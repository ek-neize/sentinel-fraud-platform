"""LLM providers behind one interface.

- TemplateProvider: offline, deterministic, extractive baseline (no key; CI and demos).
- AnthropicProvider: Claude via the Messages API with a JSON-schema structured output.
- DatabricksProvider: a Databricks Model Serving chat endpoint (JSON requested in the prompt;
  no schema enforcement, so the automatic checks matter more).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from sentinel.assistant.case import CaseFile
from sentinel.assistant.sops import Section


@dataclass
class Request:
    system: str
    user: str
    schema: dict
    case: CaseFile | None = None
    sections: list[Section] = field(default_factory=list)


@dataclass
class Generation:
    data: dict
    raw: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class Provider:
    name = "base"
    model = ""

    def generate(self, req: Request) -> Generation:  # pragma: no cover - interface
        raise NotImplementedError


def parse_json(text: str) -> dict:
    """First JSON object in text (tolerates code fences / prose around it)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class TemplateProvider(Provider):
    """Restates every fact and signal verbatim and turns retrieved SOP sections into steps.
    Grounded by construction; the baseline the LLMs must match on accuracy and beat on
    usefulness (synthesis, discrepancies, prioritised steps)."""

    name, model = "template", "extractive-v1"

    def generate(self, req: Request) -> Generation:
        t0 = time.perf_counter()
        case = req.case
        f = case.features
        facts = [{"statement": i.text, "refs": [i.id]} for i in case.facts]
        signals = [
            {
                "statement": (
                    "Context only, not a reason to act: " if i.strength == "context only" else ""
                )
                + i.text,
                "refs": [i.id],
            }
            for i in case.signals
        ]
        steps = []
        for s in req.sections:
            if s.id.startswith(("SOP-009", "SOP-011")):
                continue
            first = re.split(r"(?<=\.)\s", s.text, maxsplit=1)[0]
            steps.append(
                {"step": f"Follow {s.sop_title} ({s.heading}): {first}", "citations": [s.id]}
            )
        questions = ["Did the customer authorise this transaction?"]
        if f.get("hot_list"):
            questions.append("Why and when was the card hot-listed?")
        if f.get("n_24h", 0) >= 3:
            questions.append("Do the other transactions in the last 24 hours belong to one burst?")
        data = {
            "summary": {
                "text": f"{f.get('band')}-band alert: {case.by_id['F1'].text.split(': ', 1)[1]}",
                "refs": ["F1", "S1"],
            },
            "verified_facts": facts,
            "model_signals": signals,
            "procedure": steps,
            "open_questions": questions,
        }
        return Generation(
            data, json.dumps(data), self.name, self.model, latency_s=time.perf_counter() - t0
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self, model: str, effort: str | None = "medium", max_tokens: int = 8000, client=None
    ):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client

    def generate(self, req: Request) -> Generation:
        t0 = time.perf_counter()
        output_config = {"format": {"type": "json_schema", "schema": req.schema}}
        if self.effort:
            output_config["effort"] = self.effort
        # extra_body keeps this working on SDK versions that predate output_config.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=req.system,
            messages=[{"role": "user", "content": req.user}],
            extra_body={"output_config": output_config},
        )
        text = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Response hit max_tokens; raise assistant.max_tokens")
        u = resp.usage
        return Generation(
            parse_json(text),
            text,
            self.name,
            self.model,
            getattr(u, "input_tokens", 0),
            getattr(u, "output_tokens", 0),
            time.perf_counter() - t0,
        )


class DatabricksProvider(Provider):
    name = "databricks"

    def __init__(
        self, endpoint: str, max_tokens: int = 4000, profile: str | None = None, workspace=None
    ):
        self.model, self.max_tokens = endpoint, max_tokens
        if workspace is None:
            from databricks.sdk import WorkspaceClient

            workspace = WorkspaceClient(profile=profile or os.getenv("DATABRICKS_PROFILE"))
        self.w = workspace

    def generate(self, req: Request) -> Generation:
        t0 = time.perf_counter()
        system = (
            req.system
            + "\n\nReturn only one JSON object that matches this JSON schema:\n"
            + json.dumps(req.schema)
        )
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": req.user},
            ],
            "max_tokens": self.max_tokens,
        }
        resp = self.w.api_client.do(
            "POST", f"/serving-endpoints/{self.model}/invocations", body=body
        )
        text = resp["choices"][0]["message"]["content"]
        if isinstance(text, list):  # some endpoints return content blocks
            text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
        u = resp.get("usage", {})
        return Generation(
            parse_json(text),
            text,
            self.name,
            self.model,
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            time.perf_counter() - t0,
        )


def make_provider(name: str, cfg: dict, **kw) -> Provider:
    a = cfg.get("assistant", {})
    if name == "template":
        return TemplateProvider()
    if name == "anthropic":
        return AnthropicProvider(
            kw.get("model") or a.get("anthropic_model", "claude-sonnet-5-5"),
            a.get("anthropic_effort"),
            a.get("max_tokens", 8000),
        )
    if name == "databricks":
        return DatabricksProvider(
            kw.get("model") or a["databricks_endpoint"], profile=kw.get("profile")
        )
    raise ValueError(f"Unknown provider {name!r} (template, anthropic, databricks)")
