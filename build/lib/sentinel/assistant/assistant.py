"""Draft a brief for one case: retrieve SOP sections, generate, check, render."""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.assistant.case import CaseFile
from sentinel.assistant.checks import CheckResult, check_brief
from sentinel.assistant.prompts import SYSTEM, user_prompt
from sentinel.assistant.providers import Generation, Provider, Request
from sentinel.assistant.retrieval import SopIndex
from sentinel.assistant.schema import brief_schema
from sentinel.assistant.sops import Section, load_sections

DISCLAIMER = (
    "AI-drafted brief. The decision rests with the investigator (SOP-010 §1); "
    "model signals are not evidence (SOP-002 §1)."
)


@dataclass
class BriefResult:
    case: CaseFile
    context: list[Section]
    generation: Generation
    checks: CheckResult

    @property
    def brief(self) -> dict:
        return self.generation.data

    def cited_sections(self) -> list[Section]:
        ids = {c for s in self.brief.get("procedure", []) for c in s.get("citations", [])}
        return [s for s in self.context if s.id in ids]

    def markdown(self) -> str:
        return render_markdown(self.brief, self.checks, self.generation)


def build_index(folder=None) -> SopIndex:
    return SopIndex(load_sections(folder) if folder else load_sections())


def draft_brief(case: CaseFile, provider: Provider, index: SopIndex, cfg: dict) -> BriefResult:
    a = cfg.get("assistant", {})
    context = index.context_for(case.query(), a.get("retrieve_k", 6), a.get("pinned_sections", []))
    schema = brief_schema(
        [i.id for i in case.facts], [i.id for i in case.signals], [s.id for s in context]
    )
    req = Request(SYSTEM, user_prompt(case, context), schema, case, context)
    gen = provider.generate(req)
    return BriefResult(case, context, gen, check_brief(gen.data, case, context))


def _refs(ids) -> str:
    return " " + " ".join(f"`{i}`" for i in ids) if ids else ""


def render_markdown(
    brief: dict, checks: CheckResult | None = None, gen: Generation | None = None
) -> str:
    md = [f"**Summary.** {brief['summary']['text']}{_refs(brief['summary'].get('refs'))}", ""]
    md.append("**Verified facts** (recorded data)")
    md += [f"- {s['statement']}{_refs(s.get('refs'))}" for s in brief["verified_facts"]]
    md += ["", "**Model signals** (not evidence)"]
    md += [f"- {s['statement']}{_refs(s.get('refs'))}" for s in brief["model_signals"]]
    md += ["", "**Procedure** (from the SOPs)"]
    md += [
        f"{n}. {s['step']}{_refs(s.get('citations'))}" for n, s in enumerate(brief["procedure"], 1)
    ]
    if brief.get("open_questions"):
        md += ["", "**Open questions**"] + [f"- {q}" for q in brief["open_questions"]]
    foot = DISCLAIMER
    if gen:
        foot += f" Drafted by {gen.provider} ({gen.model})."
    if checks is not None:
        foot += (
            " Automatic checks passed."
            if checks.passed
            else f" Automatic checks flagged {checks.issues} issue(s)."
        )
    md += ["", f"_{foot}_"]
    return "\n".join(md)
