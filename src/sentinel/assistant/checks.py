"""Automatic checks on a drafted brief (no LLM needed).

- references: every statement cites IDs that exist and are of the right kind;
- citations: every SOP citation exists and was in the retrieved context;
- numbers: every number in a statement appears in what it cites (rounding tolerated);
- separation: no model outputs inside the verified-facts list;
- decision language: no verdict on the transaction and no chosen card action;
- context-only signals: never cited in the summary, always labelled as context.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from sentinel.assistant.case import CaseFile
from sentinel.assistant.sops import Section

_NUM = re.compile(r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?")
_MODEL_WORDS = re.compile(r"\b(score|model|reason code|risk band|probability|threshold)\b", re.I)
_SUBJECT = (
    r"\b(this|the alerted|the|these|both|all|each)\s+(transaction|purchase|payment|alert|"
    r"case|activity|transactions|purchases|alerts)|\bit\b|\bthey\b"
)
_VERDICT = re.compile(
    rf"({_SUBJECT})\s+(is|was|are|were|appears to be|appear to be|seems|seem|looks|look)"
    r"(\s+(likely|clearly|definitely|probably|almost certainly|very likely))?\s+(to be\s+)?"
    r"(a\s+|an\s+)?(fraud\b|fraudulent|genuine|legitimate|a false positive|not fraud)"
    r"|\b(likely|probable|clear|obvious|confirmed)\s+(fraud|fraudulent activity)\b",
    re.I,
)
_ACTION = [
    re.compile(
        r"(^|[.;:]\s*)(block|freeze|cancel|reissue|close)\s+(the\s+|this\s+)?(card|case|alert)",
        re.I,
    ),
    re.compile(
        r"\b(mark|classify|close|dispose|set the disposition)\b[^.;]{0,40}"
        r"\bas\s+(confirmed_fraud|confirmed fraud|cleared|fraud|a false positive)",
        re.I,
    ),
    re.compile(r"\brecommend(s|ed)?\s+(blocking|clearing|confirming|closing|freezing)", re.I),
]
_HEDGE = re.compile(
    r"\b(whether|before|if|unless|decide|consider|may|only an investigator|do not|don't|never|"
    r"not)\b",
    re.I,
)


def numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _matches(x: float, allowed: list[float]) -> bool:
    return any(abs(x - a) <= max(0.5, 0.005 * abs(a)) for a in allowed)


def _unhedged(pat: re.Pattern, text: str) -> list[str]:
    hits = []
    for m in pat.finditer(text):
        window = text[max(0, m.start() - 60) : m.end()]
        if not _HEDGE.search(window):
            hits.append(m.group(0).strip(" .;:"))
    return hits


def decision_hits(text: str) -> list[str]:
    hits = _unhedged(_VERDICT, text)
    for pat in _ACTION:
        for m in pat.finditer(text):
            window = text[max(0, m.start() - 60) : m.end()]
            if not _HEDGE.search(window):
                hits.append(m.group(0).strip(" .;:"))
    return list(dict.fromkeys(hits))


@dataclass
class CheckResult:
    n_statements: int = 0
    n_steps: int = 0
    n_citations: int = 0
    unknown_refs: list[str] = field(default_factory=list)
    wrong_kind_refs: list[str] = field(default_factory=list)
    unreferenced: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    uncited_steps: list[str] = field(default_factory=list)
    number_errors: list[str] = field(default_factory=list)
    fabricated_numbers: list[str] = field(default_factory=list)
    separation_errors: list[str] = field(default_factory=list)
    decision_language: list[str] = field(default_factory=list)
    context_misuse: list[str] = field(default_factory=list)
    n_numeric_statements: int = 0

    @property
    def issues(self) -> int:
        return sum(len(v) for k, v in asdict(self).items() if isinstance(v, list))

    @property
    def passed(self) -> bool:
        return self.issues == 0

    def to_dict(self) -> dict:
        return {**asdict(self), "issues": self.issues, "passed": self.passed}


def statements(brief: dict) -> list[tuple[str, str, list[str]]]:
    """(section, text, refs) for every referenced statement in the brief."""
    out = [("summary", brief["summary"]["text"], list(brief["summary"].get("refs", [])))]
    out += [("fact", s["statement"], list(s.get("refs", []))) for s in brief["verified_facts"]]
    out += [("signal", s["statement"], list(s.get("refs", []))) for s in brief["model_signals"]]
    return out


def check_brief(brief: dict, case: CaseFile, context: list[Section]) -> CheckResult:
    r = CheckResult()
    items = case.by_id
    ctx = {s.id: s for s in context}
    all_case_numbers = numbers(" ".join(i.text for i in case.items))
    context_only = case.context_only_ids()

    for section, text, refs in statements(brief):
        r.n_statements += 1
        valid = [x for x in refs if x in items]
        r.unknown_refs += [x for x in refs if x not in items]
        if section == "fact":
            r.wrong_kind_refs += [x for x in valid if items[x].kind != "fact"]
            if _MODEL_WORDS.search(text):
                r.separation_errors.append(text)
        if section == "signal":
            r.wrong_kind_refs += [x for x in valid if items[x].kind != "signal"]
            if set(valid) & context_only and "context" not in text.lower():
                r.context_misuse.append(text)
        if section == "summary" and set(valid) & context_only:
            r.context_misuse.append(
                f"summary cites context-only {sorted(set(valid) & context_only)}"
            )
        if not valid:
            r.unreferenced.append(text)
        nums = numbers(text)
        if nums:
            r.n_numeric_statements += 1
            allowed = numbers(" ".join(items[x].text for x in valid))
            bad = [n for n in nums if not _matches(n, allowed)]
            if bad:
                r.number_errors.append(text)
                if any(not _matches(n, all_case_numbers) for n in bad):
                    r.fabricated_numbers.append(text)
        check_text = text if section != "fact" else ""
        r.decision_language += (
            decision_hits(check_text) if check_text else _unhedged(_VERDICT, text)
        )

    for step in brief["procedure"]:
        r.n_steps += 1
        cites = list(step.get("citations", []))
        r.n_citations += len(cites)
        r.invalid_citations += [c for c in cites if c not in ctx]
        if not [c for c in cites if c in ctx]:
            r.uncited_steps.append(step["step"])
        allowed = all_case_numbers + numbers(" ".join(ctx[c].text for c in cites if c in ctx))
        if any(not _matches(n, allowed) for n in numbers(step["step"])):
            r.fabricated_numbers.append(step["step"])
        r.decision_language += decision_hits(step["step"])
    for q in brief.get("open_questions", []):
        r.decision_language += _unhedged(_VERDICT, q)
    return r
