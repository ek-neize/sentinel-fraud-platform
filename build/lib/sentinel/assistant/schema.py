"""Structured output of the assistant. The JSON schema is built per case so the model can
only reference fact/signal IDs that exist and SOP sections that were retrieved."""

from __future__ import annotations


def _refs(ids: list[str]) -> dict:
    return {"type": "array", "items": {"type": "string", "enum": ids}}


def _obj(props: dict) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def brief_schema(fact_ids: list[str], signal_ids: list[str], section_ids: list[str]) -> dict:
    ref_ids = fact_ids + signal_ids
    return _obj(
        {
            "summary": _obj({"text": {"type": "string"}, "refs": _refs(ref_ids)}),
            "verified_facts": {
                "type": "array",
                "items": _obj({"statement": {"type": "string"}, "refs": _refs(fact_ids)}),
            },
            "model_signals": {
                "type": "array",
                # signal IDs, plus facts when a reason code is compared with the records
                "items": _obj(
                    {"statement": {"type": "string"}, "refs": _refs(signal_ids + fact_ids)}
                ),
            },
            "procedure": {
                "type": "array",
                "items": _obj({"step": {"type": "string"}, "citations": _refs(section_ids)}),
            },
            "open_questions": {"type": "array", "items": {"type": "string"}},
        }
    )


def judge_schema() -> dict:
    verdict = {"type": "string", "enum": ["supported", "unsupported", "contradicted"]}
    return _obj(
        {
            "statements": {
                "type": "array",
                "items": _obj(
                    {"index": {"type": "integer"}, "verdict": verdict, "note": {"type": "string"}}
                ),
            },
            "citations": {
                "type": "array",
                "items": _obj(
                    {
                        "index": {"type": "integer"},
                        "supports": {"type": "boolean"},
                        "note": {"type": "string"},
                    }
                ),
            },
        }
    )
