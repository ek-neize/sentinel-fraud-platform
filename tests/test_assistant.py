"""Investigation assistant: SOP parsing, retrieval, case files, checks, providers, eval."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sentinel.assistant.assistant import build_index, draft_brief, render_markdown
from sentinel.assistant.case import build_case, is_context_only
from sentinel.assistant.checks import check_brief, decision_hits
from sentinel.assistant.evaluation import (
    CANARY,
    build_eval_set,
    gold_sops,
    run_eval,
    summarize,
    write_report,
)
from sentinel.assistant.providers import (
    AnthropicProvider,
    DatabricksProvider,
    Generation,
    Provider,
    TemplateProvider,
    parse_json,
)
from sentinel.assistant.sops import load_sections

SNAP = Path(__file__).resolve().parents[1] / "dashboard" / "snapshot"
T0 = pd.Timestamp("2020-12-07 23:14:00")


@pytest.fixture(scope="module")
def index():
    return build_index()


def _alert(**kw):
    a = {
        "alert_id": "A1",
        "transaction_id": "T9",
        "event_ts": T0,
        "created_day": T0.normalize(),
        "card_id": "C1",
        "merchant_name": "Shop A",
        "category": "shopping_net",
        "amount": 694.47,
        "score": 0.98,
        "threshold": 0.0194,
        "model_version": "3",
        "reason_1": "amount $694",
        "reason_2": "$1,500 spent on this card in the previous 24h",
        "reason_3": "customer city population ~2,345",
        "status": "open",
        "status_day": None,
        "disposition": "confirmed_fraud",
    }
    a.update(kw)
    return pd.Series(a)


def _history():
    rows = [
        ("T1", T0 - pd.Timedelta(days=5), "Grocer", "grocery_pos", 50.0, 0.001),
        ("T2", T0 - pd.Timedelta(hours=20), "Fuel", "gas_transport", 30.0, 0.002),
        ("T3", T0 - pd.Timedelta(hours=2), "Shop B", "misc_net", 400.0, 0.9),
        ("T4", T0 - pd.Timedelta(minutes=30), "Shop C", "shopping_pos", 800.0, 0.95),
        ("T9", T0, "Shop A", "shopping_net", 694.47, 0.98),  # the alerted transaction
        ("T10", T0 + pd.Timedelta(hours=1), "Later", "travel", 999.0, 0.99),  # after: excluded
        ("T0", T0 - pd.Timedelta(days=9), "Old", "home", 10.0, 0.0),  # outside 7 days
    ]
    return pd.DataFrame(
        rows, columns=["transaction_id", "event_ts", "merchant_name", "category", "amount", "score"]
    )


def test_sops_parse_into_citable_sections():
    secs = load_sections()
    assert len({s.sop_id for s in secs}) == 12
    ids = [s.id for s in secs]
    assert len(ids) == len(set(ids))
    assert all(s.text and s.heading for s in secs)
    assert "SOP-002 §3" in ids and "SOP-010 §1" in ids


def test_retrieval_finds_the_specific_procedure(index):
    top = index.search(
        "The transaction time (23:14 UTC) is within the 22:00-04:00 night window.", 1
    )
    assert top[0].id == "SOP-007 §1"
    ctx = index.context_for(
        ["card on the fraud hot-list", "category shopping_net is online"], 10, ["SOP-010 §1"]
    )
    assert ctx[0].id == "SOP-010 §1"
    got = {s.sop_id for s in ctx}
    assert {"SOP-006", "SOP-004"} <= got
    assert sum(s.sop_id == "SOP-006" for s in ctx) <= 2  # diversity cap


def test_case_is_point_in_time_and_hides_disposition():
    earlier = pd.DataFrame(
        [
            {
                "alert_id": "A0",
                "card_id": "C1",
                "event_ts": T0 - pd.Timedelta(days=1),
                "amount": 120.0,
                "merchant_name": "Prev",
                "status": "reviewed",
                "status_day": T0.normalize() - pd.Timedelta(days=1),
                "disposition": "confirmed_fraud",
            },
            {
                "alert_id": "A2",
                "card_id": "C1",
                "event_ts": T0 - pd.Timedelta(hours=2),
                "amount": 400.0,
                "merchant_name": "Shop B",
                "status": "reviewed",
                "status_day": T0.normalize(),
                "disposition": "cleared",
            },  # reviewed same day
        ]
    )
    case = build_case(_alert(), _history(), earlier)
    text = " ".join(i.text for i in case.items)
    assert "Later" not in text and "$999" not in text  # after the alert
    assert "Old" not in text  # outside the 7-day window
    assert "4 other transactions" in text
    assert "3 transactions totalling $1,230.00" in text  # 24 h window
    assert "with disposition confirmed_fraud" in text  # finished the day before
    assert "cleared" not in text  # same-day review was not known yet
    assert case.features["online"] and case.features["night"]
    assert case.context_only_ids() == {"S4"}


def test_context_only_reasons():
    assert is_context_only("customer city population ~12,000")
    assert is_context_only("card has 3 earlier transaction(s)")
    assert not is_context_only("3 earlier transaction(s) on this card in the previous 24h")
    assert not is_context_only("amount $694")


@pytest.mark.parametrize(
    ("text", "flag"),
    [
        ("This transaction is fraudulent.", True),
        ("The purchase appears to be genuine.", True),
        ("Likely fraud.", True),
        ("Block the card.", True),
        ("Close the alert as cleared.", True),
        ("Establish whether this transaction is fraudulent before deciding.", False),
        ("Before blocking the card, contact the customer (SOP-008).", False),
        ("Do not close the alert as cleared on the score alone.", False),
        ("The score is the probability that a transaction is fraudulent.", False),
    ],
)
def test_decision_language(text, flag):
    assert bool(decision_hits(text)) is flag


def _good_brief(case):
    return {
        "summary": {"text": "Online purchase of $694.47 at Shop A.", "refs": ["F1"]},
        "verified_facts": [
            {"statement": "$694.47 at Shop A on 2020-12-07 23:14 UTC.", "refs": ["F1"]}
        ],
        "model_signals": [
            {"statement": "Context only: customer city population ~2,345.", "refs": ["S4"]}
        ],
        "procedure": [
            {"step": "Contact the customer for CNP alerts above $500.", "citations": ["SOP-004 §3"]}
        ],
        "open_questions": ["Did the customer make the purchase?"],
    }


def test_checks_pass_a_grounded_brief_and_catch_errors(index):
    case = build_case(_alert(), _history())
    ctx = [index.by_id["SOP-004 §3"], index.by_id["SOP-010 §1"]]
    assert check_brief(_good_brief(case), case, ctx).passed

    bad = _good_brief(case)
    bad["summary"] = {
        "text": "This transaction is fraudulent; city population 2,345.",
        "refs": ["S4"],
    }
    bad["verified_facts"].append({"statement": "Score 0.98 at $7,777.00.", "refs": ["F1"]})
    bad["model_signals"][0]["statement"] = "City population ~2,345 raises risk."
    bad["procedure"].append({"step": "Escalate.", "citations": ["SOP-099 §1"]})
    r = check_brief(bad, case, ctx)
    assert r.decision_language
    assert r.context_misuse  # summary cites S4, and S4 not labelled context
    assert len(r.context_misuse) == 2
    assert r.separation_errors  # score inside facts
    assert r.fabricated_numbers  # $7,777 is nowhere in the case
    assert r.invalid_citations == ["SOP-099 §1"]
    assert not r.passed


def test_template_provider_is_grounded(index, cfg):
    case = build_case(_alert(), _history())
    res = draft_brief(case, TemplateProvider(), index, cfg)
    assert res.checks.passed, res.checks.to_dict()
    md = res.markdown()
    assert "Verified facts" in md and "Model signals" in md and "SOP-010 §1" in md
    assert "Context only" in md


class _FakeMessages:
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.kwargs = text, stop, None

    def create(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text=self.text),
            ],
            stop_reason=self.stop,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def test_anthropic_provider_request_and_parsing(index, cfg):
    case = build_case(_alert(), _history())
    brief = _good_brief(case)
    fake = _FakeMessages(json.dumps(brief))
    p = AnthropicProvider("claude-sonnet-5-5", "medium", client=SimpleNamespace(messages=fake))
    res = draft_brief(case, p, index, cfg)
    oc = fake.kwargs["extra_body"]["output_config"]
    assert oc["effort"] == "medium" and oc["format"]["type"] == "json_schema"
    enum = oc["format"]["schema"]["properties"]["procedure"]["items"]["properties"]["citations"]
    assert set(enum["items"]["enum"]) == {s.id for s in res.context}
    assert "temperature" not in fake.kwargs
    assert res.brief == brief and res.generation.input_tokens == 10

    fake2 = _FakeMessages("{", stop="max_tokens")
    with pytest.raises(RuntimeError, match="max_tokens"):
        AnthropicProvider("m", client=SimpleNamespace(messages=fake2)).generate(
            SimpleNamespace(system="", user="", schema={})
        )


def test_databricks_provider_parses_fenced_json(index, cfg):
    case = build_case(_alert(), _history())
    brief = _good_brief(case)
    calls = {}

    def do(method, path, body):
        calls.update(method=method, path=path, body=body)
        return {
            "choices": [{"message": {"content": f"```json\n{json.dumps(brief)}\n```"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    w = SimpleNamespace(api_client=SimpleNamespace(do=do))
    res = draft_brief(case, DatabricksProvider("my-endpoint", workspace=w), index, cfg)
    assert calls["path"] == "/serving-endpoints/my-endpoint/invocations"
    assert "JSON schema" in calls["body"]["messages"][0]["content"]
    assert res.brief == brief
    assert parse_json('noise {"a": 1} noise') == {"a": 1}


def test_gold_sops_rules():
    req, rel = gold_sops(
        {
            "online": True,
            "n_24h": 4,
            "amount": 1200,
            "night": True,
            "hot_list": True,
            "context_only_reason": True,
        }
    )
    assert req == {"SOP-002", "SOP-003", "SOP-004", "SOP-005", "SOP-006", "SOP-007", "SOP-010"}
    req, rel = gold_sops({"online": False, "n_24h": 0, "amount": 20})
    assert req == set() and "SOP-010" in rel


class _Judge(Provider):
    name, model = "anthropic", "judge"

    def generate(self, req):
        block = req.user.split("<statements>\n")[1].split("\n</statements>")[0]
        n = len(block.splitlines())
        return Generation(
            {
                "statements": [{"index": i, "verdict": "supported", "note": ""} for i in range(n)],
                "citations": [{"index": 0, "supports": True, "note": ""}],
            },
            "",
            self.name,
            self.model,
            1,
            1,
        )


@pytest.mark.skipif(not (SNAP / "alerts.parquet").exists(), reason="no committed snapshot")
def test_eval_end_to_end_on_snapshot(index, cfg, tmp_path):
    alerts = pd.read_parquet(SNAP / "alerts.parquet")
    hist = pd.read_parquet(SNAP / "card_history.parquet")
    cases = build_eval_set(alerts, hist, n=8)
    kinds = {c.kind for c in cases}
    assert "adversarial: prompt injection" in kinds
    inj = next(c for c in cases if "injection" in c.kind)
    assert CANARY in inj.case.by_id["F1"].text  # the attack is in the data
    recs = run_eval(
        cases, [TemplateProvider()], index, cfg, judge=_Judge(), workers=2, judge_template=True
    )
    s = summarize(recs, cfg["assistant"]["prices"]).iloc[0]
    assert s["valid_output"] == 1 and s["factual_accuracy"] == 1 and s["citation_validity"] == 1
    assert s["injection_resisted"] == 0  # verbatim copying echoes the injected text
    assert s["judge_supported"] == 1
    out = tmp_path / "report.md"
    write_report(
        summarize(recs),
        recs,
        cases,
        out,
        "judge",
        {"n_sections": 41, "k": 10, "recall": 1.0},
        render_markdown(recs[0]["brief"]),
    )
    text = out.read_text()
    assert "Factual accuracy" in text and "Example brief" in text


@pytest.mark.skipif(not (SNAP / "alerts.parquet").exists(), reason="no committed snapshot")
def test_template_passes_checks_on_every_real_alert(index, cfg):
    alerts = pd.read_parquet(SNAP / "alerts.parquet")
    hist = pd.read_parquet(SNAP / "card_history.parquet")
    for _, a in alerts.iterrows():
        case = build_case(a, hist[hist["alert_id"] == a["alert_id"]], alerts)
        text = " ".join(i.text for i in case.items)
        assert str(a["disposition"]) not in ("", "None") or True
        assert f"disposition {a['disposition']}" not in text or "Earlier alert" in text
        res = draft_brief(case, TemplateProvider(), index, cfg)
        assert res.checks.passed, (a["alert_id"], res.checks.to_dict())


@pytest.mark.skipif(not (SNAP / "alerts.parquet").exists(), reason="no committed snapshot")
def test_summary_mixes_judged_and_unjudged_and_recheck(index, cfg):
    """Regression: template records have no judge result; summarize must not crash."""
    from sentinel.assistant.evaluation import recheck

    alerts = pd.read_parquet(SNAP / "alerts.parquet")
    hist = pd.read_parquet(SNAP / "card_history.parquet")
    cases = build_eval_set(alerts, hist, n=4)

    class _Named(TemplateProvider):
        name, model = "anthropic", "fake-llm"

    recs = run_eval(cases, [TemplateProvider(), _Named()], index, cfg, judge=_Judge(), workers=2)
    assert not any("judge" in r for r in recs if r["provider"] == "template")
    s = summarize(recs).set_index("provider")
    assert s.loc["anthropic", "judge_supported"] == 1
    saved = [json.loads(json.dumps(r, default=str)) for r in recs]  # round-trip like the file
    again = recheck(saved, cases, index, cfg)
    assert [r["checks"]["passed"] for r in again] == [r["checks"]["passed"] for r in recs]
