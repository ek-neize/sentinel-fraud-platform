"""Evaluate the investigation assistant on a fixed eval set.

    python scripts/evaluate_assistant.py                                   # template only
    python scripts/evaluate_assistant.py --providers template,anthropic --judge anthropic
    python scripts/evaluate_assistant.py --providers anthropic,databricks --judge anthropic

Needs ANTHROPIC_API_KEY in .env for anthropic (generator or judge) and a Databricks CLI
profile for databricks. The eval set (eval/assistant/cases.jsonl) is built from the
committed dashboard snapshot on first run and reused afterwards (--rebuild to redo).
Writes docs/assistant_eval.md and eval/assistant/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from sentinel.assistant.assistant import build_index, render_markdown
from sentinel.assistant.evaluation import (
    EvalCase,
    build_eval_set,
    recheck,
    run_eval,
    summarize,
    write_report,
)
from sentinel.assistant.providers import make_provider
from sentinel.common.config import REPO_ROOT, load_assumptions

SNAP = REPO_ROOT / "dashboard" / "snapshot"
EVAL_DIR = REPO_ROOT / "eval" / "assistant"


def load_cases(n: int, rebuild: bool, snapshot=SNAP) -> list[EvalCase]:
    f = EVAL_DIR / "cases.jsonl"
    if f.exists() and not rebuild:
        return [EvalCase.from_json(x) for x in f.read_text().splitlines() if x.strip()]
    alerts = pd.read_parquet(snapshot / "alerts.parquet")
    hist = pd.read_parquet(snapshot / "card_history.parquet")
    cases = build_eval_set(alerts, hist, n)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(c.to_json() for c in cases) + "\n")
    print(f"Built {len(cases)} eval cases -> {f}")
    return cases


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--providers", default="template")
    p.add_argument("--judge", default=None, help="anthropic (uses assistant.judge_model)")
    p.add_argument("--judge-template", action="store_true", help="also judge the template")
    p.add_argument("--n", type=int, default=30, help="real cases (plus 6 adversarial)")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--profile", default=None, help="Databricks CLI profile")
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "assistant_eval.md"))
    p.add_argument(
        "--from-results",
        action="store_true",
        help="no API calls: re-check the saved briefs in eval/assistant/results.jsonl "
        "and rewrite the report",
    )
    args = p.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    cfg = load_assumptions()
    a = cfg["assistant"]
    cases = load_cases(args.n, args.rebuild)
    if args.limit:
        cases = cases[: args.limit]
    index = build_index()
    results_file = EVAL_DIR / "results.jsonl"
    if args.from_results:
        records = [json.loads(x) for x in results_file.read_text().splitlines() if x.strip()]
        records = recheck(records, cases, index, cfg)
        judge_model = next((r["judge_model"] for r in records if r.get("judge_model")), None)
        if judge_model is None and any(isinstance(r.get("judge"), dict) for r in records):
            judge_model = a["judge_model"]
        print(f"Re-checked {len(records)} saved briefs (no API calls)")
    else:
        providers = [
            make_provider(x.strip(), cfg, profile=args.profile) for x in args.providers.split(",")
        ]
        judge = make_provider("anthropic", cfg, model=a["judge_model"]) if args.judge else None
        total = len(cases) * len(providers)
        done = []

        def progress(r):
            done.append(r)
            flag = (
                "ok"
                if r.get("ok") and r["checks"]["passed"]
                else ("FAIL" if not r.get("ok") else "flagged")
            )
            print(
                f"[{len(done)}/{total}] {r['provider']:10s} {r['case_id']:20s} {flag}", flush=True
            )

        records = run_eval(
            cases, providers, index, cfg, judge, args.workers, args.judge_template, progress
        )
        (EVAL_DIR / "results.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in records) + "\n"
        )
        judge_model = a["judge_model"] if judge else None
        for r in records:
            r["judge_model"] = judge_model
    results_file.write_text("\n".join(json.dumps(r, default=str) for r in records) + "\n")
    summary = summarize(records, a.get("prices"))
    retr = [
        r["retrieval_recall"] for r in records if r.get("ok") and r["retrieval_recall"] is not None
    ]
    best = next(
        (r for r in records if r.get("ok") and r["provider"] != "template" and r["kind"] == "real"),
        None,
    ) or next(r for r in records if r.get("ok"))
    example = render_markdown(best["brief"]) if best else None
    history = []
    for f in sorted(EVAL_DIR.glob("results_v*.jsonl")):
        tag = f.stem.split("_")[-1]
        cf = EVAL_DIR / f"cases_{tag}.jsonl"
        if not cf.exists():
            continue
        old_cases = [EvalCase.from_json(x) for x in cf.read_text().splitlines() if x.strip()]
        old = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        history.append((f"{tag} (re-checked)", summarize(recheck(old, old_cases, index, cfg))))
    if history:
        history.append(("latest", summary))
    changes = EVAL_DIR / "CHANGES.md"
    write_report(
        summary,
        records,
        cases,
        args.out,
        judge_model,
        {
            "n_sections": len(index.sections),
            "k": a["retrieve_k"],
            "recall": sum(retr) / len(retr) if retr else float("nan"),
        },
        example,
        history,
        changes.read_text() if changes.exists() else None,
    )
    with pd.option_context("display.width", 160, "display.max_columns", 40):
        print(summary.set_index(["provider", "model"]).T.to_string())
    print(f"\nReport: {args.out}")
    if any(not r.get("ok") for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
