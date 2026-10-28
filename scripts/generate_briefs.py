"""Draft investigation briefs for the alerts in the dashboard snapshot.

    python scripts/generate_briefs.py                       # Claude (ANTHROPIC_API_KEY in .env)
    python scripts/generate_briefs.py --provider template   # offline, free
    git add dashboard/briefs && git commit -m "Update briefs" && git push

Writes dashboard/briefs/briefs.parquet. The public dashboard shows these drafts in the
case view, so it never needs an API key. Re-run after pulling a new snapshot.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pandas as pd

from sentinel.assistant.assistant import build_index, draft_brief
from sentinel.assistant.case import build_case
from sentinel.assistant.providers import make_provider
from sentinel.common.config import REPO_ROOT, load_assumptions

SNAP = REPO_ROOT / "dashboard" / "snapshot"
OUT = REPO_ROOT / "dashboard" / "briefs" / "briefs.parquet"


def generate(
    provider_name: str,
    snapshot=SNAP,
    out=OUT,
    limit: int = 0,
    workers: int = 4,
    cfg: dict | None = None,
    provider=None,
) -> pd.DataFrame:
    cfg = cfg or load_assumptions()
    alerts = pd.read_parquet(snapshot / "alerts.parquet")
    hist = pd.read_parquet(snapshot / "card_history.parquet")
    meta = json.loads((snapshot / "meta.json").read_text())
    if limit:
        alerts = alerts.head(limit)
    index = build_index()
    provider = provider or make_provider(provider_name, cfg)
    by_alert = {k: g for k, g in hist.groupby("alert_id")}

    def one(row):
        case = build_case(
            row,
            by_alert.get(row["alert_id"], hist.iloc[0:0]),
            alerts,
            cfg["assistant"]["history_days"],
        )
        try:
            res = draft_brief(case, provider, index, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  {row['alert_id']}: {type(e).__name__}: {e}")
            return None
        return {
            "alert_id": row["alert_id"],
            "provider": res.generation.provider,
            "model": res.generation.model,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "snapshot_generated_at": meta.get("generated_at"),
            "markdown": res.markdown(),
            "checks_passed": res.checks.passed,
            "issues": res.checks.issues,
            # what was flagged, so the dashboard can say which check fired
            "issue_details": json.dumps(
                {k: v for k, v in res.checks.to_dict().items() if isinstance(v, list) and v}
            ),
            "brief_json": json.dumps(res.brief),
            "cited_sections": json.dumps(
                [{"id": s.id, "label": s.label, "text": s.text} for s in res.cited_sections()]
            ),
            "input_tokens": res.generation.input_tokens,
            "output_tokens": res.generation.output_tokens,
        }

    rows = list(ThreadPoolExecutor(workers).map(one, [r for _, r in alerts.iterrows()]))
    df = pd.DataFrame([r for r in rows if r])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    cfg = load_assumptions()
    df = generate(args.provider, limit=args.limit, workers=args.workers, cfg=cfg)
    cost = ""
    prices = cfg["assistant"].get("prices", {})
    if len(df) and df["model"].iloc[0] in prices:
        pi, po = prices[df["model"].iloc[0]]
        cost = f", about ${(df.input_tokens.sum() * pi + df.output_tokens.sum() * po) / 1e6:.2f}"
    print(f"{len(df)} briefs -> {OUT} ({df['checks_passed'].mean():.0%} passed all checks{cost})")


if __name__ == "__main__":
    main()
