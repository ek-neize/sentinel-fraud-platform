"""Build a dashboard snapshot from mock data (CI, local demos without Databricks).

    python scripts/build_mock_snapshot.py --out /tmp/snapshot
    SENTINEL_SNAPSHOT_DIR=/tmp/snapshot streamlit run dashboard/app.py

Runs the same code as notebooks 05-08 on the mock generator. Never commit its output
as the public snapshot: the real one comes from Databricks (scripts/pull_snapshot.py).
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from sentinel.alerts.pipeline import ScoringModel, empty_queue, process_day
from sentinel.common.config import load_assumptions
from sentinel.dashboard_export import DRIFT_FEATURES, build_snapshot, write_snapshot
from sentinel.data.labels import simulate_label_feed
from sentinel.data.loader import to_canonical
from sentinel.data.mock import make_mock_raw
from sentinel.data.watchlist import build_watchlist
from sentinel.evaluation.report import run_business_evaluation
from sentinel.features.training_set import build_training_frame
from sentinel.models.training import ExperimentConfig, run_experiments
from sentinel.monitoring.checks import monitoring_history


def make_inputs(days: int = 21, bootstrap: int = 200, seed: int = 7) -> dict:
    """Everything notebook 08 reads, produced from mock data by the real pipeline code."""
    cfg = load_assumptions()
    tx = to_canonical(make_mock_raw(seed=seed))
    labels = simulate_label_feed(tx, cfg["labels"]["channels"], cfg["random_seed"])
    wl = build_watchlist(tx, labels, cfg["watchlist"], cfg["random_seed"])
    df = build_training_frame(tx, labels, wl, cfg["split"]).merge(
        tx[["transaction_id", "merchant_name"]], on="transaction_id"
    )
    train, valid, test = (
        df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test")
    )
    _, models = run_experiments(
        train,
        valid,
        [
            ExperimentConfig("champion", "xgb", "full", {"max_depth": 4, "n_estimators": 200}),
            ExperimentConfig("baseline", "logreg", "full", {"C": 0.1}),
        ],
    )
    scores = {
        n: {
            "valid": m["pipeline"].predict_proba(valid)[:, 1],
            "test": m["pipeline"].predict_proba(test)[:, 1],
        }
        for n, m in models.items()
    }
    result = run_business_evaluation(valid, test, scores, cfg, bootstrap_n=bootstrap)
    k = result.operating_k
    thr = result.thresholds["champion"][k]

    model = ScoringModel(
        models["champion"]["pipeline"],
        thr,
        "mock",
        k,
        cfg["alerting"]["max_age_days"],
        cfg["alerting"]["priority"],
    )
    label_map = test.set_index("transaction_id")["label_final"]
    queue, summaries = empty_queue(), []
    start = test["event_ts"].max().normalize() - pd.Timedelta(days=days - 1)
    for d in pd.date_range(start, periods=days, freq="D"):
        today = test[test["event_ts"].dt.normalize() == d]
        queue, s, _ = process_day(d, today, queue, model, label_map)
        summaries.append(s)

    both = pd.concat(
        [
            valid.assign(score=scores["champion"]["valid"]),
            test.assign(score=scores["champion"]["test"]),
        ],
        ignore_index=True,
    )
    both["above_threshold"] = both["score"] >= thr
    return {
        "cfg": cfg,
        "queue": queue,
        "daily_summary": pd.DataFrame(summaries),
        "scored": both,
        "eval_tables": {
            "curve": result.curve,
            "strategies": result.strategies,
            "intervals": result.intervals,
            "fairness": result.fairness,
        },
        "model_info": {
            "name": "fraud_model",
            "version": "mock",
            "threshold": thr,
            "operating_k": k,
            "pr_auc": result.headline["champion_pr_auc"],
        },
        "baseline_scores": scores["baseline"],
    }


def build(out: str, days: int = 21, bootstrap: int = 200, seed: int = 7) -> None:
    i = make_inputs(days, bootstrap, seed)
    tables, meta = build_snapshot(
        queue=i["queue"],
        daily_summary=i["daily_summary"],
        scored=i["scored"],
        features=i["scored"],
        history_txns=i["scored"],
        eval_tables=i["eval_tables"],
        model_info=i["model_info"],
        cfg=i["cfg"],
        monitoring=monitoring_history(
            i["scored"][i["scored"]["split"] == "test"],
            i["scored"][i["scored"]["split"] == "valid"],
            i["daily_summary"],
            i["cfg"],
            DRIFT_FEATURES,
        ),
    )
    meta["mock"] = True
    path = write_snapshot(tables, meta, out)
    sizes = {n: len(t) for n, t in tables.items()}
    print(f"Mock snapshot written to {path}: {sizes}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--bootstrap", type=int, default=200)
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    build(args.out, args.days, args.bootstrap)


if __name__ == "__main__":
    main()
