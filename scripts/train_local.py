"""Train and compare models locally (same code as notebook 05, pandas features).

Examples
  python scripts/train_local.py --mock                # quick check on mock data
  python scripts/train_local.py                       # real data from data/reference + synthetic
  python scripts/train_local.py --mlflow              # also log runs to ./mlruns
  mlflow ui --backend-store-uri ./mlruns              # browse them

Mirrors Databricks: train on `train` with label_at_cutoff, select on `valid` with
label_final. The test split is not used.
"""

from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd

from sentinel.common.config import REFERENCE_DIR, REPO_ROOT, SYNTHETIC_DIR, load_assumptions
from sentinel.features.training_set import build_training_frame
from sentinel.models.training import DEFAULT_EXPERIMENTS, run_experiments

SHOW = [
    "name",
    "pr_auc",
    "roc_auc",
    "topk100_recall",
    "thr100_recall",
    "thr100_precision",
    "thr100_fp_per_1000_txns",
    "thr25_recall",
    "thr25_precision",
    "thr25_amount_recall",
    "fit_seconds",
]


def load_frame(mock: bool) -> pd.DataFrame:
    cfg = load_assumptions()
    if mock:
        from sentinel.data.labels import simulate_label_feed
        from sentinel.data.loader import to_canonical
        from sentinel.data.mock import make_mock_raw
        from sentinel.data.watchlist import build_watchlist

        tx = to_canonical(make_mock_raw())
        labels = simulate_label_feed(tx, cfg["labels"]["channels"], cfg["random_seed"])
        wl = build_watchlist(tx, labels, cfg["watchlist"], cfg["random_seed"])
    else:
        tx = pd.read_parquet(REFERENCE_DIR / "transactions.parquet")
        labels = pd.read_parquet(SYNTHETIC_DIR / "label_feed.parquet")
        wl = pd.read_parquet(SYNTHETIC_DIR / "watchlist.parquet")
    return build_training_frame(tx, labels, wl, cfg["split"])


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mock", action="store_true")
    p.add_argument("--neg-rate", type=float, default=0.25)
    p.add_argument("--mlflow", action="store_true", help="log runs to ./mlruns")
    p.add_argument("--only", nargs="*", help="experiment names to run (default: all)")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    t0 = time.time()
    df = load_frame(args.mock)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    print(
        f"Features built in {time.time() - t0:,.0f}s | train {len(train):,} | valid {len(valid):,}"
    )

    configs = [c for c in DEFAULT_EXPERIMENTS if not args.only or c.name in args.only]
    on_result = None
    if args.mlflow:
        import os

        import mlflow

        from sentinel.models.tracking import log_experiment_run

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # simple local file store
        mlflow.set_tracking_uri(f"file:{REPO_ROOT / 'mlruns'}")
        mlflow.set_experiment("sentinel-fraud-local")

        def on_result(cfg, pipe, info, metrics, thresholds):
            log_experiment_run(
                cfg,
                pipe,
                info,
                metrics,
                thresholds,
                valid,
                args.neg_rate,
                extra_tags={"platform": "local", "mock": str(args.mock)},
            )

    table, _ = run_experiments(train, valid, configs, neg_rate=args.neg_rate, on_result=on_result)
    pd.set_option("display.width", 250)
    print(table[SHOW].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
