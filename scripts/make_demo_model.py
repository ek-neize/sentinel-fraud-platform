"""Train a small demo model on mock data for the Compose monitoring stack (no Databricks).

    python scripts/make_demo_model.py            # -> models/champion/ + models/demo_traffic.jsonl
    docker compose up --build

With a real model, run scripts/export_model.py instead (it writes models/champion/) and keep
models/demo_traffic.jsonl from this script for the traffic generator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.common.config import REPO_ROOT, load_assumptions
from sentinel.data.labels import simulate_label_feed
from sentinel.data.loader import to_canonical
from sentinel.data.mock import make_mock_raw
from sentinel.data.watchlist import build_watchlist
from sentinel.features.training_set import build_training_frame
from sentinel.models.training import ExperimentConfig, run_experiments
from sentinel.serving.app import TransactionFeatures

FIELDS = list(TransactionFeatures.model_fields)


def feature_record(row: pd.Series) -> dict:
    """One gold row -> the JSON body /score expects (NaN -> null)."""
    out = {}
    for k in FIELDS:
        if k not in row:
            continue
        v = row[k]
        if isinstance(v, float) and np.isnan(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default=str(REPO_ROOT / "models"))
    p.add_argument("--traffic-rows", type=int, default=5000)
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    cfg = load_assumptions()
    tx = to_canonical(make_mock_raw(seed=7))
    labels = simulate_label_feed(tx, cfg["labels"]["channels"], cfg["random_seed"])
    wl = build_watchlist(tx, labels, cfg["watchlist"], cfg["random_seed"])
    df = build_training_frame(tx, labels, wl, cfg["split"])
    train = df[df.split == "train"].reset_index(drop=True)
    valid = df[df.split == "valid"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    _, models = run_experiments(
        train,
        valid,
        [ExperimentConfig("demo", "xgb", "full", {"max_depth": 4, "n_estimators": 200})],
    )
    m = models["demo"]
    thr = float(m["thresholds"][25])
    valid_scores = m["pipeline"].predict_proba(valid)[:, 1]

    import mlflow.sklearn

    out = Path(args.out)
    dest = out / "champion"
    shutil.rmtree(dest, ignore_errors=True)
    # same serialization as the MLflow runs (the pipeline has custom transform functions)
    mlflow.sklearn.save_model(m["pipeline"], str(dest), serialization_format="cloudpickle")
    meta = {
        "model": "demo (mock data)",
        "version": "demo",
        "threshold": thr,
        "operating_k": 25,
        # validation-period reference values used by the Prometheus drift alerts
        "reference": {
            "alert_rate": float((valid_scores >= thr).mean()),
            "amount_p50": float(valid["amount"].median()),
        },
    }
    (dest / "sentinel_meta.json").write_text(json.dumps(meta, indent=2))
    sample = test.sample(min(args.traffic_rows, len(test)), random_state=1)
    with open(out / "demo_traffic.jsonl", "w") as fh:
        for _, r in sample.iterrows():
            fh.write(json.dumps(feature_record(r)) + "\n")
    print(
        f"Demo model -> {dest} (threshold {thr:.4f}, validation alert rate "
        f"{meta['reference']['alert_rate']:.2%}); traffic -> {out / 'demo_traffic.jsonl'}"
    )


if __name__ == "__main__":
    main()
