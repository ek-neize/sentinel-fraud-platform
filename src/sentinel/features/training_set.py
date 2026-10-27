"""Pandas version of gold_training_set (mirrors notebooks 03 + 04).

Used for local experiments, tests and CI. Databricks builds the same table in Spark.
"""

from __future__ import annotations

import pandas as pd

from sentinel.data.loader import assign_split
from sentinel.data.watchlist import point_in_time_join
from sentinel.features.reference import build_features


def build_training_frame(
    transactions: pd.DataFrame,
    label_feed: pd.DataFrame,
    watchlist: pd.DataFrame,
    split_cfg: dict,
    label_cutoff: str | None = None,
) -> pd.DataFrame:
    """Canonical txns (+ label feed, watchlist) -> features + split + both label views."""
    tx = transactions.drop(columns=["is_fraud", "split"], errors="ignore")
    silver = point_in_time_join(tx, watchlist).rename(columns={"event_time": "event_ts"})
    feats = build_features(silver)
    feats["split"] = assign_split(feats["event_ts"], split_cfg).astype(str)

    cutoff = pd.Timestamp(label_cutoff or split_cfg["gap_end"])
    lab = label_feed[["transaction_id", "label_known_at", "label_source"]].rename(
        columns={"label_known_at": "label_known_ts"}
    )
    out = feats.merge(lab, on="transaction_id", how="left")
    out["label_final"] = out["label_known_ts"].notna().astype(int)
    out["label_at_cutoff"] = (
        out["label_known_ts"].notna() & (out["label_known_ts"] <= cutoff)
    ).astype(int)
    return out.sort_values(["event_ts", "transaction_id"]).reset_index(drop=True)
