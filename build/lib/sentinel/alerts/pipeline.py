"""One day of operations: score -> alert -> queue -> reasons -> digest.

Pure pandas so it can be tested without Spark; notebook 07 only reads/writes tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.alerts.explain import top_reasons
from sentinel.alerts.queue import STATUS_OPEN, STATUS_REVIEWED, priority, step_queue
from sentinel.models.feature_sets import CATEGORICAL, NUMERIC

QUEUE_COLUMNS = [
    "alert_id", "transaction_id", "event_ts", "created_day", "card_id", "merchant_id",
    "merchant_name", "category", "amount", "score", "priority", "threshold", "model_version",
    "reason_1", "reason_2", "reason_3", "reason_features", "status", "status_day", "wait_days",
    "disposition",
]  # fmt: skip


@dataclass
class ScoringModel:
    pipeline: object
    threshold: float
    version: str
    k: int
    max_age_days: int = 3
    priority_mode: str = "score"


def empty_queue() -> pd.DataFrame:
    q = pd.DataFrame({c: pd.Series(dtype=object) for c in QUEUE_COLUMNS})
    for c in ("event_ts", "created_day", "status_day"):
        q[c] = pd.Series(dtype="datetime64[ns]")
    for c in ("amount", "score", "priority", "threshold"):
        q[c] = pd.Series(dtype=float)
    return q


def process_day(
    day,
    txns: pd.DataFrame,
    queue: pd.DataFrame,
    model: ScoringModel,
    labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Advance operations by one day.

    txns:   that day's transactions (gold features + merchant_name).
    queue:  current alert_queue table (all alerts ever, with status).
    labels: optional transaction_id -> 1/0; in this simulation it stands in for the
            investigator's finding when an alert is reviewed.
    Returns (updated queue, daily summary, today's digest rows).
    """
    day = pd.Timestamp(day).normalize()
    x = txns[NUMERIC + CATEGORICAL]
    scores = model.pipeline.predict_proba(x)[:, 1] if len(txns) else np.array([])
    hit = scores >= model.threshold
    new = txns.loc[hit].copy()
    new["score"] = scores[hit]
    new["priority"] = priority(new["score"], new["amount"], model.priority_mode)
    new["alert_id"] = new["transaction_id"]
    new["created_day"] = day
    new["threshold"] = model.threshold
    new["model_version"] = str(model.version)
    if len(new):
        reasons = top_reasons(model.pipeline, new[NUMERIC + CATEGORICAL], n=3)
        new = new.join(reasons[["reason_1", "reason_2", "reason_3", "reason_features"]])
    new = new.reindex(
        columns=[
            c
            for c in QUEUE_COLUMNS
            if c not in ("status", "status_day", "wait_days", "disposition")
        ]
    )

    open_mask = queue["status"] == STATUS_OPEN if len(queue) else pd.Series(dtype=bool)
    open_alerts = queue[open_mask] if len(queue) else empty_queue()
    stepped = step_queue(
        open_alerts.drop(
            columns=["status", "status_day", "wait_days", "disposition"], errors="ignore"
        ),
        new,
        day,
        model.k,
        model.max_age_days,
    )
    stepped["wait_days"] = (stepped["status_day"] - pd.to_datetime(stepped["created_day"])).dt.days
    stepped.loc[stepped["status"] == STATUS_OPEN, "wait_days"] = np.nan
    if labels is not None:
        lab = stepped["transaction_id"].map(labels).fillna(0)
        stepped["disposition"] = np.where(
            stepped["status"] == STATUS_REVIEWED,
            np.where(lab == 1, "confirmed_fraud", "cleared"),
            None,
        )
    else:
        stepped["disposition"] = None

    untouched = queue[~open_mask] if len(queue) else empty_queue()
    updated = pd.concat([untouched, stepped[QUEUE_COLUMNS]], ignore_index=True)

    reviewed = stepped[stepped["status"] == STATUS_REVIEWED]
    summary = {
        "day": day,
        "txns_scored": int(len(txns)),
        "new_alerts": int(len(new)),
        "reviewed": int(len(reviewed)),
        "expired": int((stepped["status"] == "expired").sum()),
        "open_end": int((stepped["status"] == STATUS_OPEN).sum()),
        "confirmed_fraud": int((reviewed["disposition"] == "confirmed_fraud").sum()),
        "capacity": model.k,
        "threshold": model.threshold,
        "model_version": str(model.version),
    }
    digest_rows = (
        stepped[stepped["status"].isin([STATUS_REVIEWED, STATUS_OPEN])]
        .sort_values(["status", "priority"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return updated, summary, digest_rows
