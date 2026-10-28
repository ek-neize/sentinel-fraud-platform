"""Backlog alert queue: how investigators actually work through alerts.

Each day:
1. New alerts (score >= threshold) join the open backlog.
2. Alerts older than `max_age_days` expire unreviewed (by then the card is usually
   hot-listed or the burst is over).
3. Reviewers take the `k` highest-priority open alerts.

Compared with a hard daily cap, a busy day's overflow is not lost: it waits for a
quieter day. `step_queue` is the operational unit (one day, used by the scoring job);
`simulate_queue` replays it over a whole period (used by the evaluation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATUS_OPEN, STATUS_REVIEWED, STATUS_EXPIRED = "open", "reviewed", "expired"


def priority(scores, amounts=None, mode: str = "score") -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if mode == "score":
        return s
    if mode == "expected_loss":
        if amounts is None:
            raise ValueError("expected_loss priority needs amounts")
        return s * np.asarray(amounts, dtype=float)
    raise ValueError(f"Unknown priority '{mode}' (use 'score' or 'expected_loss')")


def step_queue(
    open_alerts: pd.DataFrame,
    new_alerts: pd.DataFrame,
    day: pd.Timestamp,
    k: int,
    max_age_days: int,
) -> pd.DataFrame:
    """Advance the queue by one day.

    Both inputs need columns: alert_id, created_day, priority (plus any extras, kept).
    Returns every alert touched today with `status` and `status_day` set:
    reviewed today, expired today, or still open.
    """
    day = pd.Timestamp(day).normalize()
    pool = pd.concat([open_alerts, new_alerts], ignore_index=True)
    if pool.empty:
        return pool.assign(
            status=pd.Series(dtype=str), status_day=pd.Series(dtype="datetime64[ns]")
        )
    age = (day - pd.to_datetime(pool["created_day"]).dt.normalize()).dt.days
    expired = age >= max_age_days
    live = pool[~expired].sort_values(["priority", "alert_id"], ascending=[False, True])
    reviewed_ids = set(live["alert_id"].head(k))

    status = np.where(
        expired,
        STATUS_EXPIRED,
        np.where(pool["alert_id"].isin(reviewed_ids), STATUS_REVIEWED, STATUS_OPEN),
    )
    return pool.assign(status=status, status_day=day)


def simulate_queue(
    df: pd.DataFrame,
    scores,
    threshold: float,
    k: int,
    max_age_days: int = 3,
    priority_mode: str = "score",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """Replay the queue over a period. Returns one row per alert with its final status,
    the day it was reviewed/expired and how long it waited."""
    s = np.asarray(scores, dtype=float)
    flagged = s >= threshold
    amounts = df[amount_col].to_numpy() if amount_col in df else None
    alerts = pd.DataFrame(
        {
            "alert_id": np.arange(len(df))[flagged],
            "row": np.flatnonzero(flagged),
            "created_day": df["event_ts"].dt.normalize().to_numpy()[flagged],
            "priority": priority(s, amounts, priority_mode)[flagged],
            "score": s[flagged],
        }
    )
    days = pd.date_range(
        df["event_ts"].min().normalize(),
        df["event_ts"].max().normalize() + pd.Timedelta(days=max_age_days),
        freq="D",
    )
    by_day = {d: g for d, g in alerts.groupby("created_day")}
    open_alerts = alerts.iloc[0:0]
    finished = []
    for d in days:
        out = step_queue(open_alerts, by_day.get(d, alerts.iloc[0:0]), d, k, max_age_days)
        if out.empty:
            continue
        done = out[out["status"] != STATUS_OPEN]
        finished.append(done)
        open_alerts = out[out["status"] == STATUS_OPEN].drop(columns=["status", "status_day"])
    res = (
        pd.concat(finished, ignore_index=True)
        if finished
        else alerts.assign(status=STATUS_EXPIRED, status_day=pd.NaT)
    )
    res["wait_days"] = (res["status_day"] - res["created_day"]).dt.days
    return res.sort_values("row").reset_index(drop=True)


def reviewed_mask(n_rows: int, queue_result: pd.DataFrame) -> np.ndarray:
    """Boolean per original row: True if that transaction's alert was reviewed."""
    mask = np.zeros(n_rows, dtype=bool)
    rows = queue_result.loc[queue_result["status"] == STATUS_REVIEWED, "row"].to_numpy()
    mask[rows] = True
    return mask


def queue_stats(queue_result: pd.DataFrame, labels) -> dict:
    y = np.asarray(labels)
    q = queue_result
    rev = q[q["status"] == STATUS_REVIEWED]
    rev_fraud = rev[y[rev["row"].to_numpy()] == 1]
    return {
        "alerts": int(len(q)),
        "reviewed": int(len(rev)),
        "expired": int((q["status"] == STATUS_EXPIRED).sum()),
        "frauds_expired": int(
            (y[q.loc[q["status"] == STATUS_EXPIRED, "row"].to_numpy()] == 1).sum()
        ),
        "share_reviewed_same_day": float((rev["wait_days"] == 0).mean()) if len(rev) else np.nan,
        "mean_wait_days_fraud": float(rev_fraud["wait_days"].mean()) if len(rev_fraud) else np.nan,
        "max_wait_days": int(rev["wait_days"].max()) if len(rev) else 0,
    }
