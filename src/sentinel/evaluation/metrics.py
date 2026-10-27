"""Metrics that match how a fraud team actually works.

Accuracy is useless at ~0.5% fraud (always saying "legit" scores 99.5%). What matters:
- how well the model *ranks* fraud above legit (PR-AUC);
- with a fixed daily review budget of K alerts, how much fraud is caught (recall@K)
  and how many alerts are real (precision@K);
- how many legitimate customers get flagged (false positives per 1,000 transactions).

Two capacity views:
- `daily_topk_report`: each day, investigators review that day's K highest scores.
  Uses the day's full ranking, so it is an *upper bound* for a real-time system.
- `fixed_threshold_report`: one score threshold, chosen on validation so that the
  *average* day produces K alerts, applied unchanged to later data. This is what a
  deployed system can actually do, and busy days overflow capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(y_true, scores) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(scores)
    if y.min() == y.max():
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "base_rate": float(y.mean())}
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "base_rate": float(y.mean()),
    }


def _summarise(flagged: np.ndarray, y: np.ndarray, n_days: int, amounts=None) -> dict:
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    frauds = int((y == 1).sum())
    alerts = tp + fp
    out = {
        "alerts": alerts,
        "alerts_per_day": alerts / max(n_days, 1),
        "true_positives": tp,
        "false_positives": fp,
        "frauds": frauds,
        "precision": tp / alerts if alerts else float("nan"),
        "recall": tp / frauds if frauds else float("nan"),
        "fp_per_1000_txns": 1000 * fp / max(len(y), 1),
    }
    if amounts is not None:
        a = np.asarray(amounts, dtype=float)
        fraud_amt = a[y == 1].sum()
        out["fraud_amount"] = float(fraud_amt)
        out["fraud_amount_caught"] = float(a[flagged & (y == 1)].sum())
        out["amount_recall"] = (
            float(out["fraud_amount_caught"] / fraud_amt) if fraud_amt else float("nan")
        )
    return out


def daily_topk_report(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    k: int,
    day_col: str = "day",
    amount_col: str | None = "amount",
) -> dict:
    """Flag the K highest-scoring transactions of every day."""
    rank = df.groupby(day_col)[score_col].rank(method="first", ascending=False)
    flagged = (rank <= k).to_numpy()
    amounts = df[amount_col] if amount_col and amount_col in df else None
    rep = _summarise(flagged, df[label_col].to_numpy(), df[day_col].nunique(), amounts)
    rep["capacity_k"] = k
    return rep


def threshold_for_daily_capacity(scores, days, k: int) -> float:
    """Score threshold so that, on average, k alerts per day exceed it."""
    s = np.asarray(scores, dtype=float)
    n_days = pd.Series(days).nunique()
    target = int(round(k * n_days))
    if target <= 0:
        return float("inf")
    if target >= len(s):
        return float(s.min())
    candidate = np.sort(s)[::-1][target - 1]
    # With tied scores, `>= candidate` can flag far more than the budget; step just above
    # the tie instead, so we never exceed the target on the data used to fit it.
    if (s >= candidate).sum() > target:
        candidate = np.nextafter(candidate, np.inf)
    return float(candidate)


def fixed_threshold_report(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    threshold: float,
    k: int,
    day_col: str = "day",
    amount_col: str | None = "amount",
) -> dict:
    """Flag everything at/above `threshold`; also report how often capacity is exceeded."""
    flagged = (df[score_col] >= threshold).to_numpy()
    amounts = df[amount_col] if amount_col and amount_col in df else None
    rep = _summarise(flagged, df[label_col].to_numpy(), df[day_col].nunique(), amounts)
    per_day = pd.Series(flagged).groupby(df[day_col].to_numpy()).sum()
    rep.update(
        {
            "threshold": float(threshold),
            "capacity_k": k,
            "days_over_capacity": int((per_day > k).sum()),
            "max_alerts_in_a_day": int(per_day.max()) if len(per_day) else 0,
        }
    )
    return rep
