"""Business evaluation: alert policies, cost model, capacity curve, bootstrap CIs, fairness.

Everything here takes plain arrays/DataFrames so the Databricks notebook, the local
script and the tests share one implementation. Cost inputs come from
config/assumptions.yaml (`costs` and `detection`), never from code.

Cost model (per period, USD):
    review     = alerts x investigation_cost_per_alert
    escalation = caught frauds x escalation_cost_per_case
    friction   = false positives x false_positive_friction_cost
    fraud loss = missed amount x missed_fraud_loss_rate + missed count x missed_fraud_fixed_cost
               + caught amount x (1 - caught_fraud_recovery_rate) x missed_fraud_loss_rate
    total      = review + escalation + friction + fraud loss
    net benefit = total cost of doing nothing (every fraud missed) - total
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AGG_COLS = ["txns", "frauds", "fraud_amt", "alerts", "tp", "fp", "caught_amt"]


# ----------------------------------------------------------------------------- policies


def _day(df: pd.DataFrame) -> np.ndarray:
    return df["event_ts"].dt.floor("D").to_numpy()


def policy_threshold(scores, threshold: float) -> np.ndarray:
    """Flag every transaction scoring at/above the threshold (needs unlimited reviewers)."""
    return np.asarray(scores) >= threshold


def policy_topk(df: pd.DataFrame, scores, k: int) -> np.ndarray:
    """Each day, flag that day's k highest scores (upper bound: needs the whole day's ranking)."""
    rank = pd.Series(scores, index=df.index).groupby(_day(df)).rank(method="first", ascending=False)
    return (rank <= k).to_numpy()


def policy_threshold_capped(df: pd.DataFrame, scores, threshold: float, k: int) -> np.ndarray:
    """Deployable policy: flag above the threshold, review at most k per day (highest first).

    Alerts beyond k on a busy day are not reviewed; if they were fraud, that fraud is missed.
    """
    s = pd.Series(np.asarray(scores, dtype=float), index=df.index)
    above = s >= threshold
    rank = s.where(above).groupby(_day(df)).rank(method="first", ascending=False)
    return (above & (rank <= k)).to_numpy()


def policy_rule(
    df: pd.DataFrame, rule_col: str, k: int | None = None, priority_col: str = "amount"
) -> np.ndarray:
    """Rules-only baseline: flag rule hits; with a budget, review the k largest per day."""
    hit = df[rule_col].to_numpy() == 1
    if k is None:
        return hit
    prio = pd.Series(np.where(hit, df[priority_col].to_numpy(), np.nan), index=df.index)
    rank = prio.groupby(_day(df)).rank(method="first", ascending=False)
    return hit & (rank <= k).to_numpy()


# ----------------------------------------------------------------------------- aggregation


def daily_aggregates(
    df: pd.DataFrame, flagged, label: str = "label_final", amount: str = "amount"
) -> pd.DataFrame:
    """Per-day sums: the sufficient statistics for every metric and cost (and the bootstrap)."""
    f = np.asarray(flagged, dtype=bool)
    y = df[label].to_numpy() == 1
    a = df[amount].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "day": _day(df),
            "txns": 1,
            "frauds": y.astype(int),
            "fraud_amt": np.where(y, a, 0.0),
            "alerts": f.astype(int),
            "tp": (f & y).astype(int),
            "fp": (f & ~y).astype(int),
            "caught_amt": np.where(f & y, a, 0.0),
        }
    )
    return frame.groupby("day")[AGG_COLS].sum()


def _ratio(a, b):
    """a / b, NaN where b == 0. Works for scalars (bootstrap point) and arrays (samples)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)
    return float(out) if out.ndim == 0 else out


def _metrics(s: dict, n_days: int) -> dict:
    return {
        "alerts_per_day": _ratio(s["alerts"], n_days),
        "recall": _ratio(s["tp"], s["frauds"]),
        "precision": _ratio(s["tp"], s["alerts"]),
        "amount_recall": _ratio(s["caught_amt"], s["fraud_amt"]),
        "fp_per_1000_txns": 1000 * _ratio(s["fp"], s["txns"]),
    }


def _costs(s: dict, c: dict) -> dict:
    loss_rate = c["missed_fraud_loss_rate"]
    recovery = c["caught_fraud_recovery_rate"]
    missed_amt = s["fraud_amt"] - s["caught_amt"]
    missed_n = s["frauds"] - s["tp"]
    review = s["alerts"] * c["investigation_cost_per_alert"]
    escalation = s["tp"] * c["escalation_cost_per_case"]
    friction = s["fp"] * c["false_positive_friction_cost"]
    fraud_loss = (
        missed_amt * loss_rate
        + missed_n * c["missed_fraud_fixed_cost"]
        + s["caught_amt"] * (1 - recovery) * loss_rate
    )
    total = review + escalation + friction + fraud_loss
    do_nothing = s["fraud_amt"] * loss_rate + s["frauds"] * c["missed_fraud_fixed_cost"]
    return {
        "cost_review": review,
        "cost_escalation": escalation,
        "cost_friction": friction,
        "cost_fraud_loss": fraud_loss,
        "cost_total": total,
        "cost_do_nothing": do_nothing,
        "net_benefit": do_nothing - total,
    }


def flat_costs(cfg: dict) -> dict:
    """Merge the `costs` and `detection` sections of assumptions.yaml."""
    return {**cfg["costs"], **cfg["detection"]}


def summarize(df: pd.DataFrame, flagged, costs: dict, label: str = "label_final") -> dict:
    agg = daily_aggregates(df, flagged, label)
    s = agg.sum().to_dict()
    n_days = len(agg)
    return {
        "days": n_days,
        **{k: float(v) for k, v in s.items()},
        **_metrics(s, n_days),
        **_costs(s, costs),
    }


# ----------------------------------------------------------------------------- uncertainty


def bootstrap_ci(
    agg: pd.DataFrame, costs: dict, n: int = 1000, level: float = 0.95, seed: int = 42
) -> dict:
    """Day-level bootstrap: resample days with replacement, recompute every metric.

    Resampling whole days keeps within-day correlation (bursts, busy days) intact.
    Returns {metric: (low, high)}.
    """
    rng = np.random.default_rng(seed)
    arr = agg[AGG_COLS].to_numpy(dtype=float)
    n_days = len(arr)
    idx = rng.integers(0, n_days, size=(n, n_days))
    sums = arr[idx].sum(axis=1)  # (n, len(AGG_COLS))
    s = {col: sums[:, i] for i, col in enumerate(AGG_COLS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = {**_metrics(s, n_days), **_costs(s, costs)}
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {
        k: (float(np.nanquantile(v, lo)), float(np.nanquantile(v, hi)))
        for k, v in samples.items()
        if np.ndim(v) == 1
    }


# ----------------------------------------------------------------------------- capacity


def overflow_stats(
    df: pd.DataFrame, scores, threshold: float, k: int, label: str = "label_final"
) -> dict:
    """How often the threshold produces more alerts than the team can review."""
    above = policy_threshold(scores, threshold)
    capped = policy_threshold_capped(df, scores, threshold, k)
    per_day = pd.Series(above).groupby(_day(df)).sum()
    dropped = above & ~capped
    y = df[label].to_numpy() == 1
    return {
        "days": int(len(per_day)),
        "days_over_capacity": int((per_day > k).sum()),
        "max_alerts_in_a_day": int(per_day.max()) if len(per_day) else 0,
        "alerts_not_reviewed": int(dropped.sum()),
        "frauds_lost_to_overflow": int((dropped & y).sum()),
    }


def capacity_curve(
    test: pd.DataFrame, test_scores, thresholds: dict, costs: dict, label: str = "label_final"
) -> pd.DataFrame:
    """For each daily budget k: deployable (threshold fitted on validation, capped at k)
    and best-case (daily top-k) results on the test period."""
    rows = []
    for k, thr in sorted(thresholds.items()):
        for policy, flagged in (
            ("threshold_capped", policy_threshold_capped(test, test_scores, thr, k)),
            ("daily_topk", policy_topk(test, test_scores, k)),
        ):
            rows.append(
                {
                    "capacity_k": k,
                    "policy": policy,
                    "threshold": thr,
                    **summarize(test, flagged, costs, label),
                }
            )
    return pd.DataFrame(rows)


def choose_operating_point(curve: pd.DataFrame, policy: str = "threshold_capped") -> int:
    """Budget with the highest net benefit (ties -> smaller team)."""
    c = curve[curve["policy"] == policy].sort_values(
        ["net_benefit", "capacity_k"], ascending=[False, True]
    )
    return int(c.iloc[0]["capacity_k"])


# ----------------------------------------------------------------------------- fairness


def age_band(age: pd.Series, bounds: list[int]) -> pd.Series:
    edges = list(bounds) + [np.inf]
    labels = [
        f"<{edges[1]}" if i == 0 else (f"{lo}+" if np.isinf(hi) else f"{lo}-{hi - 1}")
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]
    return pd.cut(age, bins=edges, labels=labels, right=False).astype(str)


def fairness_report(
    df: pd.DataFrame, flagged, groups: dict[str, pd.Series], label: str = "label_final"
) -> pd.DataFrame:
    """Alert rate, recall, precision and false-positive rate per group.

    `*_ratio` columns compare each group with the overall rate (1.0 = parity). Common
    rule of thumb: investigate ratios outside 0.8-1.25.
    """
    f = np.asarray(flagged, dtype=bool)
    y = df[label].to_numpy() == 1

    def rates(mask):
        n, fr = mask.sum(), (mask & y).sum()
        neg = (mask & ~y).sum()
        al = (mask & f).sum()
        return {
            "txns": int(n),
            "fraud_rate": fr / n if n else np.nan,
            "alert_rate": al / n if n else np.nan,
            "recall": (mask & f & y).sum() / fr if fr else np.nan,
            "precision": (mask & f & y).sum() / al if al else np.nan,
            "false_positive_rate": (mask & f & ~y).sum() / neg if neg else np.nan,
        }

    overall = rates(np.ones(len(df), dtype=bool))
    rows = []
    for var, values in groups.items():
        v = np.asarray(values)
        for g in pd.unique(v):
            r = rates(v == g)
            rows.append(
                {
                    "attribute": var,
                    "group": g,
                    **r,
                    "alert_rate_ratio": r["alert_rate"] / overall["alert_rate"]
                    if overall["alert_rate"]
                    else np.nan,
                    "fpr_ratio": r["false_positive_rate"] / overall["false_positive_rate"]
                    if overall["false_positive_rate"]
                    else np.nan,
                    "recall_ratio": r["recall"] / overall["recall"]
                    if overall["recall"]
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["attribute", "group"]).reset_index(drop=True)
