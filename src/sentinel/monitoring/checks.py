"""Daily model and operations checks (run by notebook 09 and the mock snapshot).

Each check returns a row: check, subject, value, warn and critical thresholds, status
(ok / warn / critical) and a plain-language detail. Thresholds live in
config/assumptions.yaml (`monitoring`).

Why these checks. Fraud labels arrive late (customer reports within days, chargebacks up
to 60 days), so the earliest warnings cannot use labels:
- input and score drift (PSI vs the validation period)            - no labels needed
- alert volume, backlog and expiry vs the review budget           - no labels needed
- missing values in model inputs                                  - no labels needed
- precision of reviewed alerts (investigator dispositions)        - known within days
- recall on the latest *matured* cohort (all labels in)           - known after 60 days
`label_maturity` shows how much of the eventual fraud is visible at each transaction age,
which is why recent recall always looks better or worse than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.monitoring.drift import drift_table

STATUS_ORDER = {"ok": 0, "warn": 1, "critical": 2}


def _status_high(value: float, warn: float, critical: float) -> str:
    """Higher is worse."""
    if value is None or pd.isna(value):
        return "ok"
    return "critical" if value >= critical else "warn" if value >= warn else "ok"


def _status_low(value: float, warn: float, critical: float) -> str:
    """Lower is worse."""
    if value is None or pd.isna(value):
        return "ok"
    return "critical" if value < critical else "warn" if value < warn else "ok"


def _row(check, subject, value, warn, critical, status, detail):
    return {
        "check": check,
        "subject": subject,
        "value": value,
        "warn": warn,
        "critical": critical,
        "status": status,
        "detail": detail,
    }


def label_maturity(scored: pd.DataFrame, as_of: pd.Timestamp, bands: list[int]) -> pd.DataFrame:
    """Fraud visible by transaction age on `as_of`: known labels vs final labels.

    scored: event_ts, label_final, label_known_ts, above_threshold.
    """
    s = scored[scored["event_ts"] <= as_of].copy()
    s["age_days"] = (as_of - s["event_ts"]).dt.days
    edges = [*bands, 10_000]
    names = [
        f"{a}-{b - 1} days" if b < 10_000 else f"{a}+ days"
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    s["age_band"] = pd.cut(s["age_days"], edges, right=False, labels=names)
    s["fraud"] = s["label_final"] == 1
    s["known"] = s["fraud"] & pd.to_datetime(s["label_known_ts"]).le(as_of)
    s["caught"] = s["fraud"] & s["above_threshold"].astype(bool)
    s["caught_known"] = s["caught"] & s["known"]
    g = s.groupby("age_band", observed=False).agg(
        txns=("fraud", "size"),
        frauds_final=("fraud", "sum"),
        frauds_known=("known", "sum"),
        caught_final=("caught", "sum"),
        caught_known=("caught_known", "sum"),
    )
    g["label_coverage"] = g["frauds_known"] / g["frauds_final"].replace(0, np.nan)
    g["recall_known"] = g["caught_known"] / g["frauds_known"].replace(0, np.nan)
    g["recall_final"] = g["caught_final"] / g["frauds_final"].replace(0, np.nan)
    g.insert(0, "as_of", as_of.normalize())
    return g.reset_index()


def run_checks(
    scored: pd.DataFrame,
    reference: pd.DataFrame,
    daily_ops: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: dict,
    features: list[str],
) -> pd.DataFrame:
    """scored: scored transactions (event_ts, score, above_threshold, label_final,
    label_known_ts and the drift features); reference: the validation period, same
    columns; daily_ops: alert_daily_summary (day, new_alerts, reviewed, expired, open_end,
    confirmed_fraud, capacity)."""
    m = cfg["monitoring"]
    as_of = pd.Timestamp(as_of)
    start = as_of.normalize() - pd.Timedelta(days=m["window_days"] - 1)
    end = as_of.normalize() + pd.Timedelta(days=1)
    win = scored[(scored["event_ts"] >= start) & (scored["event_ts"] < end)]
    rows = []

    # 1. Drift of inputs and score (no labels needed).
    feats = [f for f in ["score", *features] if f in win and f in reference]
    if len(win) and len(reference):
        d = drift_table(reference, win.assign(period="window"), feats, ["category"])
        for r in d.itertuples():
            exempt = r.feature in m.get("drift_exempt", [])
            status = "ok" if exempt else _status_high(r.psi, m["drift_warn"], m["drift_critical"])
            name = "model score" if r.feature == "score" else r.feature
            rows.append(
                _row(
                    "drift",
                    name,
                    r.psi,
                    m["drift_warn"],
                    m["drift_critical"],
                    status,
                    f"PSI {r.psi:.3f} vs validation"
                    + (" (exempt: grows by design)" if exempt else ""),
                )
            )

    # 2. Missing values in model inputs vs reference.
    for f in features:
        if f in win and f in reference and len(win):
            inc = win[f].isna().mean() - reference[f].isna().mean()
            rows.append(
                _row(
                    "missing values",
                    f,
                    inc,
                    m["null_increase_warn"],
                    m["null_increase_critical"],
                    _status_high(inc, m["null_increase_warn"], m["null_increase_critical"]),
                    f"missing share {win[f].isna().mean():.1%} "
                    f"(reference {reference[f].isna().mean():.1%})",
                )
            )

    # 3. Operations: alert volume, backlog, expiry, reviewed precision.
    ops = daily_ops[(daily_ops["day"] >= start) & (daily_ops["day"] < end)]
    if len(ops):
        k = float(ops["capacity"].iloc[-1])
        per_day = ops["new_alerts"].mean() / k
        lo_w, hi_w = m["alert_volume_warn"]
        lo_c, hi_c = m["alert_volume_critical"]
        st = (
            "critical"
            if per_day < lo_c or per_day > hi_c
            else "warn"
            if per_day < lo_w or per_day > hi_w
            else "ok"
        )
        rows.append(
            _row(
                "alert volume",
                "new alerts per day / budget",
                per_day,
                hi_w,
                hi_c,
                st,
                f"{ops['new_alerts'].mean():.1f} new alerts/day vs budget {k:.0f}",
            )
        )
        backlog = ops["open_end"].iloc[-1] / k
        rows.append(
            _row(
                "backlog",
                "open alerts / daily budget",
                backlog,
                m["backlog_warn_days"],
                m["backlog_critical_days"],
                _status_high(backlog, m["backlog_warn_days"], m["backlog_critical_days"]),
                f"{int(ops['open_end'].iloc[-1])} open alerts at day end",
            )
        )
        expired = ops["expired"].sum() / max(ops["new_alerts"].sum(), 1)
        rows.append(
            _row(
                "expiry",
                "alerts expired unreviewed",
                expired,
                m["expired_warn"],
                m["expired_critical"],
                _status_high(expired, m["expired_warn"], m["expired_critical"]),
                f"{int(ops['expired'].sum())} of {int(ops['new_alerts'].sum())} alerts",
            )
        )
        if ops["reviewed"].sum():
            prec = ops["confirmed_fraud"].sum() / ops["reviewed"].sum()
            rows.append(
                _row(
                    "precision",
                    "reviewed alerts confirmed",
                    prec,
                    m["precision_warn"],
                    m["precision_critical"],
                    _status_low(prec, m["precision_warn"], m["precision_critical"]),
                    f"{int(ops['confirmed_fraud'].sum())} of "
                    f"{int(ops['reviewed'].sum())} reviewed alerts confirmed",
                )
            )

    # 4. Recall on the latest matured 30-day cohort (labels complete).
    mature_end = as_of - pd.Timedelta(days=m["matured_after_days"])
    cohort = scored[
        (scored["event_ts"] > mature_end - pd.Timedelta(days=30))
        & (scored["event_ts"] <= mature_end)
        & (scored["label_final"] == 1)
    ]
    if len(cohort):
        rec = cohort["above_threshold"].astype(bool).mean()
        rows.append(
            _row(
                "recall (matured)",
                f"cohort to {mature_end:%Y-%m-%d}",
                rec,
                m["recall_warn"],
                m["recall_critical"],
                _status_low(rec, m["recall_warn"], m["recall_critical"]),
                f"{int(cohort['above_threshold'].sum())} of {len(cohort)} frauds scored "
                "above the threshold",
            )
        )

    out = pd.DataFrame(rows)
    out.insert(0, "as_of", as_of.normalize())
    return out


def overall_status(checks: pd.DataFrame) -> str:
    if checks.empty:
        return "ok"
    return max(checks["status"], key=STATUS_ORDER.get)


def monitoring_history(
    scored: pd.DataFrame,
    reference: pd.DataFrame,
    daily_ops: pd.DataFrame,
    cfg: dict,
    features: list[str],
    every_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the checks as of every `every_days` across the scored period, as a daily job
    would have (backfill), plus the label-maturity table for the last date."""
    days = pd.date_range(
        scored["event_ts"].min().normalize() + pd.Timedelta(days=6),
        scored["event_ts"].max().normalize(),
        freq=f"{every_days}D",
    )
    last = scored["event_ts"].max().normalize()
    if len(days) == 0 or days[-1] != last:
        days = days.append(pd.DatetimeIndex([last]))
    checks = pd.concat(
        [
            run_checks(
                scored, reference, daily_ops, d + pd.Timedelta(hours=23, minutes=59), cfg, features
            )
            for d in days
        ],
        ignore_index=True,
    )
    checks["as_of"] = pd.to_datetime(checks["as_of"]).dt.normalize()
    maturity = label_maturity(
        scored, last + pd.Timedelta(hours=23, minutes=59), cfg["monitoring"]["label_bands"]
    )
    return checks, maturity
