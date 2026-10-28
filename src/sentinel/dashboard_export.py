"""Build the dashboard's small, pre-aggregated tables from pipeline outputs.

Used by notebook 08 (writes them as `dash_*` Delta tables and as Parquet files for the
public snapshot) and by scripts/build_mock_snapshot.py (CI and local demos). The
dashboard itself never imports `sentinel`; it only reads these tables, so its folder
deploys on its own to Streamlit Community Cloud or Databricks Apps.

Every table is small (thousands of rows at most) and contains no protected attributes
at row level: fairness appears only as group aggregates.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.monitoring.drift import drift_table

TABLES = [
    "alerts", "daily_ops", "card_history", "trend_daily", "trend_category", "trend_state",
    "monthly_perf", "drift", "eval_curve", "eval_strategies", "eval_intervals", "eval_fairness",
]  # fmt: skip
# Written by notebook 09 (monitoring); older snapshots may not have them.
OPTIONAL_TABLES = ["monitoring_checks", "label_maturity"]

ALERT_COLUMNS = [
    "alert_id", "transaction_id", "event_ts", "created_day", "card_id", "merchant_name",
    "category", "amount", "score", "priority", "status", "status_day", "wait_days",
    "disposition", "reason_1", "reason_2", "reason_3", "threshold", "model_version",
]  # fmt: skip

DRIFT_FEATURES = [
    "amount", "amt_to_prior_mean", "amt_24h", "cnt_24h", "secs_since_prev_txn",
    "home_merch_km", "hour_of_day", "card_prior_txn_count",
]  # fmt: skip

HISTORY_DAYS = 7


def _month(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").astype(str)


def build_card_history(
    alerts: pd.DataFrame, txns: pd.DataFrame, days: int = HISTORY_DAYS
) -> pd.DataFrame:
    """For each alert: the card's transactions in the `days` before it, up to the alert.

    Point-in-time: nothing after the alerted transaction is shown, as in a live case.
    txns needs transaction_id, event_ts, card_id, merchant_name, category, amount, score.
    """
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "transaction_id",
                "event_ts",
                "merchant_name",
                "category",
                "amount",
                "score",
                "is_alerted_txn",
            ]
        )
    cards = txns[txns["card_id"].isin(alerts["card_id"].unique())]
    m = alerts[["alert_id", "card_id", "event_ts", "transaction_id"]].merge(
        cards, on="card_id", suffixes=("_alert", "")
    )
    keep = (m["event_ts"] <= m["event_ts_alert"]) & (
        m["event_ts"] > m["event_ts_alert"] - pd.Timedelta(days=days)
    )
    h = m[keep].copy()
    h["is_alerted_txn"] = h["transaction_id"] == h["transaction_id_alert"]
    cols = [
        "alert_id",
        "transaction_id",
        "event_ts",
        "merchant_name",
        "category",
        "amount",
        "score",
        "is_alerted_txn",
    ]
    return h[cols].sort_values(["alert_id", "event_ts"]).reset_index(drop=True)


def build_trends(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """scored: transaction_id, event_ts, split, category, state, amount, label_final,
    above_threshold (valid + test)."""
    s = scored.assign(
        day=scored["event_ts"].dt.normalize(),
        fraud_amount=np.where(scored["label_final"] == 1, scored["amount"], 0.0),
        tp=(scored["above_threshold"] & (scored["label_final"] == 1)).astype(int),
        alerts=scored["above_threshold"].astype(int),
    )
    daily = (
        s.groupby(["day", "split"], as_index=False)
        .agg(
            txns=("transaction_id", "size"),
            frauds=("label_final", "sum"),
            fraud_amount=("fraud_amount", "sum"),
            alerts=("alerts", "sum"),
            tp=("tp", "sum"),
        )
        .sort_values("day")
    )

    test = s[s["split"] == "test"]

    def by(col):
        g = test.groupby(col, as_index=False).agg(
            txns=("transaction_id", "size"),
            frauds=("label_final", "sum"),
            fraud_amount=("fraud_amount", "sum"),
            alerts=("alerts", "sum"),
            tp=("tp", "sum"),
        )
        g["fraud_rate"] = g["frauds"] / g["txns"]
        g["alert_rate"] = g["alerts"] / g["txns"]
        g["alert_precision"] = np.where(
            g["alerts"] > 0, g["tp"] / g["alerts"].clip(lower=1), np.nan
        )
        g["recall"] = np.where(g["frauds"] > 0, g["tp"] / g["frauds"].clip(lower=1), np.nan)
        return g.sort_values("frauds", ascending=False).reset_index(drop=True)

    monthly = (
        s.assign(month=_month(s["event_ts"]))
        .groupby(["month", "split"], as_index=False)
        .agg(
            txns=("transaction_id", "size"),
            frauds=("label_final", "sum"),
            alerts=("alerts", "sum"),
            tp=("tp", "sum"),
            days=("day", "nunique"),
        )
    )
    monthly["precision"] = np.where(
        monthly["alerts"] > 0, monthly["tp"] / monthly["alerts"].clip(lower=1), np.nan
    )
    monthly["recall"] = np.where(
        monthly["frauds"] > 0, monthly["tp"] / monthly["frauds"].clip(lower=1), np.nan
    )
    monthly["alerts_per_day"] = monthly["alerts"] / monthly["days"]
    return {
        "trend_daily": daily.reset_index(drop=True),
        "trend_category": by("category"),
        "trend_state": by("state").head(15),
        "monthly_perf": monthly.sort_values("month").reset_index(drop=True),
    }


def build_drift(features: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """PSI of key inputs and the score: validation (reference) vs. each test month."""
    ref = features[features["split"] == "valid"]
    cur = features[features["split"] == "test"].assign(period=lambda d: _month(d["event_ts"]))
    numeric = [c for c in [score_col, *DRIFT_FEATURES] if c in features]
    t = drift_table(ref, cur, numeric, ["category"], period_col="period")
    t["feature"] = t["feature"].replace({score_col: "model score"})
    return t.rename(columns={"period": "month"})


def headline_from_strategies(strategies: pd.DataFrame, intervals: pd.DataFrame, k: int) -> dict:
    s = strategies.set_index("strategy")

    def row(prefix):
        hits = [n for n in s.index if n.startswith(prefix)]
        return s.loc[hits[0]] if hits else None

    out = {}
    for key, prefix in (
        ("capped", f"XGBoost champion: threshold, max {k}/day"),
        ("backlog_score", "XGBoost champion: backlog queue by score"),
        ("backlog_expected_loss", "XGBoost champion: backlog queue by expected loss"),
        ("rule", f"Watchlist rule: top {k}/day"),
        ("baseline", f"Logistic regression: threshold, max {k}/day"),
        ("overtime", "XGBoost champion: threshold, no cap"),
    ):
        r = row(prefix)
        if r is not None:
            out[key] = {
                m: float(r[m])
                for m in (
                    "recall",
                    "precision",
                    "amount_recall",
                    "fp_per_1000_txns",
                    "net_benefit",
                    "alerts_per_day",
                )
            }
    ci = intervals[intervals["strategy"].str.startswith(f"XGBoost champion: threshold, max {k}")]
    out["capped_ci"] = {r.metric: [float(r.ci_low), float(r.ci_high)] for r in ci.itertuples()}
    return out


def build_snapshot(
    *,
    queue: pd.DataFrame,
    daily_summary: pd.DataFrame,
    scored: pd.DataFrame,
    features: pd.DataFrame,
    history_txns: pd.DataFrame,
    eval_tables: dict,
    model_info: dict,
    cfg: dict,
    monitoring: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return ({table_name: DataFrame}, meta). See module docstring for inputs."""
    alerts = queue.reindex(columns=ALERT_COLUMNS).copy()
    alerts["event_ts"] = pd.to_datetime(alerts["event_ts"])
    tables = {
        "alerts": alerts.sort_values(
            ["created_day", "priority"], ascending=[False, False]
        ).reset_index(drop=True),
        "daily_ops": daily_summary.sort_values("day").reset_index(drop=True),
        "card_history": build_card_history(alerts, history_txns),
        **build_trends(scored),
        "drift": build_drift(features),
        "eval_curve": eval_tables["curve"],
        "eval_strategies": eval_tables["strategies"],
        "eval_intervals": eval_tables["intervals"],
        "eval_fairness": eval_tables["fairness"].assign(group=lambda d: d["group"].astype(str))
        if len(eval_tables["fairness"])
        else eval_tables["fairness"],
    }
    if monitoring is not None:
        tables["monitoring_checks"], tables["label_maturity"] = monitoring
    k = int(model_info["operating_k"])
    test = scored[scored["split"] == "test"]
    meta = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "model": model_info,
        "operating_k": k,
        "max_age_days": cfg["alerting"]["max_age_days"],
        "priority": cfg["alerting"]["priority"],
        "costs": {**cfg["costs"], **cfg["detection"]},
        "test_period": [f"{test['event_ts'].min():%Y-%m-%d}", f"{test['event_ts'].max():%Y-%m-%d}"]
        if len(test)
        else None,
        "test_txns": int(len(test)),
        "test_frauds": int(test["label_final"].sum()) if len(test) else 0,
        "headline": headline_from_strategies(
            eval_tables["strategies"], eval_tables["intervals"], k
        ),
        "synthetic_data": True,
    }
    return tables, meta


def write_snapshot(tables: dict[str, pd.DataFrame], meta: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        # Spark Connect's toPandas() puts PlanMetrics objects in df.attrs; pandas would
        # try to JSON-encode them into the Parquet metadata. They are never needed here.
        df = df.copy(deep=False)
        df.attrs = {}
        df.to_parquet(
            out / f"{name}.parquet",
            index=False,
            coerce_timestamps="us",
            allow_truncated_timestamps=True,
        )
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out
