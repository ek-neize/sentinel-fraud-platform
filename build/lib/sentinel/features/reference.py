"""Independent pandas implementation of the gold features.

The Databricks notebook (04_gold_features) computes features with Spark window
functions. This module computes the same features a different way (pandas
groupby / expanding / time-based rolling) so tests can check the two agree,
and that no feature ever looks at a transaction from the future.

Not used in production. If a feature changes in the notebook, change it here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VELOCITY_WINDOWS = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    """tx: one row per transaction with event_ts, card_id, amount, geo, wl_* flags."""
    df = tx.sort_values(["card_id", "event_ts", "transaction_id"]).reset_index(drop=True)
    g = df.groupby("card_id", sort=False)["amount"]

    df["card_prior_txn_count"] = df.groupby("card_id", sort=False).cumcount()
    prior_sum = g.cumsum() - df["amount"]
    df["card_prior_amt_mean"] = prior_sum / df["card_prior_txn_count"].replace(0, np.nan)
    df["card_prior_amt_std"] = g.transform(lambda s: s.expanding().std().shift(1))
    df["card_prior_amt_max"] = g.transform(lambda s: s.expanding().max().shift(1))
    df["_ts"] = (df["event_ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    df["secs_since_prev_txn"] = df.groupby("card_id", sort=False)["_ts"].diff()

    for name, secs in VELOCITY_WINDOWS.items():
        cnt = np.zeros(len(df))
        amt = np.zeros(len(df))
        for _, idx in df.groupby("card_id", sort=False).indices.items():
            t = df["_ts"].to_numpy()[idx]
            a = df["amount"].to_numpy()[idx]
            csum = np.concatenate([[0.0], np.cumsum(a)])
            lo = np.searchsorted(t, t - secs, side="left")  # first row with ts >= t - secs
            hi = np.searchsorted(t, t, side="left")  # first row with ts >= t (excludes ties)
            cnt[idx] = hi - lo
            amt[idx] = csum[hi] - csum[lo]
        df[f"cnt_{name}"] = cnt.astype("int64")
        df[f"amt_{name}"] = amt

    hour = df["event_ts"].dt.hour
    out = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"],
            "event_ts": df["event_ts"],
            "card_id": df["card_id"],
            "merchant_id": df["merchant_id"],
            "amount": df["amount"],
            "log_amount": np.log1p(df["amount"]),
            "card_prior_txn_count": df["card_prior_txn_count"],
            "card_prior_amt_mean": df["card_prior_amt_mean"],
            "card_prior_amt_std": df["card_prior_amt_std"],
            "card_prior_amt_max": df["card_prior_amt_max"],
            "amt_to_prior_mean": df["amount"] / df["card_prior_amt_mean"],
            "amt_to_prior_max": df["amount"] / df["card_prior_amt_max"],
            "amt_zscore": np.where(
                df["card_prior_amt_std"] > 0,
                (df["amount"] - df["card_prior_amt_mean"]) / df["card_prior_amt_std"],
                np.nan,
            ),
            "secs_since_prev_txn": df["secs_since_prev_txn"].astype(float),
            **{f"cnt_{k}": df[f"cnt_{k}"] for k in VELOCITY_WINDOWS},
            **{f"amt_{k}": df[f"amt_{k}"] for k in VELOCITY_WINDOWS},
            "hour_of_day": hour,
            "day_of_week": (df["event_ts"].dt.dayofweek + 1) % 7 + 1,  # Spark: 1 = Sunday
            "is_night": ((hour >= 22) | (hour < 4)).astype(int),
            "home_merch_km": _haversine_km(
                df["cust_lat"], df["cust_long"], df["merch_lat"], df["merch_long"]
            ),
            "age_at_txn": df["age_at_txn"],
            "log_city_pop": np.log1p(df["city_pop"]),
            "wl_card_hit": df["wl_card_hit"],
            "wl_merchant_hit": df["wl_merchant_hit"],
            "wl_any_hit": df["wl_any_hit"],
            "category": df["category"].astype(str),
            "state": df["state"].astype(str),
            "gender": df["gender"].astype(str),
        }
    )
    return out
