"""Generate a small offline sample in the exact raw Sparkov CSV format.

Used by CI and tests (no Kaggle credentials needed) and for quick local demos.
Mirrors the real data's fraud structure: almost every card is compromised once,
in a single burst of ~10 frauds over ~2 days, with higher amounts, online /
POS-heavy categories and night-time activity. Do NOT report results on mock data.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

CATEGORIES = [
    "misc_net",
    "grocery_pos",
    "entertainment",
    "gas_transport",
    "misc_pos",
    "grocery_net",
    "shopping_net",
    "shopping_pos",
    "food_dining",
    "personal_care",
    "health_fitness",
    "travel",
    "kids_pets",
    "home",
]
FRAUD_CATEGORIES = ["shopping_net", "grocery_pos", "misc_net", "shopping_pos", "gas_transport"]
STATES = ["ON", "TX", "NY", "PA", "CA", "OH", "MI", "IL", "FL", "AL"]  # mixed on purpose
JOBS = ["Engineer", "Teacher", "Nurse", "Accountant", "Designer", "Chemist", "Lawyer"]


def make_mock_raw(
    n_cards: int = 150,
    n_merchants: int = 120,
    start: str = "2019-01-01",
    end: str = "2021-01-01",
    txn_per_card_per_day: float = 2.5,
    fraud_card_share: float = 0.99,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    days = (t1 - t0).days

    cards = pd.DataFrame(
        {
            "cc_num": [str(4000_0000_0000_0000 + i * 7919) for i in range(n_cards)],
            "first": "Jane",
            "last": "Doe",
            "street": "1 Main St",
            "gender": rng.choice(["F", "M"], n_cards),
            "city": [f"City{i % 40}" for i in range(n_cards)],
            "state": rng.choice(STATES, n_cards),
            "zip": rng.integers(10000, 99999, n_cards).astype(str),
            "lat": rng.uniform(30, 45, n_cards),
            "long": rng.uniform(-110, -75, n_cards),
            "city_pop": rng.integers(500, 2_000_000, n_cards),
            "job": rng.choice(JOBS, n_cards),
            "dob": pd.to_datetime("1950-01-01")
            + pd.to_timedelta(rng.integers(0, 365 * 50, n_cards), unit="D"),
            "spend_scale": rng.lognormal(3.6, 0.5, n_cards),
        }
    )
    merchants = pd.DataFrame(
        {
            "merchant": [f"fraud_Merchant {i:03d} LLC" for i in range(n_merchants)],
            "category": rng.choice(CATEGORIES, n_merchants),
        }
    )

    # --- Legitimate traffic
    n_legit = int(n_cards * days * txn_per_card_per_day)
    card_idx = rng.integers(0, n_cards, n_legit)
    mer_idx = rng.integers(0, n_merchants, n_legit)
    day_off = rng.integers(0, days, n_legit)
    hour = np.clip(rng.normal(14, 4, n_legit), 0, 23.99)
    legit = pd.DataFrame(
        {
            "card_idx": card_idx,
            "mer_idx": mer_idx,
            "ts": t0 + pd.to_timedelta(day_off, unit="D") + pd.to_timedelta(hour, unit="h"),
            "amt": np.round(
                rng.lognormal(0, 0.8, n_legit) * cards["spend_scale"].values[card_idx] / 2, 2
            ),
            "is_fraud": 0,
        }
    )

    # --- One fraud burst per compromised card (~10 frauds over <= ~2 days)
    fraud_cards = np.flatnonzero(rng.random(n_cards) < fraud_card_share)
    rows = []
    fraud_mer = merchants.index[merchants["category"].isin(FRAUD_CATEGORIES)].to_numpy()
    for c in fraud_cards:
        burst_start = t0 + pd.Timedelta(days=float(rng.uniform(0, days - 3)))
        span_h = float(rng.uniform(24, 48))
        size = int(np.clip(rng.normal(10, 3), 2, 19))
        for _ in range(size):
            ts = burst_start + pd.Timedelta(hours=float(rng.uniform(0, span_h)))
            if rng.random() < 0.7:  # most fraud at night
                ts = (
                    ts.normalize()
                    + pd.Timedelta(hours=float(rng.choice([22, 23, 0, 1, 2, 3])))
                    + pd.Timedelta(minutes=float(rng.uniform(0, 59)))
                )
            rows.append(
                {
                    "card_idx": c,
                    "mer_idx": int(rng.choice(fraud_mer)),
                    "ts": ts,
                    "amt": round(float(rng.uniform(200, 1200)), 2),
                    "is_fraud": 1,
                }
            )
    df = pd.concat([legit, pd.DataFrame(rows)], ignore_index=True)
    df = df[(df["ts"] >= t0) & (df["ts"] < t1)]

    c = cards.iloc[df["card_idx"]].reset_index(drop=True)
    m = merchants.iloc[df["mer_idx"]].reset_index(drop=True)
    df = df.reset_index(drop=True)
    raw = pd.DataFrame(
        {
            "trans_date_trans_time": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": c["cc_num"],
            "merchant": m["merchant"],
            "category": m["category"],
            "amt": df["amt"],
            "first": c["first"],
            "last": c["last"],
            "gender": c["gender"],
            "street": c["street"],
            "city": c["city"],
            "state": c["state"],
            "zip": c["zip"],
            "lat": c["lat"].round(4),
            "long": c["long"].round(4),
            "city_pop": c["city_pop"],
            "job": c["job"],
            "dob": c["dob"].dt.strftime("%Y-%m-%d"),
            "trans_num": [uuid.UUID(int=int(rng.integers(0, 2**63))).hex for _ in range(len(df))],
            # Mimic the source quirk: unix_time is offset from the real date.
            "unix_time": (df["ts"] - pd.DateOffset(years=7)).astype("int64") // 10**9,
            "merch_lat": (c["lat"] + rng.normal(0, 0.8, len(df))).round(6),
            "merch_long": (c["long"] + rng.normal(0, 0.8, len(df))).round(6),
            "is_fraud": df["is_fraud"],
        }
    ).sort_values("trans_date_trans_time")
    return raw.reset_index(drop=True)
