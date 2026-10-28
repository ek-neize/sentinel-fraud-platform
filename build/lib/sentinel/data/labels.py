"""Simulate when fraud labels become *known*.

A fraud label is never available at transaction time. Two channels deliver it:
- customer_report: the cardholder notices and calls in (hours to a few days)
- chargeback:      a formal dispute (weeks)

Each fraud transaction gets one channel. Every downstream step that uses labels
(watchlist building, retraining, drift monitoring) must filter on
`label_known_at <= as_of_time`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_label_feed(transactions: pd.DataFrame, channels: dict, seed: int = 42) -> pd.DataFrame:
    """Return one row per fraud transaction with when and how its label became known.

    Non-fraud transactions are confirmed implicitly once the dispute window has
    passed, so they are not emitted on the label feed.
    """
    rng = np.random.default_rng(seed)
    fraud = transactions.loc[transactions["is_fraud"] == 1, ["transaction_id", "event_time"]]
    n = len(fraud)

    cr, cb = channels["customer_report"], channels["chargeback"]
    is_report = rng.random(n) < cr["share"]
    lag_hours = np.where(
        is_report,
        rng.uniform(cr["lag_hours_min"], cr["lag_hours_max"], n),
        rng.uniform(cb["lag_days_min"] * 24, cb["lag_days_max"] * 24, n),
    )
    feed = fraud.assign(
        label_known_at=(fraud["event_time"] + pd.to_timedelta(lag_hours, unit="h")).dt.ceil("s"),
        label="fraud",
        label_source=np.where(is_report, "customer_report", "chargeback"),
    )
    return feed.sort_values("label_known_at").reset_index(drop=True)[
        ["transaction_id", "event_time", "label_known_at", "label", "label_source"]
    ]


def labels_known_as_of(label_feed: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Labels an analyst could actually see at `as_of`."""
    return label_feed.loc[label_feed["label_known_at"] <= as_of]
