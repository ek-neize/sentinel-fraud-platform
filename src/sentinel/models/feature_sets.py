"""Which gold columns each model uses.

Must stay a subset of the columns produced by notebooks/databricks/04_gold_features.py
(a test checks this).
"""

from __future__ import annotations

WATCHLIST = ["wl_card_hit", "wl_merchant_hit", "wl_any_hit"]

# Protected attributes: kept in gold for fairness checks, never used as model inputs.
PROTECTED = ["gender", "age_at_txn"]

NUMERIC = [
    "amount",
    "log_amount",
    "card_prior_txn_count",
    "card_prior_amt_mean",
    "card_prior_amt_std",
    "card_prior_amt_max",
    "amt_to_prior_mean",
    "amt_to_prior_max",
    "amt_zscore",
    "secs_since_prev_txn",
    "cnt_1h",
    "cnt_24h",
    "cnt_7d",
    "amt_1h",
    "amt_24h",
    "amt_7d",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "home_merch_km",
    "log_city_pop",
    *WATCHLIST,
]

CATEGORICAL = ["category", "state"]

# The linear baseline gets one, non-overlapping version of each signal so its coefficients
# read cleanly (no amount expressed three ways, no nested windows splitting one effect).
# Trees keep everything: they are not confused by correlated inputs.
LR_EXCLUDE = [
    "amount",  # log_amount instead
    "amt_zscore",  # amt_to_prior_mean carries "unusual for this card"
    "card_prior_amt_mean",
    "card_prior_amt_std",
    "card_prior_amt_max",
    "amt_to_prior_max",
    "amt_1h",
    "amt_7d",
    "cnt_7d",
]

# Heavy-tailed, non-negative columns: log1p before the linear model sees them.
LOG1P = [
    "card_prior_txn_count",
    "card_prior_amt_mean",
    "card_prior_amt_std",
    "card_prior_amt_max",
    "amt_to_prior_mean",
    "amt_to_prior_max",
    "secs_since_prev_txn",
    "cnt_1h",
    "cnt_24h",
    "cnt_7d",
    "amt_1h",
    "amt_24h",
    "amt_7d",
    "home_merch_km",
]


def feature_set(name: str) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) columns for a named feature set."""
    if name == "full":
        return list(NUMERIC), list(CATEGORICAL)
    if name == "no_watchlist":
        return [c for c in NUMERIC if c not in WATCHLIST], list(CATEGORICAL)
    raise ValueError(f"Unknown feature set '{name}' (use 'full' or 'no_watchlist')")
