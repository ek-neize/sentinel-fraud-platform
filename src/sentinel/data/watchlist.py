"""Build a synthetic, leakage-safe watchlist and join it point-in-time.

Three sources:
- card hot-list      : a card is listed hours after its first fraud label becomes
                       known (usually a customer report), for a short TTL. In this
                       data each compromised card has one ~2-day burst, so a fast
                       hot-list catches the tail of the burst.
- merchant risk list : re-evaluated on a schedule; a merchant is listed when its
                       trailing fraud rate (from labels known at review time) is a
                       multiple of the global known rate.
- noise              : clean entities listed by mistake (unverified external lists).

Design rules (enforced by tests/test_labels_and_watchlist.py):
1. Listing uses only evidence known before listing: added_at >= evidence_known_at,
   with evidence taken from the lagged label feed, never from raw `is_fraud`.
2. Joins are point-in-time: a transaction matches only if
   added_at <= event_time < removed_at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WATCHLIST_COLUMNS = [
    "watchlist_id",
    "entity_type",
    "entity_value",
    "source",
    "reason",
    "evidence_known_at",
    "added_at",
    "removed_at",
]


def _hours(rng: np.random.Generator, n: int, lo: float, hi: float) -> pd.TimedeltaIndex:
    return pd.to_timedelta(rng.uniform(lo, hi, size=n), unit="h")


def _card_hotlist(confirmed: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    first = confirmed.groupby("card_id")["label_known_at"].min().reset_index()
    first = first[rng.random(len(first)) < cfg["card_listing_rate"]]
    n = len(first)
    added = first["label_known_at"].values + _hours(
        rng, n, cfg["card_listing_lag_hours_min"], cfg["card_listing_lag_hours_max"]
    )
    return pd.DataFrame(
        {
            "entity_type": "card",
            "entity_value": first["card_id"].values,
            "source": "card_hotlist",
            "reason": "fraud reported on card",
            "evidence_known_at": first["label_known_at"].values,
            "added_at": added,
            "removed_at": added + pd.Timedelta(days=cfg["card_ttl_days"]),
        }
    )


def _merchant_risk(
    transactions: pd.DataFrame, confirmed: pd.DataFrame, cfg: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Scheduled reviews: list merchants whose known trailing fraud rate stands out."""
    t_min, t_max = transactions["event_time"].min(), transactions["event_time"].max()
    window = pd.Timedelta(days=cfg["merchant_window_days"])
    reviews = pd.date_range(
        t_min.normalize() + window, t_max, freq=f"{cfg['merchant_review_every_days']}D"
    )
    tx = transactions[["event_time", "merchant_id"]]
    rows = []
    active_until: dict[str, pd.Timestamp] = {}
    for review_at in reviews:
        win_start = review_at - window
        in_win = tx[(tx["event_time"] >= win_start) & (tx["event_time"] < review_at)]
        volume = in_win.groupby("merchant_id").size()
        known = confirmed[
            (confirmed["label_known_at"] <= review_at)
            & (confirmed["event_time"] >= win_start)
            & (confirmed["event_time"] < review_at)
        ]
        frauds = known.groupby("merchant_id").size().reindex(volume.index, fill_value=0)
        global_rate = frauds.sum() / max(volume.sum(), 1)
        rate = frauds / volume
        flagged = rate[
            (volume >= cfg["merchant_min_txns"])
            & (rate >= cfg["merchant_rate_multiplier"] * global_rate)
            & (frauds > 0)
        ]
        for merchant_id, r in flagged.items():
            if active_until.get(merchant_id, pd.Timestamp.min) > review_at:
                continue  # still listed from an earlier review
            added = review_at + pd.Timedelta(
                days=float(
                    rng.uniform(
                        cfg["merchant_listing_lag_days_min"], cfg["merchant_listing_lag_days_max"]
                    )
                )
            )
            removed = added + pd.Timedelta(days=cfg["merchant_ttl_days"])
            active_until[merchant_id] = removed
            rows.append(
                {
                    "entity_type": "merchant",
                    "entity_value": merchant_id,
                    "source": "merchant_risk",
                    "reason": f"trailing fraud rate {r:.2%} vs {global_rate:.2%} overall",
                    "evidence_known_at": review_at,
                    "added_at": added,
                    "removed_at": removed,
                }
            )
    return pd.DataFrame(rows)


def _noise(
    transactions: pd.DataFrame,
    exclude: set,
    column: str,
    entity_type: str,
    n: int,
    ttl_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pool = transactions[column].drop_duplicates()
    pool = pool[~pool.isin(exclude)]
    picks = pool.sample(n=min(n, len(pool)), random_state=int(rng.integers(1e9))).values
    t_min, t_max = transactions["event_time"].min(), transactions["event_time"].max()
    added = t_min + pd.to_timedelta(
        rng.uniform(0, (t_max - t_min).total_seconds(), len(picks)), unit="s"
    )
    return pd.DataFrame(
        {
            "entity_type": entity_type,
            "entity_value": picks,
            "source": "external_list",
            "reason": "unverified external list",
            "evidence_known_at": added,
            "added_at": added,
            "removed_at": added + pd.Timedelta(days=ttl_days),
        }
    )


def build_watchlist(
    transactions: pd.DataFrame, label_feed: pd.DataFrame, cfg: dict, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    confirmed = label_feed.merge(
        transactions[["transaction_id", "card_id", "merchant_id"]], on="transaction_id"
    )
    cards = _card_hotlist(confirmed, cfg, rng)
    merchants = _merchant_risk(transactions, confirmed, cfg, rng)
    listed = set(cards["entity_value"]) | (
        set(merchants["entity_value"]) if len(merchants) else set()
    )
    parts = [
        cards,
        merchants,
        _noise(
            transactions, listed, "card_id", "card", cfg["noise_cards"], cfg["noise_ttl_days"], rng
        ),
        _noise(
            transactions,
            listed,
            "merchant_id",
            "merchant",
            cfg["noise_merchants"],
            cfg["noise_ttl_days"],
            rng,
        ),
    ]
    wl = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for col in ("evidence_known_at", "added_at", "removed_at"):
        wl[col] = pd.to_datetime(wl[col])
    wl = wl.sort_values("added_at").reset_index(drop=True)
    wl["watchlist_id"] = [f"WL{i:05d}" for i in range(1, len(wl) + 1)]
    return wl[WATCHLIST_COLUMNS]


def point_in_time_join(transactions: pd.DataFrame, watchlist: pd.DataFrame) -> pd.DataFrame:
    """Add watchlist flags using only entries active at each transaction's time.

    Adds: wl_card_hit, wl_merchant_hit, wl_any_hit, wl_ids (comma-separated).
    """
    out = transactions.copy()
    hits = []
    for entity_type, column in (("card", "card_id"), ("merchant", "merchant_id")):
        wl = watchlist.loc[watchlist["entity_type"] == entity_type]
        m = out[["transaction_id", "event_time", column]].merge(
            wl, left_on=column, right_on="entity_value", how="inner"
        )
        active = (m["added_at"] <= m["event_time"]) & (
            m["removed_at"].isna() | (m["event_time"] < m["removed_at"])
        )
        m = m.loc[active, ["transaction_id", "watchlist_id"]]
        hits.append(m)
        out[f"wl_{entity_type}_hit"] = (
            out["transaction_id"].isin(m["transaction_id"]).astype("int8")
        )

    ids = pd.concat(hits).groupby("transaction_id")["watchlist_id"].agg(",".join)
    out["wl_any_hit"] = (out["wl_card_hit"] | out["wl_merchant_hit"]).astype("int8")
    out["wl_ids"] = out["transaction_id"].map(ids)
    return out


def signal_report(joined: pd.DataFrame, by: str | None = "split") -> pd.DataFrame:
    """Lift and coverage of each watchlist flag (optionally per split)."""
    groups = joined.groupby(by, observed=True) if by else [("all", joined)]
    rows = []
    for key, g in groups:
        base = g["is_fraud"].mean()
        n_fraud = max(int(g["is_fraud"].sum()), 1)
        for flag in ("wl_card_hit", "wl_merchant_hit", "wl_any_hit"):
            hit = g[flag] == 1
            rate = g.loc[hit, "is_fraud"].mean() if hit.any() else float("nan")
            rows.append(
                {
                    "group": key,
                    "flag": flag,
                    "hit_rate": hit.mean(),
                    "fraud_rate_if_hit": rate,
                    "lift": rate / base if base > 0 else float("nan"),
                    "fraud_captured": g.loc[hit, "is_fraud"].sum() / n_fraud,
                }
            )
    return pd.DataFrame(rows).set_index(["group", "flag"])
