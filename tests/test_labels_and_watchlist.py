import pandas as pd

from sentinel.data.labels import labels_known_as_of
from sentinel.data.watchlist import point_in_time_join


def test_label_lag_within_channel_bounds(labels, cfg):
    ch = cfg["labels"]["channels"]
    lag_h = (labels["label_known_at"] - labels["event_time"]).dt.total_seconds() / 3600
    rep = labels["label_source"] == "customer_report"
    assert rep.any() and (~rep).any()
    cr = ch["customer_report"]
    assert lag_h[rep].between(cr["lag_hours_min"], cr["lag_hours_max"] + 1 / 3600).all()
    cb = ch["chargeback"]
    assert lag_h[~rep].between(cb["lag_days_min"] * 24, cb["lag_days_max"] * 24 + 1 / 3600).all()


def test_labels_known_as_of_filters_future(labels):
    as_of = labels["label_known_at"].median()
    assert (labels_known_as_of(labels, as_of)["label_known_at"] <= as_of).all()


def test_watchlist_listed_only_after_evidence_known(watchlist):
    # Leakage guard: nothing is listed before its evidence existed.
    assert (watchlist["added_at"] >= watchlist["evidence_known_at"]).all()


def test_card_hotlist_backed_by_known_labels(watchlist, labels, txns):
    """Every hot-listed card must trace to a fraud label known before listing."""
    known = labels.merge(txns[["transaction_id", "card_id"]], on="transaction_id")
    first_known = known.groupby("card_id")["label_known_at"].min()
    cards = watchlist[watchlist["source"] == "card_hotlist"]
    assert len(cards) > 0
    assert (cards["added_at"] >= cards["entity_value"].map(first_known)).all()


def test_merchant_listings_use_only_known_labels(watchlist, labels, txns, cfg):
    """Recompute each merchant listing's evidence using only labels known at review time."""
    merchants = watchlist[watchlist["source"] == "merchant_risk"]
    known = labels.merge(txns[["transaction_id", "merchant_id"]], on="transaction_id")
    window = pd.Timedelta(days=cfg["watchlist"]["merchant_window_days"])
    for _, row in merchants.iterrows():
        review = row["evidence_known_at"]
        ev = known[
            (known["merchant_id"] == row["entity_value"])
            & (known["label_known_at"] <= review)
            & (known["event_time"] >= review - window)
        ]
        assert len(ev) > 0
        assert row["added_at"] > review


def test_watchlist_carries_positive_signal(txns, watchlist):
    joined = point_in_time_join(txns, watchlist)
    hit = joined["wl_any_hit"] == 1
    assert joined.loc[hit, "is_fraud"].mean() > 2 * joined["is_fraud"].mean()


def test_watchlist_has_noise(watchlist):
    assert (watchlist["source"] == "external_list").sum() > 0


def test_point_in_time_join_respects_window(txns, watchlist):
    joined = point_in_time_join(txns, watchlist)
    hits = joined.loc[joined["wl_any_hit"] == 1, ["transaction_id", "event_time", "wl_ids"]]
    exploded = hits.assign(watchlist_id=hits["wl_ids"].str.split(",")).explode("watchlist_id")
    m = exploded.merge(watchlist, on="watchlist_id")
    assert (m["added_at"] <= m["event_time"]).all()
    assert (m["removed_at"].isna() | (m["event_time"] < m["removed_at"])).all()


def test_no_hit_before_listing():
    tx = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "event_time": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-03-01"]),
            "card_id": ["card_a"] * 3,
            "merchant_id": ["mer_x"] * 3,
        }
    )
    wl = pd.DataFrame(
        {
            "watchlist_id": ["WL1"],
            "entity_type": ["card"],
            "entity_value": ["card_a"],
            "added_at": pd.to_datetime(["2020-01-05"]),
            "removed_at": pd.to_datetime(["2020-02-01"]),
        }
    )
    out = point_in_time_join(tx, wl)
    assert out["wl_card_hit"].tolist() == [0, 1, 0]
