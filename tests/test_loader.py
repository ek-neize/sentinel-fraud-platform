from sentinel.data.loader import CANONICAL_COLUMNS, PII_COLUMNS


def test_canonical_columns_and_no_pii(txns):
    assert list(txns.columns[: len(CANONICAL_COLUMNS)]) == CANONICAL_COLUMNS
    assert not set(PII_COLUMNS) & set(txns.columns)
    assert "cc_num" not in txns.columns


def test_ids_pseudonymised_and_prefix_stripped(txns):
    assert txns["card_id"].str.match(r"^card_[0-9a-f]{12}$").all()
    assert not txns["merchant_name"].str.startswith("fraud_").any()


def test_sorted_and_unique(txns):
    assert txns["event_time"].is_monotonic_increasing
    assert txns["transaction_id"].is_unique


def test_split_is_chronological(txns):
    bounds = txns.groupby("split", observed=True)["event_time"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["gap", "min"]
    assert bounds.loc["gap", "max"] < bounds.loc["valid", "min"]
    assert bounds.loc["valid", "max"] < bounds.loc["test", "min"]
