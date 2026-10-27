import pytest
from pydantic import ValidationError

from sentinel.common.events import TransactionEvent


def _row_to_event(row) -> dict:
    d = row.drop(labels=["is_fraud", "split"], errors="ignore").to_dict()
    d["category"] = str(d["category"])
    d["state"] = str(d["state"])
    d["gender"] = str(d["gender"])
    return d


def test_canonical_rows_validate(txns):
    for _, row in txns.sample(200, random_state=0).iterrows():
        TransactionEvent(**_row_to_event(row))


def test_label_not_allowed_on_transaction_topic(txns):
    d = _row_to_event(txns.iloc[0])
    d["is_fraud"] = 0
    with pytest.raises(ValidationError):
        TransactionEvent(**d)


def test_rejects_bad_amount(txns):
    d = _row_to_event(txns.iloc[0])
    d["amount"] = -5
    with pytest.raises(ValidationError):
        TransactionEvent(**d)
