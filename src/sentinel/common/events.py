"""Kafka event contracts (v1).

Two topics keep labels out of the scoring path:
  sentinel.transactions.v1  -> TransactionEvent  (key = card_id, preserves per-card order)
  sentinel.fraud_labels.v1  -> FraudLabelEvent   (key = transaction_id)

`is_fraud` is never on the transactions topic. Labels arrive later on their own
topic, exactly as chargebacks would in production.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
TRANSACTIONS_TOPIC = "sentinel.transactions.v1"
LABELS_TOPIC = "sentinel.fraud_labels.v1"


class TransactionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    transaction_id: str = Field(min_length=1)
    event_time: datetime
    card_id: str = Field(pattern=r"^card_[0-9a-f]{12}$")
    merchant_id: str = Field(pattern=r"^mer_[0-9a-f]{10}$")
    merchant_name: str
    category: str
    amount: float = Field(gt=0, lt=1_000_000)
    currency: Literal["USD"] = "USD"
    gender: str | None = None
    age_at_txn: float | None = Field(default=None, ge=0, le=120)
    job: str | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    cust_lat: float = Field(ge=-90, le=90)
    cust_long: float = Field(ge=-180, le=180)
    city_pop: int | None = Field(default=None, ge=0)
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)


class FraudLabelEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    transaction_id: str = Field(min_length=1)
    event_time: datetime
    label_known_at: datetime
    label: Literal["fraud"]
    label_source: Literal["chargeback", "customer_report", "investigation"]


def export_json_schemas(out_dir) -> None:
    """Write JSON Schemas (for Schema Registry / docs) to `out_dir`."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, model in (
        ("transaction_event", TransactionEvent),
        ("fraud_label_event", FraudLabelEvent),
    ):
        (out / f"{name}.v1.json").write_text(json.dumps(model.model_json_schema(), indent=2))
