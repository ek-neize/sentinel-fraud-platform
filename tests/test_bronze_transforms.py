"""Run the bronze notebook's Transforms cell against local Spark.

The notebook stays self-contained for Databricks; this test executes the exact
cell titled 'Transforms' so the parsing and data-quality rules are covered by CI.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")

from sentinel.streaming.producer import replay  # noqa: E402
from sentinel.streaming.sinks import MemorySink  # noqa: E402
from tests.notebook_utils import load_cell  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "databricks" / "01_bronze_ingest.py"


def _load_transforms() -> dict:
    return load_cell(NOTEBOOK.name, "Transforms")


@pytest.fixture(scope="module")
def nb():
    return _load_transforms()


def _kafka_rows(messages, extra=()):
    rows = []
    for i, m in enumerate(list(messages) + list(extra)):
        rows.append(
            (
                m["key"].encode() if m["key"] is not None else None,
                m["value"] if isinstance(m["value"], bytes) else m["value"].encode(),
                m["topic"],
                0,
                i,
                datetime(2026, 10, 27),
                0,
                [(k, v.encode()) for k, v in m.get("headers", {}).items()],
            )
        )
    return rows


KAFKA_SCHEMA = (
    "key binary, value binary, topic string, partition int, offset long, "
    "timestamp timestamp, timestampType int, headers array<struct<key:string,value:binary>>"
)


@pytest.fixture(scope="module")
def produced(txns, labels):
    sink = MemorySink()
    replay(txns.drop(columns=["is_fraud", "split"]), labels, sink, speed=None, max_events=3000)
    return sink.messages


def test_valid_transactions_parse_clean(spark, nb, produced):
    msgs = [m for m in produced if m["topic"] == nb["TRANSACTIONS_TOPIC"]]
    raw = spark.createDataFrame(_kafka_rows(msgs), KAFKA_SCHEMA)
    out = nb["parse_transactions"](nb["kafka_envelope"](raw)).collect()
    assert len(out) == len(msgs)
    assert all(r.is_valid and not r.is_test for r in out)
    assert all(r.kafka_headers["event_type"] == "transaction" for r in out)
    assert all(r.event_ts is not None for r in out)


def test_bad_transactions_are_flagged_not_dropped(spark, nb, produced):
    good = json.loads(next(m for m in produced if m["topic"] == nb["TRANSACTIONS_TOPIC"])["value"])
    test_msg = dict(good, transaction_id="TEST-0001")
    leak = dict(good, transaction_id="leak-1", is_fraud=1)
    neg = dict(good, transaction_id="neg-1", amount=-5)
    extras = [
        {"key": good["card_id"], "value": json.dumps(test_msg), "topic": nb["TRANSACTIONS_TOPIC"]},
        {"key": good["card_id"], "value": json.dumps(leak), "topic": nb["TRANSACTIONS_TOPIC"]},
        {"key": good["card_id"], "value": json.dumps(neg), "topic": nb["TRANSACTIONS_TOPIC"]},
        {"key": "card_other00000", "value": json.dumps(good), "topic": nb["TRANSACTIONS_TOPIC"]},
        {"key": "x", "value": "not json", "topic": nb["TRANSACTIONS_TOPIC"]},
    ]
    raw = spark.createDataFrame(_kafka_rows([], extras), KAFKA_SCHEMA)
    rows = {
        (r.payload.transaction_id if r.payload and r.payload.transaction_id else r.raw_value): r
        for r in nb["parse_transactions"](nb["kafka_envelope"](raw)).collect()
    }
    assert rows["TEST-0001"].is_test and rows["TEST-0001"].is_valid
    assert "label_leak" in rows["leak-1"].dq_errors
    assert "bad_amount" in rows["neg-1"].dq_errors
    assert "key_mismatch" in rows[good["transaction_id"]].dq_errors
    assert "unparseable_json" in rows["not json"].dq_errors or not rows["not json"].is_valid
    assert len(rows) == 5


def test_labels_parse_and_check_ordering(spark, nb, produced):
    msgs = [m for m in produced if m["topic"] == nb["LABELS_TOPIC"]]
    assert msgs, "expected some labels in the first 3000 events"
    bad = dict(
        json.loads(msgs[0]["value"]), transaction_id="bad-1", label_known_at="2000-01-01T00:00:00"
    )
    extra = [{"key": "bad-1", "value": json.dumps(bad), "topic": nb["LABELS_TOPIC"]}]
    raw = spark.createDataFrame(_kafka_rows(msgs, extra), KAFKA_SCHEMA)
    out = {
        r.payload.transaction_id: r for r in nb["parse_labels"](nb["kafka_envelope"](raw)).collect()
    }
    assert "label_before_event" in out["bad-1"].dq_errors
    assert all(r.is_valid for k, r in out.items() if k != "bad-1")
