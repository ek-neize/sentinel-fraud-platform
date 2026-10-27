import json

import pandas as pd
import pytest

from sentinel.common.events import LABELS_TOPIC, TRANSACTIONS_TOPIC
from sentinel.streaming.producer import build_timeline, parse_speed, replay
from sentinel.streaming.sinks import MemorySink


@pytest.fixture(scope="module")
def stream_inputs(txns, labels):
    return txns.drop(columns=["is_fraud", "split"]), labels


def test_parse_speed():
    assert parse_speed("1d/10s") == 8640
    assert parse_speed("6h/1s") == 21600
    assert parse_speed("max") is None
    with pytest.raises(ValueError):
        parse_speed("fast")


def test_timeline_is_ordered_by_sim_time(stream_inputs):
    tx, lb = stream_inputs
    times = [t for t, *_ in build_timeline(tx, lb)]
    assert times == sorted(times)


def test_labels_selected_by_known_time(stream_inputs):
    tx, lb = stream_inputs
    start, end = pd.Timestamp("2020-07-01"), pd.Timestamp("2020-10-01")
    events = list(build_timeline(tx, lb, start, end))
    for t, _, kind, payload in events:
        assert start <= t < end
        if kind == "label":
            assert payload["label_known_at"] == t


def test_replay_routes_topics_and_never_leaks_label(stream_inputs):
    tx, lb = stream_inputs
    sink = MemorySink()
    stats = replay(tx, lb, sink, speed=None, max_events=5000)
    assert stats.invalid == 0 and stats.failed_deliveries == 0
    assert stats.transactions + stats.labels == 5000
    for m in sink.messages:
        body = json.loads(m["value"])
        assert "is_fraud" not in body
        if m["topic"] == TRANSACTIONS_TOPIC:
            assert m["key"] == body["card_id"]
            assert m["headers"]["event_type"] == "transaction"
        else:
            assert m["topic"] == LABELS_TOPIC
            assert m["key"] == body["transaction_id"]


def test_label_published_after_its_transaction(stream_inputs):
    tx, lb = stream_inputs
    sink = MemorySink()
    replay(tx, lb, sink, speed=None)
    seen_tx = set()
    for m in sink.messages:
        body = json.loads(m["value"])
        if m["topic"] == TRANSACTIONS_TOPIC:
            seen_tx.add(body["transaction_id"])
        else:
            assert body["transaction_id"] in seen_tx


def test_pacing_sleeps_in_proportion_to_sim_time(stream_inputs):
    tx, lb = stream_inputs
    clock_now = [0.0]
    slept = []

    def fake_sleep(s):
        slept.append(s)
        clock_now[0] += s

    stats = replay(
        tx,
        lb,
        MemorySink(),
        speed=86400.0,
        max_events=2000,  # 1 sim day per real second
        sleep=fake_sleep,
        clock=lambda: clock_now[0],
    )
    sim_days = (stats.sim_end - stats.sim_start).total_seconds() / 86400
    assert sum(slept) == pytest.approx(sim_days, rel=0.01)


def test_invalid_event_is_skipped_not_sent(stream_inputs):
    tx, lb = stream_inputs
    bad = tx.head(3).copy()
    bad.loc[bad.index[0], "amount"] = -1.0
    sink = MemorySink()
    stats = replay(bad, lb.iloc[0:0], sink, speed=None)
    assert stats.invalid == 1 and stats.transactions == 2
    assert len(sink.messages) == 2
