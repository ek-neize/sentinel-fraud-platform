"""Replay historical transactions and delayed labels into Kafka as a live stream.

Two event streams are merged into one simulated timeline:
- transactions at `event_time`        -> sentinel.transactions.v1 (key = card_id)
- fraud labels at `label_known_at`    -> sentinel.fraud_labels.v1 (key = transaction_id)

So a label is published only when it would have become known, never with the
transaction. `is_fraud` never appears on the transactions topic.

Pacing: `speed` = simulated seconds per real second (e.g. 8640 = 1 day per 10 s).
`speed=None` sends as fast as the broker accepts (backfill mode).
"""

from __future__ import annotations

import heapq
import logging
import math
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd

from sentinel.common.events import (
    LABELS_TOPIC,
    SCHEMA_VERSION,
    TRANSACTIONS_TOPIC,
    FraudLabelEvent,
    TransactionEvent,
)
from sentinel.streaming.sinks import Sink

log = logging.getLogger(__name__)

TX_FIELDS = [f for f in TransactionEvent.model_fields if f not in ("schema_version", "currency")]
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_speed(text: str) -> float | None:
    """'1d/10s' -> 8640.0 ; '6h/1s' -> 21600.0 ; 'max' -> None (no sleeping)."""
    text = text.strip().lower()
    if text in ("max", "inf", "none"):
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])/(\d+(?:\.\d+)?)([smhd])", text)
    if not m:
        raise ValueError(f"Bad speed '{text}'. Use e.g. 1d/10s, 6h/1s or max.")
    sim = float(m[1]) * _UNITS[m[2]]
    real = float(m[3]) * _UNITS[m[4]]
    return sim / real


@dataclass
class ReplayStats:
    transactions: int = 0
    labels: int = 0
    invalid: int = 0
    failed_deliveries: int = 0
    sim_start: pd.Timestamp | None = None
    sim_end: pd.Timestamp | None = None
    wall_seconds: float = 0.0
    invalid_examples: list[str] = field(default_factory=list)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _tx_events(tx: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, int, str, dict]]:
    for row in tx[TX_FIELDS].itertuples(index=False):
        d = {k: _clean(v) for k, v in zip(TX_FIELDS, row, strict=True)}
        for k in ("category", "state", "gender", "zip", "job", "city"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        yield d["event_time"], 0, "tx", d


def _label_events(labels: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, int, str, dict]]:
    cols = ["transaction_id", "event_time", "label_known_at", "label", "label_source"]
    for row in labels[cols].itertuples(index=False):
        d = {k: _clean(v) for k, v in zip(cols, row, strict=True)}
        yield d["label_known_at"], 1, "label", d


def build_timeline(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> Iterator[tuple[pd.Timestamp, int, str, dict]]:
    """Merge both streams in simulated-time order within [start, end).

    Labels are selected by when they become known, so a label for a transaction
    from before `start` can still arrive inside the window (as in real life).
    """
    tx = transactions
    lb = labels
    if start is not None:
        tx = tx[tx["event_time"] >= start]
        lb = lb[lb["label_known_at"] >= start]
    if end is not None:
        tx = tx[tx["event_time"] < end]
        lb = lb[lb["label_known_at"] < end]
    tx = tx.sort_values(["event_time", "transaction_id"], kind="stable")
    lb = lb.sort_values(["label_known_at", "transaction_id"], kind="stable")
    return heapq.merge(_tx_events(tx), _label_events(lb), key=lambda e: (e[0], e[1]))


def replay(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    sink: Sink,
    speed: float | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_events: int | None = None,
    log_every: int = 50_000,
    sleep=time.sleep,
    clock=time.monotonic,
) -> ReplayStats:
    stats = ReplayStats()
    wall0 = clock()
    sim0 = None
    headers_tx = {"schema_version": SCHEMA_VERSION, "event_type": "transaction"}
    headers_lb = {"schema_version": SCHEMA_VERSION, "event_type": "fraud_label"}

    for sim_time, _, kind, payload in build_timeline(transactions, labels, start, end):
        if max_events is not None and stats.transactions + stats.labels >= max_events:
            break
        if sim0 is None:
            sim0 = sim_time
            stats.sim_start = sim_time
        stats.sim_end = sim_time

        if speed is not None:
            due = wall0 + (sim_time - sim0).total_seconds() / speed
            wait = due - clock()
            if wait > 0:
                sink.poll()
                sleep(wait)

        try:
            if kind == "tx":
                event = TransactionEvent(**payload)
                sink.send(
                    TRANSACTIONS_TOPIC, event.card_id, event.model_dump_json().encode(), headers_tx
                )
                stats.transactions += 1
            else:
                event = FraudLabelEvent(**payload)
                sink.send(
                    LABELS_TOPIC, event.transaction_id, event.model_dump_json().encode(), headers_lb
                )
                stats.labels += 1
        except Exception as exc:  # contract violation: skip and record, never send bad data
            stats.invalid += 1
            if len(stats.invalid_examples) < 5:
                stats.invalid_examples.append(f"{kind} {payload.get('transaction_id')}: {exc}")
            continue

        sent = stats.transactions + stats.labels
        if sent % 1000 == 0:
            sink.poll()
        if log_every and sent % log_every == 0:
            log.info("sent %s events | sim time %s", f"{sent:,}", sim_time)

    stats.failed_deliveries = sink.flush()
    stats.wall_seconds = clock() - wall0
    return stats
