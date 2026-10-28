"""Where produced events go: Confluent Kafka, a JSONL file, or memory (tests)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


class Sink(Protocol):
    def send(self, topic: str, key: str, value: bytes, headers: dict[str, str]) -> None: ...
    def poll(self) -> None: ...
    def flush(self) -> int:
        """Block until delivered; return the number of failed deliveries."""


class MemorySink:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send(self, topic, key, value, headers):
        self.messages.append({"topic": topic, "key": key, "value": value, "headers": headers})

    def poll(self):
        pass

    def flush(self):
        return 0


class FileSink:
    """Write one JSON line per message: handy for dry runs without Kafka."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def send(self, topic, key, value, headers):
        self._fh.write(
            json.dumps({"topic": topic, "key": key, "headers": headers, "value": json.loads(value)})
            + "\n"
        )

    def poll(self):
        pass

    def flush(self):
        self._fh.close()
        return 0


class KafkaSink:
    """Idempotent Confluent Cloud producer (SASL_SSL / PLAIN)."""

    def __init__(self, bootstrap: str, api_key: str, api_secret: str) -> None:
        from confluent_kafka import Producer  # imported lazily so tests don't need it

        self.failed = 0
        self.delivered = 0
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap,
                "security.protocol": "SASL_SSL",
                "sasl.mechanisms": "PLAIN",
                "sasl.username": api_key,
                "sasl.password": api_secret,
                "client.id": "sentinel-producer",
                "enable.idempotence": True,  # no retry duplicates; needs IDEMPOTENT_WRITE ACL
                "acks": "all",
                "compression.type": "zstd",
                "linger.ms": 50,
                "batch.size": 262144,
            }
        )

    @classmethod
    def from_env(cls) -> KafkaSink:
        missing = [
            v
            for v in ("KAFKA_BOOTSTRAP", "KAFKA_PRODUCER_KEY", "KAFKA_PRODUCER_SECRET")
            if not os.environ.get(v)
        ]
        if missing:
            raise RuntimeError(f"Missing {missing}. Put them in .env (see .env.example).")
        return cls(
            os.environ["KAFKA_BOOTSTRAP"],
            os.environ["KAFKA_PRODUCER_KEY"],
            os.environ["KAFKA_PRODUCER_SECRET"],
        )

    def _on_delivery(self, err, msg) -> None:
        if err is not None:
            self.failed += 1
            if self.failed <= 10:
                log.error("Delivery failed for key=%s: %s", msg.key(), err)
        else:
            self.delivered += 1

    def send(self, topic, key, value, headers):
        while True:
            try:
                self._producer.produce(
                    topic,
                    key=key.encode(),
                    value=value,
                    headers=list(headers.items()),
                    on_delivery=self._on_delivery,
                )
                return
            except BufferError:  # local queue full: let it drain, then retry
                self._producer.poll(0.5)

    def poll(self):
        self._producer.poll(0)

    def flush(self):
        remaining = self._producer.flush(60)
        if remaining:
            log.error("%d messages still undelivered after flush timeout", remaining)
        return self.failed + remaining
