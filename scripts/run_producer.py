"""Replay transactions + delayed fraud labels into Confluent Kafka.

Examples
  # Dry run to a file (no Kafka), first 10k events
  python scripts/run_producer.py --sink file --max-events 10000

  # Backfill the training history as fast as possible
  python scripts/run_producer.py --start 2019-01-01 --end 2020-07-01 --speed max

  # "Live" demo of the test period: 1 simulated day every 10 seconds
  python scripts/run_producer.py --start 2020-07-01 --end 2021-01-01 --speed 1d/10s

Re-running a window sends duplicates; bronze keeps them and silver de-duplicates
on transaction_id, so replays are safe.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from sentinel.common.config import REFERENCE_DIR, REPO_ROOT, SYNTHETIC_DIR
from sentinel.streaming.producer import parse_speed, replay
from sentinel.streaming.sinks import FileSink, KafkaSink


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sink", choices=["kafka", "file"], default="kafka")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "replay" / "events.jsonl"))
    p.add_argument("--start", help="Simulated start (inclusive), e.g. 2020-07-01")
    p.add_argument("--end", help="Simulated end (exclusive), e.g. 2021-01-01")
    p.add_argument("--speed", default="max", help="e.g. 1d/10s, 6h/1s or max (default)")
    p.add_argument("--max-events", type=int)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tx = pd.read_parquet(REFERENCE_DIR / "transactions.parquet").drop(columns=["is_fraud", "split"])
    labels = pd.read_parquet(SYNTHETIC_DIR / "label_feed.parquet")

    if args.sink == "kafka":
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        sink = KafkaSink.from_env()
    else:
        sink = FileSink(args.out)

    stats = replay(
        tx,
        labels,
        sink,
        speed=parse_speed(args.speed),
        start=pd.Timestamp(args.start) if args.start else None,
        end=pd.Timestamp(args.end) if args.end else None,
        max_events=args.max_events,
    )
    print(
        f"\nSent {stats.transactions:,} transactions and {stats.labels:,} labels "
        f"({stats.sim_start} -> {stats.sim_end}) in {stats.wall_seconds:,.1f}s"
    )
    print(f"Invalid (skipped): {stats.invalid}   Failed deliveries: {stats.failed_deliveries}")
    for ex in stats.invalid_examples:
        print("  invalid:", ex)
    if args.sink == "file":
        print(f"Wrote {args.out}")
    return 1 if stats.failed_deliveries else 0


if __name__ == "__main__":
    sys.exit(main())
