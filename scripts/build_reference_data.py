"""Build the canonical transactions table plus synthetic label feed and watchlist.

Outputs (parquet):
  data/reference/transactions.parquet   canonical, PII-reduced, with `split`
  data/synthetic/label_feed.parquet     fraud labels with label_known_at
  data/synthetic/watchlist.parquet      leakage-safe card/merchant watchlist

These are the *inputs* to the streaming pipeline: the Kafka producer replays
transactions.parquet (without is_fraud) and label_feed.parquet on separate
topics, and the watchlist is loaded as a Delta reference table.

Usage:
  python scripts/build_reference_data.py [--raw data/raw] [--mock]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentinel.common.config import DATA_DIR, RAW_DIR, REFERENCE_DIR, SYNTHETIC_DIR, load_assumptions
from sentinel.data.labels import simulate_label_feed
from sentinel.data.loader import assign_split, read_raw, to_canonical
from sentinel.data.watchlist import build_watchlist, point_in_time_join, signal_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=None, help="Folder with fraudTrain/fraudTest.csv")
    parser.add_argument("--mock", action="store_true", help="Use data/raw_mock")
    args = parser.parse_args()

    raw_dir = Path(args.raw) if args.raw else (DATA_DIR / "raw_mock" if args.mock else RAW_DIR)
    cfg = load_assumptions()
    seed = cfg["random_seed"]

    print(f"Reading raw CSVs from {raw_dir}")
    txns = to_canonical(read_raw(raw_dir))
    txns["split"] = assign_split(txns["event_time"], cfg["split"])

    labels = simulate_label_feed(txns, cfg["labels"]["channels"], seed)
    watchlist = build_watchlist(txns, labels, cfg["watchlist"], seed)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    # Spark cannot read pandas' default nanosecond timestamps: write microseconds.
    pq = {"index": False, "coerce_timestamps": "us", "allow_truncated_timestamps": True}
    txns.to_parquet(REFERENCE_DIR / "transactions.parquet", **pq)
    labels.to_parquet(SYNTHETIC_DIR / "label_feed.parquet", **pq)
    watchlist.to_parquet(SYNTHETIC_DIR / "watchlist.parquet", **pq)

    # Is the watchlist a useful-but-imperfect signal? Target: lift ~3-10x, captures 15-25%.
    joined = point_in_time_join(txns, watchlist)
    print(f"Transactions: {len(txns):,}  fraud rate: {txns['is_fraud'].mean():.3%}")
    print(txns["split"].value_counts().to_string())
    print(f"Label feed rows: {len(labels):,}")
    print(labels["label_source"].value_counts().to_string())
    print(f"Watchlist entries: {len(watchlist):,}")
    print(watchlist.groupby(["entity_type", "source"]).size().to_string())
    print("\nWatchlist signal (all data):")
    print(signal_report(joined, by=None).round(4).to_string())


if __name__ == "__main__":
    main()
