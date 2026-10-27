"""Write a small mock dataset in raw Sparkov format (offline / CI use).

Usage:
  python scripts/make_mock_data.py [--out data/raw_mock]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentinel.common.config import DATA_DIR
from sentinel.data.mock import make_mock_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_DIR / "raw_mock"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = make_mock_raw()
    cut = pd.to_datetime(raw["trans_date_trans_time"]) < "2020-06-21"  # same cut as Kaggle files
    raw[cut].reset_index(drop=True).to_csv(out / "fraudTrain.csv")
    raw[~cut].reset_index(drop=True).to_csv(out / "fraudTest.csv")
    print(f"Wrote {len(raw):,} mock rows ({raw['is_fraud'].mean():.2%} fraud) to {out}")


if __name__ == "__main__":
    main()
