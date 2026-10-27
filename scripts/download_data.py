"""Download the Sparkov credit-card fraud dataset from Kaggle into data/raw/.

Prereqs:
  pip install -e ".[kaggle]"
  Kaggle API token at ~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token)

Usage:
  python scripts/download_data.py
"""

from __future__ import annotations

import sys
import zipfile

from sentinel.common.config import RAW_DIR
from sentinel.data.loader import RAW_FILES

DATASET = "kartik2112/fraud-detection"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / f).exists() for f in RAW_FILES):
        print(f"Raw files already present in {RAW_DIR}; nothing to do.")
        return 0
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("kaggle package missing: pip install -e '.[kaggle]'", file=sys.stderr)
        return 1

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {DATASET} -> {RAW_DIR}")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), quiet=False)
    for z in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        z.unlink()
    missing = [f for f in RAW_FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"Download finished but {missing} not found.", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
