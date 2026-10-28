"""Load the Sparkov Kaggle CSVs and map them to Sentinel's canonical schema.

Known quirks of the source data, handled here:
- `unix_time` does NOT match `trans_date_trans_time` (it is offset by years),
  so `trans_date_trans_time` is the source of truth for event time.
- Every merchant name carries a `fraud_` prefix; it is meaningless and stripped.
- The CSVs include an unnamed index column and PII-style fields (name, street)
  that are dropped. `dob` is kept only to derive age at transaction time.
- The provided train/test files are concatenated; Sentinel applies its own
  chronological split from config/assumptions.yaml.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

RAW_FILES = ("fraudTrain.csv", "fraudTest.csv")

PII_COLUMNS = ["first", "last", "street"]

CANONICAL_COLUMNS = [
    "transaction_id",
    "event_time",
    "card_id",
    "merchant_id",
    "merchant_name",
    "category",
    "amount",
    "gender",
    "age_at_txn",
    "job",
    "city",
    "state",
    "zip",
    "cust_lat",
    "cust_long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


def _stable_id(value: str, prefix: str, length: int = 12) -> str:
    """Deterministic pseudonymous id (so raw card numbers never leave bronze)."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}"


def read_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Read and concatenate the raw Kaggle CSVs exactly as downloaded."""
    raw_dir = Path(raw_dir)
    missing = [f for f in RAW_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Run scripts/download_data.py first "
            "(or scripts/make_mock_data.py for a small offline sample)."
        )
    frames = [pd.read_csv(raw_dir / f, dtype={"cc_num": str, "zip": str}) for f in RAW_FILES]
    return pd.concat(frames, ignore_index=True)


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw Sparkov columns to the canonical, PII-reduced schema."""
    df = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.drop(columns=PII_COLUMNS, errors="ignore")

    event_time = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])
    merchant_name = df["merchant"].str.removeprefix("fraud_")

    out = pd.DataFrame(
        {
            "transaction_id": df["trans_num"].astype(str),
            "event_time": event_time,
            "card_id": df["cc_num"].astype(str).map(lambda v: _stable_id(v, "card_")),
            "merchant_id": merchant_name.map(lambda v: _stable_id(v, "mer_", 10)),
            "merchant_name": merchant_name,
            "category": df["category"].astype("category"),
            "amount": df["amt"].astype(float),
            "gender": df["gender"].astype("category"),
            "age_at_txn": ((event_time - dob).dt.days / 365.25).round(1),
            "job": df["job"],
            "city": df["city"],
            "state": df["state"].astype("category"),
            "zip": df["zip"].astype(str).str.zfill(5),
            "cust_lat": df["lat"].astype(float),
            "cust_long": df["long"].astype(float),
            "city_pop": df["city_pop"].astype(int),
            "merch_lat": df["merch_lat"].astype(float),
            "merch_long": df["merch_long"].astype(float),
            "is_fraud": df["is_fraud"].astype("int8"),
        }
    )
    out = out.drop_duplicates(subset="transaction_id")
    return out.sort_values(["event_time", "transaction_id"]).reset_index(drop=True)[
        CANONICAL_COLUMNS
    ]


def assign_split(event_time: pd.Series, split_cfg: dict) -> pd.Series:
    """Chronological split label: train / gap / valid / test / out_of_range."""
    bounds = {k: pd.Timestamp(v) for k, v in split_cfg.items()}
    conditions = [
        (event_time >= bounds["train_start"]) & (event_time < bounds["train_end"]),
        (event_time >= bounds["train_end"]) & (event_time < bounds["gap_end"]),
        (event_time >= bounds["gap_end"]) & (event_time < bounds["valid_end"]),
        (event_time >= bounds["valid_end"]) & (event_time < bounds["test_end"]),
    ]
    labels = ["train", "gap", "valid", "test"]
    result = pd.Series("out_of_range", index=event_time.index, dtype="object")
    for cond, label in zip(conditions, labels, strict=True):
        result[cond] = label
    return result.astype("category")
