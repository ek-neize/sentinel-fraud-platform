"""Data profile: writes docs/data_profile.md from the reference tables.

Answers the questions that shape later steps:
- How rare is fraud, and is the rate stable over time (drift baseline)?
- Which categories / states / hours concentrate fraud?
- How does fraud amount differ from legitimate amount?
- How many transactions and frauds per day (grounds the alert-capacity choice)?
- How much signal does the watchlist carry, per split?

Usage:
  python scripts/profile_data.py
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from sentinel.common.config import REFERENCE_DIR, REPO_ROOT, SYNTHETIC_DIR, load_assumptions
from sentinel.data.watchlist import point_in_time_join, signal_report


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def main() -> None:
    cfg = load_assumptions()
    tx = pd.read_parquet(REFERENCE_DIR / "transactions.parquet")
    wl = pd.read_parquet(SYNTHETIC_DIR / "watchlist.parquet")
    tx = point_in_time_join(tx, wl)
    tx["day"] = tx["event_time"].dt.floor("D")
    tx["month"] = tx["event_time"].dt.to_period("M").astype(str)
    tx["hour"] = tx["event_time"].dt.hour

    s = []
    s.append(
        f"# Sentinel data profile\n\nGenerated {datetime.now(UTC).date()} "
        "from `transactions.parquet`.\n"
    )
    if tx["card_id"].nunique() < 500:
        s.append(
            "> ⚠️ This looks like the **mock** sample. Re-run on the Kaggle data before "
            "drawing conclusions.\n"
        )

    # Overview
    daily = tx.groupby("day").agg(txns=("transaction_id", "size"), frauds=("is_fraud", "sum"))
    overview = pd.DataFrame(
        {
            "value": [
                f"{len(tx):,}",
                f"{tx['event_time'].min():%Y-%m-%d} → {tx['event_time'].max():%Y-%m-%d}",
                f"{tx['card_id'].nunique():,}",
                f"{tx['merchant_id'].nunique():,}",
                f"{tx['category'].nunique()}",
                f"{tx['is_fraud'].mean():.3%}",
                f"{daily['txns'].median():,.0f}",
                f"{daily['frauds'].median():,.1f}",
                f"{(tx.loc[tx.is_fraud == 1, 'card_id'].nunique() / tx['card_id'].nunique()):.1%}",
            ]
        },
        index=[
            "Transactions",
            "Date range",
            "Cards",
            "Merchants",
            "Categories",
            "Fraud rate",
            "Median txns / day",
            "Median frauds / day",
            "Cards with ≥1 fraud",
        ],
    )
    s.append("## Overview\n\n" + overview.to_markdown() + "\n")

    cap = cfg["capacity"]["alerts_per_day"]
    cap_tight = cfg["capacity"]["alerts_per_day_tight"]
    avg_tx, avg_fr = daily["txns"].mean(), daily["frauds"].mean()
    lines = [
        "## Alert capacity in context\n",
        f"Average day: {avg_tx:,.0f} transactions, {avg_fr:,.1f} frauds.\n",
    ]
    for c in (cap, cap_tight):
        recall_ceiling = min(1.0, c / avg_fr) if avg_fr > 0 else float("nan")
        lines.append(
            f"- **{c} alerts/day** = top {min(1.0, c / avg_tx):.1%} of daily traffic; "
            f"precision ceiling ≈ {min(1.0, avg_fr / c):.1%}, recall ceiling ≈ {recall_ceiling:.1%}"
        )
    s.append("\n".join(lines) + "\n")

    # Split
    split = tx.groupby("split", observed=True).agg(
        rows=("transaction_id", "size"),
        frauds=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
        start=("event_time", "min"),
        end=("event_time", "max"),
    )
    s.append("## Chronological split\n\n" + _md(split, ".4f") + "\n")

    # Monthly trend (drift baseline)
    monthly = tx.groupby("month").agg(
        txns=("transaction_id", "size"),
        fraud_rate=("is_fraud", "mean"),
        avg_amount=("amount", "mean"),
    )
    monthly["txns"] = monthly["txns"].map("{:,}".format)
    s.append("## Monthly trend (drift baseline)\n\n" + _md(monthly, ".4f") + "\n")

    # Category
    cat = (
        tx.groupby("category", observed=True)
        .agg(
            txns=("transaction_id", "size"),
            frauds=("is_fraud", "sum"),
            fraud_rate=("is_fraud", "mean"),
        )
        .sort_values("fraud_rate", ascending=False)
    )
    cat["share_of_fraud"] = cat["frauds"] / cat["frauds"].sum()
    s.append("## Fraud by merchant category\n\n" + _md(cat, ".4f") + "\n")

    # State (top 10 by fraud count)
    st = (
        tx.groupby("state", observed=True)
        .agg(
            txns=("transaction_id", "size"),
            frauds=("is_fraud", "sum"),
            fraud_rate=("is_fraud", "mean"),
        )
        .sort_values("frauds", ascending=False)
        .head(10)
    )
    s.append("## Fraud by state (top 10 by count)\n\n" + _md(st, ".4f") + "\n")

    # Hour of day
    hr = tx.groupby("hour").agg(fraud_rate=("is_fraud", "mean"), frauds=("is_fraud", "sum"))
    s.append("## Fraud rate by hour of day\n\n" + _md(hr.T, ".4f") + "\n")

    # Amounts
    amt = tx.groupby("is_fraud")["amount"].describe(percentiles=[0.5, 0.9, 0.99])
    amt.index = ["legit", "fraud"]
    s.append("## Amount distribution (USD)\n\n" + _md(amt, ".2f") + "\n")

    # Fraud burst structure (drives watchlist & velocity-feature design)
    fb = tx[tx.is_fraud == 1].groupby("card_id")["event_time"].agg(["min", "max", "size"])
    burst = pd.DataFrame(
        {
            "frauds_per_card": fb["size"].describe(),
            "burst_span_days": ((fb["max"] - fb["min"]).dt.total_seconds() / 86400).describe(),
        }
    )
    s.append("## Fraud burst structure (per compromised card)\n\n" + _md(burst, ".2f") + "\n")

    # Watchlist signal per split
    wls = signal_report(tx, by="split")
    s.append(
        "## Watchlist signal (point-in-time join)\n\n"
        "Goal: a useful-but-imperfect signal (lift well above 1, far from a perfect label).\n\n"
        + _md(wls, ".4f")
        + "\n"
    )

    out = REPO_ROOT / "docs" / "data_profile.md"
    out.write_text("\n".join(s), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
