"""Build the case file for one alert: recorded facts (F*) and model signals (S*).

Point-in-time: only data that existed when the alert was raised is used. The card history
stops at the alerted transaction, and earlier alerts count as reviewed only if their
review finished on an earlier day. The alert's own disposition is never included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

NIGHT_START, NIGHT_END = 22, 4

# Reason codes that SOP-002 §3 / SOP-007 §2 treat as context only.
CONTEXT_ONLY_PATTERNS = (
    "city population",
    "customer state",
    "day of week",
    "card has ",
    "km from the customer",
    "spending pattern of this card",
)


@dataclass
class Item:
    id: str
    kind: str  # "fact" | "signal"
    text: str
    strength: str | None = None  # signals: "indicator" | "context only"


@dataclass
class CaseFile:
    alert_id: str
    transaction_id: str
    items: list[Item]
    features: dict = field(default_factory=dict)  # used by retrieval queries and gold rules

    @property
    def facts(self) -> list[Item]:
        return [i for i in self.items if i.kind == "fact"]

    @property
    def signals(self) -> list[Item]:
        return [i for i in self.items if i.kind == "signal"]

    @property
    def by_id(self) -> dict[str, Item]:
        return {i.id: i for i in self.items}

    def context_only_ids(self) -> set[str]:
        return {i.id for i in self.signals if i.strength == "context only"}

    def to_dict(self) -> dict:
        return {**asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> CaseFile:
        return cls(
            d["alert_id"],
            d["transaction_id"],
            [Item(**i) for i in d["items"]],
            d.get("features", {}),
        )

    def query(self) -> list[str]:
        """Retrieval queries: one per fact and signal, in the case's own words."""
        return [i.text for i in self.items]


def money(x: float) -> str:
    return f"${x:,.2f}"


def risk_band(score: float, threshold: float) -> str:
    if score >= max(0.9, threshold):
        return "High"
    return "Elevated" if score >= threshold else "Low"


def is_context_only(reason: str) -> bool:
    r = reason.lower()
    return any(p in r for p in CONTEXT_ONLY_PATTERNS)


def _ago(delta: pd.Timedelta) -> str:
    s = delta.total_seconds()
    if s < 3600:
        return f"{s / 60:.0f} minutes"
    if s < 86400 * 2:
        return f"{s / 3600:.1f} hours"
    return f"{s / 86400:.1f} days"


def _is_night(hour: int) -> bool:
    return hour >= NIGHT_START or hour < NIGHT_END


def build_case(
    alert: pd.Series | dict,
    history: pd.DataFrame,
    all_alerts: pd.DataFrame | None = None,
    history_days: int = 7,
) -> CaseFile:
    """alert: one row of the alerts table. history: the card's transactions (columns
    transaction_id, event_ts, merchant_name, category, amount, score), any extent; it is
    cut to the `history_days` before the alert here. all_alerts: alerts table (optional)."""
    a = pd.Series(alert)
    ts = pd.Timestamp(a["event_ts"])
    thr = float(a["threshold"])
    h = history.copy()
    h["event_ts"] = pd.to_datetime(h["event_ts"])
    h = h[(h["event_ts"] < ts) & (h["event_ts"] >= ts - pd.Timedelta(days=history_days))]
    h = h[h["transaction_id"] != a["transaction_id"]].sort_values("event_ts")
    h24 = h[h["event_ts"] >= ts - pd.Timedelta(hours=24)]
    cat = str(a["category"])
    amount = float(a["amount"])
    reasons = [
        str(a[c])
        for c in ("reason_1", "reason_2", "reason_3")
        if c in a and pd.notna(a[c]) and a[c]
    ]
    hot = any("hot-list" in r and "not on" not in r for r in reasons)

    facts: list[str] = [
        f"Alerted transaction {a['transaction_id']}: {money(amount)} at {a['merchant_name']} "
        f"(category {cat}) on {ts:%Y-%m-%d %H:%M} UTC."
    ]
    if cat.endswith("_net"):
        facts.append(f"Category {cat} is an online (card-not-present) category.")
    else:
        facts.append(f"Category {cat} is not an online category (the card was present).")
    if len(h):
        facts.append(
            f"In the {history_days} days before this transaction the card made {len(h)} other "
            f"transactions totalling {money(h['amount'].sum())}; average "
            f"{money(h['amount'].mean())}, largest {money(h['amount'].max())}."
        )
    else:
        facts.append(
            f"No other transactions on this card in the {history_days} days before this one "
            "(in the available records)."
        )
    if len(h24):
        rows = "; ".join(
            f"{r.event_ts:%m-%d %H:%M} {r.merchant_name} ({r.category}) {money(r.amount)}"
            for r in h24.sort_values("event_ts", ascending=False).head(10).itertuples()
        )
        facts.append(
            f"In the 24 hours before this transaction the card made {len(h24)} transactions "
            f"totalling {money(h24['amount'].sum())} at {h24['merchant_name'].nunique()} "
            f"different merchants. Most recent first: {rows}."
        )
    else:
        facts.append("No other transactions on this card in the 24 hours before this one.")
    if len(h):
        p = h.iloc[-1]
        facts.append(
            f"The previous transaction on this card was {_ago(ts - p['event_ts'])} earlier "
            f"({money(p['amount'])} at {p['merchant_name']}, {p['category']})."
        )
        ratio = amount / h["amount"].mean() if h["amount"].mean() > 0 else float("nan")
        cmp = "larger than" if amount > h["amount"].max() else "not larger than"
        facts.append(
            f"The alerted amount is {ratio:.1f} times the card's {history_days}-day average and "
            f"{cmp} its largest purchase in that period."
        )
    else:
        ratio = float("nan")
    night = _is_night(ts.hour)
    facts.append(
        f"The transaction time ({ts:%H:%M} UTC) is {'within' if night else 'outside'} the "
        "22:00-04:00 night window."
    )
    if hot:
        facts.append(
            "The card was on the fraud hot-list when the transaction was made (watchlist "
            "record; the listing reason and date are not in the case data)."
        )
    prior_conf = 0
    if all_alerts is not None and len(all_alerts):
        same = all_alerts[
            (all_alerts["card_id"] == a["card_id"]) & (pd.to_datetime(all_alerts["event_ts"]) < ts)
        ]
        today = ts.normalize()
        for r in same.sort_values("event_ts", ascending=False).head(5).itertuples():
            rts = pd.Timestamp(r.event_ts)
            done = (
                r.status == "reviewed"
                and pd.notna(r.status_day)
                and pd.Timestamp(r.status_day) < today
            )
            if done:
                prior_conf += r.disposition == "confirmed_fraud"
                facts.append(
                    f"Earlier alert on this card ({rts:%Y-%m-%d %H:%M} UTC, {money(r.amount)} "
                    f"at {r.merchant_name}) was reviewed on {pd.Timestamp(r.status_day):%Y-%m-%d}"
                    f" with disposition {r.disposition}."
                )
            else:
                facts.append(
                    f"Earlier alert on this card ({rts:%Y-%m-%d %H:%M} UTC, {money(r.amount)} "
                    f"at {r.merchant_name}) had not been reviewed when this alert was raised."
                )

    score = float(a["score"])
    band = risk_band(score, thr)
    signals: list[tuple[str, str | None]] = [
        (
            f"Model score {score:.3f} against operating threshold {thr:.4f}: risk band {band} "
            f"(model version {a.get('model_version', '?')}).",
            "indicator",
        )
    ]
    for i, r in enumerate(reasons, 1):
        strength = "context only" if is_context_only(r) else "indicator"
        signals.append((f"Reason code {i} (raised the score): {r}.", strength))
    if len(h24) and "score" in h24:
        m = int((h24["score"] >= thr).sum())
        signals.append(
            (
                f"{m} of the card's {len(h24)} other transactions in the previous 24 hours also "
                "scored above the threshold.",
                "indicator",
            )
        )

    items = [Item(f"F{i}", "fact", t) for i, t in enumerate(facts, 1)]
    items += [Item(f"S{i}", "signal", t, s) for i, (t, s) in enumerate(signals, 1)]
    features = {
        "category": cat,
        "online": cat.endswith("_net"),
        "amount": amount,
        "n_7d": int(len(h)),
        "n_24h": int(len(h24)),
        "spend_24h": float(h24["amount"].sum()) if len(h24) else 0.0,
        "ratio_to_avg": None if pd.isna(ratio) else float(ratio),
        "larger_than_max": bool(len(h) and amount > h["amount"].max()),
        "night": bool(night),
        "hot_list": bool(hot),
        "context_only_reason": any(is_context_only(r) for r in reasons),
        "prior_confirmed": int(prior_conf),
        "score": score,
        "band": band,
    }
    return CaseFile(str(a["alert_id"]), str(a["transaction_id"]), items, features)


def render_case(case: CaseFile) -> str:
    """Case file as the model sees it."""
    lines = ["VERIFIED FACTS (recorded data):"]
    lines += [f"[{i.id}] {i.text}" for i in case.facts]
    lines += ["", "MODEL SIGNALS (model outputs, not evidence):"]
    lines += [f"[{i.id}] ({i.strength}) {i.text}" for i in case.signals]
    return "\n".join(lines)
