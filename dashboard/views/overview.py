from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from charts import line, ref_rule, show

from views.common import data_note, md, pct, require, usd

SERIES = {"new_alerts": "New alerts", "reviewed": "Reviewed", "open_end": "Waiting in backlog"}


def render(b) -> None:
    meta = b.meta
    k = meta.get("operating_k")
    st.title("Sentinel · fraud alert operations")
    data_note(b)
    st.markdown(
        "**Question:** how can a financial institution catch suspicious transactions quickly "
        "while keeping unnecessary investigations manageable? Sentinel ranks every transaction "
        f"with an XGBoost model and sizes the alert queue to the team: **{k} reviews/day**, "
        f"overflow waits up to {meta.get('max_age_days')} days, highest risk first."
    )

    h = meta.get("headline", {})
    q = h.get("backlog_score") or h.get("capped") or {}
    cap = h.get("capped", {})
    tp = meta.get("test_period") or ["?", "?"]
    st.subheader(f"Held-out test period ({tp[0]} to {tp[1]})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Fraud caught",
        pct(q.get("recall")),
        delta=f"{100 * (q.get('recall', 0) - cap.get('recall', 0)):+.1f} pts vs hard cap"
        if cap
        else None,
    )
    c2.metric("Fraud dollars caught", pct(q.get("amount_recall")))
    c3.metric("Alerts that were fraud", pct(q.get("precision")))
    c4.metric("Net benefit vs no model", usd(q.get("net_benefit")))
    rule = h.get("rule", {})
    if rule:
        st.caption(
            md(
                f"Same {k}/day budget with the watchlist rule alone: {pct(rule.get('recall'))} of "
                f"fraud, {usd(rule.get('net_benefit'))}. Figures from notebook 06 (fixed threshold "
                "from validation, backlog queue by score)."
            )
        )

    st.subheader("Daily operations")
    if not require(b, "daily_ops"):
        return
    ops = b["daily_ops"]
    long = ops.melt(id_vars=["day"], value_vars=list(SERIES), var_name="key", value_name="alerts")
    long["series"] = long["key"].map(SERIES)
    chart = line(
        alt.Chart(long),
        "day:T",
        "alerts:Q",
        "series",
        list(SERIES.values()),
        x_title=None,
        y_title="Alerts per day",
        tooltip=[alt.Tooltip("day:T", title="Day"), "series:N", "alerts:Q"],
    )
    if k:
        chart = alt.layer(chart, ref_rule(k, axis="y", label=f"review budget {k}/day"))
    show(chart.properties(height=280), ops)

    last = ops.iloc[-1]
    day_s = f"{pd.Timestamp(last['day']):%Y-%m-%d}"
    fraud = int(last.get("confirmed_fraud", 0))
    st.caption(
        f"Last day ({day_s}): {int(last['new_alerts'])} new alerts, {int(last['reviewed'])} "
        f"reviewed ({fraud} confirmed fraud), {int(last['open_end'])} waiting, "
        f"{int(last['expired'])} expired."
    )
