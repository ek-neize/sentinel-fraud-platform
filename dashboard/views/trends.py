from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from charts import line, palette, ref_rule, show

from views.common import data_note, pct, require


def render(b) -> None:
    st.title("Trends")
    data_note(b)
    if not require(b, "trend_daily", "trend_category"):
        return

    st.subheader("Fraud and alerts per day (7-day average)")
    d = b["trend_daily"].groupby("day", as_index=False)[["frauds", "alerts", "tp"]].sum()
    d = d.sort_values("day")
    for c in ("frauds", "alerts"):
        d[f"{c}_7d"] = d[c].rolling(7, min_periods=1).mean().round(1)
    long = d.melt(
        id_vars="day", value_vars=["frauds_7d", "alerts_7d"], var_name="key", value_name="n"
    )
    long["key"] = long["key"].str.replace("_7d", "", regex=False)
    names = {"frauds": "Confirmed fraud (all labels)", "alerts": "Alerts (score ≥ threshold)"}
    long["series"] = long["key"].map(names)
    chart = line(
        alt.Chart(long),
        "day:T",
        "n:Q",
        "series",
        list(names.values()),
        x_title=None,
        y_title="Per day (7-day average)",
        y_format=",.1f",
        tooltip=[alt.Tooltip("day:T", title="Day"), "series:N", "n:Q"],
    )
    split = b["trend_daily"]
    test_start = split.loc[split["split"] == "test", "day"].min()
    if pd.notna(test_start):
        chart = alt.layer(chart, ref_rule(test_start, axis="x", label="test period starts"))
    show(chart.properties(height=280), d)
    st.caption(
        "Fraud arrives in bursts: a compromised card makes ~10 fraudulent purchases in "
        "about two days, which is why daily alert volume swings."
    )

    st.subheader("By merchant category (test period)")
    cat = b["trend_category"].sort_values("fraud_rate", ascending=False)
    p = palette()
    bars = (
        alt.Chart(cat)
        .mark_bar(color=p["s1"], cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            x=alt.X("fraud_rate:Q", title="Fraud rate", axis=alt.Axis(format=".1%")),
            y=alt.Y("category:N", sort="-x", title=None),
            tooltip=[
                "category:N",
                alt.Tooltip("txns:Q", format=","),
                alt.Tooltip("frauds:Q", format=","),
                alt.Tooltip("fraud_rate:Q", format=".2%"),
                alt.Tooltip("alert_precision:Q", format=".1%", title="Alerts that were fraud"),
                alt.Tooltip("recall:Q", format=".1%", title="Fraud above threshold"),
            ],
        )
        .properties(height=alt.Step(24))
    )
    show(bars, cat)
    top = cat.iloc[0]
    st.caption(
        f"Highest fraud rate: **{top['category']}** ({pct(top['fraud_rate'], 2)} of "
        f"transactions). Online and point-of-sale shopping and grocery categories dominate."
    )

    st.subheader("By customer state (top 15 by fraud count, test period)")
    if b["trend_state"].empty:
        return
    st_ = b["trend_state"].sort_values("frauds", ascending=False)
    bars = (
        alt.Chart(st_)
        .mark_bar(color=p["s1"], cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            x=alt.X("frauds:Q", title="Frauds"),
            y=alt.Y("state:N", sort="-x", title=None),
            tooltip=[
                "state:N",
                alt.Tooltip("txns:Q", format=","),
                "frauds:Q",
                alt.Tooltip("fraud_rate:Q", format=".2%"),
            ],
        )
        .properties(height=alt.Step(22))
    )
    show(bars, st_)
