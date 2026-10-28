from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from charts import STATUS, line, ref_rule, show

from views.common import data_note, pct, require, usd


def render(b) -> None:
    st.title("Model performance and drift")
    data_note(b)
    if not require(b, "eval_strategies", "eval_curve"):
        return
    k = b.meta.get("operating_k")

    st.subheader(f"Strategies at {k} alerts/day (test period)")
    s = b["eval_strategies"]
    t = pd.DataFrame(
        {
            "Strategy": s["strategy"],
            "Alerts/day": s["alerts_per_day"].round(1),
            "Fraud caught": s["recall"].map(pct),
            "Precision": s["precision"].map(pct),
            "$ caught": s["amount_recall"].map(pct),
            "FP / 1k txns": s["fp_per_1000_txns"].round(1),
            "Net benefit": s["net_benefit"].map(usd),
        }
    )
    st.dataframe(
        t,
        hide_index=True,
        width="stretch",
        column_config={"Strategy": st.column_config.TextColumn(width="large")},
    )
    st.caption("Precision = share of alerts that were fraud. FP = false positives.")

    st.subheader("Fraud caught vs. alert quality by review budget")
    curve = b["eval_curve"]
    champ = curve[(curve["model"] == "champion") & (curve["policy"] == "threshold_capped")]
    long = champ.melt(
        id_vars="capacity_k", value_vars=["recall", "precision"], var_name="key", value_name="share"
    )
    names = {"recall": "Fraud caught", "precision": "Alerts that were fraud"}
    long["series"] = long["key"].map(names)
    chart = line(
        alt.Chart(long),
        "capacity_k:Q",
        "share:Q",
        "series",
        list(names.values()),
        x_title="Review budget (alerts/day)",
        y_title=None,
        y_format=".0%",
        x_zero=False,
        tooltip=[
            alt.Tooltip("capacity_k:Q", title="Budget"),
            "series:N",
            alt.Tooltip("share:Q", format=".1%"),
        ],
    )
    show(
        alt.layer(chart, ref_rule(k, label=f"chosen {k}/day")).properties(height=260),
        champ[
            [
                "capacity_k",
                "recall",
                "precision",
                "amount_recall",
                "fp_per_1000_txns",
                "net_benefit",
            ]
        ],
    )
    st.caption(
        "More reviewers catch more fraud, but past the sweet spot most extra alerts are "
        "legitimate customers. The capacity planner page lets you change the costs."
    )

    if not b["eval_intervals"].empty:
        with st.expander("95% confidence intervals (day-level bootstrap)"):
            iv = b["eval_intervals"].copy()
            st.dataframe(iv, hide_index=True, width="stretch")

    st.subheader("Performance by month")
    mp = b["monthly_perf"]
    if not mp.empty:
        long = mp.melt(
            id_vars=["month", "split"],
            value_vars=["recall", "precision"],
            var_name="key",
            value_name="share",
        )
        long["series"] = long["key"].map(names)
        chart = line(
            alt.Chart(long),
            "month:O",
            "share:Q",
            "series",
            list(names.values()),
            x_title=None,
            y_title=None,
            y_format=".0%",
            x_label_angle=0,
            tooltip=["month:O", "split:N", "series:N", alt.Tooltip("share:Q", format=".1%")],
        )
        show(chart.properties(height=240), mp)
        st.caption(
            "Fixed threshold, no daily cap. Validation months (Apr–Jun) vs. test months "
            "(Jul–Dec). The latest month's recall can move as late chargebacks arrive."
        )

    st.subheader("Input and score drift (PSI vs. validation)")
    dr = b["drift"]
    if not dr.empty:
        order = ["stable", "moderate", "major"]
        heat = alt.Chart(dr).encode(
            x=alt.X("month:O", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("feature:N", title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=["month:O", "feature:N", alt.Tooltip("psi:Q", format=".3f"), "status:N"],
        )
        rects = heat.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=order, range=[STATUS[o] for o in order]),
                legend=alt.Legend(orient="top", title="PSI status"),
            )
        )
        labels = heat.mark_text(fontSize=11, color="black").encode(
            text=alt.Text("psi:Q", format=".2f")
        )
        show(alt.layer(rects, labels).properties(height=alt.Step(30)), dr)
        by_design = ["card_prior_txn_count"]
        real = dr[~dr["feature"].isin(by_design)].sort_values("psi", ascending=False)
        worst = real.iloc[0] if len(real) else dr.iloc[0]
        st.caption(
            "PSI < 0.10 stable · 0.10–0.25 moderate · > 0.25 major. PSI needs no labels, so it "
            f"warns before chargebacks arrive. Largest shift excluding by-design drift: "
            f"**{worst['feature']}** in {worst['month']} (PSI {worst['psi']:.2f}, "
            f"{worst['status']})."
        )
        if dr["feature"].isin(by_design).any():
            st.caption(
                "`card_prior_txn_count` drifts **by design**: a card's history only grows, so "
                "later months always have more prior transactions than the reference period. "
                "It is expected, not a data problem, but it means the model sees values above "
                "its training range; capping or bucketing it is a candidate for the next retrain."
            )

    st.subheader("Fairness check (gender and age are not model inputs)")
    fr = b["eval_fairness"]
    if not fr.empty:
        st.dataframe(
            fr[
                [
                    "attribute",
                    "group",
                    "txns",
                    "fraud_rate",
                    "alert_rate",
                    "recall",
                    "false_positive_rate",
                    "alert_rate_ratio",
                    "fpr_ratio",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "fraud_rate": st.column_config.NumberColumn("Fraud rate", format="%.4f"),
                "alert_rate": st.column_config.NumberColumn("Alert rate", format="%.4f"),
                "recall": st.column_config.NumberColumn("Fraud caught", format="%.3f"),
                "false_positive_rate": st.column_config.NumberColumn("FP rate", format="%.4f"),
                "alert_rate_ratio": st.column_config.NumberColumn(
                    "Alert-rate ratio", format="%.2f"
                ),
                "fpr_ratio": st.column_config.NumberColumn("FP-rate ratio", format="%.2f"),
            },
        )
        st.caption(
            "Ratios compare each group with the overall rate; outside 0.8–1.25 "
            "warrants review. Differences partly reflect different fraud rates per group."
        )
