from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from charts import line, ref_rule, show
from costs import COST_FIELDS, costs

from views.common import data_note, md, pct, require, usd

SUMS = ["txns", "frauds", "fraud_amt", "alerts", "tp", "fp", "caught_amt"]
MODELS = {"champion": "XGBoost champion", "baseline": "Logistic regression"}


def _recompute(curve: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    for _, r in curve.iterrows():
        s = {k: float(r[k]) for k in SUMS}
        rows.append({**r.to_dict(), **costs(s, c)})
    return pd.DataFrame(rows)


def render(b) -> None:
    st.title("Capacity planner")
    data_note(b)
    if not require(b, "eval_curve"):
        return
    meta = b.meta
    default_costs = meta.get("costs", {})
    k_op = int(meta.get("operating_k", 20))
    curve = b["eval_curve"]
    curve = curve[curve["policy"] == "threshold_capped"].copy()
    ks = sorted(curve["capacity_k"].unique().tolist())

    st.markdown(
        "How big should the review team be? Each budget uses one score threshold fixed on "
        "validation data and reviews at most that many alerts a day (test period, Jul–Dec 2020). "
        "Change the cost assumptions to see where the sweet spot moves."
    )
    with st.expander("Cost assumptions", expanded=False):
        cols = st.columns(3)
        c = {}
        for i, (key, label) in enumerate(COST_FIELDS.items()):
            v = float(default_costs.get(key, 0.0))
            is_rate = "rate" in key
            c[key] = cols[i % 3].number_input(
                label,
                value=v,
                min_value=0.0,
                max_value=1.0 if is_rate else 10_000.0,
                step=0.05 if is_rate else 5.0,
                key=f"cost_{key}",
            )
    k = st.select_slider(
        "Review budget (alerts/day)", options=ks, value=k_op if k_op in ks else ks[len(ks) // 2]
    )

    cur = _recompute(curve, c)
    champ = cur[cur["model"] == "champion"].set_index("capacity_k")
    row, op = champ.loc[k], champ.loc[k_op] if k_op in champ.index else champ.loc[k]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Fraud caught",
        pct(row["recall"]),
        delta=f"{100 * (row['recall'] - op['recall']):+.1f} pts vs {k_op}/day",
    )
    m2.metric("Fraud dollars caught", pct(row["amount_recall"]))
    m3.metric("Alerts that were fraud", pct(row["precision"]))
    m4.metric(
        "Net benefit (6 months)",
        usd(row["net_benefit"]),
        delta=f"{usd(row['net_benefit'] - op['net_benefit'])} vs {k_op}/day",
    )
    st.caption(
        md(
            f"Review cost {usd(row['cost_review'])} · remaining fraud loss "
            f"{usd(row['cost_fraud_loss'])} · {row['fp_per_1000_txns']:.1f} false "
            "positives per 1,000 transactions."
        )
    )

    best_k = int(champ["net_benefit"].idxmax())
    long = cur.assign(series=cur["model"].map(MODELS))
    chart = line(
        alt.Chart(long),
        "capacity_k:Q",
        "net_benefit:Q",
        "series",
        [MODELS["champion"], MODELS["baseline"]],
        x_title="Review budget (alerts/day)",
        y_title="Net benefit vs no model ($)",
        y_format="$,.0f",
        x_zero=False,
        tooltip=[
            alt.Tooltip("capacity_k:Q", title="Budget"),
            "series:N",
            alt.Tooltip("net_benefit:Q", format="$,.0f"),
            alt.Tooltip("recall:Q", format=".1%", title="Fraud caught"),
        ],
    )
    chart = alt.layer(chart, ref_rule(k, label=f"selected {k}/day"))
    show(
        chart.properties(height=300),
        cur[
            [
                "model",
                "capacity_k",
                "recall",
                "precision",
                "amount_recall",
                "cost_review",
                "cost_fraud_loss",
                "net_benefit",
            ]
        ],
    )
    st.markdown(f"With these costs the best budget is **{best_k} alerts/day**.")

    q = meta.get("headline", {}).get("backlog_score")
    if q and k == k_op:
        st.info(
            f"At {k_op}/day the deployed **backlog queue** (overflow waits up to "
            f"{meta.get('max_age_days')} days) catches {pct(q['recall'])} of fraud vs "
            f"{pct(row['recall'])} with a hard daily cap (default costs)."
        )
