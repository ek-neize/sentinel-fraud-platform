from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st
from charts import palette, show

from views.common import DISCLAIMER, data_note, md, require, risk_band

STATUS_ORDER = ["open", "reviewed", "expired"]


def _queue_table(alerts: pd.DataFrame, threshold: float) -> pd.DataFrame:
    t = alerts.copy()
    t["risk"] = [risk_band(s, threshold) for s in t["score"]]
    t["card"] = "…" + t["card_id"].astype(str).str[-4:]
    return t


def render(b) -> None:
    st.title("Alert queue")
    data_note(b)
    if not require(b, "alerts"):
        return
    alerts = b["alerts"]
    threshold = float(b.meta.get("model", {}).get("threshold", 0.5))

    days = sorted(alerts["created_day"].dropna().dt.normalize().unique(), reverse=True)
    f1, f2, f3 = st.columns([1.2, 1.6, 1.2])
    day_options = ["All days", *[f"{d:%Y-%m-%d}" for d in days]]
    day = f1.selectbox(
        "Alerts created on",
        day_options,
        index=1 if len(day_options) > 1 else 0,
        help="Defaults to the latest day; pick 'All days' for the full history.",
    )
    status = f2.multiselect("Status", STATUS_ORDER, default=["open", "reviewed"])
    min_score = f3.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)

    view = alerts[alerts["status"].isin(status) & (alerts["score"] >= min_score)]
    if day != "All days":
        view = view[view["created_day"].dt.strftime("%Y-%m-%d") == day]
    status_rank = view["status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    view = _queue_table(
        view.assign(_r=status_rank)
        .sort_values(["_r", "priority"], ascending=[True, False])
        .drop(columns="_r"),
        threshold,
    )
    st.caption(
        f"{len(view):,} alerts · open first, then highest priority · select a row to open the case"
    )

    cols = [
        "risk",
        "score",
        "amount",
        "merchant_name",
        "category",
        "card",
        "event_ts",
        "status",
        "wait_days",
        "disposition",
        "reason_1",
    ]
    event = st.dataframe(
        view[cols],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="queue_table",
        column_config={
            "risk": "Risk",
            "score": st.column_config.NumberColumn("Score", format="%.3f"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "merchant_name": "Merchant",
            "category": "Category",
            "card": "Card",
            "event_ts": st.column_config.DatetimeColumn("Time (UTC)", format="YYYY-MM-DD HH:mm"),
            "status": "Status",
            "wait_days": st.column_config.NumberColumn("Waited (days)", format="%d"),
            "disposition": "Finding",
            "reason_1": "Top reason",
        },
    )
    if view.empty:
        return
    rows = event.selection.rows if event and event.selection else []
    case = view.iloc[rows[0] if rows else 0]
    _case_panel(b, case, threshold)


def _case_panel(b, a: pd.Series, threshold: float) -> None:
    st.divider()
    st.subheader(md(f"Case · ${a['amount']:,.2f} at {a['merchant_name']}"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Recorded facts** (from the transaction and the case log)")
        st.markdown(
            f"- Time: {pd.Timestamp(a['event_ts']):%Y-%m-%d %H:%M} UTC\n"
            f"- Merchant: {a['merchant_name']} ({a['category']})\n"
            f"- Card: …{str(a['card_id'])[-4:]}\n"
            f"- Alert status: **{a['status']}**"
            + (f", waited {int(a['wait_days'])} day(s)" if pd.notna(a.get("wait_days")) else "")
            + (
                f"\n- Investigator finding: **{a['disposition']}**"
                if pd.notna(a.get("disposition")) and a.get("disposition")
                else ""
            )
        )
    with right:
        st.markdown("**Model signals** (why it was flagged; not verified)")
        st.markdown(
            f"- Risk: **{risk_band(a['score'], threshold)}** · score {a['score']:.3f} "
            f"(alert threshold {threshold:.4f})"
        )
        for c in ("reason_1", "reason_2", "reason_3"):
            if pd.notna(a.get(c)) and a.get(c):
                st.markdown(md(f"- {a[c]}"))
    st.caption(DISCLAIMER)
    _brief_panel(b, a)

    hist = b["card_history"]
    hist = hist[hist["alert_id"] == a["alert_id"]] if not hist.empty else hist
    st.markdown("**Card activity in the 7 days up to this transaction**")
    if hist.empty:
        st.info("No card history in this snapshot.")
        return
    p = palette()
    h = hist.assign(
        kind=hist["is_alerted_txn"].map({True: "This transaction", False: "Earlier transactions"})
    )
    names = ["Earlier transactions", "This transaction"]
    chart = (
        alt.Chart(h)
        .mark_circle(size=90, opacity=0.9, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("event_ts:T", title=None),
            y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=names, range=[p["s1"], p["s2"]]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("event_ts:T", title="Time", format="%Y-%m-%d %H:%M"),
                "merchant_name:N",
                "category:N",
                alt.Tooltip("amount:Q", format="$,.2f"),
                alt.Tooltip("score:Q", format=".3f", title="Risk score"),
            ],
        )
        .properties(height=240)
    )
    show(chart, hist.drop(columns=["alert_id"]), table_label="Show card transactions")


def _brief_panel(b, a: pd.Series) -> None:
    """AI-drafted brief (precomputed by scripts/generate_briefs.py; no API key needed here)."""
    brief = b.brief_for(a["alert_id"])
    with st.expander("Investigation brief (AI draft)", expanded=brief is not None, icon="📝"):
        if brief is None:
            st.caption(
                "No AI brief for this alert. Run `python scripts/generate_briefs.py` after "
                "pulling a snapshot."
            )
            return
        if brief.get("checks_passed"):
            st.success("Automatic checks passed: citations, numbers, no verdict.", icon="✅")
        else:
            st.warning(
                f"Automatic checks flagged {int(brief.get('issues', 0))} issue(s); "
                "read with care.",
                icon="⚠️",
            )
        st.markdown(md(brief["markdown"]))
        sections = json.loads(brief.get("cited_sections") or "[]")
        if sections:
            st.markdown("**Cited procedure sections**")
            for sec in sections:
                st.markdown(md(f"`{sec['id']}` {sec['label'].split(' ', 2)[-1]}"))
                st.caption(md(sec["text"]))
