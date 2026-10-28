from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from charts import STATUS, palette, show

from views.common import data_note, pct

# Same status colors as the drift heatmap; always shown with a text label.
COLORS = {"ok": STATUS["stable"], "warn": STATUS["moderate"], "critical": STATUS["major"]}
ORDER = {"ok": 0, "warn": 1, "critical": 2}
LABEL = {"ok": "✅ ok", "warn": "⚠️ warn", "critical": "⛔ critical"}


def _fmt_value(check: str, v: float) -> str:
    if pd.isna(v):
        return "–"
    if check in ("drift",):
        return f"{v:.3f}"
    if check in ("alert volume", "backlog"):
        return f"{v:.2f}×"
    return pct(v)


def render(b) -> None:
    st.title("Monitoring")
    data_note(b)
    checks, maturity = b["monitoring_checks"], b["label_maturity"]
    if checks.empty:
        st.info(
            "No monitoring data in this snapshot. In Databricks run `09_monitoring`, then "
            "`08_dashboard_snapshot`, and pull the snapshot again."
        )
        return

    last_day = checks["as_of"].max()
    latest = checks[checks["as_of"] == last_day].copy()
    worst = max(latest["status"], key=ORDER.get)
    n_warn, n_crit = (latest["status"] == "warn").sum(), (latest["status"] == "critical").sum()
    msg = (
        f"Status on {last_day:%Y-%m-%d}: **{worst.upper()}** · {n_crit} critical, "
        f"{n_warn} warning, {len(latest) - n_warn - n_crit} ok checks"
    )
    {"ok": st.success, "warn": st.warning, "critical": st.error}[worst](msg)
    st.caption(
        "Labels arrive late (customer reports within days, chargebacks up to 60 days), so "
        "the first warnings use what is known at once: drift, alert volume, backlog and "
        "missing inputs. Recall is checked only on the latest cohort whose labels are "
        "complete. Thresholds: `config/assumptions.yaml` → `monitoring`."
    )

    st.subheader("Checks on the latest run")
    not_ok = latest.sort_values("status", key=lambda s: -s.map(ORDER))
    table = pd.DataFrame(
        {
            "Status": not_ok["status"].map(LABEL),
            "Check": not_ok["check"],
            "Subject": not_ok["subject"],
            "Value": [
                _fmt_value(c, v) for c, v in zip(not_ok["check"], not_ok["value"], strict=True)
            ],
            "Detail": not_ok["detail"],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    st.subheader("Status over time (weekly backfill)")
    hist = checks.assign(rank=checks["status"].map(ORDER))
    worst_by = (
        hist.sort_values("rank")
        .groupby(["as_of", "check"], as_index=False)
        .last()[["as_of", "check", "status", "subject", "detail"]]
    )
    heat = (
        alt.Chart(worst_by)
        .mark_rect(stroke="white", strokeWidth=1, cornerRadius=2)
        .encode(
            # one column per run date (ordinal), so a check present on only some dates is
            # not stretched across the gaps
            x=alt.X(
                "yearmonthdate(as_of):O", title=None, axis=alt.Axis(format="%b %d", labelAngle=-45)
            ),
            y=alt.Y("check:N", title=None),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("as_of:T", title="As of"),
                "check:N",
                alt.Tooltip("subject:N", title="Worst item"),
                "status:N",
                "detail:N",
            ],
        )
        .properties(height=alt.Step(28))
    )
    show(heat, worst_by, table_label="Show status history")
    st.caption("Each cell shows the worst item of that check on that date.")

    if maturity.empty:
        return
    st.subheader("How much fraud is visible yet, by transaction age")
    m = maturity.copy()
    p = palette()
    bars = (
        alt.Chart(m)
        .mark_bar(color=p["s1"], cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            x=alt.X(
                "label_coverage:Q",
                title="Share of eventual fraud already labelled",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("age_band:N", title="Transaction age", sort=list(m["age_band"])),
            tooltip=[
                "age_band:N",
                alt.Tooltip("frauds_known:Q", title="Fraud labelled so far"),
                alt.Tooltip("frauds_final:Q", title="Fraud eventually labelled"),
                alt.Tooltip("label_coverage:Q", format=".0%", title="Labelled so far"),
                alt.Tooltip("recall_known:Q", format=".1%", title="Recall on labels so far"),
                alt.Tooltip("recall_final:Q", format=".1%", title="Recall on final labels"),
            ],
        )
        .properties(height=alt.Step(34))
    )
    show(bars, m, table_label="Show label-maturity table")
    young = m.iloc[0]
    st.caption(
        f"For transactions under a week old only {pct(young['label_coverage'], 0)} of the fraud "
        "that will eventually be reported is labelled yet, so performance on recent data is "
        "measured on an incomplete, biased sample (fast customer reports, not slow "
        "chargebacks). This is why recall is monitored on matured cohorts."
    )
