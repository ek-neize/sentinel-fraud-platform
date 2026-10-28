from __future__ import annotations

import pandas as pd
import streamlit as st

DISCLAIMER = (
    "Risk scores and reasons are model signals, not verified facts. "
    "An investigator reviews the evidence and makes the final decision."
)


def pct(x, digits: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def usd(x) -> str:
    if x is None or pd.isna(x):
        return "–"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def md(text: str) -> str:
    """Escape '$' so Streamlit markdown does not render amounts as LaTeX math."""
    return str(text).replace("$", "\\$")


def risk_band(score: float, threshold: float) -> str:
    if score >= max(0.9, threshold):
        return "High"
    if score >= threshold:
        return "Elevated"
    return "Low"


def data_note(b) -> None:
    meta = b.meta
    mock = " · **mock data**" if meta.get("mock") else ""
    when = meta.get("generated_at", "?")
    src = "live tables" if b.mode == "live" else f"snapshot generated {when}"
    st.caption(
        f"Source: {src}{mock} · model {meta.get('model', {}).get('name', '')} "
        f"v{meta.get('model', {}).get('version', '?')} · synthetic data (Sparkov), demo only"
    )


def require(b, *names: str) -> bool:
    empty = [n for n in names if b[n].empty]
    if empty:
        st.info(f"No data yet for: {', '.join(empty)}. Run notebook 07/08 in Databricks.")
        return False
    return True
