"""Sentinel investigation dashboard.

streamlit run dashboard/app.py                         # committed snapshot
SENTINEL_SNAPSHOT_DIR=/tmp/snap streamlit run dashboard/app.py
SENTINEL_DATA_MODE=live DATABRICKS_WAREHOUSE_ID=... streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_bundle  # noqa: E402
from views import model, monitoring, overview, planner, queue, trends  # noqa: E402

st.set_page_config(page_title="Sentinel · fraud operations", page_icon="🛡️", layout="wide")

bundle = load_bundle()

if not bundle.meta:
    st.title("Sentinel · fraud alert operations")
    st.warning(
        f"No dashboard data found ({bundle.mode}: {bundle.source}).\n\n"
        "Run notebook `08_dashboard_snapshot` in Databricks, then "
        "`python scripts/pull_snapshot.py` and commit `dashboard/snapshot/`. "
        "For a local demo: `python scripts/build_mock_snapshot.py --out /tmp/snap` and set "
        "`SENTINEL_SNAPSHOT_DIR=/tmp/snap`."
    )
    st.stop()
if bundle.missing:
    st.sidebar.warning(f"Missing: {', '.join(bundle.missing)}")


def _page(fn, title, icon, default=False):
    return st.Page(
        lambda: fn(bundle),
        title=title,
        icon=icon,
        default=default,
        url_path=title.lower().replace(" ", "-").replace("&", "and"),
    )


nav = st.navigation(
    [
        _page(overview.render, "Overview", "🏠", default=True),
        _page(queue.render, "Alert queue", "🚨"),
        _page(trends.render, "Trends", "📈"),
        _page(model.render, "Model & drift", "🧪"),
        _page(monitoring.render, "Monitoring", "🩺"),
        _page(planner.render, "Capacity planner", "🧮"),
    ]
)
with st.sidebar:
    st.caption(
        "Synthetic data (Sparkov) · demo only.\n\n"
        "[Source code](https://github.com/ek-neize/sentinel-fraud-platform)"
    )
nav.run()
