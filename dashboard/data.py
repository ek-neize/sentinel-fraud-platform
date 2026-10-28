"""Where the dashboard's data comes from: a committed snapshot or live Databricks tables.

SENTINEL_DATA_MODE=snapshot (default)  Parquet files in dashboard/snapshot/ (or
                                        SENTINEL_SNAPSHOT_DIR). Used by the public app.
SENTINEL_DATA_MODE=live                 `dash_*` Delta tables via a Databricks SQL
                                        warehouse (DATABRICKS_WAREHOUSE_ID). Used by the
                                        Databricks App; works locally with a CLI profile.

Both sources return the same tables, written by notebooks/databricks/08_dashboard_snapshot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

TABLES = [
    "alerts", "daily_ops", "card_history", "trend_daily", "trend_category", "trend_state",
    "monthly_perf", "drift", "eval_curve", "eval_strategies", "eval_intervals", "eval_fairness",
]  # fmt: skip

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "snapshot"
# AI-drafted case briefs (scripts/generate_briefs.py); used in both modes.
DEFAULT_BRIEFS = Path(__file__).resolve().parent / "briefs" / "briefs.parquet"


@dataclass
class Bundle:
    tables: dict[str, pd.DataFrame]
    meta: dict
    mode: str
    source: str
    missing: list[str] = field(default_factory=list)
    briefs: pd.DataFrame = field(default_factory=pd.DataFrame)

    def brief_for(self, alert_id: str) -> dict | None:
        if self.briefs.empty:
            return None
        hit = self.briefs[self.briefs["alert_id"] == alert_id]
        return hit.iloc[0].to_dict() if len(hit) else None

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())

    @property
    def ok(self) -> bool:
        return not self.missing and bool(self.meta)


def _fix_types(name: str, df: pd.DataFrame) -> pd.DataFrame:
    for c in ("event_ts", "created_day", "status_day", "day"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if getattr(df[c].dt, "tz", None) is not None:
                df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def _load_snapshot(folder: Path) -> Bundle:
    tables, missing = {}, []
    for name in TABLES:
        f = folder / f"{name}.parquet"
        if f.exists():
            tables[name] = _fix_types(name, pd.read_parquet(f))
        else:
            missing.append(name)
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    if not meta:
        missing.append("meta.json")
    return Bundle(tables, meta, "snapshot", str(folder), missing)


def _load_live(catalog: str, schema: str, warehouse_id: str) -> Bundle:
    from databricks import sql
    from databricks.sdk.core import Config

    cfg = Config()  # Databricks App service principal, or the local CLI profile
    tables, missing = {}, []
    with (
        sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate,
        ) as conn,
        conn.cursor() as cur,
    ):
        for name in TABLES:
            try:
                cur.execute(f"SELECT * FROM {catalog}.{schema}.dash_{name}")
                tables[name] = _fix_types(name, cur.fetchall_arrow().to_pandas())
            except Exception:  # noqa: BLE001 - show which table is missing, keep going
                missing.append(f"dash_{name}")
        try:
            cur.execute(f"SELECT json FROM {catalog}.{schema}.dash_meta")
            meta = json.loads(cur.fetchone()[0])
        except Exception:  # noqa: BLE001
            meta, missing = {}, [*missing, "dash_meta"]
    return Bundle(tables, meta, "live", f"{catalog}.{schema} via warehouse {warehouse_id}", missing)


def _load_briefs(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    except Exception:  # noqa: BLE001 - briefs are optional
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner="Loading data…")
def _load_cached(mode: str, location: tuple, briefs_path: str) -> Bundle:
    b = _load_live(*location) if mode == "live" else _load_snapshot(Path(location[0]))
    b.briefs = _load_briefs(Path(briefs_path))
    return b


def load_bundle(mode: str | None = None) -> Bundle:
    """Cached per data location, so switching snapshot folder or schema reloads."""
    mode = (mode or os.environ.get("SENTINEL_DATA_MODE", "snapshot")).lower()
    if mode == "live":
        location = (
            os.environ.get("SENTINEL_CATALOG", "workspace"),
            os.environ.get("SENTINEL_SCHEMA", "sentinel"),
            os.environ["DATABRICKS_WAREHOUSE_ID"],
        )
    else:
        location = (str(os.environ.get("SENTINEL_SNAPSHOT_DIR", DEFAULT_SNAPSHOT)),)
    briefs = os.environ.get("SENTINEL_BRIEFS_FILE", str(DEFAULT_BRIEFS))
    return _load_cached(mode, location, briefs)
