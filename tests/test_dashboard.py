"""The dashboard renders every page from a mock snapshot without errors.

Uses Streamlit's headless AppTest. The snapshot is built with the same code as
notebook 08 (sentinel.dashboard_export) so schema drift between them fails here.
"""

import json
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("xgboost")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASH = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = ["overview", "queue", "trends", "model", "monitoring", "planner"]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_mock_snapshot import build

    out = tmp_path_factory.mktemp("snapshot")
    build(str(out), days=10, bootstrap=50)
    return out


def test_snapshot_has_every_table(snapshot):
    sys.path.insert(0, str(DASH))
    from data import TABLES

    from sentinel.dashboard_export import TABLES as EXPORT_TABLES

    assert TABLES == EXPORT_TABLES
    for t in TABLES:
        assert (snapshot / f"{t}.parquet").exists(), t
    meta = json.loads((snapshot / "meta.json").read_text())
    assert {"model", "operating_k", "costs", "headline"} <= set(meta)
    assert sum(f.stat().st_size for f in snapshot.iterdir()) < 5_000_000  # small enough to commit


def test_snapshot_has_no_row_level_protected_attributes(snapshot):
    import pandas as pd

    for f in snapshot.glob("*.parquet"):
        if f.stem == "eval_fairness":
            continue
        cols = set(pd.read_parquet(f).columns)
        assert not cols & {"gender", "age_at_txn"}, f.stem


def test_card_history_is_point_in_time(snapshot):
    import pandas as pd

    alerts = pd.read_parquet(snapshot / "alerts.parquet")
    hist = pd.read_parquet(snapshot / "card_history.parquet")
    m = hist.merge(alerts[["alert_id", "event_ts"]], on="alert_id", suffixes=("", "_alert"))
    assert (m["event_ts"] <= m["event_ts_alert"]).all()
    assert m.groupby("alert_id")["is_alerted_txn"].sum().eq(1).all()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(snapshot, page, monkeypatch):
    monkeypatch.setenv("SENTINEL_SNAPSHOT_DIR", str(snapshot))
    code = f"""
import sys; sys.path.insert(0, {str(DASH)!r})
from data import load_bundle
from views import {page}
{page}.render(load_bundle())
"""
    at = AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception, at.exception[0].value


def test_app_entrypoint_and_queue_selection(snapshot, monkeypatch):
    monkeypatch.setenv("SENTINEL_SNAPSHOT_DIR", str(snapshot))
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.startswith("Sentinel")


def test_planner_reacts_to_budget(snapshot, monkeypatch):
    monkeypatch.setenv("SENTINEL_SNAPSHOT_DIR", str(snapshot))
    code = f"""
import sys; sys.path.insert(0, {str(DASH)!r})
from data import load_bundle
from views import planner
planner.render(load_bundle())
"""
    at = AppTest.from_string(code, default_timeout=60).run()
    before = at.metric[0].value
    slider = at.select_slider[0]
    slider.set_value(slider.options[0]).run()
    assert not at.exception
    assert at.metric[0].value != before or len(slider.options) == 1


def test_missing_snapshot_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_SNAPSHOT_DIR", str(tmp_path))
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No dashboard data found" in w.value for w in at.warning)


def test_dashboard_cost_model_matches_evaluation():
    sys.path.insert(0, str(DASH))
    from costs import costs as dash_costs

    from sentinel.common.config import load_assumptions
    from sentinel.evaluation.business import _costs, flat_costs

    c = flat_costs(load_assumptions())
    rng = random.Random(0)
    for _ in range(20):
        frauds = rng.randint(0, 500)
        tp = rng.randint(0, frauds)
        s = {
            "txns": 10_000,
            "frauds": frauds,
            "fraud_amt": rng.uniform(0, 1e5),
            "alerts": tp + rng.randint(0, 300),
            "tp": tp,
            "fp": rng.randint(0, 300),
        }
        s["caught_amt"] = s["fraud_amt"] * rng.random()
        assert dash_costs(s, c) == pytest.approx(_costs(s, c))


def test_write_snapshot_ignores_unserializable_attrs(tmp_path):
    """Databricks serverless toPandas() leaves PlanMetrics objects in df.attrs."""
    import pandas as pd

    from sentinel.dashboard_export import write_snapshot

    df = pd.DataFrame({"a": [1, 2]})
    df.attrs["metrics"] = [object()]
    write_snapshot({"t": df}, {"x": 1}, tmp_path)
    assert pd.read_parquet(tmp_path / "t.parquet")["a"].tolist() == [1, 2]
    assert df.attrs  # caller's frame untouched


def test_case_view_shows_ai_brief(snapshot, tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_briefs import generate

    briefs = tmp_path / "briefs.parquet"
    df = generate("template", snapshot=snapshot, out=briefs, workers=1)
    assert len(df) and df["checks_passed"].all()
    monkeypatch.setenv("SENTINEL_SNAPSHOT_DIR", str(snapshot))
    code = f"""
import sys; sys.path.insert(0, {str(DASH)!r})
from data import load_bundle
from views import queue
queue.render(load_bundle())
"""
    monkeypatch.setenv("SENTINEL_BRIEFS_FILE", str(briefs))
    at = AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception
    assert any("Automatic checks passed" in s.value for s in at.success)
    assert any("Verified facts" in m.value for m in at.markdown)
    assert any("SOP-0" in m.value for m in at.markdown)

    monkeypatch.setenv("SENTINEL_BRIEFS_FILE", str(tmp_path / "none.parquet"))
    at = AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception
    assert any("No AI brief" in c.value for c in at.caption)


def test_new_snapshot_files_bypass_the_cache(snapshot, tmp_path, monkeypatch):
    """Regression: the cache was keyed by folder only, so a pulled snapshot stayed stale."""
    import shutil

    sys.path.insert(0, str(DASH))
    from data import load_bundle

    snap = tmp_path / "snap"
    shutil.copytree(snapshot, snap)
    (snap / "monitoring_checks.parquet").unlink()
    monkeypatch.setenv("SENTINEL_SNAPSHOT_DIR", str(snap))
    assert load_bundle()["monitoring_checks"].empty
    shutil.copy(snapshot / "monitoring_checks.parquet", snap)  # new snapshot pulled
    assert not load_bundle()["monitoring_checks"].empty
