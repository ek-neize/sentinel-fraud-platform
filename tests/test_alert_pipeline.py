import pandas as pd
import pytest

from sentinel.alerts.pipeline import ScoringModel, empty_queue, process_day
from sentinel.alerts.queue import simulate_queue

pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def setup(txns, labels, watchlist, cfg):
    from sentinel.features.training_set import build_training_frame
    from sentinel.models.training import ExperimentConfig, run_experiments

    df = build_training_frame(txns, labels, watchlist, cfg["split"])
    df = df.merge(txns[["transaction_id", "merchant_name"]], on="transaction_id")
    tr, va = df[df.split == "train"], df[df.split == "valid"]
    _, m = run_experiments(
        tr, va, [ExperimentConfig("x", "xgb", "full", {"n_estimators": 60, "max_depth": 3})]
    )
    test = df[df.split == "test"].reset_index(drop=True)
    pipe = m["x"]["pipeline"]
    thr = float(pd.Series(pipe.predict_proba(va)[:, 1]).quantile(0.995))
    return pipe, thr, test


def test_daily_operations_match_offline_simulation(setup):
    pipe, thr, test = setup
    model = ScoringModel(pipe, thr, "t", k=3, max_age_days=3)
    labels = test.set_index("transaction_id")["label_final"]
    days = pd.date_range(test["event_ts"].min().normalize(), periods=20, freq="D")
    window = test[test["event_ts"] < days[-1] + pd.Timedelta(days=1)].reset_index(drop=True)

    q = empty_queue()
    summaries = []
    for d in days:
        today = window[window["event_ts"].dt.normalize() == d]
        q, s, digest = process_day(d, today, q, model, labels)
        summaries.append(s)
        assert s["reviewed"] <= 3
        assert digest["status"].isin(["reviewed", "open"]).all()

    # the operational loop and the offline simulator agree on what got reviewed
    sim = simulate_queue(window, pipe.predict_proba(window)[:, 1], thr, k=3, max_age_days=3)
    sim_reviewed = set(window.loc[sim.loc[sim["status"] == "reviewed", "row"], "transaction_id"])
    ops_reviewed = set(q.loc[q["status"] == "reviewed", "transaction_id"])
    sim_done_by_last_day = {
        t
        for t, d in zip(window.loc[sim["row"], "transaction_id"], sim["status_day"], strict=True)
        if d <= days[-1]
    } & sim_reviewed
    assert ops_reviewed == sim_done_by_last_day

    assert q["alert_id"].is_unique
    assert (
        q.loc[q["status"] == "reviewed", "disposition"].isin(["confirmed_fraud", "cleared"])
    ).all()
    new_alerts = sum(s["new_alerts"] for s in summaries)
    assert len(q) == new_alerts
    assert q.loc[q["status"] == "reviewed", "reason_1"].notna().all()
