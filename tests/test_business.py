import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.business import (
    age_band,
    bootstrap_ci,
    choose_operating_point,
    daily_aggregates,
    fairness_report,
    flat_costs,
    policy_rule,
    policy_threshold_capped,
    policy_topk,
    summarize,
)

COSTS = {
    "investigation_cost_per_alert": 25.0,
    "escalation_cost_per_case": 75.0,
    "missed_fraud_loss_rate": 0.85,
    "missed_fraud_fixed_cost": 15.0,
    "false_positive_friction_cost": 5.0,
    "caught_fraud_recovery_rate": 0.90,
}


@pytest.fixture
def days():
    # 4 days x 50 txns; 2 frauds/day worth $500 each; scores rank frauds first except day 3.
    rows = []
    for d in range(4):
        for i in range(50):
            fraud = int(i < 2)
            score = 1.0 - i / 100 + d / 1000
            if d == 3 and i == 1:
                score = 0.0
            rows.append(
                {
                    "event_ts": pd.Timestamp("2020-07-01") + pd.Timedelta(days=d, minutes=i),
                    "label_final": fraud,
                    "amount": 500.0 if fraud else 20.0,
                    "score": score,
                    "wl_any_hit": int(i in (0, 10, 11)),
                    "gender": "F" if i % 2 else "M",
                }
            )
    return pd.DataFrame(rows)


def test_cost_model_by_hand(days):
    flagged = policy_topk(days, days["score"], k=3)  # 12 alerts, 7 frauds caught of 8
    s = summarize(days, flagged, COSTS)
    assert (s["alerts"], s["tp"], s["fp"], s["frauds"]) == (12, 7, 5, 8)
    assert s["cost_review"] == 12 * 25
    assert s["cost_escalation"] == 7 * 75
    assert s["cost_friction"] == 5 * 5
    missed = 1 * 500 * 0.85 + 1 * 15
    caught_leak = 7 * 500 * 0.10 * 0.85
    assert s["cost_fraud_loss"] == pytest.approx(missed + caught_leak)
    assert s["cost_do_nothing"] == pytest.approx(8 * 500 * 0.85 + 8 * 15)
    assert s["net_benefit"] == pytest.approx(s["cost_do_nothing"] - s["cost_total"])


def test_capped_policy_never_exceeds_budget(days):
    flagged = policy_threshold_capped(days, days["score"], threshold=0.0, k=5)
    per_day = pd.Series(flagged).groupby(days["event_ts"].dt.floor("D").to_numpy()).sum()
    assert (per_day <= 5).all() and per_day.sum() == 20


def test_rule_policy_capped_by_amount(days):
    assert policy_rule(days, "wl_any_hit").sum() == 12
    capped = policy_rule(days, "wl_any_hit", k=1)
    assert capped.sum() == 4
    assert days.loc[capped, "label_final"].all()  # the $500 fraud wins each day


def test_bootstrap_interval_brackets_estimate(days):
    flagged = policy_topk(days, days["score"], k=3)
    agg = daily_aggregates(days, flagged)
    ci = bootstrap_ci(agg, COSTS, n=500, seed=1)
    point = summarize(days, flagged, COSTS)
    for m in ("recall", "precision", "net_benefit"):
        lo, hi = ci[m]
        assert lo <= point[m] <= hi


def test_choose_operating_point_prefers_smaller_on_tie():
    curve = pd.DataFrame(
        {"capacity_k": [5, 10, 20], "policy": "threshold_capped", "net_benefit": [100, 150, 150]}
    )
    assert choose_operating_point(curve) == 10


def test_fairness_parity_and_bands(days):
    flagged = policy_topk(days, days["score"], k=3)
    fr = fairness_report(days, flagged, {"gender": days["gender"]})
    assert set(fr["group"]) == {"F", "M"}
    assert fr["txns"].sum() == len(days)
    bands = age_band(pd.Series([18, 29, 30, 44, 45, 59, 60, 90]), [0, 30, 45, 60])
    assert bands.tolist() == ["<30", "<30", "30-44", "30-44", "45-59", "45-59", "60+", "60+"]


def test_end_to_end_report_on_mock(txns, labels, watchlist, cfg):
    from sentinel.evaluation.report import run_business_evaluation
    from sentinel.features.training_set import build_training_frame
    from sentinel.models.training import ExperimentConfig, run_experiments

    pytest.importorskip("xgboost")
    df = build_training_frame(txns, labels, watchlist, cfg["split"])
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    _, models = run_experiments(
        tr,
        va,
        [
            ExperimentConfig("champion", "xgb", "full", {"n_estimators": 80, "max_depth": 3}),
            ExperimentConfig("baseline", "logreg", "full", {"C": 0.1}),
        ],
    )
    scores = {
        n: {
            "valid": m["pipeline"].predict_proba(va)[:, 1],
            "test": m["pipeline"].predict_proba(te)[:, 1],
        }
        for n, m in models.items()
    }
    r = run_business_evaluation(va, te, scores, cfg, bootstrap_n=100)
    assert r.operating_k in cfg["evaluation"]["capacities"]
    assert 0 <= r.headline["recall"] <= 1
    assert r.headline["recall_ci"][0] <= r.headline["recall"] <= r.headline["recall_ci"][1]
    assert "## Headline" in r.markdown and "Fairness" in r.markdown
    assert set(r.fairness["attribute"]) == {"gender", "age_band"}
    # thresholds came from validation: test alerts/day need not equal k exactly
    assert set(r.thresholds) == {"champion", "baseline"}
    assert (
        flat_costs(cfg)["caught_fraud_recovery_rate"]
        == cfg["detection"]["caught_fraud_recovery_rate"]
    )
    assert np.isfinite(r.headline["net_benefit"])
