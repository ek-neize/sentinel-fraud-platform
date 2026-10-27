import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.metrics import (
    daily_topk_report,
    fixed_threshold_report,
    ranking_metrics,
    threshold_for_daily_capacity,
)


@pytest.fixture
def toy():
    # 3 days x 10 txns; 2 frauds per day with the highest scores on days 1-2,
    # and on day 3 one fraud is ranked low.
    rows = []
    for d in range(3):
        for i in range(10):
            fraud = int(i in (0, 1))
            score = 0.9 - 0.05 * i + 0.001 * d  # unique scores, no ties
            if d == 2 and i == 1:
                score = 0.01
            rows.append({"day": d, "score": score, "y": fraud, "amount": 100.0 * (1 + fraud)})
    return pd.DataFrame(rows)


def test_daily_topk(toy):
    rep = daily_topk_report(toy, "score", "y", k=2)
    assert rep["alerts"] == 6 and rep["alerts_per_day"] == 2
    assert rep["true_positives"] == 5 and rep["frauds"] == 6
    assert rep["precision"] == pytest.approx(5 / 6)
    assert rep["recall"] == pytest.approx(5 / 6)
    assert rep["fp_per_1000_txns"] == pytest.approx(1000 * 1 / 30)
    assert rep["amount_recall"] == pytest.approx(1000 / 1200)


def test_threshold_matches_average_capacity(toy):
    thr = threshold_for_daily_capacity(toy["score"], toy["day"], k=2)
    rep = fixed_threshold_report(toy, "score", "y", thr, k=2)
    assert rep["alerts"] == 6
    assert rep["days_over_capacity"] == 0


def test_fixed_threshold_counts_overflow():
    df = pd.DataFrame({"day": [0] * 5 + [1] * 5, "score": [0.9] * 5 + [0.1] * 5, "y": [1] * 10})
    rep = fixed_threshold_report(df, "score", "y", 0.5, k=2, amount_col=None)
    assert rep["days_over_capacity"] == 1 and rep["max_alerts_in_a_day"] == 5


def test_ranking_metrics_perfect_and_random():
    y = np.r_[np.ones(10), np.zeros(990)]
    assert ranking_metrics(y, y)["pr_auc"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert ranking_metrics(y, rng.random(1000))["pr_auc"] < 0.1


def test_threshold_never_exceeds_budget_with_ties():
    # 1000 txns over 10 days; 900 share the same (saturated) score.
    scores = np.r_[np.linspace(0.99, 0.5, 100), np.full(900, 1e-6)]
    days = np.repeat(np.arange(10), 100)
    thr = threshold_for_daily_capacity(scores, days, k=50)  # budget 500 > 100 distinct scores
    assert (scores >= thr).sum() <= 500
    assert (scores >= thr).sum() == 100
