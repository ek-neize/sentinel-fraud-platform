import numpy as np
import pandas as pd
import pytest

from sentinel.alerts.queue import (
    STATUS_EXPIRED,
    STATUS_REVIEWED,
    priority,
    queue_stats,
    reviewed_mask,
    simulate_queue,
    step_queue,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["event_ts", "score", "amount", "label_final"]).assign(
        event_ts=lambda d: pd.to_datetime(d["event_ts"])
    )


def test_overflow_waits_instead_of_being_lost():
    # Day 1: 5 alerts, capacity 2 -> 3 wait. Day 2-3: nothing new -> backlog cleared.
    df = _df([("2020-07-01 10:00", 0.9 - i / 100, 100, 1) for i in range(5)])
    q = simulate_queue(df, df["score"], threshold=0.5, k=2, max_age_days=3)
    assert (q["status"] == STATUS_REVIEWED).all()
    assert sorted(q["wait_days"].tolist()) == [0, 0, 1, 1, 2]
    assert reviewed_mask(len(df), q).all()


def test_alerts_expire_after_max_age():
    df = _df([("2020-07-01 10:00", 0.9 - i / 100, 100, 1) for i in range(5)])
    q = simulate_queue(df, df["score"], threshold=0.5, k=1, max_age_days=2)
    assert (q["status"] == STATUS_REVIEWED).sum() == 2  # one per day for 2 days
    assert (q["status"] == STATUS_EXPIRED).sum() == 3
    stats = queue_stats(q, df["label_final"])
    assert stats["frauds_expired"] == 3 and stats["max_wait_days"] == 1


def test_higher_priority_new_alert_jumps_the_backlog():
    df = _df(
        [
            ("2020-07-01 10:00", 0.6, 100, 0),
            ("2020-07-01 11:00", 0.7, 100, 0),
            ("2020-07-02 09:00", 0.99, 100, 1),
        ]
    )
    q = simulate_queue(df, df["score"], threshold=0.5, k=1, max_age_days=3)
    day2 = q[q["status_day"] == pd.Timestamp("2020-07-02")]
    assert day2["row"].tolist() == [2]  # the new 0.99 alert is reviewed first


def test_expected_loss_priority_prefers_big_amounts():
    p = priority([0.5, 0.4], [100, 1000], mode="expected_loss")
    assert p[1] > p[0]
    with pytest.raises(ValueError):
        priority([0.5], mode="nope")


def test_step_queue_never_reviews_more_than_k():
    new = pd.DataFrame(
        {
            "alert_id": range(10),
            "created_day": pd.Timestamp("2020-07-01"),
            "priority": np.linspace(1, 0, 10),
        }
    )
    out = step_queue(new.iloc[0:0], new, pd.Timestamp("2020-07-01"), k=4, max_age_days=3)
    assert (out["status"] == STATUS_REVIEWED).sum() == 4
    assert out.loc[out["status"] == STATUS_REVIEWED, "alert_id"].tolist() == [0, 1, 2, 3]
