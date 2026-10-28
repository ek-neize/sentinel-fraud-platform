import numpy as np
import pandas as pd
import pytest

from sentinel.monitoring.drift import drift_status, drift_table, psi_categorical, psi_numeric


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20000), rng.normal(size=20000)
    assert psi_numeric(a, a) == pytest.approx(0, abs=1e-9)
    assert psi_numeric(a, b) < 0.01


def test_shifted_distribution_is_major():
    rng = np.random.default_rng(1)
    assert psi_numeric(rng.normal(size=20000), rng.normal(1.0, size=20000)) > 0.25


def test_missing_values_are_their_own_bin():
    ref = np.r_[np.arange(1000.0), [np.nan] * 10]
    cur = np.r_[np.arange(1000.0), [np.nan] * 500]
    assert psi_numeric(ref, cur) > 0.1


def test_categorical_and_new_levels():
    ref = ["a"] * 500 + ["b"] * 500
    assert psi_categorical(ref, ref) == pytest.approx(0, abs=1e-9)
    assert psi_categorical(ref, ["a"] * 500 + ["c"] * 500) > 0.25


def test_drift_table_by_period():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"x": rng.normal(size=5000), "c": rng.choice(["a", "b"], 5000)})
    cur = pd.DataFrame(
        {
            "x": np.r_[rng.normal(size=3000), rng.normal(2, size=3000)],
            "c": rng.choice(["a", "b"], 6000),
            "period": ["2020-07"] * 3000 + ["2020-08"] * 3000,
        }
    )
    t = drift_table(ref, cur, ["x"], ["c"])
    x = t[t.feature == "x"].set_index("period")["status"]
    assert x["2020-07"] == "stable" and x["2020-08"] == "major"
    assert drift_status(0.15) == "moderate"
