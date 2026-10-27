import numpy as np
import pandas as pd
import pytest

from sentinel.features.training_set import build_training_frame
from sentinel.models.feature_sets import CATEGORICAL, NUMERIC, WATCHLIST, feature_set
from sentinel.models.training import (
    WEIGHT_COL,
    ExperimentConfig,
    run_experiments,
    sample_negatives,
)

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def frame(txns, labels, watchlist, cfg):
    return build_training_frame(txns, labels, watchlist, cfg["split"])


def test_training_frame_label_views(frame, cfg):
    cutoff = pd.Timestamp(cfg["split"]["gap_end"])
    assert (frame["label_at_cutoff"] <= frame["label_final"]).all()
    late = frame["label_known_ts"] > cutoff
    assert (frame.loc[late, "label_at_cutoff"] == 0).all()
    assert frame.loc[frame.split == "train", "label_at_cutoff"].sum() > 0


def test_model_features_exist_in_gold_notebook(spark):
    from tests.notebook_utils import load_cell

    gold = load_cell("04_gold_features.py", "Feature transforms")
    assert set(NUMERIC) <= set(gold["NUMERIC_FEATURES"])
    assert set(CATEGORICAL) <= set(gold["CATEGORICAL_FEATURES"])


def test_feature_sets():
    full_num, _ = feature_set("full")
    nowl_num, _ = feature_set("no_watchlist")
    assert set(WATCHLIST) <= set(full_num)
    assert not set(WATCHLIST) & set(nowl_num)
    with pytest.raises(ValueError):
        feature_set("nope")


def test_negative_sampling_reweights():
    df = pd.DataFrame({"y": [1] * 10 + [0] * 990})
    sample, w = sample_negatives(df, "y", rate=0.2, seed=0)
    assert sample["y"].sum() == 10
    assert w[sample["y"].to_numpy() == 1].tolist() == [1.0] * 10
    assert np.allclose(w[sample["y"].to_numpy() == 0], 5.0)
    # weighted negatives approximate the original count
    assert abs(w[sample["y"].to_numpy() == 0].sum() - 990) < 150


def test_presampled_weights_used_when_rate_is_one():
    df = pd.DataFrame({"y": [1, 0, 0], WEIGHT_COL: [1.0, 4.0, 4.0]})
    _, w = sample_negatives(df, "y", rate=1.0)
    assert w.tolist() == [1.0, 4.0, 4.0]


def test_models_train_and_beat_chance(frame):
    train, valid = frame[frame.split == "train"], frame[frame.split == "valid"]
    configs = [
        ExperimentConfig("lr", "logreg", "full", {"C": 0.1}),
        ExperimentConfig("xgb", "xgb", "no_watchlist", {"n_estimators": 60, "max_depth": 3}),
    ]
    table, models = run_experiments(train, valid, configs, neg_rate=0.3)
    base = valid["label_final"].mean()
    assert set(table["name"]) == {"lr", "xgb"}
    assert (table["pr_auc"] > 10 * base).all()
    for m in models.values():
        scores = m["pipeline"].predict_proba(valid)[:, 1]
        assert ((scores >= 0) & (scores <= 1)).all()
        assert set(m["thresholds"]) == {100, 25}


def test_mlflow_logging_roundtrip(frame, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # lightweight local store for tests
    from sentinel.models.tracking import log_experiment_run

    train, valid = frame[frame.split == "train"], frame[frame.split == "valid"]
    cfg = ExperimentConfig("lr_logged", "logreg", "full", {"C": 0.1})
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test")

    uris = {}

    def on_result(cfg, pipe, info, metrics, thresholds):
        uris["run"], uris["model"] = log_experiment_run(
            cfg, pipe, info, metrics, thresholds, valid, 0.3
        )

    run_experiments(train, valid, [cfg], neg_rate=0.3, on_result=on_result)
    loaded = mlflow.sklearn.load_model(uris["model"])
    assert loaded.predict_proba(valid.head(5))[:, 1].shape == (5,)
    run = mlflow.get_run(uris["run"])
    assert "pr_auc" in run.data.metrics
