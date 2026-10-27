"""Train and compare models on the chronological split.

- Train on the `train` split with `label_at_cutoff` (what was known when training).
- Negatives are down-sampled for speed; kept negatives get weight 1/rate so predicted
  probabilities stay on the original scale.
- Early stopping, model choice and the capacity threshold use the `valid` split with
  `label_final`. The `test` split is never touched here (Step 6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sentinel.evaluation.metrics import (
    daily_topk_report,
    fixed_threshold_report,
    ranking_metrics,
    threshold_for_daily_capacity,
)
from sentinel.models.feature_sets import feature_set
from sentinel.models.pipelines import make_logreg, make_xgb, make_xgb_preprocessor

TRAIN_LABEL = "label_at_cutoff"
EVAL_LABEL = "label_final"


@dataclass
class ExperimentConfig:
    name: str
    model: str  # "logreg" | "xgb"
    features: str = "full"  # "full" | "no_watchlist"
    params: dict = field(default_factory=dict)


DEFAULT_EXPERIMENTS = [
    ExperimentConfig("logreg_full", "logreg", "full", {"C": 0.1}),
    ExperimentConfig("logreg_no_watchlist", "logreg", "no_watchlist", {"C": 0.1}),
    ExperimentConfig("xgb_full_d4", "xgb", "full", {"max_depth": 4, "learning_rate": 0.1}),
    ExperimentConfig("xgb_full_d6", "xgb", "full", {"max_depth": 6, "learning_rate": 0.05}),
    ExperimentConfig("xgb_full_d8", "xgb", "full", {"max_depth": 8, "learning_rate": 0.05}),
    ExperimentConfig(
        "xgb_no_watchlist", "xgb", "no_watchlist", {"max_depth": 6, "learning_rate": 0.05}
    ),
]


WEIGHT_COL = "sample_weight"


def sample_negatives(df: pd.DataFrame, label: str, rate: float, seed: int = 42):
    """Keep all positives and `rate` of negatives; return (sample, sample_weight).

    If the frame was already sampled upstream (e.g. in Spark) and carries a
    `sample_weight` column, pass rate=1.0 and that column is used as-is.
    """
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    if rate == 1.0:
        sample = df.reset_index(drop=True)
        weight = sample[WEIGHT_COL].to_numpy() if WEIGHT_COL in sample else np.ones(len(sample))
        return sample, weight
    rng = np.random.default_rng(seed)
    keep = (df[label] == 1).to_numpy() | (rng.random(len(df)) < rate)
    sample = df.loc[keep].reset_index(drop=True)
    weight = np.where(sample[label] == 1, 1.0, 1.0 / rate)
    return sample, weight


def fit(
    cfg: ExperimentConfig,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    neg_rate: float = 0.25,
    seed: int = 42,
) -> tuple[Pipeline, dict]:
    numeric, categorical = feature_set(cfg.features)
    cols = numeric + categorical
    sample, weight = sample_negatives(train, TRAIN_LABEL, neg_rate, seed)
    y = sample[TRAIN_LABEL].to_numpy()
    t0 = time.time()
    info: dict = {"train_rows": len(sample), "train_positives": int(y.sum())}

    if cfg.model == "logreg":
        pipe = make_logreg(numeric, categorical, seed=seed, **cfg.params)
        pipe.fit(sample[cols], y, model__sample_weight=weight)
    elif cfg.model == "xgb":
        pre = make_xgb_preprocessor(numeric, categorical)
        xtr = pre.fit_transform(sample[cols])
        xva = pre.transform(valid[cols])
        clf = make_xgb(cfg.params, seed=seed)
        clf.fit(
            xtr,
            y,
            sample_weight=weight,
            eval_set=[(xva, valid[EVAL_LABEL].to_numpy())],
            verbose=False,
        )
        info["best_iteration"] = int(clf.best_iteration)
        pipe = Pipeline([("pre", pre), ("model", clf)])
    else:
        raise ValueError(f"Unknown model '{cfg.model}'")

    info["fit_seconds"] = round(time.time() - t0, 1)
    info["n_features_after_preprocessing"] = len(pipe[:-1].get_feature_names_out())
    return pipe, info


def score(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(df)[:, 1]


def evaluate(
    df: pd.DataFrame,
    scores: np.ndarray,
    capacities=(100, 25),
    label: str = EVAL_LABEL,
    thresholds: dict | None = None,
) -> dict:
    """Flat metric dict. If `thresholds` is None they are fitted on `df` (validation)."""
    d = df.assign(_score=scores, day=df["event_ts"].dt.floor("D"))
    out = {f"{k}": v for k, v in ranking_metrics(d[label], d["_score"]).items()}
    fitted = {}
    for k in capacities:
        top = daily_topk_report(d, "_score", label, k)
        thr = (thresholds or {}).get(k)
        if thr is None:
            thr = threshold_for_daily_capacity(d["_score"], d["day"], k)
        fitted[k] = thr
        fix = fixed_threshold_report(d, "_score", label, thr, k)
        for prefix, rep in (("topk", top), ("thr", fix)):
            for m in ("recall", "precision", "fp_per_1000_txns", "amount_recall", "alerts_per_day"):
                out[f"{prefix}{k}_{m}"] = rep[m]
        out[f"thr{k}_threshold"] = thr
        out[f"thr{k}_days_over_capacity"] = fix["days_over_capacity"]
    out["_thresholds"] = fitted
    return out


def run_experiments(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    configs=None,
    neg_rate: float = 0.25,
    seed: int = 42,
    on_result=None,
):
    """Fit every config, evaluate on validation; return (results table, fitted pipelines)."""
    rows, models = [], {}
    for cfg in configs or DEFAULT_EXPERIMENTS:
        pipe, info = fit(cfg, train, valid, neg_rate=neg_rate, seed=seed)
        metrics = evaluate(valid, score(pipe, valid))
        thresholds = metrics.pop("_thresholds")
        row = {"name": cfg.name, "model": cfg.model, "features": cfg.features, **info, **metrics}
        rows.append(row)
        models[cfg.name] = {
            "pipeline": pipe,
            "config": cfg,
            "thresholds": thresholds,
            "info": info,
            "metrics": metrics,
        }
        if on_result:
            on_result(cfg, pipe, info, metrics, thresholds)
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return table, models
