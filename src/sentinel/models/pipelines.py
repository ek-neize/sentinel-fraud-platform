"""Model pipelines: interpretable logistic regression baseline and XGBoost.

Both are scikit-learn Pipelines (preprocess -> model) so one object goes from raw gold
columns to a fraud score, logs cleanly to MLflow, and serves from FastAPI later.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sentinel.models.feature_sets import LOG1P

# Hour and weekday act non-linearly (night-time fraud), so the linear model one-hots them.
LR_CYCLIC_AS_CATEGORICAL = ["hour_of_day", "day_of_week"]


def _clip_log1p(x):
    return np.log1p(np.clip(x, 0, None))


def _clip_z(x):
    return np.clip(x, -20, 20)


def make_logreg(numeric: list[str], categorical: list[str], C: float = 0.1, seed: int = 42):
    """Standardised, log-transformed numerics + one-hot categoricals -> L2 logistic regression."""
    num = [c for c in numeric if c not in LR_CYCLIC_AS_CATEGORICAL and c != "amount"]
    log_cols = [c for c in num if c in LOG1P]
    z_cols = [c for c in num if c == "amt_zscore"]
    plain = [c for c in num if c not in log_cols and c not in z_cols]
    cats = list(categorical) + [c for c in LR_CYCLIC_AS_CATEGORICAL if c in numeric]

    def numeric_branch(fn):
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if fn is not None:
            steps.append(("transform", FunctionTransformer(fn, feature_names_out="one-to-one")))
        steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    pre = ColumnTransformer(
        [
            ("log", numeric_branch(_clip_log1p), log_cols),
            ("z", numeric_branch(_clip_z), z_cols),
            ("num", numeric_branch(None), plain),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
        ],
        verbose_feature_names_out=True,
    )
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=seed)
    return Pipeline([("pre", pre), ("model", clf)])


def make_xgb_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Trees need no scaling; NaN is handled natively. One-hot the categoricals."""
    return ColumnTransformer(
        [
            ("num", "passthrough", list(numeric)),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), list(categorical)),
        ],
        verbose_feature_names_out=True,
    )


DEFAULT_XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
}


def make_xgb(params: dict | None = None, seed: int = 42):
    from xgboost import XGBClassifier

    p = {**DEFAULT_XGB_PARAMS, **(params or {})}
    return XGBClassifier(**p, random_state=seed, n_jobs=-1)
