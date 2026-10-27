"""MLflow logging shared by the Databricks notebook and the local training script."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd

from sentinel.models.feature_sets import CATEGORICAL, NUMERIC
from sentinel.models.training import EVAL_LABEL, TRAIN_LABEL

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src/sentinel


def feature_importance(pipe) -> pd.DataFrame:
    """LR: signed coefficients on standardised inputs. XGBoost: total gain."""
    names = pipe[:-1].get_feature_names_out()
    model = pipe[-1]
    if hasattr(model, "coef_"):
        vals = model.coef_[0]
        return pd.DataFrame(
            {"feature": names, "coefficient": vals, "abs_coefficient": np.abs(vals)}
        ).sort_values("abs_coefficient", ascending=False)
    gain = model.get_booster().get_score(importance_type="gain")
    vals = [gain.get(f"f{i}", gain.get(n, 0.0)) for i, n in enumerate(names)]
    return pd.DataFrame({"feature": names, "gain": vals}).sort_values("gain", ascending=False)


def _pr_curve_png(y, scores, title: str, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    p, r, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def log_experiment_run(
    cfg,
    pipe,
    info: dict,
    metrics: dict,
    thresholds: dict,
    valid: pd.DataFrame,
    neg_rate: float,
    extra_tags: dict | None = None,
):
    """Log one fitted experiment to the active MLflow experiment. Returns (run_id, model_uri)."""
    import mlflow
    from mlflow.models import infer_signature

    with mlflow.start_run(run_name=cfg.name) as run:
        mlflow.log_params(
            {
                "model": cfg.model,
                "features": cfg.features,
                "neg_rate": neg_rate,
                "train_label": TRAIN_LABEL,
                "eval_label": EVAL_LABEL,
                **{f"p_{k}": v for k, v in cfg.params.items()},
            }
        )
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in {**info, **metrics}.items()
                if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)
            }
        )
        mlflow.set_tags({"stage": "validation", "project": "sentinel", **(extra_tags or {})})

        with tempfile.TemporaryDirectory() as tmp:
            feature_importance(pipe).to_csv(f"{tmp}/feature_importance.csv", index=False)
            with open(f"{tmp}/capacity_thresholds.json", "w") as fh:
                json.dump({str(k): v for k, v in thresholds.items()}, fh, indent=2)
            scores = pipe.predict_proba(valid)[:, 1]
            _pr_curve_png(
                valid[EVAL_LABEL],
                scores,
                f"{cfg.name}: PR-AUC {metrics['pr_auc']:.3f}",
                f"{tmp}/pr_curve_valid.png",
            )
            mlflow.log_artifacts(tmp)

        # Integer columns are declared as doubles so a missing value at serving time
        # does not break MLflow schema enforcement.
        x_example = valid[NUMERIC + CATEGORICAL].head(200).copy()
        ints = x_example.select_dtypes("integer").columns
        x_example[ints] = x_example[ints].astype("float64")
        kwargs = dict(
            signature=infer_signature(x_example, pipe.predict_proba(x_example)[:, 1]),
            input_example=x_example.head(3),
            code_paths=[PACKAGE_DIR],  # model loads anywhere without installing sentinel
            serialization_format="cloudpickle",  # pipeline has custom transform functions
        )
        try:  # MLflow 3
            model_info = mlflow.sklearn.log_model(pipe, name="model", **kwargs)
        except TypeError:  # MLflow 2
            model_info = mlflow.sklearn.log_model(pipe, artifact_path="model", **kwargs)
        return run.info.run_id, model_info.model_uri
