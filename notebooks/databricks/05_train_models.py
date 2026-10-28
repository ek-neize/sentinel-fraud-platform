# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Train and compare models (MLflow)
# MAGIC
# MAGIC | Model | Why |
# MAGIC |---|---|
# MAGIC | Logistic regression | interpretable baseline: one coefficient per feature |
# MAGIC | XGBoost (3 depths) | captures interactions (e.g. high amount *and* a burst *at night*) |
# MAGIC | Both without watchlist features | ablation: how much does the synthetic watchlist add? |
# MAGIC
# MAGIC **Data discipline**
# MAGIC - Train on the `train` split with `label_at_cutoff` (only labels known by the training date).
# MAGIC - Early stopping, model choice and the capacity threshold use `valid` with `label_final`.
# MAGIC - The `test` split is not read here. It is used once, in Step 6.
# MAGIC
# MAGIC Every run is logged to MLflow (params, validation metrics, model, plots). The champion is the
# MAGIC **simplest model within 0.001 PR-AUC of the best** (ties go to shallower / faster models); it is
# MAGIC registered in Unity Catalog as `<catalog>.<schema>.fraud_model` with alias **`champion`**.
# MAGIC The best logistic regression gets alias **`baseline`**.
# MAGIC
# MAGIC **Protected attributes** (`gender`, `age_at_txn`) are not model inputs. They stay in gold only
# MAGIC for the fairness check in Step 6.

# COMMAND ----------

# MAGIC %pip install -q xgboost==3.* scikit-learn==1.* matplotlib
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
dbutils.widgets.text("neg_rate", "0.25")
dbutils.widgets.dropdown("register", "true", ["true", "false"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
NS = f"{CATALOG}.{SCHEMA}"
NEG_RATE = float(dbutils.widgets.get("neg_rate"))
REGISTER = dbutils.widgets.get("register") == "true"
MODEL_NAME = f"{NS}.fraud_model"
SEED = 42
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Import the sentinel package from this Git folder
import os
import sys

REPO_ROOT = os.path.abspath("../..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from sentinel.common.config import load_assumptions  # noqa: E402
from sentinel.models.feature_sets import CATEGORICAL, NUMERIC  # noqa: E402
from sentinel.models.training import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    EVAL_LABEL,
    TRAIN_LABEL,
    WEIGHT_COL,
    run_experiments,
    select_champion,
)

CFG = load_assumptions()
print("sentinel package loaded from", REPO_ROOT)

# COMMAND ----------

# DBTITLE 1,Label maturity check (validation needs chargebacks that arrived after July 2020)
from pyspark.sql import functions as F

display(
    spark.table(f"{NS}.gold_training_set")
    .where("split = 'valid'")
    .groupBy(F.date_trunc("month", "event_ts").alias("month"))
    .agg(F.count("*").alias("rows"), F.sum("label_final").alias("frauds"),
         F.round(F.avg("label_final"), 5).alias("fraud_rate"))
    .orderBy("month")
)
max_label = spark.table(f"{NS}.silver_labels").agg(F.max("label_known_ts")).first()[0]
print("Latest label known:", max_label)
if max_label is None or str(max_label) < "2020-09-01":
    print(
        "WARNING: labels for May-June transactions are still arriving (chargebacks take up to 60 "
        "days). Stream the rest of the data first: run_producer.py --start 2020-07-01 --speed max, "
        "then 01_bronze_ingest, 03_silver, 04_gold_features."
    )

# COMMAND ----------

# DBTITLE 1,Load train (negatives sampled in Spark) and validation
cols = ["transaction_id", "event_ts", "split", TRAIN_LABEL, EVAL_LABEL, *NUMERIC, *CATEGORICAL]
gold = spark.table(f"{NS}.gold_training_set").select(*cols)

train_sdf = (
    gold.where("split = 'train'")
    .withColumn("_u", F.rand(SEED))
    .where((F.col(TRAIN_LABEL) == 1) | (F.col("_u") < NEG_RATE))
    .withColumn(WEIGHT_COL, F.when(F.col(TRAIN_LABEL) == 1, 1.0).otherwise(1.0 / NEG_RATE))
    .drop("_u")
)
train = train_sdf.toPandas()
valid = gold.where("split = 'valid'").toPandas()
for df in (train, valid):  # Arrow may return tz-aware UTC timestamps; metrics expect naive UTC
    if getattr(df["event_ts"].dt, "tz", None) is not None:
        df["event_ts"] = df["event_ts"].dt.tz_convert("UTC").dt.tz_localize(None)

print(f"train sample: {len(train):,} rows, {int(train[TRAIN_LABEL].sum()):,} frauds (known at cutoff)")
print(f"valid:        {len(valid):,} rows, {int(valid[EVAL_LABEL].sum()):,} frauds")

# COMMAND ----------

# DBTITLE 1,MLflow experiment
import mlflow
import pandas as pd

from sentinel.models.tracking import feature_importance, log_experiment_run

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{user}/sentinel-fraud"
mlflow.set_experiment(EXPERIMENT)
run_ids, model_uris = {}, {}


def log_run(cfg, pipe, info, metrics, thresholds):
    run_ids[cfg.name], model_uris[cfg.name] = log_experiment_run(
        cfg, pipe, info, metrics, thresholds, valid, NEG_RATE, extra_tags={"platform": "databricks"}
    )
    print(f"logged {cfg.name}: PR-AUC {metrics['pr_auc']:.4f}")

# COMMAND ----------

# DBTITLE 1,Run all experiments
results, models = run_experiments(train, valid, DEFAULT_EXPERIMENTS, neg_rate=1.0, seed=SEED,
                                  on_result=log_run)
show = ["name", "pr_auc", "roc_auc", "topk100_recall", "thr100_recall", "thr100_precision",
        "thr100_fp_per_1000_txns", "thr25_recall", "thr25_precision", "thr25_amount_recall",
        "thr100_days_over_capacity", "fit_seconds"]
display(results[show].round(4))

# COMMAND ----------

# DBTITLE 1,Ablation: what does the watchlist add?
r = results.set_index("name")
best_xgb_full = r[r["model"].eq("xgb") & r["features"].eq("full")]["pr_auc"].idxmax()
ablation = pd.DataFrame(
    {
        "logistic regression": [r.loc["logreg_full", "pr_auc"], r.loc["logreg_no_watchlist", "pr_auc"]],
        "xgboost": [r.loc[best_xgb_full, "pr_auc"], r.loc["xgb_no_watchlist", "pr_auc"]],
    },
    index=["with watchlist", "without watchlist"],
).T
ablation["pr_auc_gain_from_watchlist"] = ablation["with watchlist"] - ablation["without watchlist"]
display(ablation.round(4).reset_index(names="model"))

# COMMAND ----------

# DBTITLE 1,Top features: logistic regression coefficients and XGBoost gain
lr_imp = feature_importance(models["logreg_full"]["pipeline"]).head(15)
xgb_imp = feature_importance(models[best_xgb_full]["pipeline"]).head(15)
# LR coefficients: effect of +1 standard deviation (numeric) or of a level vs. its reference level
display(lr_imp[["feature", "coefficient"]].round(3))
display(xgb_imp.round(1))

# COMMAND ----------

# DBTITLE 1,Register champion (simplest model within 0.001 of best PR-AUC) and baseline (best LR)
champion = select_champion(results, models, tol=0.001)
baseline = results[results["model"] == "logreg"].iloc[0]["name"]
print("champion:", champion, "| baseline:", baseline)

if REGISTER:
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from sentinel.models.promotion import promotion_decision

    client = MlflowClient()

    def current_metrics(alias):
        """Validation metrics of the model version that holds `alias` now (None if none)."""
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, alias)
        except MlflowException:
            return None
        return dict(client.get_run(mv.run_id).data.metrics)

    for alias, name in (("champion", champion), ("baseline", baseline)):
        mv = mlflow.register_model(model_uris[name], MODEL_NAME)
        thr = models[name]["thresholds"]
        client.set_model_version_tag(MODEL_NAME, mv.version, "experiment", name)
        client.set_model_version_tag(MODEL_NAME, mv.version, "threshold_100_per_day", str(thr[100]))
        client.set_model_version_tag(MODEL_NAME, mv.version, "threshold_25_per_day", str(thr[25]))
        if alias == "champion":
            # Champion/challenger gate (config: promotion): only move @champion if better.
            decision = promotion_decision(models[name]["metrics"], current_metrics("champion"),
                                          CFG["promotion"])
            client.set_model_version_tag(MODEL_NAME, mv.version, "promotion",
                                         ("promoted: " if decision.promote else "rejected: ")
                                         + decision.reason)
            if not decision.promote:
                client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
                print(f"{MODEL_NAME} v{mv.version} -> @challenger ({name}); champion kept: "
                      f"{decision.reason}")
                continue
        client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        print(f"{MODEL_NAME} v{mv.version} -> @{alias} ({name})")
else:
    print("register = false: skipped registration")
