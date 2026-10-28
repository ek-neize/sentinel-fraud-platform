# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Business evaluation on the held-out test period
# MAGIC
# MAGIC Scores the **untouched** test months (Jul–Dec 2020) with `fraud_model@champion` (XGBoost)
# MAGIC and `fraud_model@baseline` (logistic regression), then answers the business question:
# MAGIC *how much fraud can a team of a given size catch, at what cost, and how sure are we?*
# MAGIC
# MAGIC | Output | Table / artifact |
# MAGIC |---|---|
# MAGIC | Capacity curve: recall, precision, $ caught, net benefit for 5–100 alerts/day | `gold_eval_capacity_curve` |
# MAGIC | Strategy comparison at the chosen budget (no model, watchlist rule, LR, XGBoost) | `gold_eval_strategies` |
# MAGIC | 95% day-level bootstrap intervals | `gold_eval_intervals` |
# MAGIC | Fairness by gender and age band (not model inputs) | `gold_eval_fairness` |
# MAGIC | Scores for validation + test (feeds dashboard and alerts) | `gold_scores` |
# MAGIC | Full Markdown report + capacity plot | MLflow run `test_evaluation` |
# MAGIC
# MAGIC Thresholds come from validation only. Costs come from `config/assumptions.yaml`.

# COMMAND ----------

# MAGIC %pip install -q xgboost==3.* scikit-learn==1.* matplotlib tabulate
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
dbutils.widgets.text("bootstrap_iterations", "1000")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
NS = f"{CATALOG}.{SCHEMA}"
MODEL_NAME = f"{NS}.fraud_model"
N_BOOT = int(dbutils.widgets.get("bootstrap_iterations"))
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Import the sentinel package from this Git folder
import os
import sys

REPO_ROOT = os.path.abspath("../..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from sentinel.common.config import load_assumptions  # noqa: E402
from sentinel.evaluation.report import capacity_figure, run_business_evaluation  # noqa: E402
from sentinel.models.feature_sets import CATEGORICAL, NUMERIC, PROTECTED  # noqa: E402

CFG = load_assumptions()
print("assumptions loaded; capacities:", CFG["evaluation"]["capacities"])

# COMMAND ----------

# DBTITLE 1,Label maturity check (December chargebacks arrive until late February 2021)
from pyspark.sql import functions as F

latest = spark.table(f"{NS}.silver_labels").agg(F.max("label_known_ts")).first()[0]
test_frauds = spark.table(f"{NS}.gold_training_set").where("split = 'test'").agg(F.sum("label_final")).first()[0]
print(f"latest label known: {latest} | test frauds labelled: {test_frauds}")
if latest is None or str(latest) < "2021-02-01":
    print("WARNING: late chargebacks for the test period are missing; recall will look too high. "
          "Run: python scripts/run_producer.py --start 2020-07-01 --speed max, then "
          "01_bronze_ingest, 03_silver, 04_gold_features.")

# COMMAND ----------

# DBTITLE 1,Load champion and baseline from Unity Catalog
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
models, versions = {}, {}
for alias in ("champion", "baseline"):
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    versions[alias] = mv.version
    models[alias] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    print(f"@{alias}: v{mv.version} ({mv.tags.get('experiment', '?')})")

# COMMAND ----------

# DBTITLE 1,Load validation and test, score both models
cols = ["transaction_id", "event_ts", "split", "label_final", *NUMERIC, *CATEGORICAL, *PROTECTED]
gold = spark.table(f"{NS}.gold_training_set").where("split in ('valid', 'test')").select(*cols)
df = gold.toPandas()
if getattr(df["event_ts"].dt, "tz", None) is not None:
    df["event_ts"] = df["event_ts"].dt.tz_convert("UTC").dt.tz_localize(None)
ints = df[NUMERIC].select_dtypes("integer").columns
df[ints] = df[ints].astype("float64")

valid = df[df["split"] == "valid"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
scores = {
    name: {"valid": m.predict_proba(valid[NUMERIC + CATEGORICAL])[:, 1],
           "test": m.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]}
    for name, m in models.items()
}
print(f"valid {len(valid):,} rows | test {len(test):,} rows, {int(test['label_final'].sum()):,} frauds")

# COMMAND ----------

# DBTITLE 1,Run the business evaluation
result = run_business_evaluation(valid, test, scores, CFG, bootstrap_n=N_BOOT)
k = result.operating_k
print(f"Operating budget (max net benefit): {k} alerts/day")
display(result.ranking.round(4))

# COMMAND ----------

# DBTITLE 1,Strategies at the operating budget
show = ["strategy", "alerts_per_day", "recall", "precision", "amount_recall", "fp_per_1000_txns",
        "cost_review", "cost_fraud_loss", "cost_total", "net_benefit"]
display(result.strategies[show].round(4))

# COMMAND ----------

# DBTITLE 1,Capacity curve and 95% intervals
fig = capacity_figure(result)
display(fig)
display(result.curve[result.curve["policy"] == "threshold_capped"]
        [["model", "capacity_k", "recall", "precision", "amount_recall", "fp_per_1000_txns",
          "net_benefit"]].round(4))
display(result.intervals.round(4))
print("busy days:", result.overflow)

# COMMAND ----------

# DBTITLE 1,Fairness check by gender and age band
display(result.fairness.round(4))

# COMMAND ----------

# DBTITLE 1,Persist tables for the dashboard and alerting
import pandas as pd


def save(pdf: pd.DataFrame, name: str) -> None:
    (spark.createDataFrame(pdf).write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").saveAsTable(f"{NS}.{name}"))


save(result.curve, "gold_eval_capacity_curve")
save(result.strategies, "gold_eval_strategies")
save(result.intervals, "gold_eval_intervals")
if len(result.fairness):
    save(result.fairness.assign(group=result.fairness["group"].astype(str)), "gold_eval_fairness")

thr = result.thresholds["champion"][k]
score_rows = []
for split_name, part in (("valid", valid), ("test", test)):
    score_rows.append(pd.DataFrame({
        "transaction_id": part["transaction_id"],
        "event_ts": part["event_ts"],
        "split": split_name,
        "score_champion": scores["champion"][split_name],
        "score_baseline": scores["baseline"][split_name],
        "above_threshold": scores["champion"][split_name] >= thr,
        "label_final": part["label_final"],
    }))
save(pd.concat(score_rows, ignore_index=True), "gold_scores")
print("Saved gold_eval_* tables and gold_scores")

# COMMAND ----------

# DBTITLE 1,Log the report to MLflow and tag the champion with its operating point
import tempfile

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/sentinel-fraud")
h = result.headline
with mlflow.start_run(run_name="test_evaluation"):
    mlflow.set_tags({"stage": "test", "champion_version": versions["champion"],
                     "baseline_version": versions["baseline"]})
    mlflow.log_params({"operating_k": k, "threshold": thr, "bootstrap_iterations": N_BOOT})
    mlflow.log_metrics({key: float(v) for key, v in h.items() if isinstance(v, (int, float))})
    with tempfile.TemporaryDirectory() as tmp:
        with open(f"{tmp}/results.md", "w") as fh:
            fh.write(result.markdown)
        fig.savefig(f"{tmp}/capacity_curve.png", dpi=130)
        for name, t in (("capacity_curve", result.curve), ("strategies", result.strategies),
                        ("intervals", result.intervals), ("fairness", result.fairness)):
            t.to_csv(f"{tmp}/{name}.csv", index=False)
        mlflow.log_artifacts(tmp, artifact_path="report")

client.set_model_version_tag(MODEL_NAME, versions["champion"], "operating_k", str(k))
client.set_model_version_tag(MODEL_NAME, versions["champion"], "operating_threshold", str(thr))
print(f"champion v{versions['champion']} tagged: operating_k={k}, threshold={thr:.6f}")

# COMMAND ----------

# DBTITLE 1,Report (copy this into docs/results.md)
print(result.markdown)
