# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Dashboard tables and snapshot
# MAGIC
# MAGIC Builds the dashboard's small, pre-aggregated tables from notebooks 04–07 and writes them
# MAGIC twice:
# MAGIC
# MAGIC | Output | Used by |
# MAGIC |---|---|
# MAGIC | `dash_*` Delta tables + `dash_meta` | Databricks App (`SENTINEL_DATA_MODE=live`) |
# MAGIC | Parquet files + `meta.json` in `/Volumes/<catalog>/<schema>/dashboard_snapshot/` | public Streamlit app: `python scripts/pull_snapshot.py`, then commit `dashboard/snapshot/` |
# MAGIC
# MAGIC Run after `07_score_and_alert` and `09_monitoring`. Everything is aggregated or case-level demo data on
# MAGIC synthetic transactions; gender and age appear only as group-level fairness rates.

# COMMAND ----------

# MAGIC %pip install -q xgboost==3.* scikit-learn==1.*
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
NS = f"{CATALOG}.{SCHEMA}"
MODEL_NAME = f"{NS}.fraud_model"
SNAPSHOT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/dashboard_snapshot"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Import the sentinel package from this Git folder
import os
import sys

REPO_ROOT = os.path.abspath("../..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from sentinel.common.config import load_assumptions  # noqa: E402
from sentinel.dashboard_export import (  # noqa: E402
    DRIFT_FEATURES,
    HISTORY_DAYS,
    build_snapshot,
    write_snapshot,
)

CFG = load_assumptions()

# COMMAND ----------

# DBTITLE 1,Champion model info
import json

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
pr_auc = None
try:
    pr_auc = float(mlflow.get_run(mv.run_id).data.metrics.get("pr_auc"))
except Exception:  # noqa: BLE001
    pass
MODEL_INFO = {
    "name": "fraud_model",
    "version": mv.version,
    "experiment": mv.tags.get("experiment"),
    "threshold": float(mv.tags["operating_threshold"]),
    "operating_k": int(mv.tags["operating_k"]),
    "pr_auc_validation": pr_auc,
}
print(MODEL_INFO)

# COMMAND ----------

# DBTITLE 1,Load inputs (only the columns the dashboard needs)
def to_pd(sdf):
    pdf = sdf.toPandas()
    pdf.attrs = {}  # serverless attaches PlanMetrics here; they break Parquet/JSON writes
    for c in pdf.columns:
        if str(pdf[c].dtype).startswith("datetime64") and getattr(pdf[c].dt, "tz", None):
            pdf[c] = pdf[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return pdf


queue = to_pd(spark.table(f"{NS}.alert_queue").drop("reason_features"))
daily = to_pd(spark.table(f"{NS}.alert_daily_summary"))
scores = spark.table(f"{NS}.gold_scores").select(
    "transaction_id", "split", F.col("score_champion").alias("score"), "above_threshold"
)
gold = spark.table(f"{NS}.gold_training_set").where("split in ('valid', 'test')").select(
    "transaction_id", "event_ts", "split", "card_id", "category", "state", "amount",
    "label_final", *[c for c in DRIFT_FEATURES if c != "amount"],
)
names = spark.table(f"{NS}.silver_transactions").select("transaction_id", "merchant_name")
scored = to_pd(gold.join(scores.drop("split"), "transaction_id")
               .join(names, "transaction_id", "left"))
print(f"queue {len(queue):,} | daily {len(daily):,} | scored rows {len(scored):,}")

# Card history only for cards that have alerts, within the window before their alerts.
start = queue["event_ts"].min() - pd.Timedelta(days=HISTORY_DAYS)
history = scored[scored["card_id"].isin(queue["card_id"].unique())
                 & (scored["event_ts"] >= start)]

eval_tables = {
    "curve": to_pd(spark.table(f"{NS}.gold_eval_capacity_curve")),
    "strategies": to_pd(spark.table(f"{NS}.gold_eval_strategies")),
    "intervals": to_pd(spark.table(f"{NS}.gold_eval_intervals")),
    "fairness": to_pd(spark.table(f"{NS}.gold_eval_fairness"))
    if spark.catalog.tableExists(f"{NS}.gold_eval_fairness") else pd.DataFrame(),
}

# COMMAND ----------

# DBTITLE 1,Build tables
# Monitoring tables from notebook 09, if it has run.
monitoring = None
if spark.catalog.tableExists(f"{NS}.monitoring_checks"):
    monitoring = (to_pd(spark.table(f"{NS}.monitoring_checks")),
                  to_pd(spark.table(f"{NS}.monitoring_label_maturity")))
else:
    print("monitoring_checks not found: run 09_monitoring first to include the Monitoring page")

tables, meta = build_snapshot(
    queue=queue, daily_summary=daily, scored=scored, features=scored, history_txns=history,
    eval_tables=eval_tables, model_info=MODEL_INFO, cfg=CFG, monitoring=monitoring,
)
for name, t in tables.items():
    print(f"{name:16s} {len(t):>7,} rows")
print(json.dumps(meta["headline"].get("backlog_score", {}), indent=2))

# COMMAND ----------

# DBTITLE 1,Write dash_* Delta tables (live mode) and the Parquet snapshot (public app)
for name, t in tables.items():
    (spark.createDataFrame(t).write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").saveAsTable(f"{NS}.dash_{name}"))
(spark.createDataFrame(pd.DataFrame({"json": [json.dumps(meta, default=str)]}))
 .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{NS}.dash_meta"))

spark.sql(f"CREATE VOLUME IF NOT EXISTS {NS}.dashboard_snapshot COMMENT 'Dashboard Parquet snapshot'")
write_snapshot(tables, meta, SNAPSHOT_DIR)
size_kb = sum(os.path.getsize(os.path.join(SNAPSHOT_DIR, f)) for f in os.listdir(SNAPSHOT_DIR)) / 1024
print(f"dash_* tables written; snapshot {size_kb:,.0f} KB in {SNAPSHOT_DIR}")
print("Next, on your laptop: python scripts/pull_snapshot.py && git add dashboard/snapshot && "
      "git commit -m 'Update dashboard snapshot' && git push")
