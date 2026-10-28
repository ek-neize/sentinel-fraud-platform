# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Score, alert and send the daily digest
# MAGIC
# MAGIC Runs operations day by day, like a scheduled job would:
# MAGIC
# MAGIC 1. Score that day's transactions with `fraud_model@champion`.
# MAGIC 2. Alerts = score ≥ the **operating threshold** tagged on the model by `06_evaluate`.
# MAGIC 3. **Backlog queue:** new alerts join yesterday's open ones; investigators review the
# MAGIC    `operating_k` highest-priority alerts; alerts open for `max_age_days` expire.
# MAGIC 4. Each new alert gets its **top 3 reasons** (TreeSHAP contributions, in plain words).
# MAGIC 5. Writes `alert_queue`, `alert_daily_summary` and an HTML **digest** per day to
# MAGIC    `/Volumes/<catalog>/<schema>/alerts/`. Email is sent from your laptop
# MAGIC    (`scripts/send_digest.py`) because Free Edition blocks outbound SMTP.
# MAGIC
# MAGIC In this simulation, "reviewed" alerts take the eventual label as the investigator's
# MAGIC finding (`confirmed_fraud` / `cleared`). Days already processed are skipped, so the
# MAGIC notebook can be re-run safely; use `reset = true` to start over.

# COMMAND ----------

# MAGIC %pip install -q xgboost==3.* scikit-learn==1.* tabulate
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
dbutils.widgets.text("start_day", "2020-12-01")
dbutils.widgets.text("days", "7")
dbutils.widgets.dropdown("reset", "false", ["false", "true"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
NS = f"{CATALOG}.{SCHEMA}"
MODEL_NAME = f"{NS}.fraud_model"
START = dbutils.widgets.get("start_day")
N_DAYS = int(dbutils.widgets.get("days"))
RESET = dbutils.widgets.get("reset") == "true"
ALERT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/alerts"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Import the sentinel package from this Git folder
import os
import sys

REPO_ROOT = os.path.abspath("../..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from sentinel.alerts.digest import digest_text, render_digest  # noqa: E402
from sentinel.alerts.pipeline import ScoringModel, empty_queue, process_day  # noqa: E402
from sentinel.common.config import load_assumptions  # noqa: E402
from sentinel.models.feature_sets import CATEGORICAL, NUMERIC  # noqa: E402

CFG = load_assumptions()
ALERTING = CFG["alerting"]

# COMMAND ----------

# DBTITLE 1,Load champion and its operating point (set by 06_evaluate)
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
if "operating_threshold" not in mv.tags:
    raise RuntimeError("Champion has no operating_threshold tag. Run 06_evaluate first.")
MODEL = ScoringModel(
    pipeline=mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion"),
    threshold=float(mv.tags["operating_threshold"]),
    version=mv.version,
    k=int(mv.tags["operating_k"]),
    max_age_days=int(ALERTING["max_age_days"]),
    priority_mode=ALERTING["priority"],
)
print(f"champion v{MODEL.version}: threshold {MODEL.threshold:.6f}, {MODEL.k} reviews/day, "
      f"{MODEL.max_age_days}-day expiry, priority = {MODEL.priority_mode}")

# COMMAND ----------

# DBTITLE 1,State: alert queue and processed days
import pandas as pd

spark.sql(f"CREATE VOLUME IF NOT EXISTS {NS}.alerts COMMENT 'Daily alert digests (HTML)'")
if RESET:
    for t in ("alert_queue", "alert_daily_summary"):
        spark.sql(f"DROP TABLE IF EXISTS {NS}.{t}")
    print("Reset: alert tables dropped.")


def table_exists(name: str) -> bool:
    return spark.catalog.tableExists(f"{NS}.{name}")


queue = spark.table(f"{NS}.alert_queue").toPandas() if table_exists("alert_queue") else empty_queue()
summaries = (spark.table(f"{NS}.alert_daily_summary").toPandas()
             if table_exists("alert_daily_summary") else pd.DataFrame())
done_days = set(pd.to_datetime(summaries["day"]).dt.normalize()) if len(summaries) else set()
days = [d for d in pd.date_range(START, periods=N_DAYS, freq="D") if d not in done_days]
if done_days and days and days[0] > max(done_days) + pd.Timedelta(days=1):
    print("Note: there is a gap after the last processed day; the backlog ages across it.")
print(f"{len(queue):,} alerts in queue so far; processing {len(days)} new day(s): "
      f"{[d.strftime('%Y-%m-%d') for d in days]}")

# COMMAND ----------

# DBTITLE 1,Run operations day by day
from pyspark.sql import functions as F

labels = spark.table(f"{NS}.silver_labels").select("transaction_id").withColumn("y", F.lit(1)) \
    .toPandas().set_index("transaction_id")["y"]
cols = ["transaction_id", "event_ts", "card_id", "merchant_id", *NUMERIC, *CATEGORICAL]
features = spark.table(f"{NS}.gold_features").select(*cols)
names = spark.table(f"{NS}.silver_transactions").select("transaction_id", "merchant_name")

new_summaries, last_digest = [], None
for d in days:
    txns = (features.where(F.to_date("event_ts") == F.lit(d.strftime("%Y-%m-%d")))
            .join(names, "transaction_id", "left").toPandas())
    if getattr(txns["event_ts"].dt, "tz", None) is not None:
        txns["event_ts"] = txns["event_ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    ints = txns[NUMERIC].select_dtypes("integer").columns
    txns[ints] = txns[ints].astype("float64")

    queue, summary, digest_rows = process_day(d, txns, queue, MODEL, labels)
    new_summaries.append(summary)

    html = render_digest(d, digest_rows, summary,
                         {"name": "fraud_model", "version": MODEL.version, "threshold": MODEL.threshold},
                         top_n=int(ALERTING["digest_top_n"]))
    with open(f"{ALERT_DIR}/digest_{d:%Y-%m-%d}.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    last_digest = (d, html, digest_rows, summary)
    print(f"{d:%Y-%m-%d}: {summary['txns_scored']:,} txns | {summary['new_alerts']} new alerts | "
          f"{summary['reviewed']} reviewed ({summary['confirmed_fraud']} fraud) | "
          f"{summary['open_end']} waiting | {summary['expired']} expired")

# COMMAND ----------

# DBTITLE 1,Persist queue and daily summary
if days:
    q = queue.copy()
    q["reason_features"] = q["reason_features"].apply(
        lambda v: [str(x) for x in v] if isinstance(v, (list, tuple)) else [])
    for c in ("disposition", "reason_1", "reason_2", "reason_3", "merchant_name"):
        q[c] = q[c].astype(object).where(q[c].notna(), None)
    (spark.createDataFrame(q).write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").saveAsTable(f"{NS}.alert_queue"))
    all_summaries = pd.concat([summaries, pd.DataFrame(new_summaries)], ignore_index=True)
    (spark.createDataFrame(all_summaries).write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").saveAsTable(f"{NS}.alert_daily_summary"))
    print("alert_queue and alert_daily_summary updated")
display(spark.table(f"{NS}.alert_daily_summary").orderBy("day"))

# COMMAND ----------

# DBTITLE 1,Today's digest (what the email contains)
if last_digest:
    d, html, rows, summary = last_digest
    displayHTML(html)
    print(digest_text(d, rows, summary))
    print(f"\nSaved to {ALERT_DIR}/digest_{d:%Y-%m-%d}.html — send it with: "
          "python scripts/send_digest.py --from-databricks")
