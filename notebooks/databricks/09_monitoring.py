# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · Monitoring
# MAGIC
# MAGIC Daily checks on the scored transactions and the alert queue (thresholds in
# MAGIC `config/assumptions.yaml` → `monitoring`). Labels arrive late, so the first warnings
# MAGIC use what is known immediately: input and score drift, alert volume vs the review budget,
# MAGIC backlog and expiry, missing inputs. Precision comes from investigator dispositions;
# MAGIC recall is checked only on the latest cohort whose labels are complete (60+ days old).
# MAGIC
# MAGIC | Output | Content |
# MAGIC |---|---|
# MAGIC | `monitoring_checks` | every check, as of each week of the test period (backfill) |
# MAGIC | `monitoring_label_maturity` | share of eventual fraud already labelled, by transaction age |
# MAGIC | `monitoring_log` | one row per run: overall status and counts (appended) |
# MAGIC
# MAGIC With `fail_on_critical = true` (set by the scheduled job) a critical check fails this
# MAGIC task, so the job's failure e-mail is the alert. Run after `07_score_and_alert`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
dbutils.widgets.dropdown("fail_on_critical", "false", ["true", "false"])
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FAIL_ON_CRITICAL = dbutils.widgets.get("fail_on_critical") == "true"
NS = f"{CATALOG}.{SCHEMA}"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Import the sentinel package from this Git folder
import os
import sys

REPO_ROOT = os.path.abspath("../..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from sentinel.common.config import load_assumptions  # noqa: E402
from sentinel.dashboard_export import DRIFT_FEATURES  # noqa: E402
from sentinel.monitoring.checks import monitoring_history, overall_status  # noqa: E402

CFG = load_assumptions()

# COMMAND ----------

# DBTITLE 1,Load scored transactions (validation = reference, test = monitored) and daily operations
import pandas as pd
from pyspark.sql import functions as F


def to_pd(sdf):
    pdf = sdf.toPandas()
    pdf.attrs = {}  # serverless attaches PlanMetrics here
    for c in pdf.columns:
        if str(pdf[c].dtype).startswith("datetime64") and getattr(pdf[c].dt, "tz", None):
            pdf[c] = pdf[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return pdf


scores = spark.table(f"{NS}.gold_scores").select(
    "transaction_id", F.col("score_champion").alias("score"), "above_threshold"
)
cols = [
    "transaction_id",
    "event_ts",
    "split",
    "category",
    "label_final",
    "label_known_ts",
    *[c for c in DRIFT_FEATURES],
]
scored = to_pd(
    spark.table(f"{NS}.gold_training_set")
    .where("split in ('valid', 'test')")
    .select(*cols)
    .join(scores, "transaction_id")
)
ops = to_pd(spark.table(f"{NS}.alert_daily_summary"))
reference = scored[scored["split"] == "valid"]
monitored = scored[scored["split"] == "test"]
print(f"reference {len(reference):,} | monitored {len(monitored):,} | ops days {len(ops)}")

# COMMAND ----------

# DBTITLE 1,Run the checks as of every week (backfill) and the label-maturity view
checks, maturity = monitoring_history(monitored, reference, ops, CFG, DRIFT_FEATURES)
latest = checks[checks["as_of"] == checks["as_of"].max()]
status = overall_status(latest)
display(latest[["check", "subject", "value", "warn", "critical", "status", "detail"]])
display(maturity)
print(f"Overall status as of {latest['as_of'].iloc[0]:%Y-%m-%d}: {status.upper()}")

# COMMAND ----------

# DBTITLE 1,Write monitoring tables
for name, df in (("monitoring_checks", checks), ("monitoring_label_maturity", maturity)):
    (
        spark.createDataFrame(df)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{NS}.{name}")
    )

log = pd.DataFrame(
    [
        {
            "run_ts": pd.Timestamp.now(tz="UTC").tz_localize(None),
            "as_of": latest["as_of"].iloc[0],
            "status": status,
            "n_warn": int((latest["status"] == "warn").sum()),
            "n_critical": int((latest["status"] == "critical").sum()),
            "failing": "; ".join(
                f"{r.check}: {r.subject}" for r in latest.itertuples() if r.status != "ok"
            ),
        }
    ]
)
spark.createDataFrame(log).write.format("delta").mode("append").saveAsTable(f"{NS}.monitoring_log")
display(spark.table(f"{NS}.monitoring_log").orderBy(F.desc("run_ts")).limit(10))

# COMMAND ----------

# DBTITLE 1,Alert: fail the task on a critical check (the job's failure e-mail is the alert)
if status == "critical" and FAIL_ON_CRITICAL:
    failing = latest[latest["status"] == "critical"]
    raise RuntimeError(
        "Critical monitoring checks: "
        + "; ".join(f"{r.check} / {r.subject}: {r.detail}" for r in failing.itertuples())
    )
print("No action needed" if status == "ok" else f"Status {status}: see the table above")
