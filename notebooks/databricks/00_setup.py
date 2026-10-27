# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the Unity Catalog schema and volumes Sentinel uses, and checks the Kafka
# MAGIC secrets are in place. Safe to re-run.
# MAGIC
# MAGIC | Object | Purpose |
# MAGIC |---|---|
# MAGIC | `<catalog>.sentinel` | schema for all Sentinel tables |
# MAGIC | volume `checkpoints` | Structured Streaming checkpoints |
# MAGIC | volume `landing` | reference files uploaded from the laptop (watchlist) |

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA} COMMENT 'Sentinel fraud platform'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints COMMENT 'Streaming checkpoints'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.landing COMMENT 'Uploaded reference files'")
display(spark.sql(f"SHOW VOLUMES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# DBTITLE 1,Check secrets (values are never printed)
required = {"kafka-bootstrap", "kafka-consumer-key", "kafka-consumer-secret"}
present = {s.key for s in dbutils.secrets.list("sentinel")}
missing = required - present
assert not missing, f"Missing secrets in scope 'sentinel': {sorted(missing)}"
print("Secrets OK:", sorted(required))
