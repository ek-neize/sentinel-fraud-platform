# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Load reference data: watchlist
# MAGIC
# MAGIC The watchlist is reference data, not a stream. It is built on the laptop
# MAGIC (`scripts/build_reference_data.py`), uploaded to the `landing` volume
# MAGIC (`scripts/upload_reference.sh`), and loaded here into `ref_watchlist`.
# MAGIC
# MAGIC Silver joins it **point-in-time**: a transaction matches an entry only if
# MAGIC `added_at <= event_ts < removed_at`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SRC = f"/Volumes/{CATALOG}/{SCHEMA}/landing/watchlist.parquet"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

from pyspark.sql import functions as F

wl = (
    spark.read.parquet(SRC)
    .withColumn("added_at", F.col("added_at").cast("timestamp"))
    .withColumn("removed_at", F.col("removed_at").cast("timestamp"))
    .withColumn("evidence_known_at", F.col("evidence_known_at").cast("timestamp"))
    .withColumn("loaded_at", F.current_timestamp())
)

bad = wl.filter(F.col("added_at") < F.col("evidence_known_at")).count()
assert bad == 0, f"{bad} watchlist rows listed before their evidence was known"

(
    wl.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.ref_watchlist")
)
spark.sql(
    f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.ref_watchlist IS "
    "'Synthetic card hot-list, merchant-risk list and noisy external listings. Join point-in-time.'"
)

display(
    spark.sql(
        f"""
        SELECT entity_type, source, count(*) AS entries,
               min(added_at) AS first_added, max(added_at) AS last_added
        FROM {CATALOG}.{SCHEMA}.ref_watchlist
        GROUP BY entity_type, source ORDER BY entity_type, source
        """
    )
)
