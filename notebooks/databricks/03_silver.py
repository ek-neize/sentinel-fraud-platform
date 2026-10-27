# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Silver: clean, de-duplicate, enrich
# MAGIC
# MAGIC | Source | Silver table | What happens |
# MAGIC |---|---|---|
# MAGIC | `bronze_transactions` | `silver_transactions` | valid, non-test rows; de-duplicated on `transaction_id`; flattened; **point-in-time watchlist join** |
# MAGIC | `bronze_labels` | `silver_labels` | valid rows; one label per transaction (earliest `label_known_ts`) |
# MAGIC | both | `silver_quarantine` | rows that failed bronze data-quality checks, with their `dq_errors` |
# MAGIC
# MAGIC **Incremental and replay-safe.** Bronze is read as a stream (`availableNow`) and each
# MAGIC micro-batch is `MERGE`d on `transaction_id`, so re-running this notebook or replaying Kafka
# MAGIC never creates duplicates.
# MAGIC
# MAGIC **Watchlist join.** A transaction matches an entry only if
# MAGIC `added_at <= event_ts < removed_at`. If `ref_watchlist` is reloaded with different
# MAGIC entries, run with `reset = true` to recompute the flags.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
dbutils.widgets.dropdown("reset", "false", ["false", "true"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
RESET = dbutils.widgets.get("reset") == "true"
NS = f"{CATALOG}.{SCHEMA}"
CHECKPOINT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Silver transforms
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

TX_COLUMNS = [
    "transaction_id",
    "event_ts",
    "card_id",
    "merchant_id",
    "merchant_name",
    "category",
    "amount",
    "gender",
    "age_at_txn",
    "job",
    "city",
    "state",
    "zip",
    "cust_lat",
    "cust_long",
    "city_pop",
    "merch_lat",
    "merch_long",
]


def clean_transactions(bronze: DataFrame) -> DataFrame:
    """Valid, non-test bronze rows -> flat typed columns plus lineage."""
    payload_cols = [c for c in TX_COLUMNS if c not in ("event_ts",)]
    return bronze.filter(F.col("is_valid") & ~F.col("is_test")).select(
        *[F.col(f"payload.{c}").alias(c) for c in payload_cols],
        F.col("event_ts"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.col("ingested_at").alias("bronze_ingested_at"),
    ).select(*TX_COLUMNS, "kafka_partition", "kafka_offset", "bronze_ingested_at")


def keep_first(df: DataFrame, key: str, *order_cols: str) -> DataFrame:
    """One row per key: the first by `order_cols` (e.g. earliest Kafka offset)."""
    w = Window.partitionBy(key).orderBy(*[F.col(c).asc() for c in order_cols])
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def watchlist_flags(tx: DataFrame, watchlist: DataFrame) -> DataFrame:
    """Point-in-time join: add wl_card_hit, wl_merchant_hit, wl_any_hit, wl_ids."""
    active = lambda: (F.col("added_at") <= F.col("event_ts")) & (  # noqa: E731
        F.col("removed_at").isNull() | (F.col("event_ts") < F.col("removed_at"))
    )
    hits = None
    for entity_type, column in (("card", "card_id"), ("merchant", "merchant_id")):
        wl = watchlist.filter(F.col("entity_type") == entity_type).select(
            F.col("entity_value"), "watchlist_id", "added_at", "removed_at"
        )
        h = (
            tx.select("transaction_id", "event_ts", F.col(column).alias("entity_value"))
            .join(wl, "entity_value")
            .filter(active())
            .select("transaction_id", "watchlist_id", F.lit(entity_type).alias("hit_type"))
        )
        hits = h if hits is None else hits.unionByName(h)

    agg = hits.groupBy("transaction_id").agg(
        F.max((F.col("hit_type") == "card").cast("int")).alias("wl_card_hit"),
        F.max((F.col("hit_type") == "merchant").cast("int")).alias("wl_merchant_hit"),
        F.concat_ws(",", F.array_sort(F.collect_set("watchlist_id"))).alias("wl_ids"),
    )
    out = tx.join(agg, "transaction_id", "left")
    return (
        out.withColumn("wl_card_hit", F.coalesce("wl_card_hit", F.lit(0)))
        .withColumn("wl_merchant_hit", F.coalesce("wl_merchant_hit", F.lit(0)))
        .withColumn("wl_any_hit", F.greatest("wl_card_hit", "wl_merchant_hit"))
    )


def clean_labels(bronze: DataFrame) -> DataFrame:
    labels = bronze.filter(F.col("is_valid") & ~F.col("is_test")).select(
        F.col("payload.transaction_id").alias("transaction_id"),
        F.col("event_ts"),
        F.col("label_known_ts"),
        F.col("payload.label").alias("label"),
        F.col("payload.label_source").alias("label_source"),
        F.col("ingested_at").alias("bronze_ingested_at"),
    )
    return keep_first(labels, "transaction_id", "label_known_ts")


def quarantine(bronze: DataFrame, source_table: str) -> DataFrame:
    return bronze.filter(~F.col("is_valid")).select(
        F.lit(source_table).alias("source_table"),
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_key",
        "raw_value",
        "dq_errors",
        "ingested_at",
    )

# COMMAND ----------

# DBTITLE 1,Optional reset
SILVER_TABLES = ("silver_transactions", "silver_labels", "silver_quarantine")
CHECKPOINTS = (
    "silver_transactions",
    "silver_labels",
    "silver_quarantine_tx",
    "silver_quarantine_labels",
)
if RESET:
    for t in SILVER_TABLES:
        spark.sql(f"DROP TABLE IF EXISTS {NS}.{t}")
    for c in CHECKPOINTS:
        dbutils.fs.rm(f"{CHECKPOINT_ROOT}/{c}", True)
    print("Reset done: silver tables dropped and checkpoints cleared.")

# COMMAND ----------

# DBTITLE 1,Merge helpers (create on first batch, MERGE afterwards)
def _table_exists(session, name: str) -> bool:
    return session.catalog.tableExists(name)


def upsert_transactions(batch: DataFrame, batch_id: int) -> None:
    session = batch.sparkSession
    target = f"{NS}.silver_transactions"
    wl = session.read.table(f"{NS}.ref_watchlist")
    rows = watchlist_flags(
        keep_first(clean_transactions(batch), "transaction_id", "kafka_partition", "kafka_offset"),
        wl,
    ).withColumn("silver_updated_at", F.current_timestamp())
    if not _table_exists(session, target):
        rows.write.format("delta").saveAsTable(target)
        return
    rows.createOrReplaceTempView("_silver_tx_batch")
    session.sql(
        f"""
        MERGE INTO {target} t
        USING _silver_tx_batch s
        ON t.transaction_id = s.transaction_id
        WHEN NOT MATCHED THEN INSERT *
        """
    )


def upsert_labels(batch: DataFrame, batch_id: int) -> None:
    session = batch.sparkSession
    target = f"{NS}.silver_labels"
    rows = clean_labels(batch).withColumn("silver_updated_at", F.current_timestamp())
    if not _table_exists(session, target):
        rows.write.format("delta").saveAsTable(target)
        return
    rows.createOrReplaceTempView("_silver_label_batch")
    session.sql(
        f"""
        MERGE INTO {target} t
        USING _silver_label_batch s
        ON t.transaction_id = s.transaction_id
        WHEN MATCHED AND s.label_known_ts < t.label_known_ts THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )

# COMMAND ----------

# DBTITLE 1,Run (availableNow)
def run_foreach(source: str, checkpoint: str, fn) -> None:
    (
        spark.readStream.table(f"{NS}.{source}")
        .writeStream.foreachBatch(fn)
        .option("checkpointLocation", f"{CHECKPOINT_ROOT}/{checkpoint}")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )


def run_quarantine(source: str, checkpoint: str) -> None:
    (
        quarantine(spark.readStream.table(f"{NS}.{source}"), source)
        .writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_ROOT}/{checkpoint}")
        .trigger(availableNow=True)
        .toTable(f"{NS}.silver_quarantine")
        .awaitTermination()
    )


run_foreach("bronze_transactions", "silver_transactions", upsert_transactions)
run_foreach("bronze_labels", "silver_labels", upsert_labels)
run_quarantine("bronze_transactions", "silver_quarantine_tx")
run_quarantine("bronze_labels", "silver_quarantine_labels")
print("Silver run complete.")

# COMMAND ----------

# DBTITLE 1,Silver health check
display(
    spark.sql(
        f"""
        SELECT
          (SELECT count(*) FROM {NS}.bronze_transactions)                       AS bronze_rows,
          (SELECT count(*) FROM {NS}.silver_transactions)                       AS silver_rows,
          (SELECT count(DISTINCT transaction_id) FROM {NS}.silver_transactions) AS distinct_ids,
          (SELECT count(*) FROM {NS}.silver_labels)                             AS silver_labels,
          (SELECT count(*) FROM {NS}.silver_quarantine)                         AS quarantined
        """
    )
)

# COMMAND ----------

# DBTITLE 1,Watchlist signal in silver (labels known so far)
display(
    spark.sql(
        f"""
        WITH j AS (
          SELECT t.*, CASE WHEN l.transaction_id IS NOT NULL THEN 1 ELSE 0 END AS is_fraud_known
          FROM {NS}.silver_transactions t
          LEFT JOIN {NS}.silver_labels l USING (transaction_id)
        ), base AS (SELECT avg(is_fraud_known) AS base_rate, sum(is_fraud_known) AS frauds FROM j)
        SELECT flag,
               round(avg(hit), 4)                                         AS hit_rate,
               round(sum(hit * is_fraud_known) / sum(hit), 4)             AS fraud_rate_if_hit,
               round(sum(hit * is_fraud_known) / sum(hit) / max(base_rate), 2) AS lift,
               round(sum(hit * is_fraud_known) / max(frauds), 4)          AS fraud_captured
        FROM j CROSS JOIN base
        LATERAL VIEW stack(3, 'wl_card_hit', wl_card_hit,
                              'wl_merchant_hit', wl_merchant_hit,
                              'wl_any_hit', wl_any_hit) s AS flag, hit
        GROUP BY flag ORDER BY flag
        """
    )
)
