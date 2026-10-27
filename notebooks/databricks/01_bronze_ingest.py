# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingest: Kafka → Delta
# MAGIC
# MAGIC Streams both Confluent topics into append-only bronze tables.
# MAGIC
# MAGIC | Topic | Bronze table |
# MAGIC |---|---|
# MAGIC | `sentinel.transactions.v1` | `bronze_transactions` |
# MAGIC | `sentinel.fraud_labels.v1` | `bronze_labels` |
# MAGIC
# MAGIC **Bronze rules**
# MAGIC - Keep everything: the raw message (`raw_value`) and Kafka metadata are stored as received,
# MAGIC   so any later layer can be rebuilt from bronze.
# MAGIC - Parse against the v1 contract and *flag* problems in `dq_errors`; nothing is dropped here.
# MAGIC   Silver decides what to keep.
# MAGIC - `trigger(availableNow=True)`: process everything new, then stop. Works on serverless and
# MAGIC   runs as a scheduled Job; the checkpoint makes each run pick up where the last ended.
# MAGIC
# MAGIC Topic retention is 7 days: run this at least weekly or older offsets expire.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
dbutils.widgets.dropdown("starting_offsets", "earliest", ["earliest", "latest"])
dbutils.widgets.text("max_offsets_per_trigger", "")
dbutils.widgets.dropdown("reset", "false", ["false", "true"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
STARTING_OFFSETS = dbutils.widgets.get("starting_offsets")
MAX_OFFSETS = dbutils.widgets.get("max_offsets_per_trigger").strip()
RESET = dbutils.widgets.get("reset") == "true"

CHECKPOINT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Transforms
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

TRANSACTIONS_TOPIC = "sentinel.transactions.v1"
LABELS_TOPIC = "sentinel.fraud_labels.v1"
SCHEMA_VERSION = "1.0"

# Timestamps are parsed as strings first so a malformed value is flagged, not silently null.
TX_SCHEMA = StructType(
    [
        StructField("schema_version", StringType()),
        StructField("transaction_id", StringType()),
        StructField("event_time", StringType()),
        StructField("card_id", StringType()),
        StructField("merchant_id", StringType()),
        StructField("merchant_name", StringType()),
        StructField("category", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("gender", StringType()),
        StructField("age_at_txn", DoubleType()),
        StructField("job", StringType()),
        StructField("city", StringType()),
        StructField("state", StringType()),
        StructField("zip", StringType()),
        StructField("cust_lat", DoubleType()),
        StructField("cust_long", DoubleType()),
        StructField("city_pop", LongType()),
        StructField("merch_lat", DoubleType()),
        StructField("merch_long", DoubleType()),
    ]
)

LABEL_SCHEMA = StructType(
    [
        StructField("schema_version", StringType()),
        StructField("transaction_id", StringType()),
        StructField("event_time", StringType()),
        StructField("label_known_at", StringType()),
        StructField("label", StringType()),
        StructField("label_source", StringType()),
    ]
)


def kafka_envelope(raw: DataFrame) -> DataFrame:
    """Kafka source columns -> bronze envelope (raw message + metadata)."""
    return raw.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.col("value").cast("string").alias("raw_value"),
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.map_from_entries(
            F.transform("headers", lambda h: F.struct(h["key"], h["value"].cast("string")))
        ).alias("kafka_headers"),
        F.current_timestamp().alias("ingested_at"),
    )


def _dq(checks: list[tuple[str, "F.Column"]]) -> "F.Column":
    """Array of failed check names (empty array = valid)."""
    flagged = F.array(*[F.when(cond, F.lit(name)) for name, cond in checks])
    return F.filter(flagged, lambda x: x.isNotNull())


def parse_transactions(env: DataFrame) -> DataFrame:
    p = F.from_json("raw_value", TX_SCHEMA)
    df = env.withColumn("payload", p).withColumn(
        "event_ts", F.to_timestamp("payload.event_time")
    )
    checks = [
        ("unparseable_json", F.col("payload").isNull()),
        ("bad_schema_version", F.col("payload.schema_version") != SCHEMA_VERSION),
        ("missing_transaction_id", F.col("payload.transaction_id").isNull()),
        ("missing_card_id", F.col("payload.card_id").isNull()),
        ("missing_merchant_id", F.col("payload.merchant_id").isNull()),
        ("bad_event_time", F.col("event_ts").isNull()),
        ("bad_amount", F.col("payload.amount").isNull() | (F.col("payload.amount") <= 0)),
        ("key_mismatch", F.col("kafka_key") != F.col("payload.card_id")),
        # The label must never travel with the transaction.
        ("label_leak", F.col("raw_value").contains('"is_fraud"')),
    ]
    return (
        df.withColumn("dq_errors", _dq(checks))
        .withColumn("is_valid", F.size("dq_errors") == 0)
        .withColumn("is_test", F.coalesce(F.col("payload.transaction_id").startswith("TEST-"), F.lit(False)))
    )


def parse_labels(env: DataFrame) -> DataFrame:
    p = F.from_json("raw_value", LABEL_SCHEMA)
    df = (
        env.withColumn("payload", p)
        .withColumn("event_ts", F.to_timestamp("payload.event_time"))
        .withColumn("label_known_ts", F.to_timestamp("payload.label_known_at"))
    )
    checks = [
        ("unparseable_json", F.col("payload").isNull()),
        ("bad_schema_version", F.col("payload.schema_version") != SCHEMA_VERSION),
        ("missing_transaction_id", F.col("payload.transaction_id").isNull()),
        ("bad_event_time", F.col("event_ts").isNull()),
        ("bad_label_known_at", F.col("label_known_ts").isNull()),
        ("label_before_event", F.col("label_known_ts") < F.col("event_ts")),
        ("bad_label", ~F.coalesce(F.col("payload.label") == "fraud", F.lit(False))),
    ]
    return (
        df.withColumn("dq_errors", _dq(checks))
        .withColumn("is_valid", F.size("dq_errors") == 0)
        .withColumn("is_test", F.coalesce(F.col("payload.transaction_id").startswith("TEST-"), F.lit(False)))
    )

# COMMAND ----------

# DBTITLE 1,Kafka connection (credentials from the secret scope)
def kafka_options() -> dict:
    key = dbutils.secrets.get("sentinel", "kafka-consumer-key")
    secret = dbutils.secrets.get("sentinel", "kafka-consumer-secret")
    opts = {
        "kafka.bootstrap.servers": dbutils.secrets.get("sentinel", "kafka-bootstrap"),
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.sasl.jaas.config": (
            "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required "
            f'username="{key}" password="{secret}";'
        ),
        "groupIdPrefix": "sentinel-bronze",  # matches the consumer-group ACL
        "startingOffsets": STARTING_OFFSETS,
        "includeHeaders": "true",
        "failOnDataLoss": "false",  # tolerate offsets expired by retention; logged, not fatal
    }
    if MAX_OFFSETS:
        opts["maxOffsetsPerTrigger"] = MAX_OFFSETS
    return opts

# COMMAND ----------

# DBTITLE 1,Optional reset (re-ingest from scratch)
if RESET:
    for name in ("bronze_transactions", "bronze_labels"):
        spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{name}")
        dbutils.fs.rm(f"{CHECKPOINT_ROOT}/{name}", True)
    print("Reset done: bronze tables dropped and checkpoints cleared.")

# COMMAND ----------

# DBTITLE 1,Run both streams (availableNow)
def ingest(topic: str, table: str, parse) -> dict:
    raw = spark.readStream.format("kafka").options(**kafka_options()).option("subscribe", topic).load()
    query = (
        parse(kafka_envelope(raw))
        .writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_ROOT}/{table}")
        .trigger(availableNow=True)
        .toTable(f"{CATALOG}.{SCHEMA}.{table}")
    )
    query.awaitTermination()
    progress = query.lastProgress or {}
    return {"table": table, "rows_this_run": progress.get("numInputRows", 0)}


results = [
    ingest(TRANSACTIONS_TOPIC, "bronze_transactions", parse_transactions),
    ingest(LABELS_TOPIC, "bronze_labels", parse_labels),
]
print(results)

# COMMAND ----------

# DBTITLE 1,Bronze health check
for table in ("bronze_transactions", "bronze_labels"):
    print(f"--- {table}")
    display(
        spark.sql(
            f"""
            SELECT count(*)                                    AS rows,
                   count_if(is_valid AND NOT is_test)          AS valid_rows,
                   count_if(NOT is_valid)                      AS invalid_rows,
                   count_if(is_test)                           AS test_rows,
                   count(DISTINCT payload.transaction_id)      AS distinct_txn_ids,
                   min(event_ts)                               AS min_event_ts,
                   max(event_ts)                               AS max_event_ts,
                   max(ingested_at)                            AS last_ingested_at
            FROM {CATALOG}.{SCHEMA}.{table}
            """
        )
    )
    display(
        spark.sql(
            f"""
            SELECT err, count(*) AS rows
            FROM {CATALOG}.{SCHEMA}.{table} LATERAL VIEW explode(dq_errors) e AS err
            GROUP BY err ORDER BY rows DESC
            """
        )
    )
