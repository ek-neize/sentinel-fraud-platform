# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Gold: point-in-time features and as-of labels
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |---|---|
# MAGIC | `gold_features` | one row per transaction: model features computed **only from that card's earlier transactions** |
# MAGIC | `gold_training_set` | `gold_features` + split + two label views (below) |
# MAGIC
# MAGIC **No future data.** Every history feature uses windows that end strictly before the
# MAGIC current transaction (`rowsBetween(unboundedPreceding, -1)` / `rangeBetween(-N, -1)`).
# MAGIC CI checks this two ways: Spark output equals an independent pandas implementation, and
# MAGIC changing future transactions never changes earlier rows' features.
# MAGIC
# MAGIC **Two label views.**
# MAGIC - `label_at_cutoff`: fraud labels **known by the training cutoff** (end of the gap month).
# MAGIC   This is what a model trained on that date could actually learn from. Late chargebacks are
# MAGIC   missing on purpose — that is real label noise.
# MAGIC - `label_final`: every label known now. Used only for evaluation.
# MAGIC
# MAGIC The full table is rebuilt each run (≈1.3–1.9M rows, a few minutes on serverless).
# MAGIC See `docs/features.md` for the feature dictionary.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "sentinel")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
NS = f"{CATALOG}.{SCHEMA}"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# DBTITLE 1,Split dates from config/assumptions.yaml (Git folder) with safe defaults
import os

SPLIT = {
    "train_start": "2019-01-01",
    "train_end": "2020-03-01",
    "gap_end": "2020-04-01",
    "valid_end": "2020-07-01",
    "test_end": "2021-01-01",
}
for candidate in ("../../config/assumptions.yaml", "../config/assumptions.yaml"):
    path = os.path.abspath(candidate)
    if os.path.exists(path):
        import yaml  # preinstalled on Databricks runtimes

        with open(path) as fh:
            SPLIT = yaml.safe_load(fh)["split"]
        print(f"Split dates from {path}")
        break
else:
    print("assumptions.yaml not found next to this notebook; using defaults")
LABEL_CUTOFF = SPLIT["gap_end"]  # model is trained once the gap month has passed
print(SPLIT, "| label cutoff:", LABEL_CUTOFF)

# COMMAND ----------

# DBTITLE 1,Feature transforms
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

VELOCITY_WINDOWS = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}

NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "card_prior_txn_count",
    "card_prior_amt_mean",
    "card_prior_amt_std",
    "card_prior_amt_max",
    "amt_to_prior_mean",
    "amt_to_prior_max",
    "amt_zscore",
    "secs_since_prev_txn",
    *[f"cnt_{k}" for k in VELOCITY_WINDOWS],
    *[f"amt_{k}" for k in VELOCITY_WINDOWS],
    "hour_of_day",
    "day_of_week",
    "is_night",
    "home_merch_km",
    "age_at_txn",
    "log_city_pop",
    "wl_card_hit",
    "wl_merchant_hit",
    "wl_any_hit",
]
CATEGORICAL_FEATURES = ["category", "state", "gender"]
ID_COLUMNS = ["transaction_id", "event_ts", "card_id", "merchant_id"]


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dphi = F.radians(lat2 - lat1)
    dlmb = F.radians(lon2 - lon1)
    a = F.sin(dphi / 2) ** 2 + F.cos(p1) * F.cos(p2) * F.sin(dlmb / 2) ** 2
    return 2 * r * F.asin(F.sqrt(a))


def build_features(tx: DataFrame) -> DataFrame:
    """silver_transactions -> one feature row per transaction (strictly-prior history)."""
    df = tx.withColumn("_ts", F.unix_timestamp("event_ts"))
    by_card = Window.partitionBy("card_id").orderBy("_ts", "transaction_id")
    prior = by_card.rowsBetween(Window.unboundedPreceding, -1)

    df = (
        df.withColumn("card_prior_txn_count", F.count(F.lit(1)).over(prior))
        .withColumn("card_prior_amt_mean", F.avg("amount").over(prior))
        .withColumn("card_prior_amt_std", F.stddev_samp("amount").over(prior))
        .withColumn("card_prior_amt_max", F.max("amount").over(prior))
        .withColumn("secs_since_prev_txn", F.col("_ts") - F.lag("_ts").over(by_card))
    )
    for name, secs in VELOCITY_WINDOWS.items():
        w = Window.partitionBy("card_id").orderBy("_ts").rangeBetween(-secs, -1)
        df = df.withColumn(f"cnt_{name}", F.count(F.lit(1)).over(w)).withColumn(
            f"amt_{name}", F.coalesce(F.sum("amount").over(w), F.lit(0.0))
        )

    hour = F.hour("event_ts")
    return df.select(
        *ID_COLUMNS,
        "amount",
        F.log1p("amount").alias("log_amount"),
        "card_prior_txn_count",
        "card_prior_amt_mean",
        "card_prior_amt_std",
        "card_prior_amt_max",
        (F.col("amount") / F.col("card_prior_amt_mean")).alias("amt_to_prior_mean"),
        (F.col("amount") / F.col("card_prior_amt_max")).alias("amt_to_prior_max"),
        F.when(
            F.col("card_prior_amt_std") > 0,
            (F.col("amount") - F.col("card_prior_amt_mean")) / F.col("card_prior_amt_std"),
        ).alias("amt_zscore"),
        F.col("secs_since_prev_txn").cast("double").alias("secs_since_prev_txn"),
        *[F.col(f"cnt_{k}") for k in VELOCITY_WINDOWS],
        *[F.col(f"amt_{k}") for k in VELOCITY_WINDOWS],
        hour.alias("hour_of_day"),
        F.dayofweek("event_ts").alias("day_of_week"),  # 1 = Sunday ... 7 = Saturday
        ((hour >= 22) | (hour < 4)).cast("int").alias("is_night"),
        _haversine_km(
            F.col("cust_lat"), F.col("cust_long"), F.col("merch_lat"), F.col("merch_long")
        ).alias("home_merch_km"),
        "age_at_txn",
        F.log1p("city_pop").alias("log_city_pop"),
        "wl_card_hit",
        "wl_merchant_hit",
        "wl_any_hit",
        *CATEGORICAL_FEATURES,
    )


def assign_split(df: DataFrame, split: dict) -> DataFrame:
    ts = F.col("event_ts")
    b = {k: F.to_timestamp(F.lit(v)) for k, v in split.items()}
    return df.withColumn(
        "split",
        F.when((ts >= b["train_start"]) & (ts < b["train_end"]), "train")
        .when((ts >= b["train_end"]) & (ts < b["gap_end"]), "gap")
        .when((ts >= b["gap_end"]) & (ts < b["valid_end"]), "valid")
        .when((ts >= b["valid_end"]) & (ts < b["test_end"]), "test")
        .otherwise("out_of_range"),
    )


def attach_labels(features: DataFrame, labels: DataFrame, cutoff: str) -> DataFrame:
    lab = labels.select("transaction_id", "label_known_ts", "label_source")
    cut = F.to_timestamp(F.lit(cutoff))
    return (
        features.join(lab, "transaction_id", "left")
        .withColumn("label_final", F.col("label_known_ts").isNotNull().cast("int"))
        .withColumn(
            "label_at_cutoff",
            (F.col("label_known_ts").isNotNull() & (F.col("label_known_ts") <= cut)).cast("int"),
        )
    )

# COMMAND ----------

# DBTITLE 1,Build gold tables
silver_tx = spark.read.table(f"{NS}.silver_transactions")
silver_labels = spark.read.table(f"{NS}.silver_labels")

features = build_features(silver_tx)
(
    features.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{NS}.gold_features")
)

training = attach_labels(assign_split(spark.read.table(f"{NS}.gold_features"), SPLIT), silver_labels, LABEL_CUTOFF)
(
    training.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{NS}.gold_training_set")
)
spark.sql(
    f"COMMENT ON TABLE {NS}.gold_training_set IS "
    f"'Point-in-time features + split + label_at_cutoff (known by {LABEL_CUTOFF}) + label_final'"
)
print("Gold tables written.")

# COMMAND ----------

# DBTITLE 1,Labels by split (cutoff vs final)
display(
    spark.sql(
        f"""
        SELECT split,
               count(*)                       AS rows,
               sum(label_at_cutoff)           AS frauds_known_at_cutoff,
               sum(label_final)               AS frauds_known_now,
               round(avg(label_final), 5)     AS fraud_rate_now
        FROM {NS}.gold_training_set
        GROUP BY split ORDER BY min(event_ts)
        """
    )
)

# COMMAND ----------

# DBTITLE 1,Sanity check: does velocity separate fraud? (train split)
display(
    spark.sql(
        f"""
        SELECT CASE WHEN cnt_24h = 0 THEN '0'
                    WHEN cnt_24h <= 2 THEN '1-2'
                    WHEN cnt_24h <= 5 THEN '3-5'
                    ELSE '6+' END                AS prior_txns_24h,
               count(*)                          AS rows,
               round(avg(label_final), 4)        AS fraud_rate
        FROM {NS}.gold_training_set
        WHERE split = 'train'
        GROUP BY 1 ORDER BY 1
        """
    )
)

# COMMAND ----------

# DBTITLE 1,Null rates (expected only for first-ever card transactions)
null_cols = ["card_prior_amt_mean", "card_prior_amt_std", "amt_zscore", "secs_since_prev_txn"]
display(
    spark.read.table(f"{NS}.gold_features").select(
        F.count(F.lit(1)).alias("rows"),
        *[F.round(F.avg(F.col(c).isNull().cast("int")), 5).alias(f"null_{c}") for c in null_cols],
    )
)
