"""Silver and gold notebook transforms, run on local Spark against mock data.

Key guarantees checked here:
- the Spark watchlist join equals the pandas point-in-time join;
- Spark features equal an independent pandas implementation;
- changing *future* transactions never changes *earlier* rows' features (no leakage);
- label_at_cutoff only counts labels known by the cutoff.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sentinel.data.watchlist import point_in_time_join
from sentinel.features import reference
from sentinel.streaming.producer import replay
from sentinel.streaming.sinks import MemorySink
from tests.notebook_utils import load_cell

pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def silver_nb():
    return load_cell("03_silver.py", "Silver transforms")


@pytest.fixture(scope="module")
def gold_nb():
    return load_cell("04_gold_features.py", "Feature transforms")


@pytest.fixture(scope="module")
def silver_pdf(txns, watchlist):
    """Pandas stand-in for silver_transactions (canonical txns + watchlist flags)."""
    df = point_in_time_join(txns.drop(columns=["is_fraud", "split"]), watchlist)
    df = df.rename(columns={"event_time": "event_ts"})
    for c in ("category", "state", "gender"):
        df[c] = df[c].astype(str)
    return df


def _to_spark(spark, pdf):
    return spark.createDataFrame(pdf)


def _wl_spark(spark, watchlist):
    return spark.createDataFrame(watchlist)


# ---------------------------------------------------------------- silver


def test_watchlist_join_matches_pandas(spark, silver_nb, silver_pdf, watchlist):
    tx = _to_spark(
        spark, silver_pdf.drop(columns=["wl_card_hit", "wl_merchant_hit", "wl_any_hit", "wl_ids"])
    )
    out = silver_nb["watchlist_flags"](tx, _wl_spark(spark, watchlist)).toPandas()
    merged = out.merge(silver_pdf, on="transaction_id", suffixes=("_spark", "_pd"))
    assert len(merged) == len(silver_pdf)
    for flag in ("wl_card_hit", "wl_merchant_hit", "wl_any_hit"):
        assert (merged[f"{flag}_spark"] == merged[f"{flag}_pd"]).all(), flag
    assert merged["wl_any_hit_spark"].sum() > 0


def test_bronze_to_silver_cleans_and_dedupes(spark, silver_nb, txns, labels):
    bronze_nb = load_cell("01_bronze_ingest.py", "Transforms")
    sink = MemorySink()
    replay(txns.drop(columns=["is_fraud", "split"]), labels, sink, speed=None, max_events=2000)
    msgs = [m for m in sink.messages if m["topic"] == bronze_nb["TRANSACTIONS_TOPIC"]]
    test_msg = dict(json.loads(msgs[0]["value"]), transaction_id="TEST-0001")
    replayed = msgs + msgs[:50]  # a Kafka replay duplicates 50 events
    rows = [
        (m["key"].encode(), m["value"], m["topic"], 0, i, datetime(2026, 10, 27), 0, [])
        for i, m in enumerate(replayed)
    ] + [
        (
            msgs[0]["key"].encode(),
            json.dumps(test_msg).encode(),
            msgs[0]["topic"],
            0,
            99999,
            datetime(2026, 10, 27),
            0,
            [],
        )
    ]
    schema = (
        "key binary, value binary, topic string, partition int, offset long, "
        "timestamp timestamp, timestampType int, headers array<struct<key:string,value:binary>>"
    )
    bronze = bronze_nb["parse_transactions"](
        bronze_nb["kafka_envelope"](spark.createDataFrame(rows, schema))
    )
    silver = silver_nb["keep_first"](
        silver_nb["clean_transactions"](bronze), "transaction_id", "kafka_partition", "kafka_offset"
    )
    out = silver.toPandas()
    assert len(out) == len(msgs)  # duplicates and the TEST row removed
    assert out["transaction_id"].is_unique
    assert not out["transaction_id"].str.startswith("TEST-").any()
    first_offsets = out.set_index("transaction_id")["kafka_offset"]
    assert (first_offsets < len(msgs)).all()  # kept the earliest copy


def test_clean_labels_keeps_earliest(spark, silver_nb):
    rows = pd.DataFrame(
        {
            "is_valid": [True, True, True, False],
            "is_test": [False, False, False, False],
            "payload": [
                {"transaction_id": "a", "label": "fraud", "label_source": "chargeback"},
                {"transaction_id": "a", "label": "fraud", "label_source": "customer_report"},
                {"transaction_id": "b", "label": "fraud", "label_source": "chargeback"},
                {"transaction_id": "c", "label": "fraud", "label_source": "chargeback"},
            ],
            "event_ts": pd.to_datetime(["2020-01-01"] * 4),
            "label_known_ts": pd.to_datetime(
                ["2020-02-20", "2020-01-03", "2020-02-01", "2020-01-05"]
            ),
            "ingested_at": pd.to_datetime(["2026-10-27"] * 4),
        }
    )
    schema = (
        "is_valid boolean, is_test boolean, "
        "payload struct<transaction_id:string,label:string,label_source:string>, "
        "event_ts timestamp, label_known_ts timestamp, ingested_at timestamp"
    )
    df = spark.createDataFrame(
        [
            tuple(v.to_pydatetime() if isinstance(v, pd.Timestamp) else v for v in r)
            for r in rows.itertuples(index=False)
        ],
        schema,
    )
    out = silver_nb["clean_labels"](df).toPandas().set_index("transaction_id")
    assert sorted(out.index) == ["a", "b"]  # c was invalid
    assert out.loc["a", "label_source"] == "customer_report"  # earliest known


# ---------------------------------------------------------------- gold

NUMERIC_TOL = 1e-6


def _compare(spark_df: pd.DataFrame, ref_df: pd.DataFrame, numeric_cols, cat_cols):
    m = spark_df.merge(ref_df, on="transaction_id", suffixes=("_s", "_r"))
    assert len(m) == len(ref_df) == len(spark_df)
    for c in numeric_cols:
        a = m[f"{c}_s"].astype(float).to_numpy()
        b = m[f"{c}_r"].astype(float).to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=NUMERIC_TOL, atol=NUMERIC_TOL)
        bad = ~(both_nan | close)
        assert not bad.any(), (
            f"{c}: {bad.sum()} mismatches, e.g. spark={a[bad][:3]} ref={b[bad][:3]}"
        )
    for c in cat_cols:
        assert (m[f"{c}_s"] == m[f"{c}_r"]).all(), c


def test_spark_features_match_pandas_reference(spark, gold_nb, silver_pdf):
    spark_out = gold_nb["build_features"](_to_spark(spark, silver_pdf)).toPandas()
    ref_out = reference.build_features(silver_pdf)
    _compare(spark_out, ref_out, gold_nb["NUMERIC_FEATURES"], gold_nb["CATEGORICAL_FEATURES"])


def _perturb_future(pdf: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Change everything at/after cutoff: scale amounts, drop some rows, add new ones."""
    rng = np.random.default_rng(0)
    past = pdf[pdf["event_ts"] < cutoff]
    future = pdf[pdf["event_ts"] >= cutoff].copy()
    future["amount"] = future["amount"] * 7.0
    future = future[rng.random(len(future)) > 0.2]
    extra = future.sample(200, random_state=1).copy()
    extra["transaction_id"] = [f"injected-{i}" for i in range(len(extra))]
    extra["event_ts"] = extra["event_ts"] + pd.Timedelta(minutes=5)
    return pd.concat([past, future, extra], ignore_index=True)


@pytest.mark.parametrize("engine", ["pandas", "spark"])
def test_features_ignore_the_future(engine, spark, gold_nb, silver_pdf):
    cutoff = silver_pdf["event_ts"].quantile(0.5)
    perturbed = _perturb_future(silver_pdf, cutoff)

    if engine == "pandas":
        before = reference.build_features(silver_pdf)
        after = reference.build_features(perturbed)
    else:
        before = gold_nb["build_features"](_to_spark(spark, silver_pdf)).toPandas()
        after = gold_nb["build_features"](_to_spark(spark, perturbed)).toPandas()

    past_ids = silver_pdf.loc[silver_pdf["event_ts"] < cutoff, "transaction_id"]
    b = before[before["transaction_id"].isin(past_ids)]
    a = after[after["transaction_id"].isin(past_ids)]
    _compare(b, a, gold_nb["NUMERIC_FEATURES"], gold_nb["CATEGORICAL_FEATURES"])


def test_label_views_respect_cutoff(spark, gold_nb):
    feats = spark.createDataFrame(
        pd.DataFrame(
            {
                "transaction_id": ["a", "b", "c"],
                "event_ts": pd.to_datetime(["2020-02-01", "2020-02-20", "2020-02-25"]),
            }
        )
    )
    labels = spark.createDataFrame(
        pd.DataFrame(
            {
                "transaction_id": ["a", "b"],
                "label_known_ts": pd.to_datetime(["2020-02-03", "2020-04-15"]),
                "label_source": ["customer_report", "chargeback"],
            }
        )
    )
    out = (
        gold_nb["attach_labels"](feats, labels, "2020-04-01").toPandas().set_index("transaction_id")
    )
    assert out.loc["a", ["label_final", "label_at_cutoff"]].tolist() == [1, 1]
    assert out.loc["b", ["label_final", "label_at_cutoff"]].tolist() == [
        1,
        0,
    ]  # chargeback too late
    assert out.loc["c", ["label_final", "label_at_cutoff"]].tolist() == [0, 0]


def test_feature_signal_on_mock(spark, gold_nb, silver_pdf, txns):
    """Sanity: fraud bursts should show up as higher 24h velocity."""
    out = reference.build_features(silver_pdf).merge(txns[["transaction_id", "is_fraud"]])
    assert (
        out.loc[out.is_fraud == 1, "cnt_24h"].mean()
        > 2 * out.loc[out.is_fraud == 0, "cnt_24h"].mean()
    )
