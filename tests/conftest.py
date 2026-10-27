import pandas as pd
import pytest

from sentinel.common.config import load_assumptions
from sentinel.data.labels import simulate_label_feed
from sentinel.data.loader import assign_split, to_canonical
from sentinel.data.mock import make_mock_raw
from sentinel.data.watchlist import build_watchlist


@pytest.fixture(scope="session")
def cfg():
    return load_assumptions()


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_mock_raw(n_cards=60, n_merchants=40, txn_per_card_per_day=0.5, seed=11)


@pytest.fixture(scope="session")
def txns(raw, cfg) -> pd.DataFrame:
    df = to_canonical(raw)
    df["split"] = assign_split(df["event_time"], cfg["split"])
    return df


@pytest.fixture(scope="session")
def labels(txns, cfg) -> pd.DataFrame:
    return simulate_label_feed(txns, cfg["labels"]["channels"], seed=1)


@pytest.fixture(scope="session")
def watchlist(txns, labels, cfg) -> pd.DataFrame:
    return build_watchlist(txns, labels, cfg["watchlist"], seed=1)


@pytest.fixture(scope="session")
def spark():
    """One local Spark session for all notebook-transform tests (skipped without pyspark)."""
    pytest.importorskip("pyspark")
    import os
    import time

    os.environ["TZ"] = "UTC"  # pandas naive timestamps must not shift across DST
    time.tzset()
    from pyspark.sql import SparkSession

    s = (
        SparkSession.builder.master("local[2]")
        .appName("sentinel-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .getOrCreate()
    )
    yield s
    s.stop()
