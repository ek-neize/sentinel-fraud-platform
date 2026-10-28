import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from fastapi.testclient import TestClient  # noqa: E402

from sentinel.serving.app import ModelBundle, create_app, risk_band  # noqa: E402


@pytest.fixture(scope="module")
def client(txns, labels, watchlist, cfg):
    from sentinel.features.training_set import build_training_frame
    from sentinel.models.training import ExperimentConfig, run_experiments

    df = build_training_frame(txns, labels, watchlist, cfg["split"])
    tr, va = df[df.split == "train"], df[df.split == "valid"].reset_index(drop=True)
    _, models = run_experiments(
        tr, va, [ExperimentConfig("xgb", "xgb", "full", {"n_estimators": 60, "max_depth": 3})]
    )
    bundle = ModelBundle(models["xgb"]["pipeline"], threshold=0.3, version="test")
    with TestClient(create_app(bundle)) as c:
        yield c, va


def _record(row) -> dict:
    keys = [
        "transaction_id",
        "amount",
        "card_prior_txn_count",
        "card_prior_amt_mean",
        "card_prior_amt_std",
        "card_prior_amt_max",
        "amt_to_prior_mean",
        "amt_to_prior_max",
        "amt_zscore",
        "secs_since_prev_txn",
        "cnt_1h",
        "cnt_24h",
        "cnt_7d",
        "amt_1h",
        "amt_24h",
        "amt_7d",
        "hour_of_day",
        "day_of_week",
        "home_merch_km",
        "log_city_pop",
        "wl_card_hit",
        "wl_merchant_hit",
        "category",
        "state",
    ]
    out = {}
    for k in keys:
        v = row[k]
        out[k] = None if v != v else (v.item() if hasattr(v, "item") else v)  # NaN -> None
    return out


def test_health(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == "test"


def test_score_fraud_like_transaction(client):
    c, va = client
    fraud = va[va.label_final == 1].iloc[0]
    body = c.post("/score", json=_record(fraud)).json()
    assert 0 <= body["score"] <= 1
    assert body["alert"] == (body["score"] >= 0.3)
    assert body["disclaimer"].startswith("Risk scores and reasons are model signals")
    if body["alert"]:
        assert body["reasons"] and body["reasons"][0]["weight"] > 0


def test_score_matches_pipeline(client):
    c, va = client
    rows = va.head(5)
    batch = c.post("/score/batch", json=[_record(r) for _, r in rows.iterrows()]).json()
    assert [b["transaction_id"] for b in batch] == rows["transaction_id"].tolist()
    assert c.get("/health").json()["transactions_scored"] >= 5


def test_validation_rejects_bad_input(client):
    c, va = client
    bad = _record(va.iloc[0]) | {"amount": -5}
    assert c.post("/score", json=bad).status_code == 422
    missing = {k: v for k, v in _record(va.iloc[0]).items() if k != "category"}
    assert c.post("/score", json=missing).status_code == 422


def test_no_model_returns_503():
    with TestClient(create_app(None)) as c:
        assert c.get("/health").json()["status"] in ("no_model", "ok")


def test_risk_band():
    assert risk_band(0.95, 0.2) == "high"
    assert risk_band(0.25, 0.2) == "elevated"
    assert risk_band(0.1, 0.2) == "low"
