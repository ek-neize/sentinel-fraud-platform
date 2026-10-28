import numpy as np
import pandas as pd
import pytest

from sentinel.alerts.digest import DISCLAIMER, digest_text, render_digest
from sentinel.alerts.explain import contributions, raw_feature, top_reasons
from sentinel.alerts.notify import EmailSettings, send_email, write_digest_file

pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def trained(txns, labels, watchlist, cfg):
    from sentinel.features.training_set import build_training_frame
    from sentinel.models.training import ExperimentConfig, run_experiments

    df = build_training_frame(txns, labels, watchlist, cfg["split"])
    tr, va = df[df.split == "train"], df[df.split == "valid"].reset_index(drop=True)
    _, models = run_experiments(
        tr,
        va,
        [
            ExperimentConfig("xgb", "xgb", "full", {"n_estimators": 60, "max_depth": 3}),
            ExperimentConfig("lr", "logreg", "full", {"C": 0.1}),
        ],
    )
    return {k: v["pipeline"] for k, v in models.items()}, va


def test_raw_feature_mapping():
    assert raw_feature("log__amt_24h") == "amt_24h"
    assert raw_feature("cat__category_shopping_net") == "category"
    assert raw_feature("cat__hour_of_day_22") == "hour_of_day"
    assert raw_feature("num__wl_card_hit") == "wl_card_hit"
    assert raw_feature("log__missingindicator_secs_since_prev_txn") == "secs_since_prev_txn"


def test_xgb_contributions_sum_to_margin(trained):
    import xgboost as xgb

    models, va = trained
    pipe, x = models["xgb"], va.head(40)
    c = contributions(pipe, x)
    dm = xgb.DMatrix(pipe[:-1].transform(x))
    booster = pipe[-1].get_booster()
    bias = booster.predict(dm, pred_contribs=True)[:, -1]
    margin = booster.predict(dm, output_margin=True)
    assert np.allclose(c.sum(axis=1).to_numpy() + bias, margin, atol=1e-4)
    assert "category" in c.columns and not any(col.startswith("cat__") for col in c.columns)


@pytest.mark.parametrize("model", ["xgb", "lr"])
def test_top_reasons_are_readable_and_positive(trained, model):
    models, va = trained
    pipe = models[model]
    s = pipe.predict_proba(va)[:, 1]
    top = va.assign(s=s).nlargest(5, "s")
    r = top_reasons(pipe, top, n=3)
    assert r["reason_1"].notna().all()
    assert all(all(w > 0 for w in ws) for ws in r["reason_weights"])
    assert not r["reason_1"].str.contains("__").any()  # no encoded names leak out


def _alerts():
    return pd.DataFrame(
        {
            "transaction_id": ["abc123", "def456"],
            "event_ts": pd.to_datetime(["2020-12-14 23:10", "2020-12-14 01:05"]),
            "card_id": ["card_000000001234", "card_000000005678"],
            "merchant_name": ["Shop <script>", "Gas & Go"],
            "category": ["shopping_net", "gas_transport"],
            "amount": [912.4, 15.0],
            "score": [0.97, 0.55],
            "status": ["reviewed", "open"],
            "reason_1": ["amount is 8.2x this card's average", "card is on the fraud hot-list"],
            "reason_2": ["$1,240 spent on this card in the previous 24h", None],
            "reason_3": [None, None],
        }
    )


def test_digest_html_is_escaped_and_labelled():
    html = render_digest(
        "2020-12-14",
        _alerts(),
        {"new_alerts": 2, "reviewed": 1, "open_end": 1, "expired": 0, "capacity": 20},
        {"name": "fraud_model", "version": 3, "threshold": 0.12},
    )
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert DISCLAIMER in html
    assert "…1234" in html and "High" in html and "Elevated" in html
    assert "…abc123" in html  # last characters of the transaction id
    assert "digest" in digest_text("2020-12-14", _alerts(), {}).lower()


def test_digest_file_always_written(tmp_path):
    path = write_digest_file("<html></html>", tmp_path, "2020-12-14")
    assert path.name == "digest_2020-12-14.html" and path.exists()


def test_email_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ALERT_EMAIL_ENABLED", raising=False)
    assert "disabled" in send_email("s", "<b>x</b>", "x", EmailSettings.from_env())


def test_email_reports_missing_settings(monkeypatch):
    monkeypatch.setenv("ALERT_EMAIL_ENABLED", "true")
    for v in ("SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO"):
        monkeypatch.delenv(v, raising=False)
    msg = send_email("s", "<b>x</b>", "x", EmailSettings.from_env())
    assert "missing" in msg and "SMTP_PASSWORD" in msg


def test_email_sent_through_smtp(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, user, pw):
            sent["login"] = (user, pw)

        def send_message(self, msg):
            sent["msg"] = msg

    settings = EmailSettings(
        True, "smtp.gmail.com", 465, "me@gmail.com", "app-pass", ["me@gmail.com"]
    )
    status = send_email("Sentinel", "<b>hi</b>", "hi", settings, smtp_factory=FakeSMTP)
    assert status.startswith("email sent")
    assert sent["host"] == "smtp.gmail.com" and sent["login"] == ("me@gmail.com", "app-pass")
    assert sent["msg"]["Subject"] == "Sentinel"
    assert sent["msg"].get_content_type() == "multipart/alternative"
