"""FastAPI scoring service.

    uvicorn sentinel.serving.app:app --port 8000

Prometheus metrics at /metrics (requests, latency, scores, alerts, input amounts and the
model's validation reference values used by the drift alerts in deploy/prometheus).

The model is loaded once at startup from SENTINEL_MODEL_PATH (default `models/champion`,
created by `scripts/export_model.py`), together with `sentinel_meta.json` (model version and
the operating threshold chosen in Step 6). SENTINEL_THRESHOLD overrides the threshold.

v1 scores a *feature record* (the gold-layer columns). Computing card-history features
online needs a feature store and is out of scope; batch scoring in Databricks covers the
end-to-end flow.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from sentinel.alerts.digest import DISCLAIMER
from sentinel.alerts.explain import top_reasons
from sentinel.models.feature_sets import CATEGORICAL, NUMERIC

DEFAULT_MODEL_PATH = "models/champion"


class TransactionFeatures(BaseModel):
    """One gold-layer feature record. Unknown extra fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: str | None = None
    amount: float = Field(gt=0)
    log_amount: float | None = None
    card_prior_txn_count: float = 0
    card_prior_amt_mean: float | None = None
    card_prior_amt_std: float | None = None
    card_prior_amt_max: float | None = None
    amt_to_prior_mean: float | None = None
    amt_to_prior_max: float | None = None
    amt_zscore: float | None = None
    secs_since_prev_txn: float | None = None
    cnt_1h: float = 0
    cnt_24h: float = 0
    cnt_7d: float = 0
    amt_1h: float = 0
    amt_24h: float = 0
    amt_7d: float = 0
    hour_of_day: float = Field(ge=0, le=23)
    day_of_week: float = Field(ge=1, le=7)
    is_night: float | None = None
    home_merch_km: float | None = None
    log_city_pop: float | None = None
    wl_card_hit: float = 0
    wl_merchant_hit: float = 0
    wl_any_hit: float | None = None
    category: str
    state: str

    def to_frame(self) -> pd.DataFrame:
        d = self.model_dump()
        import math

        if d["log_amount"] is None:
            d["log_amount"] = math.log1p(d["amount"])
        if d["is_night"] is None:
            d["is_night"] = float(d["hour_of_day"] >= 22 or d["hour_of_day"] < 4)
        if d["wl_any_hit"] is None:
            d["wl_any_hit"] = float(max(d["wl_card_hit"], d["wl_merchant_hit"]))
        return pd.DataFrame([d])


class Reason(BaseModel):
    feature: str
    description: str
    weight: float


class ScoreResponse(BaseModel):
    transaction_id: str | None
    score: float
    alert: bool
    risk_band: str
    threshold: float
    reasons: list[Reason]
    model_version: str
    disclaimer: str = DISCLAIMER


def risk_band(score: float, threshold: float) -> str:
    if score >= max(0.9, threshold):
        return "high"
    if score >= threshold:
        return "elevated"
    return "low"


class ModelBundle:
    def __init__(
        self, pipeline, threshold: float, version: str = "local", reference: dict | None = None
    ):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.version = str(version)
        # Validation-period reference values (from sentinel_meta.json), for drift alerts.
        self.reference = reference or {}

    @classmethod
    def from_path(cls, path: str | Path) -> ModelBundle:
        import mlflow

        path = Path(path)
        meta_file = path / "sentinel_meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        threshold = float(os.environ.get("SENTINEL_THRESHOLD", meta.get("threshold", 0.5)))
        return cls(
            mlflow.sklearn.load_model(str(path)),
            threshold,
            meta.get("version", "local"),
            meta.get("reference"),
        )

    def score(self, frame: pd.DataFrame, with_reasons: bool = True) -> list[ScoreResponse]:
        x = frame[NUMERIC + CATEGORICAL]
        scores = self.pipeline.predict_proba(x)[:, 1]
        reasons = top_reasons(self.pipeline, x) if with_reasons else None
        out = []
        for i, s in enumerate(scores):
            rs = []
            if reasons is not None:
                r = reasons.iloc[i]
                rs = [
                    Reason(feature=f, description=r[f"reason_{j + 1}"], weight=w)
                    for j, (f, w) in enumerate(
                        zip(r["reason_features"], r["reason_weights"], strict=True)
                    )
                ]
            out.append(
                ScoreResponse(
                    transaction_id=frame.iloc[i].get("transaction_id"),
                    score=float(s),
                    alert=bool(s >= self.threshold),
                    risk_band=risk_band(float(s), self.threshold),
                    threshold=self.threshold,
                    reasons=rs,
                    model_version=self.version,
                )
            )
        return out


class Metrics:
    """Prometheus metrics, one registry per app (so tests can create several apps)."""

    SCORE_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9, 0.99, 1.0)
    AMOUNT_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

    def __init__(self):
        r = self.registry = CollectorRegistry()
        self.requests = Counter(
            "sentinel_http_requests_total", "HTTP requests", ["endpoint", "status"], registry=r
        )
        self.latency = Histogram(
            "sentinel_http_request_duration_seconds",
            "Request latency",
            ["endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
            registry=r,
        )
        self.scored = Counter(
            "sentinel_transactions_scored_total", "Transactions scored", registry=r
        )
        self.alerts = Counter(
            "sentinel_alerts_total",
            "Transactions at or above the threshold",
            ["risk_band"],
            registry=r,
        )
        self.score = Histogram(
            "sentinel_score", "Model score", buckets=self.SCORE_BUCKETS, registry=r
        )
        self.amount = Histogram(
            "sentinel_input_amount",
            "Transaction amount (USD) received",
            buckets=self.AMOUNT_BUCKETS,
            registry=r,
        )
        self.model_loaded = Gauge("sentinel_model_loaded", "1 if a model is loaded", registry=r)
        self.model_info = Gauge("sentinel_model_info", "Loaded model", ["version"], registry=r)
        self.threshold = Gauge("sentinel_threshold", "Operating threshold", registry=r)
        self.ref_alert_rate = Gauge(
            "sentinel_reference_alert_rate", "Share above threshold on validation", registry=r
        )
        self.ref_amount_p50 = Gauge(
            "sentinel_reference_amount_p50", "Median amount on validation", registry=r
        )

    def set_model(self, b: ModelBundle | None) -> None:
        self.model_loaded.set(1 if b else 0)
        if b is None:
            return
        self.model_info.labels(version=b.version).set(1)
        self.threshold.set(b.threshold)
        if "alert_rate" in b.reference:
            self.ref_alert_rate.set(b.reference["alert_rate"])
        if "amount_p50" in b.reference:
            self.ref_amount_p50.set(b.reference["amount_p50"])

    def observe(self, frame: pd.DataFrame, results: list[ScoreResponse]) -> None:
        self.scored.inc(len(results))
        for amt in frame["amount"]:
            self.amount.observe(float(amt))
        for r in results:
            self.score.observe(r.score)
            if r.alert:
                self.alerts.labels(risk_band=r.risk_band).inc()


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    state: dict = {"bundle": bundle, "started": time.time(), "scored": 0}
    metrics = Metrics()
    metrics.set_model(bundle)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["bundle"] is None:
            path = os.environ.get("SENTINEL_MODEL_PATH", DEFAULT_MODEL_PATH)
            if Path(path).exists():
                state["bundle"] = ModelBundle.from_path(path)
        metrics.set_model(state["bundle"])
        yield

    app = FastAPI(title="Sentinel scoring API", version="1.0", lifespan=lifespan)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.middleware("http")
    async def record(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = getattr(route, "path", "unmatched")
            metrics.latency.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
            metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()

    def _bundle() -> ModelBundle:
        if state["bundle"] is None:
            raise HTTPException(503, "Model not loaded. Run scripts/export_model.py first.")
        return state["bundle"]

    @app.get("/health")
    def health():
        b = state["bundle"]
        return {
            "status": "ok" if b else "no_model",
            "model_version": b.version if b else None,
            "threshold": b.threshold if b else None,
            "uptime_s": round(time.time() - state["started"], 1),
            "transactions_scored": state["scored"],
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(txn: TransactionFeatures):
        frame = txn.to_frame()
        res = _bundle().score(frame)
        metrics.observe(frame, res)
        state["scored"] += 1
        return res[0]

    @app.post("/score/batch", response_model=list[ScoreResponse])
    def score_batch(txns: list[TransactionFeatures]):
        if not txns:
            return []
        if len(txns) > 1000:
            raise HTTPException(413, "At most 1000 transactions per request.")
        frame = pd.concat([t.to_frame() for t in txns], ignore_index=True)
        res = _bundle().score(frame)
        metrics.observe(frame, res)
        state["scored"] += len(res)
        return res

    return app


app = create_app()
