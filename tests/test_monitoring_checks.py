"""Monitoring checks, label maturity, promotion gate and the job definition."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sentinel.models.promotion import promotion_decision
from sentinel.monitoring.checks import label_maturity, overall_status, run_checks

ROOT = Path(__file__).resolve().parents[1]
AS_OF = pd.Timestamp("2020-12-31 23:59")


def _scored(n_days=120, per_day=200, seed=0, amount_scale=1.0, alert_share=0.02):
    rng = np.random.default_rng(seed)
    days = pd.date_range(AS_OF.normalize() - pd.Timedelta(days=n_days - 1), periods=n_days)
    ts = np.repeat(days, per_day) + pd.to_timedelta(
        rng.integers(0, 86_400, n_days * per_day), unit="s"
    )
    n = len(ts)
    fraud = rng.random(n) < 0.01
    score = np.where(fraud, rng.uniform(0.5, 1, n), rng.uniform(0, 0.3, n))
    above = (score >= 0.5) | (rng.random(n) < alert_share)
    lag = pd.to_timedelta(rng.integers(1, 60, n), unit="D")
    return pd.DataFrame(
        {
            "event_ts": ts,
            "score": score,
            "above_threshold": above,
            "label_final": fraud.astype(int),
            "label_known_ts": pd.Series(ts + lag).where(fraud),
            "amount": rng.lognormal(3.5, 1, n) * amount_scale,
            "cnt_24h": rng.poisson(3, n).astype(float),
            "card_prior_txn_count": np.arange(n, dtype=float),
            "category": rng.choice(["grocery_pos", "shopping_net"], n),
        }
    )


def _ops(new=20, reviewed=20, confirmed=14, open_end=5, expired=0, k=20, days=14):
    d = pd.date_range(AS_OF.normalize() - pd.Timedelta(days=days - 1), periods=days)
    return pd.DataFrame(
        {
            "day": d,
            "new_alerts": new,
            "reviewed": reviewed,
            "confirmed_fraud": confirmed,
            "open_end": open_end,
            "expired": expired,
            "capacity": k,
        }
    )


@pytest.fixture(scope="module")
def ref():
    return _scored(seed=1)


def _get(checks, check, subject=None):
    c = checks[checks["check"] == check]
    if subject:
        c = c[c["subject"] == subject]
    return c.iloc[0]


def test_healthy_system_is_ok(cfg, ref):
    c = run_checks(
        _scored(seed=2), ref, _ops(), AS_OF, cfg, ["amount", "cnt_24h", "card_prior_txn_count"]
    )
    assert overall_status(c) == "ok", c[c.status != "ok"][["check", "subject", "detail"]]
    # grows by design: huge PSI but exempt
    row = _get(c, "drift", "card_prior_txn_count")
    assert row["value"] > 1 and row["status"] == "ok" and "exempt" in row["detail"]
    assert set(c["check"]) >= {
        "drift",
        "missing values",
        "alert volume",
        "backlog",
        "expiry",
        "precision",
        "recall (matured)",
    }


def test_drift_volume_backlog_and_precision_problems_are_flagged(cfg, ref):
    cur = _scored(seed=3, amount_scale=4)  # amounts x4
    cur.loc[cur.index[-1400:], "cnt_24h"] = np.nan  # last week: inputs missing
    c = run_checks(
        cur,
        ref,
        _ops(new=70, confirmed=4, open_end=60, expired=30),
        AS_OF,
        cfg,
        ["amount", "cnt_24h"],
    )
    assert _get(c, "drift", "amount")["status"] == "critical"
    assert _get(c, "missing values", "cnt_24h")["status"] == "critical"
    assert _get(c, "alert volume")["status"] == "critical"  # 3.5x the budget
    assert _get(c, "backlog")["status"] == "critical"  # 3 days of capacity open
    assert _get(c, "expiry")["status"] == "critical"
    assert _get(c, "precision")["status"] == "critical"  # 20% confirmed
    assert overall_status(c) == "critical"


def test_too_few_alerts_is_also_a_warning(cfg, ref):
    c = run_checks(_scored(seed=4), ref, _ops(new=8), AS_OF, cfg, ["amount"])
    assert _get(c, "alert volume")["status"] == "warn"  # 0.4x the budget


def test_recall_uses_only_matured_cohort(cfg, ref):
    cur = _scored(seed=5)
    mature_end = AS_OF - pd.Timedelta(days=cfg["monitoring"]["matured_after_days"])
    recent = cur["event_ts"] > mature_end
    cur.loc[recent & (cur["label_final"] == 1), "above_threshold"] = False  # recent misses
    c = run_checks(cur, ref, _ops(), AS_OF, cfg, ["amount"])
    assert _get(c, "recall (matured)")["status"] == "ok"  # unaffected by the recent misses
    old = (cur["event_ts"] <= mature_end) & (cur["label_final"] == 1)
    cur.loc[old, "above_threshold"] = False
    c = run_checks(cur, ref, _ops(), AS_OF, cfg, ["amount"])
    assert _get(c, "recall (matured)")["status"] == "critical"


def test_label_maturity_grows_with_age(cfg):
    m = label_maturity(_scored(seed=6), AS_OF, cfg["monitoring"]["label_bands"])
    cov = m["label_coverage"].tolist()
    assert cov == sorted(cov) and cov[-1] == 1.0 and cov[0] < 0.5
    assert (m["frauds_known"] <= m["frauds_final"]).all()


RULES = {
    "primary_metric": "pr_auc",
    "min_primary_gain": 0.0,
    "guard_metric": "topk25_recall",
    "max_guard_drop": 0.01,
}


@pytest.mark.parametrize(
    ("cand", "champ", "promote", "why"),
    [
        ({"pr_auc": 0.80, "topk25_recall": 0.90}, None, True, "no current champion"),
        (
            {"pr_auc": 0.82, "topk25_recall": 0.90},
            {"pr_auc": 0.80, "topk25_recall": 0.90},
            True,
            "gain",
        ),
        (
            {"pr_auc": 0.79, "topk25_recall": 0.95},
            {"pr_auc": 0.80, "topk25_recall": 0.90},
            False,
            "below",
        ),
        (
            {"pr_auc": 0.85, "topk25_recall": 0.85},
            {"pr_auc": 0.80, "topk25_recall": 0.90},
            False,
            "drops",
        ),
        (
            {"pr_auc": 0.85, "topk25_recall": 0.895},
            {"pr_auc": 0.80, "topk25_recall": 0.90},
            True,
            "gain",
        ),
        ({"pr_auc": 0.85}, {"pr_auc": 0.80, "topk25_recall": 0.90}, False, "candidate lacks"),
        ({"pr_auc": 0.85, "topk25_recall": 0.9}, {"pr_auc": 0.80}, True, "champion lacks"),
    ],
)
def test_promotion_gate(cand, champ, promote, why):
    d = promotion_decision(cand, champ, RULES)
    assert d.promote is promote and why in d.reason


def test_promotion_rules_in_config(cfg):
    assert set(RULES) <= set(cfg["promotion"])


def _widgets(nb: Path) -> set[str]:
    return set(re.findall(r'dbutils\.widgets\.(?:text|dropdown)\("([a-z_]+)"', nb.read_text()))


def test_bundle_jobs_reference_real_notebooks_and_widgets():
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())
    for job in bundle["resources"]["jobs"].values():
        keys = {t["task_key"] for t in job["tasks"]}
        for t in job["tasks"]:
            nb = ROOT / t["notebook_task"]["notebook_path"]
            assert nb.exists(), nb
            assert nb.read_text().startswith("# Databricks notebook source")
            params = set(t["notebook_task"].get("base_parameters", {}))
            assert params <= _widgets(nb), (nb.name, params - _widgets(nb))
            assert {d["task_key"] for d in t.get("depends_on", [])} <= keys
    daily = bundle["resources"]["jobs"]["sentinel_daily"]
    order = [t["task_key"] for t in daily["tasks"]]
    assert order.index("monitoring") < order.index("dashboard_snapshot")
    assert daily["schedule"]["pause_status"] == "PAUSED"


def test_compose_and_prometheus_files_agree():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    prom = yaml.safe_load((ROOT / "deploy/prometheus/prometheus.yml").read_text())
    assert set(compose["services"]) >= {"api", "traffic", "prometheus", "grafana"}
    target = prom["scrape_configs"][0]["static_configs"][0]["targets"][0]
    assert target.split(":")[0] in compose["services"]
    rules = yaml.safe_load((ROOT / "deploy/prometheus/rules.yml").read_text())
    exprs = " ".join(r["expr"] for g in rules["groups"] for r in g["rules"])
    for metric in re.findall(r"sentinel_[a-z_]+", exprs):
        base = re.sub(r"_(bucket|count|sum)$", "", metric)
        assert base in (ROOT / "src/sentinel/serving/app.py").read_text(), metric
