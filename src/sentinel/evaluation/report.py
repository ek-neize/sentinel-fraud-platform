"""Step 6 orchestration: one call turns validation + test scores into every table,
the headline numbers and a Markdown report.

Rules: thresholds are fitted on validation only; the test period is scored once and
judged with `label_final` (all labels, including late chargebacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.alerts.queue import queue_stats, reviewed_mask, simulate_queue
from sentinel.evaluation.business import (
    age_band,
    bootstrap_ci,
    capacity_curve,
    choose_operating_point,
    daily_aggregates,
    fairness_report,
    flat_costs,
    overflow_stats,
    policy_rule,
    policy_threshold,
    policy_threshold_capped,
    policy_topk,
    summarize,
)
from sentinel.evaluation.metrics import ranking_metrics, threshold_for_daily_capacity

LABEL = "label_final"
CI_METRICS = ["recall", "precision", "amount_recall", "fp_per_1000_txns", "net_benefit"]


@dataclass
class EvaluationResult:
    operating_k: int
    thresholds: dict
    ranking: pd.DataFrame
    curve: pd.DataFrame
    strategies: pd.DataFrame
    intervals: pd.DataFrame
    overflow: dict
    queue: dict
    fairness: pd.DataFrame
    headline: dict
    markdown: str = ""
    figures: dict = field(default_factory=dict)


def _days(df):
    return df["event_ts"].dt.floor("D")


def run_business_evaluation(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    scores: dict[str, dict[str, np.ndarray]],
    cfg: dict,
    champion: str = "champion",
    baseline: str = "baseline",
    bootstrap_n: int | None = None,
) -> EvaluationResult:
    costs = flat_costs(cfg)
    ev = cfg["evaluation"]
    ks = list(ev["capacities"])
    n_boot = bootstrap_n or ev["bootstrap_iterations"]

    # 1. thresholds per model and budget, fitted on validation only
    thresholds = {
        m: {k: threshold_for_daily_capacity(s["valid"], _days(valid), k) for k in ks}
        for m, s in scores.items()
    }

    # 2. threshold-free ranking quality on test
    ranking = pd.DataFrame(
        [{"model": m, **ranking_metrics(test[LABEL], s["test"])} for m, s in scores.items()]
    )

    # 3. capacity curve on test
    curve = pd.concat(
        [
            capacity_curve(test, s["test"], thresholds[m], costs).assign(model=m)
            for m, s in scores.items()
        ],
        ignore_index=True,
    )
    k = choose_operating_point(curve[curve["model"] == champion])

    # 4. strategies at the operating budget
    champ_test = scores[champion]["test"]
    thr = thresholds[champion][k]
    policies = {
        "No model (every fraud missed)": np.zeros(len(test), dtype=bool),
        "Watchlist rule: review every hit": policy_rule(test, "wl_any_hit"),
        f"Watchlist rule: top {k}/day by amount": policy_rule(test, "wl_any_hit", k),
    }
    if baseline in scores:
        policies[f"Logistic regression: threshold, max {k}/day"] = policy_threshold_capped(
            test, scores[baseline]["test"], thresholds[baseline][k], k
        )
    main_name = f"XGBoost champion: threshold, max {k}/day"
    policies[main_name] = policy_threshold_capped(test, champ_test, thr, k)

    # backlog queue: overflow waits (up to max_age_days) instead of being dropped
    max_age = cfg.get("alerting", {}).get("max_age_days", 3)
    queue = {}
    backlog_names = {}
    for mode, label in (("score", "by score"), ("expected_loss", "by expected loss")):
        q = simulate_queue(test, champ_test, thr, k, max_age, priority_mode=mode)
        name = f"XGBoost champion: backlog queue {label}, {k}/day, {max_age}-day expiry"
        policies[name] = reviewed_mask(len(test), q)
        queue[mode] = queue_stats(q, test[LABEL].to_numpy())
        backlog_names[mode] = name
    policies["XGBoost champion: threshold, no cap (overtime)"] = policy_threshold(champ_test, thr)
    policies[f"XGBoost champion: daily top-{k} (upper bound)"] = policy_topk(test, champ_test, k)

    strategies = pd.DataFrame(
        [{"strategy": name, **summarize(test, f, costs)} for name, f in policies.items()]
    )

    # 5. bootstrap intervals for the policies that matter
    ci_rows = []
    for name in (
        main_name,
        f"Watchlist rule: top {k}/day by amount",
        f"Logistic regression: threshold, max {k}/day",
    ):
        if name not in policies:
            continue
        agg = daily_aggregates(test, policies[name])
        ci = bootstrap_ci(agg, costs, n=n_boot, level=ev["ci_level"])
        point = summarize(test, policies[name], costs)
        for m in CI_METRICS:
            ci_rows.append(
                {
                    "strategy": name,
                    "metric": m,
                    "estimate": point[m],
                    "ci_low": ci[m][0],
                    "ci_high": ci[m][1],
                }
            )
    intervals = pd.DataFrame(ci_rows)

    # 6. capacity overflow and fairness for the deployed policy
    overflow = overflow_stats(test, champ_test, thr, k)
    groups = {}
    if "gender" in test:
        groups["gender"] = test["gender"].astype(str)
    if "age_at_txn" in test:
        groups["age_band"] = age_band(test["age_at_txn"], ev["age_bands"])
    fairness = fairness_report(test, policies[main_name], groups) if groups else pd.DataFrame()

    main = strategies.set_index("strategy").loc[main_name]
    rule = strategies.set_index("strategy").loc[f"Watchlist rule: top {k}/day by amount"]
    ci_main = intervals[intervals["strategy"] == main_name].set_index("metric")
    headline = {
        "test_period": f"{test['event_ts'].min():%Y-%m-%d} to {test['event_ts'].max():%Y-%m-%d}",
        "test_txns": int(len(test)),
        "test_frauds": int(test[LABEL].sum()),
        "frauds_per_day": float(test[LABEL].sum() / test["event_ts"].dt.floor("D").nunique()),
        "operating_k": k,
        "recall": float(main["recall"]),
        "recall_ci": tuple(ci_main.loc["recall", ["ci_low", "ci_high"]]),
        "precision": float(main["precision"]),
        "precision_ci": tuple(ci_main.loc["precision", ["ci_low", "ci_high"]]),
        "amount_recall": float(main["amount_recall"]),
        "fp_per_1000_txns": float(main["fp_per_1000_txns"]),
        "net_benefit": float(main["net_benefit"]),
        "net_benefit_ci": tuple(ci_main.loc["net_benefit", ["ci_low", "ci_high"]]),
        "rule_recall": float(rule["recall"]),
        "rule_net_benefit": float(rule["net_benefit"]),
        "cost_do_nothing": float(main["cost_do_nothing"]),
        "champion_pr_auc": float(ranking.set_index("model").loc[champion, "pr_auc"]),
    }
    by_name = strategies.set_index("strategy")
    for mode, name in backlog_names.items():
        headline[f"backlog_{mode}_recall"] = float(by_name.loc[name, "recall"])
        headline[f"backlog_{mode}_amount_recall"] = float(by_name.loc[name, "amount_recall"])
        headline[f"backlog_{mode}_net_benefit"] = float(by_name.loc[name, "net_benefit"])
    result = EvaluationResult(
        k, thresholds, ranking, curve, strategies, intervals, overflow, queue, fairness, headline
    )
    result.markdown = to_markdown(result, cfg)
    return result


def _pct(x):
    return f"{100 * x:.1f}%"


def _usd(x):
    return f"${x:,.0f}"


def to_markdown(r: EvaluationResult, cfg: dict) -> str:
    h = r.headline
    k = r.operating_k
    c = cfg["costs"]
    lines = [
        "# Sentinel — test-period results",
        "",
        f"Held-out test period **{h['test_period']}**: {h['test_txns']:,} transactions, "
        f"{h['test_frauds']:,} frauds (~{h['frauds_per_day']:.1f}/day). Thresholds were fixed on "
        "validation (Apr–Jun 2020) and applied unchanged. 95% intervals: day-level bootstrap.",
        "",
        "## Headline",
        "",
        f"With a review budget of **{k} alerts/day**, the XGBoost champion caught "
        f"**{_pct(h['recall'])}** of fraud "
        f"(95% CI {_pct(h['recall_ci'][0])}–{_pct(h['recall_ci'][1])}) "
        f"and **{_pct(h['amount_recall'])} of fraud dollars**, with precision "
        f"{_pct(h['precision'])} "
        f"(CI {_pct(h['precision_ci'][0])}–{_pct(h['precision_ci'][1])}) and "
        f"{h['fp_per_1000_txns']:.1f} false positives per 1,000 transactions. Net benefit vs. no "
        f"model: **{_usd(h['net_benefit'])}** (CI {_usd(h['net_benefit_ci'][0])}–"
        f"{_usd(h['net_benefit_ci'][1])}); the watchlist rule with the same budget caught "
        f"{_pct(h['rule_recall'])} for {_usd(h['rule_net_benefit'])}.",
        "",
        "## Strategies at the operating budget",
        "",
    ]
    cols = [
        "strategy",
        "alerts_per_day",
        "recall",
        "precision",
        "amount_recall",
        "fp_per_1000_txns",
        "cost_total",
        "net_benefit",
    ]
    t = r.strategies[cols].copy()
    for col in ("recall", "precision", "amount_recall"):
        t[col] = t[col].map(lambda v: "–" if pd.isna(v) else _pct(v))
    for col in ("cost_total", "net_benefit"):
        t[col] = t[col].map(_usd)
    t["alerts_per_day"] = t["alerts_per_day"].map(lambda v: f"{v:.1f}")
    t["fp_per_1000_txns"] = t["fp_per_1000_txns"].map(lambda v: f"{v:.1f}")
    lines += [t.to_markdown(index=False), ""]

    cur = r.curve[(r.curve["model"] == "champion") & (r.curve["policy"] == "threshold_capped")]
    cur = cur[
        ["capacity_k", "recall", "precision", "amount_recall", "fp_per_1000_txns", "net_benefit"]
    ].copy()
    for col in ("recall", "precision", "amount_recall"):
        cur[col] = cur[col].map(_pct)
    cur["fp_per_1000_txns"] = cur["fp_per_1000_txns"].map(lambda v: f"{v:.1f}")
    cur["net_benefit"] = cur["net_benefit"].map(_usd)
    o = r.overflow
    lines += [
        "## Capacity curve (champion, deployable policy)",
        "",
        cur.to_markdown(index=False),
        "",
        f"Operating budget = highest net benefit (ties → smaller team): **{k}/day**.",
        "",
        "## Busy days",
        "",
        f"The fixed threshold exceeded {k} alerts on {o['days_over_capacity']} of {o['days']} days "
        f"(max {o['max_alerts_in_a_day']}/day). Capping review at {k}/day left "
        f"{o['alerts_not_reviewed']:,} alerts unreviewed, of which {o['frauds_lost_to_overflow']} "
        "were fraud.",
        "",
    ]
    if r.queue:
        qs = r.queue["score"]
        ql = r.queue["expected_loss"]
        max_age = cfg.get("alerting", {}).get("max_age_days", 3)
        lines += [
            "## Backlog queue (overflow waits instead of being dropped)",
            "",
            f"Same {k}/day team, unreviewed alerts carried over for up to {max_age} days, "
            "highest priority first:",
            "",
            "| Priority | Fraud caught | $ caught | Net benefit | Same-day reviews | "
            "Mean wait (fraud) | Frauds expired |",
            "|---|---|---|---|---|---|---|",
            f"| Fraud score | {_pct(h['backlog_score_recall'])} | "
            f"{_pct(h['backlog_score_amount_recall'])} | {_usd(h['backlog_score_net_benefit'])} | "
            f"{_pct(qs['share_reviewed_same_day'])} | {qs['mean_wait_days_fraud']:.2f} days | "
            f"{qs['frauds_expired']} |",
            f"| Expected loss (score × amount) | {_pct(h['backlog_expected_loss_recall'])} | "
            f"{_pct(h['backlog_expected_loss_amount_recall'])} | "
            f"{_usd(h['backlog_expected_loss_net_benefit'])} | "
            f"{_pct(ql['share_reviewed_same_day'])} | {ql['mean_wait_days_fraud']:.2f} days | "
            f"{ql['frauds_expired']} |",
            "",
            f"Hard daily cap for comparison: {_pct(h['recall'])} of fraud, "
            f"{_usd(h['net_benefit'])}. A fraud reviewed within the expiry window counts as "
            "caught; the extra exposure while it waits is reported (mean wait) but not priced.",
            "",
        ]
    if len(r.fairness):
        f = r.fairness[
            [
                "attribute",
                "group",
                "txns",
                "fraud_rate",
                "alert_rate",
                "recall",
                "false_positive_rate",
                "alert_rate_ratio",
                "fpr_ratio",
            ]
        ].copy()
        for col in ("fraud_rate", "alert_rate", "false_positive_rate"):
            f[col] = f[col].map(lambda v: f"{100 * v:.2f}%")
        f["recall"] = f["recall"].map(lambda v: "–" if pd.isna(v) else _pct(v))
        for col in ("alert_rate_ratio", "fpr_ratio"):
            f[col] = f[col].map(lambda v: f"{v:.2f}")
        lines += [
            "## Fairness check (protected attributes are not model inputs)",
            "",
            f.to_markdown(index=False),
            "",
            "Ratios compare each group with the overall rate; outside 0.8–1.25 warrants review. "
            "Differences in alert rate partly reflect different fraud rates per group.",
            "",
        ]
    lines += [
        "## Cost assumptions (config/assumptions.yaml)",
        "",
        f"Review {_usd(c['investigation_cost_per_alert'])}/alert · escalation "
        f"{_usd(c['escalation_cost_per_case'])}/caught fraud · customer friction "
        f"{_usd(c['false_positive_friction_cost'])}/false positive · missed fraud loses "
        f"{_pct(c['missed_fraud_loss_rate'])} of its amount + "
        f"{_usd(c['missed_fraud_fixed_cost'])} · "
        f"caught fraud recovers {_pct(cfg['detection']['caught_fraud_recovery_rate'])}.",
        "",
        "*Synthetic data (Sparkov): fraud patterns are cleaner than in production, so absolute "
        "numbers are optimistic; the method and the comparisons are the point.*",
    ]
    return "\n".join(lines)


def capacity_figure(result: EvaluationResult):
    """Recall / precision / net benefit vs. daily budget (matplotlib Figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur = result.curve[result.curve["policy"] == "threshold_capped"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for model, g in cur.groupby("model"):
        g = g.sort_values("capacity_k")
        axes[0].plot(g["capacity_k"], g["recall"], marker="o", label=f"{model} recall")
        axes[0].plot(
            g["capacity_k"], g["precision"], marker="o", ls="--", label=f"{model} precision"
        )
        axes[1].plot(g["capacity_k"], g["net_benefit"] / 1000, marker="o", label=model)
    for ax in axes:
        ax.axvline(result.operating_k, color="grey", lw=1, ls=":")
        ax.set_xlabel("Review budget (alerts/day)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Share")
    axes[0].set_title("Fraud caught vs. alert quality")
    axes[1].set_ylabel("Net benefit ($k)")
    axes[1].set_title("Net benefit vs. no model")
    fig.tight_layout()
    return fig
