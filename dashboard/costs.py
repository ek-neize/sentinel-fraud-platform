"""Cost model for the capacity planner (mirrors sentinel.evaluation.business._costs).

Kept here so the dashboard folder deploys without the `sentinel` package; a test checks
the two implementations agree.
"""

from __future__ import annotations

COST_FIELDS = {
    "investigation_cost_per_alert": "Review cost per alert ($)",
    "escalation_cost_per_case": "Escalation cost per caught fraud ($)",
    "false_positive_friction_cost": "Customer friction per false positive ($)",
    "missed_fraud_loss_rate": "Share of a missed fraud's amount lost",
    "missed_fraud_fixed_cost": "Fixed cost per missed fraud ($)",
    "caught_fraud_recovery_rate": "Share of a caught fraud's amount recovered",
}


def costs(s: dict, c: dict) -> dict:
    """s: sums with txns, frauds, fraud_amt, alerts, tp, fp, caught_amt."""
    loss = c["missed_fraud_loss_rate"]
    missed_amt = s["fraud_amt"] - s["caught_amt"]
    missed_n = s["frauds"] - s["tp"]
    review = s["alerts"] * c["investigation_cost_per_alert"]
    escalation = s["tp"] * c["escalation_cost_per_case"]
    friction = s["fp"] * c["false_positive_friction_cost"]
    fraud_loss = (
        missed_amt * loss
        + missed_n * c["missed_fraud_fixed_cost"]
        + s["caught_amt"] * (1 - c["caught_fraud_recovery_rate"]) * loss
    )
    total = review + escalation + friction + fraud_loss
    do_nothing = s["fraud_amt"] * loss + s["frauds"] * c["missed_fraud_fixed_cost"]
    return {
        "cost_review": review,
        "cost_escalation": escalation,
        "cost_friction": friction,
        "cost_fraud_loss": fraud_loss,
        "cost_total": total,
        "cost_do_nothing": do_nothing,
        "net_benefit": do_nothing - total,
    }
