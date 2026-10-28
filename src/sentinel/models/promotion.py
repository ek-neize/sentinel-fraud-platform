"""Champion/challenger gate: a new model replaces @champion only if it is better.

Used by notebook 05 before moving the `champion` alias. A rejected candidate is still
registered, gets the `challenger` alias and a tag explaining why, so nothing is lost.
Both models are compared on the same validation period; the test period stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


def promotion_decision(candidate: dict, champion: dict | None, rules: dict) -> Decision:
    """candidate / champion: validation metrics (as logged to MLflow)."""
    p, g = rules["primary_metric"], rules["guard_metric"]
    if champion is None:
        return Decision(True, "no current champion")
    for name, m in (("candidate", candidate), ("champion", champion)):
        missing = [x for x in (p, g) if m.get(x) is None]
        if missing:
            ok = name == "champion"  # an incomparable incumbent should not block forever
            return Decision(ok, f"{name} lacks {', '.join(missing)}")
    gain = candidate[p] - champion[p]
    drop = champion[g] - candidate[g]
    if gain < rules.get("min_primary_gain", 0.0):
        return Decision(
            False,
            f"{p} {candidate[p]:.4f} vs champion {champion[p]:.4f} "
            f"(gain {gain:+.4f} below {rules.get('min_primary_gain', 0.0)})",
        )
    if drop > rules.get("max_guard_drop", 0.0):
        return Decision(False, f"{g} drops by {drop:.4f} (allowed {rules['max_guard_drop']})")
    return Decision(
        True,
        f"{p} {candidate[p]:.4f} vs {champion[p]:.4f} (gain {gain:+.4f}); {g} change {-drop:+.4f}",
    )
