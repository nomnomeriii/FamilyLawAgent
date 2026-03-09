from __future__ import annotations

from typing import Tuple

RED_FLAGS = [
    "hide money",
    "evade",
    "harass",
    "threaten",
    "hurt",
    "hide assets",
    "coercion",
    "violence",
    "endanger",
]


def safety_classifier(prompt: str) -> Tuple[bool, str]:
    """Return (is_safe, message) for a user query."""
    lowered = (prompt or "").lower()
    if any(flag in lowered for flag in RED_FLAGS):
        return (
            False,
            "SAFETY ALERT: I cannot provide coaching for hiding assets, evading orders, "
            "or harassment. Please seek professional legal aid or emergency resources.",
        )
    return True, "Safe"
