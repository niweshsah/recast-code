"""Statistics for the evaluation: Wilson score intervals for success rates."""

from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval (default z) for a binomial proportion, as fractions in [0, 1]."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(successes: list[bool]) -> dict:
    """Success rate and its Wilson interval, in percent."""
    n, k = len(successes), int(sum(successes))
    lo, hi = wilson_interval(k, n)
    return {"trials": n, "successes": k, "rate": 100 * k / n, "ci95": [100 * lo, 100 * hi]}
