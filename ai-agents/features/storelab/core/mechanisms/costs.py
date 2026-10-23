"""Expected operating costs E C_j(D_j) (SL-16; Ass. cost l.385, Lemma load l.373).

Given Λ the load is D_j ~ Bin(n, p_j(Λ)), with p_j(Λ) linear in the policy.

- **Quadratic** C(D) = λ D² / (2B) is exact through Lemma load:
  E C = λ / (2B) {E D + n(n − 1) E_Λ p²}, a convex quadratic in the policy.
- **Overflow** C(D) = λ (D − B)⁺. h(p) = E (Bin(n, p) − B)⁺ is convex in p, with
  h'(p) = n Pr(Bin(n − 1, p) ≥ c) − n (c − B) Pr(Bin(n − 1, p) = c − 1) for
  c = ⌈B⌉ (B an integer: h'(p) = n Pr(Bin(n − 1, p) ≥ B)). The solver uses the convex
  piecewise-linear outer approximation max_k {h(p_k) + h'(p_k)(p − p_k)} (tangents,
  so it never exceeds h). Since h minus one tangent is convex on the interval where
  that tangent is the maximum, the largest gap is attained where consecutive
  tangents cross: ``Tangents.max_error`` is that maximum, exactly. Outcomes are
  evaluated with the exact h.

Per request: functions return cost / n.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .market import Cost

TANGENTS = 81  # tangent points per overflow cost: 0, 1 and 79 spread over ±8 sd around B / n


def overflow_expectation(n: int, capacity: float, p: np.ndarray | float) -> np.ndarray:
    """h(p) = E (Bin(n, p) − B)⁺."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    d = np.arange(n + 1)
    excess = np.maximum(d - capacity, 0.0)
    return stats.binom.pmf(d[None, :], n, np.atleast_1d(p)[:, None]) @ excess if p.ndim else float(
        stats.binom.pmf(d, n, float(p)) @ excess)


def overflow_slope(n: int, capacity: float, p: np.ndarray) -> np.ndarray:
    """h'(p) (module docstring)."""
    c = math.ceil(capacity)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if c <= 0:
        return np.full_like(p, float(n))
    if c > n:
        return np.zeros_like(p)
    tail = stats.binom.sf(c - 1, n - 1, p)  # Pr(Bin(n − 1, p) ≥ c)
    return n * tail + n * (c - capacity) * stats.binom.pmf(c - 1, n - 1, p)


@dataclass(frozen=True)
class Tangents:
    """Tangents of h / n (per request) at points p_k: h(p)/n ≥ a_k + s_k p."""
    points: np.ndarray
    intercept: np.ndarray
    slope: np.ndarray
    max_error: float  # per request: max_p (h(p) − max_k tangent_k(p)) / n

    def value(self, p: np.ndarray) -> np.ndarray:
        return np.max(self.intercept[None, :] + self.slope[None, :] * np.atleast_1d(p)[:, None], axis=1)


def overflow_tangents(n: int, capacity: float, count: int = TANGENTS) -> Tangents:
    centre = min(max(capacity / n, 0.0), 1.0)
    sd = math.sqrt(max(centre * (1 - centre), 1.0 / n) / n)
    points = np.unique(np.clip(np.r_[0.0, 1.0, centre + sd * np.linspace(-8, 8, count - 2)], 0.0, 1.0))
    h = overflow_expectation(n, capacity, points) / n
    s = overflow_slope(n, capacity, points) / n
    intercept = h - s * points
    # crossings of consecutive tangents (slopes increase with p because h is convex)
    order = np.argsort(points)
    points, intercept, s = points[order], intercept[order], s[order]
    crossings = []
    for k in range(len(points) - 1):
        if s[k + 1] - s[k] > 1e-15:
            crossings.append((intercept[k] - intercept[k + 1]) / (s[k + 1] - s[k]))
    crossings = np.clip(np.asarray(crossings), 0.0, 1.0)
    envelope = np.max(intercept[None, :] + s[None, :] * crossings[:, None], axis=1) if len(crossings) else np.zeros(0)
    error = float(np.max(overflow_expectation(n, capacity, crossings) / n - envelope)) if len(crossings) else 0.0
    return Tangents(points=points, intercept=intercept, slope=s, max_error=max(error, 0.0))


def expected_cost(cost: Cost | None, n: int, state_probs: np.ndarray, p_state: np.ndarray) -> float:
    """Exact E C_j(D_j) / n for one product, p_state = p_j(Λ) by state."""
    if cost is None or cost.weight == 0.0:
        return 0.0
    p = np.asarray(p_state, dtype=float)
    if cost.kind == "quadratic":
        return cost.weight / (2 * cost.capacity) * float(state_probs @ (p + (n - 1) * p * p))
    return cost.weight * float(state_probs @ overflow_expectation(n, cost.capacity, p)) / n
