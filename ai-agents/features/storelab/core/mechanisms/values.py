"""Value distributions and (ironed) virtual values for the profit rule (SL-16, A16).

``prop:profit`` (l.475) ranks by φ_j(θ) = θ − (1 − F_j(θ)) / f_j(θ), which must be
nondecreasing. Uniform values are regular. Lognormal values are not once s is large
(about s ≥ 2), so by the registered rule (A16, claims register
``profit_rule_regularity``) the profit rule uses Myerson's ironed virtual value:

    H(q) = ∫₀^q φ(F⁻¹(r)) dr = −F⁻¹(q) (1 − q),   φ̄(F⁻¹(q)) = d/dq conv H(q),

where conv H is the greatest convex minorant of H on [0, 1]. Each ironing interval
[q₁, q₂] is located on a fine grid and then solved exactly from the tangency
conditions H'(q₁) = H'(q₂) = (H(q₂) − H(q₁)) / (q₂ − q₁); on it φ̄ is that constant,
elsewhere φ̄ = φ.
"""
from __future__ import annotations

import math
from functools import cached_property
from typing import Literal

import numpy as np
from pydantic import PositiveFloat, model_validator
from scipy import integrate, optimize, stats

from ..config import Section

GRID = 20001
TANGENCY_TOL = 1e-10
HULL_GAP = 1e-6  # relative to max |H|; smaller hull gaps are rounding where q bunches near 1


class ValueDistribution(Section):
    """F_j: ``uniform`` on [0, upper] or ``lognormal`` with log θ ~ N(mu, s²)."""
    kind: Literal["uniform", "lognormal"]
    upper: PositiveFloat | None = None
    mu: float | None = None
    s: PositiveFloat | None = None

    @model_validator(mode="after")
    def _parameters(self) -> ValueDistribution:
        if (self.kind == "uniform") != (self.upper is not None) or (self.kind == "lognormal") != (
                self.mu is not None and self.s is not None):
            raise ValueError("uniform needs upper; lognormal needs mu and s")
        return self

    @property
    def _dist(self):
        if self.kind == "uniform":
            return stats.uniform(0.0, self.upper)
        return stats.lognorm(self.s, scale=math.exp(self.mu))

    def cdf(self, theta: float) -> float:
        return float(self._dist.cdf(theta))

    def ppf(self, q: float) -> float:
        return float(self._dist.ppf(q))

    def sample(self, rng: np.random.Generator, size: int | None = None):
        if self.kind == "uniform":
            return rng.uniform(0.0, self.upper, size)
        return rng.lognormal(self.mu, self.s, size)

    def virtual_value(self, theta: float) -> float:
        """φ(θ) = θ − (1 − F(θ)) / f(θ) (−∞ where f = 0 below the support's top)."""
        if self.kind == "uniform":
            return 2.0 * theta - self.upper
        if theta <= 0:
            return -math.inf
        z = (math.log(theta) - self.mu) / self.s
        # (1 − F) / f = θ s · Φ(−z) / φ(z), in log form; it overflows to −∞ only far below the support's bulk
        log_ratio = stats.norm.logsf(z) - stats.norm.logpdf(z)
        return -math.inf if log_ratio > 700 else theta - theta * self.s * math.exp(log_ratio)

    def H(self, q: float) -> float:
        return -self.ppf(q) * (1.0 - q)

    @cached_property
    def ironing(self) -> list[tuple[float, float, float]]:
        """Ironing intervals as (θ₁, θ₂, φ̄) (empty when φ is nondecreasing)."""
        if self.kind == "uniform":
            return []
        z = np.linspace(-9.0, 7.0, GRID)  # q = Φ(z); beyond z = 7, 1 − q is below float resolution of q
        q = stats.norm.cdf(z)
        theta = np.exp(self.mu + self.s * z)
        h = -theta * stats.norm.sf(z)
        points = np.c_[np.r_[0.0, q, 1.0], np.r_[0.0, h, 0.0]]
        hull = _lower_hull(points)
        out = []
        for a, b in zip(hull, hull[1:]):
            if b - a < 2:
                continue  # consecutive grid points: the hull follows H
            (x1, y1), (x2, y2) = points[a], points[b]
            inside = points[a + 1:b]
            gap = inside[:, 1] - (y1 + (y2 - y1) * (inside[:, 0] - x1) / (x2 - x1))
            if gap.max() > HULL_GAP * max(1.0, np.abs(h).max()):  # H materially above its hull: ironing
                out.append(self._tangency(x1, x2))
        return out

    def _tangency(self, q1: float, q2: float) -> tuple[float, float, float]:
        """Solve φ(θ₁) = φ(θ₂) = (H(q₂) − H(q₁)) / (q₂ − q₁) from the grid guess."""
        def equations(x):
            a, b = x
            qa, qb = stats.norm.cdf(a), stats.norm.cdf(b)
            ta, tb = math.exp(self.mu + self.s * a), math.exp(self.mu + self.s * b)
            slope = (-tb * stats.norm.sf(b) + ta * stats.norm.sf(a)) / (qb - qa)
            return [self.virtual_value(ta) - slope, self.virtual_value(tb) - slope]

        z0 = np.array([stats.norm.ppf(min(max(q1, 1e-300), 1 - 1e-16)), stats.norm.ppf(min(max(q2, 1e-300), 1 - 1e-16))])
        # the grid locates each end to within one step (0.0008 in z); search near it
        solution = optimize.least_squares(equations, z0, bounds=(z0 - 0.05, z0 + 0.05), xtol=1e-15, ftol=1e-15,
                                          gtol=1e-15, method="trf")
        residual = equations(solution.x)
        scale = max(1.0, math.exp(self.mu + self.s * max(solution.x)))  # φ grows like θ
        if not all(map(math.isfinite, residual)) or max(map(abs, residual)) > TANGENCY_TOL * scale or abs(
                solution.x[1] - solution.x[0]) < 1e-6:
            raise RuntimeError("ironing: tangency not solved for %s (residual %s)" % (self, equations(solution.x)))
        a, b = sorted(solution.x)
        ta, tb = math.exp(self.mu + self.s * a), math.exp(self.mu + self.s * b)
        return ta, tb, self.virtual_value(ta)

    def ironed(self, theta: float) -> float:
        """φ̄(θ): the ironed virtual value (= φ where φ is already monotone)."""
        for a, b, value in self.ironing:
            if a <= theta <= b:
                return value
        return self.virtual_value(theta)

    def ironed_inverse(self, c: float) -> float:
        """inf{θ ≥ 0 : φ̄(θ) ≥ c}, the bid at which the profit coefficient reaches c."""
        if self.ironed(0.0) >= c:
            return 0.0
        for a, b, value in self.ironing:
            if math.isclose(value, c, rel_tol=0.0, abs_tol=1e-15):
                return a
        high = max(1.0, self.upper or 0.0)
        while self.ironed(high) < c:
            high *= 2.0
            if high > 1e12:
                raise ValueError("no bid reaches virtual value %g" % c)
        low = 0.0
        for a, b, value in self.ironing:  # a flat part at value < c lies wholly below the answer
            if value < c:
                low = max(low, b)
        return optimize.brentq(lambda t: self.ironed(t) - c, low, high, xtol=1e-14, rtol=4 * np.finfo(float).eps,
                               maxiter=500)

    def ironed_integral(self, low: float, high: float) -> float:
        """∫_low^high φ̄(θ) dθ (adaptive quadrature, split at the ironing endpoints)."""
        if high <= low:
            return 0.0
        if self.kind == "uniform":
            return (high ** 2 - low ** 2) - self.upper * (high - low)
        cuts = sorted({low, high, *[t for a, b, _ in self.ironing for t in (a, b) if low < t < high]})
        total = 0.0
        for a, b in zip(cuts, cuts[1:]):
            flat = [v for x, y, v in self.ironing if x <= a and b <= y]
            if flat:
                total += flat[0] * (b - a)
            else:
                total += integrate.quad(self.virtual_value, a, b, epsabs=0.0, epsrel=1e-12, limit=200)[0]
        return total


def _lower_hull(points: np.ndarray) -> list[int]:
    """Indices of the lower convex hull of points sorted by x (monotone chain)."""
    hull: list[int] = []
    for i in range(len(points)):
        while len(hull) >= 2:
            (x1, y1), (x2, y2), (x3, y3) = points[hull[-2]], points[hull[-1]], points[i]
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) < 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull
