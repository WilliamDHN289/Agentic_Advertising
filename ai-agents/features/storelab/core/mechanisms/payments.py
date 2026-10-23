"""Payments (SL-16): the externality form of Prop ic(c) and the integral eq. payment.

**Externality** (surplus rules M3, set-blind M3, M4; l.452): with W_{-j}(Q) =
Σ_{i≠j} b_i X_i(Q) + G(Q),

    P_j(b) = max_Q W_{-j}(Q) − W_{-j}(Q*(b)),

one extra solve per advertiser (b_j = 0) over the same feasible set.

**Integral** (eq. payment, l.442): P_j(b) = b_j X_j(b) − ∫₀^{b_j} X_j(z, b_{-j}) dz. Under
the profit rule (prop:profit) X_j(z) = X̃_j(φ̄_j(z)), where X̃_j(c) is the allocation
with coefficient c for j. ``profile`` finds the pieces of X̃_j in c on which it is
affine: constant between breakpoints when the program is an LP, affine between kinks
when it is a QP. Pieces are accepted only on X values: an interval is one piece when
X at its third points lies on the line through its ends (every tracked component:
X_j and its human part), within the solver's accuracy (``NOISE``) or 1e-9 of the
variation of X_j over the profile, whichever is larger. A jump is located at the intersection of the tangents of
the convex value function V(c) at the interval ends; it is accepted when V there
lies on both tangents (V convex and touching both lines is piecewise linear with one
kink) and X just left and right of it equals the end values. Otherwise the interval
is split (a breakpoint at an end of the interval, where the evaluated X is
the tie-broken value, is accepted the same way). The integral then sums the pieces in bid space: ∫ over a piece is
F_a Δz + slope (∫ φ̄ dz − c_a Δz), exact for constant and affine pieces (∫ φ̄ dz in
closed form for the surplus rule and uniform values, by quadrature otherwise).
Intervals narrower than ``MIN_WIDTH`` are accepted as they are and counted.

Payments by traffic type (A1 §E5) integrate X_j^H and X_j^A the same way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .allocation import Program
from .values import ValueDistribution

INITIAL_GRID = 9
NOISE = {"lp": 1e-12, "qp": 1e-12}  # per request: accuracy of X / n and V (simplex vertices, polished QP)
RELATIVE_TOL = 1e-9  # of the variation of X_j over the profile
MIN_WIDTH = 1e-9  # relative to the coefficient range
PROBE = 1e-6  # relative to the interval searched for a jump
FLOOR_SCALE = 10.0  # profit-rule coefficients are clipped at −FLOOR_SCALE (1 + market scale), see coefficient_floor


def externality_payments(program: Program, bids: np.ndarray) -> tuple[np.ndarray, object]:
    """P_j for every advertiser under a surplus rule, and the allocation at b."""
    if program.rule.objective != "surplus":
        raise ValueError("the externality form holds for the surplus rules")
    bids = np.asarray(bids, dtype=float)
    base = program.solve(bids)
    n = program.market.requests
    out = np.zeros(len(bids))
    for k, j in enumerate(program.market.advertisers):
        if bids[k] == 0.0:
            continue
        others = bids.copy()
        others[k] = 0.0
        best_without = program.solve(others).value  # max W_{-j}, per request
        at_b = base.value - bids[k] * base.X[j] / n  # W_{-j}(Q*(b)), per request
        out[k] = n * (best_without - at_b)
    return out, base


@dataclass
class Piece:
    low: float
    high: float
    at_low: np.ndarray  # X components (per request) at the low end of the piece
    at_high: np.ndarray


@dataclass
class Profile:
    """X̃_j(c) on [c_min, c_max] as affine pieces."""
    advertiser: str
    pieces: list[Piece]
    solves: int
    unresolved: list[tuple[float, float]] = field(default_factory=list)  # accepted at MIN_WIDTH

    def components_at(self, c: float) -> np.ndarray:
        for piece in self.pieces:
            if piece.low <= c <= piece.high:
                if piece.high == piece.low:
                    return piece.at_low
                w = (c - piece.low) / (piece.high - piece.low)
                return piece.at_low + w * (piece.at_high - piece.at_low)
        raise ValueError("coefficient %g outside the profile" % c)


class _Evaluator:
    def __init__(self, program: Program, coefficients: np.ndarray, k: int):
        self.program, self.coefficients, self.k = program, np.asarray(coefficients, dtype=float), k
        self.j = program.market.advertisers[k]
        self.solves = 0

    def __call__(self, c: float) -> tuple[float, np.ndarray]:
        coefficients = self.coefficients.copy()
        coefficients[self.k] = c
        a = self.program.solve(coefficients)
        self.solves += 1
        n = self.program.market.requests
        return a.value, np.array([a.X[self.j] / n, a.X_human[self.j] / n])


def profile(program: Program, coefficients: np.ndarray, k: int, low: float, high: float,
            points: list[float] | None = None) -> Profile:
    """Affine pieces of X̃_j on [low, high] (module docstring); ``points`` are added to
    the initial grid (the images φ̄(z) of a bid grid, where the profit rule's pieces lie)."""
    f = _Evaluator(program, coefficients, k)
    span = high - low
    min_width = MIN_WIDTH * span
    grid = sorted({*np.linspace(low, high, INITIAL_GRID), *[c for c in points or [] if low < c < high]})
    evaluated = [(c, *f(c)) for c in grid]
    variation = abs(evaluated[-1][2][0] - evaluated[0][2][0])
    tol = max(NOISE[program.kind], RELATIVE_TOL * variation)
    stack = list(zip(evaluated[:-1], evaluated[1:]))[::-1]
    pieces, unresolved = [], []
    while stack:
        (ca, va, fa), (cb, vb, fb) = stack.pop()
        if cb - ca <= min_width:
            pieces.append(Piece(ca, cb, fa, fb))
            unresolved.append((ca, cb))
            continue
        t1, t2 = ca + (cb - ca) / 3, ca + 2 * (cb - ca) / 3
        p1, p2 = (t1, *f(t1)), (t2, *f(t2))
        line = lambda t: fa + (fb - fa) * (t - ca) / (cb - ca)  # noqa: E731
        if np.abs(p1[2] - line(t1)).max() <= tol and np.abs(p2[2] - line(t2)).max() <= tol:
            pieces.append(Piece(ca, cb, fa, fb))
            continue
        jump = _single_jump(f, (ca, va, fa), (cb, vb, fb), PROBE * (cb - ca), tol, NOISE[program.kind])
        if jump is not None:
            pieces += [Piece(ca, jump, fa, fa), Piece(jump, cb, fb, fb)]
            continue
        stack += [(p2, (cb, vb, fb)), (p1, p2), ((ca, va, fa), p1)]
    pieces.sort(key=lambda p: p.low)
    return Profile(program.market.advertisers[k], pieces, f.solves, unresolved)


def _single_jump(f, a, b, probe: float, tol: float, value_tol: float) -> float | None:
    """A single breakpoint of a piecewise-constant X in [a, b] (possibly at an end, where
    the evaluated value is the tie-broken one), or None."""
    (ca, va, fa), (cb, vb, fb) = a, b
    if abs(fa[0] - fb[0]) <= tol:
        return None
    c = (vb - va - fb[0] * cb + fa[0] * ca) / (fa[0] - fb[0])  # tangents of V at a and b meet here
    if not ca - probe <= c <= cb + probe:
        return None
    c = min(max(c, ca), cb)
    vc = f(c)[0] if ca < c < cb else (va if c == ca else vb)
    if abs(vc - (va + fa[0] * (c - ca))) > value_tol * max(1.0, abs(vc)) or abs(
            vc - (vb + fb[0] * (c - cb))) > value_tol * max(1.0, abs(vc)):
        return None
    if c - ca > probe and np.abs(f(c - probe)[1] - fa).max() > tol:
        return None
    if cb - c > probe and np.abs(f(c + probe)[1] - fb).max() > tol:
        return None
    return c


@dataclass
class IntegralPayment:
    advertiser: str
    bid: float
    payment: float  # total units
    payment_human: float
    payment_agent: float
    X: float  # X_j(b), total
    integral: float  # ∫₀^b X_j(z) dz, total
    profile: Profile


def coefficient_floor(program: Program) -> float:
    """The profit rule's lowest coefficient. φ̄_j(θ) → −∞ as θ → 0 for lognormal values,
    which no program can take, so the rule uses max(φ̄_j(b_j), floor) with floor =
    −10 (1 + max_j |φ̄_j(F_j⁻¹(0.99))| + max g + max λ), well below the market's
    marginal values. The clipped coefficient is still nondecreasing in the bid, so the
    rule stays incentive compatible and eq. payment is computed for it exactly; it
    differs from the unclipped rule only for bids whose φ̄ is below the floor, and
    ``floor_gap`` measures by how much (X̃_j at the floor against ten times it)."""
    market = program.market
    scale = max([abs(v.ironed(v.ppf(0.99))) for v in market.values.values()] or [0.0])
    scale += max(o.contribution for c in market.contexts for o in c.offers.values())
    scale += max([c.weight for c in market.costs.values()] or [0.0])
    return -FLOOR_SCALE * (1.0 + scale)


class BidMap:
    """z ↦ c: the identity (surplus rule) or φ̄_j clipped at the floor (profit rule)."""

    def __init__(self, values: ValueDistribution | None = None, floor: float = -math.inf):
        self.values, self.floor = values, floor

    def coefficient(self, z: float) -> float:
        return z if self.values is None else max(self.values.ironed(z), self.floor)

    def lowest(self, scale: float = 0.0) -> float:
        return 0.0 if self.values is None else self.coefficient(0.0)

    def upper_inverse(self, c: float) -> float:
        """sup{z ≥ 0 : coefficient(z) ≤ c}: the bids whose coefficient is at most c
        (on an ironing interval at value c, its upper end)."""
        if self.values is None:
            return max(c, 0.0)
        for _, high, value in self.values.ironing:
            if math.isclose(value, c, rel_tol=0.0, abs_tol=1e-12):
                return high
        return self.values.ironed_inverse(max(c, self.floor))

    def integral(self, low: float, high: float) -> float:
        if self.values is None:
            return (high * high - low * low) / 2
        return self.values.ironed_integral(low, high)


def integral_payment(program: Program, bids: np.ndarray, k: int, bid_map: BidMap, upper: float | None = None,
                     prof: Profile | None = None) -> IntegralPayment:
    """eq. payment for advertiser k at ``bids`` (the others' bids stay fixed); the
    profile covers bids up to ``upper`` (default the bid) so it can be reused with the
    same ``bid_map`` and others' bids."""
    bids = np.asarray(bids, dtype=float)
    coefs = coefficients(program, bids)
    upper = bids[k] if upper is None else upper
    low = bid_map.lowest()
    high = bid_map.coefficient(upper)
    if prof is None and high > low:
        points = [bid_map.coefficient(z) for z in np.linspace(0.0, upper, INITIAL_GRID)[1:]]
        prof = profile(program, coefs, k, low, high, points)
    integral = cumulative(prof, bid_map, bids[k]) if prof is not None else np.zeros(2)
    a = program.solve(coefs)
    j, n = program.market.advertisers[k], program.market.requests
    x, xh = a.X[j], a.X_human[j]
    pay = bids[k] * x - n * integral[0]
    pay_h = bids[k] * xh - n * integral[1]
    return IntegralPayment(j, float(bids[k]), float(pay), float(pay_h), float(pay - pay_h), float(x),
                           float(n * integral[0]), prof)


def floor_gap(program: Program, bids: np.ndarray, k: int) -> float:
    """|X̃_j(floor) − X̃_j(10 floor)| per request: how much clipping at the floor changes
    the allocation of bids below it (0 for the surplus rule)."""
    if program.rule.objective == "surplus":
        return 0.0
    f = _Evaluator(program, coefficients(program, bids), k)
    floor = coefficient_floor(program)
    return float(abs(f(floor)[1][0] - f(10.0 * floor)[1][0]))


def cumulative(prof: Profile, bid_map: BidMap, bid: float) -> np.ndarray:
    """∫₀^bid X̃(c(z)) dz per request (components), from the profile's pieces. The bids
    with c(z) ∈ (c_a, c_b] are (Z(c_a), Z(c_b)] with Z = ``upper_inverse``, so a flat
    stretch of φ̄ (ironing) belongs to the piece that ends at its value."""
    total = np.zeros(2)
    if bid <= 0.0:
        return total
    first = prof.pieces[0]
    z_first = min(bid_map.upper_inverse(first.low), bid)
    total += first.at_low * z_first  # bids whose coefficient is the profile's lowest (the floor)
    for piece in prof.pieces:
        za, zb = bid_map.upper_inverse(piece.low), min(bid_map.upper_inverse(piece.high), bid)
        if zb <= za:
            continue
        dz = zb - za
        if piece.high > piece.low:
            slope = (piece.at_high - piece.at_low) / (piece.high - piece.low)
            total += piece.at_low * dz + slope * (bid_map.integral(za, zb) - piece.low * dz)
        else:
            total += piece.at_low * dz
        if zb >= bid:
            break
    return total


def coefficients(program: Program, bids: np.ndarray) -> np.ndarray:
    """The rule's coefficient vector at bids b: b, or φ̄(b) clipped at the floor (profit rule)."""
    return np.array([bid_map(program, k).coefficient(float(b)) for k, b in enumerate(np.asarray(bids, dtype=float))])


def bid_map(program: Program, k: int) -> BidMap:
    if program.rule.objective == "surplus":
        return BidMap()
    j = program.market.advertisers[k]
    if j not in program.market.values:
        raise ValueError("the profit rule needs the value distribution of %s" % j)
    return BidMap(program.market.values[j], coefficient_floor(program))
