"""The E4 code path (SL-21S; reused by SL-21): learning with hidden types and common states.

``requirements/paper.tex`` §6 (l.676–740) and ``requirements/A1_formal_statements.md`` §3, §E1–§E3.

**Identification** (Prop identify). Alternatives are indexed 0 (the outside option) to N, so
product j is alternative j + 1. An audited family is a list of (set, winner) observations; the
comparison graph G_A has an edge w(S) → i for every i ∈ (S ∪ {0}) \\ {w(S)}. ``closure`` is its
transitive closure (paths of length ≥ 1). The choice from S' is identified if some alternative
of S' ∪ {0} reaches every other one (``identified``). ``possible_winners`` is an independent
computation: under a strict ranking consistent with the observations (a linear extension of
the order the graph generates), b can be ranked first in S' ∪ {0} unless some other alternative
of S' ∪ {0} reaches b; the choice is identified when exactly one alternative can win.
``identified_by_enumeration`` checks both against every consistent ranking (small cases).

**Bound** (Cor design(b), corrected in A1 §E1). The number of distinct choice functions is
A(N) = Σ_k N!/(N − k)! = ⌊e · N!⌋ (ordered lists of the products ranked above the outside
option), and an adaptive design with sets of at most K products needs at least
⌈log A(N) / log(K + 1)⌉ audited sets: ``design_lower_bound`` is the smallest m with
(K + 1)^m ≥ A(N), in exact integer arithmetic. ``count_choice_functions`` enumerates rankings.

**Estimation** (Prop learning(a)). Each period draws a state; each audited cell receives n₀
requests and Y_{e,c,·} is the fraction ending with each alternative. q̂ = mean of Y over periods,
with Var q̂ = (1/h) {Var_Λ q + E_Λ[q (1 − q)] / n₀} (``variance_closed_form``).

**Plug-in mechanism** (A1 §3.3) over a finite menu of policies supported on the audited cells
(ι_c(Q) the weight of cell c): X̂_j, Γ̂ and the unbiased quadratic cost estimator Ĉ_j, Ŵ_b
maximised with ties to the lowest menu index, payments by eq. payment with X̂ (exact: the
allocation is a step function of one bid, ``envelope``). ``true_values`` gives the exact X_j, Γ
and E C_j of every menu policy. Everything is in totals (n requests per period).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

OUTSIDE = 0


# ------------------------------------------------------------- identification
def comparison_graph(observations: list[tuple[tuple[int, ...], int]], alternatives: int) -> np.ndarray:
    """Adjacency (alternatives × alternatives): an edge w → i for every i ∈ (S ∪ {0}) \\ {w}.
    Sets hold alternative indices (products are 1..N)."""
    adj = np.zeros((alternatives, alternatives), dtype=bool)
    for shown, winner in observations:
        members = set(shown) | {OUTSIDE}
        if winner not in members:
            raise ValueError("the winner %d is not in %s ∪ {0}" % (winner, shown))
        for i in members - {winner}:
            adj[winner, i] = True
    return adj


def closure(adj: np.ndarray) -> np.ndarray:
    """Transitive closure: reach[a, b] if a path of length ≥ 1 leads from a to b."""
    reach = adj.copy()
    for k in range(len(adj)):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def _members(target: tuple[int, ...]) -> list[int]:
    return sorted(set(target) | {OUTSIDE})


def identified(reach: np.ndarray, target: tuple[int, ...]) -> int | None:
    """Prop identify: the alternative of S' ∪ {0} that reaches every other one, or None."""
    members = _members(target)
    for a in members:
        if all(reach[a, b] for b in members if b != a):
            return a
    return None


def possible_winners(reach: np.ndarray, target: tuple[int, ...]) -> list[int]:
    """Alternatives of S' ∪ {0} that some consistent ranking puts first (none reaches them)."""
    members = _members(target)
    return [b for b in members if not any(reach[c, b] for c in members if c != b)]


def identified_shares(reach: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Vectorised Prop identify over target sets (T, k) of products: the identified alternative
    per target, or −1."""
    T, k = targets.shape
    members = np.concatenate([np.zeros((T, 1), dtype=int), targets], axis=1)  # (T, k + 1)
    sub = reach[members[:, :, None], members[:, None, :]]  # (T, k+1, k+1)
    eye = np.eye(k + 1, dtype=bool)[None]
    beats_all = np.all(sub | eye, axis=2)  # a reaches every other member
    has = beats_all.any(axis=1)
    first = np.argmax(beats_all, axis=1)
    return np.where(has, members[np.arange(T), first], -1)


def consistent_rankings(observations: list[tuple[tuple[int, ...], int]], alternatives: int):
    """Every strict ranking (best first) of the alternatives consistent with the observations."""
    for ranking in itertools.permutations(range(alternatives)):
        position = {a: p for p, a in enumerate(ranking)}
        if all(position[w] < position[i] for shown, w in observations for i in set(shown) | {OUTSIDE} if i != w):
            yield ranking


def choice(ranking: tuple[int, ...], target: tuple[int, ...]) -> int:
    members = set(target) | {OUTSIDE}
    return next(a for a in ranking if a in members)


def identified_by_enumeration(observations, alternatives: int, target: tuple[int, ...]) -> int | None:
    choices = {choice(r, target) for r in consistent_rankings(observations, alternatives)}
    return choices.pop() if len(choices) == 1 else None


# ---------------------------------------------------------------------- bound
def choice_functions(n: int) -> int:
    """A(N) = Σ_{k=0}^{N} N!/(N − k)!, exactly."""
    return sum(math.factorial(n) // math.factorial(n - k) for k in range(n + 1))


def design_lower_bound(n: int, k: int) -> int:
    """⌈log A(N) / log(K + 1)⌉: the smallest m with (K + 1)^m ≥ A(N) (A1 §3.2(b), §E1)."""
    a, m = choice_functions(n), 0
    while (k + 1) ** m < a:
        m += 1
    return m


def paper_lower_bound(n: int, k: int) -> int:
    """The paper's (incorrect) ⌈log (N + 1)! / log(K + 1)⌉, for comparison."""
    a, m = math.factorial(n + 1), 0
    while (k + 1) ** m < a:
        m += 1
    return m


def count_choice_functions(n: int, k: int) -> int:
    """Distinct choice functions on every subset of 1..k of N products, over all strict
    rankings of the N products and the outside option (enumeration)."""
    subsets = [s for size in range(1, k + 1) for s in itertools.combinations(range(1, n + 1), size)]
    seen = set()
    for ranking in itertools.permutations(range(n + 1)):
        seen.add(tuple(choice(ranking, s) for s in subsets))
    return len(seen)


# --------------------------------------------------------------------- choices
def logit(scores: np.ndarray, sets: np.ndarray, sigma: float, outside: float) -> np.ndarray:
    """Choice probabilities (S, cells, k + 1) over (outside, the set's products in order) for
    scores (S, N) of products 1..N and sets (cells, k) of product indices 1..N."""
    v = scores[:, sets - 1]  # (S, cells, k)
    logits = np.concatenate([np.full(v.shape[:2] + (1,), outside), v], axis=2) / sigma
    logits -= logits.max(axis=2, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=2, keepdims=True)


def deterministic(scores: np.ndarray, sets: np.ndarray, outside: float) -> np.ndarray:
    """The σ → 0 winners (S, cells) as alternative indices (0 or the product index)."""
    v = scores[:, sets - 1]
    full = np.concatenate([np.full(v.shape[:2] + (1,), outside), v], axis=2)
    pos = np.argmax(full, axis=2)  # (S, cells)
    ordered = np.sort(full, axis=2)
    if np.any(ordered[:, :, -1] == ordered[:, :, -2]):
        raise ValueError("a tie in the deterministic limit (Ass. deterministic excludes it)")
    products = sets[np.arange(len(sets))[None, :], np.maximum(pos - 1, 0)]
    return np.where(pos == 0, OUTSIDE, products)


# -------------------------------------------------------------------- designs
def leader_inclusive(rng: np.random.Generator, means: np.ndarray, count: int, k: int) -> np.ndarray:
    """Every set holds the product with the highest mean score and k − 1 others drawn uniformly."""
    n = len(means)
    leader = int(np.argmax(means)) + 1
    others = [j for j in range(1, n + 1) if j != leader]
    return np.array([sorted([leader] + list(rng.choice(others, k - 1, replace=False))) for _ in range(count)])


def competitor_varying(rng: np.random.Generator, means: np.ndarray, count: int, k: int, top: int = 5) -> np.ndarray:
    """Every pair of the top products by mean score meets in a set whose other members are
    drawn from outside the top, so the pair's winner is one of them and G_A joins them
    (when the top products beat the rest); the remaining sets are drawn uniformly."""
    n = len(means)
    order = list(np.argsort(-means) + 1)
    best, rest = order[:top], order[top:]
    sets = [sorted(list(pair) + list(rng.choice(rest, k - 2, replace=False)))
            for pair in itertools.combinations(sorted(best), 2)]
    if len(sets) > count:
        raise ValueError("too few audited sets for the top pairs")
    while len(sets) < count:
        sets.append(sorted(rng.choice(np.arange(1, n + 1), k, replace=False)))
    return np.array(sets)[rng.permutation(count)]


def matched_random(rng: np.random.Generator, design: np.ndarray, swaps: int | None = None) -> np.ndarray:
    """A random design with the same per-product inclusion counts as ``design``: random
    swaps of products between sets that keep every set's products distinct."""
    sets = [list(s) for s in design]
    count, k = design.shape
    for _ in range(swaps or 50 * count * k):
        a, b = rng.integers(0, count, 2)
        i, j = rng.integers(0, k, 2)
        x, y = sets[a][i], sets[b][j]
        if a == b or x == y or y in sets[a] or x in sets[b]:
            continue
        sets[a][i], sets[b][j] = y, x
    return np.array([sorted(s) for s in sets])


def inclusion(design: np.ndarray, n: int) -> np.ndarray:
    """Per product (1..N): the share of audited sets that contain it."""
    counts = np.zeros(n)
    for s in design:
        counts[np.asarray(s) - 1] += 1
    return counts / len(design)


# ----------------------------------------------------------------- estimation
def audit(rng: np.random.Generator, probs: np.ndarray, states: np.ndarray, n0: int) -> np.ndarray:
    """Y (periods, cells, k + 1): fractions of the n₀ audit requests per cell and period, with
    the period's state ``states[e]`` and probabilities ``probs`` (S, cells, k + 1)."""
    return rng.multinomial(n0, probs[states]) / n0


def variance_closed_form(q_state: np.ndarray, state_probs: np.ndarray, h: int, n0: int) -> np.ndarray:
    """Prop learning(a): (1/h) {Var_Λ q + E_Λ[q (1 − q)] / n₀}, elementwise over q's trailing axes."""
    mean = np.tensordot(state_probs, q_state, axes=1)
    var = np.tensordot(state_probs, (q_state - mean) ** 2, axes=1)
    within = np.tensordot(state_probs, q_state * (1 - q_state), axes=1)
    return (var + within / n0) / h


# ----------------------------------------------------------------- mechanism
@dataclass
class Mechanism:
    """A menu of policies over the audited cells and the market's primitives."""
    iota: np.ndarray  # (policies, cells): ι_c(Q)
    sets: np.ndarray  # (cells, k): product indices 1..N
    products: int  # N
    advertisers: np.ndarray  # product indices 1..N
    n: int
    g: float
    lam: float
    capacity: float  # B

    def per_product(self, y: np.ndarray) -> np.ndarray:
        """Map (..., cells, k + 1) alternative fractions to (..., cells, N) product fractions."""
        out = np.zeros(y.shape[:-1] + (self.products,))
        idx = np.broadcast_to(self.sets - 1, y.shape[:-1] + (self.sets.shape[1],))
        np.put_along_axis(out, idx, y[..., 1:], axis=-1)
        return out


def true_values(m: Mechanism, q_state: np.ndarray, state_probs: np.ndarray) -> dict:
    """Exact X_j (policies, N), Γ and Σ_j E C_j (policies) of every menu policy."""
    qp = m.per_product(q_state)  # (S, cells, N)
    p = np.einsum("pc,scj->spj", m.iota, qp)  # (S, policies, N)
    x = m.n * np.tensordot(state_probs, p, axes=1)  # (policies, N)
    gamma = m.g * x.sum(axis=1)
    cost = m.lam / (2 * m.capacity) * (x + m.n * (m.n - 1) * np.tensordot(state_probs, p * p, axes=1))
    return {"X": x, "contribution": gamma, "cost": cost.sum(axis=1)}


def estimates(m: Mechanism, y: np.ndarray, n0: int) -> dict:
    """Plug-in estimates after the periods of y (periods, cells, k + 1): X̂ (policies, N), Γ̂ and
    Σ_j Ĉ_j (policies), with the unbiased within-period second moment of A1 §3.3."""
    yp = m.per_product(y)  # (E, cells, N)
    h = len(y)
    q_hat = yp.mean(axis=0)  # (cells, N)
    x = m.n * m.iota @ q_hat
    gamma = m.g * x.sum(axis=1)
    p = np.einsum("pc,ecj->epj", m.iota, yp)  # (E, policies, N): Σ_c ι_c Y
    own = np.einsum("pc,ecj->epj", m.iota ** 2, yp * yp)
    unbiased = np.einsum("pc,ecj->epj", m.iota ** 2, yp * (n0 * yp - 1) / (n0 - 1))
    second = (p * p - own + unbiased).sum(axis=0) / h  # (policies, N)
    cost = m.lam / (2 * m.capacity) * (x + m.n * (m.n - 1) * second)
    return {"X": x, "contribution": gamma, "cost": cost.sum(axis=1)}


def prefix_welfare(m: Mechanism, y: np.ndarray, n0: int, bids: np.ndarray) -> np.ndarray:
    """Ŵ_b (periods, policies) after each prefix of periods 1..e: ``estimates`` of y[:e] for
    every e at once, through cumulative sums."""
    yp = m.per_product(y)
    e = np.arange(1, len(y) + 1)[:, None, None]
    x = m.n * np.einsum("pc,ecj->epj", m.iota, np.cumsum(yp, axis=0)) / e  # (E, policies, N)
    p = np.einsum("pc,ecj->epj", m.iota, yp)
    own = np.einsum("pc,ecj->epj", m.iota ** 2, yp * yp)
    unbiased = np.einsum("pc,ecj->epj", m.iota ** 2, yp * (n0 * yp - 1) / (n0 - 1))
    second = np.cumsum(p * p - own + unbiased, axis=0) / e
    cost = (m.lam / (2 * m.capacity) * (x + m.n * (m.n - 1) * second)).sum(axis=2)
    return x[:, :, m.advertisers - 1] @ bids + m.g * x.sum(axis=2) - cost


def welfare(values: dict, ads: np.ndarray, bids: np.ndarray) -> np.ndarray:
    """W_b (policies) = Σ_j b_j X_j + Γ − Σ_j C_j."""
    return values["X"][:, ads - 1] @ bids + values["contribution"] - values["cost"]


def choose(est: dict, ads: np.ndarray, bids: np.ndarray) -> int:
    """The plug-in policy: argmax Ŵ_b, ties to the lowest menu index."""
    return int(np.argmax(welfare(est, ads, bids)))


def _right_max(candidates: np.ndarray, slopes: np.ndarray) -> int:
    """Among lines equal at a point, the one that is largest just to its right: the highest
    slope, then the lowest index."""
    steepest = candidates[slopes[candidates] == slopes[candidates].max()]
    return int(steepest.min())


def envelope(slopes: np.ndarray, intercepts: np.ndarray, top: float) -> list[tuple[float, float, int]]:
    """The upper envelope of intercepts + z slopes over z ∈ [0, top] as pieces (start, end,
    index); each piece's line is the maximum on its interior."""
    pieces, z = [], 0.0
    current = _right_max(np.flatnonzero(intercepts == intercepts.max()), slopes)
    while z < top:
        values = intercepts + z * slopes
        later = slopes > slopes[current]
        if not later.any():
            pieces.append((z, top, current))
            break
        cross = z + (values[current] - values) / np.where(later, slopes - slopes[current], 1.0)
        cross = np.where(later, cross, np.inf)
        nxt = float(cross.min())
        if nxt >= top:
            pieces.append((z, top, current))
            break
        pieces.append((z, nxt, current))
        z = nxt
        current = _right_max(np.flatnonzero(later & (cross <= nxt)), slopes)
    return pieces


def misreport_gains(m: Mechanism, est: dict, truth: dict, theta: np.ndarray, reports: np.ndarray) -> np.ndarray:
    """For each advertiser: max over reports of u_j(r) − u_j(θ_j), with u_j(r) = θ_j X_j(Q̂(r, θ_{−j}))
    − P_j(r, θ_{−j}), true X and payments by eq. payment with X̂ (others report truthfully)."""
    gains = np.zeros(len(m.advertisers))
    grid = np.unique(np.concatenate([reports, theta]))
    for k, j in enumerate(m.advertisers):
        others = theta.copy()
        others[k] = 0.0
        slopes = est["X"][:, j - 1]
        intercepts = welfare(est, m.advertisers, others)
        pieces = envelope(slopes, intercepts, float(grid.max()))
        starts = np.array([p[0] for p in pieces])
        index = np.array([p[2] for p in pieces])
        lengths = np.array([p[1] - p[0] for p in pieces])
        cum = np.concatenate([[0.0], np.cumsum(slopes[index] * lengths)])
        piece = np.clip(np.searchsorted(starts, grid, side="right") - 1, 0, len(pieces) - 1)
        integral = cum[piece] + slopes[index[piece]] * (grid - starts[piece])
        chosen = np.array([int(np.argmax(intercepts + r * slopes)) for r in grid])
        payment = grid * slopes[chosen] - integral
        utility = theta[k] * truth["X"][chosen, j - 1] - payment
        gains[k] = float(utility.max() - utility[np.searchsorted(grid, theta[k])])
    return gains


def epsilon_1(n_products: int, cells: int, h: int, rho: float) -> float:
    """ε₁ = √(log(2 N |S| / ϱ) / (2h)) (Prop learning(b))."""
    return math.sqrt(math.log(2 * n_products * cells / rho) / (2 * h))


__all__ = ["Mechanism", "audit", "choice_functions", "choose", "closure", "comparison_graph", "competitor_varying",
           "consistent_rankings", "count_choice_functions", "design_lower_bound", "deterministic", "envelope",
           "epsilon_1", "estimates", "identified", "identified_by_enumeration", "identified_shares", "inclusion",
           "leader_inclusive", "logit", "matched_random", "misreport_gains", "paper_lower_bound", "possible_winners",
           "prefix_welfare", "true_values", "variance_closed_form", "welfare"]
