"""The E3 code path (SL-20S; reused by SL-20): organic design under correlated agent demand.

``requirements/paper.tex`` §5 (l.620–675) and ``requirements/A1_formal_statements.md`` §2:
an organic policy without sponsorship is a lottery w over candidate lists, drawn
independently for every request (or, for a *quota*, a fixed number n w_l of requests per
list). Given the common state Λ, a request yields an effective choice of j with probability
p_j(Λ) = Σ_l w_l q_j(l | Λ), and (Lemma load, l.373)

    E D_j = n E_Λ p_j,   Var D_j = n E_Λ[p_j (1 − p_j)] + n² Var_Λ(p_j)
                                   (within state)        (across states).

Under a quota the within-state part is Σ_l n w_l E_Λ[q_j(l)(1 − q_j(l))] and the across part
is unchanged. ``moments`` returns both parts and, independently, the total variance from the
exact distribution of D_j (a mixture over states of binomials, or of convolutions of binomials
for a quota); the identity is checked, not assumed.

An ``Environment`` holds the candidate lists with two sets of choice probabilities: ``q`` at
the agents' logit noise (SL-16 ``response``: humans by Ass. human, agents by eq. agent-logit)
and ``q_det``, the deterministic limit σ → 0 (Ass. deterministic: each type chooses its
highest-scored alternative; exact ties are split equally). ``contested`` marks the actions
whose deterministic winner changes with the state (Lemma contested), so
Θ_j(w) = Σ_l w_l contested[l, j] and Var_Λ(p_j) ≤ Θ_j² / 4 under the deterministic limit.

``separate`` is the construction of Prop separation (A1 §2.2), ``program`` the convex organic
design problem max Σ_Λ F_Λ Σ_j p_j(Λ) ν_j(Λ) − Σ_j E C_j(D_j) / n over lotteries (net value
ν = g + η u). It is solved with SL-16's pinned solvers: quadratic costs are exact (Lemma load)
and give a QP; overflow costs give LPs, either the limit cost λ (p − ω)⁺ of Prop threshold(b)
or, at finite n, SL-16's tangent outer approximation of E (Bin(n, p) − B)⁺, whose largest
error is known exactly. Outcomes are always evaluated with exact costs.

``two_leader`` and ``diversify`` build the environments of Prop threshold and Example
ex:diversify; the closed forms of A1 §2.3 are ``y_star_quadratic``, ``objective_quadratic``,
``y_star_overflow`` and ``limit_objective``.

Everything is per request unless named a load (D) or a total.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy import stats

from ..core.mechanisms import costs as cost_model
from ..core.mechanisms.market import (AgentModel, Context, Cost, Display, Market, Offer, OrganicList, State,
                                      response)
from ..core.mechanisms.solver import solve


@dataclass
class Environment:
    n: int
    products: list[str]
    lists: list[tuple[str, ...]]
    state_probs: np.ndarray  # (S,)
    q: np.ndarray  # (S, L, J): pooled choice probabilities at the agents' noise
    q_det: np.ndarray  # (S, L, J): the deterministic limit
    agent_det: np.ndarray  # (S, L, J): Σ_m π_m of the agents' deterministic choices (no α, no humans)
    contested: np.ndarray  # (L, J): α Σ_m π_m 1{list l is j-contested for model m}
    contribution: np.ndarray  # (S, J): g(j, Λ), or the net value ν in Prop threshold
    match: np.ndarray  # (S, J): u(j, Λ)

    @property
    def index(self) -> dict[str, int]:
        return {j: i for i, j in enumerate(self.products)}

    def list_index(self, displayed: tuple[str, ...]) -> int:
        return self.lists.index(tuple(displayed))


def _det_choice(scores: np.ndarray) -> np.ndarray:
    """Deterministic choice among (outside, displayed...): the maximum, ties split equally."""
    best = scores == scores.max()
    return best / best.sum()


def environment(market: Market, lists: list[tuple[str, ...]], match: np.ndarray | None = None,
                contribution: np.ndarray | None = None) -> Environment:
    """The candidate lists of a one-context market (unlabelled, organic). ``match`` and
    ``contribution`` (S, J) override the market's state-independent u and g."""
    if len(market.contexts) != 1:
        raise ValueError("E3 environments have one context")
    context: Context = market.contexts[0]
    products = sorted(context.offers)
    index = {j: i for i, j in enumerate(products)}
    states = market.states
    probs = np.array([s.probability for s in states])
    S, L, J = len(states), len(lists), len(products)
    q, q_det, agent_det, contested = np.zeros((S, L, J)), np.zeros((S, L, J)), np.zeros((S, L, J)), np.zeros((L, J))
    alpha = context.agent_share
    for li, shown in enumerate(lists):
        cols = [index[j] for j in shown]
        r = response(market, context, tuple(shown), (False,) * len(shown))
        q[:, li, cols] = r["pooled"]
        det = np.zeros((S, len(shown)))
        for m in market.models:
            share = context.model_shares.get(m.key, 0.0)
            if share == 0.0:
                continue
            if m.position_effects and any(m.position_effects[:len(shown)]):
                raise ValueError("the deterministic limit needs β = 0 (Ass. deterministic)")
            winners = np.array([_det_choice(np.array([m.outside[context.context_id]]
                                                     + [m.scores[s.state_id][context.context_id][j] for j in shown]))
                                for s in states])  # (S, 1 + len(shown))
            det += share * winners[:, 1:]
            varies = np.ptp(winners[:, 1:], axis=0) > 0
            contested[li, cols] += alpha * share * varies
        q_det[:, li, cols] = (1 - alpha) * r["human"][None, :] + alpha * det
        agent_det[:, li, cols] = det
    if match is None:
        match = np.tile([context.offers[j].match for j in products], (S, 1))
    if contribution is None:
        contribution = np.tile([context.offers[j].contribution for j in products], (S, 1))
    return Environment(n=market.requests, products=products, lists=[tuple(x) for x in lists], state_probs=probs,
                       q=q, q_det=q_det, agent_det=agent_det, contested=contested, contribution=np.asarray(contribution, dtype=float),
                       match=np.asarray(match, dtype=float))


# ------------------------------------------------------------------ outcomes
def p_state(env: Environment, w: np.ndarray, det: bool = False) -> np.ndarray:
    """(S, J): p_j(Λ) under the lottery w."""
    return np.einsum("l,slj->sj", np.asarray(w, dtype=float), env.q_det if det else env.q)


def quota_counts(env: Environment, w: np.ndarray) -> np.ndarray:
    counts = env.n * np.asarray(w, dtype=float)
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-9:
        raise ValueError("a quota needs n w_l to be whole numbers")
    return rounded.astype(int)


def load_pmf(env: Environment, w: np.ndarray, j: int, det: bool = False, quota: bool = False) -> np.ndarray:
    """(S, n + 1): the distribution of D_j given each state."""
    q = env.q_det if det else env.q
    d = np.arange(env.n + 1)
    out = np.zeros((len(env.state_probs), env.n + 1))
    for s in range(len(env.state_probs)):
        if not quota:
            out[s] = stats.binom.pmf(d, env.n, min(max(float(np.asarray(w) @ q[s, :, j]), 0.0), 1.0))
            continue
        pmf = np.array([1.0])
        for count, prob in zip(quota_counts(env, w), q[s, :, j]):
            if count:
                pmf = np.convolve(pmf, stats.binom.pmf(np.arange(count + 1), count, min(max(prob, 0.0), 1.0)))
        out[s] = pmf
    return out


def moments(env: Environment, w: np.ndarray, det: bool = False, quota: bool = False) -> dict:
    """Per product: E D, the within- and across-state parts of Var D (Lemma load), the total
    variance from the exact distribution, Var_Λ p and, under the deterministic limit, Θ."""
    w = np.asarray(w, dtype=float)
    probs, q = env.state_probs, env.q_det if det else env.q
    p = p_state(env, w, det)
    mean_p = probs @ p
    var_p = probs @ (p - mean_p) ** 2
    if quota:
        counts = quota_counts(env, w)
        within = probs @ np.einsum("l,slj->sj", counts, q * (1 - q))
    else:
        within = env.n * (probs @ (p * (1 - p)))
    d = np.arange(env.n + 1, dtype=float)
    total, pmfs = np.zeros(len(env.products)), []
    for j in range(len(env.products)):
        pmf = probs @ load_pmf(env, w, j, det, quota)
        pmfs.append(pmf)
        mean = pmf @ d
        total[j] = pmf @ (d - mean) ** 2
    return {"mean": env.n * mean_p, "within": within, "across": env.n ** 2 * var_p, "total": total,
            "var_p": var_p, "theta": w @ env.contested, "p_state": p, "pmf": np.array(pmfs)}


def expected_cost(pmf: np.ndarray, cost: Cost) -> float:
    """E C(D) from the distribution of D (total units)."""
    d = np.arange(len(pmf), dtype=float)
    if cost.kind == "quadratic":
        c = cost.weight * d * d / (2 * cost.capacity)
    else:
        c = cost.weight * np.maximum(d - cost.capacity, 0.0)
    return float(pmf @ c)


def overflow_probability(pmf: np.ndarray, capacity: float) -> float:
    """Pr(D > B)."""
    return float(pmf[np.arange(len(pmf)) > capacity].sum())


def value(env: Environment, w: np.ndarray, det: bool = False) -> dict:
    """Γ / n and M / n."""
    p = p_state(env, w, det)
    return {"contribution": float(env.state_probs @ (p * env.contribution).sum(axis=1)),
            "match": float(env.state_probs @ (p * env.match).sum(axis=1))}


# ---------------------------------------------------------------- separation
def separate(env: Environment, w: np.ndarray, anchors: dict[str, tuple[str, ...]]) -> np.ndarray:
    """Prop separation: each list l is replaced by the lottery Σ_j ϖ_j(l) δ_{R†_j}, with
    ϖ_j(l) = Σ_m π_m Pr_Λ(w^m(l, Λ) = j) from the agents' deterministic choices (A1 §2.2).
    Every product with ϖ_j > 0 needs an anchor under which every agent type chooses it.
    The proposition assumes α ≡ 1; with humans the same lottery is used, and their
    choices follow the lists."""
    w = np.asarray(w, dtype=float)
    out = np.zeros(len(env.lists))
    for li, weight in enumerate(w):
        if weight == 0.0:
            continue
        varpi = env.state_probs @ env.agent_det[:, li, :]
        if varpi.sum() < 1 - 1e-12:
            raise ValueError("list %s: the outside option is chosen; Prop separation excludes it" % (env.lists[li],))
        for j, share in zip(env.products, varpi):
            if share == 0.0:
                continue
            if j not in anchors:
                raise ValueError("no anchor for %s" % j)
            a = env.list_index(anchors[j])
            if not np.all(env.agent_det[:, a, env.index[j]] == 1.0):
                raise ValueError("%s is not an anchor of %s" % (anchors[j], j))
            out[a] += weight * share
    return out


# ------------------------------------------------------------------- program
@dataclass
class Solution:
    w: np.ndarray
    solver_value: float  # the program's optimal value, per request
    kind: str  # "qp" | "lp"
    approximation_error: float = 0.0  # overflow tangents: largest possible cost error, per request


def program(env: Environment, costs: dict[str, Cost], mode: str) -> Solution:
    """max_w Σ_Λ F_Λ Σ_j p_j(Λ) ν_j(Λ) − Σ_j cost_j over lotteries w, with the deterministic
    choice probabilities and ν = ``env.contribution``. ``mode``: ``exact`` (quadratic costs),
    ``limit`` (overflow, λ (p − ω)⁺ with ω = B / n) or ``finite`` (overflow tangents)."""
    probs, q = env.state_probs, env.q_det
    S, L, _ = q.shape
    w = cp.Variable(L, nonneg=True)
    constraints = [cp.sum(w) == 1]
    linear = np.einsum("s,slj,sj->l", probs, q, env.contribution)
    objective = linear @ w
    kind, error = "lp", 0.0
    for j, cost in costs.items():
        if cost.weight == 0.0:
            continue
        p = q[:, :, env.index[j]] @ w  # (S,)
        if mode == "exact":
            if cost.kind != "quadratic":
                raise ValueError("mode exact needs quadratic costs")
            kind = "qp"
            scale = cost.weight / (2 * cost.capacity)
            objective -= scale * (probs @ p + (env.n - 1) * cp.sum(cp.multiply(probs, cp.square(p))))
        elif mode == "limit":
            t = cp.Variable(S, nonneg=True)
            constraints.append(t >= p - cost.capacity / env.n)
            objective -= cost.weight * (probs @ t)
        elif mode == "finite":
            tangents = cost_model.overflow_tangents(env.n, cost.capacity)
            e = cp.Variable(S)
            constraints += [e >= tangents.intercept[k] + tangents.slope[k] * p for k in range(len(tangents.points))]
            objective -= cost.weight * (probs @ e)
            error += cost.weight * tangents.max_error
        else:
            raise ValueError("unknown mode %s" % mode)
    problem = cp.Problem(cp.Maximize(objective), constraints)
    optimum = solve(problem, kind, "e3 %s" % mode)
    x = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
    return Solution(w=x / x.sum(), solver_value=optimum, kind=kind, approximation_error=error)


def exact_objective(env: Environment, w: np.ndarray, costs: dict[str, Cost]) -> float:
    """The per-request objective at w with exact expected costs (deterministic choices)."""
    p = p_state(env, w, det=True)
    out = float(env.state_probs @ (p * env.contribution).sum(axis=1))
    for j, cost in costs.items():
        out -= cost_model.expected_cost(cost, env.n, env.state_probs, p[:, env.index[j]])
    return out


def limit_cost_objective(env: Environment, w: np.ndarray, costs: dict[str, Cost]) -> float:
    """The limit objective f of Prop threshold(b) at any lottery w: overflow cost λ (p − ω)⁺."""
    p = p_state(env, w, det=True)
    out = float(env.state_probs @ (p * env.contribution).sum(axis=1))
    for j, cost in costs.items():
        out -= cost.weight * float(env.state_probs @ np.maximum(p[:, env.index[j]] - cost.capacity / env.n, 0.0))
    return out


# ------------------------------------------------------------- environments
def _one_model_market(n: int, products: list[str], scores: list[dict[str, float]], outside: float,
                      k: int) -> Market:
    """A one-context, α = 1, one-model market with equally likely states (the scores)."""
    offers = {j: Offer(contribution=0.0, match=0.0, organic=Display(request=0.0, conversion=0.0),
                       sponsored=Display(request=0.0, conversion=0.0)) for j in products}
    states = [State(state_id="s%d" % (i + 1), probability=1.0 / len(scores)) for i in range(len(scores))]
    model = AgentModel(key="m", sigma=1e-3, outside={"x": outside},
                       scores={st.state_id: {"x": sc} for st, sc in zip(states, scores)})
    context = Context(context_id="x", probability=1.0, offers=offers,
                      organic=[OrganicList(ranking=products, probability=1.0)], agent_share=1.0, model_shares={"m": 1.0})
    return Market(market_id="e3", requests=n, max_displayed=k, max_sponsored=0, advertisers=[], contexts=[context],
                  states=states, position_weights=[1.0] * k, models=[model])


TWO_LEADER_LISTS = [("1", "2"), ("1", "3"), ("2", "3")]


def two_leader(n: int, delta: float, nu_minus: float = 0.0) -> Environment:
    """Prop threshold's two-leader environment (A1 §2.3): one context, α ≡ 1, one model,
    K = 2, lists of two of {1, 2, 3}; v1 > v2 > v3 > v0 in Λ1 and v2 > v1 > v3 > v0 in Λ2;
    the leader a state favours has net value ν₋ + Δ, the other ν₋ (product 3 is never chosen)."""
    market = _one_model_market(n, ["1", "2", "3"], [{"1": 3.0, "2": 2.0, "3": 1.0}, {"1": 2.0, "2": 3.0, "3": 1.0}],
                               0.0, 2)
    nu = np.array([[nu_minus + delta, nu_minus, 0.0], [nu_minus, nu_minus + delta, 0.0]])
    return environment(market, TWO_LEADER_LISTS, match=np.zeros_like(nu), contribution=nu)


def two_leader_costs(kind: str, n: int, omega: float, lam: float) -> dict[str, Cost]:
    """Products 1 and 2 share the cost; product 3 has none. B = ω n."""
    return {j: Cost(kind=kind, weight=lam, capacity=omega * n) for j in ("1", "2")}


def head_to_head_share(env: Environment, w: np.ndarray) -> float:
    """y: the probability of showing {1, 2}."""
    return float(w[env.list_index(("1", "2"))])


def y_star_quadratic(n: int, omega: float, lam: float, delta: float) -> float:
    """Prop threshold(a): y* = min{1, Δ B / (λ (n − 1))}, B = ω n."""
    return min(1.0, delta * omega * n / (lam * (n - 1)))


def objective_quadratic(y: float, n: int, omega: float, lam: float, delta: float, nu_minus: float = 0.0) -> float:
    """J(y) / n = ν₋ + Δ (1 + y) / 2 − λ / (4B) [n + 1 + (n − 1) y²] (A1 §2.3(a))."""
    return nu_minus + delta * (1 + y) / 2 - lam / (4 * omega * n) * (n + 1 + (n - 1) * y * y)


def y_star_overflow(omega: float, lam: float, delta: float) -> float:
    """Prop threshold(b), Δ > 0 and Δ ≠ λ: 1 if Δ > λ or ω ≥ 1, else |2ω − 1|."""
    if delta <= 0 or delta == lam:
        raise ValueError("knife-edge (A1 §E6)")
    return 1.0 if (delta > lam or omega >= 1) else abs(2 * omega - 1)


def limit_objective(y: float, omega: float, lam: float, delta: float, nu_minus: float = 0.0) -> float:
    """f(y) = ν₋ + Δ (1 + y)/2 − λ [((1 + y)/2 − ω)⁺ + ((1 − y)/2 − ω)⁺]."""
    return nu_minus + delta * (1 + y) / 2 - lam * (max((1 + y) / 2 - omega, 0.0) + max((1 - y) / 2 - omega, 0.0))


DIVERSIFY_POLICIES = {  # Example ex:diversify (l.395): lists and their probabilities
    "{A,B} always": {("A", "B"): 1.0},
    "{A,C} or {B,C}": {("A", "C"): 0.5, ("B", "C"): 0.5},
    "{A,B} or {A,C}": {("A", "B"): 0.5, ("A", "C"): 0.5},
}
DIVERSIFY_VARIANCE = {"{A,B} always": 250_000.0, "{A,C} or {B,C}": 250.0, "{A,B} or {A,C}": 62_625.0}


def diversify() -> Environment:
    """Example ex:diversify: one nearly deterministic model, K = 2, n = 1000, two equally
    likely states with v_A > v_B > v_C (Λ1) and v_B > v_A > v_C (Λ2), outside below C."""
    market = _one_model_market(1000, ["A", "B", "C"], [{"A": 3.0, "B": 2.0, "C": 1.0}, {"A": 2.0, "B": 3.0, "C": 1.0}],
                               0.0, 2)
    return environment(market, [("A", "B"), ("A", "C"), ("B", "C")])


def policy(env: Environment, lottery: dict[tuple[str, ...], float]) -> np.ndarray:
    w = np.zeros(len(env.lists))
    for shown, weight in lottery.items():
        w[env.list_index(shown)] += weight
    return w


__all__ = ["DIVERSIFY_POLICIES", "DIVERSIFY_VARIANCE", "Environment", "Solution", "diversify", "environment",
           "exact_objective", "expected_cost", "head_to_head_share", "limit_cost_objective", "limit_objective",
           "load_pmf", "moments", "objective_quadratic", "overflow_probability", "p_state", "policy", "program",
           "separate", "two_leader", "two_leader_costs", "value", "y_star_overflow", "y_star_quadratic"]
