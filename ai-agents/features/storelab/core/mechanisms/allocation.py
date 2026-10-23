"""Allocation rules eq. alloc and prop:profit as convex programs (SL-16, l.436–482).

Over the policy space of ``market.ActionTable`` (a distribution over sponsored lists
for every context and organic ranking; organic integrity holds by construction), a
rule maximises

    Σ_j c_j X_j(Q^o, Q^s) + Γ(Q) − Σ_j E C_j(D_j)    s.t. M(Q) ≥ (1 − κ) M^ref

with c_j = b_j (surplus rule, eq. alloc) or c_j = φ̄_j(b_j), the ironed virtual value
(profit rule, prop:profit, A16). The rules of the comparison (Table
tab:model-benchmarks, config ``Mechanism``):

| Rule | Coefficient | Costs in the objective | Policy space |
|---|---|---|---|
| M3 | b_j | none (C ≡ 0) | all sponsored policies |
| M3_set_blind | b_j | none | set-blind: Q^s(· | x, R) the same for every R (A1 §1.6) |
| M4 | b_j | the market's C_j | all |
| profit_rule | φ̄_j(b_j) | the market's C_j | all |

Without costs, and with overflow costs (tangent outer approximation, costs.py, with
the approximation error reported), the program is an LP; quadratic costs make it a
convex QP (exact, Lemma load). The bid vector is a cvxpy parameter, so a market's
program is compiled once and re-solved for every bid (DPP). The program works per
request (X_j / n, Γ / n, ...); ``Allocation`` reports totals. An allocation's
operating cost is always the market's exact E C_j, whatever the rule optimised (M3
ignores costs, the platform still bears them).

**Ties** (eq. alloc: "a fixed bid-independent rule breaks ties"). The objective
carries η M(Q) / n with η = ``TIE_BREAK`` (1e-9): among policies whose objective
values differ by less than that, the one with the higher match quality wins. The
rule does not depend on the bids, and the mechanism is exactly eq. alloc with
G' = G + η M, so Prop ic holds for it as stated, with the externality payments
computed from the same objective (the identity with eq. payment is exact). Its
payments differ from those of the η → 0 limit by at most n η max_x,j u(x, j)
(1e-9 per request for u ≤ 1). Between breakpoints X_j is unique whatever the rule
(a face optimal for an interval of b_j has constant X_j); the rule fixes the rest of
the outcome. LPs are solved by the dual simplex and QPs are polished on their
active set (solver.py), so allocations are exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy import sparse

from . import costs as cost_model
from .market import ActionTable, Market, action_table, match_reference
from .solver import SolverFailure, solve

TIE_BREAK = 1e-9  # η: weight of match quality per request in the objective


@dataclass(frozen=True)
class Rule:
    objective: str  # "surplus" | "profit"
    costs: bool
    set_blind: bool = False


RULES = {"M3": Rule("surplus", costs=False), "M3_set_blind": Rule("surplus", costs=False, set_blind=True),
         "M4": Rule("surplus", costs=True), "profit_rule": Rule("profit", costs=True)}


@dataclass
class Allocation:
    rule: str
    coefficients: np.ndarray  # c_j per advertiser (market.advertisers order)
    y: np.ndarray  # probability of each action row within its (context, organic list) group
    value: float  # optimal objective per request, V(c), including the tie-break term η M / n
    X: dict[str, float]  # expected effective choices, totals
    X_human: dict[str, float]
    X_agent: dict[str, float]
    p_state: np.ndarray  # (states, products): p_j(Λ)
    contribution: float  # Γ
    match: float  # M
    cost: dict[str, float]  # exact E C_j(D_j)
    cost_model_error: float  # exact − modelled cost of the costs in the objective (overflow tangents), total

    @property
    def operating_value(self) -> float:  # G = Γ − Σ E C_j
        return self.contribution - math.fsum(self.cost.values())


class Program:
    """One rule on one market: ``solve(coefficients)`` for any coefficient vector.

    The program is held in standard form over x = (z, e): z the policy (one weight per
    action row, or per (context, sponsored list) when set-blind) and e the overflow
    epigraph variables (one per costed product and state),

        maximise (q₀ + Q_c c)·x − ½ Σ_r d_r (v_r·x)²   s.t.  E x = 1,  G x ≤ h,  z ≥ 0,

    and the cvxpy problem is built from the same data, so a QP solution can be
    polished on it (``polish``).
    """

    def __init__(self, market: Market, rule: str, table: ActionTable | None = None, m_ref: float | None = None):
        self.market, self.rule_name, self.rule = market, rule, RULES[rule]
        self.table = table or action_table(market)
        t, n = self.table, market.requests
        self.ads = [t.index[j] for j in market.advertisers]
        rows = len(t.rows)
        if self.rule.set_blind:
            keys = sorted({(t.groups[g][0], s) for g, s in t.rows})
            column = {k: i for i, k in enumerate(keys)}
            expand = sparse.csr_matrix((np.ones(rows), (np.arange(rows), [column[(t.groups[g][0], s)] for g, s in t.rows])),
                                       shape=(rows, len(keys)))
            groups, ngroups = [c for c, _ in keys], len(market.contexts)
        else:
            expand = sparse.identity(rows, format="csr")
            groups, ngroups = list(t.row_group), len(t.groups)
        self._expand = expand
        nz = expand.shape[1]
        to_z = lambda per_row: np.asarray(expand.T @ per_row).ravel()  # noqa: E731  a row functional in z
        x_map = t.linear(t.q_bar)  # (products, rows): X_j / n
        p_map = [t.linear(t.q_state[s]) for s in range(len(t.state_probs))]
        linear = to_z(t.weight * (t.contribution + TIE_BREAK * t.match))
        quad_vectors, quad_weights, overflow = [], [], []
        self.modelled_costs: dict[str, tuple] = {}
        if self.rule.costs:
            for j, cost in sorted(market.costs.items()):
                if cost.weight == 0.0:
                    continue
                i = t.index[j]
                if cost.kind == "quadratic":
                    linear = linear - cost.weight / (2 * cost.capacity) * to_z(x_map[i])
                    for w, pm in zip(t.state_probs, p_map):
                        quad_vectors.append(to_z(pm[i]))
                        quad_weights.append(cost.weight / cost.capacity * (n - 1) * w)
                    self.modelled_costs[j] = ("quadratic",)
                else:
                    tangents = cost_model.overflow_tangents(n, cost.capacity)
                    overflow.append((j, i, cost, tangents))
                    self.modelled_costs[j] = ("overflow", tangents)
        ne = len(overflow) * len(t.state_probs)
        self.nz, self.nx = nz, nz + ne
        q0 = np.r_[linear, np.zeros(ne)]
        g_rows, h = [], []
        for k, (j, i, cost, tangents) in enumerate(overflow):
            for s, (w, pm) in enumerate(zip(t.state_probs, p_map)):
                col = nz + k * len(t.state_probs) + s
                q0[col] = -cost.weight * w
                pz = to_z(pm[i])
                for a, b in zip(tangents.intercept, tangents.slope):  # a + b p ≤ e
                    row = np.r_[b * pz, np.zeros(ne)]
                    row[col] = -1.0
                    g_rows.append(row)
                    h.append(-a)
        if market.guardrail is not None:
            self.m_ref = m_ref if m_ref is not None else match_reference(market)
            g_rows.append(np.r_[-to_z(t.weight * t.match), np.zeros(ne)])
            h.append(-(1 - market.guardrail.kappa) * self.m_ref / n)
        else:
            self.m_ref = None
        self.q0 = q0
        self.Qc = np.zeros((self.nx, len(self.ads)))
        self.Qc[:nz] = np.asarray(expand.T @ x_map[self.ads].T)
        self.E = np.zeros((ngroups, self.nx))
        self.E[groups, np.arange(nz)] = 1.0
        self.G, self.h = (np.array(g_rows), np.array(h)) if g_rows else (np.zeros((0, self.nx)), np.zeros(0))
        self.V = np.array(quad_vectors).T if quad_vectors else np.zeros((nz, 0))  # (nz, r)
        self.d = np.array(quad_weights)
        self.kind = "qp" if quad_vectors else "lp"

        self.x = cp.Variable(self.nx)
        self.coef = cp.Parameter(len(self.ads))
        objective = self.q0 @ self.x + self.coef @ (self.Qc.T @ self.x)
        if self.kind == "qp":
            objective = objective - 0.5 * cp.sum_squares(np.sqrt(self.d)[:, None] * self.V.T @ self.x[:nz])
        constraints = [self.x[:nz] >= 0, self.E @ self.x == 1]
        if len(self.h):
            constraints.append(self.G @ self.x <= self.h)
        self.problem = cp.Problem(cp.Maximize(objective), constraints)

    @property
    def H(self) -> np.ndarray:
        """Hessian of the concave part, ½ xᵀ H x with H = V diag(d) Vᵀ on z."""
        out = np.zeros((self.nx, self.nx))
        out[:self.nz, :self.nz] = (self.V * self.d) @ self.V.T
        return out

    def value_of(self, x: np.ndarray, coefficients: np.ndarray) -> float:
        q = self.q0 + self.Qc @ coefficients
        vz = self.V.T @ x[:self.nz]
        return float(q @ x - 0.5 * (self.d * vz * vz).sum())

    def solve(self, coefficients) -> Allocation:
        coefficients = np.asarray(coefficients, dtype=float)
        self.coef.value = coefficients
        where = "%s %s" % (self.market.market_id, self.rule_name)
        value = solve(self.problem, self.kind, where)
        x = np.asarray(self.x.value, dtype=float)
        if self.kind == "qp":
            x = polish(self, x, coefficients, where)
            value = self.value_of(x, coefficients)
        y = self._expand @ x[:self.nz]
        return self.allocation(y, value)

    def allocation(self, y: np.ndarray, value: float = math.nan) -> Allocation:
        t, n, market = self.table, self.market.requests, self.market
        out = t.outcomes(y)
        cost = {j: n * cost_model.expected_cost(market.costs.get(j), n, t.state_probs, out["p_state"][:, t.index[j]])
                for j in t.products}
        error = 0.0
        for j, spec in self.modelled_costs.items():
            if spec[0] == "overflow":
                p = out["p_state"][:, t.index[j]]
                modelled = market.costs[j].weight * float(t.state_probs @ spec[1].value(p))
                error += cost[j] - n * modelled
        products = t.products
        return Allocation(rule=self.rule_name, coefficients=np.array(self.coef.value, dtype=float), y=np.asarray(y), value=value,
                          X={j: n * float(out["X"][k]) for k, j in enumerate(products)},
                          X_human={j: n * float(out["X_human"][k]) for k, j in enumerate(products)},
                          X_agent={j: n * float(out["X_agent"][k]) for k, j in enumerate(products)},
                          p_state=out["p_state"], contribution=n * out["contribution"], match=n * out["match"],
                          cost=cost, cost_model_error=error)

    def approximation_bound(self) -> float:
        """Largest possible overflow approximation error of the program, total units:
        Σ_j λ_j n max_p (h(p) − tangent envelope) / n."""
        n = self.market.requests
        return sum(self.market.costs[j].weight * n * spec[1].max_error
                   for j, spec in self.modelled_costs.items() if spec[0] == "overflow")


# ------------------------------------------------------------------- polish
POLISH_ACTIVE = 1e-7  # initial working set: z at its bound, or G x ≤ h binding, within this
POLISH_TOL = 1e-11  # per request, times max(1, max |q|): step size, multiplier signs, reduced costs, objective
POLISH_STEPS = 200
POLISH_SANITY = 1e-7  # relative: the polished objective may not fall further below the interior point's
POLISH_RCOND = 1e-10  # singular values below this (relative) are ties: no correction along them


def polish(program: Program, x0: np.ndarray, coefficients: np.ndarray, where: str = "") -> np.ndarray:
    """Exact solution of the QP by a primal active-set method started at the interior point x0.

    The working set W holds bounds z_i = 0 and rows G_r x = h_r. Each step solves the
    equality-constrained QP on W (q − H x = Eᵀν + G_Wᵀμ, E x = 1, G_W x = h_W; with
    ties the system is singular, and the solution nearest the current point is used,
    with no correction along directions whose singular value is below ``POLISH_RCOND``)
    and moves towards it as far as feasibility allows, adding the blocking bound or
    row to W. When the step is zero, a row with μ < 0 or a bound whose reduced cost
    is positive leaves W; when neither exists the point satisfies the KKT conditions
    exactly and is the optimum (a KKT point of a convex QP); as a check against gross
    errors its objective must not fall more than 1e-7 (relative) below x0's.
    The interior point is accurate in objective but not always in x along flat
    directions, so its own active set can be wrong; this repairs it. Failing in
    ``POLISH_STEPS`` steps, the solve fails (``unpolished``).
    """
    q = program.q0 + program.Qc @ coefficients
    H, E, G, h, nz, nx = program.H, program.E, program.G, program.h, program.nz, program.nx
    base = program.value_of(x0, coefficients)
    tol = POLISH_TOL * max(1.0, float(np.abs(q).max()))
    x = x0.copy()
    x[:nz] = np.maximum(x[:nz], 0.0)
    bound = np.zeros(nx, dtype=bool)
    bound[:nz] = x[:nz] <= POLISH_ACTIVE
    x[bound] = 0.0
    rows = (h - G @ x <= POLISH_ACTIVE * np.maximum(1.0, np.abs(h))) if len(h) else np.zeros(0, dtype=bool)
    for _ in range(POLISH_STEPS):
        free = ~bound
        A = np.vstack([E, G[rows]])
        b = np.r_[np.ones(len(E)), h[rows]]
        kkt = np.block([[H[np.ix_(free, free)], A[:, free].T], [A[:, free], np.zeros((len(A), len(A)))]])
        rhs = np.r_[q[free], b]
        start = np.r_[x[free], np.zeros(len(A))]
        solution = start + np.linalg.lstsq(kkt, rhs - kkt @ start, rcond=POLISH_RCOND)[0]
        target = np.zeros(nx)
        target[free] = solution[:free.sum()]
        step = target - x
        if np.abs(step).max() > tol:
            alpha, block = 1.0, None
            down = np.where(free[:nz] & (step[:nz] < 0))[0]
            if len(down):
                ratios = -x[down] / step[down]
                k = int(np.argmin(ratios))
                if ratios[k] < alpha:
                    alpha, block = ratios[k], ("bound", down[k])
            if len(h):
                rise = G @ step
                up = np.where(~rows & (rise > 0))[0]
                if len(up):
                    ratios = (h[up] - G[up] @ x) / rise[up]
                    k = int(np.argmin(ratios))
                    if ratios[k] < alpha:
                        alpha, block = ratios[k], ("row", up[k])
            if block is not None:  # move to the blocking bound or row and add it to W
                x = x + max(alpha, 0.0) * step
                if block[0] == "bound":
                    bound[block[1]] = True
                    x[block[1]] = 0.0
                else:
                    rows[block[1]] = True
                continue
        x = target  # the optimum on W (a full step): check the multipliers
        multipliers = solution[free.sum():]
        mu = np.zeros(len(h))
        mu[rows] = multipliers[len(E):]
        reduced = q - H @ x - A.T @ multipliers
        released = np.where(rows & (mu < -tol))[0] if len(h) else []
        improving = np.where(bound & (reduced > tol))[0]
        if len(released):
            rows[released[np.argmin(mu[released])]] = False
        elif len(improving):
            bound[improving[np.argmax(reduced[improving])]] = False
        else:
            feasible = (x[:nz] >= -tol).all() and (not len(h) or (G @ x <= h + tol).all())
            # KKT points of a convex QP are optimal; the objective test only guards against a gross error
            # (x0 is feasible only to the solver's tolerance, so it can score slightly higher)
            sanity = POLISH_SANITY * max(1.0, abs(base), float(np.abs(q).max()))
            if not feasible or np.abs(E @ x - 1).max() > tol or program.value_of(x, coefficients) < base - sanity:
                break
            x[:nz] = np.maximum(x[:nz], 0.0)
            return x
    raise SolverFailure("unpolished", where)
