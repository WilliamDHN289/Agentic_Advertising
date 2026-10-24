"""E3 runs on the stipulated states of A17 §3 (SL-20S; SL-20 reruns Part B on E1 states).

``python -m features.storelab.analysis.e3_run`` writes ``artifacts/results/sl20s/``:

- ``part_a.csv.gz``: Prop threshold over the A17 §3.1 grid (450 quadratic points, 150 overflow
  points with the limit objective, the same 150 at n ∈ {100, 1000, 10000}), each solved as a
  convex program (``e3.program``) and compared with the closed forms and the finite-n bound of
  A1 §2.3 under the A12 §4 tolerances. If any point fails, Part B is not run.
- ``diversify.csv.gz``: Example ex:diversify through the same code path.
- ``part_b/<alignment>/<level>.csv.gz``: one row per (market, policy).
- ``h3c.csv.gz``: sponsorship on (H3c), M3 against M4, one row per (market, mechanism, leader).

**Part B markets** (A17 §1 and §3.2, owner decisions 2026-10-24). A market is drawn from
``stream(MASTER_SEED, "sl20s-market", config_id, index)``: 10 qualities z ~ N(0, 1); the two
largest become the leaders L1 and L2, both set to their mean z_L; the other 8 are redrawn from
N(0, 1) truncated above at z_L − 0.5 (inverse CDF); then the 6 advertisers and their values
θ ~ U[0, 1] (H3c). Agent scores equal z (so the leaders lead); in each state the state adds
±0.5 to the leaders' scores in opposite directions. Match quality u = Φ(z) does not depend on
the state (*not aligned*) or is Φ(z + shift) with the state's shift (*aligned*; with two models,
model 1's shift). g = 0.1, σ = 0.3, outside score −1, K = 4, n = 1000; humans (share 1 − α)
follow A17 §1 with ζ = 0.2 + 0.6 ū, ū = E_Λ u.

**Policies.** *original*: the ranking by ū, descending; its top K maximises match quality in
the deterministic limit (both leaders shown, ties among lists broken by u). *separated*: Prop
separation's lottery over the anchors R†_L1 and R†_L2 (the original list without the other
leader), with ϖ from the deterministic limit. *composition-blind*: the original list or R†_L1
with probability 1/2 each (randomises as often as the separated policy but keeps L1 in every
set). *quota*: the separated lottery delivered as fixed counts n ϖ.

Every policy is evaluated at σ = 0.3 (exploratory metrics) and in the deterministic limit, where
the H3a exact part and Θ_j (Lemma contested) are defined (owner decision 2026-10-24). Costs on
every product: quadratic λ D²/(2B) and overflow λ (D − B)⁺, B = ω n, over A17's ω and λ.

**H3c** uses the not-aligned baseline markets (same draws): the separated lottery is the
organic policy, K^s = 1, the E2 baselines for advertisers and values (δ = −0.5), κ = 0.05, and
M4's cost is A17's baseline quadratic cost (λ = 0.5, ω = 0.5) on every product. At σ = 0.3 the
separated lottery is often more than κ below M^ref, so the paper's guardrail leaves no feasible
policy; H3c uses SL-19S's feasibility floor M ≥ T = min{(1 − κ) M^ref, M(Q^o, ∅) − 1e-8 n}
(owner decision 2026-10-24; see ``e2_markets``), and the file records M^ref, M(Q^o, ∅), T and
whether the floor binds. For each leader that is an advertiser: Q* = Q^{s*}(b) and
Q^{−j} = Q^{s*}(0, b_{−j}) from SL-16's programs, the buy-back share (the probability that j is
sponsored into a list from which the organic policy removed it) and the terms of Cor congestion:
the bid term b_j [X_j(Q*) − X_j(Q^{−j})], the other bidders' term, the contribution term, the
congestion term Σ_i [E_{Q*} C_i − E_{Q^{−j}} C_i] (in M4's objective; under M3 it is reported
but C ≡ 0), and the tie-break term. A12's H3c statistic compares buy-back with "bid term >
congestion term"; A1 §2.4's reading (buy-back only if the bid term exceeds all remaining terms)
is recorded too.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from ..core.mechanisms.allocation import TIE_BREAK
from ..core.mechanisms.evaluate import MarketRunner
from ..core.mechanisms.market import (AgentModel, Context, Cost, Display, Guardrail, Market, Offer, OrganicList,
                                      State, action_table, match_reference)
from ..core.mechanisms.solver import SolverFailure
from ..core.rng import stream
from . import e2_run as er
from . import e3
from .figures import REPO

MASTER_SEED = 2026102320
RESULTS = REPO / "artifacts" / "results" / "sl20s"
MARKETS = 200
N = 1000
K, K_S = 4, 1
ELIGIBLE = 10
ADVERTISERS = 6
SIGMA = 0.3
OUTSIDE = -1.0
SHIFT = 0.5
GAP = 0.5  # other products: z ≤ z_L − GAP
CONTRIBUTION = 0.1
KAPPA = 0.05
GAMMA = [0.30 * k ** -0.6 for k in range(1, K + 1)]
CHI = 0.5
LEADERS = ("L1", "L2")
OMEGAS = (0.3, 0.5, 0.6, 0.8, 1.0)
LAMBDAS = (0.1, 0.5, 2.0)
COST_KINDS = ("quadratic", "overflow")
H3C_COST = (0.5, 0.5)  # λ, ω of A17's baseline cost
INDIFFERENCE = 1e-6  # per request (A12 §4, H3c)
BUYBACK = 1e-6  # a buy-back share above this counts as buy-back

# Part A grid (A17 §3.1)
A_OMEGAS = (0.3, 0.5, 0.6, 0.8, 1.0)
A_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
A_DELTAS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
A_QUADRATIC_N = (10, 100, 1000)
A_LIMIT_N = 1000  # the limit objective does not depend on n; the program uses n only for B = ω n
A_FINITE_N = (100, 1000, 10000)
Y_TOLERANCE = 1e-4
OBJECTIVE_TOLERANCE = 1e-6
BOUND_SLACK = 1e-6


@dataclass(frozen=True)
class Level:
    human_share: float = 0.0
    state_probs: tuple[float, ...] = (0.5, 0.5)
    third_state: bool = False
    models: str = "one"  # one | identical | independent | opposite


LEVELS = {
    "baseline": Level(),
    "human_share=0.25": Level(human_share=0.25),
    "human_share=0.5": Level(human_share=0.5),
    "state_probs=0.7,0.3": Level(state_probs=(0.7, 0.3)),
    "states=3": Level(state_probs=(1 / 3, 1 / 3, 1 / 3), third_state=True),
    "models=2,identical": Level(models="identical"),
    "models=2,independent": Level(models="independent"),
    "models=2,opposite": Level(models="opposite"),
}
ALIGNMENTS = ("not_aligned", "aligned")
POLICIES = ("original", "separated", "composition_blind", "quota")
H3C_SOURCE = "part_b/not_aligned/baseline"


def configurations() -> list[tuple[str, str, str]]:
    return [("part_b/%s/%s" % (a, name), a, name) for a in ALIGNMENTS for name in LEVELS]


# --------------------------------------------------------------------- markets
@dataclass
class Draw:
    z: dict[str, float]
    z_leader: float
    advertisers: list[str]
    values: dict[str, float]


def draw(config_id: str, index: int) -> Draw:
    rng = stream(MASTER_SEED, "sl20s-market", config_id, index)
    z = rng.standard_normal(ELIGIBLE)
    z_leader = float(np.sort(z)[-2:].mean())
    others = np.sort(stats.norm.ppf(rng.uniform(size=ELIGIBLE - 2) * stats.norm.cdf(z_leader - GAP)))[::-1]
    names = list(LEADERS) + ["n%d" % (k + 1) for k in range(ELIGIBLE - 2)]
    scores = dict(zip(names, [z_leader, z_leader] + [float(v) for v in others]))
    advertisers = sorted(names[k] for k in rng.permutation(ELIGIBLE)[:ADVERTISERS])
    values = {j: float(v) for j, v in zip(advertisers, rng.uniform(size=ADVERTISERS))}
    return Draw(z=scores, z_leader=z_leader, advertisers=advertisers, values=values)


def _shift(state: int) -> dict[str, float]:
    """The leaders' score shift in base state 0 (+L1), 1 (+L2) or 2 (none)."""
    return [{"L1": SHIFT, "L2": -SHIFT}, {"L1": -SHIFT, "L2": SHIFT}, {"L1": 0.0, "L2": 0.0}][state]


def states(level: Level) -> list[tuple[str, float, list[int]]]:
    """(state id, probability, base state of each model)."""
    base = list(range(3 if level.third_state else 2))
    if level.models == "one":
        return [("s%d" % (b + 1), level.state_probs[b], [b]) for b in base]
    if level.models == "identical":
        return [("s%d" % (b + 1), level.state_probs[b], [b, b]) for b in base]
    if level.models == "opposite":
        return [("s%d" % (b + 1), level.state_probs[b], [b, 1 - b if b < 2 else b]) for b in base]
    return [("s%d_s%d" % (a + 1, b + 1), level.state_probs[a] * level.state_probs[b], [a, b])
            for a, b in itertools.product(base, base)]


def build(d: Draw, level: Level, alignment: str, organic: list[tuple[list[str], float]] | None = None,
          sponsorship: bool = False) -> tuple[Market, np.ndarray]:
    """The SL-16 market of a draw and u by state (S, J) in sorted product order."""
    st = states(level)
    names = sorted(d.z)
    u_state = np.array([[stats.norm.cdf(d.z[j] + (_shift(bases[0]).get(j, 0.0) if alignment == "aligned" else 0.0))
                         for j in names] for _, _, bases in st])
    probs = np.array([p for _, p, _ in st])
    u_bar = dict(zip(names, probs @ u_state))
    offers = {j: Offer(contribution=CONTRIBUTION, match=float(u_bar[j]),
                       organic=Display(request=0.6, conversion=0.2 + 0.6 * float(u_bar[j])),
                       sponsored=Display(request=0.54, conversion=0.2 + 0.6 * float(u_bar[j]))) for j in names}
    count = 1 if level.models == "one" else 2
    deltas = (-0.5,) if count == 1 else (-0.3, -0.7)
    models = [AgentModel(key="m%d" % (m + 1), sigma=SIGMA, label_effect=deltas[m], page_request_prob=CHI,
                         outside={"x": OUTSIDE},
                         scores={sid: {"x": {j: d.z[j] + _shift(bases[m]).get(j, 0.0) for j in names}}
                                 for sid, _, bases in st})
              for m in range(count)]
    ranking = original_ranking(d, u_bar)
    organic = organic or [(ranking, 1.0)]
    context = Context(context_id="x", probability=1.0, offers=offers,
                      organic=[OrganicList(ranking=r, probability=p) for r, p in organic],
                      agent_share=1.0 - level.human_share, model_shares={m.key: 1.0 / count for m in models})
    lam, omega = H3C_COST
    market = Market(market_id="sl20s", requests=N, max_displayed=K, max_sponsored=K_S if sponsorship else 0,
                    advertisers=d.advertisers if sponsorship else [], contexts=[context],
                    states=[State(state_id=sid, probability=p) for sid, p, _ in st], position_weights=GAMMA,
                    models=models,
                    costs={j: Cost(kind="quadratic", weight=lam, capacity=omega * N) for j in names} if sponsorship
                    else {},
                    guardrail=Guardrail(kappa=KAPPA) if sponsorship else None)
    return market, u_state


def original_ranking(d: Draw, u_bar: dict[str, float]) -> list[str]:
    return sorted(d.z, key=lambda j: (-u_bar[j], j))


def anchor_ranking(ranking: list[str], leader: str) -> list[str]:
    """R†_leader: the ranking with the other leader moved below position K."""
    other = LEADERS[1 - LEADERS.index(leader)]
    return [j for j in ranking if j != other] + [other]


def setup(config_id: str, alignment: str, level: Level, index: int):
    d = draw(config_id, index)
    market, u_state = build(d, level, alignment)
    ranking = market.contexts[0].organic[0].ranking
    lists = [tuple(ranking[:K])] + [tuple(anchor_ranking(ranking, L)[:K]) for L in LEADERS]
    env = e3.environment(market, lists, match=u_state)
    original = np.array([1.0, 0.0, 0.0])
    separated = e3.separate(env, original, {L: lists[1 + k] for k, L in enumerate(LEADERS)})
    policies = {"original": (original, False), "separated": (separated, False),
                "composition_blind": (np.array([0.5, 0.5, 0.0]), False), "quota": (separated, True)}
    return d, market, env, policies, ranking


# ---------------------------------------------------------------------- Part B
def _rel(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / abs(b) if b != 0 else math.inf


COST_FIELDS = ["cost_%s_w%g_l%g" % (k, w, lam) for k in COST_KINDS for w in OMEGAS for lam in LAMBDAS]
OVERFLOW_FIELDS = ["overflow_prob_w%g" % w for w in OMEGAS]
LEADER_FIELDS = ["mean", "within", "across", "total", "var_p", "theta", "ratio", "within_det", "across_det"]
PART_B_FIELDS = (["config_id", "alignment", "level", "index", "market_hash", "z_leader", "alpha", "policy", "weights",
                  "quota", "match", "contribution", "match_det", "contribution_det"]
                 + ["%s_%s" % (L, f) for L in LEADERS for f in LEADER_FIELDS]
                 + ["decomposition_error", "decomposition_error_det", "contested_bound_holds", "cost_check",
                    "exact_load_error", "exact_contribution_error", "exact_match_error", "exact_across",
                    "exact_applies"] + COST_FIELDS + OVERFLOW_FIELDS)


def evaluate_part_b(task: tuple[str, str, str, int]) -> list[dict]:
    config_id, alignment, level_name, index = task
    level = LEVELS[level_name]
    d, market, env, policies, _ = setup(config_id, alignment, level, index)
    lead = [env.index[L] for L in LEADERS]
    rows, reference = [], None
    for name in POLICIES:
        w, quota = policies[name]
        ms, md = e3.moments(env, w, quota=quota), e3.moments(env, w, det=True, quota=quota)
        vs, vd = e3.value(env, w), e3.value(env, w, det=True)
        if name == "original":
            reference = (md, vd)
        row = {"config_id": config_id, "alignment": alignment, "level": level_name, "index": index,
               "market_hash": market.market_hash(), "z_leader": d.z_leader, "alpha": 1 - level.human_share,
               "policy": name, "weights": json.dumps([float(x) for x in w]), "quota": int(quota),
               "match": vs["match"], "contribution": vs["contribution"], "match_det": vd["match"],
               "contribution_det": vd["contribution"]}
        for L, j in zip(LEADERS, lead):
            theta = float(md["theta"][j])
            row.update({"%s_mean" % L: float(ms["mean"][j]), "%s_within" % L: float(ms["within"][j]),
                        "%s_across" % L: float(ms["across"][j]), "%s_total" % L: float(ms["total"][j]),
                        "%s_var_p" % L: float(ms["var_p"][j]), "%s_theta" % L: theta,
                        "%s_ratio" % L: float(ms["var_p"][j] / (theta ** 2 / 4)) if theta > 0 else None,
                        "%s_within_det" % L: float(md["within"][j]), "%s_across_det" % L: float(md["across"][j])})
        row["decomposition_error"] = float(np.max(np.abs(ms["total"] - ms["within"] - ms["across"])
                                                  / np.maximum(ms["total"], 1.0)))
        row["decomposition_error_det"] = float(np.max(np.abs(md["total"] - md["within"] - md["across"])
                                                      / np.maximum(md["total"], 1.0)))
        row["contested_bound_holds"] = int(bool(np.all(md["var_p"] <= md["theta"] ** 2 / 4 + 1e-15)))
        check = 0.0
        if not quota:
            for kind in COST_KINDS:
                for omega in OMEGAS:
                    cost = Cost(kind=kind, weight=1.0, capacity=omega * N)
                    for j in range(len(env.products)):
                        check = max(check, abs(e3.expected_cost(ms["pmf"][j], cost) / N - e3.cost_model.expected_cost(
                            cost, N, env.state_probs, ms["p_state"][:, j])))
        row["cost_check"] = check
        ref_m, ref_v = reference
        row["exact_load_error"] = max(_rel(a, b) for a, b in zip(md["mean"], ref_m["mean"]))
        row["exact_contribution_error"] = _rel(vd["contribution"], ref_v["contribution"])
        row["exact_match_error"] = _rel(vd["match"], ref_v["match"])
        row["exact_across"] = float(np.max(md["across"]))
        row["exact_applies"] = int(alignment == "not_aligned" and level.human_share == 0.0
                                   and name in ("separated", "quota"))
        for kind in COST_KINDS:
            for omega in OMEGAS:
                for lam in LAMBDAS:
                    cost = Cost(kind=kind, weight=lam, capacity=omega * N)
                    row["cost_%s_w%g_l%g" % (kind, omega, lam)] = math.fsum(
                        e3.expected_cost(ms["pmf"][j], cost) for j in range(len(env.products))) / N
        for omega in OMEGAS:
            row["overflow_prob_w%g" % omega] = float(np.mean([e3.overflow_probability(ms["pmf"][j], omega * N)
                                                              for j in lead]))
        rows.append(row)
    return rows


# ------------------------------------------------------------------------- H3c
H3C_FIELDS = ["index", "market_hash", "mechanism", "leader", "theta", "excluded", "status", "m_ref", "m_organic",
              "threshold", "floor_binds", "slack", "buyback_share", "buyback", "bid_term", "others_term",
              "contribution_term", "cost_term", "tie_term", "remaining", "margin", "margin_check", "cost_change",
              "bid_exceeds_congestion", "congestion_indifferent", "only_if_holds", "var_p_organic", "var_p_star",
              "var_p_minus", "x_star", "x_minus"]
FLOOR_SLACK = 1e-8  # per request, as in e2_markets


def _buyback_share(prog, alloc, leader: str) -> float:
    t = prog.table
    removed = [g for g, (ci, oi) in enumerate(t.groups)
               if leader not in t.market.contexts[ci].organic[oi].ranking[:K]]
    mass = math.fsum(t.group_weight[g] for g in removed)
    if mass == 0:
        return 0.0
    bought = math.fsum(t.weight[r] * alloc.y[r] for r, (g, sponsored) in enumerate(t.rows)
                       if g in removed and leader in sponsored)
    return bought / mass


def evaluate_h3c(index: int) -> list[dict]:
    d = draw(H3C_SOURCE, index)
    level = LEVELS["baseline"]
    _, _, env, policies, ranking = setup(H3C_SOURCE, "not_aligned", level, index)
    separated = policies["separated"][0]
    organic = [(anchor_ranking(ranking, L), float(separated[1 + k])) for k, L in enumerate(LEADERS)
               if separated[1 + k] > 0]
    market, _ = build(d, level, "not_aligned", organic=organic, sponsorship=True)
    m_ref = match_reference(market)
    table = action_table(market)
    m_organic = N * table.outcomes(np.array([0.0 if s else 1.0 for _, s in table.rows]))["match"]
    threshold = (1 - KAPPA) * m_ref
    floor = m_organic < threshold
    if floor:  # the floor binds: FLOOR_SLACK gives the feasible set an interior
        threshold = m_organic - FLOOR_SLACK * N
    market = market.model_copy(update={"guardrail": Guardrail(kappa=KAPPA, m_ref=threshold / (1 - KAPPA))})
    runner = MarketRunner(market)
    theta = np.array([d.values[j] for j in market.advertisers])
    rows = []
    base = {"index": index, "market_hash": market.market_hash(), "m_ref": m_ref, "m_organic": m_organic,
            "threshold": threshold, "floor_binds": int(floor)}
    leaders = [L for L in LEADERS if L in market.advertisers]
    for mechanism in ("M3", "M4"):
        try:
            prog = runner.program(mechanism)
            star = prog.solve(theta)
            empty = np.array([1.0 if not sponsored else 0.0 for _, sponsored in prog.table.rows])
            organic_alloc = prog.allocation(empty)
            results = []
            for L in leaders:
                k = market.advertisers.index(L)
                others = theta.copy()
                others[k] = 0.0
                results.append((L, k, prog.solve(others)))
        except SolverFailure as failure:
            rows.append({**base, "mechanism": mechanism, "excluded": 1, "status": failure.status})
            continue
        for L, k, minus in results:
            bid = theta[k] * (star.X[L] - minus.X[L])
            others_term = math.fsum(theta[i] * (minus.X[a] - star.X[a])
                                    for i, a in enumerate(market.advertisers) if a != L)
            contribution_term = minus.contribution - star.contribution
            cost_change = math.fsum(star.cost.values()) - math.fsum(minus.cost.values())
            cost_term = cost_change if prog.rule.costs else 0.0
            tie_term = TIE_BREAK * (minus.match - star.match)
            remaining = others_term + contribution_term + cost_term + tie_term
            margin = bid - remaining
            # the same margin from the program values: n [V_b(Q*) − V_b(Q^{−j})]
            check = margin - N * (star.value - (minus.value + theta[k] * minus.X[L] / N))
            share = _buyback_share(prog, star, L)
            p = prog.table.index[L]
            rows.append({**base, "mechanism": mechanism, "leader": L, "theta": float(theta[k]), "excluded": 0,
                         "status": "optimal", "slack": star.match - threshold,
                         "buyback_share": share, "buyback": int(share > BUYBACK), "bid_term": bid,
                         "others_term": others_term, "contribution_term": contribution_term, "cost_term": cost_term,
                         "tie_term": tie_term, "remaining": remaining, "margin": margin, "margin_check": check,
                         "cost_change": cost_change, "bid_exceeds_congestion": int(bid > cost_change),
                         "congestion_indifferent": int(abs(bid - cost_change) <= INDIFFERENCE * N),
                         "only_if_holds": int(share <= BUYBACK or margin > INDIFFERENCE * N),
                         "var_p_organic": _var(prog.table.state_probs, organic_alloc.p_state[:, p]),
                         "var_p_star": _var(prog.table.state_probs, star.p_state[:, p]),
                         "var_p_minus": _var(prog.table.state_probs, minus.p_state[:, p]),
                         "x_star": star.X[L], "x_minus": minus.X[L]})
        if not results:
            rows.append({**base, "mechanism": mechanism, "excluded": 0, "status": "no_leader_advertiser"})
    return rows


def _var(probs: np.ndarray, p: np.ndarray) -> float:
    mean = float(probs @ p)
    return float(probs @ (p - mean) ** 2)


# ---------------------------------------------------------------------- Part A
PART_A_FIELDS = ["case", "n", "omega", "lambda", "delta", "y", "y13", "y23", "y_star", "y_error", "objective",
                 "closed_form", "relative_error", "gap", "bound", "finite_objective", "uniform_gap",
                 "approximation_error", "y_bound", "y_within_bound", "passed"]


def part_a_tasks() -> list[tuple]:
    grid = list(itertools.product(A_OMEGAS, A_LAMBDAS, A_DELTAS))
    return ([("quadratic", n, *g) for n in A_QUADRATIC_N for g in grid] + [("overflow_limit", A_LIMIT_N, *g) for g in grid]
            + [("overflow_finite", n, *g) for n in A_FINITE_N for g in grid])


def evaluate_part_a(task: tuple) -> list[dict]:
    case, n, omega, lam, delta = task
    env = e3.two_leader(n, delta)
    kind = "quadratic" if case == "quadratic" else "overflow"
    costs = e3.two_leader_costs(kind, n, omega, lam)
    mode = {"quadratic": "exact", "overflow_limit": "limit", "overflow_finite": "finite"}[case]
    sol = e3.program(env, costs, mode)
    w = sol.w
    y = e3.head_to_head_share(env, w)
    row = {"case": case, "n": n, "omega": omega, "lambda": lam, "delta": delta, "y": y,
           "y13": float(w[env.list_index(("1", "3"))]), "y23": float(w[env.list_index(("2", "3"))]),
           "approximation_error": sol.approximation_error}
    if case == "quadratic":
        y_star = e3.y_star_quadratic(n, omega, lam, delta)
        objective, closed = e3.exact_objective(env, w, costs), e3.objective_quadratic(y_star, n, omega, lam, delta)
        rel = abs(objective - closed) / abs(closed)
        row.update(y_star=y_star, y_error=abs(y - y_star), objective=objective, closed_form=closed, relative_error=rel,
                   passed=int(abs(y - y_star) <= Y_TOLERANCE and rel <= OBJECTIVE_TOLERANCE))
    else:
        y_star = e3.y_star_overflow(omega, lam, delta)
        objective, closed = e3.limit_cost_objective(env, w, costs), e3.limit_objective(y_star, omega, lam, delta)
        row.update(y_star=y_star, y_error=abs(y - y_star), objective=objective, closed_form=closed)
        if case == "overflow_limit":
            rel = abs(objective - closed) / abs(closed)
            row.update(relative_error=rel, passed=int(rel <= OBJECTIVE_TOLERANCE))
        else:
            gap, bound = closed - objective, 2 * lam / math.sqrt(n) + BOUND_SLACK
            finite = e3.exact_objective(env, w, costs)
            edge = min(delta, abs(delta - lam)) if omega < 1 else delta
            y_bound = 4 * lam / (math.sqrt(n) * edge)
            row.update(gap=gap, bound=bound, finite_objective=finite, uniform_gap=abs(finite - objective),
                       y_bound=y_bound, y_within_bound=int(abs(y - y_star) <= y_bound), passed=int(gap <= bound))
    return [row]


def diversify_rows() -> list[dict]:
    env = e3.diversify()
    a = env.index["A"]
    rows = []
    for name, lottery in e3.DIVERSIFY_POLICIES.items():
        w = e3.policy(env, lottery)
        m = e3.moments(env, w, det=True)
        rows.append({"policy": name, "within": float(m["within"][a]), "across": float(m["across"][a]),
                     "lemma_total": float(m["within"][a] + m["across"][a]), "distribution_total": float(m["total"][a]),
                     "paper": e3.DIVERSIFY_VARIANCE[name], "theta": float(m["theta"][a]),
                     "bound": float(env.n ** 2 * m["theta"][a] ** 2 / 4), "mean": float(m["mean"][a])})
    separated = e3.separate(env, e3.policy(env, {("A", "B"): 1.0}), {"A": ("A", "C"), "B": ("B", "C")})
    rows.append({"policy": "separated version of {A,B} always", "within": None, "across": None, "lemma_total": None,
                 "distribution_total": None, "paper": None, "theta": None, "bound": None, "mean": None,
                 "weights": json.dumps(dict(zip(["%s,%s" % s for s in env.lists], map(float, separated))))})
    return rows


DIVERSIFY_FIELDS = ["policy", "within", "across", "lemma_total", "distribution_total", "paper", "theta", "bound",
                    "mean", "weights"]


# ------------------------------------------------------------------------- run
def run(group: str, match: str | None, markets: int, out: Path) -> dict:
    summary = {}
    if group in ("part_a", "all") and (match is None or "part_a" in match):
        rows = [r for rs in er._map(evaluate_part_a, part_a_tasks(), "part A") for r in rs]
        er.write_rows(out / "part_a.csv.gz", PART_A_FIELDS, rows)
        er.write_rows(out / "diversify.csv.gz", DIVERSIFY_FIELDS, diversify_rows())
        failed = [r for r in rows if not r["passed"]]
        summary["part_a"] = {"points": len(rows), "failed": len(failed)}
        if failed:
            print("Part A failed at %d points: Part B is blocked" % len(failed), flush=True)
            return summary
    elif group in ("part_b", "h3c", "all"):
        path = out / "part_a.csv.gz"
        if not path.exists() or any(r["passed"] != "1" for r in er.read_rows(path)):
            raise SystemExit("Part B is blocked until every Part A point passes (run --group part_a)")
    if group in ("part_b", "all"):
        configs = [c for c in configurations() if match is None or match in c[0]]
        tasks = [(cid, a, name, i) for cid, a, name in configs for i in range(markets)]
        results = er._map(evaluate_part_b, tasks, "part B")
        by_config: dict[str, list] = {}
        for (cid, *_), rows in zip(tasks, results):
            by_config.setdefault(cid, []).extend(rows)
        for cid, rows in by_config.items():
            er.write_rows(er.result_path(out, cid), PART_B_FIELDS, rows)
            summary[cid] = {"markets": len({r["index"] for r in rows})}
    if group in ("h3c", "all") and (match is None or "h3c" in match):
        rows = [r for rs in er._map(evaluate_h3c, list(range(markets)), "H3c") for r in rs]
        er.write_rows(out / "h3c.csv.gz", H3C_FIELDS, rows)
        summary["h3c"] = {"markets": markets, "excluded": len({r["index"] for r in rows if r["excluded"]})}
    return summary


def verify_rerun(groups: list[str], committed: Path, markets: int) -> int:
    """Rerun each ``group[:match]`` into a temporary directory and compare every file it
    writes byte for byte with the committed one; the result goes to ``rerun_check.json``."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k, spec in enumerate(groups):
            group, _, match = spec.partition(":")
            scratch = Path(tmp) / str(k)
            if group in ("part_b", "h3c"):  # the gate reads Part A
                scratch.mkdir(parents=True)
                (scratch / "part_a.csv.gz").write_bytes((committed / "part_a.csv.gz").read_bytes())
            run(group, match or None, markets, scratch)
            for path in sorted(scratch.rglob("*.csv.gz")):
                name = path.relative_to(scratch).as_posix()
                if group != "part_a" and name == "part_a.csv.gz":
                    continue
                results[name] = (committed / name).exists() and path.read_bytes() == (committed / name).read_bytes()
    (committed / "rerun_check.json").write_text(json.dumps({"files": results, "identical": all(results.values()),
                                                            "workers": er.WORKERS}, indent=1, sort_keys=True) + "\n",
                                                encoding="utf-8")
    print(json.dumps(results))
    return 0 if all(results.values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.e3_run")
    parser.add_argument("--group", default="all", choices=["part_a", "part_b", "h3c", "all"])
    parser.add_argument("--match", help="only configurations whose id contains this text")
    parser.add_argument("--markets", type=int, default=MARKETS)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--verify-rerun", nargs="+", metavar="GROUP[:MATCH]",
                        help="rerun these groups into a temporary directory and compare the files byte for byte")
    args = parser.parse_args(argv)
    er.WORKERS = args.workers
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    start = time.time()
    if args.verify_rerun:
        return verify_rerun(args.verify_rerun, args.out, args.markets)
    summary = run(args.group, args.match, args.markets, args.out)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("%.0f s" % (time.time() - start), flush=True)
    failed = summary.get("part_a", {}).get("failed", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
