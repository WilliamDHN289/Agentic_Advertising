"""SL-16 mechanism validation: hand-solved markets, payment identity, incentives,
solver exclusions and runtime.

    python -m features.storelab.analysis.mechanism_validation [--quick] [--out DIR] [--workers N]

1. **Hand-solved markets** (AC-1): closed forms from the paper and A1, each checked to
   1e-8 per request (|computed − expected| / n ≤ 1e-8 max(1, |expected| / n)).
2. **Payment identity** (AC-2): on 100 random markets, for every advertiser and every
   surplus rule (M3, set-blind M3, M4), the externality payment of Prop ic(c) against
   eq. payment integrated over the breakpoints of X_j(z). Relative error |ΔP| /
   max(|P_ext|, |P_int|, n·1e-7): a payment below 1e-7 per request counts as zero and is
   compared absolutely (1e-13 per request).
3. **Incentives** (AC-3): on 100 random markets, for M3, M4 and the profit rule and
   every advertiser, X_j on a bid grid (21 bids on [0, max(2θ_j, F_j⁻¹(0.99))] and
   θ_j) must be nondecreasing (to 1e-9 per request) and no grid bid may gain more than
   1e-6 over truthful bidding (utility θ_j X_j − P_j, total units).
4. **Exclusions** (AC-4): every solve that does not return ``optimal`` (or cannot be
   polished) excludes its market and is counted; the studies report the counts, and
   a batch with a market whose guardrail cannot be met shows the market excluded
   and absent from the outcomes.
5. **Runtime** (AC-5): seconds per market and mechanism, by market size, and the
   resulting compute estimate for SL-19.

Random markets come from keyed streams of MASTER_SEED, one per market, so results do
not depend on ``--workers``. The report records a sha256 of the code it ran.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from ..core.mechanisms import gsp
from ..core.mechanisms import payments as pay
from ..core.mechanisms.allocation import Program
from ..core.mechanisms.evaluate import MECHANISMS, MarketRunner, evaluate_markets
from ..core.mechanisms.market import (AgentModel, Context, Cost, Display, Guardrail, Market, Offer, OrganicList,
                                      State, action_table, match_reference)
from ..core.mechanisms.solver import SolverFailure, check_versions
from ..core.mechanisms.values import ValueDistribution
from ..core.rng import stream

MASTER_SEED = 2026102316
HAND_TOL = 1e-8
IDENTITY_TOL, ZERO_PAYMENT = 1e-6, 1e-7  # relative; per request
MONOTONE_TOL, GAIN_TOL = 1e-9, 1e-6  # per request; total units
FULL = {"identity": 100, "incentives": 100, "runtime_repeats": 3}
QUICK = {"identity": 4, "incentives": 3, "runtime_repeats": 1}
REPORT = Path(__file__).resolve().parents[1] / "docs"
CODE = ["analysis/mechanism_validation.py", "core/mechanisms/allocation.py", "core/mechanisms/costs.py",
        "core/mechanisms/evaluate.py", "core/mechanisms/gsp.py", "core/mechanisms/market.py",
        "core/mechanisms/payments.py", "core/mechanisms/solver.py", "core/mechanisms/solver_config.json",
        "core/mechanisms/values.py"]
DET_SIGMA = 1e-3  # deterministic limit: score gaps ≥ 0.1, so logit errors are below e^-100


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in CODE:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest()


# ------------------------------------------------------------- market helpers
def offer(g: float = 0.0, u: float = 0.0, request: float = 0.0, conversion: float = 0.0,
          sponsored: tuple[float, float] | None = None) -> Offer:
    rho, zeta = sponsored or (request, conversion)
    return Offer(contribution=g, match=u, organic=Display(request=request, conversion=conversion),
                 sponsored=Display(request=rho, conversion=zeta))


def agent_market(market_id: str, scores: dict[str, dict[str, float]], rankings: list[tuple[list[str], float]],
                 advertisers: list[str], k: int, k_s: int, n: int = 1000, g: dict | None = None,
                 u: dict | None = None, label_effect: float = 0.0, states: list[State] | None = None,
                 **fields) -> Market:
    """One context, α = 1, one deterministic agent model; ``scores`` by state."""
    products = sorted(next(iter(scores.values())))
    g, u = g or {}, u or {}
    context = Context(context_id="x", probability=1.0,
                      offers={j: offer(g.get(j, 0.0), u.get(j, 0.0)) for j in products},
                      organic=[OrganicList(ranking=r, probability=p) for r, p in rankings],
                      agent_share=1.0, model_shares={"m": 1.0})
    model = AgentModel(key="m", sigma=DET_SIGMA, label_effect=label_effect, outside={"x": 0.0},
                       scores={s: {"x": v} for s, v in scores.items()})
    states = states or [State(state_id=s, probability=1.0 / len(scores)) for s in scores]
    return Market(market_id=market_id, requests=n, max_displayed=k, max_sponsored=k_s, advertisers=advertisers,
                  contexts=[context], states=states, position_weights=[0.1] * k, models=[model], **fields)


def _check(name: str, quantity: str, computed: float, expected: float, n: int) -> dict:
    error = abs(computed - expected) / n / max(1.0, abs(expected) / n)
    return {"market": name, "quantity": quantity, "computed": computed, "expected": expected,
            "error_per_request": error, "passed": bool(error <= HAND_TOL)}


# ------------------------------------------------------------- hand-solved
def hand_inclusion() -> list[dict]:
    """Example inclusion (l.339): K = 2, v_A > v_B > v_C > v_0, B included in every
    request under {A, B} or under {B, C}; it gets no choices in the first and all in
    the second."""
    out = []
    scores = {"default": {"A": 3.0, "B": 2.0, "C": 1.0}}
    for name, ranking, expected in (("inclusion {A,B}", ["A", "B", "C"], 0.0), ("inclusion {B,C}", ["B", "C", "A"], 1.0)):
        market = agent_market(name, scores, [(ranking, 1.0)], [], k=2, k_s=0)
        table = action_table(market)
        outcome = table.outcomes(np.ones(len(table.rows)))
        b = table.index["B"]
        inclusions = sum(1.0 for shown in table.displayed if "B" in shown)
        out += [_check(name, "X_B", market.requests * outcome["X"][b], market.requests * expected, market.requests),
                _check(name, "inclusions of B per request", inclusions, 1.0, 1)]
    return out


def gating_market(design: str, theta_f: float, theta_r: float) -> tuple[Market, dict]:
    """Prop gating (A1 §1.4): one context, α = 1, one deterministic model, K^s = 1 < K = 3,
    C ≡ 0, g ≡ g0, outside below every product, v_f + δ > v_r and v_r + δ > v_i."""
    scores = {"default": {"f": 2.0, "r": 1.5, "o1": 1.0, "o2": 0.8}}
    ranking = ["f", "r", "o1", "o2"] if design == "aware" else ["r", "o1", "o2", "f"]
    market = agent_market("gating %s θf=%g θr=%g" % (design, theta_f, theta_r), scores, [(ranking, 1.0)], ["f", "r"],
                          k=3, k_s=1, g={j: 0.1 for j in scores["default"]}, u={"f": 1.0, "r": 0.6, "o1": 0.2, "o2": 0.2},
                          label_effect=-0.2, guardrail=Guardrail(kappa=0.2))
    return market, {"f": theta_f, "r": theta_r}


def hand_gating() -> list[dict]:
    """(a) aware: payments 0, every request ends with f, M = M^ref = n u(f). (b) gated:
    revenue n ξ min{θ_f, θ_r}; M = n u(f) if θ_f > θ_r, n [u(f) − ξ (u(f) − u(r))]
    otherwise; ξ = min{1, κ u(f) / (u(f) − u(r))} = 0.5 here."""
    out, n, uf, ur, kappa = [], 1000, 1.0, 0.6, 0.2
    xi = min(1.0, kappa * uf / (uf - ur))
    for design, tf, tr in (("aware", 1.0, 0.6), ("gated", 1.0, 0.6), ("gated", 0.6, 1.0)):
        market, values = gating_market(design, tf, tr)
        outcome = MarketRunner(market).evaluate("M3", values)
        name = market.market_id
        if design == "aware":
            expected_revenue, expected_match, expected_xf = 0.0, n * uf, n
        else:
            expected_revenue = n * xi * min(tf, tr)
            expected_match = n * uf if tf > tr else n * (uf - xi * (uf - ur))
            expected_xf = n if tf > tr else n * (1 - xi)
        out += [_check(name, "revenue", outcome.revenue, expected_revenue, n),
                _check(name, "match quality M", outcome.match, expected_match, n),
                _check(name, "X_f", outcome.X["f"], expected_xf, n),
                _check(name, "M^ref", outcome.m_ref, n * uf, n)]
    return out


def setblind_market(n_a: int) -> tuple[Market, np.ndarray]:
    """Prop setblind(b) (A1 §1.6): advertisers 1..N_a, non-advertised o_1..o_N_a, K − 1 = 1
    filler; v_1 > v_o1 > v_2 > ... > v_oN > v_filler > v_0; R_i = (o_i, filler, ...) with
    i uniform; advertisers never organic; b_j = 1 / (N_a − j + 1); δ = 0; g constant."""
    ads = ["a%d" % j for j in range(1, n_a + 1)]
    others = ["o%d" % j for j in range(1, n_a + 1)]
    ladder = [p for pair in zip(ads, others) for p in pair] + ["filler"]
    scores = {"default": {p: 1.0 + 0.1 * (len(ladder) - i) for i, p in enumerate(ladder)}}
    rankings = [(["o%d" % i, "filler"] + [o for o in others if o != "o%d" % i] + ads, 1.0 / n_a)
                for i in range(1, n_a + 1)]
    market = agent_market("setblind N_a=%d" % n_a, scores, rankings, ads, k=2, k_s=1,
                          g={p: 0.1 for p in ladder})
    return market, np.array([1.0 / (n_a - j + 1) for j in range(1, n_a + 1)])


def hand_setblind(n_a: int = 3) -> list[dict]:
    """(V^aw − V^∅) / n = H_N / N, (V^bl − V^∅) / n = 1 / N, ratio 1 / H_N."""
    market, bids = setblind_market(n_a)
    n = market.requests
    harmonic = sum(1.0 / i for i in range(1, n_a + 1))
    aware = Program(market, "M3").solve(bids)
    blind = Program(market, "M3_set_blind").solve(bids)
    table = action_table(market)
    empty = np.array([1.0 if s == () else 0.0 for _, s in table.rows])
    v_empty = n * table.outcomes(empty)["contribution"]

    def welfare(allocation):  # Σ b_j X_j + Γ (C ≡ 0), totals
        return sum(b * allocation.X[j] for j, b in zip(market.advertisers, bids)) + allocation.contribution

    gap_aw, gap_bl = welfare(aware) - v_empty, welfare(blind) - v_empty
    name = market.market_id
    return [_check(name, "V^aw − V^∅", gap_aw, n * harmonic / n_a, n),
            _check(name, "V^bl − V^∅", gap_bl, n / n_a, n),
            _check(name, "(V^bl − V^∅) / (V^aw − V^∅) × n", n * gap_bl / gap_aw, n / harmonic, n)]


def quadratic_market(n: int = 1000, lam: float = 2.0, capacity: float = 600.0) -> Market:
    """Two advertisers A and B, K = K^s = 1, the organic list shows A; agents choose the
    displayed product; C_j(D) = λ D² / (2B) for both."""
    scores = {"default": {"A": 1.0, "B": 0.5}}
    return agent_market("quadratic two-advertiser", scores, [(["A", "B"], 1.0)], ["A", "B"], k=1, k_s=1, n=n,
                        g={"A": 1.0, "B": 1.0}, u={"A": 1.0, "B": 1.0},
                        costs={j: Cost(kind="quadratic", weight=lam, capacity=capacity) for j in "AB"})


def hand_quadratic(b_a: float = 1.0, b_b: float = 1.3) -> list[dict]:
    """M4 with y = Pr(show B): per request W(y) = b_A (1 − y) + b_B y + g − E C / n with
    E C / n = λ / (2B) [(1 − y) + (n − 1)(1 − y)² + y + (n − 1) y²], so
    y* = clip(1/2 + B (b_B − b_A) / (2 λ (n − 1))); payments by Prop ic(c)."""
    n, lam, cap = 1000, 2.0, 600.0
    market = quadratic_market(n, lam, cap)

    def w(y, a, b):
        return a * (1 - y) + b * y + 1.0 - lam / (2 * cap) * ((1 - y) + (n - 1) * (1 - y) ** 2 + y + (n - 1) * y * y)

    def y_star(a, b):
        return min(1.0, max(0.0, 0.5 + cap * (b - a) / (2 * lam * (n - 1))))

    y = y_star(b_a, b_b)
    expected_pb = n * (w(y_star(b_a, 0.0), b_a, 0.0) - (w(y, b_a, b_b) - b_b * y))
    expected_pa = n * (w(y_star(0.0, b_b), 0.0, b_b) - (w(y, b_a, b_b) - b_a * (1 - y)))
    runner = MarketRunner(market)
    outcome = runner.evaluate("M4", {"A": b_a, "B": b_b})
    integral_b = pay.integral_payment(runner.program("M4"), np.array([b_a, b_b]), 1, pay.BidMap()).payment
    name = market.market_id
    return [_check(name, "X_B", outcome.X["B"], n * y, n), _check(name, "P_A (externality)", outcome.payments["A"],
                                                               expected_pa, n),
            _check(name, "P_B (externality)", outcome.payments["B"], expected_pb, n),
            _check(name, "P_B (eq. payment)", float(integral_b), expected_pb, n),
            _check(name, "E C_A + E C_B", sum(outcome.cost.values()),
                   lam / (2 * cap) * (n * (1 - y) + n * (n - 1) * (1 - y) ** 2 + n * y + n * (n - 1) * y * y), n)]


def reserve_market(values: ValueDistribution) -> Market:
    """One advertiser A and one non-advertiser O, K = K^s = 1, the organic list shows O,
    agents choose the displayed product, equal g: the profit rule sponsors A iff
    φ̄_A(b) > 0 (Myerson reserve)."""
    scores = {"default": {"A": 2.0, "O": 1.0}}
    return agent_market("reserve %s" % values.kind, scores, [(["O", "A"], 1.0)], ["A"], k=1, k_s=1,
                        g={"A": 0.5, "O": 0.5}, values={"A": values})


def hand_reserve() -> list[dict]:
    """Uniform [0, 1]: φ(b) = 2b − 1, reserve 1/2, so at b = 0.8 X_A = n and P_A = n/2,
    at b = 0.3 X_A = 0. Lognormal (μ = 0, s = 2; ironed, φ̄ < 0 on the ironing
    interval): the reserve r solves r = (1 − F(r)) / f(r), found here by root-finding on
    the closed-form φ; at b = 2r, P_A = n r."""
    out, n = [], 1000
    uniform = ValueDistribution(kind="uniform", upper=1.0)
    runner = MarketRunner(reserve_market(uniform))
    for bid, x, p in ((0.8, n, n / 2), (0.3, 0.0, 0.0)):
        o = runner.evaluate("profit_rule", {"A": bid})
        out += [_check("reserve uniform b=%g" % bid, "X_A", o.X["A"], x, n),
                _check("reserve uniform b=%g" % bid, "P_A", o.payments["A"], p, n)]
    o = runner.evaluate("M3", {"A": 0.8})
    out.append(_check("reserve uniform, surplus rule b=0.8", "P_A (no competitor)", o.payments["A"], 0.0, n))
    lognormal = ValueDistribution(kind="lognormal", mu=0.0, s=2.0)
    dist = stats.lognorm(2.0, scale=1.0)
    r = optimize.brentq(lambda t: t - dist.sf(t) / dist.pdf(t), 1.0, 1e4, xtol=1e-14)
    o = MarketRunner(reserve_market(lognormal)).evaluate("profit_rule", {"A": 2 * r})
    out += [_check("reserve lognormal s=2 b=2r", "X_A", o.X["A"], n, n),
            _check("reserve lognormal s=2 b=2r", "P_A = n r", o.payments["A"], n * r, n)]
    return out


def gsp_market() -> tuple[Market, dict]:
    """One context, α = 0.5, one model with χ = 0.6, K = 3, K^s = 2; advertisers A, B, C
    with ρ(·, 1) = (0.5, 0.4, 0.3), ζ(·, 1) = (0.4, 0.5, 0.6), θ = (1.0, 0.9, 0.8); γ =
    (0.4, 0.3, 0.2); click bids by the human conversion rate, b^c = θ ζ = (0.4, 0.45, 0.48)."""
    rho, zeta = {"A": 0.5, "B": 0.4, "C": 0.3, "O": 0.5}, {"A": 0.4, "B": 0.5, "C": 0.6, "O": 0.5}
    scores = {"A": 1.0, "B": 0.9, "C": 0.8, "O": 1.2}
    context = Context(context_id="x", probability=1.0,
                      offers={j: offer(0.1, 0.5, 0.5, 0.5, sponsored=(rho[j], zeta[j])) for j in scores},
                      organic=[OrganicList(ranking=["O", "A", "B", "C"], probability=1.0)], agent_share=0.5,
                      model_shares={"m": 1.0})
    model = AgentModel(key="m", sigma=0.5, page_request_prob=0.6, outside={"x": 0.0},
                       scores={"default": {"x": scores}})
    market = Market(market_id="GSP hand", requests=1000, max_displayed=3, max_sponsored=2,
                    advertisers=["A", "B", "C"], contexts=[context], position_weights=[0.4, 0.3, 0.2],
                    models=[model])
    return market, {"A": 1.0, "B": 0.9, "C": 0.8}


def hand_gsp() -> list[dict]:
    """M2 ranks by b^c γ_k ρ (separable): A then B, next prices 0.072 / 0.2 = 0.36 and
    0.0432 / 0.12 = 0.36 per human click, charged clicks (1 − α) γ_k ρ = 0.1 and 0.06.
    M1 ranks by b^c [(1 − α) γ_k ρ + α χ̄] with χ̄ = 0.6: slot 1 rates (0.40, 0.38, 0.36),
    scores (0.16, 0.171, 0.1728), so C wins at 0.171 / 0.36; slot 2 rates (0.375, 0.36),
    scores (0.15, 0.162), B wins at 0.15 / 0.36; every request is charged."""
    market, values = gsp_market()
    n = market.requests
    runner = MarketRunner(market)
    out = []
    m2 = runner.evaluate("M2", values, bidding_rule="human_conversion_rate")
    out += [_check("GSP M2", "P_A", m2.payments["A"], n * 0.36 * 0.1, n),
            _check("GSP M2", "P_B", m2.payments["B"], n * 0.36 * 0.06, n),
            _check("GSP M2", "P_C", m2.payments["C"], 0.0, n)]
    result = gsp.auction(market, "M2", gsp.human_rate_bids(market, values))
    out.append({"market": "GSP M2", "quantity": "sponsored list", "computed": list(result.sponsored["x"]),
                "expected": ["A", "B"], "error_per_request": 0.0, "passed": result.sponsored["x"] == ("A", "B")})
    m1 = runner.evaluate("M1", values, bidding_rule="human_conversion_rate")
    out += [_check("GSP M1", "P_C", m1.payments["C"], n * (0.171 / 0.36) * 0.36, n),
            _check("GSP M1", "P_B", m1.payments["B"], n * (0.15 / 0.36) * 0.36, n),
            _check("GSP M1", "P_A", m1.payments["A"], 0.0, n)]
    result = gsp.auction(market, "M1", gsp.human_rate_bids(market, values))
    out.append({"market": "GSP M1", "quantity": "sponsored list", "computed": list(result.sponsored["x"]),
                "expected": ["C", "B"], "error_per_request": 0.0, "passed": result.sponsored["x"] == ("C", "B")})
    return out


def threshold_market(n: int = 1000, lam: float = 1.0, capacity: float = 300.0) -> Market:
    """Prop threshold (a) with Δ = 0: K = 2 lists from {1, 2, 3}; two equal states with
    v_1 > v_2 > v_3 > v_0 and v_2 > v_1 > v_3 > v_0; g ≡ 1 (so Δ = 0); quadratic cost on
    the leaders 1 and 2, none on 3. All three are advertisers with zero bids, so the
    sponsored policy reaches every set; the organic list is (3, 1, 2)."""
    scores = {"s1": {"1": 3.0, "2": 2.0, "3": 1.0}, "s2": {"1": 2.0, "2": 3.0, "3": 1.0}}
    return agent_market("threshold quadratic Δ=0", scores, [(["3", "1", "2"], 1.0)], ["1", "2", "3"], k=2, k_s=2,
                        n=n, g={"1": 1.0, "2": 1.0, "3": 1.0}, u={"1": 1.0, "2": 1.0, "3": 1.0},
                        costs={j: Cost(kind="quadratic", weight=lam, capacity=capacity) for j in "12"})


def hand_threshold() -> list[dict]:
    """y* = min{1, Δ B / (λ (n − 1))} = 0 and J(0) = n ν − λ n (n + 1) / (4B)."""
    n, lam, cap = 1000, 1.0, 300.0
    market = threshold_market(n, lam, cap)
    outcome = MarketRunner(market).evaluate("M4", {"1": 0.0, "2": 0.0, "3": 0.0})
    name = market.market_id
    table = action_table(market)
    allocation = MarketRunner(market).program("M4").solve(np.zeros(3))
    both = sum(table.weight[r] * allocation.y[r] for r, shown in enumerate(table.displayed) if set(shown) == {"1", "2"})
    return [_check(name, "G = J(y*)", outcome.operating_value, n * 1.0 - lam * n * (n + 1) / (4 * cap), n),
            _check(name, "y* = Pr({1, 2})", n * both, 0.0, n),
            _check(name, "X_1", outcome.X["1"], n / 2, n), _check(name, "X_2", outcome.X["2"], n / 2, n)]


def hand_solved() -> list[dict]:
    return (hand_inclusion() + hand_gating() + hand_setblind() + hand_quadratic() + hand_reserve() + hand_gsp()
            + hand_threshold())


# ------------------------------------------------------------- random markets
def random_market(index: int, cost_kind: str | None, n: int = 100) -> tuple[Market, dict[str, float]]:
    """A random market from stream(MASTER_SEED, "mechanism-market", index): N = 6, four
    advertisers, two contexts with |E(x)| = 5 and two organic rankings each, K = 3,
    K^s ∈ {1, 2}, two agent models with states and position and label effects, humans
    with page requests, the guardrail set so that no sponsorship meets it with 2% slack,
    costs of ``cost_kind`` on every product, values uniform or lognormal (s ∈ {0.5, 2},
    the latter ironed) and θ drawn from them."""
    rng = stream(MASTER_SEED, "mechanism-market", index)
    k = 3
    k_s = int(rng.integers(1, 3))
    products = ["P%d" % i for i in range(6)]
    states = [State(state_id="s%d" % s, probability=float(p)) for s, p in enumerate(rng.dirichlet([2, 2]))]
    gamma = sorted(rng.uniform(0.15, 0.4, k).tolist(), reverse=True)
    contexts = []
    for c in range(2):
        eligible = sorted(rng.choice(products, size=5, replace=False).tolist())
        offers = {j: offer(float(rng.uniform(0, 1)), float(rng.uniform(0.1, 1)), float(rng.uniform(0.2, 1)),
                           float(rng.uniform(0.1, 1)), sponsored=(float(rng.uniform(0.2, 1)), float(rng.uniform(0.1, 1))))
                  for j in eligible}
        shares = rng.dirichlet([2, 2])
        organic = [OrganicList(ranking=rng.permutation(eligible).tolist(), probability=float(p))
                   for p in rng.dirichlet([2, 2])]
        contexts.append(Context(context_id="x%d" % c, probability=[0.6, 0.4][c], offers=offers, organic=organic,
                                agent_share=float(rng.uniform(0, 1)),
                                model_shares={"m0": float(shares[0]), "m1": float(shares[1])}))
    models = [AgentModel(key="m%d" % m, sigma=float(rng.uniform(0.2, 1.0)), label_effect=float(rng.uniform(-1, 0)),
                         position_effects=sorted(rng.uniform(-0.3, 0, k).tolist(), reverse=True),
                         page_request_prob=float(rng.uniform(0, 1)),
                         outside={c.context_id: float(rng.normal(-0.5, 0.5)) for c in contexts},
                         scores={s.state_id: {c.context_id: {j: float(rng.normal()) for j in c.offers} for c in contexts}
                                 for s in states}) for m in range(2)]
    union = sorted({j for c in contexts for j in c.offers})
    advertisers = [j for j in products[:4] if j in union]
    costs = {j: Cost(kind=cost_kind, weight=float(rng.uniform(0.5, 2)), capacity=float(rng.uniform(0.15, 0.5)) * n)
             for j in union} if cost_kind else {}
    values = {}
    for j in advertisers:
        if rng.random() < 0.5:
            values[j] = ValueDistribution(kind="uniform", upper=1.0)
        else:
            values[j] = ValueDistribution(kind="lognormal", mu=-0.5, s=float(rng.choice([0.5, 2.0])))
    market = _with_guardrail(Market(market_id="random-%d" % index, requests=n, max_displayed=k, max_sponsored=k_s,
                                    advertisers=advertisers, contexts=contexts, states=states, position_weights=gamma,
                                    models=models, costs=costs, values=values))
    theta = {j: float(values[j].sample(rng)) for j in advertisers}
    return market, theta


def _cost_kind(index: int) -> str:
    return "quadratic" if index % 2 == 0 else "overflow"


def _identity_market(index: int) -> dict:
    market, theta = random_market(index, _cost_kind(index))
    runner = MarketRunner(market)
    bids = np.array([theta[j] for j in market.advertisers])
    rows, start = [], time.perf_counter()
    try:
        for rule in ("M3", "M3_set_blind", "M4"):
            program = runner.program(rule)
            ext, _ = pay.externality_payments(program, bids)
            for k, j in enumerate(market.advertisers):
                p = pay.integral_payment(program, bids, k, pay.BidMap())
                scale = max(abs(ext[k]), abs(p.payment), ZERO_PAYMENT * market.requests)
                rows.append({"rule": rule, "advertiser": j, "externality": float(ext[k]), "integral": p.payment,
                             "relative_error": abs(ext[k] - p.payment) / scale,
                             "zero": bool(max(abs(ext[k]), abs(p.payment)) < ZERO_PAYMENT * market.requests),
                             "solves": p.profile.solves if p.profile else 0,
                             "unresolved": len(p.profile.unresolved) if p.profile else 0})
    except SolverFailure as failure:
        return {"index": index, "excluded": {"status": failure.status, "where": failure.where}}
    return {"index": index, "cost": _cost_kind(index), "rows": rows, "seconds": time.perf_counter() - start}


def _incentive_market(index: int) -> dict:
    market, theta = random_market(index + 1000, _cost_kind(index))
    runner = MarketRunner(market)
    truth = np.array([theta[j] for j in market.advertisers])
    n, rows, start = market.requests, [], time.perf_counter()
    try:
        for rule in ("M3", "M4", "profit_rule"):
            program = runner.program(rule)
            for k, j in enumerate(market.advertisers):
                top = max(2 * truth[k], market.values[j].ppf(0.99))
                grid = np.unique(np.r_[np.linspace(0.0, top, 21), truth[k]])
                x, paid, prof = [], [], None
                bm = pay.bid_map(program, k)
                for bid in grid:
                    bids = truth.copy()
                    bids[k] = bid
                    if rule == "profit_rule":
                        p = pay.integral_payment(program, bids, k, bm, upper=float(grid[-1]), prof=prof)
                        prof = p.profile
                        x.append(p.X)
                        paid.append(p.payment)
                    else:
                        ext, allocation = pay.externality_payments(program, bids)
                        x.append(allocation.X[j])
                        paid.append(float(ext[k]))
                x, paid = np.array(x), np.array(paid)
                utility = truth[k] * x - paid
                truthful = utility[int(np.searchsorted(grid, truth[k]))]
                rows.append({"rule": rule, "advertiser": j, "values": market.values[j].kind,
                             "ironed": bool(market.values[j].ironing), "theta": float(truth[k]),
                             "min_increment_per_request": float(np.diff(x).min() / n),
                             "max_gain": float(utility.max() - truthful),
                             "floor_gap": pay.floor_gap(program, truth, k) if rule == "profit_rule" else 0.0})
    except SolverFailure as failure:
        return {"index": index, "excluded": {"status": failure.status, "where": failure.where}}
    return {"index": index, "cost": _cost_kind(index), "rows": rows, "seconds": time.perf_counter() - start}


# ----------------------------------------------------------------- parallel map
WORKERS = 1


def _map(function, arguments: list, label: str) -> list:
    start, out = time.time(), []
    if WORKERS <= 1:
        for count, a in enumerate(arguments, 1):
            out.append(function(a))
            _progress(label, count, len(arguments), start)
        return out
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        for count, result in enumerate(pool.map(function, arguments), 1):
            out.append(result)
            _progress(label, count, len(arguments), start)
    return out


def _progress(label: str, done: int, total: int, start: float) -> None:
    if done % max(1, total // 10) == 0 or done == total:
        print("%s %d/%d (%.0fs)" % (label, done, total, time.time() - start), flush=True)


# ------------------------------------------------------------------ studies
def identity_study(markets: int) -> dict:
    results = _map(_identity_market, list(range(markets)), "identity")
    rows = [r for res in results if "rows" in res for r in res["rows"]]
    excluded = [{"index": res["index"], **res["excluded"]} for res in results if "excluded" in res]
    by_rule = {}
    for rule in ("M3", "M3_set_blind", "M4"):
        sub = [r for r in rows if r["rule"] == rule]
        by_rule[rule] = {"payments": len(sub), "zero_payments": sum(r["zero"] for r in sub),
                         "max_relative_error": max((r["relative_error"] for r in sub), default=0.0),
                         "median_solves": float(np.median([r["solves"] for r in sub])) if sub else 0.0,
                         "unresolved_intervals": sum(r["unresolved"] for r in sub)}
    worst = max(rows, key=lambda r: r["relative_error"]) if rows else None
    return {"markets": markets, "valid_markets": markets - len(excluded), "excluded": excluded, "by_rule": by_rule,
            "worst": worst, "tolerance": IDENTITY_TOL,
            "seconds_per_market": float(np.mean([res["seconds"] for res in results if "seconds" in res])),
            "passed": bool(rows) and not excluded and all(r["relative_error"] <= IDENTITY_TOL for r in rows)}


def incentive_study(markets: int) -> dict:
    results = _map(_incentive_market, list(range(markets)), "incentives")
    rows = [r for res in results if "rows" in res for r in res["rows"]]
    excluded = [{"index": res["index"], **res["excluded"]} for res in results if "excluded" in res]
    by_rule = {}
    for rule in ("M3", "M4", "profit_rule"):
        sub = [r for r in rows if r["rule"] == rule]
        by_rule[rule] = {"advertisers": len(sub), "ironed": sum(r["ironed"] for r in sub),
                         "min_increment_per_request": min((r["min_increment_per_request"] for r in sub), default=0.0),
                         "max_gain": max((r["max_gain"] for r in sub), default=0.0),
                         "max_floor_gap": max((r["floor_gap"] for r in sub), default=0.0)}
    passed = (bool(rows) and not excluded and all(r["min_increment_per_request"] >= -MONOTONE_TOL for r in rows)
              and all(r["max_gain"] <= GAIN_TOL for r in rows))
    return {"markets": markets, "valid_markets": markets - len(excluded), "excluded": excluded, "by_rule": by_rule,
            "monotone_tolerance": MONOTONE_TOL, "gain_tolerance": GAIN_TOL, "grid_bids": 22,
            "seconds_per_market": float(np.mean([res["seconds"] for res in results if "seconds" in res])),
            "passed": passed}


def exclusion_study() -> dict:
    """A batch of two random markets and one whose guardrail cannot be met (M^ref set
    above any attainable match quality): the infeasible market is excluded from every
    mechanism and counted; the others have one outcome per mechanism."""
    feasible = [random_market(2000 + i, "quadratic") for i in range(2)]
    market, theta = random_market(2002, "quadratic")
    impossible = market.model_copy(update={"market_id": "impossible-guardrail",
                                           "guardrail": Guardrail(kappa=0.0, m_ref=10.0 * market.requests)})
    impossible = Market.model_validate(impossible.model_dump())
    batch = evaluate_markets(feasible + [(impossible, theta)], mechanisms=MECHANISMS)
    ids = sorted({o.market_id for o in batch.outcomes})
    return {"markets": 3, "excluded": batch.exclusions.entries, "excluded_count": batch.exclusions.count,
            "outcome_markets": ids, "outcomes": len(batch.outcomes),
            "passed": batch.exclusions.count == 1 and "impossible-guardrail" not in ids
            and len(batch.outcomes) == 2 * len(MECHANISMS)}


def sized_market(index: int, contexts: int, eligible: int, k: int, k_s: int, rankings: int, states: int,
                 models: int = 2, advertisers: int | None = None, n: int = 1000) -> tuple[Market, dict]:
    rng = stream(MASTER_SEED, "runtime-market", index, contexts, eligible, k, k_s, rankings, states)
    products = ["P%d" % i for i in range(eligible + 2)]
    advertisers = advertisers or eligible - 2
    state_list = [State(state_id="s%d" % s, probability=1.0 / states) for s in range(states)]
    gamma = sorted(rng.uniform(0.05, 0.2, k).tolist(), reverse=True)
    context_list = []
    for c in range(contexts):
        chosen = sorted(rng.choice(products, size=eligible, replace=False).tolist())
        offers = {j: offer(float(rng.uniform(0, 1)), float(rng.uniform(0.1, 1)), float(rng.uniform(0.2, 1)),
                           float(rng.uniform(0.1, 1)), sponsored=(float(rng.uniform(0.2, 1)), float(rng.uniform(0.1, 1))))
                  for j in chosen}
        organic = [OrganicList(ranking=rng.permutation(chosen).tolist(), probability=1.0 / rankings)
                   for _ in range(rankings)]
        context_list.append(Context(context_id="x%d" % c, probability=1.0 / contexts, offers=offers, organic=organic,
                                    agent_share=0.5, model_shares={"m%d" % m: 1.0 / models for m in range(models)}))
    model_list = [AgentModel(key="m%d" % m, sigma=0.3, label_effect=-0.5,
                             position_effects=[-0.05 * i for i in range(k)], page_request_prob=0.5,
                             outside={c.context_id: -1.0 for c in context_list},
                             scores={s.state_id: {c.context_id: {j: float(rng.normal()) for j in c.offers}
                                                  for c in context_list} for s in state_list}) for m in range(models)]
    union = sorted({j for c in context_list for j in c.offers})
    ads = union[:advertisers]
    market = Market(market_id="runtime-%d" % index, requests=n, max_displayed=k, max_sponsored=k_s, advertisers=ads,
                    contexts=context_list, states=state_list, position_weights=gamma, models=model_list,
                    costs={j: Cost(kind="quadratic", weight=1.0, capacity=0.3 * n) for j in union},
                    values={j: ValueDistribution(kind="uniform", upper=1.0) for j in ads})
    market = _with_guardrail(market)
    theta = {j: float(rng.uniform(0, 1)) for j in ads}
    return market, theta


def _with_guardrail(market: Market) -> Market:
    """The guardrail with κ such that no sponsorship meets it with 2% slack."""
    table = action_table(market)
    none = market.requests * float(sum(table.weight[r] * table.match[r] for r, (_, s) in enumerate(table.rows)
                                       if s == ()))
    m_ref = match_reference(market)
    return Market.model_validate(market.model_copy(update={"guardrail": Guardrail(
        kappa=max(0.0, 1 - 0.98 * none / m_ref), m_ref=m_ref)}).model_dump())


RUNTIME_SIZES = [
    {"label": "small", "contexts": 1, "eligible": 6, "k": 3, "k_s": 1, "rankings": 1, "states": 1},
    {"label": "E2 baseline (K = 4, K^s = 1)", "contexts": 1, "eligible": 10, "k": 4, "k_s": 1, "rankings": 2,
     "states": 2},
    {"label": "E2 K^s = 2", "contexts": 1, "eligible": 10, "k": 4, "k_s": 2, "rankings": 2, "states": 2},
    {"label": "E2 K = 8", "contexts": 1, "eligible": 10, "k": 8, "k_s": 1, "rankings": 2, "states": 2},
    {"label": "E2, 4 contexts", "contexts": 4, "eligible": 10, "k": 4, "k_s": 1, "rankings": 2, "states": 2},
]


def _runtime_case(case: tuple[int, dict]) -> dict:
    repeat, size = case
    fields = {k: v for k, v in size.items() if k != "label"}
    market, theta = sized_market(repeat, **fields)
    start = time.perf_counter()
    _ = match_reference(market.model_copy(update={"guardrail": None}))  # timed: what enumerating M^ref costs
    runner = MarketRunner(market)
    setup = time.perf_counter() - start
    seconds = {}
    for mechanism in MECHANISMS:
        t0 = time.perf_counter()
        runner.evaluate(mechanism, theta)
        seconds[mechanism] = time.perf_counter() - t0
    return {"label": size["label"], "repeat": repeat, "actions": len(runner.table.rows), "setup": setup,
            "seconds": seconds}


def runtime_study(repeats: int) -> dict:
    cases = [(r, size) for size in RUNTIME_SIZES for r in range(repeats)]
    results = _map(_runtime_case, cases, "runtime")
    table = []
    for size in RUNTIME_SIZES:
        sub = [r for r in results if r["label"] == size["label"]]
        table.append({**size, "actions": sub[0]["actions"], "setup": float(np.mean([r["setup"] for r in sub])),
                      "seconds": {m: float(np.mean([r["seconds"][m] for r in sub])) for m in MECHANISMS},
                      "total": float(np.mean([r["setup"] + sum(r["seconds"].values()) for r in sub]))})
    baseline = table[1]
    configurations, markets = 5 * 3 * 2, 200
    core = configurations * markets * baseline["total"]
    return {"repeats": repeats, "sizes": table, "workers_used": WORKERS,
            "sl19_core": {"configurations": configurations, "markets_per_configuration": markets,
                          "seconds_per_market_all_mechanisms": baseline["total"], "cpu_hours": core / 3600}}


# ------------------------------------------------------------------- report
def run(counts: dict) -> dict:
    start = time.time()
    versions = check_versions()
    hand = hand_solved()
    runtime, exclusions = runtime_study(counts["runtime_repeats"]), exclusion_study()  # the short parts first
    out = {"study": "SL-16 mechanism validation", "master_seed": MASTER_SEED, "code_fingerprint": code_fingerprint(),
           "counts": counts, "versions": versions, "hand_solved": hand,
           "identity": identity_study(counts["identity"]), "incentives": incentive_study(counts["incentives"]),
           "exclusions": exclusions, "runtime": runtime}
    out["seconds"] = round(time.time() - start, 1)
    markets = {h["market"] for h in hand}
    out["passed"] = {"AC-1": len(markets) >= 5 and all(h["passed"] for h in hand),
                     "AC-2": out["identity"]["passed"], "AC-3": out["incentives"]["passed"],
                     "AC-4": out["exclusions"]["passed"] and not out["identity"]["excluded"]
                     and not out["incentives"]["excluded"],
                     "AC-5": bool(out["runtime"]["sizes"])}
    return out


def _fmt(value) -> str:
    return ", ".join(value) if isinstance(value, list) else "%.10g" % value


def markdown(report: dict) -> str:
    hand = report["hand_solved"]
    lines = ["# SL-16 mechanism validation", "",
             "Generated by `python -m features.storelab.analysis.mechanism_validation` (master seed %d, code "
             "fingerprint `%s`, %.0f s). Solvers and versions: %s (core/mechanisms/solver_config.json)." % (
                 report["master_seed"], report["code_fingerprint"][:12], report["seconds"],
                 ", ".join("%s %s" % kv for kv in sorted(report["versions"].items()))), "",
             "## Hand-solved markets (AC-1)", "",
             "%d markets, %d checks; tolerance %g per request (|computed − expected| / n ≤ %g max(1, |expected| / n)). "
             "Deterministic agents use σ = %g with score gaps ≥ 0.1." % (
                 len({h["market"] for h in hand}), len(hand), HAND_TOL, HAND_TOL, DET_SIGMA), "",
             "| Market | Quantity | Computed | Expected | Error / request | |", "|---|---|---:|---:|---:|---|"]
    for h in hand:
        lines.append("| %s | %s | %s | %s | %.1e | %s |" % (h["market"], h["quantity"], _fmt(h["computed"]),
                                                          _fmt(h["expected"]), h["error_per_request"],
                                                          "pass" if h["passed"] else "FAIL"))
    ident = report["identity"]
    lines += ["", "## Payment identity (AC-2)", "",
              "%d random markets (%d valid; costs alternate quadratic and overflow), every advertiser, surplus rules. "
              "Relative error |P_ext − P_int| / max(|P_ext|, |P_int|, n·%g); tolerance %g: %s." % (
                  ident["markets"], ident["valid_markets"], ZERO_PAYMENT, ident["tolerance"],
                  "passed" if ident["passed"] else "FAILED"), "",
              "| Rule | Payments | Zero payments | Max relative error | Median solves per payment | Unresolved intervals |",
              "|---|---:|---:|---:|---:|---:|"]
    for rule, r in ident["by_rule"].items():
        lines.append("| %s | %d | %d | %.1e | %.0f | %d |" % (rule, r["payments"], r["zero_payments"],
                                                           r["max_relative_error"], r["median_solves"],
                                                           r["unresolved_intervals"]))
    inc = report["incentives"]
    lines += ["", "## Incentives (AC-3)", "",
              "%d random markets (%d valid), every advertiser, %d grid bids each. X_j nondecreasing to %g per request; "
              "largest gain from a grid bid over truthful bidding at most %g: %s." % (
                  inc["markets"], inc["valid_markets"], inc["grid_bids"], inc["monotone_tolerance"],
                  inc["gain_tolerance"], "passed" if inc["passed"] else "FAILED"), "",
              "| Rule | Advertisers | Ironed values | Smallest X increment / request | Largest gain | Floor gap |",
              "|---|---:|---:|---:|---:|---:|"]
    for rule, r in inc["by_rule"].items():
        lines.append("| %s | %d | %d | %.1e | %.1e | %.1e |" % (rule, r["advertisers"], r["ironed"],
                                                               r["min_increment_per_request"], r["max_gain"],
                                                               r["max_floor_gap"]))
    ex = report["exclusions"]
    lines += ["", "## Solver exclusions (AC-4)", "",
              "Accepted status: `optimal` only (QPs must also pass the active-set polish). Excluded in the identity study: "
              "%d; in the incentive study: %d. Exclusion check: %d markets, %d excluded (%s), outcomes for %s only (%d "
              "outcomes): %s." % (len(ident["excluded"]), len(inc["excluded"]), ex["markets"], ex["excluded_count"],
                                  "; ".join("%s %s" % (e["market_id"], e["status"]) for e in ex["excluded"]),
                                  ", ".join(ex["outcome_markets"]), ex["outcomes"],
                                  "passed" if ex["passed"] else "FAILED"), "",
              "## Runtime (AC-5)", "",
              "Mean seconds per market (%d repeats, one process per market). Setup: the action table and M^ref (by "
              "enumeration of the organic lists). Mechanisms: allocation and payments (externality form for "
              "the surplus rules, eq. payment for the profit rule) and the fixed-point click bids of M1 and M2." % (
                  report["runtime"]["repeats"]), "",
              "| Market | Actions | Setup | " + " | ".join(MECHANISMS) + " | Total |",
              "|---|---:|---:|" + "---:|" * len(MECHANISMS) + "---:|"]
    for size in report["runtime"]["sizes"]:
        lines.append("| %s | %d | %.3f | %s | %.2f |" % (size["label"], size["actions"], size["setup"],
                                                        " | ".join("%.3f" % size["seconds"][m] for m in MECHANISMS),
                                                        size["total"]))
    core = report["runtime"]["sl19_core"]
    lines += ["", "SL-19 sizing: the core factorial has %d configurations × %d markets; at %.2f s per E2-baseline market "
              "for all six mechanisms that is %.1f CPU hours (divide by the number of worker processes)." % (
                  core["configurations"], core["markets_per_configuration"], core["seconds_per_market_all_mechanisms"],
                  core["cpu_hours"]), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.mechanism_validation")
    parser.add_argument("--quick", action="store_true", help="a few markets (code check, not the study)")
    parser.add_argument("--out", type=Path, help="output directory (default: features/storelab/docs)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes (default: all cores)")
    args = parser.parse_args(argv)
    global WORKERS
    WORKERS = args.workers
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    report = run(QUICK if args.quick else FULL)
    out = args.out or REPORT
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, sort_keys=True, default=_json_default)
    (out / "mechanism_validation.json").write_text(text + "\n", encoding="utf-8")
    (out / "mechanism_validation.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["passed"]))
    return 0


def _json_default(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value))


if __name__ == "__main__":
    raise SystemExit(main())
