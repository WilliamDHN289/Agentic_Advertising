"""E2 markets on the stipulated parameters of A17 (SL-19S; reused by SL-19 on E1 parameters).

``requirements/A17_stipulated_parameters.md`` §1–§2 fixes the baseline (``Params``) and the
sensitivity levels (``LEVELS``); ``configurations`` lists the runs:

- **core**: α × organic regime × organic variation at the baseline (30 configurations);
- **sensitivity**: every level, one factor at a time, over α in the agent-aware regime with
  no organic variation (owner decision 2026-10-23);
- **part3**: the levels for the number of models and versions in the human-optimised regime
  (H2a part 3, owner decision 2026-10-23).

A market is drawn by ``market(config, index)`` from ``stream(MASTER_SEED, "sl19s-market",
config_id, index)``. The market inputs are therefore fixed before any mechanism runs, and every
mechanism sees the same ``Market``.

Its agent models are built as an SL-06 ``AgentParams`` (``source: stipulated``), the form
SL-19 reads from ``agent_params_e1.json``, and converted by ``models_from_agent_params``.

Draws, in order: qualities z_j; the 6 advertisers; model noise e_mj; version shifts η_mj (when
the level has versions); contributions g_j (when random); values θ_j.

**Lists** (A17 §2, A1 §1.1 and §E7):
- **human-optimised**: E(x) by u.
- **agent-aware**: the favourites F(x) first (by u), then the rest by u. The market is
  *aware-infeasible* if |F(x)| > K − K^s.
- **gated**: each favourite's second-ranked product for its type in the first K − K^s
  positions (by u, as far as they fit), then the other non-favourites by u, then the
  favourites. The market is *gated-infeasible* when the non-favourites cannot fill K
  positions, because a favourite would then be displayed organically.
- **randomised variation**: R′ swaps the unconstrained products of positions 1..K − K^s with
  those of positions K − K^s + 1..2(K − K^s). Favourites in agent-aware lists and second-ranked
  rivals in gated lists stay.

**Guardrail floor** (owner decision 2026-10-23). Under A17's κ, many gated and
randomised-list markets miss (1 − κ) M^ref before any sponsorship, so eq. mq-constraint
leaves no feasible policy. The paper assumes some sponsored policy is feasible. In the
core, sensitivity and part-3 markets the feasible set is therefore
M(Q) ≥ T = min{(1 − κ) M^ref, M(Q^o, ∅)}: sponsorship may not lower match quality below
(1 − κ) M^ref, nor below the organic design's own level when that is already lower.
No sponsorship is always feasible. Where the floor binds, T is set 1e-8 per request
below M(Q^o, ∅). Without that slack, no sponsorship would lie exactly on the boundary of
a feasible set with no interior, which interior-point solvers cannot handle. The result
files record M^ref, T and whether the floor binds. SL-16's ``Guardrail`` is passed m_ref = T / (1 − κ), so its constraint is M ≥ T. The
Prop gating and Conj gating instances keep the paper's guardrail, from which ξ is
derived.

**Screen** (A1 §E5): every positive-probability type whose favourite is in E(x) must satisfy
the labelled-favourite condition, v_f + δ_m > v₀ and v_f + δ_m > v_i for every other eligible
i. The excess is e(x) = |F(x)| − (K − K^s).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..core.agent_params import AgentParams, ModelParams, StateParams
from ..core.mechanisms.market import (Context, Cost, Display, Guardrail, Market, Offer, OrganicList,
                                      action_table, match_reference, models_from_agent_params)
from ..core.mechanisms.values import ValueDistribution
from ..core.rng import stream

MASTER_SEED = 2026102319
TASK = "x"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
REGIMES = ("human_optimised", "agent_aware", "gated")
VARIATIONS = ("none", "randomised")
MARKETS = 200
FLOOR_SLACK = 1e-8  # per request, below M(Q^o, ∅) where the floor binds
DETERMINISTIC_GAP_SIGMAS = 50.0  # the σ → 0 limit: σ = (smallest score gap) / 50, logit error < e^-50


@dataclass(frozen=True)
class Params:
    """A17 §1–§2 baseline."""
    eligible: int = 10
    rho_v: float = 0.8
    sigma: float = 0.3
    outside: float = -1.0
    models: int = 2
    delta: tuple[float, ...] | None = None  # None: A17's per-L values; else one value for every model
    beta_slope: float = 0.0  # β_k = −slope (k − 1)
    chi: float = 0.5
    gamma_scale: float = 0.30
    gamma_exponent: float = 0.6
    rho_organic: float = 0.6
    rho_sponsored: float = 0.54
    zeta: tuple[float, float] = (0.2, 0.6)  # ζ = a + b u
    advertisers: int = 6
    contribution: str = "fixed"  # "fixed": g = 0.1; "uniform": g ~ U[0.05, 0.15]
    values: tuple = ("uniform", 1.0)  # ("uniform", upper) or ("lognormal", mu, s)
    kappa: float = 0.05
    cost_weight: float = 0.5  # λ
    omega: float = 0.5  # B = ω n
    requests: int = 1000
    k: int = 4
    k_s: int = 1
    versions: int = 1  # 2: two equally likely versions per model, version 2 adds τ η
    tau: float = 0.5
    bidding: str = "fixed_point_value_per_click"

    def model_deltas(self) -> tuple[float, ...]:
        if self.delta is not None:
            return tuple(self.delta) * self.models if len(self.delta) == 1 else tuple(self.delta)
        return {1: (-0.5,), 2: (-0.3, -0.7), 4: (-0.3, -0.5, -0.5, -0.7)}[self.models]

    def gamma(self) -> list[float]:
        return [self.gamma_scale * k ** (-self.gamma_exponent) for k in range(1, self.k + 1)]

    def value_distribution(self) -> ValueDistribution:
        if self.values[0] == "uniform":
            return ValueDistribution(kind="uniform", upper=self.values[1])
        return ValueDistribution(kind="lognormal", mu=self.values[1], s=self.values[2])

    def theta_bar(self) -> float:
        """The analytic mean of the value distribution (A12 §3.3)."""
        if self.values[0] == "uniform":
            return self.values[1] / 2
        return math.exp(self.values[1] + self.values[2] ** 2 / 2)


BASELINE = Params()
LEVELS: dict[str, dict] = {  # A17 §1–§2 sensitivity levels that E2 uses
    "rho_v=0.5": {"rho_v": 0.5}, "rho_v=0.95": {"rho_v": 0.95},
    "sigma=1.0": {"sigma": 1.0}, "sigma=0.1": {"sigma": 0.1},
    "outside=0.0": {"outside": 0.0},
    "delta=0": {"delta": (0.0,)}, "delta=-1.5": {"delta": (-1.5,)},
    "beta=-0.1(k-1)": {"beta_slope": 0.1},
    "chi=0": {"chi": 0.0}, "chi=1": {"chi": 1.0},
    "models=1": {"models": 1}, "models=4": {"models": 4},
    "gamma_exponent=0.3": {"gamma_exponent": 0.3}, "gamma_exponent=1.0": {"gamma_exponent": 1.0},
    "g~U[0.05,0.15]": {"contribution": "uniform"},
    "values=lognormal(s=0.5)": {"values": ("lognormal", -0.5, 0.5)},
    "values=lognormal(s=2)": {"values": ("lognormal", -0.5, 2.0)},
    "kappa=0": {"kappa": 0.0}, "kappa=0.2": {"kappa": 0.2},
    "lambda=0.1": {"cost_weight": 0.1}, "lambda=2": {"cost_weight": 2.0},
    "K=8,Ks=2": {"k": 8, "k_s": 2},
    "versions=2": {"versions": 2},
    "bidding=human_conversion_rate": {"bidding": "human_conversion_rate"},
}
PART3_LEVELS = ("models=1", "models=4", "versions=2")


@dataclass(frozen=True)
class Configuration:
    config_id: str
    group: str  # core | sensitivity | part3
    level: str  # "baseline" or a LEVELS key
    alpha: float
    regime: str
    variation: str
    params: Params = field(default_factory=Params)


def configurations() -> list[Configuration]:
    out = []
    for alpha in ALPHAS:
        for regime in REGIMES:
            for variation in VARIATIONS:
                out.append(Configuration("core/alpha=%g/%s/%s" % (alpha, regime, variation), "core", "baseline",
                                         alpha, regime, variation))
    for level, change in LEVELS.items():
        for alpha in ALPHAS:
            out.append(Configuration("sensitivity/%s/alpha=%g" % (level, alpha), "sensitivity", level, alpha,
                                     "agent_aware", "none", dataclasses.replace(BASELINE, **change)))
    for level in PART3_LEVELS:
        for alpha in ALPHAS:
            out.append(Configuration("part3/%s/alpha=%g" % (level, alpha), "part3", level, alpha, "human_optimised",
                                     "none", dataclasses.replace(BASELINE, **LEVELS[level])))
    return out


# --------------------------------------------------------------- primitives
@dataclass
class Draw:
    """The stipulated primitives of one market (before lists and mechanisms)."""
    products: list[str]
    z: np.ndarray
    u: np.ndarray
    advertisers: list[str]
    scores: dict[str, list[np.ndarray]]  # model -> per version: v^m_j
    contribution: np.ndarray
    theta: dict[str, float]


def draw(params: Params, rng: np.random.Generator) -> Draw:
    n = params.eligible
    products = ["P%02d" % j for j in range(n)]
    z = rng.standard_normal(n)
    advertisers = sorted(rng.choice(products, size=params.advertisers, replace=False).tolist())
    scores = {}
    noise = rng.standard_normal((params.models, n))
    shifts = rng.standard_normal((params.models, n)) if params.versions == 2 else None
    for m in range(params.models):
        base = params.rho_v * z + math.sqrt(1 - params.rho_v ** 2) * noise[m]
        scores["m%d" % m] = [base] if shifts is None else [base, base + params.tau * shifts[m]]
    g = rng.uniform(0.05, 0.15, n) if params.contribution == "uniform" else np.full(n, 0.1)
    values = params.value_distribution()
    theta = {j: float(values.sample(rng)) for j in advertisers}
    return Draw(products, z, stats.norm.cdf(z), advertisers, scores, g, theta)


def agent_params(params: Params, d: Draw) -> AgentParams:
    """The market's agents as SL-06 parameters (``source: stipulated``)."""
    deltas = params.model_deltas()
    models = []
    for m, (key, versions) in enumerate(sorted(d.scores.items())):
        base = {p: float(v) for p, v in zip(d.products, versions[0])}
        states = [StateParams(state_id="v1", probability=1.0 / len(versions))]
        if len(versions) == 2:
            states.append(StateParams(state_id="v2", probability=0.5, shifts={TASK: {
                p: float(b - a) for p, a, b in zip(d.products, versions[0], versions[1])}}))
        models.append(ModelParams(model_key=key, sigma=params.sigma, label_effect=deltas[m],
                                  position_effects=[0.0 - params.beta_slope * k for k in range(params.k)],
                                  outside_value=params.outside, page_request_prob=params.chi,
                                  scores={TASK: base}, states=states))
    return AgentParams(source="stipulated", models=models,
                       provenance={"parameters": "A17 (requirements/A17_stipulated_parameters.md)",
                                   "use": "pre-E1 dry run, exploratory"})


def types(params: Params, d: Draw) -> list[dict]:
    """Every positive-probability type (model, joint state) with its unlabelled ranking."""
    out = []
    deltas = params.model_deltas()
    for m, (key, versions) in enumerate(sorted(d.scores.items())):
        for v, scores in enumerate(versions):
            order = np.argsort(-scores, kind="stable")
            best = order[0]
            favourite = d.products[best] if scores[best] > params.outside else None
            second = d.products[order[1]] if favourite is not None else None
            labelled_ok = favourite is not None and scores[best] + deltas[m] > params.outside and all(
                scores[best] + deltas[m] > scores[i] for i in range(len(scores)) if i != best)
            out.append({"model": key, "version": v, "favourite": favourite, "second": second,
                        "labelled_favourite": bool(labelled_ok)})
    return out


def lists(params: Params, d: Draw, regime: str, variation: str) -> dict:
    """The organic rankings, their constraints and the market's classification."""
    by_u = [d.products[i] for i in np.argsort(-d.u, kind="stable")]
    ts = types(params, d)
    favourites = sorted({t["favourite"] for t in ts if t["favourite"]}, key=by_u.index)
    room = params.k - params.k_s
    feasible, fixed = True, set()
    if regime == "human_optimised":
        ranking = by_u
    elif regime == "agent_aware":
        ranking = favourites + [p for p in by_u if p not in favourites]
        feasible, fixed = len(favourites) <= room, set(favourites)
    elif regime == "gated":
        seconds = sorted({t["second"] for t in ts if t["second"] and t["second"] not in favourites}, key=by_u.index)
        head = seconds[:room]
        rest = [p for p in by_u if p not in favourites and p not in head]
        ranking = head + rest + favourites
        feasible, fixed = len(by_u) - len(favourites) >= params.k, set(head)
    else:
        raise ValueError(regime)
    rankings = [ranking]
    if variation == "randomised":
        varied = list(ranking)
        for i in range(room):
            j = i + room
            if j < len(varied) and varied[i] not in fixed and varied[j] not in fixed and varied[j] not in favourites:
                varied[i], varied[j] = varied[j], varied[i]
        rankings.append(varied)
    return {"rankings": rankings, "favourites": favourites, "excess": len(favourites) - room, "feasible": feasible,
            "screen": all(t["labelled_favourite"] for t in ts if t["favourite"]), "types": ts}


def build_market(config_id: str, params: Params, d: Draw, alpha: float, rankings: list[list[str]],
                 market_id: str, guardrail: str = "floor") -> tuple[Market, dict]:
    """The market, and its guardrail terms (``floor`` or the paper's ``paper`` rule)."""
    a, b = params.zeta
    offers = {p: Offer(contribution=float(g), match=float(u),
                       organic=Display(request=params.rho_organic, conversion=a + b * float(u)),
                       sponsored=Display(request=params.rho_sponsored, conversion=a + b * float(u)))
              for p, u, g in zip(d.products, d.u, d.contribution)}
    ap = agent_params(params, d)
    models, states = models_from_agent_params(ap, {TASK: TASK})
    context = Context(context_id=TASK, probability=1.0, offers=offers,
                      organic=[OrganicList(ranking=r, probability=1.0 / len(rankings)) for r in rankings],
                      agent_share=alpha, model_shares={m.key: 1.0 / len(models) for m in models})
    plain = Market(market_id=market_id, requests=params.requests, max_displayed=params.k, max_sponsored=params.k_s,
                   advertisers=d.advertisers, contexts=[context], states=states,
                   position_weights=params.gamma(), models=models,
                   costs={p: Cost(kind="quadratic", weight=params.cost_weight, capacity=params.omega * params.requests)
                          for p in d.products},
                   values={j: params.value_distribution() for j in d.advertisers})
    m_ref = match_reference(plain)
    table = action_table(plain)
    organic = params.requests * float(sum(table.weight[r] * table.match[r] for r, (_, s) in enumerate(table.rows)
                                          if s == ()))
    paper = (1 - params.kappa) * m_ref
    threshold = paper
    if guardrail == "floor" and organic < paper:  # the floor binds; FLOOR_SLACK gives the feasible set an interior
        threshold = organic - FLOOR_SLACK * params.requests
    terms = {"m_ref": m_ref, "m_organic": organic, "threshold": threshold, "floor_binds": int(organic < paper)}
    market = Market.model_validate(plain.model_copy(update={"guardrail": Guardrail(
        kappa=params.kappa, m_ref=threshold / (1 - params.kappa))}).model_dump())
    return market, terms


def market(config: Configuration, index: int) -> tuple[Market, dict, Draw]:
    """Market ``index`` of a configuration, with its classification."""
    rng = stream(MASTER_SEED, "sl19s-market", config.config_id, index)
    d = draw(config.params, rng)
    info = lists(config.params, d, config.regime, config.variation)
    m, terms = build_market(config.config_id, config.params, d, config.alpha, info["rankings"],
                            "%s#%d" % (config.config_id, index))
    info.update(terms)
    return m, info, d


# ------------------------------------------------------- Prop gating instance
GATING_PARAMS = dataclasses.replace(BASELINE, models=1, delta=(-0.5,), beta_slope=0.0, versions=1)
GATING_KAPPAS = (0.05, 0.2)
CONJ_KAPPA = 0.05


def gating_draw(index: int, stream_key: str = "sl19s-gating") -> tuple[Draw, dict]:
    """Prop gating's hypotheses (A1 §1.4), redrawing until they hold:
    - v_f + δ > v_r and v_r + δ > v_i for every other product i;
    - u(f) = max u > u(r);
    - the outside option is below every labelled product.
    f is the agent's top product and r its second; both are advertisers."""
    p = GATING_PARAMS
    rng = stream(MASTER_SEED, stream_key, index)
    attempts = 0
    while True:
        attempts += 1
        d = draw(p, rng)
        v = d.scores["m0"][0]
        order = np.argsort(-v, kind="stable")
        f, r = order[0], order[1]
        delta = p.model_deltas()[0]
        ok = (v[f] + delta > v[r] and all(v[r] + delta > v[i] for i in order[2:])
              and d.u[f] == d.u.max() and d.u[f] > d.u[r] and (v + delta).min() > p.outside)
        if ok:
            break
    products = d.products
    others = [q for q in products if q not in (products[f], products[r])]
    chosen = sorted([products[f], products[r]] + sorted(rng.choice(others, size=p.advertisers - 2,
                                                                     replace=False).tolist()))
    values = p.value_distribution()
    d.advertisers = chosen
    d.theta = {j: float(values.sample(rng)) for j in chosen}
    gaps = np.diff(np.sort(np.r_[v, v + delta, p.outside]))
    sigma = float(gaps[gaps > 0].min() / DETERMINISTIC_GAP_SIGMAS)
    return d, {"f": products[f], "r": products[r], "sigma": sigma, "attempts": attempts}


def gating_market(index: int, kappa: float, design: str, alpha: float = 1.0) -> tuple[Market, dict]:
    """Design (a) aware or (b) gated of Prop gating; with α < 1, Conj gating's lists R^a
    and R^b (A1 §1.5): R^a = (f, r, the rest by u), R^b = R^a without f (f last, never shown)."""
    d, info = gating_draw(index)
    by_u = [d.products[i] for i in np.argsort(-d.u, kind="stable")]
    f, r = info["f"], info["r"]
    rest = [q for q in by_u if q not in (f, r)]
    ranking = [f, r] + rest if design == "aware" else [r] + rest + [f]
    params = dataclasses.replace(GATING_PARAMS, sigma=info["sigma"], kappa=kappa, cost_weight=0.0)
    m, terms = build_market("gating", params, d, alpha, [ranking], "gating/kappa=%g/alpha=%g/%s#%d" % (
        kappa, alpha, design, index), guardrail="paper")
    m = Market.model_validate(m.model_copy(update={"costs": {}}).model_dump())  # C ≡ 0
    xi = min(1.0, kappa * float(d.u[d.products.index(f)]) / float(d.u[d.products.index(f)] - d.u[d.products.index(r)]))
    return m, {**info, **terms, "xi": xi, "theta_f": d.theta[f], "theta_r": d.theta[r],
               "u_f": float(d.u[d.products.index(f)]), "u_r": float(d.u[d.products.index(r)])}
