"""Markets for the mechanism solvers (SL-16; ``requirements/paper.tex`` l.184–424, A1 §0).

A market fixes what every mechanism needs:

- contexts x with probability μ(x), eligible sets E(x), an organic policy Q^o(·|x) over
  rankings of E(x), the agent share α(x) and the model mix π_m(x);
- the advertisers, at most K displayed and at most K^s sponsored products;
- the common state Λ (discrete, probabilities F_Λ), drawn once per period;
- buyer response: humans follow Ass. human with page requests (Ass. retrieval): a
  human requests the page of the product in position k with probability γ_k ρ_j(x, ℓ)
  and then chooses it with probability ζ_j(x, ℓ), so w_j = ρ_j ζ_j; an agent of model
  m chooses by the logit eq. agent-logit with s_j = v_j^m(x, Λ) + δ_m ℓ_j + β_{m,k_j}
  and requests each displayed page with probability χ_m;
- contributions g(x, j) ≥ 0, match quality u(x, j), operating costs C_j and the
  period's n requests; optionally the guardrail eq. mq-constraint and the value
  distributions F_j (profit rule).

``ActionTable`` enumerates the feasible allocations a = (R, S^s): for each context and
organic ranking R in the support of Q^o, every ordered sponsored list S^s of distinct
eligible advertisers with |S^s| ≤ K^s. The displayed list is S^s (labelled) followed
by the first K − |S^s| products of R not in S^s (organic integrity, l.188). For every
action it holds the exact choice probabilities q_j(x, a | Λ) per state, their mean
q̄_j(x, a) and the human and agent components, so every policy outcome below is
linear in the policy (Lemma convexity, l.419). The agent probabilities are those of
SL-06 (backends/simulated.py ``choice_probs``; core may not import backends, and a
test checks equality).

Everything is per request: X_j / n, p_j^Q(Λ), Γ / n and M / n.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from ..agent_params import AgentParams, HumanParams
from ..config import Ident, NonEmptyStr, Probability, Section, canonical_json
from ..protocol import ProductId
from .values import ValueDistribution

MREF_ENUMERATION_LIMIT = 4_000_000  # ordered lists per context for M^ref (|E| = 10, K = 8: 1.8M, ~1.5 s); beyond, pass m_ref
DEFAULT_STATE = "default"


class Display(Section):
    """Human response to product j in context x under one label (Ass. retrieval)."""
    request: Probability  # ρ_j(x, ℓ)
    conversion: Probability  # ζ_j(x, ℓ); w_j(x, ℓ) = ρ ζ


class Offer(Section):
    """Product j in context x: g(x, j), u(x, j) and the human response by label."""
    contribution: NonNegativeFloat = 0.0  # g(x, j)
    match: float = 0.0  # u(x, j)
    organic: Display
    sponsored: Display


class OrganicList(Section):
    ranking: list[ProductId] = Field(min_length=1)  # a ranking of E(x)
    probability: PositiveFloat  # Q^o(R | x)


class Context(Section):
    context_id: Ident
    probability: PositiveFloat  # μ(x)
    offers: dict[ProductId, Offer] = Field(min_length=1)  # E(x)
    organic: list[OrganicList] = Field(min_length=1)  # Q^o(· | x)
    agent_share: Probability  # α(x)
    model_shares: dict[Ident, Probability] = {}  # π_m(x)

    @model_validator(mode="after")
    def _organic(self) -> Context:
        eligible = sorted(self.offers)
        for entry in self.organic:
            if sorted(entry.ranking) != eligible:
                raise ValueError("context %s: an organic ranking must order every eligible product once"
                                 % self.context_id)
        if not math.isclose(sum(e.probability for e in self.organic), 1.0, abs_tol=1e-9):
            raise ValueError("context %s: organic probabilities must sum to 1" % self.context_id)
        if self.agent_share > 0 and not math.isclose(sum(self.model_shares.values()), 1.0, abs_tol=1e-9):
            raise ValueError("context %s: model shares must sum to 1" % self.context_id)
        return self


class State(Section):
    state_id: Ident
    probability: PositiveFloat


class AgentModel(Section):
    """Model m: v_j^m(x, Λ) per state and context, δ_m, β_{m,k}, σ_m, v₀^m(x), χ_m."""
    key: Ident
    sigma: PositiveFloat
    label_effect: float = 0.0  # δ_m
    position_effects: list[float] = []  # β_{m,k}, index k - 1; at least K entries (or none: all 0)
    page_request_prob: Probability = 0.0  # χ_m
    outside: dict[Ident, float]  # context -> v₀^m(x)
    scores: dict[Ident, dict[Ident, dict[ProductId, float]]]  # state -> context -> product -> v_j^m(x, Λ)


class Cost(Section):
    """C_j: ``quadratic`` λ D² / (2B), ``overflow`` λ (D − B)⁺ (Ass. cost, eq. quad-cost)."""
    kind: Literal["quadratic", "overflow"]
    weight: NonNegativeFloat  # λ_j
    capacity: PositiveFloat  # B_j


class Guardrail(Section):
    """eq. mq-constraint: M(Q^o, Q^s) ≥ (1 − κ) M^ref; M^ref is enumerated unless given."""
    kappa: float = Field(ge=0.0, lt=1.0)
    m_ref: PositiveFloat | None = None  # in total (not per request) units


class Market(Section):
    market_id: NonEmptyStr
    requests: PositiveInt  # n
    max_displayed: PositiveInt  # K
    max_sponsored: NonNegativeInt  # K^s
    advertisers: list[ProductId]
    contexts: list[Context] = Field(min_length=1)
    states: list[State] = Field(default_factory=lambda: [State(state_id=DEFAULT_STATE, probability=1.0)],
                                min_length=1)
    position_weights: list[PositiveFloat] = Field(min_length=1)  # γ^H_k, index k - 1
    models: list[AgentModel] = []
    costs: dict[ProductId, Cost] = {}
    guardrail: Guardrail | None = None
    values: dict[ProductId, ValueDistribution] = {}  # F_j, needed by the profit rule

    @model_validator(mode="after")
    def _consistent(self) -> Market:
        k = self.max_displayed
        if self.max_sponsored > k:
            raise ValueError("max_sponsored (K^s) must not exceed max_displayed (K)")
        if len(self.position_weights) < k:
            raise ValueError("position_weights: need K = %d entries" % k)
        if any(a < b for a, b in zip(self.position_weights, self.position_weights[1:])):
            raise ValueError("position_weights: must be nonincreasing (Ass. human)")
        if not math.isclose(sum(c.probability for c in self.contexts), 1.0, abs_tol=1e-9):
            raise ValueError("context probabilities must sum to 1")
        if not math.isclose(sum(s.probability for s in self.states), 1.0, abs_tol=1e-9):
            raise ValueError("state probabilities must sum to 1")
        ids = [c.context_id for c in self.contexts]
        if len(set(ids)) != len(ids) or len({s.state_id for s in self.states}) != len(self.states):
            raise ValueError("duplicate context or state id")
        products = set(self.products)
        if not set(self.advertisers) <= products or len(set(self.advertisers)) != len(self.advertisers):
            raise ValueError("advertisers must be distinct eligible products")
        if not set(self.costs) <= products or not set(self.values) <= set(self.advertisers):
            raise ValueError("costs need eligible products and values need advertisers")
        keys = {m.key for m in self.models}
        if len(keys) != len(self.models):
            raise ValueError("duplicate model key")
        for m in self.models:
            if m.position_effects and len(m.position_effects) < k:
                raise ValueError("model %s: position_effects need K = %d entries" % (m.key, k))
            for c in self.contexts:
                if c.context_id not in m.outside:
                    raise ValueError("model %s: no outside score for context %s" % (m.key, c.context_id))
                for s in self.states:
                    missing = set(c.offers) - set(m.scores.get(s.state_id, {}).get(c.context_id, {}))
                    if missing:
                        raise ValueError("model %s: no score in state %s, context %s for %s"
                                         % (m.key, s.state_id, c.context_id, sorted(missing)[:3]))
        for c in self.contexts:
            if c.agent_share > 0 and set(c.model_shares) - keys:
                raise ValueError("context %s: model shares for unknown models" % c.context_id)
        return self

    @property
    def products(self) -> list[str]:
        return sorted({j for c in self.contexts for j in c.offers})

    def market_hash(self) -> str:
        """sha256 of the canonical JSON of the market (the CRN check of SL-19)."""
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ response
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def response(market: Market, context: Context, displayed: tuple[str, ...], labels: tuple[bool, ...]) -> dict:
    """Choice probabilities of the displayed list (position order) by state: ``human``
    q^H_j (Ass. human), ``agents`` Σ_m π_m q^{A_m}_j per state, ``pooled`` q_j(x, a | Λ)
    of eq. mixed-q, and the expected page requests per request of humans and agents."""
    gamma = np.asarray(market.position_weights[:len(displayed)])
    human_terms = [context.offers[j].sponsored if lab else context.offers[j].organic
                   for j, lab in zip(displayed, labels)]
    human = gamma * np.array([d.request * d.conversion for d in human_terms])
    if human.sum() > 1.0 + 1e-12:
        raise ValueError("market %s, context %s: human choice probabilities of %s sum to %.6g > 1 (Ass. human)"
                         % (market.market_id, context.context_id, displayed, human.sum()))
    agents = np.zeros((len(market.states), len(displayed)))
    chi = 0.0
    for m in market.models:
        share = context.model_shares.get(m.key, 0.0)
        if share == 0.0:
            continue
        chi += share * m.page_request_prob
        beta = np.asarray(m.position_effects[:len(displayed)]) if m.position_effects else np.zeros(len(displayed))
        delta = m.label_effect * np.asarray(labels, dtype=float)
        for s, state in enumerate(market.states):
            scores = m.scores[state.state_id][context.context_id]
            values = np.array([m.outside[context.context_id]] + [scores[j] for j in displayed]) + np.r_[0.0, delta + beta]
            agents[s] += share * _softmax(values / m.sigma)[1:]
    alpha = context.agent_share
    return {"human": human, "agents": agents, "pooled": (1 - alpha) * human[None, :] + alpha * agents,
            "human_requests": (1 - alpha) * gamma * np.array([d.request for d in human_terms]),
            "agent_requests": alpha * chi * np.ones(len(displayed))}


def displayed_list(ranking: list[str], sponsored: tuple[str, ...], k: int) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    """S^s first (labelled), then the first K − |S^s| products of R not in S^s."""
    organic = [j for j in ranking if j not in sponsored][:k - len(sponsored)]
    return tuple(sponsored) + tuple(organic), (True,) * len(sponsored) + (False,) * len(organic)


def sponsored_lists(eligible_advertisers: list[str], k_s: int) -> list[tuple[str, ...]]:
    """Every ordered list of distinct eligible advertisers of length 0..K^s."""
    return [s for size in range(k_s + 1) for s in itertools.permutations(eligible_advertisers, size)]


# --------------------------------------------------------------- action table
@dataclass
class ActionTable:
    """Feasible actions as rows; arrays are per request (see the module docstring)."""
    market: Market
    products: list[str]
    groups: list[tuple[int, int]]  # (context index, organic list index)
    group_weight: np.ndarray  # μ(x) Q^o(R | x)
    rows: list[tuple[int, tuple[str, ...]]]  # (group index, sponsored list)
    displayed: list[tuple[str, ...]]
    row_group: np.ndarray
    weight: np.ndarray  # μ(x) Q^o(R | x) of the row's group
    q_state: np.ndarray  # (states, rows, products): q_j(x, a | Λ)
    q_bar: np.ndarray  # (rows, products): q̄_j(x, a)
    q_human: np.ndarray  # (rows, products): (1 − α) q^H_j
    q_agent: np.ndarray  # (rows, products): α Σ_m π_m E_Λ q^{A_m}_j
    contribution: np.ndarray  # (rows,): Σ_j g(x, j) q̄_j
    match: np.ndarray  # (rows,): Σ_j u(x, j) q̄_j
    state_probs: np.ndarray

    @property
    def index(self) -> dict[str, int]:
        return {j: i for i, j in enumerate(self.products)}

    def linear(self, per_row: np.ndarray) -> np.ndarray:
        """The matrix mapping a row policy y (Σ over each group's rows = 1) to the
        per-request outcome Σ_rows weight · per_row."""
        return (per_row * self.weight[:, None]).T

    def outcomes(self, y: np.ndarray) -> dict:
        """Per-request outcomes of a row policy y."""
        w = self.weight * y
        p_state = np.einsum("r,srj->sj", w, self.q_state)
        return {"X": w @ self.q_bar, "X_human": w @ self.q_human, "X_agent": w @ self.q_agent,
                "p_state": p_state, "contribution": float(w @ self.contribution), "match": float(w @ self.match)}

    def find(self, group: int, sponsored: tuple[str, ...]) -> int:
        for r, (g, s) in enumerate(self.rows):
            if g == group and s == tuple(sponsored):
                return r
        raise KeyError("no action %s in group %d" % (sponsored, group))


def action_table(market: Market) -> ActionTable:
    products = market.products
    index = {j: i for i, j in enumerate(products)}
    advertisers = set(market.advertisers)
    probs = market_state_probs(market)
    groups, group_weight, rows, displayed, row_group = [], [], [], [], []
    q_state, q_human, q_agent, contribution, match = [], [], [], [], []
    for ci, context in enumerate(market.contexts):
        lists = sponsored_lists([j for j in sorted(context.offers) if j in advertisers], market.max_sponsored)
        for oi, organic in enumerate(context.organic):
            group = len(groups)
            groups.append((ci, oi))
            group_weight.append(context.probability * organic.probability)
            for sponsored in lists:
                shown, labels = displayed_list(organic.ranking, sponsored, market.max_displayed)
                r = response(market, context, shown, labels)
                cols = [index[j] for j in shown]
                qs = np.zeros((len(market.states), len(products)))
                qs[:, cols] = r["pooled"]
                qh, qa = np.zeros(len(products)), np.zeros(len(products))
                qh[cols] = (1 - context.agent_share) * r["human"]
                qa[cols] = context.agent_share * (probs @ r["agents"])
                qbar = probs @ qs
                rows.append((group, sponsored))
                displayed.append(shown)
                row_group.append(group)
                q_state.append(qs)
                q_human.append(qh)
                q_agent.append(qa)
                contribution.append(math.fsum(context.offers[j].contribution * qbar[index[j]] for j in shown))
                match.append(math.fsum(context.offers[j].match * qbar[index[j]] for j in shown))
    q_state = np.transpose(np.array(q_state), (1, 0, 2))
    row_group = np.array(row_group)
    group_weight = np.array(group_weight)
    return ActionTable(market=market, products=products, groups=groups, group_weight=group_weight, rows=rows,
                       displayed=displayed, row_group=row_group, weight=group_weight[row_group], q_state=q_state,
                       q_bar=np.einsum("s,srj->rj", probs, q_state), q_human=np.array(q_human),
                       q_agent=np.array(q_agent), contribution=np.array(contribution), match=np.array(match),
                       state_probs=probs)


# ---------------------------------------------------------------------- M^ref
def match_reference(market: Market) -> float:
    """M^ref = max over organic policies of M(Q^o, ∅), in total units: per context the
    best unlabelled ordered list of min(K, |E(x)|) eligible products, by vectorised
    enumeration (``response`` computes the same probabilities one list at a time)."""
    if market.guardrail is not None and market.guardrail.m_ref is not None:
        return market.guardrail.m_ref
    total = 0.0
    for context in market.contexts:
        eligible = sorted(context.offers)
        length = min(market.max_displayed, len(eligible))
        count = math.perm(len(eligible), length)
        if count > MREF_ENUMERATION_LIMIT:
            raise ValueError("context %s: %d organic lists exceed the M^ref enumeration limit; pass guardrail.m_ref"
                             % (context.context_id, count))
        lists = np.array(list(itertools.permutations(range(len(eligible)), length)), dtype=np.int64)
        best = -math.inf
        for chunk in np.array_split(lists, max(1, len(lists) // 100_000)):
            best = max(best, float(_list_match(market, context, eligible, chunk).max()))
        total += context.probability * best
    return market.requests * total


def _list_match(market: Market, context: Context, eligible: list[str], lists: np.ndarray) -> np.ndarray:
    """Σ_j u(x, j) q̄_j(x, (R, ∅)) for each unlabelled ordered list (rows of indices)."""
    length = lists.shape[1]
    gamma = np.asarray(market.position_weights[:length])
    w = np.array([context.offers[j].organic.request * context.offers[j].organic.conversion for j in eligible])
    u = np.array([context.offers[j].match for j in eligible])
    q = (1 - context.agent_share) * gamma[None, :] * w[lists]
    for m in market.models:
        share = context.model_shares.get(m.key, 0.0)
        if share == 0.0 or context.agent_share == 0.0:
            continue
        beta = np.asarray(m.position_effects[:length]) if m.position_effects else np.zeros(length)
        for state in market.states:
            v = np.array([m.scores[state.state_id][context.context_id][j] for j in eligible])
            logits = np.concatenate([np.full((len(lists), 1), m.outside[context.context_id]), v[lists] + beta], axis=1)
            q += context.agent_share * share * state.probability * _softmax(logits / m.sigma)[:, 1:]
    return (q * u[lists]).sum(axis=1)


def market_state_probs(market: Market) -> np.ndarray:
    return np.array([s.probability for s in market.states])


# ------------------------------------------------- SL-06 agent parameters
def models_from_agent_params(params: AgentParams, task_of: dict[str, str]) -> tuple[list[AgentModel], list[State]]:
    """AgentModel objects and the market's states from SL-06 ``AgentParams``. Each
    SL-06 model has its own discrete states, drawn independently per period
    (``draw_state``), so the market state is their product. ``task_of`` maps each
    context to the task whose scores it uses."""
    combos = list(itertools.product(*[[(m.model_key, s) for s in m.states] for m in params.models]))
    states = [State(state_id="__".join(s.state_id for _, s in combo) if len(combo) > 1 else combo[0][1].state_id,
                    probability=math.prod(s.probability for _, s in combo)) for combo in combos]
    models = []
    for mi, m in enumerate(params.models):
        scores = {}
        for state, combo in zip(states, combos):
            own = combo[mi][1]
            scores[state.state_id] = {x: {j: v + own.shifts.get(task, {}).get(j, 0.0)
                                          for j, v in m.scores[task].items()} for x, task in task_of.items()}
        models.append(AgentModel(key=m.model_key, sigma=m.sigma, label_effect=m.label_effect,
                                 position_effects=m.position_effects, page_request_prob=m.page_request_prob,
                                 outside={x: m.outside_value for x in task_of}, scores=scores))
    return models, states


def human_displays(params: HumanParams, task: str, product: str, request: dict[bool, float] | None = None) -> dict:
    """``organic`` and ``sponsored`` Display entries for an SL-06 human product weight
    w_j(x, ℓ): with no page-request split, ρ = w and ζ = 1."""
    out = {}
    for label, name in ((False, "organic"), (True, "sponsored")):
        w = params.weight(task, product, label)
        rho = request[label] if request else w
        out[name] = Display(request=rho, conversion=w / rho if rho > 0 else 0.0)
    return out
