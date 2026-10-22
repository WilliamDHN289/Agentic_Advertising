"""Simulated decision-makers (SL-06): the paper's agent and human models, exactly.

Agent model m (``v1.tex`` l.289): for a presentation a with displayed set S(a),
s_j = v_j^m(x) + shift_j(Λ) + δ_m ℓ_j + β_{m,k_j} and

    q_j = exp(s_j / σ) / D,   q_0 = exp(v₀ / σ) / D,   D = exp(v₀ / σ) + Σ_i exp(s_i / σ).

Human model (Ass. human, l.236): q_j^H = γ_{k_j} w_j(x, ℓ_j), q_0^H = 1 − Σ q_j^H; the
weights must keep Σ q_j^H ≤ 1. The legacy cascade human (``HumanSim``) stays in
core/legacy_env.py, where the legacy runner uses it (SL-04).

``choice_probs`` gives the exact probabilities for the solvers, for one state or
averaged over F_Λ. The runner samples by Gumbel-max on keyed streams: a trial draws
from ``stream(seed, "agent", trial_key)`` (``"human"`` for the human model) one
Gumbel variate per alternative (outside first, then candidates in position order)
and then one uniform per candidate; argmax(logit + Gumbel) is the choice. Under the
``browser_tool`` interface each displayed product that is not chosen is requested
with probability χ_m (A2), in position order, at most ``max_turns`` − 1 times, so the
actions are ``RequestPage``… then ``Select`` or ``Leave``, the types the LLM
backends emit. Other interfaces have no product pages.

The state Λ is a period-level draw, not a trial-level one: the caller passes it
(``draw_state`` gives a keyed draw). A model with a single state needs none.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.agent_params import AgentParams, HumanParams, ModelParams, StateParams, load_agent_params
from ..core.config import RunConfig, Section
from ..core.protocol import Action, Leave, Observation, RequestPage, Select, TrialOutcome
from ..core.rng import stream
from ..core.tasks import TaskStimuli

PAGE_INTERFACE = "browser_tool"
SIMULATED_MODEL_KEY = "simulated"


class ChoiceProbs(Section):
    outside: float
    products: dict[str, float]  # candidate id -> probability, in position order

    def as_list(self, observation: Observation) -> list[float]:
        """[q_0, q_j for candidates in position order]."""
        return [self.outside] + [self.products[c.product_id] for c in observation.candidates]


# ---------------------------------------------------------------------- agents
def _state(params: ModelParams, state_id: str | None) -> StateParams:
    if state_id is None:
        if len(params.states) != 1:
            raise ValueError("model %s has %d states; pass the period's state_id"
                             % (params.model_key, len(params.states)))
        return params.states[0]
    return params.state(state_id)


def agent_logits(params: ModelParams, observation: Observation, state_id: str | None = None) -> np.ndarray:
    """[v₀/σ, s_j/σ for candidates in position order]."""
    task = observation.task.task_id
    scores = params.scores.get(task)
    if scores is None:
        raise KeyError("model %s has no scores for task %s" % (params.model_key, task))
    shifts = _state(params, state_id).shifts.get(task, {})
    values = [params.outside_value]
    for c in observation.candidates:
        if c.product_id not in scores:
            raise KeyError("model %s has no score for %s in task %s" % (params.model_key, c.product_id, task))
        if c.position > len(params.position_effects):
            raise ValueError("model %s has no position effect for position %d" % (params.model_key, c.position))
        values.append(scores[c.product_id] + shifts.get(c.product_id, 0.0)
                      + params.label_effect * c.sponsored_label + params.position_effects[c.position - 1])
    return np.asarray(values, dtype=np.float64) / params.sigma


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


def _probs(observation: Observation, values: np.ndarray) -> ChoiceProbs:
    return ChoiceProbs(outside=float(values[0]),
                       products={c.product_id: float(p) for c, p in zip(observation.candidates, values[1:])})


def choice_probs(params: ModelParams, observation: Observation, state_id: str | None = None,
                 average: bool = False) -> ChoiceProbs:
    """q(x, a | Λ) for ``state_id``; with ``average``, q̄(x, a) = E_Λ q(x, a | Λ)."""
    if average:
        values = sum(s.probability * _softmax(agent_logits(params, observation, s.state_id)) for s in params.states)
    else:
        values = _softmax(agent_logits(params, observation, state_id))
    return _probs(observation, values)


def gumbel_max(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray | int:
    """argmax(logits + Gumbel noise) along the last axis: a draw from softmax(logits)."""
    noise = rng.gumbel(size=np.shape(logits))
    choice = np.argmax(np.asarray(logits) + noise, axis=-1)
    return int(choice) if np.ndim(choice) == 0 else choice


def draw_state(params: ModelParams, seed: int, *keys: str | int) -> str:
    """The period's state, drawn from F_Λ with ``stream(seed, "state", model_key, *keys)``."""
    probabilities = np.array([s.probability for s in params.states])
    index = stream(seed, "state", params.model_key, *keys).choice(len(params.states), p=probabilities / probabilities.sum())
    return params.states[int(index)].state_id


def _actions(observation: Observation, choice: int, uniforms: np.ndarray, request_prob: float, interface: str,
             max_turns: int) -> list[Action]:
    requests: list[Action] = []
    if interface == PAGE_INTERFACE:
        for index, (c, u) in enumerate(zip(observation.candidates, uniforms)):
            if index + 1 != choice and u < request_prob and len(requests) < max_turns - 1:
                requests.append(RequestPage(product_id=c.product_id))
    final = Leave() if choice == 0 else Select(product_id=observation.candidates[choice - 1].product_id)
    return requests + [final]


def _outcome(actions: list[Action], backend: str, model: str) -> TrialOutcome:
    final = actions[-1]
    common = dict(page_requests=[a.product_id for a in actions if isinstance(a, RequestPage)], turns=len(actions),
                  attempts=1, backend=backend, requested_model=model, returned_model=model)
    if isinstance(final, Select):
        return TrialOutcome(status="choice", chosen_id=final.product_id, **common)
    return TrialOutcome(status="outside", **common)


class SimulatedAgent:
    """Agent model m as a backend: ``act`` gives the actions, ``decide`` the outcome."""

    backend = "simulated"

    def __init__(self, params: ModelParams, seed: int, max_turns: int = 8):
        self.params, self.seed, self.max_turns = params, seed, max_turns
        self.model = "%s:%s" % (SIMULATED_MODEL_KEY, params.model_key)

    def choice_probs(self, observation: Observation, state_id: str | None = None,
                     average: bool = False) -> ChoiceProbs:
        return choice_probs(self.params, observation, state_id, average)

    def act(self, observation: Observation, interface: str, trial_key: str,
            state_id: str | None = None) -> list[Action]:
        logits = agent_logits(self.params, observation, state_id)
        rng = stream(self.seed, "agent", trial_key)
        choice = gumbel_max(logits, rng)
        uniforms = rng.random(len(observation.candidates))
        return _actions(observation, choice, uniforms, self.params.page_request_prob, interface, self.max_turns)

    def decide(self, observation: Observation, interface: str, trial_key: str,
               state_id: str | None = None) -> TrialOutcome:
        return _outcome(self.act(observation, interface, trial_key, state_id), self.backend, self.model)


# ---------------------------------------------------------------------- humans
def human_probs(params: HumanParams, observation: Observation) -> ChoiceProbs:
    """q_j^H = γ_{k_j} w_j(x, ℓ_j) and q_0^H = 1 − Σ q_j^H (Ass. human)."""
    task = observation.task.task_id
    values = []
    for c in observation.candidates:
        if c.position > len(params.position_weights):
            raise ValueError("no human position weight for position %d" % c.position)
        values.append(params.position_weights[c.position - 1] * params.weight(task, c.product_id, c.sponsored_label))
    total = math.fsum(values)
    if total > 1.0 + 1e-12:
        raise ValueError("human choice probabilities sum to %.6g > 1 for %s (Ass. human)"
                         % (total, observation.presentation_id))
    return _probs(observation, np.asarray([max(0.0, 1.0 - total)] + values))


class SimulatedHuman:
    """The human model as a backend; it requests no pages."""

    backend = "human_model"

    def __init__(self, params: HumanParams, seed: int):
        self.params, self.seed, self.model = params, seed, "human_model:position_weights"

    def choice_probs(self, observation: Observation) -> ChoiceProbs:
        return human_probs(self.params, observation)

    def act(self, observation: Observation, interface: str, trial_key: str) -> list[Action]:
        with np.errstate(divide="ignore"):
            logits = np.log(np.asarray(human_probs(self.params, observation).as_list(observation)))
        choice = gumbel_max(logits, stream(self.seed, "human", trial_key))
        return _actions(observation, choice, np.ones(len(observation.candidates)), 0.0, interface, 1)

    def decide(self, observation: Observation, interface: str, trial_key: str) -> TrialOutcome:
        return _outcome(self.act(observation, interface, trial_key), self.backend, self.model)


# ------------------------------------------------------------------ parameters
def params_from_config(config: RunConfig, stimuli: Sequence[TaskStimuli], base: Path | None = None) -> AgentParams:
    """The run's AgentParams. ``stipulated``: every configured model (or one model
    ``simulated`` if none) gets the config's σ, δ, β, v₀ and χ, baseline scores
    v_j^m(x) = u(x, j) from SL-05 and a single state; humans under
    ``humans.model: position_weights`` get those γ_k and product weights 1.
    ``estimated``: the pinned ``params_file`` (relative paths from ``base``)."""
    simulated = config.agents.simulated
    if simulated is None:
        raise ValueError("agents.simulated is not configured")
    if simulated.parameters == "estimated":
        path = Path(simulated.params_file.path)
        return load_agent_params(path if path.is_absolute() or base is None else Path(base) / path,
                                 simulated.params_file.sha256)
    scores = {t.task.task_id: dict(t.utility) for t in stimuli}
    keys = [m.key for m in config.agents.models] or [SIMULATED_MODEL_KEY]
    models = [ModelParams(model_key=key, sigma=simulated.sigma, label_effect=simulated.label_effect,
                          position_effects=simulated.position_effects, outside_value=simulated.outside_value,
                          page_request_prob=simulated.page_request_prob, scores=scores) for key in keys]
    humans = (HumanParams(position_weights=config.humans.position_weights)
              if config.humans.model == "position_weights" else None)
    return AgentParams(source="stipulated", models=models, humans=humans)


# ------------------------------------------------------------------ load model
def load_moments(n: int, state_probs: Sequence[float], p_by_state: Sequence[float]) -> tuple[float, float]:
    """Lemma load: E[D] = n E_Λ p and Var(D) = n E_Λ[p(1 − p)] + n² Var_Λ(p)."""
    weights = np.asarray(state_probs, dtype=np.float64)
    p = np.asarray(p_by_state, dtype=np.float64)
    if weights.shape != p.shape or not math.isclose(weights.sum(), 1.0, abs_tol=1e-12):
        raise ValueError("state_probs must match p_by_state and sum to 1")
    mean_p = float(weights @ p)
    return n * mean_p, n * float(weights @ (p * (1 - p))) + n * n * float(weights @ (p - mean_p) ** 2)


def policy_choice_prob(params: ModelParams, policy: Sequence[tuple[Observation, float]], product_id: str,
                       state_id: str | None = None) -> float:
    """p_j^Q(Λ) (eq. pQ) for a policy given as (observation, probability) pairs."""
    return math.fsum(weight * choice_probs(params, obs, state_id).products.get(product_id, 0.0)
                     for obs, weight in policy)


def simulate_loads(params: ModelParams, policy: Sequence[tuple[Observation, float]], n: int, periods: int,
                   rng: np.random.Generator, chunk: int = 500) -> dict[str, np.ndarray]:
    """Per period: draw Λ from F_Λ, give each of n requests a set from ``policy`` and
    a Gumbel-max choice; returns each displayed product's load D_j per period."""
    observations = [obs for obs, _ in policy]
    set_probs = np.array([w for _, w in policy], dtype=np.float64)
    state_probs = np.array([s.probability for s in params.states], dtype=np.float64)
    width = 1 + max(len(o.candidates) for o in observations)
    logits = np.full((len(params.states), len(observations), width), -np.inf)
    for si, state in enumerate(params.states):
        for oi, obs in enumerate(observations):
            row = agent_logits(params, obs, state.state_id)
            logits[si, oi, :len(row)] = row
    products = sorted({c.product_id for o in observations for c in o.candidates})
    column = {pid: i for i, pid in enumerate(products)}
    # alternative index -> product column (-1: outside or padding)
    lookup = np.full((len(observations), width), -1)
    for oi, obs in enumerate(observations):
        for k, c in enumerate(obs.candidates):
            lookup[oi, k + 1] = column[c.product_id]
    loads = np.zeros((periods, len(products)), dtype=np.int64)
    for start in range(0, periods, chunk):
        size = min(chunk, periods - start)
        states = rng.choice(len(params.states), size=size, p=state_probs / state_probs.sum())
        sets = rng.choice(len(observations), size=(size, n), p=set_probs / set_probs.sum())
        choice = gumbel_max(logits[states[:, None], sets], rng)
        chosen = lookup[sets, choice]
        for pid, col in column.items():
            loads[start:start + size, col] = (chosen == col).sum(axis=1)
    return {pid: loads[:, col] for pid, col in column.items()}
