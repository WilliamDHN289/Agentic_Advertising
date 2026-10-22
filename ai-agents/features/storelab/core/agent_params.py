"""Parameters of the simulated decision-makers (SL-06), shared with SL-15.

``AgentParams`` is the content of ``agent_params.json``
(contracts/storelab/agent_params.schema.json): SL-15 writes it from E1 estimates and
backends/simulated.py reads it, as does a config's ``agents.simulated.params_file``.
Stipulated runs build the same object from the config (``params_from_config`` in
backends/simulated.py).

Agent model m (``v1.tex`` l.289): the score of displayed product j is
s_j = v_j^m(x) + shift_j(Λ) + δ_m ℓ_j + β_{m,k_j}, the outside option scores v₀^m,
and choice is logit with scale σ_m. ``scores`` hold v_j^m(x) per task and product;
each discrete state Λ has a probability and per-product score shifts. Estimated
parameters come from a conditional logit, where σ is not identified: SL-15 reports
v/σ, δ/σ and β/σ with ``sigma`` = 1. χ_m (``page_request_prob``) is the probability
that the agent requests the page of an included product it does not choose (A2).

Human model (Ass. human, l.236): q_j^H = γ_k w_j(x, ℓ) with nonincreasing position
weights γ_k (A11) and product weights w per label; a product without an entry has
``default_weight``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from .config import Ident, NonEmptyStr, Probability, Section
from .protocol import ProductId

AGENT_PARAMS_SCHEMA_VERSION = 1
DEFAULT_STATE = "default"


class StateParams(Section):
    state_id: Ident
    probability: Probability
    shifts: dict[NonEmptyStr, dict[ProductId, float]] = {}  # task_id -> product -> added to v


class ModelParams(Section):
    model_key: Ident
    sigma: PositiveFloat
    label_effect: float  # δ_m
    position_effects: list[float] = []  # β_{m,k}, index k - 1; positions beyond the list are an error
    outside_value: float  # v₀^m
    page_request_prob: Probability  # χ_m
    scores: dict[NonEmptyStr, dict[ProductId, float]] = Field(min_length=1)  # task_id -> product -> v_j^m(x)
    states: list[StateParams] = Field(default_factory=lambda: [StateParams(state_id=DEFAULT_STATE, probability=1.0)],
                                      min_length=1)

    @model_validator(mode="after")
    def _states(self) -> ModelParams:
        ids = [s.state_id for s in self.states]
        if len(set(ids)) != len(ids):
            raise ValueError("states: duplicate state_id")
        if not math.isclose(sum(s.probability for s in self.states), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("states: probabilities must sum to 1")
        if any(s.probability <= 0 for s in self.states):
            raise ValueError("states: every probability must be positive")
        for state in self.states:
            for task, shifts in state.shifts.items():
                unknown = set(shifts) - set(self.scores.get(task, {}))
                if unknown:
                    raise ValueError("states.%s: shifts for products without a score in task %s: %s"
                                     % (state.state_id, task, sorted(unknown)[:5]))
        return self

    def state(self, state_id: str) -> StateParams:
        for state in self.states:
            if state.state_id == state_id:
                return state
        raise KeyError("model %s has no state %r" % (self.model_key, state_id))


class LabelWeights(Section):
    organic: NonNegativeFloat  # w_j(x, 0)
    sponsored: NonNegativeFloat  # w_j(x, 1)


class HumanParams(Section):
    position_weights: list[PositiveFloat] = Field(min_length=1)  # γ_k, index k - 1
    product_weights: dict[NonEmptyStr, dict[ProductId, LabelWeights]] = {}
    default_weight: LabelWeights = Field(default_factory=lambda: LabelWeights(organic=1.0, sponsored=1.0))

    @model_validator(mode="after")
    def _nonincreasing(self) -> HumanParams:
        if any(a < b for a, b in zip(self.position_weights, self.position_weights[1:])):
            raise ValueError("position_weights: must be nonincreasing (Ass. human)")
        return self

    def weight(self, task_id: str, product_id: str, sponsored: bool) -> float:
        w = self.product_weights.get(task_id, {}).get(product_id, self.default_weight)
        return w.sponsored if sponsored else w.organic


class AgentParams(Section):
    schema_version: Literal[1] = AGENT_PARAMS_SCHEMA_VERSION
    source: Literal["stipulated", "estimated"]
    models: list[ModelParams] = Field(min_length=1)
    humans: HumanParams | None = None
    provenance: dict[NonEmptyStr, str] = {}  # e.g. the estimating run ids and code version

    @model_validator(mode="after")
    def _unique_models(self) -> AgentParams:
        keys = [m.model_key for m in self.models]
        if len(set(keys)) != len(keys):
            raise ValueError("models: duplicate model_key")
        return self

    def model(self, model_key: str) -> ModelParams:
        for m in self.models:
            if m.model_key == model_key:
                return m
        raise KeyError("no parameters for model %s" % model_key)


def dump_agent_params(params: AgentParams) -> str:
    return json.dumps(params.model_dump(mode="json"), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def load_agent_params(path: Path, sha256: str | None = None) -> AgentParams:
    """Read an agent_params.json; with ``sha256``, refuse a file with other bytes."""
    data = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("%s: sha256 %s does not match the pinned %s" % (path, hashlib.sha256(data).hexdigest(), sha256))
    return AgentParams.model_validate_json(data)
