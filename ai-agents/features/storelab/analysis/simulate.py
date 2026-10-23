"""Synthetic E1 data with known truth for the SL-15 validation studies.

``synthetic_design`` builds the E1 design of SL-05 (``core/design.build_design``)
for synthetic tasks: ``tasks`` tasks of ``v`` products each, every K in ``set_sizes``,
r repetitions, labels at ``label_rate``, presentations ``per_set`` (A15). ``Truth``
gives the SL-06 agent model's parameters; ``simulate`` draws one choice per trial from
``backends.simulated.agent_logits`` (the model the runner uses) by Gumbel-max on a
keyed stream, so the estimators are checked against the data-generating process,
not against themselves. A two-state ``Truth`` with ``mixture="trial"`` draws the
state per trial (a mixed logit, which violates IIA: the power alternative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backends.simulated import agent_logits
from ..core.agent_params import ModelParams, StateParams
from ..core.design import Design, build_design
from ..core.protocol import Observation, Task
from ..core.rng import stream
from .data import ChoiceData, from_presentations

# The e1_pilot shape: 4 tasks, v = 10, K in {2, 4, 8}, r = 10, label rate 0.3.
PILOT_SHAPE = {"tasks": 4, "v": 10, "set_sizes": (2, 4, 8), "repetitions": 10, "label_rate": 0.3}


def synthetic_design(seed: int, tasks: int = 4, v: int = 10, set_sizes=(2, 4, 8), repetitions: int = 10,
                     label_rate: float = 0.3) -> list[Design]:
    designs = []
    for t in range(tasks):
        task_id = "task%d" % t
        eligible = ["%s_p%02d" % (task_id, j) for j in range(v)]
        for k in set_sizes:
            designs.append(build_design(task_id, eligible, k, repetitions, label_rate, "per_set", seed))
    return designs


@dataclass
class Truth:
    """SL-06 agent parameters on the original scale (σ applied by agent_logits)."""

    sigma: float
    label_effect: float = -0.5
    position_effects: list[float] = field(default_factory=lambda: [0.0, -0.1, -0.2, -0.3, -0.35, -0.4, -0.45, -0.5])
    outside_value: float = 4.0
    score_range: tuple[float, float] = (3.6, 4.8)
    second_state_shift: float = 0.0  # > 0: two equally likely states with reversed score shifts
    mixture: str = "trial"

    def params(self, designs: list[Design]) -> ModelParams:
        scores = {}
        for d in designs:
            if d.task_id not in scores:
                low, high = self.score_range
                ids = d.eligible_ids
                scores[d.task_id] = {pid: low + (high - low) * j / (len(ids) - 1) for j, pid in enumerate(ids)}
        states = [StateParams(state_id="default", probability=1.0)]
        if self.second_state_shift:
            shift = self.second_state_shift
            up = {t: {pid: shift * (1 if j % 2 else -1) for j, pid in enumerate(sorted(s))} for t, s in scores.items()}
            down = {t: {pid: -value for pid, value in s.items()} for t, s in up.items()}
            states = [StateParams(state_id="a", probability=0.5, shifts=up),
                      StateParams(state_id="b", probability=0.5, shifts=down)]
        return ModelParams(model_key="truth", sigma=self.sigma, label_effect=self.label_effect,
                           position_effects=self.position_effects, outside_value=self.outside_value,
                           page_request_prob=0.0, scores=scores, states=states)

    def on_sigma_scale(self) -> dict[str, float]:
        """The identified parameters: δ/σ and (β_k − β_1)/σ."""
        out = {"delta": self.label_effect / self.sigma}
        for k in range(2, len(self.position_effects) + 1):
            out["beta_%d" % k] = (self.position_effects[k - 1] - self.position_effects[0]) / self.sigma
        return out


def _observation(task_id: str, presentation) -> Observation:
    return Observation(task=Task(task_id=task_id, instruction="synthetic"), set_id=presentation.set_id,
                       presentation_id=presentation.presentation_id, candidates=presentation.candidates)


def simulate(designs: list[Design], truth: Truth, seed: int, replication: int | str) -> ChoiceData:
    """One synthetic data set: every (presentation, repetition) of the designs once."""
    params = truth.params(designs)
    rng = stream(seed, "validation", "replication", replication)
    rows = []
    for d in designs:
        for s in d.sets:
            presentation = d.presentation_for(s.set_id, 0)
            obs = _observation(d.task_id, presentation)
            logits = np.stack([agent_logits(params, obs, state.state_id) for state in params.states])
            for rep in range(d.repetitions):
                state = 0 if len(params.states) == 1 else int(rng.integers(len(params.states)))
                choice = int(np.argmax(logits[state] + rng.gumbel(size=logits.shape[1])))
                chosen = None if choice == 0 else presentation.candidates[choice - 1].product_id
                rows.append((d.task_id, presentation, rep, chosen))
    return from_presentations(rows)
