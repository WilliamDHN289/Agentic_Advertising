"""Cost estimate, reservations and cost reports of runs (SL-11).

``core/cost.py`` holds the formulas; this module gives them the prices
(``pricing.py``), the token counts (``tokens.py``) and the rendered prompts of a
plan's stimuli, and reads and writes the run directory:

| Function | Used by | Result |
|---|---|---|
| ``PlanEstimate`` | ``api.plan_run`` (the ``estimate_cost`` hook of ``runner.plan``) | the plan's USD; ``PlanError`` for a model with no price |
| ``write_estimate`` | ``api.plan_run`` | ``reports/cost_estimate.json`` and ``.md`` |
| ``reservation`` | ``api.execute_run`` (the runner's ``cost``) | each trial's maximum cost, reserved before it starts |
| ``cost_report`` | ``cli cost RUN`` | ``reports/cost.json`` and ``.md``: estimate vs actual per model, cache savings, the SL-18 gate |

Only sandbox runs call paid providers: offline and replay plans are estimated at $0
and reserve nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .core.config import RunConfig
from .core.cost import (GATE_TOLERANCE, CostError, Estimator, ModelEstimate, actual_markdown, compare, estimate,
                        estimate_markdown, estimate_report, jsonl, trial_prompt)
from .core.protocol import Observation, TrialOutcome
from .core.render import build_observation
from .core.runner import PlanError, Stimuli, Trial
from .core.store import LocalStore, write_bytes
from .pricing import price
from .tokens import count_tokens, token_counting

PAID_MODES = ("sandbox",)


def _rates(model: str) -> tuple[float, float]:
    rate = price(model)
    return rate.input_usd_per_1m, rate.output_usd_per_1m


def observations(config: RunConfig, stimuli: Stimuli) -> Callable[[Trial], Observation]:
    """A trial's observation, rendered once per presentation (as the runner does)."""
    specs = {t.task.task_id: t.task for t in stimuli.tasks}
    designs = {s.set_id: d for d in stimuli.designs for s in d.sets}
    memo: dict[str, Observation] = {}

    def observation(trial: Trial) -> Observation:
        if trial.presentation_id not in memo:
            presentation = designs[trial.set_id].presentation_for(trial.set_id, trial.rep)
            memo[trial.presentation_id] = build_observation(config, specs[trial.task_id], presentation,
                                                            stimuli.products)
        return memo[trial.presentation_id]

    return observation


def estimator(config: RunConfig, stimuli: Stimuli) -> Estimator:
    observation = observations(config, stimuli)
    return Estimator(config, lambda trial: trial_prompt(observation(trial), trial.interface), count_tokens, _rates)


class PlanEstimate:
    """The ``estimate_cost`` hook of ``runner.plan``; keeps the rows for the report."""

    def __init__(self, stimuli: Stimuli):
        self.stimuli = stimuli
        self.rows: list[ModelEstimate] | None = None

    def __call__(self, config: RunConfig, trials: list[Trial], mode: str) -> float:
        if mode not in PAID_MODES:
            return 0.0
        try:
            self.rows = estimate(estimator(config, self.stimuli), trials)
        except CostError as error:
            raise PlanError("cost estimate: %s" % error) from None
        return sum(r.usd for r in self.rows)


def _write(path: Path, report: dict, markdown: str) -> None:
    write_bytes(path.with_suffix(".json"), (json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    write_bytes(path.with_suffix(".md"), markdown.encode("utf-8"))


def write_estimate(store: LocalStore, config: RunConfig, rows: list[ModelEstimate], *, mode: str,
                   run_id: str) -> Path:
    report = estimate_report(config, rows, mode=mode, run_id=run_id, token_counting=token_counting())
    path = store.reports_dir / "cost_estimate.json"
    _write(path, report, estimate_markdown(report))
    return path


def reservation(config: RunConfig, stimuli: Stimuli, mode: str) -> Callable[[Trial, TrialOutcome | None], float]:
    """The runner's ``cost``: a trial's maximum cost before it starts, nothing after
    (its provider calls are charged one by one through ``log_call``)."""
    if mode not in PAID_MODES:
        return lambda trial, outcome: 0.0
    try:
        costs = estimator(config, stimuli)
    except CostError as error:
        raise ValueError("cost estimate: %s" % error) from None
    return lambda trial, outcome: costs.trial(trial).max_usd if outcome is None else 0.0


def cost_report(run_dir: Path, tolerance: float = GATE_TOLERANCE) -> dict:
    """Estimate vs actual of a run, written to ``reports/cost.json`` and ``.md``."""
    store = LocalStore(run_dir)
    manifest = store.read_manifest()
    planned, stimuli = store.read_plan()
    trials = [Trial.model_validate(t) for t in planned]
    try:
        costs = estimator(manifest.config, Stimuli.model_validate(stimuli))
    except CostError as error:
        raise ValueError("cost estimate: %s" % error) from None
    report = compare(costs, trials, store.records(), jsonl(store.invocation_log), jsonl(store.cache_log),
                     tolerance=tolerance)
    report.update(run_id=manifest.run_id, mode=manifest.mode.resolved, config_hash=manifest.config_hash,
                  status=manifest.status_history[-1].status)
    _write(store.reports_dir / "cost.json", report, actual_markdown(report))
    return report
