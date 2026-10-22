"""Runner (SL-07): mode resolution, planning, execution, error policy and run status.

``plan`` resolves the mode once (tickets.md section 3.4), enumerates the trials of
an E1 design with their ``trial_key`` (section 3.6) and writes a PLANNED manifest,
then QUEUED or PENDING_APPROVAL by the cost estimate (SL-11 hook). ``execute`` runs
the pending trials through one ``Backend`` per model key, at most ``max_concurrency``
at a time, each on its own thread with a per-trial timeout, records every outcome once, applies the error policy
(section 3.5) and moves the run through its states:

| Trial outcome | Run effect |
|---|---|
| error/bad_request, error/auth, error/model_unavailable | FAILED at once |
| error/provider_unavailable (also a trial over ``trial_timeout_s``) | breaker: ``failure_threshold`` in a row → PAUSED, auto-resume after ``cooldown_s`` at most ``max_auto_resumes`` times, then FAILED |
| error/rate_limited | rate-limited share of the trials finished in the last ``rate_limit_window_s`` above ``rate_limit_pause_share`` (once at least ``failure_threshold`` finished) → PAUSED, auto-resumed like the breaker |
| a returned model id differing from the one seen before | outcome kept, both ids in the manifest; PAUSED (``model_drift``), resumed only by an operator |
| refusal, invalid, error/cache_miss | recorded; count as missing in their cell |
| budget would be exceeded | ``not_run``/budget; LIMIT_REACHED |
| kill switch | ``not_run``/kill_switch; CANCELLED |

A backend's outcome is recorded as it is: the runner never substitutes a choice,
switches backend or model, re-asks, or drops a trial. A backend that raises fails
the run and leaves its trial pending. At the end every planned trial has a final
record and the run is COMPLETED, or COMPLETED_WITH_GAPS when a cell's missing share
(invalid, refusal, error) exceeds ``missingness_threshold`` or a replay cache miss
occurred. A cell is (task, K, interface, model, temperature).
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from pydantic import Field, NonNegativeInt, PositiveInt

from .catalog_amazon import Product
from .config import Ident, Interface, NonEmptyStr, RunConfig, Section, Sha256, canonical_json
from .design import Design
from .manifest import Manifest, append_status, new_manifest, record_model_access
from .protocol import PROTOCOL_VERSION, Observation, TrialOutcome
from .records import TrialRecord, trial_key
from .render import TEMPLATES, build_observation
from .store import ArtifactStore
from .tasks import TaskStimuli

# Provider -> environment variables that hold its credentials (any one suffices).
CREDENTIAL_ENV = {"openai": ("OPENAI_API_KEY",), "anthropic": ("ANTHROPIC_API_KEY",),
                  "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
KILL_SWITCH_ENV = "LOADTEST_LLM_KILL_SWITCH_FORCE"  # llm_guard.py:128 (SL-09)
FATAL_CODES = ("bad_request", "auth", "model_unavailable")
MISSING_STATUSES = ("invalid", "refusal", "error", "not_run")
TERMINAL = ("COMPLETED", "COMPLETED_WITH_GAPS", "CANCELLED", "FAILED")
AUTO_RESUME = "auto-resume after cooldown"


class PlanError(ValueError):
    pass


class Trial(Section):
    trial_key: Sha256
    task_id: NonEmptyStr
    k: PositiveInt
    set_id: NonEmptyStr
    presentation_id: NonEmptyStr
    interface: Interface
    model_key: Ident
    temperature: float | None
    rep: NonNegativeInt

    @property
    def cell(self) -> tuple:
        return self.task_id, self.k, self.interface, self.model_key, self.temperature


class Stimuli(Section):
    """What the trials show: SL-05 tasks, designs and the eligible products."""

    tasks: list[TaskStimuli] = Field(min_length=1)
    designs: list[Design] = Field(min_length=1)
    products: dict[NonEmptyStr, Product]


# ------------------------------------------------------------------------ mode
def detect_credentials(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    environ = os.environ if environ is None else environ
    return {provider: any(environ.get(name) for name in names) for provider, names in CREDENTIAL_ENV.items()}


def resolve_mode(config: RunConfig, credentials: Mapping[str, bool]) -> tuple[str, str]:
    """(resolved mode, reason), or PlanError (tickets.md section 3.4)."""
    providers = list(dict.fromkeys(m.provider for m in config.agents.models if m.provider in CREDENTIAL_ENV))
    missing = [p for p in providers if not credentials.get(p)]
    backend = config.agents.offline_backend
    if config.mode == "offline":
        return "offline", "mode offline; backend %s" % backend
    if config.mode == "replay":
        return "replay", "mode replay of run %s" % config.source_run_id
    if config.mode == "sandbox":
        if not providers:
            raise PlanError("mode sandbox: no API provider is configured")
        if missing:
            raise PlanError("mode sandbox: no credentials for %s; sandbox never falls back to offline"
                            % ", ".join(missing))
        return "sandbox", "mode sandbox; credentials for %s" % ", ".join(providers)
    if not providers:
        return "offline", "mode auto: no API provider configured; backend %s" % backend
    if len(missing) == len(providers):
        return "offline", "no credentials for %s; backend %s" % (", ".join(providers), backend)
    if missing:
        raise PlanError("mode auto: credentials for %s but none for %s; add them or remove those models (the "
                        "design is not changed to fit)" % (", ".join(p for p in providers if p not in missing),
                                                           ", ".join(missing)))
    return "sandbox", "mode auto: credentials for every configured provider (%s)" % ", ".join(providers)


# ------------------------------------------------------------------------ plan
def plan_trials(config: RunConfig, stimuli: Stimuli) -> list[Trial]:
    """Every (presentation, repetition, interface, model, temperature) of the designs,
    in design order."""
    if not config.agents.models:
        raise PlanError("agents.models: no model to run")
    trials = []
    for design in stimuli.designs:
        if design.repetitions != config.repetitions:
            raise PlanError("design %s/k%d has %d repetitions, config %d"
                            % (design.task_id, design.k, design.repetitions, config.repetitions))
        for s in design.sets:
            for rep in range(config.repetitions):
                presentation = design.presentation_for(s.set_id, rep)
                for interface in config.interfaces:
                    for m in config.agents.models:
                        for temperature in m.temperatures:
                            key = trial_key(config.config_hash, design.task_id, s.set_id,
                                            presentation.presentation_id, interface, m.key, temperature, rep)
                            trials.append(Trial(trial_key=key, task_id=design.task_id, k=design.k, set_id=s.set_id,
                                                presentation_id=presentation.presentation_id, interface=interface,
                                                model_key=m.key, temperature=temperature, rep=rep))
    if len({t.trial_key for t in trials}) != len(trials):
        raise PlanError("duplicate trial keys")
    return trials


@dataclass
class Plan:
    manifest: Manifest
    trials: list[Trial]
    estimate_usd: float | None


def plan(config: RunConfig, stimuli: Stimuli, credentials: Mapping[str, bool], *,
         estimate_cost: Callable[[RunConfig, list[Trial], str], float | None] | None = None,
         overrides: Mapping | None = None, run_id: str | None = None,
         now: datetime.datetime | None = None) -> Plan:
    if config.environment.catalog != "amazon_reviews_2023":
        raise PlanError("the runner plans E1 designs (catalog amazon_reviews_2023); got %s"
                        % config.environment.catalog)
    unknown = sorted(set(config.protocol.prompt_templates.values()) - set(TEMPLATES))
    if unknown:
        raise PlanError("protocol.prompt_templates: no renderer for %s" % unknown)
    resolved, reason = resolve_mode(config, credentials)
    trials = plan_trials(config, stimuli)
    # Offline and replay runs call no paid provider.
    estimate = estimate_cost(config, trials, resolved) if estimate_cost else (
        0.0 if resolved in ("offline", "replay") else None)
    now = now or _utcnow()
    manifest = new_manifest(config, resolved_mode=resolved, mode_reason=reason, governance="local",
                            protocol_version=PROTOCOL_VERSION, overrides=overrides, run_id=run_id, now=now,
                            prompt_templates={t: canonical_json(TEMPLATES[t][2])
                                              for t in config.protocol.prompt_templates.values()})
    approval = config.budget.approval_usd
    if estimate is None:
        append_status(manifest, "PENDING_APPROVAL", now, "no cost estimate for mode %s (SL-11)" % resolved)
    elif approval is not None and estimate > approval:
        append_status(manifest, "PENDING_APPROVAL", now, "estimate $%.2f above approval_usd $%.2f"
                      % (estimate, approval))
    else:
        append_status(manifest, "QUEUED", now, "%d trials; estimate $%.2f" % (len(trials), estimate))
    return Plan(manifest=manifest, trials=trials, estimate_usd=estimate)


# ------------------------------------------------------------------ interfaces
class Backend(Protocol):
    backend: str  # protocol Backend value, e.g. "simulated"
    model: str  # the model id outcomes name

    def decide(self, trial: Trial, observation: Observation) -> TrialOutcome: ...


class Governance(Protocol):
    kind: str

    def authorize(self, trial: Trial, estimated_usd: float) -> str | None:
        """None to run the trial, "budget" or "kill_switch" to stop it."""

    def record(self, trial: Trial, outcome: TrialOutcome, cost_usd: float) -> None: ...

    def cancel_requested(self) -> bool: ...


def kill_switch_engaged(environ: Mapping[str, str]) -> bool:
    """``llm_guard.py`` semantics for the forced value: set to anything but "0" engages it."""
    value = environ.get(KILL_SWITCH_ENV)
    return value is not None and value.strip() != "0"


class LocalGovernance:
    """Budget, kill switch, cancel flag, a JSONL call log of finished trials and a JSONL
    invocation log of provider calls (one line per call, SL-09) in the run directory."""

    kind = "local"

    def __init__(self, run_id: str, max_usd: float, *, call_log: Path | None = None, cancel_flag: Path | None = None,
                 environ: Mapping[str, str] | None = None, invocation_log: Path | None = None):
        self.run_id, self.max_usd = run_id, max_usd
        self.call_log, self.cancel_flag, self.invocation_log = call_log, cancel_flag, invocation_log
        self.environ = os.environ if environ is None else environ
        self.spent_usd = 0.0
        self.lock = threading.Lock()
        if call_log is not None and Path(call_log).is_file():
            for line in Path(call_log).read_text(encoding="utf-8").splitlines():
                try:
                    self.spent_usd += float(json.loads(line).get("cost_usd", 0.0))
                except ValueError:
                    continue  # a line cut short by a kill

    def authorize(self, trial: Trial, estimated_usd: float) -> str | None:
        if kill_switch_engaged(self.environ):
            return "kill_switch"
        if self.spent_usd + estimated_usd > self.max_usd + 1e-12:
            return "budget"
        return None

    def record(self, trial: Trial, outcome: TrialOutcome, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        if self.call_log is not None:
            line = json.dumps({"run_id": self.run_id, "trial_key": trial.trial_key, "model_key": trial.model_key,
                               "status": outcome.status, "error_code": outcome.error_code, "cost_usd": cost_usd,
                               "at_utc": _utc_text()}, separators=(",", ":"))
            with open(self.call_log, "a", encoding="utf-8") as out:
                out.write(line + "\n")

    def cancel_requested(self) -> bool:
        return self.cancel_flag is not None and Path(self.cancel_flag).exists()

    # CallGuard (backends/provider.py), called by the adapters on their threads.
    def before_attempt(self, provider: str, model: str) -> str | None:
        return "kill_switch" if kill_switch_engaged(self.environ) else None

    def after_attempt(self, provider: str, model: str, error: BaseException | None) -> None:
        return None

    def log_call(self, call: Mapping) -> None:
        if self.invocation_log is None:
            return
        line = json.dumps({**call, "at_utc": _utc_text()}, separators=(",", ":"), default=str)
        with self.lock, open(self.invocation_log, "a", encoding="utf-8") as out:
            out.write(line + "\n")
            out.flush()
            os.fsync(out.fileno())


# ------------------------------------------------------------------- execution
def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _utc_text() -> str:
    return _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Stop:
    status: str  # PAUSED, FAILED, LIMIT_REACHED, CANCELLED
    reason: str
    auto_resume: bool = False


class _Monitor:
    """Run effects of finished trials: breaker, rate-limit window, fatal codes, drift."""

    def __init__(self, config: RunConfig, manifest: Manifest, clock: Callable[[], float]):
        self.policy, self.manifest, self.clock = config.failure_policy, manifest, clock
        self.unavailable = 0
        self.window: collections.deque = collections.deque()

    def observe(self, trial: Trial, outcome: TrialOutcome, at: datetime.datetime) -> Stop | None:
        code, policy = outcome.error_code, self.policy
        if outcome.status == "error" and code in FATAL_CODES:
            return Stop("FAILED", "%s on trial %s (model %s)" % (code, trial.trial_key[:12], trial.model_key))
        drift = None
        if outcome.returned_model is not None:
            record = next((m for m in self.manifest.models if m.key == trial.model_key), None)
            if record is not None:
                before = list(record.returned_ids)
                record_model_access(self.manifest, trial.model_key, outcome.returned_model, at)
                if before and outcome.returned_model not in before:
                    drift = Stop("PAUSED", "model_drift: %s returned %s after %s" % (
                        trial.model_key, outcome.returned_model, ", ".join(before)))
        self.unavailable = self.unavailable + 1 if code == "provider_unavailable" else 0
        now = self.clock()
        self.window.append((now, code == "rate_limited"))
        while self.window and self.window[0][0] < now - policy.rate_limit_window_s:
            self.window.popleft()
        if drift is not None:  # resuming after drift is a human decision, so it wins over auto-resumable pauses
            return drift
        if self.unavailable >= policy.breaker.failure_threshold:
            return Stop("PAUSED", "breaker_open: %d consecutive provider_unavailable" % self.unavailable, True)
        limited = sum(flag for _, flag in self.window)
        if (len(self.window) >= policy.breaker.failure_threshold
                and limited / len(self.window) > policy.rate_limit_pause_share):
            return Stop("PAUSED", "rate_limit_storm: %d of %d trials in %gs rate limited"
                        % (limited, len(self.window), policy.rate_limit_window_s), True)
        return None


@dataclass
class Result:
    status: str
    reason: str | None
    executed: int  # records written by this call


def _set(manifest: Manifest, store: ArtifactStore, status: str, reason: str | None) -> None:
    append_status(manifest, status, _utcnow(), reason)
    store.write_manifest(manifest)


def execute(manifest: Manifest, trials: list[Trial], stimuli: Stimuli, store: ArtifactStore,
            backends: Mapping[str, Backend], governance: Governance, *,
            cost: Callable[[Trial, TrialOutcome | None], float] = lambda trial, outcome: 0.0,
            clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
            flush_every: int = 50) -> Result:
    """Run the trials without a final record; returns the run's new status."""
    config = manifest.config
    current = manifest.status_history[-1].status
    if current in TERMINAL:
        raise ValueError("run %s is %s" % (manifest.run_id, current))
    if current == "PENDING_APPROVAL":
        raise ValueError("run %s awaits approval: %s" % (manifest.run_id, manifest.status_history[-1].reason))
    missing_backends = sorted({t.model_key for t in trials} - set(backends))
    if missing_backends:
        raise ValueError("no backend for models %s" % missing_backends)
    policy = config.failure_policy
    writer = store.open_writer(flush_every)
    records = writer.known
    specs = {t.task.task_id: t.task for t in stimuli.tasks}
    designs = {s.set_id: d for d in stimuli.designs for s in d.sets}
    observations: dict[str, Observation] = {}

    def observation(trial: Trial) -> Observation:
        if trial.presentation_id not in observations:
            presentation = designs[trial.set_id].presentation_for(trial.set_id, trial.rep)
            observations[trial.presentation_id] = build_observation(config, specs[trial.task_id], presentation,
                                                                    stimuli.products)
        return observations[trial.presentation_id]

    def write(trial: Trial, outcome: TrialOutcome, started: datetime.datetime) -> None:
        obs = observation(trial)
        record = TrialRecord(run_id=manifest.run_id, trial_key=trial.trial_key, config_hash=manifest.config_hash,
                             task_id=trial.task_id, set_id=trial.set_id, presentation_id=trial.presentation_id,
                             interface=trial.interface, model_key=trial.model_key, temperature=trial.temperature,
                             rep=trial.rep, candidate_ids=sorted(obs.candidate_ids), outcome=outcome,
                             started_utc=started, finished_utc=max(started, _utcnow()))
        writer.append(record)

    auto_resumes = sum(e.status == "QUEUED" and (e.reason or "").startswith(AUTO_RESUME)
                       for e in manifest.status_history)
    executed_before = writer.written
    _set(manifest, store, "RUNNING", None)
    stop: Stop | None = None
    try:
        while True:
            stop = _drive(trials, records, backends, governance, observation, write, cost,
                          _Monitor(config, manifest, clock), clock, policy.trial_timeout_s,
                          config.budget.max_concurrency)
            if stop is not None and stop.auto_resume:
                if auto_resumes >= policy.breaker.max_auto_resumes:
                    stop = Stop("FAILED", "%s; auto-resume limit %d reached" % (stop.reason,
                                                                                policy.breaker.max_auto_resumes))
                else:
                    writer.flush()
                    _set(manifest, store, "PAUSED", stop.reason)
                    sleep(policy.breaker.cooldown_s)
                    auto_resumes += 1
                    _set(manifest, store, "QUEUED", "%s (%d of %d)" % (AUTO_RESUME, auto_resumes,
                                                                       policy.breaker.max_auto_resumes))
                    _set(manifest, store, "RUNNING", None)
                    continue
            break
    finally:
        writer.flush()
    if stop is not None:
        _set(manifest, store, stop.status, stop.reason)
        return Result(stop.status, stop.reason, writer.written - executed_before)
    status, reason = completion(trials, records, policy.missingness_threshold)
    _set(manifest, store, status, reason)
    return Result(status, reason, writer.written - executed_before)


def _start(function, *args) -> cf.Future:
    """Run ``function`` on its own daemon thread: a call abandoned after the trial
    timeout holds no worker and cannot keep the process alive."""
    future: cf.Future = cf.Future()

    def target() -> None:
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)

    threading.Thread(target=target, daemon=True).start()
    return future


def _drive(trials, records, backends, governance, observation, write, cost, monitor, clock, timeout,
           concurrency) -> Stop | None:
    """Submit pending trials and record their outcomes until done or a stop."""
    pending = iter([t for t in trials if t.trial_key not in records or records[t.trial_key].outcome.status
                    == "not_run"])
    inflight: dict[cf.Future, tuple[Trial, datetime.datetime, float]] = {}
    stop: Stop | None = None

    def merge(new: Stop | None) -> None:
        nonlocal stop
        if new is not None and (stop is None or (new.status == "FAILED" and stop.status != "FAILED")):
            stop = new

    def finish(trial: Trial, outcome: TrialOutcome, started: datetime.datetime) -> None:
        try:
            write(trial, outcome, started)
        except ValueError as error:  # e.g. a chosen id outside the candidates: a backend defect
            merge(Stop("FAILED", "backend %s returned an invalid outcome for trial %s: %s"
                       % (backends[trial.model_key].backend, trial.trial_key[:12], error)))
            return
        governance.record(trial, outcome, cost(trial, outcome))
        if outcome.status == "not_run" and outcome.error_code == "kill_switch":  # engaged while in flight
            merge(Stop("CANCELLED", "kill_switch"))
        merge(monitor.observe(trial, outcome, _utcnow()))

    while True:
        while stop is None and len(inflight) < concurrency:
            trial = next(pending, None)
            if trial is None:
                break
            backend = backends[trial.model_key]
            if governance.cancel_requested():
                merge(Stop("CANCELLED", "cancelled by user"))
                break
            verdict = governance.authorize(trial, cost(trial, None))
            if verdict is not None:
                outcome = TrialOutcome(status="not_run", error_code=verdict, backend=backend.backend,
                                       requested_model=backend.model)
                write(trial, outcome, _utcnow())
                merge(Stop("CANCELLED", "kill_switch") if verdict == "kill_switch" else
                      Stop("LIMIT_REACHED", "budget: spent $%.4f of max_usd" % getattr(governance, "spent_usd", 0.0)))
                break
            obs = observation(trial)
            inflight[_start(backend.decide, trial, obs)] = (trial, _utcnow(), clock() + timeout)
        if not inflight:
            return stop
        wait_s = max(0.0, min(deadline for _, _, deadline in inflight.values()) - clock())
        done, _ = cf.wait(list(inflight), timeout=wait_s, return_when=cf.FIRST_COMPLETED)
        for future in done:
            trial, started, _ = inflight.pop(future)
            try:
                outcome = future.result()
            except Exception as error:  # a defect, not a provider failure: stop, leave the trial pending
                merge(Stop("FAILED", "backend %s raised %s on trial %s: %s" % (
                    backends[trial.model_key].backend, type(error).__name__, trial.trial_key[:12], error)))
                continue
            finish(trial, outcome, started)
        now = clock()
        for future, (trial, started, deadline) in list(inflight.items()):
            if now >= deadline and not future.done():
                del inflight[future]  # abandoned: its late result is ignored
                backend = backends[trial.model_key]
                finish(trial, TrialOutcome(status="error", error_code="provider_unavailable", attempts=1,
                                           latency_s=timeout, backend=backend.backend,
                                           requested_model=backend.model), started)


# ------------------------------------------------------------------- summaries
def missingness(trials: list[Trial], records: Mapping[str, TrialRecord]) -> dict[tuple, tuple[int, int]]:
    """cell -> (missing, planned); a trial without a final record counts as missing."""
    cells: dict[tuple, list[int]] = {}
    for trial in trials:
        entry = cells.setdefault(trial.cell, [0, 0])
        entry[1] += 1
        record = records.get(trial.trial_key)
        if record is None or record.outcome.status in MISSING_STATUSES:
            entry[0] += 1
    return {cell: (m, n) for cell, (m, n) in cells.items()}


def completion(trials: list[Trial], records: Mapping[str, TrialRecord], threshold: float) -> tuple[str, str]:
    unfinished = sum(t.trial_key not in records or records[t.trial_key].outcome.status == "not_run" for t in trials)
    if unfinished:
        raise RuntimeError("%d trials have no final record" % unfinished)
    gaps = {cell: mn for cell, mn in missingness(trials, records).items() if mn[0] / mn[1] > threshold}
    misses = sum(records[t.trial_key].outcome.error_code == "cache_miss" for t in trials)
    if not gaps and not misses:
        return "COMPLETED", "%d trials; no cell above missingness %g" % (len(trials), threshold)
    parts = []
    if gaps:
        worst = max(gaps.items(), key=lambda item: item[1][0] / item[1][1])
        parts.append("%d cells above missingness %g (worst %s: %d of %d)" % (len(gaps), threshold, list(worst[0]),
                                                                           *worst[1]))
    if misses:
        parts.append("%d replay cache misses" % misses)
    return "COMPLETED_WITH_GAPS", "; ".join(parts)


# Fields left out of record_set_hash. Run identity: a replay or a rerun of the same
# design is another run with its own run_id and, since the mode is part of the
# config, its own config_hash and trial keys. Volatile: wall-clock values.
RUN_IDENTITY_FIELDS = ("run_id", "config_hash", "trial_key")
VOLATILE_FIELDS = ("started_utc", "finished_utc", "outcome.latency_s")


def comparable(record: TrialRecord) -> dict:
    """The record without RUN_IDENTITY_FIELDS and VOLATILE_FIELDS."""
    data = record.model_dump(mode="json")
    for path in RUN_IDENTITY_FIELDS + VOLATILE_FIELDS:
        *parents, leaf = path.split(".")
        node = data
        for part in parents:
            node = node[part]
        del node[leaf]
    return data


def record_set_hash(records: Mapping[str, TrialRecord]) -> str:
    """sha256 over the comparable records sorted by their trial coordinates (task, set,
    presentation, interface, model, temperature, repetition), so a killed-and-resumed
    run, a rerun and a replay of the same design give the same hash."""
    rows = sorted((canonical_json([r.task_id, r.set_id, r.presentation_id, r.interface, r.model_key, r.temperature,
                                   r.rep]), comparable(r)) for r in records.values())
    return hashlib.sha256(canonical_json([row for _, row in rows]).encode("utf-8")).hexdigest()


def summary(manifest: Manifest, trials: list[Trial], records: Mapping[str, TrialRecord]) -> dict:
    statuses = collections.Counter(records[t.trial_key].outcome.status for t in trials if t.trial_key in records)
    finished = sum(statuses[s] for s in statuses if s != "not_run")
    cells = missingness(trials, records)
    threshold = manifest.config.failure_policy.missingness_threshold
    last = manifest.status_history[-1]
    return {
        "run_id": manifest.run_id, "status": last.status, "reason": last.reason,
        "mode": manifest.mode.model_dump(), "planned": len(trials), "finished": finished,
        "pending": len(trials) - finished, "by_status": dict(sorted(statuses.items())),
        "refusal_rate": statuses["refusal"] / finished if finished else 0.0,
        "cells": len(cells), "cells_above_missingness": sum(m / n > threshold for m, n in cells.values()),
        "record_set_hash": record_set_hash({t.trial_key: records[t.trial_key] for t in trials
                                            if t.trial_key in records}),
    }
