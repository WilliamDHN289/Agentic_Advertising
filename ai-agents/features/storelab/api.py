"""Python API of the runner (SL-07); ``cli.py`` mirrors it.

    plan_run(config)      plan a run: resolve the mode, build the stimuli, write the run directory
    run(config)           plan and execute
    execute_run(run_dir)  execute a planned or interrupted run (``resume`` without a status change)
    resume(run_dir)       continue a PAUSED, LIMIT_REACHED or interrupted run
    status(run_dir)       a summary: status, counts by outcome, missingness, record-set hash
    cancel(run_dir)       stop a running run (cancel flag) or cancel a stopped one
    export(run_dir, fmt)  write export/trials.<parquet|csv|jsonl.gz>, one row per planned trial
    replay(run_dir)       a new run with mode replay of a finished run, served from the cache (SL-08)

Run directories live under ``outputs.root`` (default ``runs``) as ``<run_id>/``. The
offline backends are the simulated agents (SL-06) and the local LLM (SL-04, text and
structured interfaces; it has no tool calling for ``browser_tool``, SL-10) behind the
response cache; sandbox runs call the SL-09 provider
adapters (``backends/provider.py``; tests pass mocks as ``providers``) with the keys of
``StorelabSettings``, and only models that are pinned, allowed and priced. A replay plans the
source run's stimuli again under its own config_hash and serves every call from the
cache (``read_only``), so it makes no provider call.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from . import e1_stimuli
from .backends.browser_tool import simulated_decision
from .backends.llm import LLMBackend, OllamaProvider, Provider
from .backends.provider import CallGuard, adapters_for
from .backends.simulated import SimulatedAgent, params_from_config
from .core.cache import LocalAuditLog, LocalCacheStore, ResponseCache
from .core.config import RunConfig, resolve_config, trial_affecting
from .core.protocol import Observation, TrialOutcome
from .core.records import TrialRecord, write_jsonl_gz, write_parquet
from .core.manifest import append_status
from .core.runner import (TERMINAL, Decision, LocalGovernance, Plan, PlanError, Result, Stimuli, Trial,
                          detect_credentials, execute, plan, summary)
from .core.storefront import PAGE_INTERFACE
from .core.store import LocalStore, write_bytes
from .governance import Platform, PlatformGovernance
from .legacy_baseline import default_artifact_root
from .settings import StorelabSettings, model_problems

EXPORT_FORMATS = ("parquet", "csv", "jsonl")
# What a replay's config may change relative to its source run.
REPLAY_FREE_FIELDS = ("mode", "source_run_id", "cache.policy")


def default_cache_root() -> Path:
    return default_artifact_root() / "cache"


def open_cache(cache_root: Path | None = None) -> ResponseCache:
    root = Path(cache_root or default_cache_root())
    return ResponseCache(LocalCacheStore(root), LocalAuditLog(root / "audit.jsonl"))


def default_snapshot(config: RunConfig) -> Path:
    dataset = config.environment.dataset
    if dataset is None:
        raise ValueError("environment.dataset: no snapshot configured")
    return default_artifact_root() / "datasets" / dataset.name / dataset.sha256


def build_stimuli(config: RunConfig, snapshot: Path | None = None) -> Stimuli:
    built = e1_stimuli.build(config, snapshot or default_snapshot(config))
    return Stimuli(tasks=built.tasks, designs=built.designs, products=built.products)


def _without(data: dict, paths) -> dict:
    data = json.loads(json.dumps(data))
    for path in paths:
        *parents, leaf = path.split(".")
        node = data
        for part in parents:
            node = node.get(part, {})
        node.pop(leaf, None)
    return data


def replay_source(config: RunConfig, root: Path | None = None) -> tuple[Path, Stimuli]:
    """The source run of a replay config and its stimuli; PlanError if the configs
    differ in anything but REPLAY_FREE_FIELDS."""
    try:
        source_dir = resolve_run_dir(config.source_run_id, root or config.outputs.root)
    except FileNotFoundError as error:
        raise PlanError("replay: source run %s not found (%s)" % (config.source_run_id, error))
    store = LocalStore(source_dir)
    source = store.read_manifest().config
    mine, theirs = (_without(trial_affecting(c), REPLAY_FREE_FIELDS) for c in (config, source))
    if mine != theirs:
        differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        raise PlanError("replay: config differs from source run %s in %s (only %s may change)"
                        % (config.source_run_id, differing, ", ".join(REPLAY_FREE_FIELDS)))
    return source_dir, load_plan(store)[1]


def plan_run(config: RunConfig, *, snapshot: Path | None = None, root: Path | None = None,
             environ: Mapping[str, str] | None = None, overrides: Mapping[str, Any] | None = None,
             run_id: str | None = None, estimate_cost=None) -> tuple[Path, Plan]:
    if config.mode == "replay":
        _, stimuli = replay_source(config, root)
    else:
        stimuli = build_stimuli(config, snapshot)
    result = plan(config, stimuli, detect_credentials(environ), overrides=overrides, run_id=run_id,
                  estimate_cost=estimate_cost)
    if result.manifest.mode.resolved == "sandbox":
        problems = model_problems(config, StorelabSettings.from_env(environ))
        if problems:
            raise PlanError("mode sandbox: %s" % "; ".join(problems))
    run_dir = Path(root or config.outputs.root) / result.manifest.run_id
    if run_dir.exists():
        raise FileExistsError("run directory exists: %s" % run_dir)
    store = LocalStore(run_dir)
    store.write_plan([t.model_dump(mode="json") for t in result.trials], stimuli.model_dump(mode="json"))
    store.write_manifest(result.manifest)
    return run_dir, result


def load_plan(store: LocalStore) -> tuple[list[Trial], Stimuli]:
    trials, stimuli = store.read_plan()
    return [Trial.model_validate(t) for t in trials], Stimuli.model_validate(stimuli)


# -------------------------------------------------------------------- backends
class SimulatedBackend:
    backend = "simulated"

    def __init__(self, agent: SimulatedAgent):
        self.agent, self.model = agent, agent.model

    def decide(self, trial: Trial, observation: Observation) -> TrialOutcome | Decision:
        if trial.interface == PAGE_INTERFACE:  # through the storefront's tools, as a model would (SL-10)
            return simulated_decision(self.agent, trial, observation, self.agent.max_turns)
        return self.agent.decide(observation, trial.interface, trial.trial_key)


def backends_for(config: RunConfig, stimuli: Stimuli, resolved_mode: str, base: Path | None = None, *,
                 providers: Mapping[str, Provider] | None = None, cache: ResponseCache | None = None,
                 run_id: str | None = None, guard: CallGuard | None = None) -> dict:
    """One backend per model key. ``providers`` maps provider names to adapters
    (SL-09; tests pass mocks); the local model defaults to OllamaProvider. ``guard``
    receives one ``log_call`` per provider call."""
    policy = config.cache.policy
    providers = dict(providers or {})
    if resolved_mode == "replay":
        return {m.key: LLMBackend(config, m.key, provider=None, cache=cache, policy="read_only", run_id=run_id)
                for m in config.agents.models}
    if resolved_mode == "sandbox":
        missing = sorted({m.provider for m in config.agents.models} - set(providers))
        if missing:
            raise ValueError("mode sandbox: no provider adapter for %s" % ", ".join(missing))
    elif config.agents.offline_backend == "local_llm":
        providers.setdefault("ollama", OllamaProvider(config))
    else:
        params = params_from_config(config, stimuli.tasks, base=base)
        return {m.key: SimulatedBackend(SimulatedAgent(params.model(m.key), seed=config.seed,
                                                       max_turns=config.protocol.max_turns))
                for m in config.agents.models}
    if PAGE_INTERFACE in config.interfaces and "ollama" in providers:
        raise ValueError("browser_tool needs tool calling, which the local model backend lacks; use the simulated "
                         "backend or an API provider")
    return {m.key: LLMBackend(config, m.key, provider=providers[m.provider], cache=cache if policy != "off" else None,
                              policy=policy, run_id=run_id, guard=guard) for m in config.agents.models}


# ------------------------------------------------------------------- execution
def execute_run(run_dir: Path, *, environ: Mapping[str, str] | None = None, base: Path | None = None,
                flush_every: int = 50, backends: Mapping | None = None, sleep=time.sleep,
                providers: Mapping[str, Provider] | None = None, cache_root: Path | None = None,
                platform: Platform | None = None, http_clients: Mapping[str, Any] | None = None) -> Result:
    """``providers`` replaces the adapters built from ``StorelabSettings`` (tests);
    ``http_clients`` gives those adapters their SDK HTTP clients (recording);
    ``platform`` selects ``PlatformGovernance`` instead of ``LocalGovernance``."""
    store = LocalStore(run_dir)
    store.acquire()
    try:
        manifest = store.read_manifest()
        trials, stimuli = load_plan(store)
        if manifest.status_history[-1].status == "RUNNING":  # killed or crashed mid-run
            append_status(manifest, "STALLED", _now(), "the previous process ended without a final status")
            append_status(manifest, "QUEUED", _now(), "resumed")
            store.write_manifest(manifest)
        config = manifest.config
        if platform is not None:
            governance = PlatformGovernance(manifest.run_id, config.budget.max_usd, platform,
                                            cancel_flag=store.cancel_flag)
        else:
            governance = LocalGovernance(manifest.run_id, config.budget.max_usd, call_log=store.call_log,
                                         cancel_flag=store.cancel_flag, environ=environ,
                                         invocation_log=store.invocation_log)
        if backends is None:
            cache = open_cache(cache_root) if config.cache.policy != "off" else None
            if manifest.mode.resolved == "sandbox" and providers is None:
                providers = adapters_for(config, StorelabSettings.from_env(environ), governance, http_clients)
            backends = backends_for(config, stimuli, manifest.mode.resolved, base, providers=providers,
                                    cache=cache, run_id=manifest.run_id, guard=governance)
        store.cancel_flag.unlink(missing_ok=True)
        return execute(manifest, trials, stimuli, store, backends, governance, sleep=sleep, flush_every=flush_every)
    finally:
        store.release()


def run(config: RunConfig, **kwargs) -> tuple[Path, Result | None]:
    """Plan and execute; returns (run_dir, None) if the plan awaits approval."""
    plan_kwargs = {k: kwargs.pop(k) for k in ("snapshot", "root", "overrides", "run_id", "estimate_cost")
                   if k in kwargs}
    run_dir, planned = plan_run(config, environ=kwargs.get("environ"), **plan_kwargs)
    if planned.manifest.status_history[-1].status != "QUEUED":
        return run_dir, None
    return run_dir, execute_run(run_dir, **kwargs)


def replay_config(source_run_dir: Path) -> RunConfig:
    """The source run's config with mode replay, its run id and cache policy read_only."""
    source = LocalStore(source_run_dir).read_manifest()
    data = source.config.model_dump(mode="json")
    data.update(mode="replay", source_run_id=source.run_id)
    data["cache"]["policy"] = "read_only"
    return resolve_config(data)


def replay(source_run_dir: Path, *, root: Path | None = None, cache_root: Path | None = None,
           **kwargs) -> tuple[Path, Result | None]:
    """Plan and execute a replay of ``source_run_dir`` next to it (or under ``root``)."""
    source_run_dir = Path(source_run_dir)
    return run(replay_config(source_run_dir), root=root or source_run_dir.parent, cache_root=cache_root, **kwargs)


def resume(run_dir: Path, *, max_usd: float | None = None, **kwargs) -> Result:
    """Continue a run that is not finished. ``max_usd`` raises the budget of a
    LIMIT_REACHED run (budget is not part of config_hash)."""
    store = LocalStore(run_dir)
    manifest = store.read_manifest()
    last = manifest.status_history[-1]
    if last.status in TERMINAL:
        raise ValueError("run %s is %s and cannot be resumed" % (manifest.run_id, last.status))
    if last.status == "PENDING_APPROVAL":
        raise ValueError("run %s awaits approval: %s" % (manifest.run_id, last.reason))
    reason = "resumed by operator"
    if max_usd is not None:
        budget = manifest.config.budget.model_copy(update={"max_usd": max_usd})
        manifest.config = manifest.config.model_copy(update={"budget": budget})
        reason += "; max_usd raised to %g" % max_usd
    if last.status in ("PAUSED", "LIMIT_REACHED") or max_usd is not None:
        append_status(manifest, "QUEUED", _now(), reason)
        store.write_manifest(manifest)
    return execute_run(run_dir, **kwargs)


def cancel(run_dir: Path) -> str:
    """'requested' if a process is executing the run (it stops at the next trial),
    else the run is CANCELLED now; 'already <status>' for a finished run."""
    store = LocalStore(run_dir)
    manifest = store.read_manifest()
    last = manifest.status_history[-1].status
    if last in TERMINAL:
        return "already %s" % last
    try:
        store.acquire()
    except RuntimeError:
        store.cancel_flag.touch()
        return "requested"
    try:
        append_status(manifest, "CANCELLED", _now(), "cancelled by user")
        store.write_manifest(manifest)
    finally:
        store.release()
    return "cancelled"


def status(run_dir: Path) -> dict:
    store = LocalStore(run_dir)
    trials, _ = load_plan(store)
    return summary(store.read_manifest(), trials, store.records())


def records_in_plan_order(run_dir: Path) -> list[TrialRecord]:
    store = LocalStore(run_dir)
    trials, _ = load_plan(store)
    records = store.records()
    return [records[t.trial_key] for t in trials if t.trial_key in records]


CSV_COLUMNS = ("trial_key", "run_id", "config_hash", "task_id", "set_id", "presentation_id", "interface", "model_key",
               "temperature", "rep", "candidate_ids", "status", "chosen_id", "page_requests", "turns", "attempts",
               "error_code", "invalid_reason", "input_tokens", "output_tokens", "latency_s", "backend",
               "requested_model", "returned_model", "raw_ref", "started_utc", "finished_utc")


def _csv_row(record: TrialRecord) -> list:
    data = record.model_dump(mode="json")
    outcome = data.pop("outcome")
    usage = outcome.pop("usage")
    flat = {**data, **outcome, **usage}
    flat["candidate_ids"] = "|".join(flat["candidate_ids"])
    flat["page_requests"] = "|".join(flat["page_requests"])
    return ["" if flat.get(c) is None else flat[c] for c in CSV_COLUMNS]


def export(run_dir: Path, fmt: str) -> Path:
    if fmt not in EXPORT_FORMATS:
        raise ValueError("format %s: one of %s" % (fmt, ", ".join(EXPORT_FORMATS)))
    records = records_in_plan_order(run_dir)
    store = LocalStore(run_dir)
    if fmt == "parquet":
        path = store.export_dir / "trials.parquet"
        store.export_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(path, records, TrialRecord)
    elif fmt == "jsonl":
        path = store.export_dir / "trials.jsonl.gz"
        store.export_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_gz(path, records, TrialRecord)
    else:
        path = store.export_dir / "trials.csv"
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(_csv_row(r) for r in records)
        write_bytes(path, buffer.getvalue().encode("utf-8"))
    return path


def resolve_run_dir(value: str | os.PathLike, root: Path | None = None) -> Path:
    """A run directory given as a path, or as a run id under ``root`` (default ``runs``)."""
    path = Path(value)
    if (path / "manifest.json").is_file():
        return path
    candidate = Path(root or "runs") / str(value)
    if (candidate / "manifest.json").is_file():
        return candidate
    raise FileNotFoundError("no run at %s or %s" % (path, candidate))


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)
