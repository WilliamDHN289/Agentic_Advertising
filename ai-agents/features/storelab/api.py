"""Python API of the runner (SL-07); ``cli.py`` mirrors it.

    plan_run(config)      plan a run: resolve the mode, build the stimuli, write the run directory
    run(config)           plan and execute
    execute_run(run_dir)  execute a planned or interrupted run (``resume`` without a status change)
    resume(run_dir)       continue a PAUSED, LIMIT_REACHED or interrupted run
    status(run_dir)       a summary: status, counts by outcome, missingness, record-set hash
    cancel(run_dir)       stop a running run (cancel flag) or cancel a stopped one
    export(run_dir, fmt)  write export/trials.<parquet|csv|jsonl.gz>, one row per planned trial

Run directories live under ``outputs.root`` (default ``runs``) as ``<run_id>/``. The
offline backends are the simulated agents (SL-06) and the local LLM (SL-04, text and
structured interfaces); provider adapters (SL-09) and replay (SL-08) are not built
yet, so planning works in those modes but executing them is refused.
"""
from __future__ import annotations

import csv
import datetime
import io
import os
import time
from pathlib import Path
from typing import Any, Mapping

from . import e1_stimuli
from .backends.local_llm import LocalLLM
from .backends.simulated import SimulatedAgent, params_from_config
from .core.config import RunConfig
from .core.protocol import Observation, TrialOutcome
from .core.records import TrialRecord, write_jsonl_gz, write_parquet
from .core.manifest import append_status
from .core.runner import (TERMINAL, LocalGovernance, Plan, Result, Stimuli, Trial, detect_credentials, execute, plan,
                          summary)
from .core.store import LocalStore, write_bytes
from .legacy_baseline import default_artifact_root

EXPORT_FORMATS = ("parquet", "csv", "jsonl")


def default_snapshot(config: RunConfig) -> Path:
    dataset = config.environment.dataset
    if dataset is None:
        raise ValueError("environment.dataset: no snapshot configured")
    return default_artifact_root() / "datasets" / dataset.name / dataset.sha256


def build_stimuli(config: RunConfig, snapshot: Path | None = None) -> Stimuli:
    built = e1_stimuli.build(config, snapshot or default_snapshot(config))
    return Stimuli(tasks=built.tasks, designs=built.designs, products=built.products)


def plan_run(config: RunConfig, *, snapshot: Path | None = None, root: Path | None = None,
             environ: Mapping[str, str] | None = None, overrides: Mapping[str, Any] | None = None,
             run_id: str | None = None) -> tuple[Path, Plan]:
    stimuli = build_stimuli(config, snapshot)
    result = plan(config, stimuli, detect_credentials(environ), overrides=overrides, run_id=run_id)
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

    def decide(self, trial: Trial, observation: Observation) -> TrialOutcome:
        return self.agent.decide(observation, trial.interface, trial.trial_key)


class LocalLLMBackend:
    """One ollama model on the text or structured view; the call seed comes from the trial key."""

    backend = "local_llm"

    def __init__(self, config: RunConfig, model_key: str):
        self.config = config
        self.spec = next(m for m in config.agents.models if m.key == model_key)
        self.model = self.spec.model

    def decide(self, trial: Trial, observation: Observation) -> TrialOutcome:
        policy = self.config.failure_policy
        client = LocalLLM(self.spec.model, temperature=trial.temperature, top_p=self.spec.top_p,
                          num_predict=self.spec.max_output_tokens, timeout_s=policy.trial_timeout_s,
                          max_attempts=policy.max_attempts, backoff_s=policy.backoff_initial_s)
        return client.decide(observation.views[trial.interface], observation.candidate_ids,
                             seed=int(trial.trial_key[:8], 16))


def backends_for(config: RunConfig, stimuli: Stimuli, resolved_mode: str, base: Path | None = None) -> dict:
    if resolved_mode == "sandbox":
        raise NotImplementedError("mode sandbox needs the provider adapters (SL-09)")
    if resolved_mode == "replay":
        raise NotImplementedError("mode replay needs the response cache (SL-08)")
    if config.agents.offline_backend == "local_llm":
        if "browser_tool" in config.interfaces:
            raise NotImplementedError("browser_tool with a local LLM needs the tool loop (SL-10)")
        return {m.key: LocalLLMBackend(config, m.key) for m in config.agents.models}
    params = params_from_config(config, stimuli.tasks, base=base)
    return {m.key: SimulatedBackend(SimulatedAgent(params.model(m.key), seed=config.seed,
                                                   max_turns=config.protocol.max_turns))
            for m in config.agents.models}


# ------------------------------------------------------------------- execution
def execute_run(run_dir: Path, *, environ: Mapping[str, str] | None = None, base: Path | None = None,
                flush_every: int = 50, backends: Mapping | None = None, sleep=time.sleep) -> Result:
    store = LocalStore(run_dir)
    store.acquire()
    try:
        manifest = store.read_manifest()
        trials, stimuli = load_plan(store)
        if manifest.status_history[-1].status == "RUNNING":  # killed or crashed mid-run
            append_status(manifest, "STALLED", _now(), "the previous process ended without a final status")
            append_status(manifest, "QUEUED", _now(), "resumed")
            store.write_manifest(manifest)
        if backends is None:
            backends = backends_for(manifest.config, stimuli, manifest.mode.resolved, base)
        governance = LocalGovernance(manifest.run_id, manifest.config.budget.max_usd, call_log=store.call_log,
                                     cancel_flag=store.cancel_flag, environ=environ)
        store.cancel_flag.unlink(missing_ok=True)
        return execute(manifest, trials, stimuli, store, backends, governance, sleep=sleep, flush_every=flush_every)
    finally:
        store.release()


def run(config: RunConfig, **kwargs) -> tuple[Path, Result | None]:
    """Plan and execute; returns (run_dir, None) if the plan awaits approval."""
    plan_kwargs = {k: kwargs.pop(k) for k in ("snapshot", "root", "overrides", "run_id") if k in kwargs}
    run_dir, planned = plan_run(config, environ=kwargs.get("environ"), **plan_kwargs)
    if planned.manifest.status_history[-1].status != "QUEUED":
        return run_dir, None
    return run_dir, execute_run(run_dir, **kwargs)


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
