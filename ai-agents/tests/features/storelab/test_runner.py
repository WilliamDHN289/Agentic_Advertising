"""SL-07 runner: error policy table (AC-3), idempotent records and exports (AC-2), mode
resolution (AC-4), plan counts (AC-5), store and lock behaviour, CLI.

Writes only to temporary directories. Kill-and-resume with real processes (AC-1) is
in test_runner_kill.py.
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import io
import itertools
import json
import math
import os
import shutil
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path

from features.storelab import api, cli
from features.storelab.config_files import read_config_data
from features.storelab.core.config import apply_overrides, resolve_config
from features.storelab.core.protocol import TrialOutcome, Usage
from features.storelab.core.records import read_jsonl_gz
from features.storelab.core.runner import (LocalGovernance, PlanError, detect_credentials, execute, missingness, plan,
                                           plan_trials, record_set_hash, resolve_mode)
from features.storelab.core.store import LocalStore, RunLocked, effective, gzip_members
from tests.features.storelab.test_catalog_amazon import fixture_snapshot

CONFIG = Path(cli.__file__).resolve().parent / "configs" / "e1_pilot.yaml"
GPT = {"key": "gpt_4o_20240806", "provider": "openai", "model": "gpt-4o-2024-08-06", "temperatures": [None]}
SMALL = {  # 1 task, v = 4, K = 2: 6 sets x 5 repetitions x 1 interface x 1 model = 30 trials
    "environment.categories": ["Home_and_Kitchen"], "environment.tasks_per_category": 1,
    "design.eligible_size": 4, "design.set_sizes": [2], "repetitions": 5, "interfaces": ["text"],
    "protocol.prompt_templates": {"text": "e1_text@1"}, "agents.models": [GPT], "budget.max_concurrency": 1,
    "failure_policy.trial_timeout_s": 5.0,
}
ROOT = Path(tempfile.mkdtemp())
SNAPSHOT = fixture_snapshot(ROOT)
DATASET = {"environment.dataset": {"name": "amazon_reviews_2023", "sha256": SNAPSHOT.name}}


def tearDownModule():
    shutil.rmtree(ROOT, ignore_errors=True)


def config(**overrides):
    return resolve_config(apply_overrides(read_config_data(CONFIG), {**DATASET, **overrides}))


def small_config(**overrides):
    return config(**{**SMALL, **overrides})


STIMULI = {}


def stimuli_for(cfg):
    key = cfg.config_hash
    if key not in STIMULI:
        STIMULI[key] = api.build_stimuli(cfg, SNAPSHOT)
    return STIMULI[key]


# --------------------------------------------------------------- fault backend
def outcome(status="choice", chosen=None, code=None, returned="gpt-4o-2024-08-06", reason=None):
    return TrialOutcome(status=status, chosen_id=chosen, error_code=code, invalid_reason=reason, attempts=1,
                        backend="openai", requested_model="gpt-4o-2024-08-06", returned_model=returned,
                        usage=Usage(input_tokens=100, output_tokens=10))


class FaultBackend:
    """Scripted provider: script(call index, trial, observation) -> TrialOutcome,
    "choice", an exception to raise, or a number of seconds to hang first."""

    backend, model = "openai", "gpt-4o-2024-08-06"

    def __init__(self, script):
        self.script, self.calls, self.lock = script, [], threading.Lock()

    def decide(self, trial, observation):
        with self.lock:
            index = len(self.calls)
            self.calls.append(trial.trial_key)
        result = self.script(index, trial, observation)
        if isinstance(result, tuple):  # (seconds to hang, then this result)
            time.sleep(result[0])
            result = result[1]
        if isinstance(result, Exception):
            raise result
        if result == "choice":
            return outcome(chosen=observation.candidates[0].product_id)
        return result


def first(n, bad, then="choice"):
    return lambda i, trial, obs: bad if i < n else then


def every(n, bad):
    return lambda i, trial, obs: bad if i % n == n - 1 else "choice"


class Harness:
    """A planned SMALL run in a temporary directory, executed with a FaultBackend."""

    def __init__(self, script, cfg=None, environ=None, cost=lambda trial, outcome: 0.0, max_usd=None):
        self.cfg = cfg or small_config()
        if max_usd is not None:
            self.cfg = resolve_config(apply_overrides(self.cfg.model_dump(mode="json"), {"budget.max_usd": max_usd}))
        self.stimuli = stimuli_for(self.cfg)
        self.plan = plan(self.cfg, self.stimuli, {})
        self.store = LocalStore(Path(tempfile.mkdtemp(dir=ROOT)) / self.plan.manifest.run_id)
        self.store.write_plan([t.model_dump(mode="json") for t in self.plan.trials],
                              self.stimuli.model_dump(mode="json"))
        self.store.write_manifest(self.plan.manifest)
        self.backend = FaultBackend(script)
        self.environ, self.cost, self.sleeps = environ or {}, cost, []

    def execute(self, flush_every=50):
        manifest = self.store.read_manifest()
        governance = LocalGovernance(manifest.run_id, manifest.config.budget.max_usd, call_log=self.store.call_log,
                                     cancel_flag=self.store.cancel_flag, environ=self.environ)
        return execute(manifest, self.plan.trials, self.stimuli, self.store, {GPT["key"]: self.backend}, governance,
                       cost=self.cost, sleep=self.sleeps.append, flush_every=flush_every)

    @property
    def records(self):
        return self.store.records()

    @property
    def history(self):
        return [(e.status, e.reason) for e in self.store.read_manifest().status_history]


# ------------------------------------------------------------- error policy
POLICY_ROWS = [
    # (row, script, config overrides, expected outcomes {(status, code): n}, run status, reason prefix)
    ("429 within the window share", every(10, outcome("error", code="rate_limited")), {},
     {("choice", None): 27, ("error", "rate_limited"): 3}, "COMPLETED_WITH_GAPS", "1 cells above missingness"),
    ("429 storm: PAUSED, auto-resumed, then FAILED", lambda i, t, o: outcome("error", code="rate_limited"),
     {"failure_policy.breaker.max_auto_resumes": 1},
     {("error", "rate_limited"): 10}, "FAILED", "rate_limit_storm"),
    ("5xx / connection error: breaker PAUSED, auto-resume, recovers", first(5, outcome("error", code="provider_unavailable")),
     {}, {("error", "provider_unavailable"): 5, ("choice", None): 25}, "COMPLETED_WITH_GAPS", "1 cells"),
    ("5xx persistent: breaker exhausts auto-resumes, FAILED", lambda i, t, o: outcome("error", code="provider_unavailable"),
     {"failure_policy.breaker.max_auto_resumes": 2}, {("error", "provider_unavailable"): 15}, "FAILED",
     "breaker_open"),
    ("timeout: a trial over trial_timeout_s", first(1, (0.6, "choice")), {"failure_policy.trial_timeout_s": 0.1},
     {("error", "provider_unavailable"): 1, ("choice", None): 29}, "COMPLETED", "30 trials"),
    ("400 bad request: FAILED on first occurrence", first(3, "choice", outcome("error", code="bad_request")), {},
     {("choice", None): 3, ("error", "bad_request"): 1}, "FAILED", "bad_request"),
    ("401 / 403: FAILED", first(2, "choice", outcome("error", code="auth")), {},
     {("choice", None): 2, ("error", "auth"): 1}, "FAILED", "auth"),
    ("404 model not found: FAILED", first(4, "choice", outcome("error", code="model_unavailable")), {},
     {("choice", None): 4, ("error", "model_unavailable"): 1}, "FAILED", "model_unavailable"),
    ("returned snapshot changes: outcome kept, PAUSED", lambda i, t, o: outcome(
        chosen=o.candidates[0].product_id, returned="gpt-4o-2024-08-06" if i < 7 else "gpt-4o-2024-11-20"), {},
     {("choice", None): 8}, "PAUSED", "model_drift"),
    ("safety refusal: behavioural data", every(3, outcome("refusal")), {},
     {("choice", None): 20, ("refusal", None): 10}, "COMPLETED_WITH_GAPS", "1 cells"),
    ("unparseable, unknown id, turn limit: invalid", every(4, outcome("invalid", reason="unknown_id")), {},
     {("choice", None): 23, ("invalid", None): 7}, "COMPLETED_WITH_GAPS", "1 cells"),
    ("replay cache miss: error, run incomplete", first(1, outcome("error", code="cache_miss")), {},
     {("error", "cache_miss"): 1, ("choice", None): 29}, "COMPLETED_WITH_GAPS", "1 replay cache misses"),
]


class ErrorPolicyTest(unittest.TestCase):
    """AC-3: per row of tickets.md section 3.5, the trial status, error code and run status."""

    def check(self, harness, expected, run_status, reason):
        result = harness.execute()
        records = harness.records
        counts = Counter((r.outcome.status, r.outcome.error_code) for r in records.values())
        self.assertEqual(dict(counts), expected)
        self.assertEqual(result.status, run_status)
        self.assertTrue(result.reason.startswith(reason), result.reason)
        self.assertEqual(harness.history[-1][0], run_status)
        # No trial is a choice after a failed backend call: each record is what the backend said.
        self.assertEqual(len(records), len(harness.backend.calls))
        return result

    def test_rows(self):
        for row, script, overrides, expected, run_status, reason in POLICY_ROWS:
            with self.subTest(row=row):
                harness = Harness(script, small_config(**overrides))
                self.check(harness, expected, run_status, reason)
                statuses = [s for s, _ in harness.history]
                cooldown = harness.cfg.failure_policy.breaker.cooldown_s
                if row.startswith("429 storm"):
                    self.assertEqual(statuses.count("PAUSED"), 1)
                    self.assertEqual(harness.sleeps, [cooldown])
                    self.assertIn("auto-resume limit 1 reached", harness.history[-1][1])
                if row.startswith("5xx / connection"):
                    self.assertEqual(harness.sleeps, [cooldown])
                    self.assertEqual(statuses[-4:], ["PAUSED", "QUEUED", "RUNNING", "COMPLETED_WITH_GAPS"])
                    self.assertIn(("QUEUED", "auto-resume after cooldown (1 of 3)"), harness.history)
                if row.startswith("5xx persistent"):
                    self.assertEqual(harness.sleeps, [cooldown, cooldown])
                if row.startswith("timeout"):
                    record = next(r for r in harness.records.values() if r.outcome.status == "error")
                    self.assertEqual(record.outcome.latency_s, 0.1)
                if row.startswith("returned snapshot"):
                    self.assertEqual(harness.sleeps, [])  # resuming is a human decision
                    manifest = harness.store.read_manifest()
                    self.assertEqual(manifest.models[0].returned_ids, ["gpt-4o-2024-08-06", "gpt-4o-2024-11-20"])
                    drifted = [r for r in harness.records.values() if r.outcome.returned_model == "gpt-4o-2024-11-20"]
                    self.assertEqual(len(drifted), 1)  # the outcome is kept
                if row.startswith("400") or row.startswith("401") or row.startswith("404"):
                    self.assertEqual(len(harness.backend.calls), sum(expected.values()))  # nothing after it

    def test_drift_resumes_only_by_operator_and_then_completes(self):
        harness = Harness(lambda i, t, o: outcome(chosen=o.candidates[0].product_id,
                                                  returned="gpt-4o-2024-08-06" if i < 7 else "gpt-4o-2024-11-20"))
        self.assertEqual(harness.execute().status, "PAUSED")
        api.resume(harness.store.run_dir, backends={GPT["key"]: harness.backend}, environ={})
        self.assertEqual(harness.history[-1][0], "COMPLETED")
        self.assertIn(("QUEUED", "resumed by operator"), harness.history)
        self.assertEqual(len(harness.records), 30)

    def test_budget_limit_then_resume_with_a_higher_budget(self):
        harness = Harness(lambda i, t, o: "choice", cost=lambda trial, outcome: 0.25, max_usd=2.0)
        result = harness.execute()
        self.assertEqual((result.status, result.reason[:7]), ("LIMIT_REACHED", "budget:"))
        counts = Counter((r.outcome.status, r.outcome.error_code) for r in harness.records.values())
        self.assertEqual(dict(counts), {("choice", None): 8, ("not_run", "budget"): 1})
        api.resume(harness.store.run_dir, max_usd=100.0, backends={GPT["key"]: harness.backend}, environ={})
        self.assertEqual(harness.history[-1][0], "COMPLETED")
        self.assertEqual(Counter(r.outcome.status for r in harness.records.values()), {"choice": 30})
        lines = sum(1 for _ in harness.store.iter_records())
        self.assertEqual(lines, 31)  # the superseded not_run line stays in the append-only shard
        self.assertEqual(len(api.records_in_plan_order(harness.store.run_dir)), 30)

    def test_kill_switch_cancels_without_calls(self):
        harness = Harness(lambda i, t, o: "choice", environ={"LOADTEST_LLM_KILL_SWITCH_FORCE": "1"})
        result = harness.execute()
        self.assertEqual((result.status, result.reason), ("CANCELLED", "kill_switch"))
        self.assertEqual(harness.backend.calls, [])
        self.assertEqual([(r.outcome.status, r.outcome.error_code, r.outcome.attempts)
                          for r in harness.records.values()], [("not_run", "kill_switch", 0)])

    def test_user_cancel_stops_at_the_next_trial(self):
        holder = {}

        def script(i, trial, obs):
            if i == 4:
                holder["store"].cancel_flag.touch()
            return "choice"

        harness = Harness(script)
        holder["store"] = harness.store
        result = harness.execute()
        self.assertEqual((result.status, result.reason), ("CANCELLED", "cancelled by user"))
        self.assertEqual(len(harness.records), 5)

    def test_a_backend_exception_fails_the_run_and_leaves_the_trial_pending(self):
        harness = Harness(first(3, "choice", RuntimeError("adapter bug")))
        result = harness.execute()
        self.assertEqual(result.status, "FAILED")
        self.assertIn("raised RuntimeError", result.reason)
        self.assertEqual(len(harness.backend.calls), 4)
        self.assertEqual(len(harness.records), 3)
        self.assertNotIn(harness.backend.calls[3], harness.records)

    def test_a_choice_outside_the_candidates_is_never_recorded(self):
        harness = Harness(first(2, "choice", outcome(chosen="NOT-A-CANDIDATE")))
        result = harness.execute()
        self.assertEqual(result.status, "FAILED")
        self.assertIn("invalid outcome", result.reason)
        self.assertEqual(len(harness.records), 2)

    def test_terminal_and_unapproved_runs_do_not_execute(self):
        harness = Harness(lambda i, t, o: "choice")
        harness.execute()
        with self.assertRaisesRegex(ValueError, "COMPLETED"):
            harness.execute()
        with self.assertRaisesRegex(ValueError, "cannot be resumed"):
            api.resume(harness.store.run_dir)


# ------------------------------------------------------------- idempotency
class IdempotencyTest(unittest.TestCase):
    """AC-2: executing a trial twice leaves one record; exports have no duplicate trial_key."""

    def test_a_second_execution_records_nothing_new(self):
        harness = Harness(lambda i, t, o: "choice")
        self.assertEqual(harness.execute().executed, 30)
        manifest = harness.store.read_manifest()
        manifest.status_history[-1].status = "QUEUED"  # force a second pass over a finished run
        harness.store.write_manifest(manifest)
        self.assertEqual(harness.execute().executed, 0)
        self.assertEqual(len(harness.backend.calls), 30)
        self.assertEqual(sum(1 for _ in harness.store.iter_records()), 30)

    def test_the_writer_refuses_a_second_final_record(self):
        harness = Harness(lambda i, t, o: "choice")
        harness.execute()
        record = next(iter(harness.records.values()))
        writer = harness.store.open_writer()
        self.assertFalse(writer.append(record))
        self.assertFalse(writer.append(record.model_copy(update={"outcome": outcome("outside", returned=None)})))
        writer.flush()
        self.assertEqual(sum(1 for _ in harness.store.iter_records()), 30)

    def test_effective_records(self):
        harness = Harness(lambda i, t, o: "choice")
        harness.execute()
        record = next(iter(harness.records.values()))
        not_run = record.model_copy(update={"outcome": TrialOutcome(status="not_run", error_code="budget",
                                                                    backend="openai")})
        self.assertIs(effective([not_run, record])[record.trial_key], record)
        self.assertIs(effective([record, not_run])[record.trial_key], record)
        later = record.model_copy(update={"outcome": outcome("outside", returned=None)})
        self.assertIs(effective([record, later])[record.trial_key], record)

    def test_exports_have_one_row_per_trial(self):
        harness = Harness(lambda i, t, o: "choice", cost=lambda trial, outcome: 1.0, max_usd=10.0)
        harness.execute()
        api.resume(harness.store.run_dir, max_usd=1000.0, backends={GPT["key"]: harness.backend}, environ={})
        run_dir = harness.store.run_dir
        rows = list(csv.DictReader(io.StringIO(api.export(run_dir, "csv").read_text(encoding="utf-8"))))
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({r["trial_key"] for r in rows}), 30)
        self.assertEqual({r["status"] for r in rows}, {"choice"})
        jsonl = read_jsonl_gz(api.export(run_dir, "jsonl"), __import__(
            "features.storelab.core.records", fromlist=["TrialRecord"]).TrialRecord)
        self.assertEqual(len({r.trial_key for r in jsonl}), len(jsonl))
        self.assertEqual(len(jsonl), 30)
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            return
        from features.storelab.core.records import TrialRecord, read_parquet
        table = read_parquet(api.export(run_dir, "parquet"), TrialRecord)
        self.assertEqual(sorted(r.trial_key for r in table), sorted(r.trial_key for r in jsonl))


# ----------------------------------------------------------- mode resolution
class ModeTest(unittest.TestCase):
    """AC-4: auto without credentials → offline with the reason; partial credentials →
    plan error naming the missing providers."""

    def test_resolution_table(self):
        pilot = config()  # openai, anthropic and google models
        none = dict.fromkeys(("openai", "anthropic", "google"), False)
        self.assertEqual(resolve_mode(pilot, none),
                         ("offline", "no credentials for openai, anthropic, google; backend simulated"))
        self.assertEqual(resolve_mode(pilot, dict.fromkeys(none, True))[0], "sandbox")
        for present, missing in ((["openai"], "anthropic, google"), (["openai", "google"], "anthropic")):
            with self.subTest(present=present):
                with self.assertRaisesRegex(PlanError, "none for %s" % missing):
                    resolve_mode(pilot, {**none, **dict.fromkeys(present, True)})
        sandbox = config(mode="sandbox")
        with self.assertRaisesRegex(PlanError, "no credentials for anthropic, google; sandbox never falls back"):
            resolve_mode(sandbox, {**none, "openai": True})
        self.assertEqual(resolve_mode(config(mode="offline"), dict.fromkeys(none, True))[0], "offline")
        replay = config(mode="replay", source_run_id="1b4e28ba-2fa1-4d2e-883f-0016d3cca427", **{"cache.policy": "read_only"})
        self.assertEqual(resolve_mode(replay, none)[0], "replay")

    def test_credentials_come_from_the_environment(self):
        self.assertEqual(detect_credentials({}), {"openai": False, "anthropic": False, "google": False})
        self.assertEqual(detect_credentials({"GEMINI_API_KEY": "x", "OPENAI_API_KEY": ""}),
                         {"openai": False, "anthropic": False, "google": True})

    def test_the_plan_records_the_resolved_mode_and_reason(self):
        cfg = small_config(**{"agents.models": [GPT, {"key": "claude_sonnet_4_20250514", "provider": "anthropic",
                                                      "model": "claude-sonnet-4-20250514", "temperatures": [None]}]})
        run_dir, planned = api.plan_run(cfg, snapshot=SNAPSHOT, root=ROOT / "plans", environ={})
        manifest = LocalStore(run_dir).read_manifest()
        self.assertEqual((manifest.mode.requested, manifest.mode.resolved), ("auto", "offline"))
        self.assertEqual(manifest.mode.reason, "no credentials for openai, anthropic; backend simulated")
        self.assertEqual(manifest.status_history[0].reason, manifest.mode.reason)
        self.assertEqual(manifest.status_history[-1].status, "QUEUED")
        with self.assertRaisesRegex(PlanError, "credentials for openai but none for anthropic"):
            api.plan_run(cfg, snapshot=SNAPSHOT, root=ROOT / "plans", environ={"OPENAI_API_KEY": "sk-test"})
        with self.assertRaisesRegex(NotImplementedError, "SL-09"):
            api.backends_for(cfg, planned and stimuli_for(cfg), "sandbox")

    def test_unestimated_sandbox_runs_wait_for_approval(self):
        cfg = small_config()
        planned = plan(cfg, stimuli_for(cfg), {"openai": True})
        self.assertEqual(planned.manifest.mode.resolved, "sandbox")
        self.assertEqual(planned.manifest.status_history[-1].status, "PENDING_APPROVAL")
        priced = plan(cfg, stimuli_for(cfg), {"openai": True}, estimate_cost=lambda c, t, m: 80.0)
        self.assertIn("above approval_usd", priced.manifest.status_history[-1].reason)
        cheap = plan(cfg, stimuli_for(cfg), {"openai": True}, estimate_cost=lambda c, t, m: 1.0)
        self.assertEqual(cheap.manifest.status_history[-1].status, "QUEUED")


# ------------------------------------------------------------------ plan count
class PlanCountTest(unittest.TestCase):
    """AC-5: plan trial counts equal the closed-form count from the design."""

    def test_the_pilot_count(self):
        cfg = config()
        trials = plan_trials(cfg, stimuli_for(cfg))
        v, r = cfg.design.eligible_size, cfg.repetitions
        sets_per_task = math.comb(v, 2) + 2 * v * (v - 1) // (4 * 3) + 3  # all pairs, (10,4,2) BIBD, 3-block covering
        temperatures = sum(len(m.temperatures) for m in cfg.agents.models)
        expected = 4 * sets_per_task * r * len(cfg.interfaces) * temperatures
        self.assertEqual(expected, 4 * 63 * 10 * 3 * 12)
        self.assertEqual(len(trials), expected)
        self.assertEqual(len({t.trial_key for t in trials}), expected)

    def test_count_formula_over_variants(self):
        variants = [{}, {"design.randomisation": "per_repetition"}, {"repetitions": 3, "design.set_sizes": [2, 4]},
                    {"interfaces": ["text", "structured"],
                     "protocol.prompt_templates": {"text": "e1_text@1", "structured": "e1_structured@1"}},
                    {"design.eligible_size": 13, "design.set_sizes": [2, 4, 8], "repetitions": 2}]
        for change in variants:
            cfg = config(**change)
            stimuli = stimuli_for(cfg)
            with self.subTest(change=change):
                expected = (sum(len(d.sets) for d in stimuli.designs) * cfg.repetitions * len(cfg.interfaces)
                            * sum(len(m.temperatures) for m in cfg.agents.models))
                self.assertEqual(len(plan_trials(cfg, stimuli)), expected)
                for d in stimuli.designs:  # the closed forms where the design has one
                    if d.k == 2:
                        self.assertEqual(len(d.sets), math.comb(len(d.eligible_ids), 2))
                    if d.design_type == "bibd":
                        lam = d.lambda_min
                        v = len(d.eligible_ids)
                        self.assertEqual(len(d.sets), lam * v * (v - 1) // (d.k * (d.k - 1)))

    def test_cells(self):
        cfg = small_config()
        trials = plan_trials(cfg, stimuli_for(cfg))
        cells = missingness(trials, {})
        self.assertEqual(len(cells), 1)
        self.assertEqual(list(cells.values()), [(30, 30)])


# ------------------------------------------------------------------ store
class StoreTest(unittest.TestCase):
    def test_a_truncated_tail_is_skipped(self):
        harness = Harness(lambda i, t, o: "choice")
        harness.execute(flush_every=7)
        shard = harness.store.shards()[0]
        data = shard.read_bytes()
        members = list(gzip_members(data))
        self.assertGreater(len(members), 1)
        with open(shard, "ab") as out:
            out.write(gzip.compress(b'{"partial": true}\n', mtime=0)[:-9])
        self.assertEqual(len(harness.store.records()), 30)
        self.assertEqual(list(gzip_members(b"not gzip")), [])

    def test_one_process_per_run(self):
        store = LocalStore(Path(tempfile.mkdtemp(dir=ROOT)) / "run")
        store.acquire()
        with self.assertRaises(RunLocked):
            store.acquire()
        store.release()
        store.lock_path.write_text("999999999")  # a dead process
        store.acquire()
        self.assertEqual(store.lock_path.read_text(), str(os.getpid()))
        store.release()


# --------------------------------------------------------------------- CLI
class CliTest(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def sets(self, **extra):
        values = {**DATASET, **SMALL, "budget.max_concurrency": 4, **extra}
        return list(itertools.chain.from_iterable(("--set", "%s=%s" % (k, json.dumps(v))) for k, v in values.items()))

    def test_plan_resume_status_export_cancel(self):
        root = Path(tempfile.mkdtemp(dir=ROOT))
        code, out, err = self.run_cli("plan", str(CONFIG), "--snapshot", str(SNAPSHOT), "--root", str(root),
                                      *self.sets())
        self.assertEqual(code, 0, err)
        planned = json.loads(out)
        self.assertEqual((planned["trials"], planned["status"], planned["mode"]["resolved"]), (30, "QUEUED", "offline"))
        code, out, err = self.run_cli("resume", planned["run_id"], "--root", str(root))
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["status"], "COMPLETED")
        code, out, _ = self.run_cli("status", planned["run_dir"])
        self.assertEqual(json.loads(out)["finished"], 30)
        code, out, _ = self.run_cli("export", planned["run_dir"], "--format", "csv")
        self.assertTrue(json.loads(out)["export"].endswith("trials.csv"))
        code, out, _ = self.run_cli("cancel", planned["run_dir"])
        self.assertEqual(json.loads(out)["cancel"], "already COMPLETED")
        code, _, err = self.run_cli("resume", planned["run_dir"])
        self.assertEqual(code, 2)
        self.assertIn("cannot be resumed", err)

    def test_run_and_errors(self):
        root = Path(tempfile.mkdtemp(dir=ROOT))
        code, out, err = self.run_cli("run", str(CONFIG), "--snapshot", str(SNAPSHOT), "--root", str(root),
                                      *self.sets())
        self.assertEqual(code, 0, err)
        summary = json.loads(out)
        self.assertEqual((summary["status"], summary["planned"], summary["pending"]), ("COMPLETED", 30, 0))
        code, _, err = self.run_cli("run", str(CONFIG), "--snapshot", str(SNAPSHOT), "--root", str(root),
                                    *self.sets(mode="sandbox"))
        self.assertEqual(code, 2)
        self.assertIn("no credentials for openai", err)
        code, _, err = self.run_cli("status", "no-such-run", "--root", str(root))
        self.assertEqual(code, 2)

    def test_cancel_a_queued_run(self):
        root = Path(tempfile.mkdtemp(dir=ROOT))
        _, out, _ = self.run_cli("plan", str(CONFIG), "--snapshot", str(SNAPSHOT), "--root", str(root), *self.sets())
        run_dir = json.loads(out)["run_dir"]
        _, out, _ = self.run_cli("cancel", run_dir)
        self.assertEqual(json.loads(out)["cancel"], "cancelled")
        self.assertEqual(api.status(Path(run_dir))["status"], "CANCELLED")


class RecordSetHashTest(unittest.TestCase):
    def test_hash_ignores_run_id_and_timing_only(self):
        harness = Harness(lambda i, t, o: "choice")
        harness.execute()
        records = harness.records
        key = next(iter(records))
        moved = dict(records)
        moved[key] = records[key].model_copy(update={"run_id": "6f1a2c3d-4b5e-4f60-9a1b-2c3d4e5f6a7b"})
        self.assertEqual(record_set_hash(moved), record_set_hash(records))
        changed = dict(records)
        changed[key] = records[key].model_copy(update={"outcome": outcome("outside", returned=None)})
        self.assertNotEqual(record_set_hash(changed), record_set_hash(records))


if __name__ == "__main__":
    unittest.main()
