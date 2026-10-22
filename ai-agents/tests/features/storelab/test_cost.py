"""SL-11 cost estimation, metering and budget enforcement: the estimator against a
hand count (AC-1), the budget under concurrency with a mock provider (AC-2), the
refusal of an unpriced model (AC-3), and the estimate-vs-actual report and gate that
SL-18 applies after the pilot (AC-4).

Uses mock providers; writes only to temporary directories.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import re
import threading
import time
import unittest
import unittest.mock
from collections import Counter

from features.storelab import api, cli, tokens
from features.storelab.core.cache import LLMRequest, ProviderResponse
from features.storelab.core.cost import (CostError, Estimator, Meter, TrialPrompt, call_usd, estimate,
                                         heuristic_tokens, jsonl, logged_spend, trial_prompt)
from features.storelab.core.protocol import Usage
from features.storelab.core.runner import LocalGovernance, PlanError, hold_until, plan_trials
from features.storelab.core.runner import _start as runner_start
from features.storelab.core.storefront import SYSTEM_PROMPT
from features.storelab.core.store import LocalStore
from features.storelab.costs import estimator as run_estimator, observations
from features.storelab.pricing import call_cost_usd
from pydantic import ValidationError
from tests.features.storelab.test_cache import Env

CLAUDE = {"key": "claude", "provider": "anthropic", "model": "claude-sonnet-4-5-20250929",
          "temperatures": [None, 0.0], "max_output_tokens": 100}  # $3 / $15 per 1M tokens
GEMINI = {"key": "gemini", "provider": "google", "model": "gemini-2.0-flash-001",
          "temperatures": [None], "max_output_tokens": 50}  # $0.10 / $0.40 per 1M tokens
GPT = {"key": "gpt", "provider": "openai", "model": "gpt-4o-2024-08-06", "temperatures": [None]}
KEYS = {"ANTHROPIC_API_KEY": "test", "GOOGLE_API_KEY": "test", "OPENAI_API_KEY": "test"}
ID = re.compile(r"ID: (\S+) \|")


def hand_config(env: Env, **overrides):
    """2 tasks (one per category), v = 4, K in {2, 3} (6 and 3 sets), r = 2, text and
    browser_tool, Claude at two temperatures with E[turns] = 3, Gemini at one with
    E[turns] unset (max_turns = 4)."""
    return env.config(**{"environment.categories": ["Home_and_Kitchen", "Electronics"],
                         "environment.tasks_per_category": 1, "design.eligible_size": 4, "design.set_sizes": [2, 3],
                         "repetitions": 2, "interfaces": ["text", "browser_tool"],
                         "protocol.prompt_templates": {"text": "e1_text@1", "browser_tool": "e1_browser_tool@1"},
                         "protocol.max_turns": 4, "agents.models": [CLAUDE, GEMINI],
                         "budget.expected_turns": {"claude": 3.0}, **overrides})


def rates(model):
    return {"claude-sonnet-4-5-20250929": (3.0, 15.0), "gemini-2.0-flash-001": (0.10, 0.40)}[model]


def fixed_prompt(trial):
    # text: 700 characters = 240 tokens; browser_tool: 1,400 = 480 tokens, longest page 700 = 240.
    if trial.interface == "text":
        return TrialPrompt("x" * 700)
    return TrialPrompt("y" * 1400, ("p" * 350, "q" * 700))


# ----------------------------------------------------------------------- AC-1
class HandCountTest(unittest.TestCase):
    """AC-1: on a fixture design the call count equals the hand count exactly and the
    USD is within ±1% of the hand computation."""

    def setUp(self):
        self.env = Env()
        self.addCleanup(self.env.cleanup)
        self.config = hand_config(self.env)
        self.trials = plan_trials(self.config, api.build_stimuli(self.config, self.env.snapshot))

    def test_call_count_and_usd_equal_the_hand_count(self):
        rows = {r.model_key: r for r in estimate(Estimator(self.config, fixed_prompt, lambda p, m, t:
                                                           heuristic_tokens(t), rates), self.trials)}
        # Presentations: 2 tasks x (6 + 3) sets x r = 2 -> 36.
        # Claude: 36 x 2 temperatures x 2 interfaces = 144 trials; calls 36 x 2 x (1 + 3) = 288.
        # Gemini: 36 x 1 x 2 = 72 trials; calls 36 x (1 + 4) = 180.
        self.assertEqual(len(self.trials), 216)
        self.assertEqual((rows["claude"].trials, rows["gemini"].trials), (144, 72))
        self.assertEqual((rows["claude"].calls, rows["gemini"].calls), (288.0, 180.0))
        self.assertEqual(sum(r.calls for r in rows.values()), 468.0)
        # Claude text call: 240 in x $3 + 100 out x $15 = $0.00222. browser_tool, T = 3, base 480,
        # step 240 + 100: in 3 x 480 + 340 x 3 = 2,460, out 300 -> $0.00738 + $0.0045 = $0.01188.
        # Per presentation 2 x (0.00222 + 0.01188) = $0.0282; x 36 = $1.0152.
        # Gemini text: 240 x $0.10 + 50 x $0.40 = $0.000044. browser_tool, T = 4, step 290:
        # in 4 x 480 + 290 x 6 = 3,660, out 200 -> $0.000446. Per presentation $0.00049; x 36 = $0.01764.
        hand = {"claude": 1.0152, "gemini": 0.01764}
        for key, usd in hand.items():
            with self.subTest(model=key):
                self.assertLessEqual(abs(rows[key].usd - usd) / usd, 0.01)
        self.assertLessEqual(abs(sum(r.usd for r in rows.values()) - 1.03284) / 1.03284, 0.01)
        # Maxima: Claude's 4th turn sends 480 + 3 x 340 = 1,500 tokens -> $0.0045 + $0.0015.
        self.assertAlmostEqual(rows["claude"].max_call_usd, 0.006)
        self.assertAlmostEqual(rows["claude"].max_trial_usd, (4 * 480 + 340 * 6) * 3e-6 + 400 * 15e-6)

    def test_the_rendered_prompts_are_counted(self):
        """The real prompts: the view, or system prompt + tools + results page and the pages."""
        stimuli = api.build_stimuli(self.config, self.env.snapshot)
        estimator = run_estimator(self.config, stimuli)
        text = next(t for t in self.trials if t.interface == "text" and t.model_key == "claude")
        browser = next(t for t in self.trials if t.interface == "browser_tool" and t.model_key == "gemini"
                       and t.presentation_id == text.presentation_id)
        observation = observations(self.config, stimuli)(text)
        view = observation.views["text"]
        self.assertEqual(estimator.trial(text).input_tokens, heuristic_tokens(view))
        prompt = trial_prompt(observation, "browser_tool")
        self.assertTrue(prompt.first.startswith(SYSTEM_PROMPT) and prompt.first.endswith(
            observation.views["browser_tool"]))
        self.assertEqual(sorted(prompt.pages), sorted(observation.pages.values()))
        base = heuristic_tokens(prompt.first)
        step = max(heuristic_tokens(p) for p in prompt.pages) + 50
        self.assertEqual(estimator.trial(browser).input_tokens, 4 * base + step * 6)
        self.assertEqual(estimator.trial(browser).output_tokens, 200)

    def test_openai_prompts_are_counted_with_tiktoken(self):
        config = hand_config(self.env, **{"agents.models": [GPT], "budget.expected_turns": {},
                                          "interfaces": ["text"], "protocol.prompt_templates": {"text": "e1_text@1"}})
        stimuli = api.build_stimuli(config, self.env.snapshot)
        trial = plan_trials(config, stimuli)[0]
        estimator = run_estimator(config, stimuli)
        view = observations(config, stimuli)(trial).views["text"]
        self.assertEqual(estimator.trial(trial).input_tokens, len(tokens.encoding().encode(view)))
        self.assertNotEqual(estimator.trial(trial).input_tokens, heuristic_tokens(view))
        self.assertEqual(tokens.encoding().encode("hello world"), [24912, 2375])  # o200k_base ids
        self.assertEqual(tokens.count_tokens("anthropic", "claude-x", "x" * 700), 240)
        with self.assertRaisesRegex(CostError, "cl100k_base"):
            tokens.count_tokens("openai", "gpt-4-0613", "text")

    def test_the_tokenizer_is_tiktokens_o200k_base(self):
        """The pinned file has the sha256 tiktoken expects, and the split pattern and
        special tokens are those of tiktoken's own o200k_base definition."""
        import tiktoken_ext.openai_public as public

        source = inspect.getsource(public.o200k_base)
        tree = ast.parse(source)
        expected_hash = next(k.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                             for k in node.keywords if k.arg == "expected_hash")
        joined = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(
            node.func, ast.Attribute) and node.func.attr == "join")
        self.assertEqual(expected_hash, tokens.ENCODING_SHA256)
        self.assertEqual("|".join(ast.literal_eval(joined.args[0])), tokens.PAT_STR)
        self.assertEqual(tokens.SPECIAL_TOKENS, {public.ENDOFTEXT: 199999, public.ENDOFPROMPT: 200018})
        self.assertTrue(tokens.encoding_path().is_file())

    def test_heuristic(self):
        self.assertEqual([heuristic_tokens("x" * n) for n in (0, 1, 35, 36, 700)], [0, 1, 12, 13, 240])

    def test_expected_turns_must_name_a_model_and_stay_within_max_turns(self):
        for turns, message in (({"nobody": 2.0}, "not a key in agents.models"),
                               ({"claude": 5.0}, "above protocol.max_turns 4"), ({"claude": 0.5}, "greater than")):
            with self.subTest(turns=turns), self.assertRaisesRegex(ValidationError, message):
                hand_config(self.env, **{"budget.expected_turns": turns})

    def test_a_sandbox_plan_writes_the_estimate_report(self):
        config = hand_config(self.env, **{"mode": "sandbox", "budget.approval_usd": 100.0, "budget.max_usd": 100.0})
        run_dir, planned = api.plan_run(config, snapshot=self.env.snapshot, root=self.env.runs, environ=KEYS)
        report = json.loads((run_dir / "reports" / "cost_estimate.json").read_text(encoding="utf-8"))
        self.assertEqual(planned.manifest.status_history[-1].status, "QUEUED")
        self.assertAlmostEqual(report["totals"]["usd"], planned.estimate_usd)
        self.assertEqual((report["totals"]["trials"], report["totals"]["calls"]), (216, 468.0))
        self.assertEqual(report["expected_turns"], {"claude": 3.0, "gemini": 4.0})
        self.assertEqual(report["spend_bound_usd"], 100.0 + config.budget.max_concurrency * report["max_call_usd"])
        self.assertIn("| claude (`claude-sonnet-4-5-20250929`) | 144 | 288 | 3 |",
                      (run_dir / "reports" / "cost_estimate.md").read_text(encoding="utf-8"))
        path = self.env.root / "sandbox.json"
        path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), unittest.mock.patch.dict("os.environ", KEYS):
            self.assertEqual(cli.main(["plan", str(path), "--snapshot", str(self.env.snapshot),
                                       "--root", str(self.env.runs)]), 0)
        printed = json.loads(out.getvalue())
        self.assertAlmostEqual(printed["estimate_usd"], planned.estimate_usd)
        self.assertTrue(printed["cost_estimate"].endswith("reports/cost_estimate.json"))

    def test_offline_plans_cost_nothing(self):
        run_dir, planned = api.plan_run(hand_config(self.env, mode="auto"), snapshot=self.env.snapshot,
                                        root=self.env.runs, environ={})
        self.assertEqual((planned.manifest.mode.resolved, planned.estimate_usd), ("offline", 0.0))
        self.assertFalse((run_dir / "reports").exists())


# ----------------------------------------------------------------- mock provider
class MeteredProvider:
    """Anthropic stand-in: picks the first candidate, bills the prompt's heuristic
    tokens and the output cap times ``factor``, and records concurrency."""

    name, adapter_version = "anthropic", "mock-metered@1"

    def __init__(self, factor: float = 1.0, delay: float = 0.02):
        self.factor, self.delay = factor, delay
        self.calls, self.active, self.peak, self.lock = 0, 0, 0, threading.Lock()

    def complete(self, request: LLMRequest) -> ProviderResponse:
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        prompt = request.messages[-1].content
        usage = Usage(input_tokens=round(heuristic_tokens(prompt) * self.factor),
                      output_tokens=round(request.params["max_output_tokens"] * self.factor))
        return ProviderResponse(text='{"choice": "%s", "reason": "fits"}' % ID.findall(prompt)[0],
                                returned_model=request.requested_model, usage=usage)


def budget_config(env: Env, **overrides):
    """One task, v = 4, K = 2 (6 sets), r = 10, text, Claude: 60 single-call trials."""
    return env.config(**{"agents.models": [dict(CLAUDE, temperatures=[None], max_output_tokens=256)],
                         "mode": "sandbox", "repetitions": 10, "cache.policy": "off", "budget.max_concurrency": 4,
                         "budget.approval_usd": None, **overrides})


def spend(run_dir) -> list[dict]:
    return list(jsonl(LocalStore(run_dir).invocation_log))


def run(env: Env, config, provider, **kwargs):
    return api.run(config, snapshot=env.snapshot, root=env.runs, environ=KEYS, providers={"anthropic": provider},
                   cache_root=env.cache_root, **kwargs)


# ----------------------------------------------------------------------- AC-2
class BudgetTest(unittest.TestCase):
    """AC-2: spend ≤ max_usd + max_concurrency × max per-call cost; the run ends
    LIMIT_REACHED and can be resumed."""

    def setUp(self):
        self.env = Env()
        self.addCleanup(self.env.cleanup)

    def check_limited_then_resumed(self, factor: float):
        config = budget_config(self.env, **{"budget.max_usd": 0.1})
        provider = MeteredProvider(factor)
        run_dir, result = run(self.env, config, provider)
        estimate_report = json.loads((run_dir / "reports" / "cost_estimate.json").read_text(encoding="utf-8"))
        calls = spend(run_dir)
        spent = sum(c["cost_usd"] for c in calls)
        max_call = max(c["cost_usd"] for c in calls)  # the largest per-call cost actually billed
        self.assertEqual(result.status, "LIMIT_REACHED")
        self.assertGreater(provider.peak, 1)  # trials were in flight together
        self.assertLessEqual(spent, 0.1 + config.budget.max_concurrency * max_call + 1e-12)
        self.assertLessEqual(spent, estimate_report["max_usd"] + 4 * max(max_call, estimate_report["max_call_usd"]))
        records = LocalStore(run_dir).records()
        counts = Counter((r.outcome.status, r.outcome.error_code) for r in records.values())
        self.assertEqual(counts[("not_run", "budget")], 1)
        self.assertEqual(counts[("choice", None)], len(calls))
        self.assertTrue(0 < len(calls) < 60)
        # The spend is read back on resume: each call once, from the invocation log.
        self.assertAlmostEqual(LocalGovernance("x", 0.0, invocation_log=LocalStore(run_dir).invocation_log,
                                               call_log=LocalStore(run_dir).call_log).spent_usd, spent)
        resumed = api.resume(run_dir, max_usd=10.0, environ=KEYS, providers={"anthropic": provider},
                             cache_root=self.env.cache_root)
        self.assertEqual(resumed.status, "COMPLETED")
        after = spend(run_dir)
        self.assertEqual((provider.calls, len(after), len({c["trial_key"] for c in after})), (60, 60, 60))
        self.assertEqual(Counter(r.outcome.status for r in LocalStore(run_dir).records().values()), {"choice": 60})
        return spent, config

    def test_within_the_estimate_the_reservations_keep_spend_under_max_usd(self):
        spent, config = self.check_limited_then_resumed(1.0)
        self.assertLessEqual(spent, 0.1 + 1e-12)  # each call cost exactly its reservation

    def test_calls_dearer_than_estimated_stay_within_the_concurrency_bound(self):
        self.check_limited_then_resumed(1.5)

    def test_a_lost_record_is_rerun_from_the_cache_and_not_charged_twice(self):
        """A killed run keeps its invocation log but may lose unflushed records; the
        rerun trials hit the cache (read_write) and the spend counts each call once."""
        config = budget_config(self.env, **{"budget.max_usd": 0.1, "cache.policy": "read_write"})
        provider = MeteredProvider()
        run_dir, result = run(self.env, config, provider)
        self.assertEqual(result.status, "LIMIT_REACHED")
        store = LocalStore(run_dir)
        made, spent = provider.calls, sum(c["cost_usd"] for c in spend(run_dir))
        for shard in store.shards():  # every record lost
            shard.unlink()
        resumed = api.resume(run_dir, max_usd=10.0, environ=KEYS, providers={"anthropic": provider},
                             cache_root=self.env.cache_root)
        self.assertEqual(resumed.status, "COMPLETED")
        hits = list(jsonl(store.cache_log))
        self.assertEqual((len(hits), provider.calls), (made, 60))
        self.assertEqual(len(spend(run_dir)), 60)
        self.assertAlmostEqual(sum(h["cost_usd"] for h in hits), spent)
        self.assertAlmostEqual(LocalGovernance("x", 0.0, invocation_log=store.invocation_log).spent_usd,
                               sum(c["cost_usd"] for c in spend(run_dir)))

    def test_meter(self):
        meter = Meter(1.0, spent_usd=0.25)
        self.assertTrue(meter.reserve("a", 0.5))
        self.assertFalse(meter.reserve("b", 0.3))  # 0.25 + 0.5 + 0.3 > 1
        meter.charge("a", 0.2)  # a's reservation shrinks to 0.3
        self.assertAlmostEqual((meter.spent_usd, meter.reserved_usd), (0.45, 0.3))
        self.assertTrue(meter.reserve("b", 0.25))
        self.assertEqual(meter.settle("a"), 0.2)
        self.assertAlmostEqual((meter.spent_usd, meter.reserved_usd), (0.45, 0.25))
        meter.charge("a", 0.1)  # a late call of a released trial is still spend
        self.assertAlmostEqual(meter.spent_usd, 0.55)
        with self.assertRaisesRegex(CostError, "no cost_usd"):
            call_usd({"trial_key": "t", "cost_usd": None})

    def test_an_abandoned_trial_keeps_its_reservation_until_its_call_returns(self):
        meter = Meter(1.0)
        self.assertTrue(meter.reserve("slow", 0.6))
        returned = threading.Event()
        future = runner_start(lambda: returned.wait(5))
        hold_until(meter, "slow", future)
        meter.settle("slow")  # recorded as timed out
        self.assertAlmostEqual(meter.reserved_usd, 0.6)
        self.assertFalse(meter.reserve("next", 0.5))
        meter.charge("slow", 0.55)  # the late call bills
        returned.set()
        future.result(5)
        self.assertAlmostEqual((meter.spent_usd, meter.reserved_usd), (0.55, 0.0))
        self.assertFalse(meter.reserve("next", 0.5))
        self.assertTrue(meter.reserve("next", 0.45))

    def test_the_runner_holds_the_reservation_of_a_timed_out_trial(self):
        config = budget_config(self.env, **{"budget.max_usd": 10.0, "failure_policy.trial_timeout_s": 0.05,
                                            "repetitions": 1, "failure_policy.breaker.failure_threshold": 100})
        provider = MeteredProvider(delay=0.3)
        held = []
        original = LocalGovernance.hold

        def spy(governance, trial, until):
            held.append((trial.trial_key, governance.reserved_usd))
            original(governance, trial, until)

        with unittest.mock.patch.object(LocalGovernance, "hold", spy):
            run_dir, result = run(self.env, config, provider)
        self.assertEqual(result.status, "COMPLETED_WITH_GAPS")
        self.assertEqual(len(held), 6)  # every trial timed out
        self.assertTrue(all(reserved > 0 for _, reserved in held))
        self.assertEqual({r.outcome.error_code for r in LocalStore(run_dir).records().values()},
                         {"provider_unavailable"})

    def test_logged_spend_skips_a_line_cut_short(self):
        store = LocalStore(self.env.root / "r")
        store.run_dir.mkdir(parents=True)
        store.invocation_log.write_text('{"trial_key":"a","cost_usd":0.5}\n{"trial_key":"b","cost_u', encoding="utf-8")
        store.call_log.write_text('{"trial_key":"a","cost_usd":0.25}\n', encoding="utf-8")
        self.assertEqual(logged_spend(store.invocation_log, store.call_log), 0.75)


# ----------------------------------------------------------------------- AC-3
class UnpricedModelTest(unittest.TestCase):
    """AC-3: a model with no price refuses the plan."""

    def test_the_plan_is_refused_and_no_run_is_written(self):
        env = Env()
        self.addCleanup(env.cleanup)
        unpriced = {"key": "gpt_new", "provider": "openai", "model": "gpt-4o-2024-11-20", "temperatures": [None]}
        config = budget_config(env, **{"agents.models": [CLAUDE, unpriced]})
        with self.assertRaisesRegex(PlanError, "gpt_new \\(gpt-4o-2024-11-20\\): no price"):
            api.plan_run(config, snapshot=env.snapshot, root=env.runs, environ=KEYS)
        self.assertFalse(env.runs.exists() and any(env.runs.iterdir()))
        with self.assertRaisesRegex(CostError, "no price"):
            Estimator(config, fixed_prompt, lambda p, m, t: 1, rates)
        err = io.StringIO()
        path = env.root / "unpriced.json"
        path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
        with contextlib.redirect_stderr(err), unittest.mock.patch.dict("os.environ", KEYS):
            self.assertEqual(cli.main(["plan", str(path), "--snapshot", str(env.snapshot), "--root", str(env.runs)]), 2)
        self.assertIn("no price", err.getvalue())


# ----------------------------------------------------------------------- AC-4
class EstimateVsActualTest(unittest.TestCase):
    """AC-4 (the parts that need no pilot): per model |estimate − actual| / actual, the
    cache savings, and the ≤ 25% gate SL-18 applies after the pilot."""

    def setUp(self):
        self.env = Env()
        self.addCleanup(self.env.cleanup)

    def finished(self, factor, **overrides):
        config = budget_config(self.env, **{"budget.max_usd": 10.0, **overrides})
        run_dir, result = run(self.env, config, MeteredProvider(factor, delay=0.0))
        self.assertEqual(result.status, "COMPLETED")
        return run_dir

    def cli_cost(self, run_dir) -> tuple[int, dict]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["cost", str(run_dir)])
        return code, json.loads(out.getvalue())

    def test_an_accurate_estimate_passes_the_gate(self):
        run_dir = self.finished(1.0)
        report = api.cost_report(run_dir)
        (row,) = report["models"]
        self.assertEqual((row["finished_trials"], row["calls"], row["cache_hits"]), (60, 60, 0))
        self.assertAlmostEqual(row["relative_error"], 0.0, places=9)
        self.assertAlmostEqual(row["actual_usd"], sum(c["cost_usd"] for c in spend(run_dir)))
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(self.cli_cost(run_dir)[0], 0)
        self.assertIn("**passed**", (run_dir / "reports" / "cost.md").read_text(encoding="utf-8"))

    def test_an_estimate_off_by_more_than_25_percent_fails_the_gate(self):
        run_dir = self.finished(1.5)
        report = api.cost_report(run_dir)
        (row,) = report["models"]
        self.assertAlmostEqual(row["relative_error"], -1 / 3, places=2)  # (1 - 1.5) / 1.5
        self.assertEqual(report["gate"], {"passed": False, "failing_models": ["claude"]})
        code, printed = self.cli_cost(run_dir)
        self.assertEqual((code, printed["gate"]["passed"]), (1, False))
        self.assertTrue(api.cost_report(run_dir, tolerance=0.4)["gate"]["passed"])

    def test_cache_hits_are_reported_as_savings(self):
        source = self.finished(1.0, **{"cache.policy": "read_write"})
        replay_dir, result = api.replay(source, cache_root=self.env.cache_root, environ=KEYS)
        self.assertEqual(result.status, "COMPLETED")
        self.assertFalse((replay_dir / "reports" / "cost_estimate.json").exists())  # replay plans cost $0
        original, replayed = api.cost_report(source)["models"][0], api.cost_report(replay_dir)["models"][0]
        self.assertEqual((replayed["calls"], replayed["billed_usd"], replayed["cache_hits"]), (0, 0.0, 60))
        self.assertAlmostEqual(replayed["cache_saved_usd"], original["billed_usd"])
        self.assertAlmostEqual(replayed["relative_error"], 0.0, places=9)

    def test_a_model_without_actual_cost_fails_the_gate(self):
        run_dir, _ = api.plan_run(budget_config(self.env, **{"budget.max_usd": 10.0}), snapshot=self.env.snapshot,
                                  root=self.env.runs, environ=KEYS)
        report = api.cost_report(run_dir)
        self.assertEqual((report["models"][0]["relative_error"], report["gate"]["passed"]), (None, False))

    def test_usage_is_priced_like_the_invocation_log(self):
        run_dir = self.finished(1.0)
        for call in spend(run_dir):
            self.assertAlmostEqual(call["cost_usd"], call_cost_usd(call["requested_model"], Usage(
                input_tokens=call["input_tokens"], output_tokens=call["output_tokens"])))


if __name__ == "__main__":
    unittest.main()
