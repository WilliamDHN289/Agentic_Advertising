"""SL-04 tests for the ported original environment (core/legacy_env.py).

AC-1 runs tests/features/storelab/legacy_equivalence.py in two fresh processes, the
vendored original and the port; AC-2 checks keyed-mode order invariance and resume.
Read-only.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path

from features.storelab.config_files import load_config
from features.storelab.core import legacy_env
from features.storelab.core.legacy_env import LegacyRun, episode_plan, record_set_hash, run_episodes
from features.storelab.core.protocol import Completion

HERE = Path(__file__).resolve().parent
AI_AGENTS = HERE.parents[2]
CONFIG = load_config(AI_AGENTS / "features" / "storelab" / "configs" / "legacy_2026_08.yaml")
LEGACY = CONFIG.environment.legacy
KEYED = LEGACY.model_copy(update={"rng_mode": "keyed"})


def harness(side: str) -> dict:
    result = subprocess.run([sys.executable, "-B", str(HERE / "legacy_equivalence.py"), side], cwd=AI_AGENTS,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def stub(prompt: str, seed: int) -> Completion:
    """A deterministic model: picks a listed id from a hash of the prompt; one reply in
    seven is unparseable, one in eleven names an id that is not on the page."""
    ids = re.findall(r'(?:id=|"id": ")([A-Z]{3}-\d{4})', prompt)
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    if h % 7 == 0:
        return Completion(text="I would pick the second one.", attempts=1)
    choice = "ZZZ-9999" if h % 11 == 0 else ids[h % len(ids)]
    return Completion(text=json.dumps({"choice": choice, "extras": ids[-1:], "reason": "stub"}), attempts=1)


class InterruptedRun(Exception):
    pass


class LegacySharedEquivalenceTest(unittest.TestCase):
    """AC-1: the legacy_shared port and the original produce byte-identical record
    hashes for the step 2 items, each in a fresh process with arms in the original order."""

    @classmethod
    def setUpClass(cls):
        cls.original, cls.port = harness("original"), harness("port")

    def test_every_item_hash_is_identical(self):
        self.assertEqual(sorted(self.original), sorted(self.port))
        for item in ("catalog", "pages_first_90", "pages_all_520", "prompts", "human_sim", "ivt_saved",
                     "replay_rows", "replay_events"):
            with self.subTest(item=item):
                self.assertEqual(self.port[item], self.original[item])
        self.assertEqual((self.port["catalog_size"], self.port["pages"]), (2004, 520))

    def test_the_port_reproduces_the_saved_run(self):
        self.assertEqual(self.port["ivt_saved_matches"], 520)  # every saved IVT score and filter flag
        self.assertEqual(self.port["replay_rows"], self.port["saved_rows"])  # every episodes.json row

    def test_replayed_events_equal_the_saved_events_up_to_the_platform_libm(self):
        sys.path.insert(0, str(HERE))
        import legacy_equivalence as h

        replies, current = h.saved_replies(), {}
        run = LegacyRun.from_config(CONFIG, lambda prompt, seed: Completion(text=current["reply"], attempts=1))
        self.assertEqual((run.rng_mode, run.fallback), ("legacy_shared", "first_item"))
        mine = []
        for spec in episode_plan(LEGACY):
            current["reply"] = replies.get(spec.episode, "")
            mine.extend(run.run(spec)["events"])
        self.assertEqual(len(mine), len(h.EVENTS))
        if platform.machine() == "x86_64":  # the platform of the 2026-08-14 run
            self.assertEqual(mine, h.EVENTS)
            return
        # Elsewhere random.gauss can differ in the last bit (libm); only t may differ.
        worst = 0.0
        for a, b in zip(mine, h.EVENTS):
            self.assertEqual({k: v for k, v in a.items() if k != "t"}, {k: v for k, v in b.items() if k != "t"})
            worst = max(worst, abs(a["t"] - b["t"]) / max(1.0, abs(b["t"])))
        self.assertLess(worst, 1e-14)


class KeyedOrderInvarianceTest(unittest.TestCase):
    """AC-2: three random episode orders plus one run interrupted and resumed at a
    random point all produce the same record-set hash."""

    @classmethod
    def setUpClass(cls):
        cls.plan = episode_plan(KEYED)
        cls.reference = record_set_hash(run_episodes(LegacyRun(KEYED, seed=CONFIG.seed, complete=stub), cls.plan))

    def test_three_random_orders_give_the_same_record_set_hash(self):
        rng = random.Random(404)
        for attempt in range(3):
            order = list(self.plan)
            rng.shuffle(order)
            with self.subTest(order=attempt):
                self.assertNotEqual([s.episode for s in order], [s.episode for s in self.plan])
                records = run_episodes(LegacyRun(KEYED, seed=CONFIG.seed, complete=stub), order)
                self.assertEqual(record_set_hash(records), self.reference)

    def test_a_run_interrupted_mid_episode_and_resumed_gives_the_same_hash(self):
        rng = random.Random(505)
        order = list(self.plan)
        rng.shuffle(order)
        stop_at, calls = rng.randrange(50, 400), [0]

        def failing(prompt, seed):
            calls[0] += 1
            if calls[0] == stop_at:
                raise InterruptedRun("killed during model call %d" % stop_at)
            return stub(prompt, seed)

        done: dict = {}
        with self.assertRaises(InterruptedRun):
            run_episodes(LegacyRun(KEYED, seed=CONFIG.seed, complete=failing), order, done)
        self.assertTrue(0 < len(done) < len(order))
        resumed = run_episodes(LegacyRun(KEYED, seed=CONFIG.seed, complete=stub), order, dict(done))
        self.assertEqual(len(resumed), len(order))
        self.assertEqual(record_set_hash(resumed), self.reference)

    def test_legacy_shared_is_order_dependent_so_the_check_has_power(self):
        shared = LEGACY.model_copy(update={"rng_mode": "legacy_shared"})
        order = list(self.plan)
        random.Random(606).shuffle(order)
        first = record_set_hash(run_episodes(LegacyRun(shared, seed=CONFIG.seed, complete=stub), self.plan))
        second = record_set_hash(run_episodes(LegacyRun(shared, seed=CONFIG.seed, complete=stub), order))
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, self.reference)

    def test_keyed_streams_depend_on_the_master_seed(self):
        other = record_set_hash(run_episodes(LegacyRun(KEYED, seed=CONFIG.seed + 1, complete=stub), self.plan))
        self.assertNotEqual(other, self.reference)


class OutcomeTest(unittest.TestCase):
    """Substitution only under the legacy fallback, flagged; failed calls are errors."""

    def episode(self, arm: str, reply: Completion, fallback: str, rng_mode: str = "legacy_shared") -> dict:
        legacy = LEGACY.model_copy(update={"rng_mode": rng_mode})
        spec = next(s for s in episode_plan(legacy) if s.arm == arm)
        return LegacyRun(legacy, seed=CONFIG.seed, fallback=fallback, complete=lambda p, s: reply).run(spec)

    def test_first_item_fallback_substitutes_and_flags_it(self):
        record = self.episode("gui_agent", Completion(text="no json here", attempts=1), "first_item")
        self.assertEqual(record["outcome"]["status"], "choice")
        self.assertTrue(record["outcome"]["substituted"])
        self.assertEqual(record["outcome"]["invalid_reason"], "no_json")
        self.assertEqual(record["row"]["chosen_rank"], 1)

    def test_without_the_fallback_an_unparseable_reply_buys_nothing(self):
        for arm in ("gui_agent", "chatbot", "api", "gui_agent_2x2"):
            with self.subTest(arm=arm):
                record = self.episode(arm, Completion(text='{"choice": "ZZZ-9999"}', attempts=1), "none", "keyed")
                self.assertEqual(record["outcome"]["status"], "invalid")
                self.assertEqual(record["outcome"]["invalid_reason"], "unknown_id")
                self.assertFalse(record["outcome"]["substituted"])
                self.assertEqual(record["row"]["basket_size"], 0)
                self.assertFalse({e["kind"] for e in record["events"]} & {"click", "cart", "purchase"})

    def test_a_failed_call_is_an_error_even_under_the_fallback(self):
        record = self.episode("api", Completion(error_code="provider_unavailable", attempts=3), "first_item")
        self.assertEqual(record["outcome"], {"status": "error", "chosen_sku": None, "invalid_reason": None,
                                             "error_code": "provider_unavailable", "substituted": False,
                                             "raw": None, "attempts": 3})
        self.assertEqual(record["events"], [])

    def test_raw_replies_are_kept(self):
        text = '{"choice": "ZZZ-9999", "extras": [], "reason": "x"}'
        self.assertEqual(self.episode("chatbot", Completion(text=text, attempts=1), "none", "keyed")["outcome"]["raw"], text)

    def test_the_run_refuses_the_fallback_with_keyed_streams(self):
        with self.assertRaises(ValueError):
            LegacyRun(KEYED, seed=CONFIG.seed, fallback="first_item")


class PlanTest(unittest.TestCase):
    def test_the_plan_is_the_original_run_order(self):
        sys.path.insert(0, str(HERE))
        import legacy_equivalence as h

        self.assertEqual([s.episode for s in episode_plan(LEGACY)], [row["episode"] for row in h.ROWS])
        counts = defaultdict(int)
        for spec in episode_plan(LEGACY):
            counts[spec.arm] += 1
        self.assertEqual(dict(counts), {"human_sim": 90, "chatbot": 90, "gui_agent": 90, "api": 90, "gui_agent_2x2": 160})

    def test_template_ids_match_the_legacy_config(self):
        self.assertEqual({"legacy_" + k: v for k, v in legacy_env.TEMPLATE_IDS.items()},
                         dict(CONFIG.protocol.prompt_templates))


if __name__ == "__main__":
    unittest.main()
