"""SL-03 protocol tests: parser fixtures, the no-foreign-Select property, and the
observation, action and outcome types. Standard library and pydantic only:

    cd ai-agents && python -B -m unittest discover -s tests/features/storelab -t . -v
"""
from __future__ import annotations

import json
import random
import string
import unittest
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from features.storelab.core.protocol import (ACTION, Invalid, Leave, Observation, RequestPage, Select,
                                             TrialOutcome, parse_for, parse_reply)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "parser_cases.json").read_text(encoding="utf-8"))
PARSER_REASONS = {"no_json", "schema_violation", "unknown_id", "multiple_choices", "empty"}


def summary(action) -> dict:
    out = {"action": action.type}
    if isinstance(action, Select):
        out["product_id"] = action.product_id
    if isinstance(action, Invalid):
        out["reason"] = action.reason
    return out


def observation(ids, views=None) -> Observation:
    return Observation(task={"task_id": "t1", "instruction": "Buy one desk lamp under $40."}, set_id="s1",
                       presentation_id="p1", views=views or {},
                       candidates=[{"product_id": pid, "position": i + 1, "sponsored_label": i == 0}
                                   for i, pid in enumerate(ids)])


class ParserFixtureTest(unittest.TestCase):
    """AC-1: 100% of parser fixtures map to the expected Action or invalid reason."""

    def test_every_fixture_maps_to_its_expected_action(self):
        for case in FIXTURES["cases"]:
            with self.subTest(case=case["id"]):
                allowed = FIXTURES["candidate_sets"][case["candidates"]]
                self.assertEqual(summary(parse_reply(case["output"], allowed)), case["expected"], case["shape"])

    def test_fixtures_cover_every_outcome_and_are_about_forty(self):
        cases = FIXTURES["cases"]
        self.assertGreaterEqual(len(cases), 40)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        expected = Counter(c["expected"].get("reason", c["expected"]["action"]) for c in cases)
        self.assertEqual(set(expected), PARSER_REASONS | {"select", "leave"})
        for case in cases:
            self.assertTrue(case["shape"].strip(), case["id"])

    def test_parse_for_uses_the_observation_candidates(self):
        obs = observation(["DES-0219", "DES-0007"])
        self.assertEqual(parse_for(obs, '{"choice": "DES-0007", "reason": "x"}'), Select(product_id="DES-0007"))
        self.assertEqual(parse_for(obs, '{"choice": "COF-0123", "reason": "x"}').reason, "unknown_id")


class NoForeignSelectPropertyTest(unittest.TestCase):
    """AC-2: with random strings as model output, no Select is ever produced for an id
    outside the observation."""

    ITERATIONS = 20000

    @staticmethod
    def asin(rng: random.Random) -> str:
        return "B0" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    def outside_ids(self, rng, inside) -> list[str]:
        pid = rng.choice(inside)
        return [self.asin(rng), pid.lower(), pid + "X", pid[:-1], " " + pid[1:], pid.replace("0", "O"),
                pid + " " + rng.choice(inside), "%s, %s" % (pid, rng.choice(inside)), "COF-0123", "1", ""]

    def check(self, generate, seed):
        rng = random.Random(seed)
        counts = Counter()
        for _ in range(self.ITERATIONS):
            inside = [self.asin(rng) for _ in range(rng.randint(1, 8))]
            text, foreign = generate(rng, inside)
            action = parse_reply(text, inside)
            self.assertIsInstance(action, (Select, Leave, Invalid))
            if isinstance(action, Select):
                self.assertIn(action.product_id, inside, text)
                counts["select"] += 1
            counts["foreign_offered"] += foreign
        return counts

    def test_random_strings(self):
        alphabet = string.printable + '{}{}[]""::,,é中\U0001f600'

        def generate(rng, inside):
            parts = []
            for _ in range(rng.randint(0, 12)):
                kind = rng.random()
                if kind < 0.5:
                    parts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30))))
                elif kind < 0.75:
                    parts.append(rng.choice(['{"choice": "', '"}', '{"choice":', "null", '"reason": "x"', "```json"]))
                elif kind < 0.85:  # occasionally a whole reply, for an inside or a foreign id
                    parts.append(json.dumps({"choice": rng.choice(inside + self.outside_ids(rng, inside)),
                                             "reason": "x"}))
                else:
                    parts.append(rng.choice(inside + self.outside_ids(rng, inside)))
            return "".join(parts), 0

        counts = self.check(generate, seed=3)
        self.assertGreater(counts["select"], 0)

    def test_json_shaped_replies_with_foreign_ids(self):
        def value(rng, inside):
            r = rng.random()
            if r < 0.3:
                return rng.choice(inside), False
            if r < 0.8:
                return rng.choice(self.outside_ids(rng, inside)), True
            return rng.choice([None, 3, 1.5, True, [rng.choice(inside)], [self.asin(rng), rng.choice(inside)], {}]), False

        def generate(rng, inside):
            foreign, objects = 0, []
            for _ in range(rng.choice([1, 1, 1, 2, 3])):
                choice, is_foreign = value(rng, inside)
                foreign |= is_foreign
                obj = {"choice": choice}
                if rng.random() < 0.85:
                    obj["reason"] = rng.choice(["cheapest", "best value", "fits", "x " * rng.randint(0, 30), 7, None])
                if rng.random() < 0.15:
                    obj[rng.choice(["extras", "selection", "Choice"])] = rng.choice(inside)
                text = json.dumps(obj, ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 2]))
                if rng.random() < 0.1:
                    text = text[:rng.randint(0, len(text))]
                if rng.random() < 0.1:
                    text = text.replace('"choice"', '"choice": "%s", "choice"' % rng.choice(inside), 1)
                objects.append(rng.choice(["%s", "```json\n%s\n```", "Answer: %s", "%s\nThanks."]) % text)
            return "\n".join(objects), foreign

        counts = self.check(generate, seed=5)
        self.assertGreater(counts["select"], 1000)
        self.assertGreater(counts["foreign_offered"], 5000)

    def test_wrapped_truncated_conflicting_or_overflowing_replies_are_never_choices(self):
        """Variants of a valid reply that the strict format rules out (SL-03 review round 1)."""
        rng = random.Random(11)
        variants = [
            lambda r, a, b: "[%s]" % r,  # array root
            lambda r, a, b: "[%s, %s]" % (r, r),
            lambda r, a, b: '{"answer": %s' % r,  # truncated outer object
            lambda r, a, b: '{"note": oops, "reply": %s}' % r,  # malformed outer object
            lambda r, a, b: '[ %s' % r,  # unclosed array
            lambda r, a, b: '%s\n{"choice": "%s", "choice": "%s", "reason": "x"}' % (r, b, a),  # conflicting copy
            lambda r, a, b: '{"choice": "%s", "choice": "%s", "reason": "x"}\n%s' % (b, a, r),
            lambda r, a, b: r[:-1] + ', "score": %se999}' % rng.choice(["", "-"]),  # overflow
            lambda r, a, b: '{"choice": "%s"}' % a,  # no reason
            lambda r, a, b: '{"choice": "%s", "reason": null}' % a,
            lambda r, a, b: r[:rng.randrange(1, len(r))],  # truncated reply
        ]
        for _ in range(self.ITERATIONS):
            inside = [self.asin(rng) for _ in range(rng.randint(2, 8))]
            a, b = rng.sample(inside, 2)
            reply = json.dumps({"choice": a, "reason": "cheapest"})
            for index, variant in enumerate(variants):
                text = rng.choice(["%s", "```json\n%s\n```", "Answer: %s"]) % variant(reply, a, b)
                action = parse_reply(text, inside)
                self.assertIsInstance(action, Invalid, (index, text))

    def test_random_edits_of_valid_replies(self):
        def generate(rng, inside):
            text = list(json.dumps({"choice": rng.choice(inside), "reason": "cheapest"}))
            for _ in range(rng.randint(1, 4)):
                i = rng.randrange(len(text) + 1)
                op = rng.random()
                if op < 0.4 and text:
                    text[min(i, len(text) - 1)] = rng.choice(string.ascii_uppercase + string.digits + '{}"')
                elif op < 0.7:
                    text.insert(i, rng.choice(string.ascii_uppercase + string.digits + ' "{}'))
                elif text:
                    del text[min(i, len(text) - 1)]
            return "".join(text), 1

        counts = self.check(generate, seed=7)
        self.assertGreater(counts["select"], 0)


class ObservationAndActionTest(unittest.TestCase):
    def test_candidates_need_unique_ids_and_positions_one_to_k(self):
        self.assertEqual(observation(["A1", "B2", "C3"]).candidate_ids, frozenset({"A1", "B2", "C3"}))
        with self.assertRaises(ValidationError):
            observation(["A1", "A1"])
        with self.assertRaises(ValidationError):
            Observation(task={"task_id": "t", "instruction": "x"}, set_id="s", presentation_id="p",
                        candidates=[{"product_id": "A1", "position": 2, "sponsored_label": False}])

    def test_observation_hash_covers_the_rendered_views(self):
        a = observation(["A1", "B2"], {"text": "1. A1\n2. B2"})
        self.assertEqual(a.observation_hash(), observation(["A1", "B2"], {"text": "1. A1\n2. B2"}).observation_hash())
        self.assertNotEqual(a.observation_hash(), observation(["A1", "B2"], {"text": "1. A1\n2. B2 "}).observation_hash())
        self.assertNotEqual(a.observation_hash(), observation(["B2", "A1"], {"text": "1. A1\n2. B2"}).observation_hash())

    def test_actions_round_trip_through_the_discriminated_union(self):
        for action in (RequestPage(product_id="A1"), Select(product_id="A1"), Leave(),
                       Invalid(reason="max_turns", detail="8 turns")):
            self.assertEqual(ACTION.validate_json(ACTION.dump_json(action)), action)
        with self.assertRaises(ValidationError):
            ACTION.validate_python({"type": "buy", "product_id": "A1"})


class OutcomeTest(unittest.TestCase):
    """Every non-choice outcome is labelled (S3)."""

    VALID = [
        {"status": "choice", "chosen_id": "A1", "attempts": 1},
        {"status": "outside", "attempts": 1},
        {"status": "invalid", "invalid_reason": "unknown_id", "attempts": 1},
        {"status": "refusal", "attempts": 1},
        {"status": "error", "error_code": "rate_limited", "attempts": 4},
        {"status": "not_run", "error_code": "kill_switch"},
    ]
    INVALID = [
        ({"status": "choice"}, "chosen_id"),
        ({"status": "error", "chosen_id": "A1", "error_code": "auth"}, "chosen_id"),
        ({"status": "error"}, "error_code"),
        ({"status": "error", "error_code": "budget"}, "error_code"),
        ({"status": "not_run", "error_code": "rate_limited"}, "error_code"),
        ({"status": "not_run", "error_code": "budget", "attempts": 1}, "attempts"),
        ({"status": "invalid"}, "invalid_reason"),
        ({"status": "outside", "invalid_reason": "empty"}, "invalid_reason"),
        ({"status": "refusal", "error_code": "auth"}, "error_code"),
    ]

    def test_valid_outcomes(self):
        for fields in self.VALID:
            with self.subTest(**fields):
                self.assertEqual(TrialOutcome(backend="openai", **fields).status, fields["status"])

    def test_unexplained_or_contradictory_outcomes_are_rejected(self):
        for fields, named in self.INVALID:
            with self.subTest(**fields), self.assertRaises(ValidationError) as caught:
                TrialOutcome(backend="openai", **fields)
            self.assertIn(named, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
