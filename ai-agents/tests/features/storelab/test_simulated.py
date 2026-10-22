"""SL-06 simulated backend: analytic checks, Example diversify (AC-1), sampling vs
choice_probs (AC-2), seeds and order invariance (AC-3), protocol conformance (AC-4),
and the agent_params schema.

Writes only to temporary directories.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
from pydantic import ValidationError

from features.storelab import cli, e1_stimuli
from features.storelab.backends.local_llm import LocalLLM
from features.storelab.backends.simulated import (ChoiceProbs, SimulatedAgent, SimulatedHuman, agent_logits,
                                                  choice_probs, draw_state, human_probs, load_moments,
                                                  params_from_config, policy_choice_prob, simulate_loads)
from features.storelab.config_files import read_config_data
from features.storelab.core.agent_params import (AgentParams, HumanParams, LabelWeights, ModelParams, StateParams,
                                                 dump_agent_params, load_agent_params)
from features.storelab.core.config import apply_overrides, canonical_json, resolve_config
from features.storelab.core.design import chi2_sf
from features.storelab.core.protocol import Candidate, Completion, Observation, Task, parse_reply
from features.storelab.core.records import trial_key
from features.storelab.core.render import build_observation
from tests.features.storelab import protocol_conformance as conformance
from tests.features.storelab.test_catalog_amazon import fixture_snapshot

CONFIG = Path(cli.__file__).resolve().parent / "configs" / "e1_pilot.yaml"
CONTRACT = Path(cli.__file__).resolve().parents[2] / "contracts" / "storelab" / "agent_params.schema.json"


def observation(ids, labels=None, task="t", set_id=None) -> Observation:
    labels = labels or [False] * len(ids)
    name = set_id or "s-" + "-".join(ids)
    return Observation(task=Task(task_id=task, instruction="choose"), set_id=name, presentation_id=name + "/p0",
                       candidates=[Candidate(product_id=i, position=k + 1, sponsored_label=lab)
                                   for k, (i, lab) in enumerate(zip(ids, labels))])


def model(sigma=1.0, **changes) -> ModelParams:
    base = dict(model_key="m", sigma=sigma, label_effect=-0.5, position_effects=[0.0, -0.1, -0.2, -0.3, -0.35, -0.4,
                                                                                  -0.45, -0.5],
                outside_value=-1.0, page_request_prob=0.3,
                scores={"t": {"P%d" % i: 0.25 * i for i in range(8)}})
    return ModelParams(**{**base, **changes})


EIGHT = observation(["P%d" % i for i in (5, 1, 7, 0, 3, 6, 2, 4)],
                    [True, False, False, True, False, True, False, False])


def chi2_p(counts: list[int], probs: list[float]) -> float:
    n = sum(counts)
    stat = sum((c - n * p) ** 2 / (n * p) for c, p in zip(counts, probs))
    return chi2_sf(stat, len(counts) - 1)


# --------------------------------------------------------------------- analytic
class AnalyticTest(unittest.TestCase):
    def test_probabilities_sum_to_one_including_the_outside_option(self):
        rng = random.Random(3)
        for _ in range(200):
            k = rng.randint(1, 8)
            ids = rng.sample(["P%d" % i for i in range(8)], k)
            obs = observation(ids, [rng.random() < 0.3 for _ in ids])
            params = model(sigma=rng.choice([1e-3, 0.05, 1.0, 20.0]), outside_value=rng.uniform(-3, 3))
            probs = choice_probs(params, obs)
            self.assertAlmostEqual(math.fsum(probs.as_list(obs)), 1.0, places=12)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs.as_list(obs)))

    def test_the_logit_formula(self):
        params = model()
        probs = choice_probs(params, EIGHT)
        scores = [params.scores["t"][c.product_id] + params.label_effect * c.sponsored_label
                  + params.position_effects[c.position - 1] for c in EIGHT.candidates]
        denominator = math.exp(params.outside_value) + sum(math.exp(s) for s in scores)
        self.assertAlmostEqual(probs.outside, math.exp(params.outside_value) / denominator, places=14)
        for c, s in zip(EIGHT.candidates, scores):
            self.assertAlmostEqual(probs.products[c.product_id], math.exp(s) / denominator, places=14)

    def test_small_noise_limit(self):
        """Lemma small-noise: with σ = 1e-4 the argmax gets probability ≥ 1 − 1e-9."""
        for outside, winner in ((-1.0, "P7"), (5.0, None)):
            params = model(sigma=1e-4, position_effects=[0.0] * 8, outside_value=outside, label_effect=0.0)
            probs = choice_probs(params, EIGHT)
            best = probs.outside if winner is None else probs.products[winner]
            with self.subTest(outside=outside):
                self.assertGreaterEqual(best, 1 - 1e-9)
        tight = model(sigma=1e-4, position_effects=[0.0] * 8, label_effect=0.0,
                      scores={"t": {"A": 1.0, "B": 1.0 - 0.01}})
        self.assertGreaterEqual(choice_probs(tight, observation(["B", "A"])).products["A"], 1 - 1e-9)

    def test_states_shift_scores_and_average_with_their_probabilities(self):
        params = model(states=[StateParams(state_id="calm", probability=0.25),
                               StateParams(state_id="hot", probability=0.75, shifts={"t": {"P0": 3.0}})])
        calm, hot = choice_probs(params, EIGHT, "calm"), choice_probs(params, EIGHT, "hot")
        self.assertGreater(hot.products["P0"], calm.products["P0"])
        mean = choice_probs(params, EIGHT, average=True)
        for pid in mean.products:
            self.assertAlmostEqual(mean.products[pid], 0.25 * calm.products[pid] + 0.75 * hot.products[pid], places=14)
        with self.assertRaisesRegex(ValueError, "pass the period's state_id"):
            choice_probs(params, EIGHT)
        self.assertIn(draw_state(params, 9, "period", 0), ("calm", "hot"))
        self.assertEqual(draw_state(params, 9, "period", 0), draw_state(params, 9, "period", 0))
        drawn = Counter(draw_state(params, 9, "period", i) for i in range(4000))
        self.assertAlmostEqual(drawn["hot"] / 4000, 0.75, delta=0.03)

    def test_missing_inputs_are_errors(self):
        with self.assertRaisesRegex(KeyError, "no scores for task"):
            agent_logits(model(), observation(["P1"], task="other"))
        with self.assertRaisesRegex(KeyError, "no score for X"):
            agent_logits(model(), observation(["X"]))
        with self.assertRaisesRegex(ValueError, "no position effect"):
            agent_logits(model(position_effects=[0.0]), observation(["P1", "P2"]))

    def test_human_model(self):
        humans = HumanParams(position_weights=[0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
                             product_weights={"t": {"P5": LabelWeights(organic=0.5, sponsored=0.25)}})
        probs = human_probs(humans, EIGHT)
        self.assertAlmostEqual(probs.products["P5"], 0.3 * 0.25)  # position 1, sponsored
        self.assertAlmostEqual(probs.products["P1"], 0.2 * 1.0)
        self.assertAlmostEqual(math.fsum(probs.as_list(EIGHT)), 1.0, places=14)
        self.assertAlmostEqual(probs.outside, 1 - math.fsum(probs.as_list(EIGHT)[1:]), places=14)
        with self.assertRaisesRegex(ValueError, "sum to"):
            human_probs(HumanParams(position_weights=[0.6, 0.5]), observation(["A", "B"]))
        with self.assertRaises(ValidationError):
            HumanParams(position_weights=[0.1, 0.2])


# ------------------------------------------------------------- Example diversify
def diversify_model() -> ModelParams:
    """One nearly deterministic model, two equally likely states (v1.tex l.388):
    Λ1: v_A > v_B > v_C; Λ2: v_B > v_A > v_C; the outside option below C."""
    return ModelParams(model_key="m", sigma=1e-3, label_effect=0.0, position_effects=[0.0, 0.0], outside_value=0.0,
                       page_request_prob=0.0, scores={"x": {"A": 2.0, "B": 2.0, "C": 1.0}},
                       states=[StateParams(state_id="l1", probability=0.5, shifts={"x": {"A": 1.0}}),
                               StateParams(state_id="l2", probability=0.5, shifts={"x": {"B": 1.0}})])


DIVERSIFY = {  # policy -> Var(D_A) stated in the example
    "every request {A,B}": ([(["A", "B"], 1.0)], 250_000.0),
    "{A,C} or {B,C}": ([(["A", "C"], 0.5), (["B", "C"], 0.5)], 250.0),
    "{A,B} or {A,C}": ([(["A", "B"], 0.5), (["A", "C"], 0.5)], 62_625.0),
}


def diversify_policy(sets):
    return [(observation(ids, task="x"), w) for ids, w in sets]


class DiversifyTest(unittest.TestCase):
    """AC-1: Var(D_A) = 250,000 / 250 / 62,625 for n = 1,000."""

    def test_the_variances_follow_exactly_from_lemma_load(self):
        params = diversify_model()
        for name, (sets, variance) in DIVERSIFY.items():
            policy = diversify_policy(sets)
            p = [policy_choice_prob(params, policy, "A", s.state_id) for s in params.states]
            mean, var = load_moments(1000, [0.5, 0.5], p)
            with self.subTest(policy=name):
                self.assertEqual(var, variance)
                self.assertEqual(mean, {250_000.0: 500.0, 250.0: 500.0, 62_625.0: 750.0}[variance])

    def test_simulation_is_within_three_standard_errors(self):
        params = diversify_model()
        for i, (name, (sets, variance)) in enumerate(DIVERSIFY.items()):
            loads = simulate_loads(params, diversify_policy(sets), n=1000, periods=20_000,
                                   rng=np.random.default_rng(20_000 + i))["A"].astype(np.float64)
            centred = loads - loads.mean()
            sample_var = centred @ centred / (len(loads) - 1)
            # standard error of the sample variance from the sample's 2nd and 4th moments
            m2, m4 = (centred ** 2).mean(), (centred ** 4).mean()
            se = math.sqrt(max(m4 - m2 ** 2 * (len(loads) - 3) / (len(loads) - 1), 0.0) / len(loads))
            with self.subTest(policy=name, sample_var=sample_var, se=se):
                self.assertGreater(se, 0.0)
                self.assertLessEqual(abs(sample_var - variance), 3 * se)

    def test_lemma_load_formula(self):
        mean, var = load_moments(10, [1.0], [0.3])
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(var, 10 * 0.3 * 0.7)
        mean, var = load_moments(100, [0.2, 0.8], [0.1, 0.6])
        self.assertAlmostEqual(mean, 100 * 0.5)
        self.assertAlmostEqual(var, 100 * (0.2 * 0.09 + 0.8 * 0.24) + 100 ** 2 * (0.2 * 0.16 + 0.8 * 0.01))
        with self.assertRaises(ValueError):
            load_moments(10, [0.5], [0.1, 0.2])


# ------------------------------------------------------------------ sampling
class SamplingTest(unittest.TestCase):
    """AC-2: empirical choice frequencies match choice_probs (χ², fixed seed, p > 0.001)."""

    N = 20_000

    def counts(self, decide, obs) -> list[int]:
        index = {c.product_id: k + 1 for k, c in enumerate(obs.candidates)}
        counts = [0] * (len(obs.candidates) + 1)
        for i in range(self.N):
            outcome = decide("trial-%d" % i)
            counts[0 if outcome.status == "outside" else index[outcome.chosen_id]] += 1
        return counts

    def test_agent_frequencies(self):
        for sigma in (1.0, 0.3):
            params = model(sigma=sigma, outside_value=0.5)
            agent = SimulatedAgent(params, seed=2026)
            counts = self.counts(lambda key: agent.decide(EIGHT, "text", key), EIGHT)
            probs = choice_probs(params, EIGHT).as_list(EIGHT)
            p = chi2_p(counts, probs)
            with self.subTest(sigma=sigma, p=p):
                self.assertGreater(p, 0.001)
                self.assertTrue(all(c > 0 for c in counts))

    def test_frequencies_in_a_given_state(self):
        params = model(states=[StateParams(state_id="a", probability=0.5),
                               StateParams(state_id="b", probability=0.5, shifts={"t": {"P0": 2.0, "P7": -2.0}})])
        agent = SimulatedAgent(params, seed=7)
        for state in ("a", "b"):
            counts = self.counts(lambda key: agent.decide(EIGHT, "structured", key, state), EIGHT)
            p = chi2_p(counts, choice_probs(params, EIGHT, state).as_list(EIGHT))
            with self.subTest(state=state, p=p):
                self.assertGreater(p, 0.001)

    def test_human_frequencies(self):
        humans = HumanParams(position_weights=[0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02],
                             product_weights={"t": {"P5": LabelWeights(organic=0.9, sponsored=0.6)}})
        human = SimulatedHuman(humans, seed=11)
        counts = self.counts(lambda key: human.decide(EIGHT, "text", key), EIGHT)
        p = chi2_p(counts, human_probs(humans, EIGHT).as_list(EIGHT))
        self.assertGreater(p, 0.001)

    def test_page_requests_follow_chi(self):
        params = model(page_request_prob=0.3)
        agent = SimulatedAgent(params, seed=5)
        requested = eligible = 0
        for i in range(self.N // 4):
            outcome = agent.decide(EIGHT, "browser_tool", "page-%d" % i)
            eligible += len(EIGHT.candidates) - (outcome.status == "choice")
            requested += len(outcome.page_requests)
            self.assertNotIn(outcome.chosen_id, outcome.page_requests)
        rate, se = requested / eligible, math.sqrt(0.3 * 0.7 / eligible)
        self.assertLessEqual(abs(rate - 0.3), 4 * se)
        for interface in ("text", "structured"):
            self.assertEqual(agent.decide(EIGHT, interface, "page-0").page_requests, [])

    def test_page_requests_respect_max_turns(self):
        agent = SimulatedAgent(model(page_request_prob=1.0), seed=5, max_turns=3)
        for i in range(50):
            actions = agent.act(EIGHT, "browser_tool", "turns-%d" % i)
            self.assertLessEqual(len(actions), 3)
            self.assertEqual(len(actions), 3)


# ------------------------------------------------------- seeds, order, resume
class InterruptedRun(Exception):
    pass


class E1Trials:
    """E1 trials from the SL-05 fixture snapshot, as a runner would enumerate them."""

    def __init__(self, root: Path, seed: int):
        snapshot = fixture_snapshot(root)
        overrides = {"environment.dataset": {"name": "amazon_reviews_2023", "sha256": snapshot.name},
                     "repetitions": 3, "seed": seed,
                     "agents.models": [
                         {"key": "gpt_4o_20240806", "provider": "openai", "model": "gpt-4o-2024-08-06",
                          "temperatures": [None]},
                         {"key": "gemini_20_flash_001", "provider": "google", "model": "gemini-2.0-flash-001",
                          "temperatures": [0.0]}]}
        self.config = resolve_config(apply_overrides(read_config_data(CONFIG), overrides))
        built = e1_stimuli.build(self.config, snapshot)
        self.stimuli = built.tasks
        self.params = params_from_config(self.config, built.tasks)
        specs = {t.task.task_id: t.task for t in built.tasks}
        self.trials = []
        for design in built.designs:
            for s in design.sets:
                for rep in range(self.config.repetitions):
                    presentation = design.presentation_for(s.set_id, rep)
                    obs = build_observation(self.config, specs[design.task_id], presentation, built.products)
                    for interface in self.config.interfaces:
                        for m in self.config.agents.models:
                            for temperature in m.temperatures:
                                key = trial_key(self.config.config_hash, design.task_id, s.set_id,
                                                presentation.presentation_id, interface, m.key, temperature, rep)
                                self.trials.append((key, m.key, interface, obs))

    def agents(self, seed: int | None = None):
        return {m.model_key: SimulatedAgent(m, seed=self.config.seed if seed is None else seed,
                                            max_turns=self.config.protocol.max_turns) for m in self.params.models}


def run_trials(agents, trials, done: dict | None = None, fail_at: int | None = None) -> dict:
    done = {} if done is None else done
    calls = 0
    for key, model_key, interface, obs in trials:
        if key in done:
            continue
        calls += 1
        if calls == fail_at:
            raise InterruptedRun("killed at trial %d" % calls)
        done[key] = agents[model_key].decide(obs, interface, key)
    return done


def record_set_hash(records: dict) -> str:
    rows = sorted([key, outcome.model_dump(mode="json")] for key, outcome in records.items())
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


class SeedAndOrderTest(unittest.TestCase):
    """AC-3: same seed → identical choices; three random orders plus a run
    interrupted and resumed at a random point give the same record-set hash."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.e1 = E1Trials(cls.root, seed=2026102101)
        cls.reference = run_trials(cls.e1.agents(), cls.e1.trials)
        cls.reference_hash = record_set_hash(cls.reference)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_the_trials_cover_the_design_and_all_outcomes(self):
        self.assertEqual(len(self.e1.trials), len({key for key, *_ in self.e1.trials}))
        self.assertEqual(len(self.reference), len(self.e1.trials))
        statuses = Counter(o.status for o in self.reference.values())
        self.assertEqual(set(statuses), {"choice", "outside"})
        self.assertTrue(any(o.page_requests for o in self.reference.values()))

    def test_same_seed_gives_identical_choices(self):
        again = run_trials(self.e1.agents(), self.e1.trials)
        self.assertEqual(again, self.reference)
        other = run_trials(self.e1.agents(seed=self.e1.config.seed + 1), self.e1.trials)
        changed = sum(other[k].chosen_id != self.reference[k].chosen_id for k in self.reference)
        self.assertGreater(changed, len(self.reference) // 10)

    def test_three_random_orders_give_the_same_record_set_hash(self):
        rng = random.Random(404)
        for attempt in range(3):
            order = list(self.e1.trials)
            rng.shuffle(order)
            with self.subTest(order=attempt):
                self.assertNotEqual([t[0] for t in order], [t[0] for t in self.e1.trials])
                self.assertEqual(record_set_hash(run_trials(self.e1.agents(), order)), self.reference_hash)

    def test_a_run_interrupted_and_resumed_gives_the_same_hash(self):
        rng = random.Random(505)
        order = list(self.e1.trials)
        rng.shuffle(order)
        stop_at = rng.randrange(100, len(order) - 100)
        done: dict = {}
        with self.assertRaises(InterruptedRun):
            run_trials(self.e1.agents(), order, done, fail_at=stop_at)
        self.assertEqual(len(done), stop_at - 1)
        rng.shuffle(order)  # resume in yet another order, with fresh backend objects
        resumed = run_trials(self.e1.agents(), order, dict(done))
        self.assertEqual(record_set_hash(resumed), self.reference_hash)

    def test_stipulated_parameters_come_from_the_config_and_u(self):
        cfg, params = self.e1.config, self.e1.params
        self.assertEqual([m.model_key for m in params.models], [m.key for m in cfg.agents.models])
        simulated = cfg.agents.simulated
        for m in params.models:
            self.assertEqual((m.sigma, m.label_effect, m.position_effects, m.outside_value, m.page_request_prob),
                             (simulated.sigma, simulated.label_effect, simulated.position_effects,
                              simulated.outside_value, simulated.page_request_prob))
            self.assertEqual([s.state_id for s in m.states], ["default"])
            self.assertEqual(m.scores, {t.task.task_id: t.utility for t in self.e1.stimuli})  # v = u
        self.assertIsNone(params.humans)
        with_humans = resolve_config(apply_overrides(cfg.model_dump(mode="json"), {
            "humans": {"model": "position_weights", "position_weights": [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]}}))
        humans = params_from_config(with_humans, self.e1.stimuli).humans
        self.assertEqual((humans.position_weights[0], humans.default_weight.organic, humans.default_weight.sponsored),
                         (0.3, 1.0, 1.0))


# ------------------------------------------------------------------ conformance
class ConformanceTest(unittest.TestCase):
    """AC-4: the simulated backends pass the SL-03 protocol conformance checks and emit
    the Action types the LLM backends emit."""

    def test_simulated_agent(self):
        agent = SimulatedAgent(model(page_request_prob=0.4, outside_value=0.8), seed=3)
        seen = set()
        for interface in ("text", "structured", "browser_tool"):
            for i in range(300):
                key = "%s-%d" % (interface, i)
                actions = agent.act(EIGHT, interface, key)
                outcome = agent.decide(EIGHT, interface, key)
                conformance.check_outcome(self, EIGHT, interface, outcome, actions)
                self.assertEqual((outcome.backend, outcome.requested_model), ("simulated", "simulated:m"))
                seen |= {type(a).__name__ for a in actions}
        self.assertEqual(seen, {"RequestPage", "Select", "Leave"})

    def test_simulated_human(self):
        human = SimulatedHuman(HumanParams(position_weights=[0.2] * 4), seed=3)
        obs = observation(["A", "B", "C", "D"])
        seen = set()
        for interface in ("text", "browser_tool"):
            for i in range(300):
                actions = human.act(obs, interface, "h-%d" % i)
                outcome = human.decide(obs, interface, "h-%d" % i)
                conformance.check_outcome(self, obs, interface, outcome, actions)
                self.assertEqual(outcome.backend, "human_model")
                seen |= {type(a).__name__ for a in actions}
        self.assertEqual(seen, {"Select", "Leave"})

    def test_the_local_llm_backend_emits_the_same_action_types(self):
        replies = ['{"choice": "P1", "reason": "best"}', '{"choice": null, "reason": "none fit"}', "no json here"]
        obs = observation(["P0", "P1"])
        seen = set()
        for reply in replies:
            llm = LocalLLM("qwen2:7b")
            llm.complete = lambda prompt, seed=None, text=reply: Completion(text=text, attempts=1)
            outcome = llm.decide("prompt", obs.candidate_ids)
            actions = [parse_reply(reply, obs.candidate_ids)]
            conformance.check_outcome(self, obs, "text", outcome, actions)
            seen |= {type(a).__name__ for a in actions}
        self.assertEqual(seen, {"Select", "Leave", "Invalid"})

    def test_the_checker_rejects_nonconforming_output(self):
        obs = observation(["A", "B"])
        agent = SimulatedAgent(model(sigma=1e-3, scores={"t": {"A": 1.0, "B": 0.0}}), seed=1)  # always A
        outcome = agent.decide(obs, "text", "k")
        self.assertEqual(outcome.chosen_id, "A")
        bad = [outcome.model_copy(update={"page_requests": ["A"]}),
               outcome.model_copy(update={"status": "outside", "chosen_id": None})]
        for wrong in bad:
            with self.assertRaises(AssertionError):
                conformance.check_outcome(self, obs, "text", wrong, agent.act(obs, "text", "k"))


# --------------------------------------------------------------------- schema
class AgentParamsSchemaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def params(self) -> AgentParams:
        return AgentParams(source="estimated", models=[model(), diversify_model().model_copy(update={"model_key": "d"})],
                           humans=HumanParams(position_weights=[0.3, 0.2]), provenance={"run": "x"})

    def test_round_trip_and_contract(self):
        params = self.params()
        text = dump_agent_params(params)
        path = self.tmp / "agent_params.json"
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(load_agent_params(path, digest), params)
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_agent_params(path, "0" * 64)
        schema = json.loads(CONTRACT.read_text(encoding="utf-8"))
        jsonschema.validate(json.loads(text), schema)
        broken = json.loads(text)
        broken["models"][0]["sigma"] = 0
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    def test_estimated_parameters_load_through_the_config(self):
        path = self.tmp / "agent_params.json"
        path.write_text(dump_agent_params(self.params()), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        data = apply_overrides(read_config_data(CONFIG), {"agents.simulated": {
            "parameters": "estimated", "params_file": {"path": "agent_params.json", "sha256": digest}}})
        self.assertEqual(params_from_config(resolve_config(data), [], base=self.tmp), self.params())
        data["agents"]["simulated"]["params_file"]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            params_from_config(resolve_config(data), [], base=self.tmp)

    def test_invalid_parameters_are_rejected(self):
        for change in ({"states": [StateParams(state_id="a", probability=0.4)]},
                       {"states": [StateParams(state_id="a", probability=0.5),
                                   StateParams(state_id="a", probability=0.5)]},
                       {"states": [StateParams(state_id="a", probability=1.0, shifts={"t": {"ZZ": 1.0}})]},
                       {"sigma": 0.0}, {"page_request_prob": 1.5}, {"scores": {}}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                model(**change)
        with self.assertRaises(ValidationError):
            AgentParams(source="stipulated", models=[model(), model()])

    def test_choice_probs_type(self):
        probs = choice_probs(model(), EIGHT)
        self.assertIsInstance(probs, ChoiceProbs)
        self.assertEqual(list(probs.products), [c.product_id for c in EIGHT.candidates])


if __name__ == "__main__":
    unittest.main()
