"""SL-16 mechanism and allocation solvers: hand-solved markets (AC-1), the payment
identity (AC-2), incentives (AC-3), solver exclusions (AC-4) and the runtime profile
(AC-5), from the committed validation study and from fast direct checks of the
solvers, the market model, costs and virtual values.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import unittest
from pathlib import Path
from unittest import mock

import cvxpy as cp
import numpy as np
from features.storelab.analysis import mechanism_validation as mv
from features.storelab.backends.simulated import choice_probs, human_probs
from features.storelab.core.agent_params import AgentParams, HumanParams, LabelWeights, ModelParams, StateParams
from features.storelab.core.mechanisms import allocation, costs, gsp, solver
from features.storelab.core.mechanisms import payments as pay
from features.storelab.core.mechanisms.evaluate import MECHANISMS, MarketRunner, evaluate_markets
from features.storelab.core.mechanisms.market import (Context, Cost, Guardrail, Market, Offer, OrganicList,
                                                      action_table, displayed_list, human_displays, match_reference,
                                                      models_from_agent_params, response)
from features.storelab.core.mechanisms.values import ValueDistribution
from features.storelab.core.protocol import Candidate, Observation, Task

REPORT = Path(mv.__file__).resolve().parents[1] / "docs" / "mechanism_validation.json"
REQUIREMENTS = Path(mv.__file__).resolve().parents[3] / "requirements-storelab.txt"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


class CommittedStudyTest(unittest.TestCase):
    """The committed study was produced by the current code at the full counts."""

    def test_the_report_matches_the_code_and_the_registered_counts(self):
        data = report()
        self.assertEqual(data["code_fingerprint"], mv.code_fingerprint())
        self.assertEqual(data["counts"], mv.FULL)
        self.assertEqual(data["master_seed"], mv.MASTER_SEED)
        self.assertEqual(data["versions"], solver.config()["packages"])
        self.assertEqual(data["passed"], {"AC-1": True, "AC-2": True, "AC-3": True, "AC-4": True, "AC-5": True})
        self.assertTrue((REPORT.parent / "mechanism_validation.md").read_text(encoding="utf-8").startswith(
            "# SL-16 mechanism validation"))


# ----------------------------------------------------------------------- AC-1
class HandSolvedTest(unittest.TestCase):
    """AC-1: at least 5 hand-solved markets match to 1e-8."""

    def test_the_hand_solved_markets_match_their_closed_forms(self):
        rows = mv.hand_solved()
        self.assertGreaterEqual(len({r["market"] for r in rows}), 5)
        for row in rows:
            with self.subTest(market=row["market"], quantity=row["quantity"]):
                self.assertTrue(row["passed"], row)
                self.assertLessEqual(row["error_per_request"], 1e-8)
        self.assertEqual([(r["market"], r["quantity"]) for r in report()["hand_solved"]],
                         [(r["market"], r["quantity"]) for r in rows])

    def test_the_closed_forms_cover_the_rules(self):
        names = " ".join(r["market"] for r in report()["hand_solved"])
        for fragment in ("inclusion", "gating aware", "gating gated", "setblind", "quadratic", "reserve uniform",
                         "reserve lognormal", "GSP M1", "GSP M2", "threshold"):
            self.assertIn(fragment, names)


# ------------------------------------------------------------- market model
def observation(ids, labels, task="t") -> Observation:
    return Observation(task=Task(task_id=task, instruction="choose"), set_id="s", presentation_id="s/p0",
                       candidates=[Candidate(product_id=i, position=k + 1, sponsored_label=lab)
                                   for k, (i, lab) in enumerate(zip(ids, labels))])


class ResponseTest(unittest.TestCase):
    def setUp(self):
        self.params = AgentParams(source="stipulated", models=[
            ModelParams(model_key="a", sigma=0.4, label_effect=-0.3, position_effects=[0.0, -0.1, -0.2],
                        outside_value=-0.5, page_request_prob=0.3, scores={"t": {"P1": 0.2, "P2": 0.9, "P3": -0.1}},
                        states=[StateParams(state_id="u", probability=0.3, shifts={"t": {"P1": 0.5}}),
                                StateParams(state_id="v", probability=0.7)]),
            ModelParams(model_key="b", sigma=1.2, label_effect=0.1, position_effects=[0.0, 0.0, -0.3],
                        outside_value=0.2, page_request_prob=0.8, scores={"t": {"P1": 0.0, "P2": 0.3, "P3": 0.6}},
                        states=[StateParams(state_id="p", probability=0.5), StateParams(state_id="q", probability=0.5,
                                                                                      shifts={"t": {"P3": -1.0}})])],
            humans=HumanParams(position_weights=[0.3, 0.2, 0.1],
                               product_weights={"t": {"P2": LabelWeights(organic=0.8, sponsored=0.5)}}))
        models, states = models_from_agent_params(self.params, {"x": "t"})
        offers = {j: Offer(**human_displays(self.params.humans, "t", j)) for j in ("P1", "P2", "P3")}
        self.context = Context(context_id="x", probability=1.0, offers=offers,
                               organic=[OrganicList(ranking=["P1", "P2", "P3"], probability=1.0)], agent_share=0.6,
                               model_shares={"a": 0.25, "b": 0.75})
        self.market = Market(market_id="sl06", requests=10, max_displayed=3, max_sponsored=1, advertisers=["P2"],
                             contexts=[self.context], states=states, position_weights=[0.3, 0.2, 0.1], models=models)

    def test_states_are_the_product_of_the_models_states(self):
        self.assertEqual([s.state_id for s in self.market.states], ["u__p", "u__q", "v__p", "v__q"])
        self.assertTrue(np.allclose([s.probability for s in self.market.states], [0.15, 0.15, 0.35, 0.35]))

    def test_agent_and_human_probabilities_equal_sl06(self):
        shown, labels = ("P2", "P3", "P1"), (True, False, False)
        r = response(self.market, self.context, shown, labels)
        obs = observation(shown, labels)
        for s, state in enumerate(self.market.states):
            a_state, b_state = state.state_id.split("__")
            expected = (0.25 * np.array(choice_probs(self.params.model("a"), obs, a_state).as_list(obs)[1:])
                        + 0.75 * np.array(choice_probs(self.params.model("b"), obs, b_state).as_list(obs)[1:]))
            np.testing.assert_allclose(r["agents"][s], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(r["human"], human_probs(self.params.humans, obs).as_list(obs)[1:], atol=1e-15)
        np.testing.assert_allclose(r["pooled"], 0.4 * r["human"][None, :] + 0.6 * r["agents"], atol=1e-15)

    def test_the_displayed_list_keeps_organic_integrity(self):
        self.assertEqual(displayed_list(["a", "b", "c", "d"], ("c",), 3), (("c", "a", "b"), (True, False, False)))
        self.assertEqual(displayed_list(["a", "b", "c", "d"], (), 3), (("a", "b", "c"), (False, False, False)))
        table = action_table(self.market)
        self.assertEqual(sorted(s for _, s in table.rows), [(), ("P2",)])

    def test_human_probabilities_above_one_are_rejected(self):
        heavy = self.context.model_copy(update={"offers": {j: mv.offer(request=1.0, conversion=1.0) for j in
                                                           ("P1", "P2", "P3")}})
        market = self.market.model_copy(update={"position_weights": [0.6, 0.5, 0.4]})
        with self.assertRaisesRegex(ValueError, "sum to"):
            response(market, heavy, ("P1", "P2", "P3"), (False, False, False))

    def test_the_vectorised_reference_equals_enumeration(self):
        market, _ = mv.random_market(5, None)
        expected = 0.0
        for c in market.contexts:
            best = -math.inf
            for shown in itertools.permutations(sorted(c.offers), market.max_displayed):
                qbar = np.array([s.probability for s in market.states]) @ response(market, c, shown, (False,) * 3)["pooled"]
                best = max(best, sum(c.offers[j].match * q for j, q in zip(shown, qbar)))
            expected += c.probability * best
        self.assertAlmostEqual(match_reference(market.model_copy(update={"guardrail": None})),
                               market.requests * expected, places=11)


# ------------------------------------------------------------------- costs
class CostTest(unittest.TestCase):
    def test_overflow_slope_and_tangent_error(self):
        for n, capacity in ((1000, 600), (100, 37.5), (20, 0.5)):
            p = np.linspace(0.01, 0.99, 37)
            h = 1e-6
            numeric = (costs.overflow_expectation(n, capacity, p + h) - costs.overflow_expectation(n, capacity, p - h)) / (2 * h)
            np.testing.assert_allclose(costs.overflow_slope(n, capacity, p), numeric, rtol=1e-6, atol=1e-6)
            tangents = costs.overflow_tangents(n, capacity)
            fine = np.linspace(0, 1, 100001)
            gap = costs.overflow_expectation(n, capacity, fine) / n - tangents.value(fine)
            self.assertGreaterEqual(gap.min(), -1e-12)  # an outer approximation never exceeds h
            self.assertLessEqual(gap.max(), tangents.max_error + 1e-12)  # the reported error is the maximum
            self.assertGreater(tangents.max_error, 0.95 * gap.max())

    def test_quadratic_cost_is_exact_through_lemma_load(self):
        n, w, p = 1000, np.array([0.5, 0.5]), np.array([1.0, 0.5])
        cost = Cost(kind="quadratic", weight=2.0, capacity=600)
        second_moment = float(w @ (n * p * (1 - p) + (n * p) ** 2))
        self.assertAlmostEqual(n * costs.expected_cost(cost, n, w, p), 2.0 / 1200 * second_moment, places=9)


# ------------------------------------------------------------------ values
class ValuesTest(unittest.TestCase):
    def test_uniform_values_are_regular(self):
        u = ValueDistribution(kind="uniform", upper=2.0)
        self.assertEqual(u.ironing, [])
        self.assertAlmostEqual(u.ironed(1.5), 1.0)
        self.assertAlmostEqual(u.ironed_inverse(0.0), 1.0)
        self.assertAlmostEqual(u.ironed_integral(0.5, 1.5), (1.5 ** 2 - 0.5 ** 2) - 2.0)

    def test_lognormal_ironing_is_monotone_and_tangent(self):
        self.assertEqual(ValueDistribution(kind="lognormal", mu=0.0, s=1.0).ironing, [])
        for mu, s in ((0.0, 2.0), (-0.5, 2.5), (1.0, 3.0)):
            d = ValueDistribution(kind="lognormal", mu=mu, s=s)
            self.assertEqual(len(d.ironing), 1)
            low, high, value = d.ironing[0]
            self.assertAlmostEqual(d.virtual_value(low), value, places=9)
            self.assertAlmostEqual(d.virtual_value(high) / high, value / high, places=12)
            # the hull's chord: H(q2) − H(q1) = φ̄ (q2 − q1)
            self.assertAlmostEqual(d.H(d.cdf(high)) - d.H(d.cdf(low)), value * (d.cdf(high) - d.cdf(low)), places=10)
            grid = np.exp(mu + s * np.linspace(-6, 6, 3001))
            ironed = np.array([d.ironed(t) for t in grid])
            self.assertTrue((np.diff(ironed) >= -1e-12).all())
            self.assertFalse((np.diff([d.virtual_value(t) for t in grid]) >= -1e-12).all())  # it needed ironing
            for c in (value - 1.0, value + 1e-3, 0.0, 2.0):
                self.assertAlmostEqual(d.ironed(d.ironed_inverse(c)), c, places=8)


# -------------------------------------------------------------- AC-2, AC-3
class PaymentIdentityTest(unittest.TestCase):
    """AC-2: externality and integral payments agree within 1e-6 relative on 100 random
    markets (surplus rule)."""

    def test_the_committed_identity_study(self):
        data = report()["identity"]
        self.assertEqual((data["markets"], data["valid_markets"], data["excluded"]), (100, 100, []))
        self.assertEqual(sorted(data["by_rule"]), ["M3", "M3_set_blind", "M4"])
        for rule, r in data["by_rule"].items():
            with self.subTest(rule=rule):
                self.assertGreater(r["payments"] - r["zero_payments"], 100)
                self.assertLessEqual(r["max_relative_error"], 1e-6)
        self.assertTrue(data["passed"])

    def test_the_identity_on_a_fresh_market(self):
        result = mv._identity_market(0)
        self.assertEqual(result["cost"], "quadratic")
        for row in result["rows"]:
            with self.subTest(rule=row["rule"], advertiser=row["advertiser"]):
                self.assertLessEqual(row["relative_error"], 1e-6)

    def test_payments_by_traffic_type_add_up(self):
        market, theta = mv.random_market(7, "quadratic")
        outcome = MarketRunner(market).evaluate("M4", theta, by_type=True)
        for j in market.advertisers:
            self.assertAlmostEqual(outcome.payments_human[j] + outcome.payments_agent[j], outcome.payments[j], places=9)
            self.assertLessEqual(outcome.identity_gap[j], 1e-6 * max(1.0, abs(outcome.payments[j])))


class IncentiveTest(unittest.TestCase):
    """AC-3: on 100 random markets X_j is monotone in b_j and no grid deviation gains more
    than 1e-6 (M3, M4, profit rule)."""

    def test_the_committed_incentive_study(self):
        data = report()["incentives"]
        self.assertEqual((data["markets"], data["valid_markets"], data["excluded"]), (100, 100, []))
        self.assertEqual(sorted(data["by_rule"]), ["M3", "M4", "profit_rule"])
        for rule, r in data["by_rule"].items():
            with self.subTest(rule=rule):
                self.assertGreaterEqual(r["min_increment_per_request"], -1e-9)
                self.assertLessEqual(r["max_gain"], 1e-6)
                self.assertGreater(r["ironed"], 0)  # irregular values were covered
        self.assertTrue(data["passed"])

    def test_truthful_bidding_on_a_fresh_market(self):
        market, theta = mv.random_market(3, None)
        program = MarketRunner(market).program("M3")
        truth = np.array([theta[j] for j in market.advertisers])
        for k, j in enumerate(market.advertisers):
            grid = np.unique(np.r_[np.linspace(0, 2.0, 11), truth[k]])
            x, paid = [], []
            for bid in grid:
                bids = truth.copy()
                bids[k] = bid
                ext, a = pay.externality_payments(program, bids)
                x.append(a.X[j])
                paid.append(ext[k])
            utility = truth[k] * np.array(x) - np.array(paid)
            self.assertGreaterEqual(np.diff(x).min(), -1e-9)
            self.assertLessEqual(utility.max() - utility[int(np.searchsorted(grid, truth[k]))], 1e-9)


# ----------------------------------------------------------------------- GSP
class GSPTest(unittest.TestCase):
    def test_fixed_point_bids_and_non_convergence_are_reported(self):
        market, values = mv.gsp_market()
        runner = MarketRunner(market)
        outcome = runner.evaluate("M1", values, bidding_rule="fixed_point_value_per_click")
        self.assertTrue(outcome.converged)
        self.assertGreater(outcome.iterations, 1)
        with mock.patch.object(gsp, "MAX_ITERATIONS", 1):
            stalled = runner.evaluate("M1", values, bidding_rule="fixed_point_value_per_click")
        self.assertFalse(stalled.converged)
        self.assertEqual(stalled.iterations, 1)

    def test_m2_charges_no_agent_request(self):
        market, values = mv.gsp_market()
        agents_only = Market.model_validate(market.model_copy(update={"contexts": [
            market.contexts[0].model_copy(update={"agent_share": 1.0})]}).model_dump())
        outcome = MarketRunner(agents_only).evaluate("M2", values, bidding_rule="human_conversion_rate")
        self.assertEqual(sum(outcome.payments.values()), 0.0)
        self.assertGreater(sum(MarketRunner(agents_only).evaluate("M1", values, bidding_rule="human_conversion_rate")
                               .payments.values()), 0.0)


# ----------------------------------------------------------------------- AC-4
class ExclusionTest(unittest.TestCase):
    """AC-4: solver failures are excluded and counted; none is used."""

    def test_the_committed_exclusion_check(self):
        data = report()
        self.assertTrue(data["exclusions"]["passed"])
        self.assertEqual(data["exclusions"]["excluded_count"], 1)
        self.assertNotIn("impossible-guardrail", data["exclusions"]["outcome_markets"])

    def test_an_infeasible_market_is_excluded_from_every_mechanism(self):
        market, theta = mv.random_market(9, None)
        impossible = Market.model_validate(market.model_copy(update={
            "market_id": "impossible", "guardrail": Guardrail(kappa=0.0, m_ref=10.0 * market.requests)}).model_dump())
        batch = evaluate_markets([(market, theta), (impossible, theta)], mechanisms=("M3", "M4"))
        self.assertEqual(batch.exclusions.count, 1)
        self.assertEqual(batch.exclusions.entries[0]["status"], "infeasible")
        self.assertEqual({o.market_id for o in batch.outcomes}, {market.market_id})
        self.assertEqual(len(batch.outcomes), 2)

    def test_an_inaccurate_status_is_a_failure_and_its_market_is_not_used(self):
        x = cp.Variable()
        problem = cp.Problem(cp.Maximize(x), [x <= 1])

        def inaccurate(*args, **kwargs):
            problem._status = "optimal_inaccurate"
            return 1.0

        with mock.patch.object(problem, "solve", side_effect=inaccurate):
            with self.assertRaises(solver.SolverFailure) as raised:
                solver.solve(problem, "lp", "test")
        self.assertEqual(raised.exception.status, "optimal_inaccurate")

        market, theta = mv.random_market(4, None)
        real = allocation.solve

        def flaky(problem, kind, where=""):
            if where.startswith(market.market_id):
                raise solver.SolverFailure("optimal_inaccurate", where)
            return real(problem, kind, where)

        other, other_theta = mv.random_market(6, None)
        with mock.patch.object(allocation, "solve", side_effect=flaky):
            batch = evaluate_markets([(market, theta), (other, other_theta)], mechanisms=MECHANISMS)
        self.assertEqual(batch.exclusions.count, 1)
        self.assertEqual(batch.exclusions.entries[0]["status"], "optimal_inaccurate")
        self.assertNotIn(market.market_id, {o.market_id for o in batch.outcomes})
        self.assertEqual(len(batch.outcomes), len(MECHANISMS))

    def test_a_failed_interior_point_qp_is_solved_by_the_fallbacks(self):
        """Clarabel failing (any non-optimal status) does not exclude the market when a
        fallback path finds the same exact optimum: the vertex-started active set, then ADMM."""
        market, theta = mv.random_market(2, "quadratic")
        bids = np.array([theta[j] for j in market.advertisers])
        reference = MarketRunner(market).program("M4").solve(bids)
        real = solver.solve

        def broken(problem, kind, where=""):
            if kind == "qp":
                raise solver.SolverFailure("user_limit", where)
            return real(problem, kind, where)

        for disable_vertex in (False, True):
            program = MarketRunner(market).program("M4")
            with self.subTest(vertex_path=not disable_vertex), mock.patch.object(allocation, "solve", side_effect=broken):
                if disable_vertex:
                    with mock.patch.object(allocation, "FALLBACK_STEPS", 0):
                        result = program.solve(bids)
                else:
                    result = program.solve(bids)
            self.assertEqual(program.fallbacks, ["admm after user_limit" if disable_vertex else "vertex after user_limit"])
            for j in market.products:
                self.assertAlmostEqual(result.X[j], reference.X[j], delta=1e-9 * market.requests)
            self.assertAlmostEqual(result.value, reference.value, places=10)

    def test_a_qp_that_cannot_be_polished_fails(self):
        market, theta = mv.random_market(2, "quadratic")
        program = MarketRunner(market).program("M4")
        self.assertEqual(program.kind, "qp")
        with mock.patch.object(allocation, "POLISH_STEPS", 0), mock.patch.object(allocation, "FALLBACK_STEPS", 0), \
                mock.patch.object(allocation, "polish", side_effect=solver.SolverFailure("unpolished", "test")):
            with self.assertRaises(solver.SolverFailure) as raised:
                program.solve(np.array([theta[j] for j in market.advertisers]))
        self.assertEqual(raised.exception.status, "unpolished")


# ----------------------------------------------------------------- AC-5, pins
class RuntimeTest(unittest.TestCase):
    """AC-5: runtime per market is reported and used to size SL-19."""

    def test_the_committed_runtime_profile(self):
        data = report()["runtime"]
        self.assertEqual([s["label"] for s in data["sizes"]], [s["label"] for s in mv.RUNTIME_SIZES])
        for size in data["sizes"]:
            self.assertEqual(sorted(size["seconds"]), sorted(MECHANISMS))
            self.assertGreater(size["total"], 0.0)
        core = data["sl19_core"]
        self.assertEqual((core["configurations"], core["markets_per_configuration"]), (30, 200))
        self.assertAlmostEqual(core["cpu_hours"], 30 * 200 * core["seconds_per_market_all_mechanisms"] / 3600)
        self.assertIn("SL-19 sizing", (REPORT.parent / "mechanism_validation.md").read_text(encoding="utf-8"))


class SolverConfigTest(unittest.TestCase):
    def test_versions_are_pinned_and_installed(self):
        installed = solver.check_versions()
        pins = dict(re.findall(r"^([a-z]+)==([0-9.]+)", REQUIREMENTS.read_text(encoding="utf-8"), re.M))
        for name, version in installed.items():
            self.assertEqual(pins.get(name), version, name)

    def test_the_platform_check_covers_every_package(self):
        check = solver.config()["platform_check"]
        for platform in ("linux_x86_64", "macos_arm64"):
            for name in ("cvxpy", "clarabel", "osqp", "scs"):
                with self.subTest(platform=platform, package=name):
                    wheel = check[platform][name]["wheel"]
                    self.assertIn("-%s-" % solver.config()["packages"][name], wheel)
                    self.assertTrue(re.search("x86_64" if platform == "linux_x86_64" else "arm64|universal2", wheel))
                    self.assertRegex(check[platform][name]["sha256"], "^[0-9a-f]{64}$")

    def test_each_program_class_has_one_pinned_solver(self):
        programs = solver.config()["programs"]
        self.assertEqual(sorted(programs), ["lp", "qp", "qp_fallback"])
        self.assertEqual(solver.config()["accepted_status"], ["optimal"])
        market, _ = mv.random_market(1, "quadratic")
        self.assertEqual(MarketRunner(market).program("M4").kind, "qp")
        self.assertEqual(MarketRunner(market).program("M3").kind, "lp")


if __name__ == "__main__":
    unittest.main()
