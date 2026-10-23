"""SL-15 choice-model estimation and the E1 test suite: recovery (AC-1), separation
(AC-2) and IIA size (AC-3), from the committed validation study and from fast checks
of the estimators themselves; agent_params.json round trip into SL-06 (AC-4).
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
from features.storelab import api, cli
from features.storelab.analysis import clogit, consistency, iia, page_requests, validation, versions
from features.storelab.analysis.data import ChoiceData, from_presentations, from_records
from features.storelab.analysis.simulate import Truth, _observation, simulate, synthetic_design
from features.storelab.backends.simulated import SimulatedAgent, choice_probs, params_from_config
from features.storelab.core.agent_params import dump_agent_params, load_agent_params
from features.storelab.core.config import apply_overrides, resolve_config
from features.storelab.core.design import build_design
from features.storelab.core.protocol import TrialOutcome
from tests.features.storelab.test_cache import Env

REPORT = Path(validation.__file__).resolve().parents[1] / "docs" / "analysis_validation.json"
DESIGNS = synthetic_design(7)


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


class CommittedStudyTest(unittest.TestCase):
    """The committed study was produced by the current code at the full replication counts."""

    def test_the_report_matches_the_code_and_the_registered_counts(self):
        data = report()
        self.assertEqual(data["code_fingerprint"], validation.code_fingerprint())
        self.assertEqual(data["replications"], validation.FULL)
        self.assertEqual(data["master_seed"], validation.MASTER_SEED)
        self.assertTrue((REPORT.parent / "analysis_validation.md").read_text(encoding="utf-8").startswith(
            "# SL-15 analysis validation"))


# ----------------------------------------------------------------------- AC-1
class RecoveryTest(unittest.TestCase):
    """AC-1: for sigma in {1, 0.3}, 95% CI coverage of delta/sigma and beta/sigma in
    [0.92, 0.98] and |bias| < 0.1 SE over 200 replications."""

    def test_the_committed_recovery_study(self):
        """Coverage is required; the bias half of AC-1 is waived by the owner
        (agent-workflow/waivers/SL-15.json) and is reported with its Monte Carlo error."""
        blocks = {b["sigma"]: b for b in report()["recovery"]}
        self.assertEqual(sorted(blocks), [0.3, 1.0])
        for sigma, block in blocks.items():
            self.assertEqual((block["replications"], block["estimator"]), (200, "firth"))
            self.assertEqual([r["parameter"] for r in block["parameters"]],
                             ["delta"] + ["beta_%d" % k for k in range(2, 9)])
            for row in block["parameters"]:
                with self.subTest(sigma=sigma, parameter=row["parameter"]):
                    self.assertGreaterEqual(row["profile_coverage"], 0.92)
                    self.assertLessEqual(row["profile_coverage"], 0.98)
                    self.assertTrue(math.isfinite(row["bias_over_se"]))
                    self.assertAlmostEqual(row["bias_mc_se_over_se"], 200 ** -0.5)
            self.assertTrue(block["coverage_passed"])

    def test_score_and_information_are_the_derivatives_of_the_log_likelihood(self):
        data = simulate(DESIGNS, Truth(sigma=1.0), 1, 0)
        model = clogit.Model(data)
        rng = np.random.default_rng(0)
        theta = rng.normal(0, 0.3, model.p)
        score, info, _ = model.score_info(theta)
        h = 1e-5
        numeric = np.array([(model.loglik(theta + h * e) - model.loglik(theta - h * e)) / (2 * h)
                            for e in np.eye(model.p)])
        np.testing.assert_allclose(score, numeric, rtol=1e-5, atol=1e-4)
        hessian = np.array([(model.score_info(theta + h * e)[0] - model.score_info(theta - h * e)[0]) / (2 * h)
                            for e in np.eye(model.p)])
        np.testing.assert_allclose(-hessian, info, rtol=1e-4, atol=1e-3)

    def test_the_firth_score_is_the_gradient_of_half_log_det_information(self):
        """Including the outside option's term (z_0 = -x_bar): the modified score must be
        the exact gradient of the penalised log likelihood."""
        data = simulate(synthetic_design(3, tasks=1, set_sizes=(2, 4), repetitions=3), Truth(sigma=1.0), 3, 0)
        model = clogit.Model(data)
        theta = np.random.default_rng(1).normal(0, 0.5, model.p)
        _, info, flat = model.score_info(theta)
        analytic = model.firth_score(theta, info, flat, model.active)

        def half_logdet(t):
            return 0.5 * np.linalg.slogdet(model.score_info(t)[1])[1]

        h = 1e-5
        numeric = np.array([(half_logdet(theta + h * e) - half_logdet(theta - h * e)) / (2 * h)
                            for e in np.eye(model.p)])
        np.testing.assert_allclose(analytic, numeric, atol=1e-7)

    def test_a_large_sample_recovers_the_truth_and_profile_matches_wald(self):
        designs = synthetic_design(3, repetitions=40)
        truth = Truth(sigma=0.3)
        result = clogit.fit(simulate(designs, truth, 3, 0), profile=["delta", "beta_4"])
        self.assertTrue(result.firth)  # registered: Firth in every cell
        self.assertFalse(result.separation.separated)
        for name, value in truth.on_sigma_scale().items():
            self.assertLess(abs(result.value(name) - value), 4 * result.std_error(name), name)
        for name in ("delta", "beta_4"):
            low, high = result.ci[name]
            wald = result.wald_ci(name)
            self.assertLess(abs(low - wald[0]) + abs(high - wald[1]), 0.2 * (wald[1] - wald[0]))
            self.assertLess(low, result.value(name))
            self.assertGreater(high, result.value(name))

    def test_the_fit_reproduces_the_data_generating_probabilities_scale(self):
        """Scores are v/sigma relative to the outside option: their differences recover the truth's."""
        designs = synthetic_design(5, repetitions=60)
        truth = Truth(sigma=1.0)
        result = clogit.fit(simulate(designs, truth, 5, 0))
        params = truth.params(designs)
        for task, scores in result.scores().items():
            for product, value in scores.items():
                expected = (params.scores[task][product] - truth.outside_value) / truth.sigma
                self.assertLess(abs(value - expected), 0.5, (task, product))


# ----------------------------------------------------------------------- AC-2
class SeparationTest(unittest.TestCase):
    """AC-2: for sigma = 0.05, separation is flagged in >= 95% of the replications
    where it occurs, and penalised estimates are returned rather than a failure."""

    def test_the_committed_separation_study(self):
        s = report()["separation"]
        self.assertEqual(s["replications"], 200)
        self.assertGreater(s["occurs"], 0)
        self.assertGreaterEqual(s["flag_share"], 0.95)
        self.assertEqual(s["fit_failures"], 0)
        self.assertEqual(s["finite_firth_fits"], 200)
        self.assertTrue(s["passed"])

    def test_a_never_chosen_product_is_flagged_and_firth_returns_finite_estimates(self):
        data = simulate(DESIGNS, Truth(sigma=1.0), 2, 0)
        chosen = data.chosen_product()
        target = 0  # make product 0 never chosen: reassign its choices to the outside option
        forced = data.chosen.copy()
        forced[chosen == target] = -1
        separated = ChoiceData(data.products, data.prod, data.pos, data.label, forced, data.set_ids, data.reps)
        self.assertIn("never chosen: %s/%s" % data.products[target], validation.separation_patterns(separated))
        check = clogit.detect_separation(separated)
        self.assertTrue(check.separated)
        self.assertIn("v:%s/%s" % data.products[target], check.parameters)
        result = clogit.fit(separated)
        self.assertTrue(result.firth and result.converged)
        self.assertTrue(np.isfinite(result.theta).all())
        mle = clogit.fit(separated, penalty="none", detect=False)
        self.assertLess(result.theta[target], 0)
        self.assertGreater(result.theta[target], mle.theta[target])  # Firth pulls the diverging score back

    def test_clean_data_is_not_flagged(self):
        check = clogit.detect_separation(simulate(DESIGNS, Truth(sigma=1.0), 2, 1))
        self.assertFalse(check.separated)
        self.assertEqual(check.parameters, [])

    def test_near_deterministic_data_gives_a_finite_firth_fit(self):
        result = clogit.fit(simulate(DESIGNS, Truth(sigma=0.05), 2, 2))
        self.assertTrue(result.separation.separated and result.firth and result.converged)
        self.assertTrue(np.isfinite(result.theta).all())
        self.assertLess(abs(result.label_effect - (-10.0)), 3.0)


# ----------------------------------------------------------------------- AC-3
class IIATest(unittest.TestCase):
    """AC-3: empirical size of the IIA tests at alpha = 0.05 in [0.03, 0.07]; power reported."""

    def test_the_committed_size_and_power_study(self):
        data = report()
        size, power = data["iia_size"], data["iia_power"]
        self.assertEqual(size["replications"], 1000)
        for test in ("triple_odds_monte_carlo", "hausman_mcfadden_1df"):
            with self.subTest(test=test):
                self.assertGreaterEqual(size["rejection_rate"][test], 0.03)
                self.assertLessEqual(size["rejection_rate"][test], 0.07)
                self.assertIn(test, power["rejection_rate"])  # reported, no threshold
        self.assertTrue(size["passed"])

    def test_the_triple_statistic_equals_a_direct_count(self):
        data = simulate(DESIGNS, Truth(sigma=1.0), 4, 0)
        result = clogit.fit(data, penalty="none", detect=False)
        scores = result.theta[:len(data.products)]
        triples = iia.TripleOdds(data, scores)
        statistic, informative = triples.statistic(triples.counts(data.chosen))
        chosen = data.chosen_product()
        sets: dict[str, dict] = {}
        for row, set_id in enumerate(data.set_ids):
            entry = sets.setdefault(set_id, {"members": set(int(p) for p in data.prod[row] if p >= 0), "n": {}})
            entry["n"][int(chosen[row])] = entry["n"].get(int(chosen[row]), 0) + 1
        total, count = 0.0, 0
        groups: dict = {}
        for entry in sets.values():
            groups.setdefault((data.products[min(entry["members"])][0], len(entry["members"])), []).append(entry)
        for entries in groups.values():
            products = sorted(set().union(*[e["members"] for e in entries]))
            for a, i in enumerate(products):
                for j in products[a + 1:]:
                    hold = [e for e in entries if {i, j} <= e["members"]]
                    for k in products:
                        if k in (i, j) or not scores[k] > max(scores[i], scores[j]) or len(hold) < 2:
                            continue
                        with_k = [e for e in hold if k in e["members"]]
                        without = [e for e in hold if k not in e["members"]]
                        if not with_k or not without:
                            continue
                        n = [sum(e["n"].get(p, 0) for e in group) for group in (with_k, without) for p in (i, j)]
                        if min(n) > 0:
                            count += 1
                            total += (np.log((n[0] + .5) / (n[1] + .5)) - np.log((n[2] + .5) / (n[3] + .5))) ** 2
        self.assertEqual(informative, count)
        self.assertAlmostEqual(statistic, total, places=9)

    def test_the_tests_return_valid_p_values(self):
        data = simulate(DESIGNS, Truth(sigma=1.0, second_state_shift=1.5), 4, 1)
        result = clogit.fit(data, penalty="none", detect=False)
        for method in ("monte_carlo", "permutation"):
            out = iia.triple_odds_test(data, result, 49, 4, "t", method)
            self.assertTrue(1 / 50 <= out.p_value <= 1.0)
            self.assertEqual(out.method, method)
        again = iia.triple_odds_test(data, result, 49, 4, "t")
        self.assertEqual(again.p_value, iia.triple_odds_test(data, result, 49, 4, "t").p_value)  # seeded
        hm = iia.hausman_mcfadden(data)
        self.assertEqual(hm.df, 1)
        self.assertTrue(0.0 <= hm.p_value <= 1.0)
        self.assertEqual(sorted(hm.dropped), sorted({t for t, _ in data.products}))
        reduced = iia.without_products(data, {0})
        self.assertFalse((reduced.prod == 0).any())
        self.assertFalse((reduced.chosen_product() == 0).any())

    def test_the_quick_study_runs_end_to_end(self):
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out)
        counts = {"recovery": 2, "separation": 2, "iia": 2, "mc": 9, "permutation": 9}
        result = validation.run(counts)
        self.assertEqual(set(result["passed"]), {"AC-1 coverage", "AC-1 bias", "AC-2", "AC-3"})
        self.assertIn("## IIA tests", validation.markdown(result))


# ------------------------------------------------------- other statistics
class StatisticsTest(unittest.TestCase):
    def test_modal_shares_and_eligibility(self):
        design = build_design("t", ["a", "b", "c"], 2, 10, 0.0, "per_set", 1)
        rows = []
        for s in design.sets[:2]:
            presentation = design.presentation_for(s.set_id, 0)
            first = presentation.candidates[0].product_id
            rows += [("t", presentation, r, first if r < 9 else None) for r in range(10)]
        third = design.presentation_for(design.sets[2].set_id, 0)
        rows += [("t", third, r, None) for r in range(7)]  # 7 valid repeats: ineligible
        shares = consistency.modal_shares(from_presentations(rows))
        self.assertEqual(shares.shares, {design.sets[0].set_id: 0.9, design.sets[1].set_id: 0.9})
        self.assertEqual(shares.ineligible, [design.sets[2].set_id])
        test = consistency.modal_share_test(from_presentations(rows), 99, 1, "m")
        self.assertAlmostEqual(test["estimate"], 0.9)
        self.assertEqual((test["p_support"], test["p_against"]), (0.0, 1.0))

    def test_score_gaps(self):
        data = simulate(DESIGNS, Truth(sigma=0.3), 6, 0)
        result = clogit.fit(data)
        gaps = consistency.score_gaps(data, result)
        self.assertEqual(len(gaps), len(set(data.set_ids)))
        row = 0
        slots = data.prod[row] >= 0
        values = np.sort(np.concatenate([[0.0], result.theta[data.prod[row][slots]]
                                         + result.label_effect * data.label[row][slots]]))[::-1]
        self.assertAlmostEqual(gaps[data.set_ids[row]], values[0] - values[1])

    def test_version_comparison_finds_a_changed_favourite(self):
        designs = synthetic_design(8, tasks=2, repetitions=20)
        a = Truth(sigma=0.3)
        data_a = simulate(designs, a, 8, 0)
        # version B: the order of products reversed within each task
        flipped = Truth(sigma=0.3, score_range=(4.8, 3.6))
        data_b = simulate(designs, flipped, 8, 1)
        comparison = versions.compare(data_a, data_b, 19, 8, "v")
        self.assertEqual(comparison.counts()["counts"]["raw_changes"], 2)
        for task in comparison.tasks:
            self.assertLess(task.spearman, -0.8)
            self.assertTrue(task.credible_change)
        utility = {t: {pid: float(i) for i, pid in enumerate(sorted(s))} for t, s in a.params(designs).scores.items()}
        changes = comparison.credible_changes("pair", "text", utility)
        self.assertEqual({c["higher_u"] for c in changes}, {False})  # B's favourite is the lowest-u product

    def test_page_request_rate(self):
        env = Env()
        self.addCleanup(env.cleanup)
        config = env.config(**{"mode": "auto", "interfaces": ["browser_tool"],
                               "protocol.prompt_templates": {"browser_tool": "e1_browser_tool@1"}})
        run_dir, result = api.run(config, snapshot=env.snapshot, root=env.runs, environ={})
        records = list(api.records_in_plan_order(run_dir))
        rate = page_requests.page_request_rate(records)
        others = sum(len(r.candidate_ids) - (r.outcome.chosen_id is not None) for r in records)
        self.assertEqual((rate.trials, rate.non_chosen), (len(records), others))
        self.assertLess(abs(rate.rate - config.agents.simulated.page_request_prob), 0.1)
        low, high = rate.ci()
        self.assertTrue(low <= rate.rate <= high)
        self.assertEqual(rate.decide_input()["counts"], {"requested": rate.requested})

    def test_records_give_complete_case_and_sensitivity_data(self):
        env = Env()
        self.addCleanup(env.cleanup)
        config = env.config(mode="auto")
        run_dir, _ = api.run(config, snapshot=env.snapshot, root=env.runs, environ={})
        records = list(api.records_in_plan_order(run_dir))
        _, stimuli = api.load_plan(__import__("features.storelab.core.store", fromlist=["LocalStore"]).LocalStore(
            run_dir))
        broken = records[0].model_copy(update={"outcome": TrialOutcome(status="invalid", invalid_reason="no_json",
                                                                       backend="simulated")})
        records = [broken] + records[1:]
        primary = from_records(records, stimuli.designs)
        sensitivity = from_records(records, stimuli.designs, missing="outside")
        self.assertEqual((primary.n_trials, sensitivity.n_trials), (len(records) - 1, len(records)))
        self.assertEqual(sensitivity.chosen[0], -1)


# ----------------------------------------------------------------------- AC-4
class AgentParamsTest(unittest.TestCase):
    """AC-4: agent_params.json validates and loads into SL-06 (round trip)."""

    def test_estimates_round_trip_into_the_simulated_backend(self):
        data = simulate(DESIGNS, Truth(sigma=0.3), 9, 0)
        result = clogit.fit(data)
        params = clogit.agent_params_from_fits({"gpt_4o_20240806": (result, 0.25)}, {"study": "sl15-test"})
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        path = directory / "agent_params.json"
        path.write_text(dump_agent_params(params), encoding="utf-8")
        schema = json.loads((cli.CONTRACTS / "agent_params.schema.json").read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(json.loads(path.read_text(encoding="utf-8")))
        loaded = load_agent_params(path)
        self.assertEqual(loaded, params)
        model = loaded.model("gpt_4o_20240806")
        self.assertEqual((model.sigma, model.outside_value, model.position_effects[0]), (1.0, 0.0, 0.0))
        fitted, fitted_outside = clogit.Model(data).probabilities(np.nan_to_num(result.theta))
        agent = SimulatedAgent(model, seed=1)
        for design in DESIGNS:
            for s in design.sets:
                presentation = design.presentation_for(s.set_id, 0)
                probs = agent.choice_probs(_observation(design.task_id, presentation))
                row = data.set_ids.index(s.set_id)
                self.assertAlmostEqual(probs.outside, fitted_outside[row], places=12)
                for slot, c in enumerate(presentation.candidates):
                    self.assertAlmostEqual(probs.products[c.product_id], fitted[row, slot], places=12)
        self.assertEqual(choice_probs(model, _observation(DESIGNS[0].task_id, DESIGNS[0].presentations[0])).products,
                         agent.choice_probs(_observation(DESIGNS[0].task_id, DESIGNS[0].presentations[0])).products)

    def test_a_config_pins_and_loads_the_estimated_file(self):
        import hashlib

        env = Env()
        self.addCleanup(env.cleanup)
        data = simulate(DESIGNS, Truth(sigma=1.0), 9, 1)
        params = clogit.agent_params_from_fits({"gpt_4o_20240806": (clogit.fit(data), 0.1)}, {})
        path = env.root / "agent_params.json"
        path.write_text(dump_agent_params(params), encoding="utf-8")
        config = env.config(mode="auto")
        data_config = apply_overrides(config.model_dump(mode="json"), {
            "agents.simulated": {"parameters": "estimated", "params_file": {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}})
        loaded = params_from_config(resolve_config(data_config), [])
        self.assertEqual(loaded, params)


if __name__ == "__main__":
    unittest.main()
