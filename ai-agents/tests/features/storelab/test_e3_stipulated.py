"""SL-20S: E3 Part A (prop:threshold) and Part B on A17's stipulated states.

- AC-1: the closed forms of A1 §2.3 and the convex program agree under the A12 §4 tolerances,
  and the committed Part A covers the whole A17 §3.1 grid with every point passing;
- AC-2: Example ex:diversify through the E3 code path;
- AC-3: in not-aligned states the variance components sum to the total, and the separated
  policy matches the original exactly with no across-state variance (deterministic limit);
- AC-4: reproducibility (committed rows regenerate; a seeded rerun was byte for byte equal)
  and the label on Part B tables and figures;
- AC-5: H3a–H3c are reported with the registered statistics, labelled exploratory.

Also: the E3 environment against SL-16's probabilities and costs, and the Part B markets
against A17 §3.2 and the owner decisions of 2026-10-24.
"""
from __future__ import annotations

import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from features.storelab.analysis import e2_run, e3, e3_report, e3_run
from features.storelab.analysis.figures import FIGURES, LABEL
from features.storelab.core.mechanisms import costs as cost_model
from features.storelab.core.mechanisms.market import Cost, action_table
from scipy import stats


class PartATest(unittest.TestCase):
    """AC-1 on fresh solves of the convex program."""

    def test_closed_forms(self):
        self.assertEqual(e3.y_star_quadratic(1000, 0.5, 1.0, 0.0), 0.0)
        self.assertAlmostEqual(e3.y_star_quadratic(1000, 0.5, 1.0, 0.4), 0.4 * 500 / 999)
        self.assertEqual(e3.y_star_quadratic(10, 1.0, 0.25, 1.6), 1.0)
        self.assertEqual(e3.y_star_overflow(0.3, 1.0, 0.4), 0.4)  # |2ω − 1|
        self.assertEqual(e3.y_star_overflow(0.3, 1.0, 1.6), 1.0)  # Δ > λ
        self.assertEqual(e3.y_star_overflow(1.0, 4.0, 0.05), 1.0)  # ω ≥ 1
        with self.assertRaises(ValueError):
            e3.y_star_overflow(0.5, 1.0, 1.0)  # knife-edge
        # J(y)/n is the exact objective under symmetric policies
        env = e3.two_leader(100, 0.4)
        costs = e3.two_leader_costs("quadratic", 100, 0.5, 2.0)
        for y in (0.0, 0.3, 1.0):
            w = e3.policy(env, {("1", "2"): y, ("1", "3"): (1 - y) / 2, ("2", "3"): (1 - y) / 2})
            self.assertAlmostEqual(e3.exact_objective(env, w, costs), e3.objective_quadratic(y, 100, 0.5, 2.0, 0.4),
                                   places=13)
            self.assertAlmostEqual(e3.limit_cost_objective(env, w, e3.two_leader_costs("overflow", 100, 0.6, 1.0)),
                                   e3.limit_objective(y, 0.6, 1.0, 0.4), places=13)

    def test_the_program_reaches_the_closed_forms(self):
        for n, omega, lam, delta in ((10, 0.3, 4.0, 0.05), (1000, 0.8, 0.25, 1.6), (100, 0.6, 1.0, 0.4)):
            row = e3_run.evaluate_part_a(("quadratic", n, omega, lam, delta))[0]
            self.assertLessEqual(row["y_error"], 1e-4)
            self.assertLessEqual(row["relative_error"], 1e-6)
            self.assertAlmostEqual(row["y13"], row["y23"], places=6)  # unique and symmetric (A1 §E6)
            for case in ("overflow_limit", "overflow_finite"):
                row = e3_run.evaluate_part_a((case, n if case == "overflow_finite" else 1000, omega, lam, delta))[0]
                self.assertTrue(row["passed"], row)

    def test_the_grid_is_a17s(self):
        tasks = e3_run.part_a_tasks()
        self.assertEqual(sum(t[0] == "quadratic" for t in tasks), 450)
        self.assertEqual(sum(t[0] == "overflow_limit" for t in tasks), 150)
        self.assertEqual(sum(t[0] == "overflow_finite" for t in tasks), 450)
        self.assertEqual({t[1] for t in tasks if t[0] == "overflow_finite"}, {100, 1000, 10000})
        self.assertFalse(any(t[4] == t[3] for t in tasks))  # no Δ = λ (A1 §E6)

    def test_the_committed_part_a_passes_everywhere(self):
        rows = e2_run.read_rows(e3_run.RESULTS / "part_a.csv.gz")
        self.assertEqual(len(rows), 1050)
        self.assertEqual({r["passed"] for r in rows}, {"1"})


class DiversifyTest(unittest.TestCase):
    """AC-2."""

    def test_example_diversify_through_the_e3_code_path(self):
        env = e3.diversify()
        a = env.index["A"]
        for name, lottery in e3.DIVERSIFY_POLICIES.items():
            m = e3.moments(env, e3.policy(env, lottery), det=True)
            self.assertEqual(m["within"][a] + m["across"][a], e3.DIVERSIFY_VARIANCE[name], name)
            self.assertAlmostEqual(m["total"][a] / e3.DIVERSIFY_VARIANCE[name], 1.0, places=10)
        m = e3.moments(env, e3.policy(env, e3.DIVERSIFY_POLICIES["{A,B} or {A,C}"]), det=True)
        self.assertEqual((m["within"][a], m["across"][a]), (125.0, 62_500.0))
        self.assertEqual(m["theta"][a], 0.5)
        self.assertEqual(m["across"][a], env.n ** 2 * m["theta"][a] ** 2 / 4)  # the bound is attained
        separated = e3.separate(env, e3.policy(env, {("A", "B"): 1.0}), {"A": ("A", "C"), "B": ("B", "C")})
        self.assertEqual(list(separated), [0.0, 0.5, 0.5])


class EnvironmentTest(unittest.TestCase):
    """The E3 code path against SL-16 and against hand calculations."""

    def setUp(self):
        self.level = e3_run.LEVELS["human_share=0.25"]
        self.d, self.market, self.env, self.policies, self.ranking = e3_run.setup(
            "part_b/not_aligned/human_share=0.25", "not_aligned", self.level, 3)

    def test_probabilities_equal_sl16s(self):
        table = action_table(self.market)  # the market's organic policy is the original ranking, no sponsorship
        self.assertEqual(len(table.rows), 1)
        cols = [table.index[j] for j in self.env.products]
        np.testing.assert_allclose(self.env.q[:, 0, :], table.q_state[:, 0, cols], rtol=0, atol=1e-15)

    def test_the_deterministic_limit_and_contested_sets(self):
        env = self.env
        original = env.lists[0]
        self.assertEqual(original[:2], e3_run.LEADERS)
        for s, (_, _, bases) in enumerate(e3_run.states(self.level)):
            favoured = "L1" if bases[0] == 0 else "L2"
            np.testing.assert_array_equal(env.agent_det[s, 0], [1.0 if j == favoured else 0.0 for j in env.products])
        alpha = 1 - self.level.human_share
        self.assertEqual(env.contested[0, env.index["L1"]], alpha)  # the head-to-head list is contested
        self.assertEqual(env.contested[1, env.index["L1"]], 0.0)  # its anchor is not
        for k, leader in enumerate(e3_run.LEADERS):  # anchors: every agent type chooses the leader
            self.assertTrue(np.all(env.agent_det[:, 1 + k, env.index[leader]] == 1.0))

    def test_moments_and_costs(self):
        env = self.env
        rng = np.random.default_rng(7)
        w = rng.dirichlet(np.ones(len(env.lists)))
        for det in (False, True):
            m = e3.moments(env, w, det=det)
            np.testing.assert_allclose(m["within"] + m["across"], m["total"], rtol=1e-9)
            p = e3.p_state(env, w, det)
            for j in range(len(env.products)):  # Lemma load by hand
                self.assertAlmostEqual(m["across"][j], env.n ** 2 * np.var(p[:, j]), delta=1e-9 * max(1.0, m["across"][j]))
                for cost in (Cost(kind="quadratic", weight=0.5, capacity=500.0),
                             Cost(kind="overflow", weight=2.0, capacity=300.0)):
                    self.assertAlmostEqual(e3.expected_cost(m["pmf"][j], cost) / env.n,
                                           cost_model.expected_cost(cost, env.n, env.state_probs, p[:, j]), places=12)
        quota = self.policies["quota"][0]
        m = e3.moments(env, quota, quota=True)
        np.testing.assert_allclose(m["within"] + m["across"], m["total"], rtol=1e-9)
        # under a quota the within part sums the per-list binomial variances
        counts = e3.quota_counts(env, quota)
        j = env.index["L1"]
        by_hand = env.state_probs @ np.array([sum(c * q * (1 - q) for c, q in zip(counts, env.q[s, :, j]))
                                              for s in range(len(env.state_probs))])
        self.assertAlmostEqual(m["within"][j], by_hand, places=9)


class MarketTest(unittest.TestCase):
    """Part B markets against A17 §3.2 and the owner decisions of 2026-10-24."""

    def test_draws(self):
        for index in range(20):
            d = e3_run.draw("part_b/aligned/baseline", index)
            z = np.array(sorted(d.z.values()))
            self.assertEqual(d.z["L1"], d.z["L2"])
            self.assertEqual(d.z["L1"], d.z_leader)
            self.assertTrue(all(d.z[j] <= d.z_leader - e3_run.GAP for j in d.z if j not in e3_run.LEADERS))
            self.assertEqual(len(z), 10)
            self.assertEqual(len(d.advertisers), 6)
            self.assertTrue(all(0 <= v <= 1 for v in d.values.values()))
        # the leaders are the mean of the two largest of 10 standard normal draws
        rng = e3_run.stream(e3_run.MASTER_SEED, "sl20s-market", "part_b/aligned/baseline", 0)
        first = rng.standard_normal(10)
        self.assertAlmostEqual(e3_run.draw("part_b/aligned/baseline", 0).z_leader, np.sort(first)[-2:].mean())

    def test_states_scores_and_match_quality(self):
        d = e3_run.draw("part_b/aligned/baseline", 1)
        for alignment in e3_run.ALIGNMENTS:
            market, u = e3_run.build(d, e3_run.LEVELS["baseline"], alignment)
            self.assertEqual([s.probability for s in market.states], [0.5, 0.5])
            m = market.models[0]
            self.assertEqual((m.sigma, m.outside["x"], m.label_effect), (0.3, -1.0, -0.5))
            names = sorted(d.z)
            self.assertEqual(m.scores["s1"]["x"]["L1"] - d.z["L1"], 0.5)
            self.assertEqual(m.scores["s2"]["x"]["L1"] - d.z["L1"], -0.5)
            self.assertEqual(m.scores["s1"]["x"]["n1"], d.z["n1"])
            i = names.index("L1")
            if alignment == "not_aligned":
                self.assertEqual(u[0, i], u[1, i])
                self.assertAlmostEqual(u[0, i], stats.norm.cdf(d.z["L1"]))
            else:
                self.assertAlmostEqual(u[0, i], stats.norm.cdf(d.z["L1"] + 0.5))
                self.assertAlmostEqual(u[1, i], stats.norm.cdf(d.z["L1"] - 0.5))
        self.assertEqual(len(e3_run.states(e3_run.LEVELS["models=2,independent"])), 4)
        opposite = e3_run.states(e3_run.LEVELS["models=2,opposite"])
        self.assertEqual([b for _, _, b in opposite], [[0, 1], [1, 0]])
        self.assertEqual([p for _, p, _ in e3_run.states(e3_run.LEVELS["states=3"])], [1 / 3] * 3)

    def test_the_original_list_maximises_match_quality_in_the_deterministic_limit(self):
        for alignment in e3_run.ALIGNMENTS:
            d, market, env, policies, ranking = e3_run.setup("part_b/%s/baseline" % alignment, alignment,
                                                             e3_run.LEVELS["baseline"], 5)
            subsets = [tuple(s) for s in itertools.combinations(sorted(d.z), e3_run.K)]
            full = e3.environment(market, subsets, match=env.match)
            best = max(e3.value(full, np.eye(len(subsets))[k], det=True)["match"] for k in range(len(subsets)))
            self.assertAlmostEqual(e3.value(env, policies["original"][0], det=True)["match"], best, places=14)

    def test_policies(self):
        d, market, env, policies, ranking = e3_run.setup("part_b/not_aligned/state_probs=0.7,0.3", "not_aligned",
                                                         e3_run.LEVELS["state_probs=0.7,0.3"], 0)
        np.testing.assert_allclose(policies["separated"][0], [0.0, 0.7, 0.3])  # ϖ = the state probabilities
        np.testing.assert_array_equal(policies["composition_blind"][0], [0.5, 0.5, 0.0])
        self.assertTrue(policies["quota"][1])
        self.assertNotIn("L2", env.lists[1])
        self.assertNotIn("L1", env.lists[2])


def summary() -> dict:
    return json.loads(e3_report.SUMMARY.read_text(encoding="utf-8"))


class ResultsTest(unittest.TestCase):
    """AC-3 to AC-5 on the committed result files and report."""

    def test_the_gates_pass(self):
        g = e3_report.gates(e3_report.load())
        self.assertTrue(g["passed"])
        self.assertTrue(g["part_a_passed"] and g["diversify_passed"] and g["separation_exact_passed"])
        self.assertLessEqual(g["separation_exact"]["max_across"], 1e-12)
        for v in g["decomposition"].values():
            self.assertLessEqual(max(v["max_error"], v["max_error_det"]), 1e-9)
        self.assertEqual(set(g["markets"].values()), {200})
        self.assertGreaterEqual(len(g["separation_exact"]["configurations"]), 1)

    def test_a_seeded_rerun_reproduces_the_committed_rows(self):
        for config_id, alignment, level in (("part_b/aligned/states=3", "aligned", "states=3"),
                                            ("part_b/not_aligned/models=2,independent", "not_aligned",
                                             "models=2,independent")):
            rows = [r for i in range(2) for r in e3_run.evaluate_part_b((config_id, alignment, level, i))]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "x.csv.gz"
                e2_run.write_rows(path, e3_run.PART_B_FIELDS, rows)
                fresh = e2_run.read_rows(path)
            self.assertEqual(fresh, e2_run.read_rows(e2_run.result_path(e3_run.RESULTS, config_id))[:len(fresh)])
        rows = [r for i in range(2) for r in e3_run.evaluate_h3c(i)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv.gz"
            e2_run.write_rows(path, e3_run.H3C_FIELDS, rows)
            fresh = e2_run.read_rows(path)
        self.assertEqual(fresh, e2_run.read_rows(e3_run.RESULTS / "h3c.csv.gz")[:len(fresh)])
        self.assertTrue(summary()["rerun_identical"])

    def test_tables_and_figures_carry_their_labels(self):
        report = e3_report.REPORT.read_text(encoding="utf-8")
        self.assertIn(LABEL, report)
        self.assertIn("not been independently reviewed", report)
        self.assertNotRegex(report, r"\b(Supported|Not supported)\b")
        for k, name in enumerate(e3_report.FIGURE_NAMES):
            label = e3_report.PART_A_LABEL if k < 2 else LABEL
            for suffix in (".pdf", ".png", ".csv"):
                self.assertTrue((FIGURES / "sl20s" / (name + suffix)).exists(), name + suffix)
            self.assertTrue((FIGURES / "sl20s" / (name + ".csv")).read_text(encoding="utf-8").startswith("# " + label))
        for name in ("part_b_means.csv", "part_b_paired_differences.csv"):
            self.assertTrue((e3_report.TABLES / name).read_text(encoding="utf-8").startswith("# " + LABEL))

    def test_the_registered_statistics_are_reported_and_exploratory(self):
        s = summary()["statistics"]
        for key in ("H3a", "H3b", "H3c"):
            self.assertEqual(s[key]["label"], "exploratory")
        self.assertNotIn("decision", json.dumps(s))
        self.assertEqual(set(s["H3a"]["pr_part"]), {c for c, _, _ in e3_run.configurations()})
        for m in ("M3", "M4"):
            h = s["H3c"][m]
            self.assertGreater(h["pairs"], 0)
            self.assertLessEqual(h["max_margin_check"], 1e-9)
            self.assertIn("coincidence", h)
            self.assertIn("buyback_share_ci95", h)

    def test_h3c_records_the_guardrail_floor(self):
        rows = [r for r in e2_run.read_rows(e3_run.RESULTS / "h3c.csv.gz") if r["excluded"] == "0"]
        for r in rows:
            m_ref, organic, threshold = float(r["m_ref"]), float(r["m_organic"]), float(r["threshold"])
            if r["floor_binds"] == "1":
                self.assertTrue(math.isclose(threshold, organic - 1e-8 * e3_run.N, rel_tol=0, abs_tol=1e-9))
                self.assertLess(organic, (1 - e3_run.KAPPA) * m_ref)
            else:
                self.assertTrue(math.isclose(threshold, (1 - e3_run.KAPPA) * m_ref))
            if r["leader"]:
                self.assertGreaterEqual(float(r["slack"]), -1e-6)


if __name__ == "__main__":
    unittest.main()
