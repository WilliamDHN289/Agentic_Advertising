"""SL-21S: E4 learning on A17's stipulated state variance.

- AC-1: reachability and identification on hand-built graphs (against every consistent
  ranking), and the corrected Cor design(b) bound with its count checked by enumeration;
- AC-2: the committed H4b implementation check passes in every checked cell;
- AC-3: every cell has R = 200 completed replications; committed rows regenerate and a seeded
  rerun was byte for byte equal;
- AC-4: the label on every table and figure;
- AC-5: H4a–H4d and the recovery analysis are reported and labelled exploratory.

Also: the designs, the estimators and the plug-in mechanism against direct computations.
"""
from __future__ import annotations

import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from features.storelab.analysis import e2_run, e4, e4_report, e4_run
from features.storelab.analysis.figures import FIGURES, LABEL

A, B, C, D = 1, 2, 3, 4


def all_subsets(n: int, k: int = 4):
    return [s for size in range(1, k + 1) for s in itertools.combinations(range(1, n + 1), size)]


class GraphTest(unittest.TestCase):
    """AC-1: Prop identify on hand-built audited families, against every consistent ranking."""

    GRAPHS = {
        # Example inclusion: B always appears with A, so A beats B and nothing is learnt about B against C
        "inclusion": ([((A, B), A), ((A, B, C), A)], 3, {(B, C): None, (A, C): A, (A, B, C): A}),
        # a chain A > B > C: {A, C} is identified through B
        "chain": ([((A, B), A), ((B, C), B)], 3, {(A, C): A, (A, B, C): A, (C,): None}),
        # the outside option: 0 beats A, B beats 0, so B is chosen from {A, B}
        "outside": ([((A,), 0), ((B,), B)], 2, {(A, B): B, (A,): 0, (B,): B}),
        # two components: nothing links A and C
        "disconnected": ([((A, B), A), ((C, D), C)], 4, {(A, C): None, (A, B): A, (C, D): C, (B, D): None}),
        # every pair compared: a full tournament identifies every set
        "tournament": ([((A, B), A), ((A, C), A), ((A, D), A), ((B, C), B), ((B, D), B), ((C, D), C)], 4,
                       {s: min(s) for s in itertools.chain.from_iterable(itertools.combinations((A, B, C, D), k)
                                                                         for k in (2, 3, 4))}),
        # leader-inclusive: A in every set and always chosen (Cor design(a)): sets without A are not identified
        "leader_inclusive": ([((A, B, C), A), ((A, C, D), A), ((A, B, D), A)], 4,
                             {(B, C): None, (C, D): None, (B, C, D): None, (A, B): A, (A, D): A}),
    }

    def test_hand_built_graphs(self):
        self.assertGreaterEqual(len(self.GRAPHS), 5)
        for name, (observations, n, expected) in self.GRAPHS.items():
            reach = e4.closure(e4.comparison_graph(observations, n + 1))
            for target in all_subsets(n):
                with self.subTest(graph=name, target=target):
                    truth = e4.identified_by_enumeration(observations, n + 1, target)
                    self.assertEqual(e4.identified(reach, target), truth)
                    winners = e4.possible_winners(reach, target)
                    self.assertEqual(winners[0] if len(winners) == 1 else None, truth)
                    got = e4.identified_shares(reach, np.array([target]))[0]
                    self.assertEqual(None if got == -1 else int(got), truth)
                    if target in expected:
                        self.assertEqual(truth, expected[target])

    def test_random_families_against_enumeration(self):
        rng = np.random.default_rng(11)
        for _ in range(40):
            n = int(rng.integers(3, 6))
            ranking = tuple(int(x) for x in rng.permutation(n + 1))
            family = [tuple(sorted(rng.choice(np.arange(1, n + 1), int(rng.integers(1, 4)), replace=False)))
                      for _ in range(int(rng.integers(1, 6)))]
            observations = [(s, e4.choice(ranking, s)) for s in family]
            reach = e4.closure(e4.comparison_graph(observations, n + 1))
            for target in all_subsets(n):
                truth = e4.identified_by_enumeration(observations, n + 1, target)
                self.assertEqual(e4.identified(reach, target), truth)
                if truth is not None:
                    self.assertEqual(truth, e4.choice(ranking, target))


class BoundTest(unittest.TestCase):
    """AC-1: the corrected bound ⌈log ⌊e · N!⌋ / log(K + 1)⌉ (A1 §E1)."""

    def test_count_and_bound(self):
        for n in range(1, 13):
            self.assertEqual(e4.choice_functions(n), math.floor(math.e * math.factorial(n)))
        for n in range(2, 6):
            for k in range(2, min(4, n) + 1):
                self.assertEqual(e4.count_choice_functions(n, k), e4.choice_functions(n))
        for n, k in ((5, 4), (10, 4), (30, 4), (7, 2)):
            m = e4.design_lower_bound(n, k)
            self.assertTrue((k + 1) ** m >= e4.choice_functions(n) > (k + 1) ** (m - 1))
            self.assertEqual(m, math.ceil(math.log(e4.choice_functions(n)) / math.log(k + 1)))
        self.assertEqual((e4.design_lower_bound(5, 4), e4.paper_lower_bound(5, 4)), (4, 5))  # A1 §E1's example

    def test_the_committed_enumeration_covers_n_up_to_7(self):
        rows = e2_run.read_rows(e4_run.RESULTS / "bound.csv.gz")
        checked = {(int(r["n"]), int(r["k"])) for r in rows if r["enumerated"]}
        self.assertEqual(checked, {(n, k) for n in range(2, 8) for k in range(2, min(4, n) + 1)})
        for r in rows:
            if r["enumerated"]:
                self.assertEqual(r["enumerated"], r["choice_functions"])


class DesignTest(unittest.TestCase):
    def test_designs(self):
        for n_x in e4_run.N_XS:
            mk = e4_run.market(n_x, 2)
            ds, iota, means = e4_run.designs(mk, n_x, 1.0, 2)
            leader = int(np.argmax(means)) + 1
            top = sorted(int(j) + 1 for j in np.argsort(-means)[:5])
            for name, sets in ds.items():
                self.assertEqual(sets.shape, (2 * n_x, e4_run.K))
                self.assertTrue(all(len(set(s)) == e4_run.K for s in sets), name)
            self.assertTrue(all(leader in s for s in ds["leader_inclusive"]))
            for pair in itertools.combinations(top, 2):  # every top pair meets with non-top members only
                self.assertTrue(any(set(pair) <= set(s) and len(set(s) & set(top)) == 2 for s in ds["competitor_varying"]))
            np.testing.assert_array_equal(e4.inclusion(ds["random"], n_x), e4.inclusion(ds["competitor_varying"], n_x))
            self.assertFalse(np.array_equal(ds["random"], ds["competitor_varying"]))
            np.testing.assert_allclose(iota.sum(axis=1), 1.0)
            self.assertTrue(np.all((iota > 0).sum(axis=1) == e4_run.MENU_CELLS))
            self.assertEqual(len(e4_run.targets(means)), 25)


class EstimatorTest(unittest.TestCase):
    def setUp(self):
        self.mk = e4_run.market(10, 4)
        ds, self.iota, _ = e4_run.designs(self.mk, 10, 2.0, 4)
        self.sets = ds["competitor_varying"]
        self.q = e4.logit(e4_run.scores(self.mk, 2.0), self.sets, e4_run.SIGMA, e4_run.OUTSIDE_SCORE)
        self.m = e4_run._mechanism(self.mk, self.sets, self.iota, 10)

    def test_logit_and_closed_form(self):
        np.testing.assert_allclose(self.q.sum(axis=2), 1.0)
        v = e4_run.scores(self.mk, 2.0)[1, self.sets[0] - 1]
        e = np.exp(np.r_[e4_run.OUTSIDE_SCORE, v] / e4_run.SIGMA)
        np.testing.assert_allclose(self.q[1, 0], e / e.sum())
        q = self.q[:, 0, 1]
        var = e4.variance_closed_form(q, e4_run.STATE_PROBS, 7, 3)
        mean = q.mean()
        self.assertAlmostEqual(var, (np.mean((q - mean) ** 2) + np.mean(q * (1 - q)) / 3) / 7)

    def test_estimates_are_unbiased_and_prefixes_agree(self):
        rng = np.random.default_rng(3)
        n0, h, reps = 3, 4, 4000
        truth = e4.true_values(self.m, self.q, e4_run.STATE_PROBS)
        x = np.zeros_like(truth["X"])
        cost = np.zeros_like(truth["cost"])
        for _ in range(reps):
            y = e4.audit(rng, self.q, rng.integers(0, 2, h), n0)
            est = e4.estimates(self.m, y, n0)
            x += est["X"] / reps
            cost += est["cost"] / reps
        np.testing.assert_allclose(x, truth["X"], atol=6.0)  # X totals up to 1000; MC error a few units
        np.testing.assert_allclose(cost, truth["cost"], rtol=0.03)
        y = e4.audit(rng, self.q, rng.integers(0, 2, 12), n0)
        prefix = e4.prefix_welfare(self.m, y, n0, self.mk["theta"])
        for e in (1, 5, 12):
            np.testing.assert_allclose(prefix[e - 1], e4.welfare(e4.estimates(self.m, y[:e], n0), self.m.advertisers,
                                                                  self.mk["theta"]), atol=1e-9)

    def test_envelope_and_truthfulness(self):
        rng = np.random.default_rng(9)
        for _ in range(300):
            slopes, intercepts = rng.normal(size=16) * 50, rng.normal(size=16) * 20
            pieces = e4.envelope(slopes, intercepts, 2.0)
            self.assertEqual(pieces[0][0], 0.0)
            self.assertEqual(pieces[-1][1], 2.0)
            for z in np.linspace(0.001, 1.999, 50):
                k = next(p[2] for p in pieces if p[0] <= z <= p[1])
                self.assertAlmostEqual(intercepts[k] + z * slopes[k], (intercepts + z * slopes).max(), places=9)
        truth = e4.true_values(self.m, self.q, e4_run.STATE_PROBS)
        gains = e4.misreport_gains(self.m, truth, truth, self.mk["theta"], e4_run.REPORTS)
        np.testing.assert_allclose(gains, 0.0, atol=1e-9)  # exact estimates: truthful reporting is optimal


def summary() -> dict:
    return json.loads(e4_report.SUMMARY.read_text(encoding="utf-8"))


class ResultsTest(unittest.TestCase):
    def test_the_checks_pass_and_every_cell_is_complete(self):
        g = e4_report.gates(e4_report.load())
        self.assertTrue(g["passed"])
        self.assertTrue(g["bound"]["passed"] and g["h4b_check"]["passed"] and g["h4a_icheck"]["passed"])
        self.assertTrue(g["complete"])
        self.assertEqual(len(g["h4b_check"]["cells"]), 8)
        for cell in g["h4b_check"]["cells"]:
            self.assertLessEqual(cell["bias_in_se"], 3.0)
            self.assertLessEqual(abs(cell["variance_ratio"] - 1), 0.05)
            self.assertEqual(cell["replications"], 10_000)

    def test_a_seeded_rerun_reproduces_the_committed_rows(self):
        for kind, function, fields, cell in (("learning", e4_run.evaluate_learning, e4_run.LEARNING_FIELDS, (30, 0.5, 100)),
                                             ("recovery", e4_run.evaluate_recovery, e4_run.RECOVERY_FIELDS, (10, 2.0, 2))):
            rows = [r for i in range(2) for r in function((*cell, i))]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "x.csv.gz"
                e2_run.write_rows(path, fields, rows)
                fresh = e2_run.read_rows(path)
            committed = e2_run.read_rows(e2_run.result_path(e4_run.RESULTS, e4_run.cell_id(kind, *cell)))
            self.assertEqual(fresh, committed[:len(fresh)])
        rows = e4_run.evaluate_identification((10, 0.5, 0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv.gz"
            e2_run.write_rows(path, e4_run.IDENTIFICATION_FIELDS, rows)
            fresh = e2_run.read_rows(path)
        committed = [r for r in e2_run.read_rows(e4_run.RESULTS / "identification.csv.gz")
                     if r["n_x"] == "10" and r["f"] == "0.5" and r["replication"] == "0"]
        self.assertEqual(fresh, committed)
        self.assertTrue(summary()["rerun_identical"])

    def test_tables_and_figures_carry_the_label(self):
        report = e4_report.REPORT.read_text(encoding="utf-8")
        self.assertIn(LABEL, report)
        self.assertIn("not been independently reviewed", report)
        self.assertNotRegex(report, r"\b(Supported|Not supported)\b")
        for name in e4_report.FIGURE_NAMES:
            for suffix in (".pdf", ".png", ".csv"):
                self.assertTrue((FIGURES / "sl21s" / (name + suffix)).exists(), name + suffix)
            self.assertTrue((FIGURES / "sl21s" / (name + ".csv")).read_text(encoding="utf-8").startswith("# " + LABEL))
        for name in ("h4a.csv", "h4b.csv", "h4c.csv", "h4d.csv", "recovery.csv"):
            self.assertTrue((e4_report.TABLES / name).read_text(encoding="utf-8").startswith("# " + LABEL))

    def test_the_registered_statistics_are_reported_and_exploratory(self):
        s = summary()["statistics"]
        for key in ("H4a", "H4b", "H4c", "H4d", "recovery"):
            self.assertEqual(s[key]["label"], "exploratory")
        self.assertNotIn("decision", json.dumps(s))
        self.assertEqual(len(s["H4b"]["cells"]), 32)
        self.assertEqual(len(s["H4c"]["cells"]), 32)
        for v in s["H4c"]["cells"].values():
            for x in v["by_h"]:
                self.assertEqual(x["pairs"], 200 * 4)
        report = e4_report.REPORT.read_text(encoding="utf-8")
        self.assertIn("Recovery after a version change (exploratory)", report)


if __name__ == "__main__":
    unittest.main()
