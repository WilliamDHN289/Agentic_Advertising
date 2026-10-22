"""SL-05 set design: coverage (AC-1), label rate (AC-2), design hash (AC-3), diagnostics."""
from __future__ import annotations

import itertools
import math
import unittest

from pydantic import ValidationError

from features.storelab.core.design import (BIBD_TABLE, Z_99, Design, blocks_for, build_design, chi2_sf, designs_hash,
                                           diagnose, label_rate_check, lambda_range, pair_counts, schonheim_bound)
from features.storelab.core.rng import stream

PILOT_SEED = 2026102101  # configs/e1_pilot.yaml


def ids(v: int) -> list[str]:
    return ["P%03d" % i for i in range(v)]


def pilot_designs(seed: int, randomisation: str = "per_set") -> list[Design]:
    """The E1 pilot's shape: 4 tasks, v = 10, K in {2, 4, 8}, r = 10, label rate 0.3."""
    return [build_design("task-%d" % t, ids(10), k, 10, 0.3, randomisation, seed) for t in range(4)
            for k in (2, 4, 8)]


def covered_pairs(design: Design) -> dict[tuple[str, str], int]:
    counts = {pair: 0 for pair in itertools.combinations(design.eligible_ids, 2)}
    for s in design.sets:
        for pair in itertools.combinations(s.product_ids, 2):
            counts[pair] += 1
    return counts


class ConstructionTest(unittest.TestCase):
    def test_every_table_entry_is_a_bibd(self):
        expected_lambda = {(10, 4): 2, (13, 4): 1, (16, 4): 1, (15, 8): 4, (16, 8): 7}
        self.assertEqual(set(BIBD_TABLE), set(expected_lambda))
        for (v, k), (_, blocks) in BIBD_TABLE.items():
            lam = expected_lambda[v, k]
            with self.subTest(v=v, k=k):
                self.assertTrue(all(len(set(b)) == k and set(b) <= set(range(v)) for b in blocks))
                self.assertEqual(len(set(blocks)), len(blocks))
                self.assertEqual(len(blocks), lam * v * (v - 1) // (k * (k - 1)))  # b = λ v (v-1) / (k (k-1))
                self.assertEqual(lambda_range(v, blocks), (lam, lam))

    def test_coverings_start_at_the_schonheim_bound(self):
        self.assertEqual([schonheim_bound(v, k) for v, k in ((10, 8), (12, 4), (30, 4), (30, 8))], [3, 12, 75, 19])
        design_type, _, blocks = blocks_for(10, 8, stream(1, "x"))
        self.assertEqual((design_type, len(blocks)), ("pair_covering", 3))
        self.assertEqual(blocks_for(10, 10, stream(1, "x"))[0], "complete")
        with self.assertRaises(ValueError):
            blocks_for(10, 11, stream(1, "x"))


class CoverageTest(unittest.TestCase):
    def test_every_pair_co_occurs_for_every_k_and_eligible_size(self):
        """AC-1: v = 10..30 (tickets.md: eligible sets of 10-30), K in {2, 4, 8}."""
        for v in range(10, 31):
            for k in (2, 4, 8):
                design = build_design("t", ids(v), k, 1, 0.3, "per_set", 5)
                counts = covered_pairs(design)
                with self.subTest(v=v, k=k, design_type=design.design_type):
                    self.assertGreaterEqual(min(counts.values()), 1)
                    self.assertEqual((design.lambda_min, design.lambda_max),
                                     (min(counts.values()), max(counts.values())))
                    if k == 2:
                        self.assertEqual((design.design_type, len(design.sets)), ("all_pairs", v * (v - 1) // 2))
                    elif (v, k) in BIBD_TABLE:
                        self.assertEqual((design.design_type, design.lambda_min), ("bibd", design.lambda_max))
                    else:
                        self.assertEqual(design.design_type, "pair_covering")
                        self.assertGreaterEqual(len(design.sets), schonheim_bound(v, k))


class PresentationTest(unittest.TestCase):
    def test_per_set_repeats_one_draw(self):
        design = build_design("t", ids(10), 4, 10, 0.3, "per_set", 5)
        self.assertEqual(len(design.presentations), len(design.sets))
        for s in design.sets:
            shown = {design.presentation_for(s.set_id, r).model_dump_json() for r in range(10)}
            self.assertEqual(len(shown), 1)
            p = design.presentation_for(s.set_id, 0)
            self.assertEqual(sorted(c.product_id for c in p.candidates), s.product_ids)
            self.assertEqual([c.position for c in p.candidates], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            design.presentation_for(design.sets[0].set_id, 10)

    def test_per_repetition_draws_each_repetition(self):
        design = build_design("t", ids(10), 8, 10, 0.3, "per_repetition", 5)
        self.assertEqual(len(design.presentations), 10 * len(design.sets))
        for s in design.sets:
            shown = [design.presentation_for(s.set_id, r) for r in range(10)]
            self.assertEqual([p.draw for p in shown], list(range(10)))
            self.assertGreater(len({tuple(c.product_id for c in p.candidates) for p in shown}), 1)

    def test_a_design_must_be_consistent(self):
        data = build_design("t", ids(10), 4, 10, 0.3, "per_set", 5).model_dump()
        data["presentations"] = data["presentations"][:-1]
        with self.assertRaises(ValidationError):
            Design.model_validate(data)


class LabelRateTest(unittest.TestCase):
    def test_the_pilot_label_rate_is_within_the_99_percent_bound(self):
        """AC-2, for the pilot design at the configured seed."""
        check = label_rate_check(pilot_designs(PILOT_SEED))
        self.assertEqual(check.draws, 4 * (45 * 2 + 15 * 4 + 3 * 8))  # one draw per product per set (per_set)
        self.assertAlmostEqual(check.bound, Z_99 * math.sqrt(0.21 / check.draws))
        self.assertTrue(check.within, check)

    def test_the_bound_holds_about_99_percent_of_the_time(self):
        """Labels are independent Bernoulli(0.3) draws: across seeds the rate leaves the
        99% bound rarely, and it is never forced inside."""
        checks = [label_rate_check(pilot_designs(seed)) for seed in range(200)]
        inside = sum(c.within for c in checks)
        self.assertGreaterEqual(inside, 194)
        rates = [c.rate for c in checks]
        self.assertAlmostEqual(sum(rates) / len(rates), 0.3, delta=0.01)
        self.assertGreater(max(rates) - min(rates), 0.03)


class DesignHashTest(unittest.TestCase):
    def test_same_seed_same_hash_different_seed_different_hash(self):
        """AC-3."""
        for k in (2, 4, 8):
            for randomisation in ("per_set", "per_repetition"):
                a = build_design("t", ids(12), k, 10, 0.3, randomisation, 41)
                with self.subTest(k=k, randomisation=randomisation):
                    self.assertEqual(build_design("t", list(reversed(ids(12))), k, 10, 0.3, randomisation,
                                                  41).design_hash(), a.design_hash())
                    others = {build_design("t", ids(12), k, 10, 0.3, randomisation, s).design_hash()
                              for s in range(42, 52)}
                    self.assertNotIn(a.design_hash(), others)
                    self.assertEqual(len(others), 10)
        self.assertEqual(designs_hash(pilot_designs(PILOT_SEED)), designs_hash(pilot_designs(PILOT_SEED)))
        self.assertNotEqual(designs_hash(pilot_designs(PILOT_SEED)), designs_hash(pilot_designs(PILOT_SEED + 1)))

    def test_the_hash_covers_every_parameter(self):
        base = build_design("t", ids(10), 4, 10, 0.3, "per_set", 5).design_hash()
        for other in (build_design("u", ids(10), 4, 10, 0.3, "per_set", 5),
                      build_design("t", ids(10), 4, 9, 0.3, "per_set", 5),
                      build_design("t", ids(10), 4, 10, 0.31, "per_set", 5),
                      build_design("t", ids(11), 4, 10, 0.3, "per_set", 5)):
            self.assertNotEqual(other.design_hash(), base)


class DiagnosticsTest(unittest.TestCase):
    def test_counts_match_the_design(self):
        design = build_design("t", ids(13), 4, 10, 0.3, "per_repetition", 9)
        diag = diagnose(design)
        index = {pid: i for i, pid in enumerate(design.eligible_ids)}
        expected = pair_counts(13, [tuple(index[p] for p in s.product_ids) for s in design.sets])
        self.assertEqual(diag.cooccurrence, expected.tolist())
        self.assertEqual((diag.lambda_min, diag.lambda_max), (1, 1))
        for b in diag.products:
            self.assertEqual(b.sets, 4)  # (13,4,1): every point is in r = 4 blocks
            self.assertEqual(sum(b.position_counts), b.draws)
            self.assertEqual(b.draws, 4 * 10)
            self.assertTrue(0.0 <= b.p_value <= 1.0)
        self.assertEqual(diag.label_draws, sum(b.draws for b in diag.products))
        self.assertEqual(diag.labelled, sum(b.labelled for b in diag.products))

    def test_chi2_survival_function(self):
        for x in (0.0, 0.1, 1.0, 3.841458820694124, 10.0, 40.0):
            with self.subTest(x=x):
                self.assertAlmostEqual(chi2_sf(x, 1), math.erfc(math.sqrt(x / 2)), places=12)
                self.assertAlmostEqual(chi2_sf(x, 2), math.exp(-x / 2), places=12)
                self.assertAlmostEqual(chi2_sf(x, 4), math.exp(-x / 2) * (1 + x / 2), places=12)
        self.assertAlmostEqual(chi2_sf(11.070497693516351, 5), 0.05, places=10)
        self.assertAlmostEqual(chi2_sf(14.067140449340169, 7), 0.05, places=10)
        with self.assertRaises(ValueError):
            chi2_sf(-1, 3)


if __name__ == "__main__":
    unittest.main()
