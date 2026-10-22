"""SL-05 end to end: E1 stimuli from a fixture snapshot, files and CLI.

Writes only to temporary directories.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from features.storelab import cli, e1_stimuli
from features.storelab.config_files import read_config_data
from features.storelab.core.config import apply_overrides, resolve_config
from tests.features.storelab import e1_fixture
from tests.features.storelab.test_catalog_amazon import fixture_snapshot

CONFIG = Path(cli.__file__).resolve().parent / "configs" / "e1_pilot.yaml"


class E1StimuliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.snapshot = fixture_snapshot(cls.root)
        cls.overrides = {"environment.dataset": {"name": "amazon_reviews_2023", "sha256": cls.snapshot.name}}
        cls.config = resolve_config(apply_overrides(read_config_data(CONFIG), cls.overrides))
        cls.built = e1_stimuli.build(cls.config, cls.snapshot)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_tasks_designs_and_checks(self):
        built, cfg = self.built, self.config
        self.assertEqual([t.task.leaf_path[-1] for t in built.tasks], ["Mugs", "Kettles", "Headphones", "Chargers"])  # config order
        self.assertEqual(len(built.designs), 4 * len(cfg.design.set_sizes))
        for t in built.tasks:  # AC-5
            expected = e1_fixture.expected(t.task.category, t.task.leaf_path[-1])
            self.assertEqual({k: getattr(t, k) for k in expected}, expected)
            self.assertEqual(sorted(t.utility), t.eligible_ids)
            self.assertEqual(len(t.eligible_ids), cfg.design.eligible_size)
        for d in built.designs:  # AC-1
            pairs = {p: 0 for p in itertools.combinations(d.eligible_ids, 2)}
            for s in d.sets:
                for p in itertools.combinations(s.product_ids, 2):
                    pairs[p] += 1
            self.assertGreaterEqual(min(pairs.values()), 1)
            self.assertEqual((d.lambda_min, d.lambda_max), (min(pairs.values()), max(pairs.values())))
        self.assertEqual([d.design_type for d in built.designs[:3]], ["all_pairs", "bibd", "pair_covering"])
        self.assertTrue(built.label_rate.within)  # AC-2
        self.assertEqual(set(built.products), {pid for t in built.tasks for pid in t.eligible_ids})
        self.assertTrue(all(len(p.description) <= cfg.environment.description_chars for p in built.products.values()))

    def test_outputs_are_reproducible_and_seed_dependent(self):
        out_a, out_b = self.root / "a", self.root / "b"
        manifest = e1_stimuli.write(self.built, out_a)
        e1_stimuli.write(e1_stimuli.build(self.config, self.snapshot), out_b)
        names = ["MANIFEST.json", "design_report.md", "designs.json", "diagnostics.json", "screening.json",
                 "tasks.json"]
        self.assertEqual(sorted(p.name for p in out_a.iterdir()), names)
        for name in names[1:]:
            self.assertEqual((out_a / name).read_bytes(), (out_b / name).read_bytes(), name)
        other = resolve_config(apply_overrides(read_config_data(CONFIG), {**self.overrides, "seed": 7}))
        rebuilt = e1_stimuli.build(other, self.snapshot)
        self.assertNotEqual(e1_stimuli.write(rebuilt, self.root / "c")["designs_hash"], manifest["designs_hash"])
        self.assertEqual(manifest["snapshot_sha256"], self.snapshot.name)
        self.assertEqual(manifest["templates"].keys(), {"e1_text@1", "e1_structured@1", "e1_browser_tool@1"})
        report = (out_a / "design_report.md").read_text(encoding="utf-8")
        for text in ("## Task `electronics-1`", "λ_min = 1", "Pair co-occurrence", "position counts", "99% bound",
                     "u defined for 10 of 10 eligible products"):
            self.assertIn(text, report)
        designs = json.loads((out_a / "designs.json").read_text(encoding="utf-8"))
        self.assertEqual(designs["designs_hash"], manifest["designs_hash"])
        self.assertEqual([d["design_hash"] for d in designs["designs"]],
                         [d.design_hash() for d in self.built.designs])

    def test_the_snapshot_must_be_the_configured_one(self):
        other = resolve_config(apply_overrides(read_config_data(CONFIG), {
            "environment.dataset": {"name": "amazon_reviews_2023", "sha256": "e" * 64}}))
        with self.assertRaisesRegex(ValueError, "is not environment.dataset"):
            e1_stimuli.build(other, self.snapshot)
        with self.assertRaisesRegex(ValueError, "must name"):
            e1_stimuli.build(resolve_config(read_config_data(CONFIG)), self.snapshot)

    def test_cli(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        dataset = "environment.dataset={name: amazon_reviews_2023, sha256: %s}" % self.snapshot.name
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(["e1-stimuli", str(CONFIG), "--snapshot", str(self.snapshot), "--out",
                             str(self.root / "cli"), "--set", dataset])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["label_rate_within_bound"], True)
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(cli.main(["e1-stimuli", str(CONFIG), "--snapshot", str(self.snapshot), "--out",
                                       str(self.root / "cli2")]), 2)


if __name__ == "__main__":
    unittest.main()
