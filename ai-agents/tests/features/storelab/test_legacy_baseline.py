"""SL-00 acceptance tests: the original StoreLab frozen as a hash-identified baseline.

Standard library only and read-only, so they run unchanged under the ai-agents
interpreter (3.11) and the original one (3.9.13):

    cd ai-agents && python -B -m unittest discover -s tests/features/storelab -t . -v

Comparisons with the original directory (STORELAB_LEGACY_SOURCE, default
~/Desktop/agentic_amazon/5_storelab) are skipped when it is absent.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from features.storelab import legacy_baseline as lb

PROVENANCE = lb.LEGACY_DIR / "PROVENANCE.md"
DEFECTS = lb.LEGACY_DIR / "DEFECTS.md"
SOURCE = lb.default_source()
HAVE_SOURCE = all((SOURCE / name).is_file() for name in lb.VENDORED_FILES)
NO_SOURCE = "original StoreLab directory not found at %s" % SOURCE
REF = re.compile(r"`([A-Za-z_]+\.py):(\d+)(?:-(\d+))?`")
# sha256 of the sorted-key JSON of every field of every SKU; identical under 3.9.13 and 3.11.
CATALOG_SHA256 = "9491773fa9aa2601d95f6396b39a08c6887f77a7bd053dff9e756b91872bbe07"


def table(path, heading):
    """Body rows of the first Markdown table after ``heading``, as lists of cells."""
    rows = []
    for line in path.read_text(encoding="utf-8").split(heading, 1)[1].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not set("".join(cells)) <= set("-: "):
            rows.append(cells)
    return rows[1:]


def unquote(cell):
    return cell.strip("`")


class VendoredCodeTest(unittest.TestCase):
    """AC-1: sha256 of every vendored file equals that of its source file."""

    def test_directory_holds_exactly_the_vendored_code(self):
        self.assertEqual(lb.check_vendored(), [])

    def test_vendored_hashes_match_provenance(self):
        recorded = {unquote(r[0]): unquote(r[1]) for r in table(PROVENANCE, "## Files") if r[4] == "vendored"}
        self.assertEqual(sorted(recorded), sorted(lb.VENDORED_FILES))
        for name, digest in recorded.items():
            self.assertEqual(lb.sha256_file(lb.LEGACY_DIR / name), digest, name)

    @unittest.skipUnless(HAVE_SOURCE, NO_SOURCE)
    def test_vendored_hashes_match_source(self):
        self.assertEqual(lb.check_vendored(SOURCE), [])

    @unittest.skipUnless(HAVE_SOURCE, NO_SOURCE)
    def test_provenance_lists_every_source_file_with_its_hash_and_size(self):
        rows = table(PROVENANCE, "## Files")
        for path, digest, size, _mtime, _disposition in rows:
            source_file = SOURCE / unquote(path)
            self.assertEqual((lb.sha256_file(source_file), source_file.stat().st_size),
                             (unquote(digest), int(size)), path)
        on_disk = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*")
                   if p.is_file() and p.name != ".DS_Store"}
        self.assertEqual({unquote(r[0]) for r in rows}, on_disk)


class SnapshotTest(unittest.TestCase):
    """The saved run data, stored in the artifact store under its content address."""

    def test_snapshot_matches_its_manifest_and_address(self):
        self.assertEqual(lb.check_snapshot(), [])

    def test_provenance_records_the_stored_hashes(self):
        manifest = json.loads((lb.snapshot_dir() / "MANIFEST.json").read_text(encoding="utf-8"))
        stored = {unquote(r[0]): unquote(r[1]) for r in table(PROVENANCE, "## Files") if r[4] == "stored"}
        self.assertEqual(stored, {e["path"]: e["sha256"] for e in manifest["files"]})

    @unittest.skipUnless(HAVE_SOURCE, NO_SOURCE)
    def test_snapshot_matches_source(self):
        self.assertEqual(lb.check_snapshot(source=SOURCE), [])

    def test_tampered_or_extra_files_are_reported(self):
        try:
            tmp = tempfile.TemporaryDirectory()
        except OSError as exc:
            self.skipTest("no writable temporary directory: %s" % exc)
        with tmp:
            root = Path(tmp.name)
            shutil.copytree(lb.snapshot_dir(), lb.snapshot_dir(root))
            self.assertEqual(lb.check_snapshot(root), [])
            with open(lb.snapshot_dir(root) / "logs/run.log", "ab") as handle:
                handle.write(b"\n")
            (lb.snapshot_dir(root) / "results/extra.json").write_text("{}")
            problems = lb.check_snapshot(root)
        self.assertTrue(any(p.startswith("logs/run.log: stored bytes") for p in problems), problems)
        self.assertTrue(any("results/extra.json" in p for p in problems), problems)


class ReproductionTest(unittest.TestCase):
    """AC-2: analyze.py on the saved data regenerates findings.md with zero diff."""

    def setUp(self):
        self.expected = (lb.snapshot_dir() / lb.FINDINGS).read_bytes()

    def test_in_memory_run_has_zero_diff(self):
        actual = lb.regenerate_findings(sys.executable, "in-memory")
        self.assertEqual(lb.findings_diff(self.expected, actual), [])

    def test_run_in_the_original_layout_has_zero_diff(self):
        try:
            tempfile.TemporaryDirectory().cleanup()
        except OSError as exc:
            self.skipTest("no writable temporary directory: %s" % exc)
        actual = lb.regenerate_findings(sys.executable, "copy")
        self.assertEqual(lb.findings_diff(self.expected, actual), [])

    def test_diff_detects_a_single_changed_byte(self):
        changed = self.expected.replace(b"18.89%", b"18.88%", 1)
        self.assertNotEqual(changed, self.expected)
        self.assertNotEqual(lb.findings_diff(self.expected, changed), [])


class CatalogTest(unittest.TestCase):
    """AC-3: build_catalog() yields 2,004 SKUs, 712 of them ad-eligible."""

    def test_build_catalog_counts(self):
        summary = lb.catalog_summary(sys.executable)
        self.assertEqual((summary["skus"], summary["categories"], summary["ad_eligible"]), (2004, 6, 712))
        self.assertEqual(summary["catalog_sha256"], CATALOG_SHA256)

    def test_counts_agree_with_the_original_run_log(self):
        self.assertEqual(lb.run_log_catalog(), {"skus": 2004, "categories": 6, "ad_eligible": 712})


class DefectListTest(unittest.TestCase):
    """AC-4: every defect has a file:line reference and a target ticket."""

    defects = table(DEFECTS, "## Defect list")
    checks = table(DEFECTS, "## Location check")

    def cited(self):
        return {m.group(0): row for row in self.defects for m in REF.finditer(row[2])}

    def test_every_defect_has_a_location_and_a_target_ticket(self):
        self.assertGreaterEqual(len(self.defects), 5)
        for ident, _defect, locations, target in self.defects:
            self.assertRegex(locations, REF, ident + " has no file:line reference")
            self.assertRegex(target, r"\bSL-\d{2}\b", ident + " has no target ticket")

    def test_every_cited_location_is_checked(self):
        self.assertEqual(set(self.cited()), {row[0] for row in self.checks})

    def test_locations_point_at_the_quoted_code(self):
        for location, code in self.checks:
            name, first, last = REF.fullmatch(location).groups()
            lines = (lb.LEGACY_DIR / name).read_text(encoding="utf-8").splitlines()
            first, last = int(first), int(last or first)
            self.assertLessEqual(last, len(lines), location)
            self.assertIn(unquote(code), "\n".join(lines[first - 1:last]), location)

    def test_defects_named_in_the_ticket_keep_their_targets(self):
        # tickets.md SL-00, "Known defects to record". The ticket cites storelab.py l.271
        # for the IVT stream; the RNG is created on l.272.
        expected = {
            "storelab.py:123": ["SL-04"], "storelab.py:131": ["SL-04"],
            "storelab.py:272": ["SL-04"], "storelab.py:325": ["SL-04"],
            "run_experiment.py:165-170": ["SL-04"],
            "agents.py:61-73": ["SL-03", "SL-07"], "agents.py:212-214": ["SL-03", "SL-07"],
            "agents.py:264-266": ["SL-03", "SL-07"], "agents.py:307-309": ["SL-03", "SL-07"],
            "run_experiment.py:228-236": ["SL-02"],
            "agents.py:44": ["SL-02"], "agents.py:56": ["SL-02"],
            "agents.py:150-152": ["legacy only"],
        }
        cited = self.cited()
        for location, targets in expected.items():
            row = cited.get("`%s`" % location)
            self.assertIsNotNone(row, location + " is not in the defect list")
            for target in targets:
                self.assertIn(target, row[3], location)


if __name__ == "__main__":
    unittest.main()
