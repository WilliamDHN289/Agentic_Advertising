"""SL-01 claims register and pre-registration: the register validates against its
schema with every field of H1a–H5b filled in (AC-1); decide.py labels a synthetic
results file for every claim without manual input (AC-2); the direction audit covers
every acceptance criterion of tickets.md (AC-3); the E1 registration timing check
from run manifests (AC-4, machinery).
"""
from __future__ import annotations

import contextlib
import copy
import datetime
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import jsonschema
from features.storelab import cli
from features.storelab.analysis import decide as decide_module
from features.storelab.analysis.decide import (ResultsFile, binom_cdf, binom_sf, count_p_values, decide,
                                               decide_file, holm)
from features.storelab.analysis.register import (CLAIM_IDS, REGISTER_PATH, REPO_ROOT, BoundTest, Register,
                                                 claim_section, load_register, read_register_data,
                                                 register_sha256)
from features.storelab.analysis.registration import Receipts, check_e1, load_receipts
from features.storelab.analysis import registration
from features.storelab.core.manifest import append_status
from pydantic import ValidationError

DOCS = REGISTER_PATH.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "analysis" / "synthetic_results.json"
SCHEMA = cli.CONTRACTS / "claims_register.schema.json"


def required_paths(schema: dict, node: dict, data, path: str = ""):
    """Every required field of ``data`` under ``node`` (resolving $ref), with its value."""
    if "$ref" in node:
        node = schema["$defs"][node["$ref"].split("/")[-1]]
    if "discriminator" in node or "oneOf" in node or "anyOf" in node:
        for option in node.get("oneOf", node.get("anyOf", [])):
            target = schema["$defs"][option["$ref"].split("/")[-1]] if "$ref" in option else option
            kind = target.get("properties", {}).get("kind", {}).get("const")
            if kind is not None and isinstance(data, dict) and data.get("kind") == kind:
                yield from required_paths(schema, target, data, path)
        return
    if isinstance(data, dict) and "properties" in node:
        for name in node.get("required", []):
            yield path + "." + name, data.get(name)
        for name, sub in node["properties"].items():
            if name in data:
                yield from required_paths(schema, sub, data[name], path + "." + name)
    elif isinstance(data, list) and "items" in node:
        for i, item in enumerate(data):
            yield from required_paths(schema, node["items"], item, "%s[%d]" % (path, i))


def blank(value) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and not (value.strip() if isinstance(value, str)
                                                                            else value))


# ----------------------------------------------------------------------- AC-1
class RegisterSchemaTest(unittest.TestCase):
    """AC-1: the register validates against its schema; every claim H1a–H5b has all
    required fields filled in."""

    def setUp(self):
        self.data = read_register_data()
        self.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_the_register_validates_against_the_published_schema(self):
        jsonschema.Draft202012Validator.check_schema(self.schema)
        jsonschema.Draft202012Validator(self.schema).validate(self.data)
        register = load_register()
        self.assertEqual([c.id for c in register.claims], list(CLAIM_IDS))
        self.assertEqual(json.loads(cli.schema_text("claims_register.schema.json")), self.schema)

    def test_every_claim_has_every_required_field_filled_in(self):
        claim_schema = self.schema["$defs"]["Claim"]
        top = set(claim_schema["required"])
        for field in ("id", "sources", "statement", "experiment", "unit_of_analysis", "metric", "estimator", "test",
                      "alpha", "ci_level", "multiplicity_family", "tolerance_or_margin", "min_n", "decision_rule",
                      "decision", "interpretation", "status"):
            self.assertIn(field, top)  # the ticket's per-claim fields are all required by the schema
        for claim in self.data["claims"]:
            with self.subTest(claim=claim["id"]):
                for path, value in required_paths(self.schema, claim_schema, claim, claim["id"]):
                    self.assertFalse(blank(value) and not path.endswith(".amendments"), path)
                for outcome in ("supported", "not_supported", "inconclusive"):
                    self.assertTrue(claim["interpretation"][outcome].strip())

    def test_the_schema_rejects_a_blank_or_missing_field(self):
        validator = jsonschema.Draft202012Validator(self.schema)
        for mutate in (lambda c: c.pop("decision_rule"), lambda c: c.update(metric=""),
                       lambda c: c["interpretation"].pop("inconclusive"), lambda c: c.update(extra="x"),
                       lambda c: c["decision"]["groups"][0]["tests"][0].update(kind="vote")):
            data = copy.deepcopy(self.data)
            mutate(data["claims"][0])
            with self.subTest(mutation=mutate):
                self.assertFalse(validator.is_valid(data))
                with self.assertRaises(ValidationError):
                    Register.model_validate(data)
        data = copy.deepcopy(self.data)
        data["claims"].pop()  # H5b missing
        with self.assertRaisesRegex(ValidationError, "exactly once"):
            Register.model_validate(data)

    def test_sources_and_documents_cite_the_current_files(self):
        register = load_register()
        for source in register.sources:
            with self.subTest(source=source.path):
                self.assertEqual(hashlib.sha256((REPO_ROOT / source.path).read_bytes()).hexdigest(), source.sha256)
        digests = {"features/storelab/docs/claims_register.yaml": register_sha256(),
                   **{s.path: s.sha256 for s in register.sources}}
        for name, plan in register.registrations.items():
            text = (DOCS / Path(plan.document).name).read_text(encoding="utf-8")
            with self.subTest(document=plan.document):
                for path, digest in digests.items():
                    self.assertIn("| `%s` | `%s` |" % (path, digest), text)
                for claim_id in plan.covers:  # each claim's section is the register's, verbatim
                    self.assertIn(claim_section(register.claim(claim_id)), text)

    def test_shared_definitions_and_amendments(self):
        register = load_register()
        shared = register.shared_definitions
        self.assertEqual(shared["ground_truth_utility"]["id"], "shrunk_rating@1")
        self.assertEqual(shared["thresholds"]["H1b_modal_share"], 0.80)
        self.assertEqual(shared["thresholds"]["missingness"], [0.10, 0.25])
        self.assertEqual(set(shared["effective_choice"]["per_interface"]), {"text", "structured", "browser_tool",
                                                                              "field_E5"})
        self.assertIn("A1 §E2", " ".join(register.claim("H4d").amendments))  # upper bound, not sqrt(h)
        self.assertIn("A1 §E3", " ".join(register.claim("H4b").amendments))
        self.assertIn("A1 §E5", " ".join(register.claim("H2a").amendments))
        self.assertIn("A1 §E1", " ".join(register.claim("H4a").amendments))
        gap = next(t for t in register.claim("H1b").decision.groups[0].tests if t.name == "score_gap")
        self.assertAlmostEqual(gap.threshold, 1.3862943611198906)
        self.assertEqual({c.registration for c in register.claims if c.experiment == "E5"}, {"E5"})


# ----------------------------------------------------------------------- AC-2
# Hand-derived labels of the synthetic results (fixtures/analysis/synthetic_results.json).
EXPECTED = {
    "H1a": "supported",  # 2 cells in the family (one below the floor): Holm 0.002, 0.004 <= 0.05; 2 of 3 cells
    "H1b": "inconclusive",  # one cell supported, one flagged (15% missing) whose sensitivity disagrees, one excluded
    "H1c": "supported",  # sign: 2 of 3 supported (one not supported, 1/3 is not > 1/3); heterogeneity p 0.01
    "H1d": "not_supported",  # credible 10/25 supported; 0 raw of 40: P(X <= 0) = 0.9^40, Holm 0.030 < 0.05
    "H1e": "inconclusive",  # 60/250 supported; 30/220 Wilson p 0.0375 > 0.025
    "H2a": "supported",
    "H2b": "supported",
    "H2c": "inconclusive",  # M2 intercept neither equivalent nor outside the margin
    "H3a": "supported",  # identity passes; exposure 2 of 3 categories
    "H3b": "inconclusive",  # 98% of grid points pass: implementation error
    "H3c": "supported",  # 12 buy-backs of 300: CP lower bound > 0; M4 0.97 >= 0.95
    "H4a": "supported",
    "H4b": "not_supported",  # slope interval entirely outside [-0.55, -0.45]
    "H4c": "supported",  # 5 of 200 violations: CP upper bound <= 0.15
    "H4d": "inconclusive",
    "H5a": "supported",
    "H5b": "inconclusive",  # one arm supported, one inconclusive (aggregation all)
}


class DecideTest(unittest.TestCase):
    """AC-2: decide.py maps a synthetic results file to a decision label for every claim
    without manual input."""

    def setUp(self):
        self.register = load_register()
        self.data = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def labels(self, data=None):
        return decide(ResultsFile.model_validate(data or self.data), self.register)["claims"]

    def test_the_synthetic_results_get_the_hand_derived_labels(self):
        labels = self.labels()
        self.assertEqual(set(labels), set(CLAIM_IDS))
        self.assertEqual({k: v["label"] for k, v in labels.items()}, EXPECTED)
        self.assertEqual({k for k, v in labels.items() if v["implementation_error"]}, {"H3b"})
        self.assertTrue(labels["H3b"]["blocks_reporting"])
        self.assertEqual(labels["H1d"]["groups"]["alignment"]["display"], "aligned")  # 9 of 10: p = 11/1024
        cells = labels["H1b"]["groups"]["near_determinism"]["cells"]
        self.assertIn("sensitivity analysis gives inconclusive", cells["gpt_4o_20240806|browser_tool"]["reason"])
        self.assertIn("excluded: missing rate 0.3", cells["gemini_20_flash_001|text"]["reason"])
        floor = labels["H1a"]["groups"]["iia"]["cells"]["gpt_4o_20240806|structured"]
        self.assertEqual((floor["label"], floor["reason"]), ("inconclusive", "below the floor: n = 12 < 20"))
        holm_cells = labels["H1a"]["groups"]["iia"]["cells"]
        self.assertAlmostEqual(holm_cells["gpt_4o_20240806|text"]["tests"]["triple_odds"]["p_support_adjusted"],
                               0.002)
        self.assertEqual(labels["H1c"]["interpretation"], self.register.claim("H1c").interpretation.supported)

    def test_the_command_line_labels_the_file(self):
        out, path = io.StringIO(), Path(tempfile.mkdtemp()) / "labels.json"
        self.addCleanup(shutil.rmtree, path.parent)
        with contextlib.redirect_stdout(out):
            self.assertEqual(decide_module.main([str(FIXTURE), "--out", str(path)]), 0)
        printed = json.loads(out.getvalue())
        self.assertEqual(printed["H3b"], "inconclusive (implementation error)")
        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["register_sha256"], register_sha256())
        self.assertEqual(written["results_sha256"], hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
        self.assertEqual({k: v["label"] for k, v in decide_file(FIXTURE)["claims"].items()}, EXPECTED)

    def test_any_complete_results_get_a_label_for_every_claim(self):
        """No manual input: random statistics in the registered shape always map to labels."""
        rng = random.Random(20261022)
        for _ in range(200):
            data = copy.deepcopy(self.data)
            for groups in data["claims"].values():
                for cells in groups.values():
                    for cell in cells:
                        for result in list(cell["tests"].values()) + list((cell.get("sensitivity") or {}).values()):
                            if "p_support" in result:
                                p = rng.random()
                                result["p_support"], result["p_against"] = p, 1.0 - p
                            if "p_value" in result:
                                result["p_value"] = rng.random()
                            for key in result.get("counts", {}):
                                result["counts"][key] = rng.randint(0, result["n"])
                            if "credible_changes" in result.get("counts", {}):  # credible changes are raw changes
                                counts = result["counts"]
                                counts["credible_changes"] = rng.randint(0, counts["raw_changes"])
                            for key, value in result.get("metrics", {}).items():
                                result["metrics"][key] = value * rng.choice([0.5, 1.0, 1e6])
                            if "share" in result:
                                result["share"] = rng.random()
                        if "missing_rate" in cell:
                            cell["missing_rate"] = rng.choice([0.0, 0.05, 0.2, 0.3])
            labels = self.labels(data)
            self.assertEqual(set(labels), set(CLAIM_IDS))
            self.assertTrue(all(v["label"] in ("supported", "not_supported", "inconclusive")
                                for v in labels.values()))

    def test_incomplete_or_inconsistent_results_are_refused(self):
        cases = {
            "a claim missing": lambda d: d["claims"].pop("H4d"),
            "a group missing": lambda d: d["claims"]["H2a"].pop("payments_rise_with_excess"),
            "a test missing": lambda d: d["claims"]["H1b"]["near_determinism"][0]["tests"].pop("score_gap"),
            "no sensitivity": lambda d: d["claims"]["H1a"]["iia"][0].pop("sensitivity"),
            "no floor field": lambda d: d["claims"]["H1e"]["page_requests"][0].pop("n"),
            "a metric missing": lambda d: d["claims"]["H3b"]["closed_forms"][0]["tests"]["grid"]["metrics"].popitem(),
            "two cells for single": lambda d: d["claims"]["H4d"]["exponent"].append(
                dict(d["claims"]["H4d"]["exponent"][0], cell="other")),
            "both directions": lambda d: d["claims"]["H4d"]["exponent"][0]["tests"]["log_log_slope"].update(
                p_support=0.001, p_against=0.001),
            "no threshold value": lambda d: d["claims"]["H2b"]["monotone_in_alpha"][0]["tests"]["increment"].pop(
                "threshold"),
        }
        for name, mutate in cases.items():
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.subTest(case=name), self.assertRaises((ValueError, ValidationError)):
                self.labels(data)
        data = dict(self.data, register_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "produced for register"):
            decide(ResultsFile.model_validate(data), self.register, register_sha256())

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(holm([0.5]), [0.5])
        self.assertEqual(holm([0.9, 0.8]), [1.0, 1.0])  # 2 x 0.8 caps at 1; step-down keeps the max

    def test_count_rules_equal_their_interval_rules(self):
        """The p-values from counts decide exactly as the Clopper–Pearson and Wilson
        bounds do, computed independently by bisection and by the Wilson formula."""
        def cp_lower(k, n, conf):
            lo, hi = 0.0, 1.0
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if binom_sf(k, n, mid) < 1 - conf else (lo, mid)
            return lo if k else 0.0

        def wilson_lower(k, n, z):
            p = k / n
            return (p + z * z / (2 * n) - z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)

        z975 = 1.959963984540054
        cp = BoundTest(kind="bound", name="t", side="above", threshold=0.1, method="clopper_pearson",
                       support_count="k", against_count="k", alpha_support=0.05, alpha_against=0.05)
        wilson = cp.model_copy(update={"method": "wilson", "alpha_support": 0.025, "alpha_against": 0.025})
        result = decide_module.StatResult
        for n in (20, 57, 200):
            for k in range(0, n + 1, max(1, n // 20)):
                with self.subTest(n=n, k=k):
                    p_cp, _ = count_p_values(cp, result(n=n, counts={"k": k}), "x")
                    self.assertEqual(p_cp <= 0.05, cp_lower(k, n, 0.95) >= 0.1 - 1e-12)
                    p_w, _ = count_p_values(wilson, result(n=n, counts={"k": k}), "x")
                    self.assertEqual(p_w <= 0.025, wilson_lower(k, n, z975) >= 0.1)
        self.assertAlmostEqual(binom_cdf(0, 40, 0.1), 0.9 ** 40)
        self.assertAlmostEqual(binom_sf(9, 10, 0.5), 11 / 1024)


# ----------------------------------------------------------------------- AC-3
class DirectionAuditTest(unittest.TestCase):
    """AC-3 (reviewer checklist item): the audit lists every acceptance criterion of
    tickets.md once, verbatim, with a verdict and a reason."""

    def test_the_audit_covers_every_acceptance_criterion(self):
        criteria, ticket = [], None
        for line in (REPO_ROOT / "tickets.md").read_text(encoding="utf-8").splitlines():
            heading = re.match(r"^### (SL-\d+[a-z]?) · ", line)
            if heading:
                ticket, number = heading.group(1), 0
                continue
            item = re.match(r"^- \[ \] (.*)$", line)
            if item and ticket:
                number += 1
                criteria.append((ticket, "AC-%d" % number, item.group(1)))
        rows = []
        for line in (DOCS / "acceptance_direction_audit.md").read_text(encoding="utf-8").splitlines():
            if re.match(r"^\| SL-\d+", line):
                cells = [c.replace("\\|", "|").strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
                rows.append(cells)
        self.assertGreaterEqual(len(criteria), 100)
        self.assertEqual([tuple(r[:3]) for r in rows], criteria)
        for row in rows:
            self.assertEqual(row[3], "No", row[:2])
            self.assertTrue(row[4], row[:2])


# ----------------------------------------------------------------------- AC-4
class RegistrationTimingTest(unittest.TestCase):
    """AC-4 machinery: the E1 registration must be earlier than the first SL-18 trial,
    read from the manifests. Running it on SL-18 needs SL-17 first (see the handoff)."""

    def setUp(self):
        from tests.features.storelab.test_cache import Env
        from features.storelab import api

        self.env = Env()
        self.addCleanup(self.env.cleanup)
        config = self.env.config(mode="auto")
        self.runs = []
        for start in ("2027-03-02T09:00:00", "2027-03-01T12:00:00"):
            run_dir, planned = api.plan_run(config, snapshot=self.env.snapshot, root=self.env.runs, environ={})
            manifest = planned.manifest
            at = datetime.datetime.fromisoformat(start).replace(tzinfo=datetime.timezone.utc)
            append_status(manifest, "RUNNING", at, None)
            append_status(manifest, "COMPLETED", at + datetime.timedelta(hours=5), "done")
            (run_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
            self.runs.append((run_dir, manifest))

    def receipts(self, when: str | None) -> Receipts:
        data = load_receipts().model_dump(mode="json")
        if when is not None:
            data["registrations"][0].update(status="filed", url="https://osf.io/abcd1", registered_utc=when)
        return Receipts.model_validate(data)

    def test_registered_before_the_first_trial_passes_and_after_fails(self):
        manifests = [m for _, m in self.runs]
        early = check_e1(self.receipts("2027-03-01T11:59:59Z"), manifests)
        self.assertEqual((early["passed"], early["first_trial_utc"]), (True, "2027-03-01T12:00:00+00:00"))
        self.assertEqual(early["first_run_id"], self.runs[1][1].run_id)
        late = check_e1(self.receipts("2027-03-01T12:00:00Z"), manifests)
        self.assertFalse(late["passed"])
        self.assertIn("at or after the first trial of run %s" % self.runs[1][1].run_id, late["reason"])

    def test_the_committed_receipts_are_not_filed_so_the_check_fails(self):
        self.assertEqual(load_receipts().get("E1").status, "not_filed")
        result = check_e1(load_receipts(), [m for _, m in self.runs])
        self.assertEqual((result["passed"], result["reason"]), (False, "the E1 registration is not filed"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(registration.main(["check-e1", *[str(d) for d, _ in self.runs]]), 1)
        self.assertFalse(json.loads(out.getvalue())["passed"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(registration.main(["check-e1"]), 1)  # no manifests: nothing to compare

    def test_a_filed_receipt_needs_its_url_and_time(self):
        data = load_receipts().model_dump(mode="json")
        data["registrations"][0]["status"] = "filed"
        with self.assertRaisesRegex(ValidationError, "url and registered_utc"):
            Receipts.model_validate(data)
        path = self.env.root / "receipts.yaml"
        data["registrations"][0].update(url="https://osf.io/abcd1", registered_utc="2027-03-01T00:00:00Z")
        path.write_text(json.dumps(data), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(registration.main(["check-e1", str(self.runs[0][0]), "--receipts", str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
