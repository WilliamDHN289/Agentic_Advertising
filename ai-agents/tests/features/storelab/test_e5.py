"""SL-22 E5 field protocol and analysis: the randomisation test's size and power (AC-1),
the data contract and PII scan (AC-2), and the pre-registration's citation of the
protocol version (AC-3); plus direct checks of the request table, H5a, H5b and the
decision-rule integration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from pathlib import Path

import jsonschema
import numpy as np
from features.storelab import cli
from features.storelab.analysis import e5, e5_validation
from features.storelab.analysis.decide import ResultsFile, decide_claim
from features.storelab.analysis.register import load_register, register_sha256

DOCS = Path(e5.__file__).resolve().parents[1] / "docs"
REPORT = DOCS / "e5_validation.json"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


class CommittedStudyTest(unittest.TestCase):
    def test_the_report_matches_the_code_and_the_registered_counts(self):
        data = report()
        self.assertEqual(data["code_fingerprint"], e5_validation.code_fingerprint())
        self.assertEqual(data["counts"], e5_validation.FULL)
        self.assertEqual((data["master_seed"], data["protocol_version"]), (e5_validation.MASTER_SEED,
                                                                          e5.PROTOCOL_VERSION))
        self.assertEqual(data["passed"], {"AC-1": True, "AC-2": True})
        self.assertTrue((DOCS / "e5_validation.md").read_text(encoding="utf-8").startswith(
            "# SL-22 E5 analysis validation"))


# ----------------------------------------------------------------------- AC-1
class H5aTest(unittest.TestCase):
    """AC-1: permutation-test size at α = 0.05 in [0.03, 0.07] over 1,000 synthetic
    datasets; power at the designed effect within ±5 points of the analytic target."""

    def test_the_committed_size_and_power(self):
        a = report()["h5a"]
        self.assertEqual((a["datasets"], a["draws"]), (1000, 10_000))
        self.assertEqual(a["n"], e5_validation.designed_n())
        for name, size in a["checks"].items():
            with self.subTest(check=name):
                self.assertGreaterEqual(size, 0.03)
                self.assertLessEqual(size, 0.07)
        self.assertAlmostEqual(a["analytic_power"], 0.80, delta=0.001)
        self.assertLessEqual(abs(a["simulated_power"] - a["analytic_power"]), 0.05)
        self.assertTrue(a["passed"])

    def table(self, seed=3, n=4000, tau=0.0):
        return e5_validation.synthetic_table(np.random.default_rng(seed), n, tau)

    def test_cell_draws_equal_the_statistic_of_explicit_assignments(self):
        """The randomisation distribution is drawn as counts per (arm, choice, net) type;
        any explicit assignment of requests gives the same statistic through its counts."""
        table = self.table()
        rand = e5._Randomisation(table, 10, np.random.default_rng(0))
        z = np.random.default_rng(1).random(table.n) < 0.5
        keys = np.stack([table.strong, table.choice, table.net], axis=1)
        counts = np.array([np.sum(z & (keys == t).all(axis=1)) for t in rand.types])
        for tau0 in (-0.1, 0.0, 0.07):
            direct_net = table.net * np.exp(-tau0 * table.strong)
            parts = []
            for arm in (z, ~z):
                n_a, c_a = direct_net[arm], table.choice[arm]
                mn, mc = n_a.mean(), c_a.mean()
                var = (n_a.var() / mn ** 2 + c_a.var() / mc ** 2 - 2 * np.cov(n_a, c_a, bias=True)[0, 1] / (mn * mc))
                parts.append((math.log(mn / mc), var / arm.sum()))
            direct = (parts[0][0] - parts[1][0]) / math.sqrt(parts[0][1] + parts[1][1])
            self.assertAlmostEqual(rand.statistic(tau0, counts[None, :])[0], direct, places=10)

    def test_tost_labels_agree_with_the_intervals(self):
        for seed, tau in ((1, 0.0), (2, 0.0), (3, 0.09), (4, -0.25)):
            result = e5.h5a(self.table(seed, 20000, tau), draws=1999, seed=seed)
            with self.subTest(seed=seed):
                inside = -0.10 < result.ci90[0] and result.ci90[1] < 0.10
                self.assertEqual(result.p_support <= 0.05, inside)
                outside = result.ci95[0] > 0.10 or result.ci95[1] < -0.10
                self.assertEqual(result.p_against < 0.05, outside)
                self.assertLess(result.ci95[0], result.ci90[0])
                self.assertGreater(result.ci95[1], result.ci90[1])

    def test_one_assignment_probability_is_required(self):
        table = self.table()
        table.probability[:10] = 0.4
        with self.assertRaisesRegex(ValueError, "one assignment probability"):
            e5.h5a(table, draws=99)

    def test_the_power_formula(self):
        args = (0.5, 0.30, 0.22, 0.495)
        n = e5.h5a_required_n(0.8, *args)
        self.assertGreaterEqual(e5.h5a_power(n, *args), 0.8)
        self.assertLess(e5.h5a_power(n - 1, *args), 0.8)
        self.assertLess(e5.h5a_power(n, *args, tau=0.05), e5.h5a_power(n, *args))
        se = e5.h5a_standard_error(n, *args)
        self.assertAlmostEqual(se, math.sqrt((1 - 0.495) / 0.495 * (1 / (n * 0.5 * 0.22) + 1 / (n * 0.5 * 0.30))))
        with self.assertRaises(ValueError):
            e5.h5a_required_n(0.8, *args, tau=0.1)


# ----------------------------------------------------------------------- H5b
class H5bTest(unittest.TestCase):
    def test_cells_follow_arm_and_identified_traffic_type(self):
        table = e5_validation.synthetic_table(np.random.default_rng(5), 3000)
        self.assertEqual([c for c, _ in e5.h5b_cells(table)], ["strong/human", "strong/agent", "weak/human",
                                                                "weak/agent"])
        unknown = copy.copy(table)
        unknown.traffic = np.full(table.n, "unknown")
        self.assertEqual([c for c, _ in e5.h5b_cells(unknown)], ["strong", "weak"])

    def test_cross_fitting_uses_other_folds_only(self):
        x = np.array([1, 1, 0, 0])
        y = np.array([1, 0, 1, 0])
        fold = np.array([0, 1, 0, 1])
        loss = e5._cross_fitted_loss(x, y, fold, 2)
        p = (0 + 0.5) / (1 + 1.0)  # row 0 (x = 1, fold 0) is predicted from row 1 only (y = 0)
        self.assertAlmostEqual(loss[0], -math.log(p))
        self.assertAlmostEqual(e5._auc(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])), 0.5)
        self.assertAlmostEqual(e5._auc(np.array([1, 0]), np.array([1, 0])), 1.0)

    def test_the_committed_h5b_checks(self):
        b = report()["h5b"]
        self.assertLessEqual(b["null"]["cell_supported_rate"], b["null_size_bound"] + 3 * math.sqrt(
            b["null_size_bound"] / (b["datasets"] * b["null"]["cells"])))
        self.assertLessEqual(abs(b["effect"]["cell_supported_rate"] - b["analytic_power"]), 0.05)


class DecisionTest(unittest.TestCase):
    def test_events_to_labels_through_the_registered_rules(self):
        """Synthetic event log → request table → H5a, H5b → results file → decide.py."""
        events = e5.parse_events(json.dumps(e) for e in e5_validation.synthetic_events(11, 400))
        table = e5.request_table(events)
        self.assertGreater(table.n, 300)
        h5a = e5.h5a(table, draws=999)
        cells = e5.h5b(table, bootstrap=499)
        results = ResultsFile.model_validate({"results_version": 1, "register_sha256": register_sha256(), "claims": {
            "H5a": {"value_invariance": [{"cell": "all", "n": h5a.n, "tests": {"log_ratio": h5a.decide_input()}}]},
            "H5b": {"predicts_better": [c.decide_input() for c in cells]}}})
        register = load_register()
        for claim_id in ("H5a", "H5b"):  # the E5 registration's claims (decide() needs every claim's results)
            label = decide_claim(register.claim(claim_id), results.claims[claim_id])
            self.assertIn(label["label"], {"supported", "not_supported", "inconclusive"})


# ----------------------------------------------------------------------- AC-2
class ContractTest(unittest.TestCase):
    """AC-2: schema fixtures validate; a PII field scan finds no email, name or address fields."""

    def setUp(self):
        self.schema = json.loads(e5_validation.CONTRACT.read_text(encoding="utf-8"))
        self.lines = e5_validation.FIXTURE.read_text(encoding="utf-8").splitlines()

    def test_the_committed_schema_is_current_and_the_fixtures_validate(self):
        self.assertEqual(e5_validation.CONTRACT.read_text(encoding="utf-8"), cli.schema_text("e5_event.schema.json"))
        validator = jsonschema.Draft202012Validator(self.schema)
        kinds = set()
        for line in self.lines:
            record = json.loads(line)
            kinds.add(record["event_type"])
            self.assertEqual(list(validator.iter_errors(record)), [])
        self.assertEqual(kinds, {"assignment", "click", "qualifying_event", "purchase", "return"})
        self.assertEqual(len(e5.parse_events(self.lines)), len(self.lines))

    def test_no_personal_data_fields(self):
        self.assertEqual(e5.pii_fields(self.schema), [])
        for line in self.lines:
            self.assertEqual(e5.pii_fields(json.loads(line)), [])
        planted = {"properties": {"customer_email": {}, "full_name": {}, "billing_address": {}, "phone": {},
                                  "price": {}}}
        self.assertEqual(sorted(e5.pii_fields(planted)), ["/properties/billing_address", "/properties/customer_email",
                                                           "/properties/full_name", "/properties/phone"])
        self.assertEqual(e5.pii_fields({"note": "mail jane@example.org"}), ["/note"])

    def test_the_schema_rejects_extra_and_identifying_fields(self):
        validator = jsonschema.Draft202012Validator(self.schema)
        record = json.loads(self.lines[0])
        for change in ({"email": "a@b.co"}, {"request_id": "plain-id-42"}, {"arm": "medium"}):
            with self.subTest(change=change):
                self.assertFalse(validator.is_valid({**record, **change}))

    def test_windows_and_exclusions(self):
        def event(n, request, event_type, minutes, **extra):
            return {"schema_version": 1, "protocol_version": e5.PROTOCOL_VERSION, "event_id": e5.digest("e%d" % n),
                    "request_id": e5.digest(request), "event_time_utc": "2026-11-0%dT%02d:%02d:00Z" % (
                        2 + minutes // 1440, (minutes % 1440) // 60, minutes % 60), "product_id": "SKU-1",
                    "event_type": event_type, **extra}

        assign = dict(arm="strong", assignment_probability=0.5, candidate_set_id="S", candidate_product_ids=["SKU-1",
                                                                                                          "SKU-2"],
                      traffic_type="agent")
        order = e5.digest("o")
        raw = [event(0, "kept", "assignment", 0, **assign),
               event(1, "kept", "qualifying_event", 5, kind="add_to_cart"),
               event(2, "kept", "purchase", 10, order_id=order, quantity=1),
               event(3, "returned", "assignment", 0, **assign),
               event(4, "returned", "qualifying_event", 5, kind="confirmed_selection"),
               event(5, "returned", "purchase", 10, order_id=e5.digest("p"), quantity=1),
               event(6, "returned", "return", 1440 * 2, order_id=e5.digest("p"), quantity=1),
               event(7, "test", "assignment", 0, **{**assign, "exclusion": "internal_test"}),
               event(8, "orphan", "click", 3),
               event(2, "kept", "purchase", 10, order_id=order, quantity=1)]  # duplicate event id
        table = e5.request_table(e5.parse_events(json.dumps(r) for r in raw))
        self.assertEqual(table.n, 2)
        self.assertEqual(table.excluded["internal_test"], 1)
        self.assertEqual(table.excluded["no_assignment"], 1)
        self.assertEqual(table.excluded["duplicate_event"], 1)
        rows = dict(zip(table.request_id, zip(table.choice, table.purchase, table.net)))
        self.assertEqual(rows[e5.digest("kept")], (1, 1, 1))
        self.assertEqual(rows[e5.digest("returned")], (1, 1, 0))


# ----------------------------------------------------------------------- AC-3
class PreregistrationTest(unittest.TestCase):
    """AC-3: the E5 pre-registration (SL-01) cites this protocol version."""

    def test_the_pre_registration_cites_the_current_protocol(self):
        protocol = DOCS / "e5_protocol.md"
        digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
        prereg = (DOCS / "preregistration_E5.md").read_text(encoding="utf-8")
        self.assertIn("| `features/storelab/docs/e5_protocol.md` (`%s`) | `%s` |" % (e5.PROTOCOL_VERSION, digest),
                      prereg)
        self.assertIn("**Protocol version:** `%s`" % e5.PROTOCOL_VERSION, prereg)
        self.assertIn("**Protocol version: `%s`**" % e5.PROTOCOL_VERSION, protocol.read_text(encoding="utf-8"))
        self.assertEqual(load_register().registrations["E5"].document, "docs/preregistration_E5.md")


if __name__ == "__main__":
    unittest.main()
