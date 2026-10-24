"""SL-19S: the E2 dry run on A17's stipulated parameters.

- AC-2: generated markets match A17 (parameter values, regime constraints, screen);
- AC-1: the committed result files have ≥ 200 valid markets per configuration, < 1%
  exclusions, and one market hash per market across mechanisms;
- AC-3: the implementation gates (Prop gating gain; zero set-awareness gap with
  deterministic lists);
- AC-4: reproducibility (a seeded rerun of committed configurations is byte for byte
  equal) and the label on tables and figures;
- AC-5: the registered H2 statistics are reported, labelled exploratory, with no
  confirmatory label recorded.
"""
from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
from features.storelab import cli
from features.storelab.analysis import e2_markets as em
from features.storelab.analysis import e2_report, e2_run
from features.storelab.analysis.figures import FIGURES, LABEL
from features.storelab.config_files import load_config
from features.storelab.core.agent_params import AgentParams, dump_agent_params
from features.storelab.core.mechanisms.market import action_table
from features.storelab.core.rng import stream
from scipy import stats

CONFIGS = Path(cli.__file__).resolve().parent / "configs"
A17_LEVELS = {  # requirements/A17_stipulated_parameters.md §1–§2, the levels E2 uses
    "rho_v": (0.5, 0.95), "sigma": (1.0, 0.1), "outside": (0.0,), "delta": ((0.0,), (-1.5,)), "beta_slope": (0.1,),
    "chi": (0.0, 1.0), "models": (1, 4), "gamma_exponent": (0.3, 1.0), "contribution": ("uniform",),
    "values": (("lognormal", -0.5, 0.5), ("lognormal", -0.5, 2.0)), "kappa": (0.0, 0.2), "cost_weight": (0.1, 2.0),
    "k": (8,), "versions": (2,), "bidding": ("human_conversion_rate",)}


def sample_markets(configs, indices=range(4)):
    for config in configs:
        for index in indices:
            yield (config, index, *em.market(config, index))


class A17Test(unittest.TestCase):
    """AC-2: generated markets match A17."""

    def test_the_baseline_and_levels_are_a17s(self):
        b = em.BASELINE
        self.assertEqual((b.eligible, b.rho_v, b.sigma, b.outside, b.models, b.chi, b.advertisers, b.kappa,
                          b.cost_weight, b.omega, b.requests, b.k, b.k_s, b.versions, b.tau),
                         (10, 0.8, 0.3, -1.0, 2, 0.5, 6, 0.05, 0.5, 0.5, 1000, 4, 1, 1, 0.5))
        self.assertEqual(b.model_deltas(), (-0.3, -0.7))
        self.assertEqual(dataclasses.replace(b, models=1).model_deltas(), (-0.5,))
        self.assertEqual(dataclasses.replace(b, models=4).model_deltas(), (-0.3, -0.5, -0.5, -0.7))
        np.testing.assert_allclose(dataclasses.replace(b, k=8).gamma(),
                                   [0.300, 0.198, 0.155, 0.131, 0.114, 0.102, 0.093, 0.086], atol=6e-4)
        self.assertEqual((b.rho_organic, b.rho_sponsored, b.zeta, b.values, b.theta_bar()),
                         (0.6, 0.54, (0.2, 0.6), ("uniform", 1.0), 0.5))
        seen: dict[str, set] = {}
        for change in em.LEVELS.values():
            for key, value in change.items():
                seen.setdefault(key, set()).add(value)
        self.assertEqual(seen.pop("k_s"), {2})
        self.assertEqual({k: set(v) for k, v in A17_LEVELS.items()}, seen)
        self.assertEqual(len(em.LEVELS), 24)
        groups = [c.group for c in em.configurations()]
        self.assertEqual((groups.count("core"), groups.count("sensitivity"), groups.count("part3")), (30, 120, 15))

    def test_the_config_records_the_baseline(self):
        config = load_config(CONFIGS / "e2_stipulated.yaml")
        self.assertEqual(config.seed, em.MASTER_SEED)
        self.assertEqual((config.agents.simulated.sigma, config.agents.simulated.outside_value,
                          config.agents.simulated.page_request_prob), (0.3, -1.0, 0.5))
        np.testing.assert_allclose(config.humans.position_weights, dataclasses.replace(em.BASELINE, k=8).gamma(),
                                   atol=1e-6)
        self.assertEqual(config.mechanism.markets_per_configuration, em.MARKETS)
        self.assertEqual(config.mechanism.agent_shares, list(em.ALPHAS))

    def test_the_stipulated_agent_params_file(self):
        path = CONFIGS / "e2_stipulated_agent_params.json"
        params = AgentParams.model_validate_json(path.read_bytes())
        self.assertEqual(params.source, "stipulated")
        schema = json.loads((cli.CONTRACTS / "agent_params.schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), schema)
        config = em.configurations()[0]
        draw = em.draw(config.params, stream(em.MASTER_SEED, "sl19s-market", config.config_id, 0))
        self.assertEqual(path.read_text(encoding="utf-8"), dump_agent_params(em.agent_params(config.params, draw)))

    def test_market_parameter_values(self):
        configs = [c for c in em.configurations() if c.config_id in (
            "core/alpha=0.5/human_optimised/none", "sensitivity/models=4/alpha=0.25", "sensitivity/K=8,Ks=2/alpha=1",
            "sensitivity/values=lognormal(s=2)/alpha=0.75", "sensitivity/g~U[0.05,0.15]/alpha=0.5",
            "sensitivity/beta=-0.1(k-1)/alpha=0.5", "sensitivity/versions=2/alpha=0.5")]
        for config, index, market, info, draw in sample_markets(configs, range(2)):
            p = config.params
            with self.subTest(config=config.config_id, index=index):
                context = market.contexts[0]
                self.assertEqual((market.requests, market.max_displayed, market.max_sponsored, len(context.offers),
                                  len(market.advertisers), context.agent_share), (1000, p.k, p.k_s, 10, 6, config.alpha))
                np.testing.assert_allclose(market.position_weights, p.gamma())
                np.testing.assert_allclose(draw.u, stats.norm.cdf(draw.z))
                for j, u in zip(draw.products, draw.u):
                    offer = context.offers[j]
                    self.assertAlmostEqual(offer.match, u)
                    self.assertEqual((offer.organic.request, offer.sponsored.request), (0.6, 0.54))
                    self.assertAlmostEqual(offer.organic.conversion, 0.2 + 0.6 * u)
                    self.assertTrue(0.05 <= offer.contribution <= 0.15 if p.contribution == "uniform"
                                    else offer.contribution == 0.1)
                    self.assertEqual((market.costs[j].kind, market.costs[j].weight, market.costs[j].capacity),
                                     ("quadratic", p.cost_weight, p.omega * 1000))
                self.assertEqual(len(market.models), p.models)
                self.assertEqual(len(market.states), 2 ** p.models if p.versions == 2 else 1)
                for m, delta in zip(market.models, p.model_deltas()):
                    self.assertEqual((m.sigma, m.label_effect, m.page_request_prob, m.outside["x"]),
                                     (p.sigma, delta, p.chi, p.outside))
                    np.testing.assert_allclose(m.position_effects, [-p.beta_slope * k for k in range(p.k)])
                self.assertEqual(set(market.values), set(market.advertisers))
                self.assertAlmostEqual(market.guardrail.kappa, p.kappa)

    def test_scores_follow_the_a17_formula(self):
        p = dataclasses.replace(em.BASELINE, versions=2)
        rng, replay = stream(1, "t"), stream(1, "t")
        d = em.draw(p, rng)
        z = replay.standard_normal(10)
        replay.choice(d.products, size=6, replace=False)
        noise, shifts = replay.standard_normal((2, 10)), replay.standard_normal((2, 10))
        for m in range(2):
            base = 0.8 * z + math.sqrt(1 - 0.64) * noise[m]
            np.testing.assert_allclose(d.scores["m%d" % m][0], base)
            np.testing.assert_allclose(d.scores["m%d" % m][1], base + 0.5 * shifts[m])

    def test_regime_constraints_and_screen(self):
        configs = [c for c in em.configurations() if c.group == "core" and c.alpha == 0.5] + [
            c for c in em.configurations() if c.config_id in ("sensitivity/models=4/alpha=0.5",
                                                              "sensitivity/K=8,Ks=2/alpha=0.5")]
        checked = {"agent_aware": 0, "gated": 0}
        for config, index, market, info, draw in sample_markets(configs, range(6)):
            p, favourites = config.params, set(info["favourites"])
            room = p.k - p.k_s
            with self.subTest(config=config.config_id, index=index):
                rankings = [o.ranking for o in market.contexts[0].organic]
                self.assertEqual(len(rankings), 2 if config.variation == "randomised" else 1)
                if config.regime == "agent_aware" and info["feasible"]:
                    for ranking in rankings:  # every favourite among the first K − K^s organic products
                        self.assertTrue(favourites <= set(ranking[:room]))
                    checked["agent_aware"] += 1
                if config.regime == "gated" and info["feasible"]:
                    table = action_table(market)
                    for shown, (_, sponsored) in zip(table.displayed, table.rows):
                        organic = [j for j in shown if j not in sponsored]
                        self.assertFalse(favourites & set(organic))  # no favourite displayed organically
                    checked["gated"] += 1
                # the screen, recomputed from the market's scores
                ok = True
                for m in market.models:
                    for state in market.states:
                        v = m.scores[state.state_id]["x"]
                        best = max(v, key=v.get)
                        if v[best] > m.outside["x"]:
                            ok &= all(v[best] + m.label_effect > w for j, w in v.items() if j != best) and (
                                v[best] + m.label_effect > m.outside["x"])
                self.assertEqual(bool(info["screen"]), ok)
                self.assertEqual(info["excess"], len(favourites) - room)
                paper = (1 - p.kappa) * info["m_ref"]
                floor = paper if info["m_organic"] >= paper else info["m_organic"] - em.FLOOR_SLACK * 1000
                self.assertEqual(info["floor_binds"], int(info["m_organic"] < paper))
                self.assertAlmostEqual(info["threshold"], floor)
                self.assertAlmostEqual(market.guardrail.m_ref * (1 - p.kappa), floor)
        self.assertGreater(checked["agent_aware"], 5)
        self.assertGreater(checked["gated"], 5)

    def test_randomised_lists_vary_the_unconstrained_products(self):
        config = next(c for c in em.configurations() if c.config_id == "core/alpha=0.5/human_optimised/randomised")
        market, info, _ = em.market(config, 0)
        r, varied = info["rankings"]
        room = config.params.k - config.params.k_s
        self.assertEqual(varied[:room], r[room:2 * room])
        self.assertEqual(varied[room:2 * room], r[:room])

    def test_the_gating_instance_meets_prop_gatings_hypotheses(self):
        for index in range(20):
            draw, info = em.gating_draw(index)
            v, delta = draw.scores["m0"][0], -0.5
            f, r = draw.products.index(info["f"]), draw.products.index(info["r"])
            with self.subTest(index=index):
                self.assertGreater(v[f] + delta, v[r])
                self.assertTrue(all(v[r] + delta > v[i] for i in range(10) if i not in (f, r)))
                self.assertEqual(draw.u[f], draw.u.max())
                self.assertGreater((v + delta).min(), -1.0)
                self.assertTrue({info["f"], info["r"]} <= set(draw.advertisers))
        market, info = em.gating_market(0, 0.05, "gated")
        table = action_table(market)
        self.assertLess(np.abs(table.q_bar.sum(axis=1) - 1).max(), 1e-15)  # deterministic limit: no outside choice
        self.assertTrue(set(np.unique(np.round(table.q_bar, 12))) <= {0.0, 1.0})


# ------------------------------------------------------------- committed results
def summary() -> dict:
    return json.loads(e2_report.SUMMARY.read_text(encoding="utf-8"))


class ResultsTest(unittest.TestCase):
    """AC-1 and AC-3 on the committed result files."""

    def test_every_configuration_has_200_valid_markets_and_one_hash_per_market(self):
        data = summary()
        configs = {c.config_id for c in em.configurations()}
        self.assertEqual(set(data["configurations"]), configs)
        for config_id, row in data["configurations"].items():
            with self.subTest(config=config_id):
                self.assertGreaterEqual(row["valid"], 200)
                self.assertLess(row["excluded"] / row["markets"], 0.01)
                self.assertTrue(row["one_hash_per_market"])

    def test_the_implementation_gates_pass(self):
        gates = summary()["gates"]
        self.assertLessEqual(gates["gating"]["max_relative_error"], 1e-6)
        self.assertLessEqual(gates["gating"]["mean_minus_formula_in_mc_se"], 3.0)
        self.assertLessEqual(gates["setblind"]["max_relative_gap"], 1e-6)
        self.assertGreaterEqual(gates["setblind"]["markets"], 200 * 150)  # core without variation, sensitivity, part 3
        self.assertTrue(gates["passed"])

    def test_a_seeded_rerun_reproduces_the_committed_rows(self):
        """The first markets of committed files, regenerated, give the same CSV lines; whole
        configurations were rerun byte for byte by ``e2_run --verify-rerun`` (summary)."""
        for config_id in ("core/alpha=0.5/gated/randomised", "sensitivity/models=4/alpha=0.75"):
            config = next(c for c in em.configurations() if c.config_id == config_id)
            rows = [r for i in range(2) for r in e2_run.evaluate_market((config, i))]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "x.csv.gz"
                e2_run.write_rows(path, e2_run.MARKET_FIELDS + e2_run.OUTCOME_FIELDS, rows)
                fresh = e2_run.read_rows(path)
            committed = e2_run.read_rows(e2_run.result_path(e2_run.RESULTS, config_id))[:len(fresh)]
            self.assertEqual(fresh, committed)
        self.assertTrue(summary()["rerun_identical"])

    def test_tables_and_figures_carry_the_label(self):
        report = e2_report.REPORT.read_text(encoding="utf-8")
        self.assertIn(LABEL, report)
        self.assertNotRegex(report, r"\b(Supported|Not supported)\b")
        for name in e2_report.FIGURE_NAMES:
            for suffix in (".pdf", ".png", ".csv"):
                self.assertTrue((FIGURES / "sl19s" / (name + suffix)).exists(), name + suffix)
            self.assertTrue((FIGURES / "sl19s" / (name + ".csv")).read_text(encoding="utf-8").startswith("# " + LABEL))

    def test_the_registered_statistics_are_reported_and_exploratory(self):
        stats_ = summary()["statistics"]
        for key in ("H2a", "H2b", "H2c"):
            self.assertIn(key, stats_)
            self.assertEqual(stats_[key]["label"], "exploratory")
        self.assertNotIn("decision", json.dumps(stats_))


if __name__ == "__main__":
    unittest.main()
