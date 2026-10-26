"""SL-19S report: tables, figures and statistics from the committed E2 result files.

    python -m features.storelab.analysis.e2_report

One command regenerates everything from ``artifacts/results/sl19s/`` (written by
``e2_run``):
- ``docs/e2_stipulated_results.md`` and ``.json``;
- the seven figures in ``artifacts/figures/sl19s/`` (PDF, PNG and a CSV of the plotted data);
- the full tables as CSVs in ``artifacts/results/sl19s/tables/``.

Every table and figure is labelled "Stipulated parameters (A17), pre-E1, exploratory".

**Implementation gates** (AC-3; a failure stops the report):
- Prop gating: every market's relative error ≤ 1e-6, and the mean of gain / (n ξ) within
  3 Monte Carlo SE of E min{θ_f, θ_r} = θ̄ / 3 (A12 §4), per κ;
- with deterministic organic lists, |V^aw − V^bl| / V^aw ≤ 1e-6 in every market;
  V = Σ_j θ_j X_j + Γ is the surplus objective of M3 and of set-blind M3.

**Registered statistics** (claims register H2a–H2c and Conj gating) are computed as
written:
- means and slopes with 95% intervals from a market bootstrap stratified by
  configuration (B = 2,000);
- one-sided p-values for each bound and Holm across the registered families.

They are recorded as "exploratory": no confirmatory label is set, by the A17 rule. The one
label recorded is H2a part 1's "inconclusive" where a configuration has fewer than 200
screen-passing markets: A12 §2 fixes it whenever a minimum sample size is not met, whatever the
data (owner decision 2026-10-24).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..core.rng import stream
from . import e2_markets as em
from . import e2_run as er
from .decide import holm
from .figures import LABEL, REPO, figure, save

DOCS = Path(__file__).resolve().parents[1] / "docs"
REPORT, SUMMARY = DOCS / "e2_stipulated_results.md", DOCS / "e2_stipulated_results.json"
TABLES = er.RESULTS / "tables"
B = 2000
MECHANISMS = ("M1", "M2", "M3_set_blind", "M3", "M4", "profit_rule")
FIGURE_NAMES = ("1_payments_by_traffic_type", "2_profit_and_match_quality", "3_decomposition", "4_setblind_gap",
                "5_gating_gain", "6_agent_payments_vs_excess", "7_agent_payments_by_screen")
SUBSETS = ("all", "paper_feasible")  # A17 §5.1: all markets, and those where the guardrail floor does not bind
SUBSET_NAMES = {"all": "all markets", "paper_feasible": "paper-feasible subset"}
GAP_TOL, GATING_TOL, MC_SE = 1e-6, 1e-6, 3.0
MARGIN = 0.01
FLOOR = 200  # A12 §5: valid markets per configuration (SL-19); H2a part 1 counts screen-passing ones
E_MIN_UNIFORM = 1.0 / 3.0  # E min{θ_f, θ_r} for i.i.d. U[0, 1] values (A1 §1.4)


class GateFailure(RuntimeError):
    pass


# --------------------------------------------------------------------- data
def _num(value: str):
    if value in ("", None):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load() -> tuple[dict, dict]:
    markets: dict[str, list[dict]] = {}
    for config in em.configurations():
        rows = [{k: _num(v) for k, v in r.items()} for r in er.read_rows(er.result_path(er.RESULTS, config.config_id))]
        markets[config.config_id] = rows
    gating = {name: [{k: _num(v) for k, v in r.items()} for r in er.read_rows(er.result_path(er.RESULTS, name))]
              for name in er.gating_tasks(1)}
    return markets, gating


def paper_feasible(markets: dict) -> dict:
    """The paper-feasible subset (A17 §5.1): markets where the guardrail floor does not bind."""
    return {cid: [r for r in rows if not r["floor_binds"]] for cid, rows in markets.items()}


def floor_share(markets: dict, config: str) -> float:
    """Share of valid markets where the floor binds, over ``config % α`` for every α."""
    rows = [r for a in em.ALPHAS for r in _valid(markets[config % a], "M3")]
    return sum(r["floor_binds"] for r in rows) / len(rows)


def by_market(rows: list[dict]) -> dict[int, dict[str, dict]]:
    out: dict[int, dict[str, dict]] = {}
    for r in rows:
        out.setdefault(r["index"], {})[r["mechanism"]] = r
    return out


# ---------------------------------------------------------------- bootstrap
def _rng(*keys):
    return stream(em.MASTER_SEED, "sl19s-bootstrap", *keys)


def boot_mean(values: np.ndarray, key: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    draws = _rng(key).integers(0, len(values), size=(B, len(values)))
    return values[draws].mean(axis=1)


def boot_slope(x: np.ndarray, y: np.ndarray, strata: np.ndarray, key: str) -> tuple[float, float, np.ndarray]:
    """OLS slope and intercept; bootstrap slopes resampling markets within strata."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rng = _rng(key)
    out = np.empty((B, 2))
    for b in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out[b] = np.polyfit(x[idx], y[idx], 1)
    return float(slope), float(intercept), out


def interval(draws: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))]


def tost(draws: np.ndarray, margin: float = MARGIN) -> dict:
    """Bootstrap duals of the registered equivalence rule (A12 §2)."""
    p_support = max(float(np.mean(draws <= -margin)), float(np.mean(draws >= margin)))
    p_against = min(1.0, 2 * min(float(np.mean(draws <= margin)), float(np.mean(draws >= -margin))))
    return {"ci90": interval(draws, 0.90), "ci95": interval(draws), "p_support": p_support, "p_against": p_against}


def bound(draws: np.ndarray, threshold: float, side: str) -> dict:
    """One-sided p-values: support = share beyond the threshold on the wrong side."""
    below, above = float(np.mean(draws < threshold)), float(np.mean(draws > threshold))
    p_support, p_against = (below, above) if side == "above" else (above, below)
    return {"ci95": interval(draws), "p_support": p_support, "p_against": p_against}


# ------------------------------------------------------------------ checks
def counts(markets: dict) -> dict:
    out = {}
    for config_id, rows in markets.items():
        idx = by_market(rows)
        valid = [i for i, mech in idx.items() if not next(iter(mech.values()))["excluded"]]
        hashes = all(len({r["market_hash"] for r in mech.values()}) == 1 and set(mech) == set(MECHANISMS)
                     for mech in idx.values())
        first = [next(iter(idx[i].values())) for i in valid]
        out[config_id] = {"markets": len(idx), "valid": len(valid), "excluded": len(idx) - len(valid),
                          "one_hash_per_market": hashes,
                          "screen_pass": sum(r["screen"] for r in first), "feasible": sum(r["feasible"] for r in first),
                          "floor_binds": sum(r["floor_binds"] for r in first),
                          "qp_fallbacks": sum(r["qp_fallbacks"] or 0 for i in valid for r in idx[i].values()),
                          "exclusion_statuses": sorted({next(iter(idx[i].values()))["status"] for i in idx
                                                        if i not in valid})}
    return out


def gates(markets: dict, gating: dict) -> dict:
    out = {"gating": {}, "setblind": {}}
    errors, z_scores, per_kappa = [], [], {}
    for kappa in em.GATING_KAPPAS:
        rows = [r for r in gating["gating/i_part/kappa=%g" % kappa] if not r["excluded"]]
        rel = np.array([r["relative_error"] for r in rows])
        loss = np.array([abs(r["match_loss"] - r["expected_match_loss"]) / 1000 for r in rows])
        ratio = np.array([r["gain"] / (1000 * r["xi"]) for r in rows])
        se = ratio.std(ddof=1) / math.sqrt(len(ratio))
        z = abs(ratio.mean() - E_MIN_UNIFORM) / se
        per_kappa["%g" % kappa] = {"markets": len(rows), "max_relative_error": float(rel.max()),
                                   "max_match_loss_error_per_request": float(loss.max()),
                                   "mean_gain_over_n_xi": float(ratio.mean()), "closed_form": 1.0 / 3.0,
                                   "mc_se": float(se), "mean_minus_formula_in_mc_se": float(z)}
        errors.append(rel.max())
        z_scores.append(z)
    out["gating"] = {"by_kappa": per_kappa, "max_relative_error": float(max(errors)),
                     "mean_minus_formula_in_mc_se": float(max(z_scores))}
    gaps = []
    for config_id, rows in markets.items():
        if not config_id.endswith("/none") and not config_id.startswith(("sensitivity/", "part3/")):
            continue
        for mech in by_market(rows).values():
            if mech["M3"]["excluded"]:
                continue
            gaps.append(abs(mech["M3"]["welfare"] - mech["M3_set_blind"]["welfare"]) / abs(mech["M3"]["welfare"]))
    out["setblind"] = {"markets": len(gaps), "max_relative_gap": float(max(gaps))}
    out["passed"] = bool(out["gating"]["max_relative_error"] <= GATING_TOL
                         and out["gating"]["mean_minus_formula_in_mc_se"] <= MC_SE
                         and out["setblind"]["max_relative_gap"] <= GAP_TOL)
    return out


# --------------------------------------------------------------- statistics
def _valid(rows: list[dict], mechanism: str, screen: bool = False, feasible: bool = False) -> list[dict]:
    return [r for r in rows if r["mechanism"] == mechanism and not r["excluded"]
            and (not screen or r["screen"]) and (not feasible or r["feasible"])]


def psi_a(r: dict) -> float:
    """ψ_A = Σ_j P^A_j / (n E_μ[α] θ̄) (A12 §3.3)."""
    return r["revenue_agent"] / (1000 * r["alpha"] * r["theta_bar"])


def psi_tot(r: dict) -> float:
    """ψ_tot = Σ_j P_j / (n θ̄)."""
    return r["revenue"] / (1000 * r["theta_bar"])


def statistics(markets: dict, gating: dict, prefix: str = "") -> dict:
    """The registered statistics; ``prefix`` keys the bootstrap streams of a subset."""
    core = {k: v for k, v in markets.items() if k.startswith("core/")}
    h2a = {"label": "exploratory", "part1": {}, "part2": {}, "part3": {}}
    # part 1: zero agent payments under agent-aware lists (screen-passing, aware-feasible markets), Holm
    keys, p_support, p_against = [], [], []
    for alpha in em.ALPHAS[1:]:
        for variation in em.VARIATIONS:
            cid = "core/alpha=%g/agent_aware/%s" % (alpha, variation)
            rows = _valid(core[cid], "M3", screen=True, feasible=True)
            if len(rows) < 2:  # too few screen-passing, aware-feasible markets for an interval
                h2a["part1"][cid] = {"n": len(rows), "insufficient": True}
                continue
            values = np.array([psi_a(r) for r in rows])
            t = tost(boot_mean(values, prefix + cid + "/psi_a"))
            h2a["part1"][cid] = {"n": len(rows), "mean_psi_a": float(values.mean()), **t,
                                 "below_registered_floor": len(rows) < FLOOR}
            if len(rows) < FLOOR:
                h2a["part1"][cid]["label_by_floor_rule"] = "inconclusive"
            keys.append(cid)
            p_support.append(t["p_support"])
            p_against.append(t["p_against"])
    for cid, ps, pa in zip(keys, holm(p_support), holm(p_against)):
        h2a["part1"][cid].update(p_support_holm=ps, p_against_holm=pa)
    cells = [core["core/alpha=%g/agent_aware/%s" % (a, v)] for a in em.ALPHAS[1:] for v in em.VARIATIONS]
    passing = sum(len(_valid(c, "M3", screen=True, feasible=True)) for c in cells)
    valid = sum(len(_valid(c, "M3")) for c in cells)
    h2a["screen_pass"] = {"passing": passing, "valid": valid, "rate": passing / valid,
                          "range": [min(len(_valid(c, "M3", screen=True, feasible=True)) / len(_valid(c, "M3"))
                                        for c in cells),
                                    max(len(_valid(c, "M3", screen=True, feasible=True)) / len(_valid(c, "M3"))
                                        for c in cells)]}
    # part 2: total payments against alpha, agent-aware
    rows = [r for cid, v in core.items() if "/agent_aware/" in cid for r in _valid(v, "M3", screen=True, feasible=True)]
    x = np.array([r["alpha"] for r in rows])
    y = np.array([psi_tot(r) for r in rows])
    strata = np.array([r["config_id"] for r in rows])
    slope, intercept, draws = boot_slope(x, y, strata, prefix + "h2a/part2")
    at1 = np.array([psi_tot(r) for r in rows if r["alpha"] == 1.0])
    h2a["part2"] = {"n": len(rows), "slope": slope, "slope_bound": bound(draws[:, 0], 0.0, "below"),
                    "psi_tot_at_alpha_1": float(at1.mean()), "psi_tot_at_alpha_1_tost": tost(boot_mean(at1, prefix + "h2a/at1"))}
    # part 3: agent payments against the excess, human-optimised (core none + part3 levels)
    rows = [r for cid, v in markets.items() if ("/human_optimised/none" in cid or cid.startswith("part3/"))
            for r in _valid(v, "M3") if r["alpha"] > 0]
    x = np.array([r["excess"] for r in rows], dtype=float)
    y = np.array([psi_a(r) for r in rows])
    slope, intercept, draws = boot_slope(x, y, np.array([r["config_id"] for r in rows]), prefix + "h2a/part3")
    h2a["part3"] = {"n": len(rows), "excess_values": sorted({int(v) for v in x}), "slope": slope,
                    "slope_bound": bound(draws[:, 0], 0.0, "above")}
    # H2b: Conj gating increments
    conj = {alpha: {r["index"]: r for r in gating["gating/conj/alpha=%g" % alpha] if not r["excluded"]}
            for alpha in em.ALPHAS}
    common = sorted(set.intersection(*[set(v) for v in conj.values()]))
    delta = {alpha: np.array([conj[alpha][i]["gain"] for i in common]) for alpha in em.ALPHAS}
    tau = 0.01 * float(delta[1.0].mean())
    increments, ps, pa = [], [], []
    for a, b in zip(em.ALPHAS, em.ALPHAS[1:]):
        diff = delta[b] - delta[a]
        t = bound(boot_mean(diff, prefix + "h2b/%g-%g" % (a, b)), -tau, "above")
        increments.append({"from": a, "to": b, "mean": float(diff.mean()), **t})
        ps.append(t["p_support"])
        pa.append(t["p_against"])
    for inc, p1, p2 in zip(increments, holm(ps), holm(pa)):
        inc.update(p_support_holm=p1, p_against_holm=p2)
    h2b = {"label": "exploratory", "markets": len(common), "tau": tau,
           "mean_gain": {"%g" % a: float(delta[a].mean()) for a in em.ALPHAS}, "increments": increments}
    # H2c
    rows = [r for v in core.values() for r in _valid(v, "M2")]
    x = np.array([1 - r["alpha"] for r in rows])
    y = np.array([psi_tot(r) for r in rows])
    slope, intercept, draws = boot_slope(x, y, np.array([r["config_id"] for r in rows]), prefix + "h2c/m2")
    m2 = {"n": len(rows), "slope": slope, "intercept": intercept, "slope_bound": bound(draws[:, 0], 0.0, "above"),
          "intercept_tost": tost(draws[:, 1])}
    chi_configs = {0.5: ["core/alpha=%g/agent_aware/none" % a for a in em.ALPHAS],
                   0.0: ["sensitivity/chi=0/alpha=%g" % a for a in em.ALPHAS],
                   1.0: ["sensitivity/chi=1/alpha=%g" % a for a in em.ALPHAS]}
    rows = [(chi, r) for chi, cids in chi_configs.items() for cid in cids for r in _valid(markets[cid], "M1")
            if r["payment_per_choice"] == r["payment_per_choice"]]
    x = np.array([r["alpha"] * chi for chi, r in rows])
    y = np.array([r["payment_per_choice"] / r["theta_bar"] for _, r in rows])
    slope, intercept, draws = boot_slope(x, y, np.array([r["config_id"] for _, r in rows]), prefix + "h2c/m1")
    m1 = {"n": len(rows), "slope": slope, "slope_bound": bound(draws[:, 0], 0.0, "above")}
    rows = [r for v in core.values() for r in _valid(v, "M3")]
    blind = {(r["config_id"], r["index"]): r for v in core.values() for r in _valid(v, "M3_set_blind")}
    x = np.array([r["variation"] == "randomised" for r in rows], dtype=float)
    y = np.array([(r["welfare"] - blind[(r["config_id"], r["index"])]["welfare"]) / r["welfare"] for r in rows])
    slope, intercept, draws = boot_slope(x, y, np.array([r["config_id"] for r in rows]), prefix + "h2c/gap")
    gap = {"n": len(rows), "slope_on_variation": slope, "slope_bound": bound(draws[:, 0], 0.0, "above")}
    return {"H2a": h2a, "H2b": h2b, "H2c": {"label": "exploratory", "m2_proportional": m2,
                                             "m1_rises_with_alpha_chi": m1, "setblind_gap_grows": gap}}


# ------------------------------------------------------------- tables, figs
def means_table(markets: dict) -> list[dict]:
    """Means by configuration and mechanism, for all markets and the paper-feasible subset."""
    out = []
    for subset, data in zip(SUBSETS, (markets, paper_feasible(markets))):
        prefix = "" if subset == "all" else subset + "/"
        for cid, rows in data.items():
            for mech in MECHANISMS:
                v = _valid(rows, mech)
                if v:
                    out.append(_means_row(cid, subset, mech, v, prefix))
    return out


def _means_row(cid: str, subset: str, mech: str, v: list[dict], prefix: str) -> dict:
    row = {"config_id": cid, "subset": subset, "mechanism": mech, "markets": len(v)}
    for metric in ("revenue", "revenue_human", "revenue_agent", "contribution", "cost", "match", "slack", "profit"):
        values = np.array([r[metric] for r in v]) / 1000
        draws = boot_mean(values, prefix + "%s/%s/%s" % (cid, mech, metric))
        row[metric] = float(values.mean())
        row[metric + "_ci"] = interval(draws)
    return row


def paired_table(markets: dict) -> list[dict]:
    """Paired differences between adjacent mechanisms, for all markets and the paper-feasible subset."""
    out = []
    chain = ("M1", "M2", "M3_set_blind", "M3", "M4")
    for subset in SUBSETS:
        prefix = "" if subset == "all" else subset + "/"
        for cid, rows in markets.items():
            if not cid.startswith("core/"):
                continue
            idx = {i: m for i, m in by_market(rows).items() if not m["M1"]["excluded"]
                   and (subset == "all" or not m["M1"]["floor_binds"])}
            if not idx:
                continue
            for a, b in zip(chain, chain[1:]):
                for metric in ("profit", "revenue", "match"):
                    diff = np.array([m[b][metric] - m[a][metric] for m in idx.values()]) / 1000
                    out.append({"config_id": cid, "subset": subset, "from": a, "to": b, "metric": metric,
                                "markets": len(idx), "mean": float(diff.mean()),
                                "ci95": interval(boot_mean(diff, prefix + "%s/%s-%s/%s" % (cid, a, b, metric)))})
    return out


def figures(markets: dict, gating: dict, means: list[dict], stats: dict) -> list[str]:
    paths = []
    lookup = {(r["config_id"], r["subset"], r["mechanism"]): r for r in means}
    styles = {"M1": "C0", "M2": "C1", "M3_set_blind": "C2", "M3": "C3", "M4": "C4", "profit_rule": "C5"}
    binds = {regime: floor_share(markets, "core/alpha=%%g/%s/none" % regime) for regime in em.REGIMES}

    def panel_title(regime: str, text: str) -> str:
        return "%s, %s\n(floor binds in %.0f%% of markets)" % (regime.replace("_", "-"), text, 100 * binds[regime])

    def plot_mechanisms(ax, config, metric, rows, tag):
        for mech in MECHANISMS:
            ys = [lookup[(config % a, "all", mech)][metric] for a in em.ALPHAS]
            ax.plot(em.ALPHAS, ys, marker="o", ms=3, color=styles[mech], label=mech.replace("M3_set_blind", "set-blind M3"),
                    linestyle="--" if mech == "M3_set_blind" else "-", zorder=3 if mech == "M3_set_blind" else 2)
            rows += [(*tag, mech, a, y) for a, y in zip(em.ALPHAS, ys)]

    all_note = ("All valid markets, including those where the guardrail floor binds (A17 §5.1); the paper-feasible "
                "subset is in tables/means.csv.\nSet-blind M3 (dashed) equals M3 without organic variation.")
    # 1 payments by traffic type
    fig, axes = figure(2, 3, 9.0, 5.6)
    rows = []
    for col, regime in enumerate(em.REGIMES):
        for row_k, (metric, name) in enumerate((("revenue_agent", "agent"), ("revenue_human", "human"))):
            ax = axes[row_k][col]
            if regime == "agent_aware" and name == "agent":  # split by the A1 §E5 screen in figure 7 (A17 §5.2)
                ax.axis("off")
                ax.text(0.5, 0.5, "agent-aware, agent payments:\nsee figure 7, split by the\nA1 §E5 screen, with the "
                        "δ = 0 level", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                continue
            plot_mechanisms(ax, "core/alpha=%%g/%s/none" % regime, metric, rows, (regime, name))
            ax.set_title(panel_title(regime, "%s payments" % name))
            ax.set_xlabel("agent share α")
            ax.set_ylabel("payments per request")
    paths += save(fig, "sl19s", FIGURE_NAMES[0], ["regime", "traffic", "mechanism", "alpha", "payment_per_request"],
                  rows, "Payments by traffic type against α (no organic variation; all markets)", legend=6,
                  note=all_note)
    # 2 profit and match quality
    fig, axes = figure(2, 3, 9.0, 5.6)
    rows = []
    for col, regime in enumerate(em.REGIMES):
        for row_k, metric in enumerate(("profit", "match")):
            ax = axes[row_k][col]
            plot_mechanisms(ax, "core/alpha=%%g/%s/none" % regime, metric, rows, (regime, metric))
            ax.set_title(panel_title(regime, "profit" if metric == "profit" else "match quality"))
            ax.set_xlabel("agent share α")
            ax.set_ylabel("per request")
    paths += save(fig, "sl19s", FIGURE_NAMES[1], ["regime", "metric", "mechanism", "alpha", "value_per_request"], rows,
                  "Profit and match quality against α (no organic variation; all markets)", legend=6, note=all_note)
    # 3 decomposition
    chain = ("M1", "M2", "M3_set_blind", "M3", "M4")
    fig, axes = figure(1, 3, 9.0, 3.8)
    rows = []
    for col, regime in enumerate(em.REGIMES):
        ax = axes[0][col]
        for a in em.ALPHAS:
            ys = [lookup[("core/alpha=%g/%s/none" % (a, regime), "all", m)]["profit"] for m in chain]
            ax.plot(range(len(chain)), ys, marker="o", ms=3, label="α = %g" % a)
            rows += [(regime, a, m, y) for m, y in zip(chain, ys)]
        ax.set_xticks(range(len(chain)), [m.replace("M3_set_blind", "set-blind M3") for m in chain], rotation=30)
        ax.set_title(panel_title(regime, "profit"))
        ax.set_ylabel("profit per request")
    paths += save(fig, "sl19s", FIGURE_NAMES[2], ["regime", "alpha", "mechanism", "profit_per_request"], rows,
                  "Decomposition M1 → M2 → set-blind M3 → M3 → M4 (profit, no organic variation; all markets)",
                  legend=5, note="All valid markets, including those where the guardrail floor binds (A17 §5.1); "
                                 "the paper-feasible subset is in tables/means.csv.")
    # 4 set-blind gap, all markets and the paper-feasible subset
    fig, axes = figure(1, 1, 5.6, 3.8)
    ax = axes[0][0]
    rows = []
    for k, regime in enumerate(em.REGIMES):
        for v, variation in enumerate(em.VARIATIONS):
            for s, subset in enumerate(SUBSETS):
                values = []
                for a in em.ALPHAS:
                    cid = "core/alpha=%g/%s/%s" % (a, regime, variation)
                    idx = {i: m for i, m in by_market(markets[cid]).items() if not m["M3"]["excluded"]
                           and (subset == "all" or not m["M3"]["floor_binds"])}
                    values += [(m["M3"]["welfare"] - m["M3_set_blind"]["welfare"]) / m["M3"]["welfare"]
                               for m in idx.values()]
                values = np.array(values)
                key = "fig4/%s/%s" % (regime, variation) + ("" if subset == "all" else "/" + subset)
                ci = interval(boot_mean(values, key))
                ax.errorbar(v + 0.12 * (k - 1) + 0.04 * (2 * s - 1), values.mean(), yerr=_yerr([values.mean()], [ci]),
                            fmt="o" if subset == "all" else "s", mfc="C%d" % k if subset == "all" else "white",
                            color="C%d" % k, capsize=2, ms=4,
                            label="%s, %s" % (regime.replace("_", "-"), SUBSET_NAMES[subset]) if v == 0 else None)
                rows.append((regime, variation, subset, float(values.mean()), ci[0], ci[1], len(values)))
    ax.set_xticks([0, 1], ["none (deterministic)", "randomised"])
    ax.set_xlabel("organic variation")
    ax.set_ylabel("(V^aw − V^bl) / V^aw")
    paths += save(fig, "sl19s", FIGURE_NAMES[3], ["regime", "variation", "subset", "mean_relative_gap", "ci_low",
                                                 "ci_high", "markets"], rows,
                  "Set-awareness gap against organic variation", legend=2,
                  note="Filled: all markets; open: the paper-feasible subset, where the floor does not bind (A17 §5.1).")
    # 5 gating gain against alpha, on the markets valid at every α (the H2b sample)
    fig, axes = figure(1, 1, 5.0, 3.4)
    ax = axes[0][0]
    h2b = stats["H2b"]
    conj = {a: {r["index"]: r for r in gating["gating/conj/alpha=%g" % a] if not r["excluded"]} for a in em.ALPHAS}
    common = sorted(set.intersection(*[set(v) for v in conj.values()]))
    means = [h2b["mean_gain"]["%g" % a] / 1000 for a in em.ALPHAS]
    cis = [interval(boot_mean(np.array([conj[a][i]["gain"] for i in common]) / 1000, "fig5/%g" % a)) for a in em.ALPHAS]
    ax.errorbar(em.ALPHAS, means, yerr=_yerr(means, cis), fmt="o-", capsize=3)
    ax.set_xlabel("agent share α")
    ax.set_ylabel("ΔΠ(α) = Π_gated − Π_aware, per request")
    paths += save(fig, "sl19s", FIGURE_NAMES[4], ["alpha", "mean_gain_per_request", "ci_low", "ci_high", "markets"],
                  [(a, m, c[0], c[1], len(common)) for a, m, c in zip(em.ALPHAS, means, cis)],
                  "Gating gain against α (Conj gating, κ = %g)" % em.CONJ_KAPPA,
                  note="The %d markets valid at every α; the paper's guardrail (the floor does not apply)." % len(common))
    # 6 agent payments against excess, all markets and the paper-feasible subset
    fig, axes = figure(1, 1, 5.0, 3.6)
    ax = axes[0][0]
    rows = [r for cid, v in markets.items() if ("/human_optimised/none" in cid or cid.startswith("part3/"))
            for r in _valid(v, "M3") if r["alpha"] > 0]
    table = []
    for s, subset in enumerate(SUBSETS):
        for e in sorted({r["excess"] for r in rows}):
            values = np.array([psi_a(r) for r in rows if r["excess"] == e and (subset == "all" or not r["floor_binds"])])
            ci = interval(boot_mean(values, "fig6/%d" % e + ("" if subset == "all" else "/" + subset)))
            ax.errorbar(e + 0.08 * (2 * s - 1), values.mean(), yerr=_yerr([values.mean()], [ci]),
                        fmt="o" if subset == "all" else "s", mfc="C3" if subset == "all" else "white", capsize=3,
                        color="C3", label=SUBSET_NAMES[subset] if e == min(r["excess"] for r in rows) else None)
            table.append((subset, int(e), float(values.mean()), ci[0], ci[1], len(values)))
    ax.set_xlabel("e(x) = |F(x)| − (K − K^s)")
    ax.set_ylabel("ψ_A under M3")
    paths += save(fig, "sl19s", FIGURE_NAMES[5], ["subset", "excess", "mean_psi_a", "ci_low", "ci_high", "markets"],
                  table, "Agent payments against excess favourites (human-optimised lists)", legend=2)
    # 7 agent payments under agent-aware lists by the A1 §E5 screen, with the δ = 0 level (A17 §5.2)
    fig, axes = figure(2, 3, 10.4, 6.2)
    rows = []
    alphas = em.ALPHAS[1:]
    columns = (("screen_pass", "passes the A1 §E5 screen", "core/alpha=%g/agent_aware/none", True),
               ("screen_fail", "fails the screen", "core/alpha=%g/agent_aware/none", False),
               ("delta_0", "δ = 0 level (all pass)", "sensitivity/delta=0/alpha=%g", None))
    for row_k, subset in enumerate(SUBSETS):
        for col, (group, text, config, screen) in enumerate(columns):
            ax = axes[row_k][col]
            ax.axhspan(-MARGIN, MARGIN, color="0.88", zorder=0, label="H2a margin ±0.01")
            for mech in MECHANISMS:
                ys, cis, ns = [], [], []
                for a in alphas:
                    values = np.array([psi_a(r) for r in _valid(markets[config % a], mech, feasible=True)
                                       if (screen is None or bool(r["screen"]) == screen)
                                       and (subset == "all" or not r["floor_binds"])])
                    ns.append(len(values))
                    ys.append(float(values.mean()) if len(values) else float("nan"))
                    cis.append(interval(boot_mean(values, "fig7/%s/%s/%s/%g" % (subset, group, mech, a)))
                               if len(values) > 1 else [float("nan")] * 2)
                ax.plot(alphas, ys, marker="o", ms=3, color=styles[mech], zorder=3 if mech == "M3_set_blind" else 2,
                        linestyle="--" if mech == "M3_set_blind" else "-",
                        label=mech.replace("M3_set_blind", "set-blind M3"))
                rows += [(subset, group, mech, a, y, c[0], c[1], n) for a, y, c, n in zip(alphas, ys, cis, ns)]
            ax.set_title("%s\n%s; markets per α: %s" % (text, SUBSET_NAMES[subset], "/".join(str(n) for n in ns)),
                         fontsize=8)
            ax.set_xlabel("agent share α")
            ax.set_ylabel("ψ_A")
    paths += save(fig, "sl19s", FIGURE_NAMES[6], ["subset", "group", "mechanism", "alpha", "mean_psi_a", "ci_low",
                                                 "ci_high", "markets"], rows,
                  "Agent payments under agent-aware lists, by the A1 §E5 screen (no organic variation)", legend=7,
                  note="ψ_A: agent payments in θ̄ per agent request, under each mechanism; aware-feasible markets. Top "
                       "row: all markets;\nbottom row: the paper-feasible subset, where the guardrail floor does not "
                       "bind (A17 §5.1). 95% intervals are in the CSV.")
    return [str(p.relative_to(REPO)) for p in paths]


def _yerr(means, intervals) -> list[list[float]]:
    """Error-bar lengths from percentile intervals (0 where rounding puts a mean on its bound)."""
    return [[max(0.0, m - c[0]) for m, c in zip(means, intervals)], [max(0.0, c[1] - m) for m, c in zip(means, intervals)]]


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    text.write("# " + LABEL + "\n")
    keys = list(rows[0])
    writer.writerow(keys)
    for row in rows:
        writer.writerow([json.dumps(row[k]) if isinstance(row[k], list) else
                         repr(row[k]) if isinstance(row[k], float) else row[k] for k in keys])
    path.write_text(text.getvalue(), encoding="utf-8")


def markdown(summary: dict, means: list[dict], paired: list[dict]) -> str:
    c, g, s = summary["configurations"], summary["gates"], summary["statistics"]
    total = sum(v["markets"] for v in c.values())
    lines = ["# SL-19S E2 dry run on stipulated parameters", "", "**%s.** Results of "
             "`python -m features.storelab.analysis.e2_run` (master seed %d), regenerated by "
             "`python -m features.storelab.analysis.e2_report`. These are stand-ins (A17), not estimates of any "
             "agent, and no confirmatory label is recorded. The one label recorded is H2a part 1's *inconclusive*, which "
             "the A12 floor rule fixes whatever the data (owner decision 2026-10-24). Review status: accepted by the "
             "owner without independent review (2026-10-24); SL-24 is its independent audit, which also added the "
             "screen split, the δ = 0 level and the paper-feasible subset (A17 §5)." % (LABEL, em.MASTER_SEED), "",
             "## Design", "",
             "- %d configurations, %d markets, six mechanisms each (SL-16 solvers). Core: α × organic regime × organic "
             "variation at the A17 baseline (30). Sensitivity: the 24 A17 levels E2 uses, over α in the agent-aware regime "
             "without organic variation (120). Part 3 of H2a: the model and version levels in the human-optimised regime "
             "(15). Owner decisions 2026-10-23." % (len(c), total),
             "- Guardrail floor (owner decision 2026-10-23, A17 §5.1): M(Q) ≥ min{(1 − κ) M^ref, M(Q^o, ∅) − 1e-8 n}; "
             "it binds in %d of %d markets (core, no organic variation: %s). Where it binds, gated-regime results "
             "describe the floor, not the paper's mechanism, so every table and figure it affects reports the "
             "*paper-feasible subset* (markets where it does not bind) next to all markets, or states which one it "
             "shows." % (sum(v["floor_binds"] for v in c.values()), sum(v["valid"] for v in c.values()),
                         ", ".join("%s %.0f%%" % (k.replace("_", "-"), 100 * v)
                                   for k, v in summary["floor_share"].items())),
             "- Prop gating and Conj gating use the paper's guardrail: 200 markets per κ ∈ {0.05, 0.2}, and Conj "
             "gating at κ = 0.05 over α on the %d markets whose gated design meets the paper's guardrail at every α "
             "(drawn until at least 200; at α = 0 and 0.25 the gated list often cannot meet it)." % (
                 s["H2b"]["markets"]),
             "- A configuration with solver exclusions is topped up with further markets (index 200, 201, ...) "
             "until 200 are valid; `e2_run` does this itself, so a seeded rerun reproduces it.", "",
             "## Markets (AC-1)", "",
             "Every configuration: %d–%d valid markets, at most %d excluded (solver failure) per configuration, one "
             "input hash per market across the six mechanisms in every configuration: %s. Screen-failing (A1 §E5) "
             "and aware-infeasible (`cor:aware`) markets are classified and left out of H2a part 1 (counts in "
             "`e2_stipulated_results.json`). QP solves that needed SL-16's fallback path: %d." % (
                 min(v["valid"] for v in c.values()), max(v["valid"] for v in c.values()),
                 max(v["excluded"] for v in c.values()),
                 "yes" if all(v["one_hash_per_market"] for v in c.values()) else "NO",
                 sum(v["qp_fallbacks"] for v in c.values())), "",
             "## Implementation gates (AC-3)", "",
             "| Gate | Result | Tolerance |", "|---|---|---|"]
    for kappa, v in g["gating"]["by_kappa"].items():
        lines.append("| Prop gating, κ = %s: max relative error of Π_b − Π_a against n ξ min{θ_f, θ_r} (%d markets) "
                     "| %.1e | 1e-6 |" % (kappa, v["markets"], v["max_relative_error"]))
        lines.append("| Prop gating, κ = %s: mean gain / (n ξ) %.4f against E min = 1/3 | %.2f MC SE | 3 MC SE |" % (
            kappa, v["mean_gain_over_n_xi"], v["mean_minus_formula_in_mc_se"]))
    lines += ["| Set-blind gap with deterministic organic lists, max (V^aw − V^bl)/V^aw (%d markets) | %.1e | 1e-6 |" % (
        g["setblind"]["markets"], g["setblind"]["max_relative_gap"]), "",
        "Both gates pass." if g["passed"] else "**A GATE FAILED: the report is blocked.**", "",
        "## Registered statistics (exploratory, no confirmatory label)", "", "*%s.*" % LABEL, "",
        "**H2a part 1** (agent-aware; ψ_A under M3; screen-passing, aware-feasible markets; Holm over the 8 "
        "configurations):", "", "| Configuration | n | mean ψ_A | 90% interval | 95% interval | p (inside margin), Holm | "
        "p (outside margin), Holm |", "|---|---:|---:|---|---|---:|---:|"]
    for cid, v in s["H2a"]["part1"].items():
        if v.get("insufficient"):
            lines.append("| %s | %d | — | — | — | — | — |" % (cid, v["n"]))
            continue
        lines.append("| %s | %d | %.4f | [%.4f, %.4f] | [%.4f, %.4f] | %.3f | %.3f |" % (
            cid, v["n"], v["mean_psi_a"], *v["ci90"], *v["ci95"], v["p_support_holm"], v["p_against_holm"]))
    sp = s["H2a"]["screen_pass"]
    lines += ["", "**H2a part 1 is inconclusive by rule** in every configuration below the floor of %d "
              "screen-passing markets (A12 §2: inconclusive whenever the minimum sample size is not met; recorded "
              "without recomputation, owner decision 2026-10-24). The intervals above are reported as computed." % FLOOR,
              "", "**Screen pass rate.** %d of %d valid markets (%.1f%%; %.1f–%.1f%% per configuration) pass the A1 §E5 "
              "screen and are aware-feasible. This is itself an exploratory result of the dry run: under the stipulated "
              "label penalty (δ = (−0.3, −0.7)), the zero-agent-payment prediction of `prop:rents` applies only in a "
              "minority of markets. The rest fail the screen (a sponsored favourite would lose to its rival, so bids "
              "change agent outcomes) or are aware-infeasible, and are reported separately as boundary cases." % (sp["passing"], sp["valid"], 100 * sp["rate"], 100 * sp["range"][0],
                                               100 * sp["range"][1])]
    p2, p3 = s["H2a"]["part2"], s["H2a"]["part3"]
    lines += ["", "**H2a part 2**: slope of ψ_tot on α %.4f, 95%% interval [%.4f, %.4f] (n = %d); ψ_tot(α = 1) = %.4f, "
              "90%% interval [%.4f, %.4f]." % (p2["slope"], *p2["slope_bound"]["ci95"], p2["n"], p2["psi_tot_at_alpha_1"],
                                              *p2["psi_tot_at_alpha_1_tost"]["ci90"]),
              "", "**H2a part 3**: slope of ψ_A on e(x) %.4f, 95%% interval [%.4f, %.4f] (n = %d, e(x) ∈ %s)." % (
                  p3["slope"], *p3["slope_bound"]["ci95"], p3["n"], p3["excess_values"]), ""]
    h2b = s["H2b"]
    lines += ["**H2b, Conj gating** (%d markets, τ = %.3f = 1%% of ΔΠ(1); Holm over the 4 increments):" % (
        h2b["markets"], h2b["tau"]), "", "| α step | mean increment of ΔΠ | 95% interval | p (≥ −τ), Holm |",
        "|---|---:|---|---:|"]
    for inc in h2b["increments"]:
        lines.append("| %g → %g | %.3f | [%.3f, %.3f] | %.3f |" % (inc["from"], inc["to"], inc["mean"], *inc["ci95"],
                                                               inc["p_support_holm"]))
    h2c = s["H2c"]
    lines += ["", "**H2c**: M2 revenue (ψ units) on E[1 − α]: slope %.4f, 95%% interval [%.4f, %.4f]; intercept %.4f, "
              "90%% interval [%.4f, %.4f]. M1 payment per effective choice (θ̄ units) on αχ̄: slope %.4f, 95%% interval "
              "[%.4f, %.4f]. Set-awareness gap on organic variation: slope %.4f, 95%% interval [%.4f, %.4f]." % (
                  h2c["m2_proportional"]["slope"], *h2c["m2_proportional"]["slope_bound"]["ci95"],
                  h2c["m2_proportional"]["intercept"], *h2c["m2_proportional"]["intercept_tost"]["ci90"],
                  h2c["m1_rises_with_alpha_chi"]["slope"], *h2c["m1_rises_with_alpha_chi"]["slope_bound"]["ci95"],
                  h2c["setblind_gap_grows"]["slope_on_variation"], *h2c["setblind_gap_grows"]["slope_bound"]["ci95"]),
              ""]
    lines += paper_feasible_section(s, summary["statistics_paper_feasible"])
    lines += ["", "## Core means by mechanism (per request, no organic variation)", "", "*%s.*" % LABEL, "",
              "All markets, and (last four columns) the paper-feasible subset, where the guardrail floor does not bind "
              "(A17 §5.1).", "",
              "| α | Regime | Mechanism | Payments | Agent | Human | Profit | Match quality | Guardrail slack "
              "| Paper-feasible markets | Agent | Profit | Match quality |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    feasible = {(r["config_id"], r["mechanism"]): r for r in means if r["subset"] == "paper_feasible"}
    for r in means:
        if r["subset"] != "all" or not (r["config_id"].startswith("core/") and r["config_id"].endswith("/none")):
            continue
        _, a, regime, _ = r["config_id"].split("/")
        f = feasible.get((r["config_id"], r["mechanism"]))
        tail = "| %d | %.4f | %.4f | %.4f |" % (f["markets"], f["revenue_agent"], f["profit"], f["match"]) if f else \
            "| 0 | — | — | — |"
        lines.append("| %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f %s" % (
            a.split("=")[1], regime, r["mechanism"], r["revenue"], r["revenue_agent"], r["revenue_human"], r["profit"],
            r["match"], r["slack"], tail))
    lines += ["", "Full tables (every configuration and both subsets, with 95% intervals; paired differences between "
              "adjacent mechanisms): `artifacts/results/sl19s/tables/`.", "", "## Figures", "", "*%s.*" % LABEL, ""]
    pdfs = [p for p in summary["figures"] if p.endswith(".pdf")]
    lines += ["- `%s`: %s" % (p, FIGURE_CAPTIONS[Path(p).stem]) for p in pdfs]
    return "\n".join(lines) + "\n"


FIGURE_CAPTIONS = {
    "1_payments_by_traffic_type": "all markets (the share where the floor binds is in each panel title); agent-aware "
                                  "agent payments are in figure 7",
    "2_profit_and_match_quality": "all markets (the share where the floor binds is in each panel title)",
    "3_decomposition": "all markets (the share where the floor binds is in each panel title)",
    "4_setblind_gap": "all markets and the paper-feasible subset",
    "5_gating_gain": "the Conj gating markets valid at every α, under the paper's guardrail (the floor does not apply)",
    "6_agent_payments_vs_excess": "all markets and the paper-feasible subset",
    "7_agent_payments_by_screen": "agent-aware lists: screen-passing and screen-failing markets separately, with the "
                                  "δ = 0 level (A17 §5.2); all markets and the paper-feasible subset",
}


def paper_feasible_section(s: dict, f: dict) -> list[str]:
    """The registered statistics on all markets and on the paper-feasible subset (A17 §5.1)."""
    def ci(v):
        return "%.4f [%.4f, %.4f]" % (v["slope"], *v["slope_bound"]["ci95"])

    rows = [("H2a part 2: slope of ψ_tot on α", ci(s["H2a"]["part2"]), s["H2a"]["part2"]["n"], ci(f["H2a"]["part2"]),
             f["H2a"]["part2"]["n"]),
            ("H2a part 3: slope of ψ_A on e(x)", ci(s["H2a"]["part3"]), s["H2a"]["part3"]["n"], ci(f["H2a"]["part3"]),
             f["H2a"]["part3"]["n"]),
            ("H2c: M2 revenue on E[1 − α]", ci(s["H2c"]["m2_proportional"]), s["H2c"]["m2_proportional"]["n"],
             ci(f["H2c"]["m2_proportional"]), f["H2c"]["m2_proportional"]["n"]),
            ("H2c: M1 payment per effective choice on αχ̄", ci(s["H2c"]["m1_rises_with_alpha_chi"]),
             s["H2c"]["m1_rises_with_alpha_chi"]["n"], ci(f["H2c"]["m1_rises_with_alpha_chi"]),
             f["H2c"]["m1_rises_with_alpha_chi"]["n"])]
    gap_all, gap_f = s["H2c"]["setblind_gap_grows"], f["H2c"]["setblind_gap_grows"]
    rows.append(("H2c: set-awareness gap on organic variation",
                 "%.4f [%.4f, %.4f]" % (gap_all["slope_on_variation"], *gap_all["slope_bound"]["ci95"]), gap_all["n"],
                 "%.4f [%.4f, %.4f]" % (gap_f["slope_on_variation"], *gap_f["slope_bound"]["ci95"]), gap_f["n"]))
    out = ["**Paper-feasible subset (A17 §5.1).** The statistics above use all markets. The same statistics on the "
           "markets where the guardrail floor does not bind (H2a part 1 there: %s screen-passing, aware-feasible "
           "markets per configuration, all below the floor of %d). H2b uses the paper's guardrail, so the floor does "
           "not apply to it." % ("–".join(str(x) for x in (
               min(v["n"] for v in f["H2a"]["part1"].values()), max(v["n"] for v in f["H2a"]["part1"].values()))),
               FLOOR), "",
           "| Statistic | All markets: estimate [95% interval] | n | Paper-feasible: estimate [95% interval] | n |",
           "|---|---|---:|---|---:|"]
    out += ["| %s | %s | %d | %s | %d |" % row for row in rows]
    return out


def main(argv: list[str] | None = None) -> int:
    markets, gating = load()
    summary = {"label": LABEL, "master_seed": em.MASTER_SEED, "configurations": counts(markets)}
    summary["gates"] = gates(markets, gating)
    if not summary["gates"]["passed"]:
        print(json.dumps(summary["gates"], indent=1))
        raise GateFailure("an implementation gate failed; no report is written (SL-19S AC-3)")
    summary["statistics"] = statistics(markets, gating)
    summary["statistics_paper_feasible"] = statistics(paper_feasible(markets), gating, "paper_feasible/")
    summary["floor_share"] = {regime: floor_share(markets, "core/alpha=%%g/%s/none" % regime) for regime in em.REGIMES}
    means, paired = means_table(markets), paired_table(markets)
    _write_csv(TABLES / "means.csv", means)
    _write_csv(TABLES / "paired_differences.csv", paired)
    summary["figures"] = figures(markets, gating, means, summary["statistics"])
    check = er.RESULTS / "rerun_check.json"
    summary["rerun_identical"] = json.loads(check.read_text(encoding="utf-8"))["identical"] if check.exists() else None
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(markdown(summary, means, paired), encoding="utf-8")
    print("report written; gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
