"""SL-20S report: tables, figures and ``docs/e3_stipulated_results.md`` from the result files.

``python -m features.storelab.analysis.e3_report`` reads ``artifacts/results/sl20s/`` (written by
``e3_run``) and regenerates everything; no solver runs.

**Gates** (a failure blocks the report, and Part B is blocked by Part A):
- Part A: 100% of the A17 §3.1 grid points pass the A12 §4 H3b tolerances (quadratic:
  |y − y*| ≤ 1e-4 and objective relative error ≤ 1e-6; overflow limit: objective relative
  error ≤ 1e-6; overflow finite n: f(y*) − f(y_n) ≤ 2λ/√n + 1e-6);
- Example ex:diversify: Var(D_A) = 250,000, 250 and 62,625 (125 + 62,500) through the E3 code
  path, by Lemma load and from the exact distribution, within 1e-9 relative, and its separated
  version is {A, C} or {B, C};
- Part B, not-aligned states with α = 1 (the hypotheses of Prop separation): the separated and
  quota policies match the original's expected loads, contributions and match quality within
  1e-9 relative, with across-state variance n² Var_Λ(p_j) ≤ 1e-12, in the deterministic limit
  (A12 §4, H3a I-part);
- every Part B row: within + across = total variance within 1e-9 relative (at σ = 0.3 and in
  the deterministic limit), and Var_Λ(p_j) ≤ Θ_j² / 4 in the deterministic limit (Lemma contested).

**Registered statistics** (A12 §4: H3a, H3b, H3c) are computed as written and labelled
exploratory; no confirmatory label is recorded (A17 §0). Intervals are percentile bootstraps over
markets (B = 2,000) with seeds from the master seed.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import numpy as np

from ..core.rng import stream
from . import e2_run as er
from . import e3
from . import e3_run as e3r
from .e2_report import _num, interval
from .figures import LABEL, REPO, figure, save

DOCS = Path(__file__).resolve().parents[1] / "docs"
REPORT = DOCS / "e3_stipulated_results.md"
SUMMARY = DOCS / "e3_stipulated_results.json"
TABLES = e3r.RESULTS / "tables"
B = 2000
PART_A_NOTE = "E3 Part A: implementation check of prop:threshold (closed forms; no stipulated parameters)"
PART_A_LABEL = "%s · %s" % (LABEL, PART_A_NOTE)  # tables; figures carry LABEL with PART_A_NOTE beneath (SL-24)
N_COLORS = {10: "C0", 100: "C1", 1000: "C2", 10000: "C3"}
OMEGA_MARKERS = {0.3: ("o", 70), 0.5: ("s", 44), 0.8: ("^", 26), 1.0: ("D", 10)}  # nested: all visible where equal
FIGURE_NAMES = ("1_numeric_vs_closed_form", "2_y_star_vs_delta", "3_variance_decomposition", "4_cost_match_tradeoff",
                "5_h3c_buyback")
POLICY_NAMES = {"original": "original", "separated": "separated", "composition_blind": "composition-blind",
                "quota": "quota (separated)"}
EXACT_TOLERANCE = 1e-9
ACROSS_TOLERANCE = 1e-12
DECOMPOSITION_TOLERANCE = 1e-9
RATIO_THRESHOLD = 0.5  # A12 §4, H3a Pr-part
RATIO_SHARE = 2 / 3
COINCIDENCE = 0.95  # A12 §4, H3c under M4
BASE_COST = "cost_quadratic_w0.5_l0.5"


class GateFailure(RuntimeError):
    pass


def _rows(path: Path) -> list[dict]:
    return [{k: _num(v) for k, v in r.items()} for r in er.read_rows(path)]


def load() -> dict:
    out = e3r.RESULTS
    return {"part_a": _rows(out / "part_a.csv.gz"), "diversify": _rows(out / "diversify.csv.gz"),
            "part_b": {cid: _rows(er.result_path(out, cid)) for cid, _, _ in e3r.configurations()},
            "h3c": _rows(out / "h3c.csv.gz")}


def _rng(*keys):
    return stream(e3r.MASTER_SEED, "sl20s-bootstrap", *keys)


def boot_mean(values, key: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[_rng(key).integers(0, len(values), size=(B, len(values)))].mean(axis=1)


def boot_ratio(numerator: dict, denominator: dict, key: str) -> np.ndarray:
    """Σ numerator / Σ denominator over markets resampled with replacement (pairs cluster by market)."""
    markets = sorted(denominator)
    num = np.array([numerator.get(m, 0.0) for m in markets])
    den = np.array([denominator[m] for m in markets])
    draws = _rng(key).integers(0, len(markets), size=(B, len(markets)))
    return num[draws].sum(axis=1) / np.maximum(den[draws].sum(axis=1), 1e-300)


# --------------------------------------------------------------------- gates
def gates(data: dict) -> dict:
    a = data["part_a"]
    part_a = {}
    for case in ("quadratic", "overflow_limit", "overflow_finite"):
        rows = [r for r in a if r["case"] == case]
        part_a[case] = {"points": len(rows), "passed": sum(r["passed"] for r in rows),
                        "max_y_error": max(r["y_error"] for r in rows)}
        if case != "overflow_finite":
            part_a[case]["max_relative_error"] = max(r["relative_error"] for r in rows)
        else:
            part_a[case]["max_gap_over_bound"] = max(r["gap"] / r["bound"] for r in rows)
            part_a[case]["max_uniform_gap_over_lambda_root_n"] = max(
                r["uniform_gap"] / (r["lambda"] / math.sqrt(r["n"])) for r in rows)
            part_a[case]["y_within_a1_bound"] = sum(r["y_within_bound"] for r in rows)
    part_a_ok = all(v["passed"] == v["points"] for v in part_a.values()) and sum(
        v["points"] for v in part_a.values()) == len(e3r.part_a_tasks())
    div = {r["policy"]: r for r in data["diversify"]}
    diversify = {}
    for name, expected in e3.DIVERSIFY_VARIANCE.items():
        r = div[name]
        diversify[name] = {"within": r["within"], "across": r["across"], "lemma_total": r["lemma_total"],
                           "distribution_total": r["distribution_total"], "paper": expected,
                           "ok": abs(r["lemma_total"] - expected) <= EXACT_TOLERANCE * expected
                           and abs(r["distribution_total"] - expected) <= EXACT_TOLERANCE * expected}
    separated = json.loads(div["separated version of {A,B} always"]["weights"])
    diversify_ok = all(v["ok"] for v in diversify.values()) and separated == {"A,B": 0.0, "A,C": 0.5, "B,C": 0.5} \
        and div["{A,B} or {A,C}"]["within"] == 125.0 and div["{A,B} or {A,C}"]["across"] == 62_500.0
    exact_rows = [r for rows in data["part_b"].values() for r in rows if r["exact_applies"]]
    exact = {"rows": len(exact_rows),
             "configurations": sorted({r["config_id"] for r in exact_rows}),
             "max_load_error": max(r["exact_load_error"] for r in exact_rows),
             "max_contribution_error": max(r["exact_contribution_error"] for r in exact_rows),
             "max_match_error": max(r["exact_match_error"] for r in exact_rows),
             "max_across": max(r["exact_across"] for r in exact_rows)}
    exact_ok = (exact["max_load_error"] <= EXACT_TOLERANCE and exact["max_contribution_error"] <= EXACT_TOLERANCE
                and exact["max_match_error"] <= EXACT_TOLERANCE and exact["max_across"] <= ACROSS_TOLERANCE)
    all_rows = [r for rows in data["part_b"].values() for r in rows]
    decomposition = {
        alignment: {"rows": len(rows), "max_error": max(r["decomposition_error"] for r in rows),
                    "max_error_det": max(r["decomposition_error_det"] for r in rows)}
        for alignment in e3r.ALIGNMENTS for rows in [[r for r in all_rows if r["alignment"] == alignment]]}
    decomposition_ok = all(max(v["max_error"], v["max_error_det"]) <= DECOMPOSITION_TOLERANCE
                           for v in decomposition.values())
    contested_ok = all(r["contested_bound_holds"] for r in all_rows)
    markets = {cid: len({r["index"] for r in rows}) for cid, rows in data["part_b"].items()}
    complete = all(n == e3r.MARKETS for n in markets.values()) and all(
        len(rows) == e3r.MARKETS * len(e3r.POLICIES) for rows in data["part_b"].values())
    cost_check = max(r["cost_check"] for r in all_rows)
    return {"part_a": part_a, "part_a_passed": part_a_ok, "diversify": diversify, "diversify_separated": separated,
            "diversify_passed": diversify_ok, "separation_exact": exact, "separation_exact_passed": exact_ok,
            "decomposition": decomposition, "decomposition_passed": decomposition_ok,
            "contested_bound_passed": contested_ok, "markets": markets, "complete": complete,
            "cost_check_max": cost_check,
            "passed": part_a_ok and diversify_ok and exact_ok and decomposition_ok and contested_ok and complete}


# ---------------------------------------------------------------- statistics
def _by_policy(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["policy"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["index"])
    return out


def _mean_ci(values, key: str) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "ci95": interval(boot_mean(values, key))}


def statistics(data: dict, g: dict) -> dict:
    h3a_pr = {}
    for cid, rows in data["part_b"].items():
        blind = _by_policy(rows)["composition_blind"]
        r = np.array([np.mean([x["L1_ratio"], x["L2_ratio"]]) for x in blind])
        r_det = np.array([np.mean([x["%s_across_det" % L] / (e3r.N ** 2 * x["%s_theta" % L] ** 2 / 4)
                                   for L in e3r.LEADERS]) for x in blind])
        share = (r >= RATIO_THRESHOLD).astype(float)
        h3a_pr[cid] = {"markets": len(r), "mean_ratio": _mean_ci(r, cid + "/ratio"),
                       "share_ratio_at_least_half": _mean_ci(share, cid + "/share"),
                       "registered_rule_met": bool(share.mean() >= RATIO_SHARE), "mean_ratio_det": float(r_det.mean())}
    pairs = [r for r in data["h3c"] if not r["excluded"] and r.get("leader") not in (None, "")]
    h3c = h3c_statistics(pairs, "")
    h3c["markets"] = len({r["index"] for r in data["h3c"]})
    h3c["excluded_markets"] = len({r["index"] for r in data["h3c"] if r["excluded"]})
    h3c["no_leader_advertiser"] = len({r["index"] for r in data["h3c"] if r.get("status") == "no_leader_advertiser"})
    # A17 §5.1: the paper-feasible subset, markets where the guardrail floor does not bind
    h3c["paper_feasible"] = h3c_statistics([r for r in pairs if not r["floor_binds"]], "paper_feasible/")
    return {"H3a": {"label": "exploratory", "exact_part": g["separation_exact"], "pr_part": h3a_pr},
            "H3b": {"label": "exploratory", "part_a": g["part_a"]},
            "H3c": {"label": "exploratory", **h3c}}


def h3c_statistics(pairs: list[dict], prefix: str) -> dict:
    """H3c under M3 and M4 on (market, leader) pairs; ``prefix`` keys a subset's bootstrap."""
    h3c = {}
    for mechanism in ("M3", "M4"):
        rows = [r for r in pairs if r["mechanism"] == mechanism]
        per_market = {}
        for r in rows:
            per_market.setdefault(r["index"], []).append(r)
        count = {m: float(len(v)) for m, v in per_market.items()}
        bought = {m: float(sum(r["buyback"] for r in v)) for m, v in per_market.items()}
        stats_ = {"pairs": len(rows), "markets": len(per_market),
                  "buyback_share": sum(bought.values()) / len(rows),
                  "buyback_share_ci95": interval(boot_ratio(bought, count, prefix + mechanism + "/buyback")),
                  "mean_buyback_probability": float(np.mean([r["buyback_share"] for r in rows])),
                  "exposure_organic": float(np.mean([r["var_p_organic"] for r in rows])),
                  "exposure_star": float(np.mean([r["var_p_star"] for r in rows])),
                  "exposure_star_buyback_pairs": float(np.mean([r["var_p_star"] for r in rows if r["buyback"]]))
                  if any(r["buyback"] for r in rows) else None,
                  "only_if_share": float(np.mean([r["only_if_holds"] for r in rows])),
                  "only_if_failures": sum(1 for r in rows if not r["only_if_holds"]),
                  "only_if_failures_at_zero_bid": sum(1 for r in rows if not r["only_if_holds"]
                                                      and r["x_star"] == r["x_minus"]),
                  "max_margin_check": float(max(abs(r["margin_check"]) for r in rows)),
                  "floor_binds_markets": len({r["index"] for r in rows if r["floor_binds"]})}
        decided = [r for r in rows if not r["congestion_indifferent"]]
        agree = {}
        decided_count = {}
        for r in decided:
            decided_count[r["index"]] = decided_count.get(r["index"], 0.0) + 1
            agree[r["index"]] = agree.get(r["index"], 0.0) + float(r["buyback"] == r["bid_exceeds_congestion"])
        stats_.update(pairs_decided=len(decided), indifferent=len(rows) - len(decided),
                      coincidence=sum(agree.values()) / max(len(decided), 1),
                      coincidence_ci95=interval(boot_ratio(agree, decided_count, prefix + mechanism + "/coincide")))
        if mechanism == "M3":
            stats_["registered_rule_met"] = bool(stats_["buyback_share_ci95"][0] > 0)
        else:
            stats_["registered_rule_met"] = bool(stats_["coincidence"] >= COINCIDENCE)
        h3c[mechanism] = stats_
    return h3c


def means_table(data: dict) -> list[dict]:
    out = []
    fields = (["L1_mean", "L1_within", "L1_across", "L1_total", "L1_within_det", "L1_across_det", "match", "match_det",
               "contribution"] + e3r.COST_FIELDS + e3r.OVERFLOW_FIELDS)
    for cid, rows in data["part_b"].items():
        for policy, prs in _by_policy(rows).items():
            row = {"config_id": cid, "policy": policy, "markets": len(prs)}
            for f in fields:
                values = np.array([r[f] for r in prs], dtype=float)
                row[f] = float(values.mean())
                if f in ("L1_across", "match", BASE_COST):
                    row[f + "_ci95"] = interval(boot_mean(values, "%s/%s/%s" % (cid, policy, f)))
            out.append(row)
    return out


PAIRS = (("separated", "original"), ("composition_blind", "original"), ("quota", "separated"))


def paired_table(data: dict) -> list[dict]:
    out = []
    for cid, rows in data["part_b"].items():
        p = _by_policy(rows)
        for a, b in PAIRS:
            for f in ("match", "L1_across", BASE_COST, "cost_overflow_w0.5_l0.5", "overflow_prob_w0.5"):
                diff = np.array([x[f] - y[f] for x, y in zip(p[a], p[b])])
                out.append({"config_id": cid, "difference": "%s - %s" % (a, b), "metric": f,
                            "mean": float(diff.mean()),
                            "ci95": interval(boot_mean(diff, "%s/%s-%s/%s" % (cid, a, b, f)))})
    return out


# -------------------------------------------------------------------- figures
def figures(data: dict, stats_: dict, means: list[dict]) -> list[str]:
    paths = []
    a = data["part_a"]
    fig, axes = figure(1, 3, 9.0, 3.6)
    rows = []
    for ax, (case, title) in zip(axes[0], (("quadratic", "quadratic cost (450 points)"),
                                           ("overflow_limit", "overflow, limit objective (150)"),
                                           ("overflow_finite", "overflow, finite n (450)"))):
        pts = [r for r in a if r["case"] == case]
        for n in sorted({r["n"] for r in pts}):
            sub = [r for r in pts if r["n"] == n]
            ax.scatter([r["y_star"] for r in sub], [r["y"] for r in sub], s=10, alpha=0.6, color=N_COLORS[n],
                       label="n = %d" % n)
            rows += [(case, n, r["omega"], r["lambda"], r["delta"], r["y_star"], r["y"]) for r in sub]
        ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, linestyle="--", label="y = y*")
        ax.set_title(title)
        ax.set_xlabel("closed-form y*")
        ax.set_ylabel("numeric y")
    paths += save(fig, "sl20s", FIGURE_NAMES[0], ["case", "n", "omega", "lambda", "delta", "y_star", "y"], rows,
                  "Numeric optimum against the closed form of prop:threshold", legend=5, note=PART_A_NOTE)

    fig, axes = figure(1, 2, 8.4, 3.8)
    rows = []
    deltas = np.linspace(0.0, 1.7, 171)
    omegas = tuple(OMEGA_MARKERS)
    for ax, case in zip(axes[0], ("quadratic", "overflow_limit")):
        for k, omega in enumerate(omegas):
            if case == "quadratic":
                line = [e3.y_star_quadratic(1000, omega, 1.0, d) for d in deltas]
            else:
                line = [e3.y_star_overflow(omega, 1.0, d) if 0 < d != 1.0 else math.nan for d in deltas]
            color = "C%d" % k
            # Below 1 each ω has its own line; where y* = 1 the lines coincide, so they are drawn with interleaved
            # dashes (every ω stays visible) and the numeric points are nested markers.
            ax.plot(deltas, [y if y < 1 else math.nan for y in line], linewidth=1.0, color=color, label="ω = %g" % omega)
            capped = [1.0 if y >= 1 else math.nan for y in line]
            ax.plot(deltas, capped, linewidth=1.6, color=color, linestyle=(2.5 * k, (2.5, 2.5 * (len(omegas) - 1))))
            rows += [(case, "closed form", omega, float(d), y) for d, y in zip(deltas, line)]
            pts = [r for r in a if r["case"] == case and r["n"] == 1000 and r["lambda"] == 1.0 and r["omega"] == omega]
            marker, size = OMEGA_MARKERS[omega]
            ax.scatter([r["delta"] for r in pts], [r["y"] for r in pts], s=size, marker=marker, facecolors="none",
                       edgecolors=color, linewidths=1.0, zorder=3)
            rows += [(case, "numeric", omega, r["delta"], r["y"]) for r in pts]
        ax.axvline(1.0, color="0.75", linewidth=0.8, zorder=0)
        if case == "overflow_limit":
            ax.text(1.35, 0.9, "Δ > λ: y* = 1\nfor every ω", ha="center", va="top", fontsize=7, color="0.3")
            ax.text(0.5, 0.9, "Δ < λ: y* = |2ω − 1|", ha="center", va="top", fontsize=7, color="0.3")
        ax.set_ylim(-0.05, 1.08)
        ax.set_title("%s cost, λ = 1, n = 1000" % ("quadratic" if case == "quadratic" else "overflow (limit)"))
        ax.set_xlabel("Δ (value of the state's better match)")
        ax.set_ylabel("head-to-head share y*")
    paths += save(fig, "sl20s", FIGURE_NAMES[1], ["case", "series", "omega", "delta", "y"], rows,
                  "Optimal head-to-head share against Δ (lines: closed form; open markers: numeric)", legend=4,
                  note=PART_A_NOTE + ". Grey line: Δ = λ (knife-edge, excluded).")

    no_floor = "Organic policies only: no sponsorship, so the guardrail floor does not apply (A17 §5.1)."
    table = {(m["config_id"], m["policy"]): m for m in means}
    fig, axes = figure(1, 2, 8.0, 3.6)
    rows = []
    for ax, alignment in zip(axes[0], e3r.ALIGNMENTS):
        cid = "part_b/%s/baseline" % alignment
        x = np.arange(len(e3r.POLICIES))
        within = [table[(cid, p)]["L1_within"] for p in e3r.POLICIES]
        across = [table[(cid, p)]["L1_across"] for p in e3r.POLICIES]
        ax.bar(x, within, color="#4C72B0", label="within state E[Var(D|Λ)]")
        ax.bar(x, across, bottom=within, color="#DD8452", label="across states Var(E[D|Λ])")
        ax.set_xticks(x, [POLICY_NAMES[p] for p in e3r.POLICIES], rotation=15)
        ax.set_title("%s, baseline (σ = 0.3)" % alignment.replace("_", " "))
        ax.set_ylabel("Var(D_L1), choices²")
        rows += [(alignment, p, w, c) for p, w, c in zip(e3r.POLICIES, within, across)]
    paths += save(fig, "sl20s", FIGURE_NAMES[2], ["alignment", "policy", "within", "across"], rows,
                  "Variance of the leader's load by policy (Lemma load)", legend=2, note=no_floor)

    fig, axes = figure(2, 2, 8.0, 6.4)
    rows = []
    for col, alignment in enumerate(e3r.ALIGNMENTS):
        cid = "part_b/%s/baseline" % alignment
        for row_i, kind in enumerate(e3r.COST_KINDS):
            ax = axes[row_i][col]
            for p in e3r.POLICIES:
                cost = [table[(cid, p)]["cost_%s_w%g_l0.5" % (kind, w)] for w in e3r.OMEGAS]
                match = table[(cid, p)]["match"]
                style = {"separated": {"linewidth": 3.0, "alpha": 0.5}, "quota": {"linestyle": "--", "marker": "s"}}
                ax.plot(cost, [match] * len(cost), **{"marker": "o", "markersize": 3, "linewidth": 1.0,
                                                      **style.get(p, {})}, label=POLICY_NAMES[p])
                rows += [(alignment, kind, p, w, c, match) for w, c in zip(e3r.OMEGAS, cost)]
            ax.set_title("%s, %s cost, λ = 0.5" % (alignment.replace("_", " "), kind))
            ax.set_xlabel("expected cost per request (ω = 1 → 0.3)")
            ax.set_ylabel("match quality per request")
    paths += save(fig, "sl20s", FIGURE_NAMES[3], ["alignment", "cost_kind", "policy", "omega", "cost", "match"], rows,
                  "Cost and match quality by policy across capacity shares ω", legend=4, note=no_floor)

    h = stats_["H3c"]
    subsets = (("all markets", h), ("paper-feasible subset", h["paper_feasible"]))
    fig, axes = figure(1, 2, 8.4, 3.8)
    rows = []
    ax = axes[0][0]
    width = 0.36
    for k, (name, v) in enumerate(subsets):
        shares = [v[m]["buyback_share"] for m in ("M3", "M4")]
        cis = [v[m]["buyback_share_ci95"] for m in ("M3", "M4")]
        yerr = [[max(s - c[0], 0.0) for s, c in zip(shares, cis)], [max(c[1] - s, 0.0) for s, c in zip(shares, cis)]]
        ax.bar(np.arange(2) + width * (k - 0.5), shares, width=width, yerr=yerr, capsize=3,
               color=["#4C72B0", "#C44E52"], alpha=1.0 if k == 0 else 0.45, edgecolor="0.2",
               hatch=None if k == 0 else "//")
        rows += [("buyback_share", name, m, s, c[0], c[1], v[m]["markets"]) for m, s, c in zip(("M3", "M4"), shares, cis)]
    ax.set_xticks([0, 1], ["M3 (C ≡ 0)", "M4 (congestion priced)"])
    ax.set_ylabel("pairs with buy-back (share)")
    ax.set_ylim(0, 1)
    ax.set_title("buy-back of sponsored inclusion\n(solid: all %d markets; hatched: %d paper-feasible)" % (
        h["M3"]["markets"], h["paper_feasible"]["M3"]["markets"]), fontsize=8)
    ax = axes[0][1]
    for k, (name, v) in enumerate(subsets):
        organic = [v[m]["exposure_organic"] for m in ("M3", "M4")]
        star = [v[m]["exposure_star"] for m in ("M3", "M4")]
        offset = 0.4 * (k - 0.5)
        ax.bar(np.arange(2) + offset - 0.09, organic, width=0.18, color="0.6", edgecolor="0.2",
               hatch=None if k == 0 else "//", label="organic (separated)" if k == 0 else None)
        ax.bar(np.arange(2) + offset + 0.09, star, width=0.18, color="#DD8452", edgecolor="0.2",
               hatch=None if k == 0 else "//", label="with sponsorship Q*(b)" if k == 0 else None)
        rows += [("exposure_organic", name, m, x, None, None, v[m]["markets"]) for m, x in zip(("M3", "M4"), organic)]
        rows += [("exposure_star", name, m, x, None, None, v[m]["markets"]) for m, x in zip(("M3", "M4"), star)]
    ax.set_xticks([0, 1], ["M3", "M4"])
    ax.set_ylabel("leader exposure Var_Λ(p_j)")
    ax.set_title("common-shock exposure\n(left pair: all markets; right, hatched: paper-feasible)", fontsize=8)
    paths += save(fig, "sl20s", FIGURE_NAMES[4], ["series", "subset", "mechanism", "value", "ci_low", "ci_high",
                                                 "markets"], rows,
                  "H3c: leaders buy back sponsored inclusion under M3 and M4", legend=2,
                  note="All markets use the guardrail floor; in the paper-feasible subset it does not bind, so the "
                       "paper's guardrail holds there (A17 §5.1).")
    return [p.relative_to(REPO).as_posix() for p in paths]


def _write_csv(path: Path, rows: list[dict], label: str = LABEL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    text.write("# " + label + "\n")
    writer = csv.writer(text, lineterminator="\n")
    keys = list(rows[0])
    writer.writerow(keys)
    for row in rows:
        writer.writerow([json.dumps(row[k]) if isinstance(row[k], list) else
                         repr(row[k]) if isinstance(row[k], float) else row[k] for k in keys])
    path.write_text(text.getvalue(), encoding="utf-8")


# ------------------------------------------------------------------- markdown
def _ci(v) -> str:
    return "[%.4f, %.4f]" % tuple(v)


def markdown(summary: dict, means: list[dict]) -> str:
    g, s = summary["gates"], summary["statistics"]
    pa = g["part_a"]
    lines = [
        "# SL-20S E3: Part A implementation check and Part B on stipulated states", "",
        "Results of `python -m features.storelab.analysis.e3_run` (master seed %d), regenerated by "
        "`python -m features.storelab.analysis.e3_report`." % e3r.MASTER_SEED, "",
        "**Review status.** SL-20S was accepted by the owner without an independent review (2026-10-24); SL-24 is "
        "its independent audit, which also added the paper-feasible subset of H3c and the overflow panel fix "
        "(A17 §5). Part A is **not final**: SL-20 must have it reviewed before carrying it over unchanged.", "",
        "**Part B is labelled: %s.** Its states are stand-ins (A17 §3.2), not estimates of any agent, and no "
        "confirmatory label is recorded. Part A uses no stipulated parameters." % LABEL, "",
        "## Design", "",
        "- **Part A** (A17 §3.1): the two-leader environment of `prop:threshold` (ν₋ = 0, deterministic agent), "
        "solved as a convex program over the three lists with SL-16's pinned solvers (QP for quadratic cost; LPs for "
        "the overflow limit objective and for the finite-n overflow cost through SL-16's tangents, whose largest error "
        "is known); 450 + 150 + 450 points.",
        "- **Part B** (A17 §1, §3.2; owner decisions 2026-10-24): 10 products, the two leaders set to the mean of the "
        "two largest of 10 N(0, 1) draws, the others redrawn below z_L − 0.5; scores equal z; K = 4; n = 1000; "
        "σ = 0.3; 2 equally likely states shifting the leaders by ±0.5. Policies: original (top K by u; "
        "match-quality-maximising in the deterministic limit), its separated version (Prop separation, ϖ from the "
        "deterministic limit), composition-blind (original or R†_L1, 1/2 each), and a quota implementation of the "
        "separated version. The H3a exact part and Θ_j use the exact σ → 0 limit; every other Part B metric is at "
        "σ = 0.3. %d configurations (aligned / not aligned × baseline and 7 sensitivity levels), %d markets each; "
        "costs on every product, quadratic and overflow, ω ∈ {0.3, 0.5, 0.6, 0.8, 1}, λ ∈ {0.1, 0.5, 2}." % (
            len(g["markets"]), e3r.MARKETS),
        "- **H3c**: sponsorship on the not-aligned baseline markets (K^s = 1, 6 advertisers, θ ~ U[0, 1], δ = −0.5), "
        "M3 against M4 (quadratic cost λ = 0.5, ω = 0.5), with SL-19S's guardrail feasibility floor (owner decision "
        "2026-10-24: the separated organic lottery is below 0.95 M^ref in most markets at σ = 0.3).", "",
        "## Part A: `prop:threshold` (AC-1, H3b)", "",
        "| Case | Points | Passed | Largest error | Tolerance |", "|---|---:|---:|---|---|",
        "| quadratic, n ∈ {10, 100, 1000} | %d | %d | \\|y − y*\\| %.1e; objective %.1e relative | 1e-4; 1e-6 |" % (
            pa["quadratic"]["points"], pa["quadratic"]["passed"], pa["quadratic"]["max_y_error"],
            pa["quadratic"]["max_relative_error"]),
        "| overflow, limit objective | %d | %d | objective %.1e relative | 1e-6 |" % (
            pa["overflow_limit"]["points"], pa["overflow_limit"]["passed"], pa["overflow_limit"]["max_relative_error"]),
        "| overflow, finite n ∈ {100, 1000, 10000} | %d | %d | f(y*) − f(y_n) at most %.3f of 2λ/√n + 1e-6 | "
        "2λ/√n + 1e-6 |" % (pa["overflow_finite"]["points"], pa["overflow_finite"]["passed"],
                            pa["overflow_finite"]["max_gap_over_bound"]), "",
        "%s Also reported (not registered): |f_n(y_n) − f(y_n)| is at most %.3f of λ/√n, and |y_n − y*| is within "
        "A1 §2.3's bound at %d of %d finite-n points." % (
            "All %d points pass: Part B is not blocked." % sum(v["points"] for v in pa.values())
            if g["part_a_passed"] else "**Part A FAILED: Part B is blocked.**",
            pa["overflow_finite"]["max_uniform_gap_over_lambda_root_n"], pa["overflow_finite"]["y_within_a1_bound"],
            pa["overflow_finite"]["points"]), "",
        "## Example `ex:diversify` (AC-2)", "",
        "| Policy | Within state | Across states | Lemma load | Exact distribution | Paper |", "|---|---:|---:|---:|---:|---:|"]
    for name, v in g["diversify"].items():
        lines.append("| %s | %s | %s | %s | %s | %s |" % (name, "{:,.0f}".format(v["within"]), "{:,.0f}".format(v["across"]),
                                                       "{:,.0f}".format(v["lemma_total"]),
                                                       "{:,.6f}".format(v["distribution_total"]),
                                                       "{:,.0f}".format(v["paper"])))
    ex = g["separation_exact"]
    lines += ["", "The separated version of {A,B} always is %s, and in the last row Θ_A = 1/2 and the across-state "
              "part n²/16 = 62,500 attains the `lem:contested` bound. Reproduced: %s." % (
                  ", ".join("%s: %g" % (k, v) for k, v in g["diversify_separated"].items() if v),
                  "yes" if g["diversify_passed"] else "NO"), "",
              "## Part B checks (AC-3)", "",
              "| Check | Result | Tolerance |", "|---|---|---|",
              "| Separated and quota vs original, not aligned, α = 1 (%d rows in %d configurations, deterministic limit): "
              "loads, contributions, match quality | %.1e, %.1e, %.1e relative | 1e-9 |" % (
                  ex["rows"], len(ex["configurations"]), ex["max_load_error"], ex["max_contribution_error"],
                  ex["max_match_error"]),
              "| Their across-state variance n² Var_Λ(p_j), every product | %.1e | 1e-12 |" % ex["max_across"]]
    for alignment, v in g["decomposition"].items():
        lines.append("| Within + across = total variance, %s (%d rows; σ = 0.3 / deterministic) | %.1e / %.1e relative "
                     "| 1e-9 |" % (alignment.replace("_", " "), v["rows"], v["max_error"], v["max_error_det"]))
    lines += ["| Var_Λ(p_j) ≤ Θ_j²/4 (`lem:contested`), every product, policy and market (deterministic) | %s | — |" % (
        "holds" if g["contested_bound_passed"] else "FAILS"),
        "| Exact distribution against SL-16's expected cost (i.i.d. policies) | %.1e per request | — |" % g["cost_check_max"],
        "", "All Part B checks pass." if g["passed"] else "**A CHECK FAILED: the report is blocked.**", "",
        "The levels with humans (1 − α > 0) are outside Prop separation (α ≡ 1): there the separated lottery is "
        "reported but not checked.", "",
        "## Registered statistics (exploratory, no confirmatory label)", "", "*%s.*" % LABEL, "",
        "**H3a, exact part**: as in the checks above (deterministic limit).", "",
        "**H3a, Pr-part** (composition-blind exposure ratio r = Var_Λ(p_j) / (Θ_j²/4), mean of the two leaders; "
        "Var_Λ at σ = 0.3, Θ_j in the deterministic limit; each market is a category, so the rule reads: r ≥ 0.5 in "
        "at least 2/3 of markets):", "",
        "| Configuration | Mean r | 95% interval | Share with r ≥ 0.5 | 95% interval | Rule met | Mean r (deterministic) |",
        "|---|---:|---|---:|---|---|---:|"]
    for cid, v in s["H3a"]["pr_part"].items():
        lines.append("| %s | %.3f | %s | %.3f | %s | %s | %.3f |" % (
            cid[len("part_b/"):], v["mean_ratio"]["mean"], _ci(v["mean_ratio"]["ci95"]),
            v["share_ratio_at_least_half"]["mean"], _ci(v["share_ratio_at_least_half"]["ci95"]),
            "yes" if v["registered_rule_met"] else "no", v["mean_ratio_det"]))
    lines += ["", "At σ = 0.3 the ratio can exceed 1: the anchor R†_L1 is not contested in the deterministic limit, "
              "but in L1's unfavoured state its score equals the cap of the other products' scores, so its logit choice "
              "probability still moves with the state. The deterministic Θ_j does not count that exposure.", "",
              "**H3b**: Part A above; every grid point passes.", ""]
    h = s["H3c"]
    lines += ["**H3c** (%d markets, %d with no leader among the advertisers; (market, leader) pairs with the leader an "
              "advertiser; intervals resample markets):" % (h["markets"], h["no_leader_advertiser"]), "",
              "| Mechanism | Pairs | Buy-back share | 95% interval | Mean buy-back probability | Exposure organic → Q* | "
              "Buy-back iff bid term > congestion term | 95% interval | A1 reading (buy-back only if bid term > "
              "remaining terms) |", "|---|---:|---:|---|---:|---|---:|---|---:|"]
    for subset, data_ in (("all markets", h), ("paper-feasible subset", h["paper_feasible"])):
        for m in ("M3", "M4"):
            v = data_[m]
            lines.append("| %s, %s (%d markets) | %d | %.3f | %s | %.3f | %.4f → %.4f | %.3f (%d decided, %d "
                         "indifferent) | %s | %.3f |" % (
                             m, subset, v["markets"], v["pairs"], v["buyback_share"], _ci(v["buyback_share_ci95"]),
                             v["mean_buyback_probability"], v["exposure_organic"], v["exposure_star"], v["coincidence"],
                             v["pairs_decided"], v["indifferent"], _ci(v["coincidence_ci95"]), v["only_if_share"]))
    lines += ["", "Registered rules (A12 §4), read as exploratory: under M3 the buy-back share's 95%% lower bound is "
              "%s 0; under M4 buy-back coincides with \"bid term > congestion term\" in %.1f%% of decided pairs "
              "(registered: ≥ 95%%). The congestion term is the last sum of `cor:congestion` with Q^{−j} = "
              "Q^{s*}(0, b_{−j}); pairs within 1e-6 per request of indifference are excluded. The margin identity "
              "n[V_b(Q*) − V_b(Q^{−j})] = bid term − remaining terms holds to %.1e. The floor binds in %d markets; in the paper-feasible subset (A17 §5.1: the %d markets "
              "where it does not bind, so the paper's guardrail holds) the M3 lower bound is %s 0 and the M4 "
              "coincidence is %.1f%%." % (
                  "above" if h["M3"]["registered_rule_met"] else "not above", 100 * h["M4"]["coincidence"],
                  max(h["M3"]["max_margin_check"], h["M4"]["max_margin_check"]), h["M4"]["floor_binds_markets"],
                  h["paper_feasible"]["M4"]["markets"],
                  "above" if h["paper_feasible"]["M3"]["registered_rule_met"] else "not above",
                  100 * h["paper_feasible"]["M4"]["coincidence"]),
              "", "The coincidence column is registered for M4 only; under M3 the congestion term is not in the "
              "objective and the column is descriptive. Buy-back pairs where Q^{−j} buys the leader back too (a zero "
              "bid gives the same allocation, so both sides of the A1 reading are 0; the η M tie-break prefers the "
              "list) account for %d of the %d M3 pairs and %d of the %d M4 pairs where the A1 reading does not "
              "hold." % (h["M3"]["only_if_failures_at_zero_bid"], h["M3"]["only_if_failures"],
                         h["M4"]["only_if_failures_at_zero_bid"], h["M4"]["only_if_failures"]), ""]
    lines += ["## Policies (baseline, per request, σ = 0.3)", "", "*%s.*" % LABEL, "",
              "| Alignment | Policy | Var(D_L1) within | across | Match quality | Quadratic cost (ω = λ = 0.5) | "
              "Pr(D_L > B), ω = 0.5 |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in means:
        if not m["config_id"].endswith("/baseline"):
            continue
        lines.append("| %s | %s | %.1f | %.1f | %.4f | %.4f | %.3f |" % (
            m["config_id"].split("/")[1].replace("_", " "), POLICY_NAMES[m["policy"]], m["L1_within"], m["L1_across"],
            m["match"], m[BASE_COST], m["overflow_prob_w0.5"]))
    lines += ["", "Every configuration, cost setting and paired difference (separated − original, composition-blind − "
              "original, quota − separated) with 95% intervals: `artifacts/results/sl20s/tables/`.", "",
              "## Figures", "", "Every figure carries \"%s\"; the Part A figures add \"%s\" beneath it." % (
                  LABEL, PART_A_NOTE), ""]
    captions = {"1_numeric_vs_closed_form": "Part A", "2_y_star_vs_delta": "Part A; for Δ > λ the overflow optimum is "
                "y* = 1 at every ω", "3_variance_decomposition": "Part B; organic policies, the floor does not apply",
                "4_cost_match_tradeoff": "Part B; organic policies, the floor does not apply",
                "5_h3c_buyback": "H3c; all markets and the paper-feasible subset"}
    lines += ["- `%s`: %s" % (p, captions[Path(p).stem]) for p in summary["figures"] if p.endswith(".pdf")]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="python -m features.storelab.analysis.e3_report").parse_args(argv)
    data = load()
    g = gates(data)
    if not g["passed"]:
        print(json.dumps(g, indent=1, default=str))
        print("a gate failed: no report is written")
        return 1
    s = statistics(data, g)
    means = means_table(data)
    paired = paired_table(data)
    _write_csv(TABLES / "part_a.csv", [{k: r[k] for k in e3r.PART_A_FIELDS} for r in data["part_a"]], PART_A_LABEL)
    _write_csv(TABLES / "part_b_means.csv", means)
    _write_csv(TABLES / "part_b_paired_differences.csv", paired)
    paths = figures(data, s, means)
    check = e3r.RESULTS / "rerun_check.json"
    summary = {"label": LABEL, "part_a_label": PART_A_LABEL, "master_seed": e3r.MASTER_SEED, "gates": g,
               "statistics": s, "figures": paths, "review_status": "accepted by the owner without independent review (2026-10-24); independent audit: SL-24",
               "rerun_identical": json.loads(check.read_text(encoding="utf-8"))["identical"] if check.exists() else None}
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    REPORT.write_text(markdown(summary, means), encoding="utf-8")
    print("report written; gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
