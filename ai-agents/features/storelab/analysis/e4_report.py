"""SL-21S report: tables, figures and ``docs/e4_stipulated_results.md`` from the result files.

``python -m features.storelab.analysis.e4_report`` reads ``artifacts/results/sl21s/`` (written by
``e4_run``) and regenerates everything.

**Gates** (a failure blocks the report):
- the choice-function count A(N) = ⌊e · N!⌋ equals the enumeration for every N ≤ 7 and
  K = 2..4, and the corrected Cor design(b) bound is computed from it (A1 §E1);
- the H4b implementation check (A12 §4) passes in every checked cell: |bias| ≤ 3 MC SE and the
  variance within ±5% of the Prop learning(a) closed form, 10,000 replications;
- the H4a I-check: in every replication and state, the Prop identify criterion and the
  independent possible-winner rule agree on every target set, and every identified choice is
  the true one;
- every cell has R = 200 completed replications.

**Registered statistics** (A12 §4, H4b and H4d as corrected in A1 §E2–§E3) are computed as
written and labelled exploratory; no confirmatory label is recorded (A17 §0). Intervals are
percentile bootstraps over replications (B = 2,000) with seeds from the master seed;
Clopper–Pearson intervals for H4c's violation share.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy import stats

from ..core.rng import stream
from . import e2_run as er
from . import e4_run as e4r
from .e2_report import _num, interval
from .figures import LABEL, REPO, figure, save

DOCS = Path(__file__).resolve().parents[1] / "docs"
REPORT = DOCS / "e4_stipulated_results.md"
SUMMARY = DOCS / "e4_stipulated_results.json"
TABLES = e4r.RESULTS / "tables"
B = 2000
FIGURE_NAMES = ("1_identification_vs_inclusion", "2_rmse", "3_misreport_gain", "4_cumulative_loss")
DESIGN_NAMES = {"leader_inclusive": "leader-inclusive", "competitor_varying": "competitor-varying",
                "random": "random (matched)"}
SLOPE_RANGE = (-0.55, -0.45)  # A12 §4, H4b
VIOLATION_LIMIT = 3 * e4r.RHO  # A12 §4, H4c
EXPONENT_LIMIT = 0.6  # A12 §4, H4d (A1 §E2)


class GateFailure(RuntimeError):
    pass


def _rows(path: Path) -> list[dict]:
    return [{k: _num(v) for k, v in r.items()} for r in er.read_rows(path)]


def cells() -> list[tuple[int, float, int]]:
    return [(n, f, n0) for n in e4r.N_XS for f in e4r.FACTORS for n0 in e4r.N0S]


def load() -> dict:
    out = e4r.RESULTS
    return {"bound": _rows(out / "bound.csv.gz"), "check": _rows(out / "h4b_check.csv.gz"),
            "identification": _rows(out / "identification.csv.gz"),
            "learning": {c: _rows(er.result_path(out, e4r.cell_id("learning", *c))) for c in cells()},
            "recovery": {c: _rows(er.result_path(out, e4r.cell_id("recovery", *c))) for c in cells()}}


def _rng(*keys):
    return stream(e4r.MASTER_SEED, "sl21s-bootstrap", *[str(k) for k in keys])


def _draws(count: int, *keys) -> np.ndarray:
    return _rng(*keys).integers(0, count, size=(B, count))


# --------------------------------------------------------------------- gates
def gates(data: dict) -> dict:
    small = [r for r in data["bound"] if r["enumerated"] is not None]
    bound_ok = all(r["enumerated"] == r["choice_functions"] == r["floor_e_factorial"] for r in small) and {
        (r["n"], r["k"]) for r in small} == {(n, k) for n in range(2, 8) for k in range(2, min(e4r.K, n) + 1)}
    example = next(r for r in small if (r["n"], r["k"]) == (5, 4))  # A1 §E1: corrected 4, paper 5
    check = data["check"]
    check_ok = len(check) == len(e4r.CHECK_CELLS) and all(r["passed"] for r in check) and {
        (r["h"], r["n0"], r["f"]) for r in check} == set(e4r.CHECK_CELLS)
    ident = data["identification"]
    ident_ok = all(r["check_equal"] and r["identified_correct"] for r in ident)
    complete = (len(ident) == len(e4r.N_XS) * len(e4r.FACTORS) * e4r.REPLICATIONS * len(e4r.DESIGNS)
                and all(len(v) == e4r.REPLICATIONS * len(e4r.HS) for v in data["learning"].values())
                and all(len(v) == e4r.REPLICATIONS * len(e4r.DESIGNS) for v in data["recovery"].values())
                and all(len({r["replication"] for r in v}) == e4r.REPLICATIONS
                        for d in (data["learning"], data["recovery"]) for v in d.values()))
    return {"bound": {"checked": len(small), "passed": bound_ok, "example_n5_k4": {
                "corrected": example["corrected_bound"], "paper": example["paper_bound"]},
                      "designs": [{k: r[k] for k in ("n", "k", "corrected_bound", "paper_bound", "audited_sets")}
                                  for r in data["bound"] if r["audited_sets"] is not None]},
            "h4b_check": {"cells": check, "passed": check_ok,
                          "max_bias_in_se": max(r["bias_in_se"] for r in check),
                          "max_variance_deviation": max(abs(r["variance_ratio"] - 1) for r in check)},
            "h4a_icheck": {"rows": len(ident), "passed": ident_ok},
            "complete": complete, "passed": bound_ok and check_ok and ident_ok and complete}


# ---------------------------------------------------------------- statistics
def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS slopes of every row of y (draws, len(x)) on x."""
    xc = x - x.mean()
    return (y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)


def h4a(data: dict) -> dict:
    out = {}
    for n in e4r.N_XS:
        for f in e4r.FACTORS:
            rows = [r for r in data["identification"] if r["n_x"] == n and r["f"] == f]
            by = {d: np.array([r["share"] for r in sorted(rows, key=lambda r: r["replication"]) if r["design"] == d])
                  for d in e4r.DESIGNS}
            freq = {d: float(np.mean([r["inclusion_top5"] for r in rows if r["design"] == d])) for d in e4r.DESIGNS}
            leader = {d: float(np.mean([r["inclusion_leader"] for r in rows if r["design"] == d])) for d in e4r.DESIGNS}
            all_k = {d: float(np.mean([r["share_all_k"] for r in rows if r["design"] == d])) for d in e4r.DESIGNS}
            draws = _draws(e4r.REPLICATIONS, "h4a", n, f)
            cell = {"designs": {d: {"share": float(by[d].mean()), "ci95": interval(by[d][draws].mean(axis=1)),
                                    "inclusion_top5": freq[d], "inclusion_leader": leader[d], "share_all_k": all_k[d]}
                                for d in e4r.DESIGNS}}
            for other in ("leader_inclusive", "random"):
                diff = by["competitor_varying"] - by[other]
                ci = interval(diff[draws].mean(axis=1))
                cell["competitor_varying - " + other] = {"mean": float(diff.mean()), "ci95": ci,
                                                         "lower_bound_above_0": bool(ci[0] > 0)}
            out["N_x=%d/f=%g" % (n, f)] = cell
    return {"label": "exploratory", "cells": out}


def _matrix(rows: list[dict], field: str) -> np.ndarray:
    ordered = sorted(rows, key=lambda r: (r["replication"], r["h"]))
    return np.array([r[field] for r in ordered]).reshape(e4r.REPLICATIONS, len(e4r.HS))


def h4b_h4c_h4d(data: dict) -> tuple[dict, dict, dict]:
    logh = np.log(np.array(e4r.HS, dtype=float))
    b_out, c_out, d_out = {}, {}, {}
    for cell, rows in data["learning"].items():
        key = "N_x=%d/f=%g/n0=%d" % cell
        draws = _draws(e4r.REPLICATIONS, "h4b", *cell)
        count = rows[0]["estimates"]
        sse, closed_sum = _matrix(rows, "sse"), _matrix(rows, "closed_variance_sum")  # (reps, h)
        rmse, closed = np.sqrt(sse.mean(axis=0) / count), np.sqrt(closed_sum.mean(axis=0) / count)
        slope = _slope(logh, np.log(rmse))
        ci = interval(_slopes(logh, np.log(np.sqrt(sse[draws].mean(axis=1) / count))))
        b_out[key] = {"rmse": rmse.tolist(), "rmse_closed_form": closed.tolist(), "ratio": (rmse / closed).tolist(),
                      "slope": slope, "ci95": ci, "within_registered_range": bool(SLOPE_RANGE[0] <= ci[0]
                                                                                 and ci[1] <= SLOPE_RANGE[1])}
        gains = np.stack([_matrix(rows, "gain_%d" % a) for a in range(e4r.ADVERTISERS)], axis=2)  # (reps, h, ads)
        bounds = np.stack([_matrix(rows, "bound_%d" % a) for a in range(e4r.ADVERTISERS)], axis=2)
        per_h = []
        for k, h in enumerate(e4r.HS):
            pairs = gains[:, k, :].size
            violations = int((gains[:, k, :] > bounds[:, k, :]).sum())
            upper = 1.0 if violations == pairs else float(stats.beta.ppf(0.975, violations + 1, pairs - violations))
            per_h.append({"h": h, "pairs": pairs, "violations": violations, "share": violations / pairs,
                          "clopper_pearson_upper": upper, "within_limit": bool(upper <= VIOLATION_LIMIT),
                          "mean_max_gain": float(gains[:, k, :].max(axis=1).mean()),
                          "mean_bound": float(bounds[:, k, :].mean()),
                          "mean_gain_over_bound": float((gains[:, k, :] / bounds[:, k, :]).mean())})
        max_gain = gains.max(axis=2)  # (reps, h)
        g_slope = _slope(logh, max_gain.mean(axis=0))
        g_ci = interval(_slopes(logh, max_gain[draws].mean(axis=1)))
        c_out[key] = {"by_h": per_h, "max_gain_slope_on_log_h": g_slope, "ci95": g_ci,
                      "upper_bound_below_0": bool(g_ci[1] < 0)}
        loss = _matrix(rows, "cumulative_loss")
        mean = loss.mean(axis=0)
        entry = {"mean_cumulative_loss": mean.tolist(), "zero_loss_share": float(np.mean(loss[:, -1] == 0))}
        if np.all(mean > 0):
            means = loss[draws].mean(axis=1)  # (B, h)
            positive = np.all(means > 0, axis=1)
            boot = _slopes(logh, np.log(np.where(means > 0, means, 1.0)))[positive]
            ci = interval(boot)
            entry.update(slope=_slope(logh, np.log(mean)), ci95=ci, bootstrap_draws=len(boot),
                         upper_bound_at_most_limit=bool(ci[1] <= EXPONENT_LIMIT),
                         lower_bound_above_limit=bool(ci[0] > EXPONENT_LIMIT))
        else:
            entry.update(slope=None, ci95=None, note="mean cumulative loss is 0 at some h: bounded loss")
        d_out[key] = entry
    return ({"label": "exploratory", "cells": b_out}, {"label": "exploratory", "cells": c_out},
            {"label": "exploratory", "cells": d_out})


def recovery(data: dict) -> dict:
    out = {}
    for cell, rows in data["recovery"].items():
        key = "N_x=%d/f=%g/n0=%d" % cell
        out[key] = {d: {"mean_periods": float(np.mean([r["recovery"] for r in rows if r["design"] == d])),
                        "median_periods": float(np.median([r["recovery"] for r in rows if r["design"] == d])),
                        "censored_share": float(np.mean([r["censored"] for r in rows if r["design"] == d])),
                        "share_pre": float(np.mean([r["share_pre"] for r in rows if r["design"] == d]))}
                    for d in e4r.DESIGNS}
    return {"label": "exploratory", "cells": out}


def statistics(data: dict, g: dict) -> dict:
    b, c, d = h4b_h4c_h4d(data)
    b["implementation_check"] = g["h4b_check"]
    return {"H4a": h4a(data), "H4b": b, "H4c": c, "H4d": d, "recovery": recovery(data)}


# -------------------------------------------------------------------- figures
def figures(data: dict, s: dict) -> list[str]:
    paths = []
    fig, axes = figure(1, 2, 8.0, 3.4)
    rows = []
    colours = {"leader_inclusive": "#C44E52", "competitor_varying": "#4C72B0", "random": "#55A868"}
    for ax, n in zip(axes[0], e4r.N_XS):
        for d in e4r.DESIGNS:
            xs, ys, lo, hi = [], [], [], []
            for f in e4r.FACTORS:
                v = s["H4a"]["cells"]["N_x=%d/f=%g" % (n, f)]["designs"][d]
                xs.append(v["inclusion_top5"])
                ys.append(v["share"])
                lo.append(v["share"] - v["ci95"][0])
                hi.append(v["ci95"][1] - v["share"])
                rows.append((n, d, f, v["inclusion_top5"], v["inclusion_leader"], v["share"], *v["ci95"]))
            ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="o", color=colours[d], capsize=2,
                        label=DESIGN_NAMES[d])
        ax.set_title("N_x = %d (points: f = 0, 0.5, 1, 2)" % n)
        ax.set_xlabel("mean inclusion frequency of the top 5 products")
        ax.set_ylabel("identified share of target sets")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
    paths += save(fig, "sl21s", FIGURE_NAMES[0], ["n_x", "design", "f", "inclusion_top5", "inclusion_leader", "share",
                                                  "ci_low", "ci_high"], rows,
                  "H4a: identification share by design against inclusion frequency")

    fig, axes = figure(1, 2, 8.0, 3.4)
    rows = []
    ax = axes[0][0]
    hs = np.array(e4r.HS, dtype=float)
    for f in e4r.FACTORS:
        v = s["H4b"]["cells"]["N_x=10/f=%g/n0=10" % f]
        ax.plot(hs, v["rmse"], marker="o", markersize=3, label="f = %g" % f)
        rows += [("rmse_vs_h", 10, f, 10, int(h), r, c) for h, r, c in zip(hs, v["rmse"], v["rmse_closed_form"])]
    ref = s["H4b"]["cells"]["N_x=10/f=1/n0=10"]["rmse"][0] * (hs / hs[0]) ** -0.5
    ax.plot(hs, ref, color="0.4", linestyle="--", linewidth=0.8, label="slope −½")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("audit periods h")
    ax.set_ylabel("RMSE of q̂")
    ax.set_title("against h (N_x = 10, n₀ = 10)")
    ax.legend(loc="lower left")
    ax = axes[0][1]
    n0s = np.array(e4r.N0S, dtype=float)
    k = e4r.HS.index(20)
    for f in e4r.FACTORS:
        emp = [s["H4b"]["cells"]["N_x=10/f=%g/n0=%d" % (f, n0)]["rmse"][k] for n0 in e4r.N0S]
        closed = [s["H4b"]["cells"]["N_x=10/f=%g/n0=%d" % (f, n0)]["rmse_closed_form"][k] for n0 in e4r.N0S]
        line, = ax.plot(n0s, closed, linewidth=1.0, label="f = %g (closed form)" % f)
        ax.scatter(n0s, emp, s=14, color=line.get_color())
        rows += [("rmse_vs_n0", 10, f, int(n0), 20, e, c) for n0, e, c in zip(n0s, emp, closed)]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("audit requests per cell and period n₀")
    ax.set_ylabel("RMSE of q̂")
    ax.set_title("against n₀ (N_x = 10, h = 20; points: simulated)")
    ax.legend(loc="lower left")
    paths += save(fig, "sl21s", FIGURE_NAMES[1], ["panel", "n_x", "f", "n0", "h", "rmse", "rmse_closed_form"], rows,
                  "H4b: estimation error against h and n₀ (Prop learning(a))")

    fig, axes = figure(1, 2, 8.0, 3.4)
    rows = []
    for ax, n in zip(axes[0], e4r.N_XS):
        for f in e4r.FACTORS:
            v = s["H4c"]["cells"]["N_x=%d/f=%g/n0=10" % (n, f)]["by_h"]
            ax.plot(hs, [x["mean_max_gain"] for x in v], marker="o", markersize=3, label="max gain, f = %g" % f)
            rows += [(n, f, 10, x["h"], x["mean_max_gain"], x["mean_bound"], x["share"]) for x in v]
        v = s["H4c"]["cells"]["N_x=%d/f=1/n0=10" % n]["by_h"]
        ax.plot(hs, [x["mean_bound"] for x in v], color="0.3", linestyle="--", label="mean bound 2θ n ε₁")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("audit periods h")
        ax.set_ylabel("misreport gain (effective-choice value)")
        ax.set_title("N_x = %d, n₀ = 10" % n)
        ax.legend(loc="center right")
    paths += save(fig, "sl21s", FIGURE_NAMES[2], ["n_x", "f", "n0", "h", "mean_max_gain", "mean_bound",
                                                  "violation_share"], rows,
                  "H4c: largest misreport gain against the bound of Prop learning(d)")

    fig, axes = figure(1, 2, 8.0, 3.4)
    rows = []
    for ax, n in zip(axes[0], e4r.N_XS):
        for f in e4r.FACTORS:
            v = s["H4d"]["cells"]["N_x=%d/f=%g/n0=10" % (n, f)]["mean_cumulative_loss"]
            ax.plot(hs, v, marker="o", markersize=3, label="f = %g" % f)
            rows += [(n, f, 10, int(h), x) for h, x in zip(hs, v)]
        ax.set_xscale("log")
        ax.set_xlabel("periods h")
        ax.set_ylabel("mean cumulative loss (W units)")
        ax.set_title("N_x = %d, n₀ = 10" % n)
        ax.legend(loc="upper left")
    paths += save(fig, "sl21s", FIGURE_NAMES[3], ["n_x", "f", "n0", "h", "mean_cumulative_loss"], rows,
                  "H4d: cumulative loss of the plug-in mechanism against h")
    return [p.relative_to(REPO).as_posix() for p in paths]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    text.write("# " + LABEL + "\n")
    writer = csv.writer(text, lineterminator="\n")
    keys = list(rows[0])
    writer.writerow(keys)
    for row in rows:
        writer.writerow([json.dumps(row[k]) if isinstance(row[k], (list, dict)) else
                         repr(row[k]) if isinstance(row[k], float) else row[k] for k in keys])
    path.write_text(text.getvalue(), encoding="utf-8")


def tables(s: dict) -> None:
    _write_csv(TABLES / "h4a.csv", [{"cell": k, "design": d, **v} for k, c in s["H4a"]["cells"].items()
                                    for d, v in c["designs"].items()])
    _write_csv(TABLES / "h4b.csv", [{"cell": k, **v} for k, v in s["H4b"]["cells"].items()])
    _write_csv(TABLES / "h4c.csv", [{"cell": k, **x} for k, v in s["H4c"]["cells"].items() for x in v["by_h"]])
    _write_csv(TABLES / "h4d.csv", [{"cell": k, "mean_cumulative_loss": v["mean_cumulative_loss"],
                                     "zero_loss_share": v["zero_loss_share"], "slope": v["slope"], "ci95": v["ci95"]}
                                    for k, v in s["H4d"]["cells"].items()])
    _write_csv(TABLES / "recovery.csv", [{"cell": k, "design": d, **v} for k, c in s["recovery"]["cells"].items()
                                         for d, v in c.items()])


# ------------------------------------------------------------------- markdown
def _ci(v) -> str:
    return "[%.3f, %.3f]" % tuple(v)


def markdown(summary: dict) -> str:
    g, s = summary["gates"], summary["statistics"]
    ex = g["bound"]["example_n5_k4"]
    lines = [
        "# SL-21S E4: learning on stipulated state variance", "",
        "**%s.** Results of `python -m features.storelab.analysis.e4_run` (master seed %d), regenerated by "
        "`python -m features.storelab.analysis.e4_report`. The state variance is a stand-in (A17 §4), not an estimate; "
        "no confirmatory label is recorded." % (LABEL, e4r.MASTER_SEED), "",
        "**Review status.** SL-21S has not been independently reviewed: the owner accepts the pre-E1 tickets after "
        "SL-19S without an independent review (2026-10-24).", "",
        "## Design", "",
        "- A17 §4: one context, one model, σ = 0.3, v_j ~ N(0, 1), outside score −1, K = 4; two equally likely states, "
        "state 2 adds τ η_j with τ = 0.5 √f, f ∈ {0, 0.5, 1, 2}; N_x ∈ {10, 30}; h ∈ {5, 10, 20, 50, 100}; "
        "n₀ ∈ {2, 10, 100, 1000}; R = %d replications per cell. Designs audit 2 N_x sets: leader-inclusive, "
        "competitor-varying (every pair of the top 5 meets in a set filled from outside the top 5), and random sets "
        "matched to it on per-product inclusion counts." % e4r.REPLICATIONS,
        "- Owner decisions 2026-10-24: H4a target sets are the subsets of the top 5 products with 2–4 members, with "
        "choices in the deterministic limit state by state; the H4b check uses 8 cells fixed before the run; H4c/H4d "
        "use a menu of 16 seeded mixtures over 4 competitor-varying cells and 4 advertisers drawn uniformly, with "
        "reports on a grid of [0, 2] in steps of 0.005; recovery restarts estimation at a version change after "
        "period 50 of 100.", "",
        "## Implementation checks", "",
        "- **Cor design(b), corrected (A1 §E1)**: A(N) = ⌊e · N!⌋ equals the enumerated number of choice functions for "
        "all %d (N, K) with N ≤ 7, K = 2..4: %s. At N = 5, K = 4 the corrected bound is %d audited sets (the paper's "
        "formula gives %d). For the designs: %s." % (
            g["bound"]["checked"], "yes" if g["bound"]["passed"] else "NO", ex["corrected"], ex["paper"],
            "; ".join("N_x = %d: at least %d sets (paper %d), each design audits %d" % (
                d["n"], d["corrected_bound"], d["paper_bound"], d["audited_sets"]) for d in g["bound"]["designs"])),
        "- **H4a I-check**: the Prop identify criterion and the possible-winner rule agree on every target set in all "
        "%d (replication, design) rows and both states, and every identified choice is the true one: %s." % (
            g["h4a_icheck"]["rows"], "yes" if g["h4a_icheck"]["passed"] else "NO"), "",
        "**H4b implementation check** (Prop learning(a), 10,000 replications per cell, N_x = 10):", "",
        "| h | n₀ | f | q̄ | |bias| / MC SE | Variance / closed form | Passed |", "|---:|---:|---:|---:|---:|---:|---|"]
    for r in g["h4b_check"]["cells"]:
        lines.append("| %d | %d | %g | %.4f | %.2f | %.4f | %s |" % (r["h"], r["n0"], r["f"], r["q_bar"], r["bias_in_se"],
                                                                  r["variance_ratio"], "yes" if r["passed"] else "NO"))
    lines += ["", "All checks pass; every cell has R = %d completed replications." % e4r.REPLICATIONS
              if g["passed"] else "**A CHECK FAILED: the report is blocked.**", "",
              "## Registered statistics (exploratory, no confirmatory label)", "", "*%s.*" % LABEL, "",
              "**H4a** (identified share of target sets, mean over replications and the two states; paired differences "
              "with 95% intervals):", "",
              "| Cell | Leader-inclusive | Competitor-varying | Random (matched) | CV − LI | CV − random | Inclusion top 5 "
              "(LI / CV / random) |", "|---|---:|---:|---:|---|---|---|"]
    for key, c in s["H4a"]["cells"].items():
        d = c["designs"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f %s | %.3f %s | %.2f / %.2f / %.2f |" % (
            key, d["leader_inclusive"]["share"], d["competitor_varying"]["share"], d["random"]["share"],
            c["competitor_varying - leader_inclusive"]["mean"], _ci(c["competitor_varying - leader_inclusive"]["ci95"]),
            c["competitor_varying - random"]["mean"], _ci(c["competitor_varying - random"]["ci95"]),
            d["leader_inclusive"]["inclusion_top5"], d["competitor_varying"]["inclusion_top5"],
            d["random"]["inclusion_top5"]))
    lines += ["", "Registered comparison (A12 §4): competitor-varying minus leader-inclusive, 95%% lower bound > 0, in "
              "%d of %d cells. Competitor-varying minus random at equal inclusion counts: lower bound > 0 in %d of %d." % (
                  sum(c["competitor_varying - leader_inclusive"]["lower_bound_above_0"] for c in s["H4a"]["cells"].values()),
                  len(s["H4a"]["cells"]),
                  sum(c["competitor_varying - random"]["lower_bound_above_0"] for c in s["H4a"]["cells"].values()),
                  len(s["H4a"]["cells"])), "",
              "**H4b** (log–log slope of RMSE on h, registered: 95%% interval within [−0.55, −0.45]; the closed-form "
              "slope is exactly −½). Within the range in %d of %d (N_x, f, n₀) cells; simulated RMSE over the closed "
              "form, across all cells and h: %.3f–%.3f." % (
                  sum(v["within_registered_range"] for v in s["H4b"]["cells"].values()), len(s["H4b"]["cells"]),
                  min(min(v["ratio"]) for v in s["H4b"]["cells"].values()),
                  max(max(v["ratio"]) for v in s["H4b"]["cells"].values())),
              "", "Point slopes lie in [%.3f, %.3f]; %d of %d intervals contain −½. Of the %d intervals outside the "
              "range, %d pass its upper end (−0.45) and %d its lower end (−0.55); their widths are %.3f–%.3f (the "
              "range is 0.10 wide)%s." % (
                  min(v["slope"] for v in s["H4b"]["cells"].values()), max(v["slope"] for v in s["H4b"]["cells"].values()),
                  sum(v["ci95"][0] <= -0.5 <= v["ci95"][1] for v in s["H4b"]["cells"].values()), len(s["H4b"]["cells"]),
                  sum(not v["within_registered_range"] for v in s["H4b"]["cells"].values()),
                  sum(v["ci95"][1] > SLOPE_RANGE[1] for v in s["H4b"]["cells"].values()),
                  sum(v["ci95"][0] < SLOPE_RANGE[0] for v in s["H4b"]["cells"].values()),
                  min((v["ci95"][1] - v["ci95"][0] for v in s["H4b"]["cells"].values() if not v["within_registered_range"]),
                      default=0.0),
                  max((v["ci95"][1] - v["ci95"][0] for v in s["H4b"]["cells"].values() if not v["within_registered_range"]),
                      default=0.0),
                  ", all in cells with f > 0, where the state term dominates the error" if all(
                      "/f=0/" not in k for k, v in s["H4b"]["cells"].items() if not v["within_registered_range"]) else ""),
              "",
              "| Cell | Slope | 95% interval | In range | RMSE at h = 5 … 100 |", "|---|---:|---|---|---|"]
    for key, v in s["H4b"]["cells"].items():
        lines.append("| %s | %.3f | %s | %s | %s |" % (key, v["slope"], _ci(v["ci95"]),
                                                       "yes" if v["within_registered_range"] else "no",
                                                       ", ".join("%.4f" % x for x in v["rmse"])))
    c = s["H4c"]["cells"]
    worst = max(x["clopper_pearson_upper"] for v in c.values() for x in v["by_h"])
    lines += ["", "Dependence on n₀ is compared with the closed form (A1 §E3), not with \"flat\": see the RMSE ratios "
              "in `tables/h4b.csv` and figure 2.", "",
              "**H4c** (misreport gain against 2θ_j n ε₁, ϱ = 0.05): violations in %d of %d (replication, advertiser, h) "
              "pairs; the largest Clopper–Pearson 95%% upper bound of a cell's violation share is %.3f (registered: "
              "≤ 0.15) and every cell is within it: %s. The largest gain relative to its bound, averaged in a cell, is "
              "%.4f. The slope of the mean largest gain on log h has a 95%% upper bound below 0 in %d of %d cells." % (
                  sum(x["violations"] for v in c.values() for x in v["by_h"]),
                  sum(x["pairs"] for v in c.values() for x in v["by_h"]), worst,
                  "yes" if all(x["within_limit"] for v in c.values() for x in v["by_h"]) else "no",
                  max(x["mean_gain_over_bound"] for v in c.values() for x in v["by_h"]),
                  sum(v["upper_bound_below_0"] for v in c.values()), len(c)), "",
              "| Cell | Mean largest gain, h = 5 … 100 | Slope on log h | 95% interval |", "|---|---|---:|---|"]
    for key, v in c.items():
        lines.append("| %s | %s | %.3f | %s |" % (key, ", ".join("%.3f" % x["mean_max_gain"] for x in v["by_h"]),
                                                  v["max_gain_slope_on_log_h"], _ci(v["ci95"])))
    d = s["H4d"]["cells"]
    defined = [v for v in d.values() if v["slope"] is not None]
    lines += ["", "**H4d** (log–log slope of mean cumulative loss on h, corrected in A1 §E2: registered 95%% upper bound "
              "≤ 0.6; a slope near 0, bounded loss, is consistent with the theory). Slope defined in %d of %d cells%s; "
              "upper bound ≤ 0.6 in %d, lower bound > 0.6 in %d." % (
                  len(defined), len(d), "" if len(defined) == len(d) else " (elsewhere the mean loss is 0 at some h)",
                  sum(v["upper_bound_at_most_limit"] for v in defined),
                  sum(v["lower_bound_above_limit"] for v in defined)), "",
              "| Cell | Mean cumulative loss, h = 5 … 100 | Share with zero loss at h = 100 | Slope | 95% interval |",
              "|---|---|---:|---:|---|"]
    for key, v in d.items():
        lines.append("| %s | %s | %.3f | %s | %s |" % (key, ", ".join("%.3f" % x for x in v["mean_cumulative_loss"]),
                                                       v["zero_loss_share"],
                                                       "—" if v["slope"] is None else "%.3f" % v["slope"],
                                                       "—" if v["ci95"] is None else _ci(v["ci95"])))
    lines += ["", "## Recovery after a version change (exploratory)", "", "*%s. Exploratory analysis, not covered "
              "by the theory.*" % LABEL, "",
              "Post-change periods until the identified target-set choices are all correct for the new version and "
              "cover the pre-change share (mean / median, share censored at 50):", "",
              "| Cell | Leader-inclusive | Competitor-varying | Random (matched) |", "|---|---|---|---|"]
    for key, v in s["recovery"]["cells"].items():
        lines.append("| %s | %s |" % (key, " | ".join("%.1f / %.0f / %.2f" % (v[x]["mean_periods"], v[x]["median_periods"],
                                                                             v[x]["censored_share"]) for x in e4r.DESIGNS)))
    lines += ["", "## Figures", "", "*%s.*" % LABEL, ""]
    lines += ["- `%s`" % p for p in summary["figures"] if p.endswith(".pdf")]
    lines += ["", "Tables for every cell: `artifacts/results/sl21s/tables/`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="python -m features.storelab.analysis.e4_report").parse_args(argv)
    data = load()
    g = gates(data)
    if not g["passed"]:
        print(json.dumps(g, indent=1, default=str))
        print("a gate failed: no report is written")
        return 1
    s = statistics(data, g)
    tables(s)
    paths = figures(data, s)
    check = e4r.RESULTS / "rerun_check.json"
    summary = {"label": LABEL, "master_seed": e4r.MASTER_SEED, "gates": g, "statistics": s, "figures": paths,
               "review_status": "not independently reviewed",
               "rerun_identical": json.loads(check.read_text(encoding="utf-8"))["identical"] if check.exists() else None}
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    REPORT.write_text(markdown(summary), encoding="utf-8")
    print("report written; gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
