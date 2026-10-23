"""SL-15 validation studies on synthetic E1 data with known truth.

    python -m features.storelab.analysis.validation [--quick] [--out DIR] [--workers N]

Data come from ``simulate.py`` (the SL-06 agent model on the SL-05 design for the
e1_pilot shape: 4 tasks, v = 10, K in {2, 4, 8}, r = 10, labels 0.3, per_set).

1. **Recovery** at σ ∈ {1, 0.3}, 200 replications, with the registered estimator
   (Firth in every cell): for δ/σ and β_k/σ (k = 2..8), the coverage of the 95%
   profile penalised-likelihood interval (and of the Wald interval) and the bias
   relative to the estimator's standard error, SE = the standard deviation of the
   estimates over the replications (AC-1: coverage in [0.92, 0.98]). The ticket's
   point condition |bias| < 0.1 SE has a Monte Carlo error of 0.071 SE at 200
   replications; by owner waiver (agent-workflow/waivers/SL-15.json) it is replaced by
   (i) no check's bias significantly exceeds 0.1 SE at familywise 5%: |bias/SE| <=
   0.1 + z_{1-0.05/16} / sqrt(200) = 0.293 for each of the 16 (σ, parameter) checks
   (one-sided, Bonferroni), and (ii) the mean |bias/SE| over the 16 checks is <= 0.1.
   ``assess`` computes both from the stored estimates (``--rescore`` re-applies it to a
   written report without rerunning the study).
2. **Separation** at σ = 0.05, 200 replications: separation *occurs* when one of its
   simple sufficient patterns holds (a product never chosen, or always chosen, when
   displayed; the outside option never, or always, chosen in a task), checked from the
   counts independently of the LP; the share of those replications the LP flags, and
   whether a finite Firth fit is returned in every replication (AC-2).
3. **IIA size** under the logit (IIA holds, σ = 1) and **power** under a two-state
   mixture drawn per trial (score shifts ±1.5, which violates IIA), 1,000 replications
   each: rejection rates at α = 0.05 of the registered triple odds Monte Carlo test
   (B = 999), the 1-df Hausman–McFadden test and, for the record, the set-level
   permutation version (B = 199) (AC-3: size in [0.03, 0.07]).

All draws come from keyed streams of ``MASTER_SEED``, one per replication, so the
result does not depend on how many worker processes run them (``--workers``). The report records a sha256 of
the code it ran (``code_fingerprint``) so a stale report is detected.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import functools
import hashlib
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats

from . import clogit, iia
from .data import ChoiceData
from .simulate import Truth, simulate, synthetic_design

MASTER_SEED = 2026102215
ALPHA = 0.05
COVERAGE_RANGE, BIAS_LIMIT, SIZE_RANGE, FLAG_SHARE = (0.92, 0.98), 0.1, (0.03, 0.07), 0.95
BIAS_FAMILY_ALPHA, BIAS_MEAN_LIMIT = 0.05, 0.1  # waiver SL-15 AC-1: (i) familywise level, (ii) mean |bias/SE|
FULL = {"recovery": 200, "separation": 200, "iia": 1000, "mc": 999, "permutation": 199}
QUICK = {"recovery": 6, "separation": 4, "iia": 6, "mc": 49, "permutation": 19}
REPORT = Path(__file__).resolve().parents[1] / "docs"
CODE = ["analysis/clogit.py", "analysis/data.py", "analysis/iia.py", "analysis/simulate.py", "analysis/validation.py",
        "backends/simulated.py", "core/design.py"]


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in CODE:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest()


def _progress(label: str, done: int, total: int, start: float) -> None:
    if done % max(1, total // 10) == 0 or done == total:
        print("%s %d/%d (%.0fs)" % (label, done, total, time.time() - start), flush=True)


# ------------------------------------------------------------- parallel map
WORKERS = 1  # set by main(); replications are keyed, so the result does not depend on it


def _map(function, arguments: list[tuple], label: str) -> list:
    """``function(*a)`` for each a, in order, on WORKERS processes."""
    start, out = time.time(), []
    if WORKERS <= 1:
        for n, a in enumerate(arguments, 1):
            out.append(function(*a))
            _progress(label, n, len(arguments), start)
        return out
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        for n, result in enumerate(pool.map(function, *zip(*arguments)), 1):
            out.append(result)
            _progress(label, n, len(arguments), start)
    return out


@functools.lru_cache(maxsize=1)
def _designs():
    return synthetic_design(MASTER_SEED)


# ------------------------------------------------------------------ recovery
def _recovery_rep(sigma: float, r: int) -> tuple:
    truth = Truth(sigma=sigma)
    target = truth.on_sigma_scale()
    names = list(target)
    data = simulate(_designs(), truth, MASTER_SEED, "recovery-%g-%d" % (sigma, r))
    result = clogit.fit(data, profile=names)
    return ([result.value(n) for n in names], [result.std_error(n) for n in names],
            [result.ci[n][0] <= target[n] <= result.ci[n][1] for n in names],
            [result.wald_ci(n)[0] <= target[n] <= result.wald_ci(n)[1] for n in names],
            bool(result.separation.separated), data.n_trials)


def recovery(sigma: float, replications: int) -> dict:
    target = Truth(sigma=sigma).on_sigma_scale()
    names = list(target)
    reps = _map(_recovery_rep, [(sigma, r) for r in range(replications)], "recovery sigma=%g" % sigma)
    est, se, prof, wald = (np.array([rep[i] for rep in reps], dtype=float) for i in range(4))
    rows = []
    for k, name in enumerate(names):
        sd = float(est[:, k].std(ddof=1)) if replications > 1 else float("nan")
        bias = float(est[:, k].mean() - target[name])
        rows.append({"parameter": name, "truth": target[name], "mean": float(est[:, k].mean()), "bias": bias,
                     "sd": sd, "bias_over_se": bias / sd if sd > 0 else float("nan"),
                     "bias_mc_se_over_se": 1 / math.sqrt(replications), "mean_wald_se": float(se[:, k].mean()),
                     "profile_coverage": float(prof[:, k].mean()), "wald_coverage": float(wald[:, k].mean())})
    return {"sigma": sigma, "replications": replications, "trials": reps[0][5],
            "separated": sum(rep[4] for rep in reps), "estimator": "firth", "parameters": rows}


def bias_criterion(blocks: list[dict]) -> dict:
    """The waiver's bias test over every (σ, parameter) check of the recovery blocks:
    (i) |bias/SE| <= 0.1 + z_{1 - α/m} / sqrt(R) for each of the m checks, (ii) mean
    |bias/SE| <= 0.1. Uses only the stored bias_over_se."""
    replications = {b["replications"] for b in blocks}
    if len(replications) != 1:
        raise ValueError("recovery blocks with different replication counts")
    values = [(b["sigma"], r["parameter"], abs(r["bias_over_se"])) for b in blocks for r in b["parameters"]]
    z = float(stats.norm.ppf(1 - BIAS_FAMILY_ALPHA / len(values)))
    limit = BIAS_LIMIT + z / math.sqrt(replications.pop())
    worst = max(values, key=lambda v: v[2])
    mean = float(np.mean([v[2] for v in values]))
    each = all(v[2] <= limit for v in values)
    return {"checks": len(values), "family_alpha": BIAS_FAMILY_ALPHA, "z": z, "check_limit": limit,
            "max_abs_bias_over_se": worst[2], "max_at": {"sigma": worst[0], "parameter": worst[1]},
            "each_check_passed": each, "mean_abs_bias_over_se": mean, "mean_limit": BIAS_MEAN_LIMIT,
            "mean_passed": mean <= BIAS_MEAN_LIMIT, "passed": each and mean <= BIAS_MEAN_LIMIT,
            "point_condition_misses": sum(v[2] >= BIAS_LIMIT for v in values)}


# ---------------------------------------------------------------- separation
def separation_patterns(data: ChoiceData) -> list[str]:
    """Simple sufficient patterns of separation, from the counts alone."""
    chosen = data.chosen_product()
    found = []
    shown = np.zeros(len(data.products))
    picked = np.zeros(len(data.products))
    for row in range(data.n_trials):
        products = data.prod[row][data.prod[row] >= 0]
        shown[products] += 1
        if chosen[row] >= 0:
            picked[chosen[row]] += 1
    for index, (task, product) in enumerate(data.products):
        if shown[index] and picked[index] == 0:
            found.append("never chosen: %s/%s" % (task, product))
        elif shown[index] and picked[index] == shown[index]:
            found.append("always chosen: %s/%s" % (task, product))
    task_of = np.array([data.products[p][0] for p in data.prod[:, 0]])
    for task in data.tasks:
        outside = chosen[task_of == task] < 0
        if not outside.any():
            found.append("outside never chosen: %s" % task)
        elif outside.all():
            found.append("outside always chosen: %s" % task)
    return found


def _separation_rep(r: int) -> tuple:
    data = simulate(_designs(), Truth(sigma=0.05), MASTER_SEED, "separation-%d" % r)
    patterns = bool(separation_patterns(data))
    try:
        result = clogit.fit(data)
    except Exception:  # a failure is what AC-2 rules out; count it
        return patterns, None, False, float("nan")
    ok = bool(result.firth and result.converged and np.isfinite(result.theta).all())
    return patterns, bool(result.separation.separated), ok, result.value("delta")


def separation(replications: int) -> dict:
    truth = Truth(sigma=0.05)
    reps = _map(_separation_rep, [(r,) for r in range(replications)], "separation")
    failures = sum(flag is None for _, flag, _, _ in reps)
    occurs = sum(p for p, flag, _, _ in reps if flag is not None)
    flagged = sum(bool(p and flag) for p, flag, _, _ in reps)
    only_lp = sum(bool(flag and not p) for p, flag, _, _ in reps)
    finite = sum(ok for _, _, ok, _ in reps)
    deltas = [d for _, flag, _, d in reps if flag is not None]
    share = flagged / occurs if occurs else float("nan")
    return {"sigma": 0.05, "replications": replications, "occurs": occurs, "flagged_when_occurs": flagged,
            "flag_share": share, "flagged_without_simple_pattern": only_lp, "fit_failures": failures,
            "finite_firth_fits": finite, "truth_delta": truth.on_sigma_scale()["delta"],
            "mean_firth_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "passed": bool(occurs) and share >= FLAG_SHARE and failures == 0 and finite == replications}


# ----------------------------------------------------------------------- IIA
def _iia_rep(label: str, shift: float, r: int, mc: int, permutation: int) -> tuple:
    data = simulate(_designs(), Truth(sigma=1.0, second_state_shift=shift), MASTER_SEED, "iia-%s-%d" % (label, r))
    result = clogit.fit(data, detect=False)  # the registered (Firth) fit
    mc_test = iia.triple_odds_test(data, result, mc, MASTER_SEED, "%s-%d" % (label, r))
    permuted = iia.triple_odds_test(data, result, permutation, MASTER_SEED, "%s-%d" % (label, r), "permutation")
    return mc_test.p_value, iia.hausman_mcfadden(data).p_value, permuted.p_value, mc_test.informative


def iia_rates(label: str, truth: Truth, replications: int, mc: int, permutation: int) -> dict:
    reps = _map(_iia_rep, [(label, truth.second_state_shift, r, mc, permutation) for r in range(replications)],
                "iia %s" % label)
    names = ("triple_odds_monte_carlo", "hausman_mcfadden_1df", "triple_odds_permutation")
    rates = {name: float(np.mean(np.array([rep[i] for rep in reps]) <= ALPHA)) for i, name in enumerate(names)}
    return {"condition": label, "replications": replications, "rejection_rate": rates,
            "rate_mc_se": math.sqrt(ALPHA * (1 - ALPHA) / replications),
            "median_informative_triples": float(np.median([rep[3] for rep in reps])), "mc_replicates": mc,
            "permutation_replicates": permutation}


def run(counts: dict) -> dict:
    start = time.time()
    out = {"study": "SL-15 validation", "master_seed": MASTER_SEED, "code_fingerprint": code_fingerprint(),
           "replications": counts, "recovery": [recovery(s, counts["recovery"]) for s in (1.0, 0.3)],
           "separation": separation(counts["separation"])}
    size = iia_rates("null", Truth(sigma=1.0), counts["iia"], counts["mc"], counts["permutation"])
    power = iia_rates("mixture", Truth(sigma=1.0, second_state_shift=1.5), counts["iia"], counts["mc"],
                      counts["permutation"])
    out.update(iia_size=size, iia_power=power, seconds=round(time.time() - start, 1))
    return assess(out)


def assess(report: dict) -> dict:
    """Set the pass/fail fields of a report from its stored estimates and rates."""
    for block in report["recovery"]:
        for stale in ("bias_passed", "passed"):  # the ticket's point condition, replaced by bias_criterion
            block.pop(stale, None)
        block["coverage_passed"] = all(COVERAGE_RANGE[0] <= r["profile_coverage"] <= COVERAGE_RANGE[1]
                                       for r in block["parameters"])
    report["bias"] = bias_criterion(report["recovery"])
    size, registered = report["iia_size"], ("triple_odds_monte_carlo", "hausman_mcfadden_1df")
    size["passed"] = all(SIZE_RANGE[0] <= size["rejection_rate"][t] <= SIZE_RANGE[1] for t in registered)
    report["passed"] = {"AC-1 coverage": all(r["coverage_passed"] for r in report["recovery"]),
                        "AC-1 bias": report["bias"]["passed"], "AC-2": report["separation"]["passed"],
                        "AC-3": size["passed"]}
    return report


def markdown(report: dict) -> str:
    lines = ["# SL-15 analysis validation", "",
             "Generated by `python -m features.storelab.analysis.validation` (master seed %d, code fingerprint `%s`, "
             "%.0f s). Synthetic data from the SL-06 agent model on the SL-05 design in the e1_pilot shape (4 tasks, "
             "v = 10, K in {2, 4, 8}, r = 10, label rate 0.3, per_set; %d trials per data set). Truth: scores evenly "
             "spaced in [3.6, 4.8], outside 4.0, delta = -0.5, beta = (0, -0.1, -0.2, -0.3, -0.35, -0.4, -0.45, -0.5), "
             "all divided by sigma." % (report["master_seed"], report["code_fingerprint"][:12], report["seconds"],
                                        report["recovery"][0]["trials"]), "",
             "## Recovery (AC-1)", "",
             "Registered estimator: Firth's penalised likelihood in every cell; intervals are profile penalised "
             "likelihood. SE is the standard deviation of the estimates over the replications. The Monte Carlo error of "
             "bias/SE is 1/sqrt(R) = %.3f, and of a coverage 0.95 is %.3f, so with 16 parameter checks per criterion "
             "a correct estimator can still miss a band by chance." % (
                 report["recovery"][0]["parameters"][0]["bias_mc_se_over_se"],
                 math.sqrt(0.95 * 0.05 / report["recovery"][0]["replications"])), ""]
    for block in report["recovery"]:
        lines += ["**sigma = %g** (%d replications; estimator Firth; separation in %d): coverage %s." % (
            block["sigma"], block["replications"], block["separated"],
            "passed" if block["coverage_passed"] else "FAILED"), "",
            "| Parameter | Truth | Mean | Bias / SE | Profile coverage | Wald coverage | SE (sd) | Mean Wald SE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in block["parameters"]:
            lines.append("| %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %.4f | %.4f |" % (
                r["parameter"], r["truth"], r["mean"], r["bias_over_se"], r["profile_coverage"], r["wald_coverage"],
                r["sd"], r["mean_wald_se"]))
        lines.append("")
    b = report["bias"]
    lines += ["**Bias** (owner waiver, agent-workflow/waivers/SL-15.json, in place of the point condition |bias| < 0.1 "
              "SE, which %d of the %d checks miss): (i) every check has |bias/SE| <= 0.1 + %.3f / sqrt(%d) = %.3f "
              "(one-sided, Bonferroni over %d at familywise %g); largest %.3f (%s at sigma = %g): %s. (ii) Mean |bias/SE| "
              "over the %d checks %.3f <= %.1f: %s. Bias %s." % (
                  b["point_condition_misses"], b["checks"], b["z"], report["recovery"][0]["replications"],
                  b["check_limit"], b["checks"], b["family_alpha"], b["max_abs_bias_over_se"], b["max_at"]["parameter"],
                  b["max_at"]["sigma"], "passed" if b["each_check_passed"] else "FAILED", b["checks"],
                  b["mean_abs_bias_over_se"], b["mean_limit"], "passed" if b["mean_passed"] else "FAILED",
                  "passed" if b["passed"] else "FAILED"), ""]
    s = report["separation"]
    lines += ["## Separation (AC-2)", "",
              "sigma = 0.05, %d replications. Separation occurred (a simple pattern held) in %d; the LP flagged %d of "
              "them (%.3f) and %d more with no simple pattern. Fit failures: %d. Finite, converged Firth fits: %d of "
              "%d. Mean Firth delta/sigma %.2f (truth %.1f): %s." % (
                  s["replications"], s["occurs"], s["flagged_when_occurs"], s["flag_share"],
                  s["flagged_without_simple_pattern"], s["fit_failures"], s["finite_firth_fits"], s["replications"],
                  s["mean_firth_delta"], s["truth_delta"], "passed" if s["passed"] else "FAILED"), "",
              "## IIA tests: size (AC-3) and power", "",
              "Rejection rates at alpha = 0.05 over %d replications (Monte Carlo SE %.3f). Size: data from the logit "
              "(IIA holds). Power: a two-state mixture drawn per trial (score shifts +-1.5). Registered: the triple "
              "odds Monte Carlo test (B = %d) and the 1-df Hausman-McFadden test; the set-level permutation version "
              "(B = %d) is shown because it is anti-conservative." % (
                  report["iia_size"]["replications"], report["iia_size"]["rate_mc_se"],
                  report["iia_size"]["mc_replicates"], report["iia_size"]["permutation_replicates"]), "",
              "| Test | Size | Power |", "|---|---:|---:|"]
    for name in ("triple_odds_monte_carlo", "hausman_mcfadden_1df", "triple_odds_permutation"):
        lines.append("| %s | %.3f | %.3f |" % (name, report["iia_size"]["rejection_rate"][name],
                                                report["iia_power"]["rejection_rate"][name]))
    lines += ["", "Median informative triples per data set: %.0f (null), %.0f (mixture). Size %s." % (
        report["iia_size"]["median_informative_triples"], report["iia_power"]["median_informative_triples"],
        "passed" if report["iia_size"]["passed"] else "FAILED"), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.validation")
    parser.add_argument("--quick", action="store_true", help="a few replications (code check, not the study)")
    parser.add_argument("--out", type=Path, help="output directory (default: features/storelab/docs)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes (default: all cores)")
    parser.add_argument("--rescore", action="store_true",
                        help="re-apply the criteria to the report in --out from its stored estimates (no study run)")
    args = parser.parse_args(argv)
    out = args.out or REPORT
    if args.rescore:
        report = assess(json.loads((out / "analysis_validation.json").read_text(encoding="utf-8")))
        _write(report, out)
        return 0
    global WORKERS
    WORKERS = args.workers
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"  # one BLAS thread per worker process (spawned workers read it at import)
    _write(run(QUICK if args.quick else FULL), out)
    return 0


def _write(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, sort_keys=True, default=lambda value: value.item())  # numpy scalars
    (out / "analysis_validation.json").write_text(text + "\n", encoding="utf-8")
    (out / "analysis_validation.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["passed"], default=lambda value: value.item()))


if __name__ == "__main__":
    raise SystemExit(main())
