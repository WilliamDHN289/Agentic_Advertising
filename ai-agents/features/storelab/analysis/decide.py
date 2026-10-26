"""Decision labels from a results file and the claims register (SL-01).

    python -m features.storelab.analysis.decide RESULTS.json [--register PATH] [--out PATH]

The results file gives, per claim and group, one entry per cell with each test's
statistics (``ResultsFile``; schema ``contracts/storelab/analysis_results.schema.json``).
Nothing is decided by hand: ``decide`` applies the registered rules
(``docs/claims_register.yaml``, conventions and A12):

1. a cell below the group's floor, or with more than 25% missing trials, is left out
   of the family and labelled inconclusive;
2. for each test, the one-sided p-values ``p_support`` and ``p_against`` are read, or
   computed from counts (exact binomial for Clopper–Pearson rules, the score test for
   Wilson rules; each is the exact dual of the registered interval bound), and Holm's
   step-down adjustment is applied within the family;
3. a test is supported when adjusted ``p_support`` ≤ ``alpha_support`` (< for a
   strict threshold), not supported when adjusted ``p_against`` < ``alpha_against``;
   ``rejects_null`` is never "not supported"; a failed ``implementation_check`` is
   inconclusive and flags an implementation error. A ``monte_carlo`` check (A12 §6.2)
   fails only if an estimate is beyond its bound by more than z·SE_MC, z = z_{1-0.05/m}
   over the test's m Monte Carlo estimates. A ``contains`` test (A12 §6.3) needs each
   cell's Bonferroni interval, at level 1 - alpha/m over the m cells, to contain the
   exact value: too wide, it needs the one rerun at ``rerun_factor`` times the
   replications and is inconclusive if still too wide; excluding the value is an
   implementation error;
4. a cell is supported if all its tests are, not supported if any is not; a cell
   with 10–25% missing (or differential missingness) keeps its label only if the
   sensitivity analysis gives the same one;
5. cells are aggregated (two_thirds, all, single) and primary groups combined.

The H1d alignment test (``first_credible_change``) takes the credible changes
themselves and derives its counts: each (provider version pair, task) counts once,
through the first interface in text → structured → browser_tool order that shows a
credible change, ties in u excluded, pooled across providers.

Inconsistent input (a missing claim, group, cell field or test; both directions at
once) raises ``ValueError``: a label is never produced from incomplete results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, ValidationError

from .register import (CLAIM_IDS, REGISTER_PATH, BoundTest, Claim, ContainsTest, Group, ImplementationCheck,
                       Name, Register, RejectsNullTest, ShareAtLeast, Strict, load_register, register_sha256)

RESULTS_VERSION = 1
MISSING_FLAG, MISSING_EXCLUDE = 0.10, 0.25  # A12 §3.4
MC_ALPHA = 0.05  # A12 §6.2: familywise level of the Monte Carlo tolerances of one criterion
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


# ------------------------------------------------------------------ results
class CredibleChange(Strict):
    """One credible favourite change (H1d alignment): its unit, the interface that
    showed it and whether the new favourite has the higher u (null: a tie in u)."""

    pair: Annotated[str, Field(min_length=1)]  # provider version pair
    task: Annotated[str, Field(min_length=1)]
    interface: Literal["text", "structured", "browser_tool"]
    higher_u: bool | None


class McEstimate(Strict):
    """One Monte Carlo estimate of a ``monte_carlo`` check, with its Monte Carlo SE."""

    estimate: float
    se: Annotated[float, Field(ge=0.0)]


class Rerun(Strict):
    """The one rerun of a ``contains`` cell whose interval was too wide (A12 §6.3)."""

    n: Annotated[int, Field(gt=0)]  # replications
    ci: tuple[float, float]
    ci_level: Annotated[float, Field(gt=0.0, lt=1.0)]


class StatResult(Strict):
    p_support: Probability | None = None
    p_against: Probability | None = None
    p_value: Probability | None = None
    n: NonNegativeInt | None = None  # trials of a count rule; replications behind a contains interval
    counts: dict[Name, NonNegativeInt] = {}
    metrics: dict[Name, float] = {}
    monte_carlo: dict[Name, list[McEstimate]] = {}  # the estimates of a monte_carlo check
    share: Probability | None = None
    changes: list[CredibleChange] = []  # a first_credible_change rule's input
    threshold: float | None = None  # the value of a registered threshold_rule
    estimate: float | None = None  # reported only
    ci: tuple[float, float] | None = None  # reported only, except by a contains test
    ci_level: Annotated[float, Field(gt=0.0, lt=1.0)] | None = None  # a contains test's interval level
    rerun: Rerun | None = None  # a contains test's rerun


class CellResult(Strict):
    cell: Annotated[str, Field(min_length=1)]
    n: float | None = None
    power: Probability | None = None
    missing_rate: Probability | None = None
    differential_missingness: bool = False
    tests: dict[Name, StatResult]
    sensitivity: dict[Name, StatResult] | None = None  # missing trials coded as the outside option


class ResultsFile(Strict):
    results_version: Literal[1]
    register_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    claims: dict[Literal[CLAIM_IDS], dict[Name, list[CellResult]]]


def results_schema() -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **ResultsFile.model_json_schema()}


# ------------------------------------------------------------- statistics
def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    k, adjusted, running = len(p_values), [0.0] * len(p_values), 0.0
    for rank, i in enumerate(sorted(range(k), key=p_values.__getitem__)):
        running = max(running, min(1.0, (k - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _log_pmf(i: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for X ~ Bin(n, p)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    terms = [_log_pmf(i, n, p) for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X ≥ k)."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    terms = [_log_pmf(i, n, p) for i in range(k, n + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_ppf(q: float) -> float:
    """The q quantile of N(0, 1), by bisection on ``norm_sf`` (0 < q < 1)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if norm_sf(mid) > 1.0 - q else (lo, mid)
    return (lo + hi) / 2


def count_p_values(test: BoundTest, result: StatResult, where: str) -> tuple[float, float]:
    """(p_support, p_against) of a Clopper–Pearson or Wilson rule from counts."""
    n = result.n
    if n is None or n == 0 or test.support_count not in result.counts or test.against_count not in result.counts:
        raise ValueError("%s: needs n > 0 and counts %s, %s" % (where, test.support_count, test.against_count))
    k_s, k_a = result.counts[test.support_count], result.counts[test.against_count]
    if k_s > n or k_a > n:
        raise ValueError("%s: a count exceeds n" % where)
    t_s = test.threshold
    t_a = test.against_threshold if test.against_threshold is not None else t_s
    above = test.side == "above"
    if test.method == "clopper_pearson":
        # lower bound >= t  <=>  P(X >= k | t) <= a;  upper bound < t  <=>  P(X <= k | t) < a
        p_support = binom_sf(k_s, n, t_s) if above else binom_cdf(k_s, n, t_s)
        p_against = binom_cdf(k_a, n, t_a) if above else binom_sf(k_a, n, t_a)
        return p_support, p_against
    for t in (t_s, t_a):
        if not 0.0 < t < 1.0:
            raise ValueError("%s: a Wilson rule needs a threshold in (0, 1)" % where)

    def z(k: int, t: float) -> float:  # the score statistic, whose inversion is the Wilson interval
        return (k / n - t) / math.sqrt(t * (1.0 - t) / n)

    if above:
        return norm_sf(z(k_s, t_s)), norm_sf(-z(k_a, t_a))
    return norm_sf(-z(k_s, t_s)), norm_sf(z(k_a, t_a))


def first_credible_changes(test: BoundTest, result: StatResult, where: str) -> tuple[StatResult, dict]:
    """``result`` with n and the aligned count derived per ``test.first_credible_change``,
    and a summary of the units kept and dropped."""
    order = test.first_credible_change.order
    if result.n is not None or result.counts:
        raise ValueError("%s: counts are derived from the changes; give changes only" % where)
    seen: dict[tuple[str, str, str], CredibleChange] = {}
    for change in result.changes:
        key = (change.pair, change.task, change.interface)
        if key in seen:
            raise ValueError("%s: change of pair %s, task %s listed twice for %s" % (where, *key))
        if change.interface not in order:
            raise ValueError("%s: interface %s is not in the registered order" % (where, change.interface))
        seen[key] = change
    first: dict[tuple[str, str], CredibleChange] = {}
    for change in sorted(result.changes, key=lambda c: order.index(c.interface)):
        first.setdefault((change.pair, change.task), change)
    kept = [c for c in first.values() if c.higher_u is not None]
    derived = result.model_copy(update={"n": len(kept), "counts": {"aligned": sum(c.higher_u for c in kept)}})
    return derived, {"changes": len(result.changes), "units": len(first), "ties_excluded": len(first) - len(kept),
                     "n": derived.n, "aligned": derived.counts["aligned"],
                     "by_interface": {i: sum(c.interface == i for c in kept) for i in order}}


# ---------------------------------------------------------------- decisions
def _p_pair(test, result: StatResult, where: str) -> tuple[float, float | None]:
    if isinstance(test, RejectsNullTest):
        if result.p_value is None:
            raise ValueError("%s: needs p_value" % where)
        return result.p_value, None
    if isinstance(test, BoundTest) and test.method != "p_values":
        return count_p_values(test, result, where)
    if result.p_support is None or result.p_against is None:
        raise ValueError("%s: needs p_support and p_against" % where)
    if isinstance(test, BoundTest) and test.threshold_rule is not None and result.threshold is None:
        raise ValueError("%s: needs the threshold value (%s)" % (where, test.threshold_rule))
    return result.p_support, result.p_against


def _check(op: str, value: float, bound: float) -> bool:
    return value <= bound if op == "<=" else value >= bound if op == ">=" else value == bound


def _implementation_check(test: ImplementationCheck, result: StatResult, where: str) -> dict:
    """Every check of one cell; ``monte_carlo`` checks net of Monte Carlo error (A12 §6.2),
    reported with the raw tolerance and the adjusted bound."""
    failed, adjusted = [], {}
    monte_carlo = [c for c in test.checks if c.monte_carlo]
    for check in monte_carlo:
        if not result.monte_carlo.get(check.metric):
            raise ValueError("%s: Monte Carlo estimates of %s missing" % (where, check.metric))
    m = sum(len(result.monte_carlo[c.metric]) for c in monte_carlo)
    z = norm_ppf(1.0 - MC_ALPHA / m) if m else None
    for check in test.checks:
        if not check.monte_carlo:
            if check.metric not in result.metrics:
                raise ValueError("%s: metric %s missing" % (where, check.metric))
            if not _check(check.op, result.metrics[check.metric], check.bound):
                failed.append("%s = %g, not %s %g" % (check.metric, result.metrics[check.metric], check.op,
                                                     check.bound))
            continue
        sign = 1.0 if check.op == "<=" else -1.0
        estimates = result.monte_carlo[check.metric]
        worst = max(estimates, key=lambda e: sign * (e.estimate - check.bound) - z * e.se)
        bound = check.bound + sign * z * worst.se
        adjusted[check.metric] = {"raw_bound": check.bound, "z": z, "m": m,
                                  "raw_pass": all(_check(check.op, e.estimate, check.bound) for e in estimates),
                                  "worst_estimate": worst.estimate, "worst_se": worst.se, "adjusted_bound": bound}
        if not _check(check.op, worst.estimate, bound):
            failed.append("%s = %g, not %s %g + %s %.3f x %g (Monte Carlo-adjusted)" % (
                check.metric, worst.estimate, check.op, check.bound, "+" if sign > 0 else "-", z, worst.se))
    entry = {"label": "inconclusive" if failed else "supported", "implementation_error": bool(failed),
             "failed_checks": failed}
    if adjusted:
        entry["monte_carlo"] = adjusted
    return entry


def _contains(test: ContainsTest, result: StatResult, level: float, where: str) -> dict:
    """A12 §6.3 for one cell: the Bonferroni interval must contain ``test.value``."""
    def read(ci, ci_level, what):
        if ci is None or ci_level is None:
            raise ValueError("%s: needs the %s interval and its level" % (where, what))
        if abs(ci_level - level) > 1e-9:
            raise ValueError("%s: the %s interval has level %g; Bonferroni over the cells needs %g"
                             % (where, what, ci_level, level))
        lo, hi = ci
        if lo > hi:
            raise ValueError("%s: the %s interval is reversed" % (where, what))
        return lo <= test.value <= hi, (hi - lo) / 2

    if result.n is None or result.n == 0:
        raise ValueError("%s: needs n, the replications behind the interval" % where)
    contains, half = read(result.ci, result.ci_level, "first")
    entry = {"ci": list(result.ci), "ci_level": level, "half_width": half, "rerun": None}
    if contains and half > test.max_half_width:
        rerun = result.rerun
        if rerun is None:
            raise ValueError("%s: half-width %g > %g; needs the rerun with %d x %d replications"
                             % (where, half, test.max_half_width, test.rerun_factor, result.n))
        if rerun.n != test.rerun_factor * result.n:
            raise ValueError("%s: the rerun has %d replications, not %d x %d" % (where, rerun.n, test.rerun_factor,
                                                                                result.n))
        contains, half = read(rerun.ci, rerun.ci_level, "rerun")
        entry["rerun"] = {"n": rerun.n, "ci": list(rerun.ci), "half_width": half}
    elif result.rerun is not None:
        raise ValueError("%s: a rerun is only run when the first interval contains %g and is too wide"
                         % (where, test.value))
    if not contains:
        entry.update(label="inconclusive", implementation_error=True,
                     failed_checks=["the interval excludes %g" % test.value])
    elif half > test.max_half_width:
        entry.update(label="inconclusive", implementation_error=False,
                     reason="precision too low: half-width %g > %g after the rerun" % (half, test.max_half_width))
    else:
        entry.update(label="supported", implementation_error=False)
    return entry


def _test_labels(group: Group, test, cells: list[tuple[str, StatResult]], claim_id: str) -> dict[str, dict]:
    """Label of one test in every eligible cell, with Holm (or, for a contains test,
    Bonferroni) across the cells."""
    out: dict[str, dict] = {}
    if isinstance(test, ImplementationCheck):
        for cell, result in cells:
            out[cell] = _implementation_check(test, result, "%s/%s/%s/%s" % (claim_id, group.name, cell, test.name))
        return out
    if isinstance(test, ContainsTest):
        level = 1.0 - test.alpha / len(cells)
        for cell, result in cells:
            out[cell] = _contains(test, result, level, "%s/%s/%s/%s" % (claim_id, group.name, cell, test.name))
        return out
    if isinstance(test, ShareAtLeast):
        for cell, result in cells:
            if result.share is None:
                raise ValueError("%s/%s/%s/%s: needs share" % (claim_id, group.name, cell, test.name))
            out[cell] = {"label": "supported" if result.share >= test.threshold else "not_supported",
                         "share": result.share}
        return out
    pairs = [_p_pair(test, result, "%s/%s/%s/%s" % (claim_id, group.name, cell, test.name)) for cell, result in cells]
    supports = [p for p, _ in pairs]
    againsts = [p for _, p in pairs]
    if group.multiplicity == "holm":
        supports = holm(supports)
        if not isinstance(test, RejectsNullTest):
            againsts = holm(againsts)
    strict = isinstance(test, BoundTest) and test.strict
    for (cell, result), p_s, p_a in zip(cells, supports, againsts):
        supported = p_s < test.alpha_support if strict else p_s <= test.alpha_support
        against = p_a is not None and p_a < test.alpha_against
        if supported and against:
            raise ValueError("%s/%s/%s/%s: supported and not supported at once; inconsistent p-values"
                             % (claim_id, group.name, cell, test.name))
        label = "supported" if supported else "not_supported" if against else "inconclusive"
        entry = {"label": label, "p_support_adjusted": p_s, "p_against_adjusted": p_a}
        if result.threshold is not None:
            entry["threshold"] = result.threshold
        if result.estimate is not None:
            entry["estimate"] = result.estimate
        out[cell] = entry
    return out


def _combine(labels: list[str]) -> str:
    if labels and all(label == "supported" for label in labels):
        return "supported"
    return "not_supported" if "not_supported" in labels else "inconclusive"


def _cell_labels(group: Group, cells: list[tuple[str, dict[str, StatResult]]], claim_id: str) -> dict[str, dict]:
    by_test = {t.name: _test_labels(group, t, [(c, r[t.name]) for c, r in cells], claim_id) for t in group.tests}
    out = {}
    for cell, _ in cells:
        tests = {name: by_test[name][cell] for name in by_test}
        out[cell] = {"label": _combine([t["label"] for t in tests.values()]), "tests": tests,
                     "implementation_error": any(t.get("implementation_error") for t in tests.values())}
    return out


def _aggregate(aggregation: str, labels: list[str]) -> str:
    if aggregation == "single":
        return labels[0]
    if aggregation == "all":
        return _combine(labels)
    if not labels:
        return "inconclusive"
    if sum(label == "supported" for label in labels) >= 2 * len(labels) / 3:
        return "supported"
    if sum(label == "not_supported" for label in labels) > len(labels) / 3:
        return "not_supported"
    return "inconclusive"


def decide_group(group: Group, cells: list[CellResult], claim_id: str) -> dict:
    where = "%s/%s" % (claim_id, group.name)
    cells = list(cells)  # cells with derived counts replace their entries here, not in the caller's results
    if not cells:
        raise ValueError("%s: no cells" % where)
    if group.aggregation == "single" and len(cells) != 1:
        raise ValueError("%s: aggregation single needs exactly one cell, got %d" % (where, len(cells)))
    names = [c.cell for c in cells]
    if len(set(names)) != len(names):
        raise ValueError("%s: duplicate cells" % where)
    expected = {t.name for t in group.tests}
    report: dict[str, dict] = {}
    eligible = []
    units: dict[str, dict] = {}
    for i, c in enumerate(cells):
        if set(c.tests) != expected:
            raise ValueError("%s/%s: tests %s, registered %s" % (where, c.cell, sorted(c.tests), sorted(expected)))
        derived = {}
        for t in group.tests:
            if isinstance(t, BoundTest) and t.first_credible_change is not None:
                derived[t.name], units[c.cell] = first_credible_changes(t, c.tests[t.name], "%s/%s/%s" % (
                    where, c.cell, t.name))
        if derived:
            n = next(iter(derived.values())).n
            if c.n is not None and c.n != n:
                raise ValueError("%s/%s: n = %g, but the changes give %d units" % (where, c.cell, c.n, n))
            c = cells[i] = c.model_copy(update={"n": n, "tests": {**c.tests, **derived}})
        if group.floor is not None:
            value = getattr(c, group.floor.field, None)
            if value is None:
                raise ValueError("%s/%s: the floor needs %s" % (where, c.cell, group.floor.field))
            if value < group.floor.min:
                report[c.cell] = {"label": "inconclusive", "reason": "below the floor: %s = %g < %g"
                                  % (group.floor.field, value, group.floor.min)}
                continue
        if group.missingness:
            if c.missing_rate is None or c.sensitivity is None or set(c.sensitivity) != expected:
                raise ValueError("%s/%s: the missingness rules need missing_rate and a sensitivity result for "
                                 "every test" % (where, c.cell))
            if c.missing_rate > MISSING_EXCLUDE:
                report[c.cell] = {"label": "inconclusive", "reason": "excluded: missing rate %g > %g"
                                  % (c.missing_rate, MISSING_EXCLUDE)}
                continue
        eligible.append(c)
    primary = _cell_labels(group, [(c.cell, c.tests) for c in eligible], claim_id) if eligible else {}
    sensitivity = (_cell_labels(group, [(c.cell, c.sensitivity) for c in eligible], claim_id)
                   if eligible and group.missingness else {})
    for c in eligible:
        entry = primary[c.cell]
        if group.missingness:
            entry["sensitivity_label"] = sensitivity[c.cell]["label"]
            flagged = c.missing_rate > MISSING_FLAG
            differential = group.differential_missingness and c.differential_missingness
            if (flagged or differential) and entry["sensitivity_label"] != entry["label"]:
                entry["reason"] = ("missing rate %g > %g" % (c.missing_rate, MISSING_FLAG) if flagged else
                                   "differential missingness") + "; the sensitivity analysis gives %s" % (
                    entry["sensitivity_label"])
                entry["label"] = "inconclusive"
        report[c.cell] = entry
    for cell, summary in units.items():
        report[cell]["units"] = summary
    labels = [report[name]["label"] for name in names]
    out = {"role": group.role, "aggregation": group.aggregation, "label": _aggregate(group.aggregation, labels),
           "implementation_error": any(r.get("implementation_error") for r in report.values()),
           "cells": {name: report[name] for name in names}}
    if group.labels:
        out["display"] = group.labels[out["label"]]
    return out


def decide_claim(claim: Claim, results: dict[str, list[CellResult]]) -> dict:
    expected = {g.name for g in claim.decision.groups}
    if set(results) != expected:
        raise ValueError("%s: groups %s, registered %s" % (claim.id, sorted(results), sorted(expected)))
    groups = {g.name: decide_group(g, results[g.name], claim.id) for g in claim.decision.groups}
    error = any(g["implementation_error"] for g in groups.values())
    label = _combine([groups[g.name]["label"] for g in claim.decision.groups if g.role == "primary"])
    return {"label": "inconclusive" if error else label, "implementation_error": error,
            "blocks_reporting": error, "kind": claim.kind, "experiment": claim.experiment,
            "interpretation": getattr(claim.interpretation, "inconclusive" if error else label), "groups": groups}


def decide(results: ResultsFile, register: Register, register_digest: str | None = None) -> dict:
    """A label for every claim of the register; ValueError on incomplete results."""
    if results.register_sha256 is not None and register_digest is not None and results.register_sha256 != register_digest:
        raise ValueError("results were produced for register %s, not %s" % (results.register_sha256[:12],
                                                                             register_digest[:12]))
    missing = [c.id for c in register.claims if c.id not in results.claims]
    if missing:
        raise ValueError("no results for %s" % missing)
    return {"register_sha256": register_digest, "claims": {c.id: decide_claim(c, results.claims[c.id])
                                                           for c in register.claims}}


def decide_file(results_path: Path, register_path: Path | None = None) -> dict:
    register_path = Path(register_path or REGISTER_PATH)
    data = Path(results_path).read_bytes()
    out = decide(ResultsFile.model_validate_json(data), load_register(register_path), register_sha256(register_path))
    out["results_sha256"] = hashlib.sha256(data).hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.decide")
    parser.add_argument("results", type=Path)
    parser.add_argument("--register", type=Path, help="claims register (default docs/claims_register.yaml)")
    parser.add_argument("--out", type=Path, help="write the labels as JSON here")
    args = parser.parse_args(argv)
    try:
        labels = decide_file(args.results, args.register)
    except (ValidationError, ValueError, OSError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    text = json.dumps(labels, indent=1, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({claim: entry["label"] + (" (implementation error)" if entry["implementation_error"] else "")
                      for claim, entry in labels["claims"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
