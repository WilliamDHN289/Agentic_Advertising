"""SL-22 synthetic validation of the E5 analysis.

    python -m features.storelab.analysis.e5_validation [--quick] [--out DIR] [--workers N]

Synthetic data stand in for the partner's baselines (``BASELINE``, protocol §8): each
request is randomised to the strong-competitor arm with π = 0.5; traffic is human with
probability 0.6, otherwise agent. The focal product's qualifying event (effective choice)
has rate 0.30 in the weak arm and 0.22 in the strong arm (strong competitors take
choices). A purchase follows a choice with probability 0.55 and is returned with
probability 0.10, so net purchases per choice are 0.495 in both arms. Humans request
the page of a product they choose, and others with probability 0.15. Agents request
each displayed page with probability 0.5, whatever they choose.

**H5a** (AC-1), 1,000 datasets per condition, 10,000 randomisation draws each:
- size of the Fisher randomisation test of τ = 0 at α = 0.05 when nothing differs
  between arms (sharp null) and when only choice rates differ (ratios equal);
- size of the registered TOST at the margin (true τ = ±0.10): the rate of a
  *supported* label;
- power at the designed effect (τ = 0, n from the power formula for 80%) against the
  analytic target.

AC-1 requires each size in [0.03, 0.07] and the power within ±5 points of the target.

**H5b** (reported), 1,000 datasets: the rate of a supported cell when event and click
predict equally well (size of the one-sided 0.025 test), and power at a designed
advantage against ``h5b_power``.

**Contract** (AC-2): the schema fixtures validate, and the PII field scan finds nothing
in the schema or the fixtures (and catches planted fields).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import jsonschema
import numpy as np

from ..core.rng import stream
from . import e5

MASTER_SEED = 2026102322
ALPHA, SIZE_RANGE, POWER_TOLERANCE, POWER_TARGET = 0.05, (0.03, 0.07), 0.05, 0.80
BASELINE = {"pi": 0.5, "human_share": 0.6, "choice_weak": 0.30, "choice_strong": 0.22, "purchase_per_choice": 0.55,
            "return_rate": 0.10, "human_other_click": 0.15, "agent_click": 0.5}
FULL = {"datasets": 1000, "draws": e5.DRAWS, "h5b_datasets": 1000, "bootstrap": e5.BOOTSTRAP}
QUICK = {"datasets": 20, "draws": 999, "h5b_datasets": 20, "bootstrap": 499}
REPORT = Path(__file__).resolve().parents[1] / "docs"
CONTRACT = Path(__file__).resolve().parents[3] / "contracts" / "storelab" / "e5_event.schema.json"
FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "features" / "storelab" / "fixtures" / "e5_events.jsonl"
CODE = ["analysis/e5.py", "analysis/e5_validation.py", "analysis/intervals.py"]


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in CODE:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest()


def net_per_choice(baseline: dict) -> float:
    return baseline["purchase_per_choice"] * (1 - baseline["return_rate"])


def designed_n() -> int:
    b = BASELINE
    return e5.h5a_required_n(POWER_TARGET, b["pi"], b["choice_weak"], b["choice_strong"], net_per_choice(b))


def synthetic_table(rng: np.random.Generator, n: int, tau: float = 0.0, choice_strong: float | None = None,
                    baseline: dict = BASELINE) -> e5.RequestTable:
    """Requests from the module docstring's model; the strong arm's net purchases per
    choice are e^τ times the weak arm's (by its return rate when τ ≠ 0)."""
    b = baseline
    strong = rng.random(n) < b["pi"]
    human = rng.random(n) < b["human_share"]
    rate = np.where(strong, b["choice_strong"] if choice_strong is None else choice_strong, b["choice_weak"])
    choice = rng.random(n) < rate
    purchase = choice & (rng.random(n) < b["purchase_per_choice"])
    kept_strong = min(1.0, (1 - b["return_rate"]) * math.exp(tau))
    keep = np.where(strong, kept_strong, 1 - b["return_rate"])
    net = purchase & (rng.random(n) < keep)
    click = np.where(human, choice | (rng.random(n) < b["human_other_click"]), rng.random(n) < b["agent_click"])
    return e5.RequestTable.from_arrays(strong, b["pi"], np.where(human, "human", "agent"), choice, click, purchase,
                                       net)


# ---------------------------------------------------------------- H5a study
CONDITIONS = {
    "sharp_null": {"tau": 0.0, "choice_strong": BASELINE["choice_weak"]},  # nothing differs between arms
    "weak_null": {"tau": 0.0},  # choice rates differ, ratios equal
    "margin_upper": {"tau": e5.MARGIN},
    "margin_lower": {"tau": -e5.MARGIN},
}


def _h5a_dataset(args: tuple) -> dict:
    condition, index, n, draws = args
    spec = CONDITIONS[condition]
    rng = stream(MASTER_SEED, "e5-synthetic", condition, index)
    table = synthetic_table(rng, n, spec["tau"], spec.get("choice_strong"))
    result = e5.h5a(table, draws=draws, seed=MASTER_SEED, analysis_id="%s-%d" % (condition, index))
    return {"estimate": result.estimate, "p_no_effect": result.p_no_effect, "p_support": result.p_support,
            "p_against": result.p_against, "covered": result.ci95[0] <= spec["tau"] <= result.ci95[1]}


def h5a_study(datasets: int, draws: int) -> dict:
    n = designed_n()
    b = BASELINE
    out = {"n": n, "draws": draws, "datasets": datasets, "conditions": {}}
    for condition in CONDITIONS:
        rows = _map(_h5a_dataset, [(condition, i, n, draws) for i in range(datasets)], "h5a %s" % condition)
        out["conditions"][condition] = {
            "fisher_rejection_rate": float(np.mean([r["p_no_effect"] <= ALPHA for r in rows])),
            "supported_rate": float(np.mean([r["p_support"] <= ALPHA for r in rows])),
            "not_supported_rate": float(np.mean([r["p_against"] < ALPHA for r in rows])),
            "ci95_coverage": float(np.mean([r["covered"] for r in rows])),
            "mean_estimate": float(np.mean([r["estimate"] for r in rows]))}
    analytic = e5.h5a_power(n, b["pi"], b["choice_weak"], b["choice_strong"], net_per_choice(b))
    c = out["conditions"]
    mc = math.sqrt(ALPHA * (1 - ALPHA) / datasets)
    out["analytic_power"] = analytic
    out["simulated_power"] = c["weak_null"]["supported_rate"]
    out["rate_mc_se"] = mc
    out["checks"] = {
        "fisher_size_sharp_null": c["sharp_null"]["fisher_rejection_rate"],
        "fisher_size_weak_null": c["weak_null"]["fisher_rejection_rate"],
        "tost_size_upper_margin": c["margin_upper"]["supported_rate"],
        "tost_size_lower_margin": c["margin_lower"]["supported_rate"]}
    out["passed"] = (all(SIZE_RANGE[0] <= v <= SIZE_RANGE[1] for v in out["checks"].values())
                     and abs(out["simulated_power"] - analytic) <= POWER_TOLERANCE)
    return out


# ---------------------------------------------------------------- H5b study
H5B_DESIGN = {"choice": 0.30, "purchase_given_choice": 0.55, "event_accuracy": 0.90, "click_accuracy_null": 0.90,
              "click_accuracy_effect": 0.85, "n": 6000}


def _h5b_table(rng: np.random.Generator, n: int, click_accuracy: float) -> e5.RequestTable:
    """A purchase driver (the intent to buy) and two noisy indicators of it: the
    qualifying event and the click, correct with the given accuracies. Equal
    accuracies: equal predictive power (the null)."""
    d = H5B_DESIGN
    strong = rng.random(n) < 0.5
    purchase = rng.random(n) < d["choice"] * d["purchase_given_choice"]
    event = np.where(rng.random(n) < d["event_accuracy"], purchase, ~purchase)
    click = np.where(rng.random(n) < click_accuracy, purchase, ~purchase)
    return e5.RequestTable.from_arrays(strong, 0.5, np.full(n, "unknown"), event, click, purchase, purchase)


def _h5b_dataset(args: tuple) -> dict:
    condition, index, bootstrap = args
    d = H5B_DESIGN
    accuracy = d["click_accuracy_null"] if condition == "null" else d["click_accuracy_effect"]
    rng = stream(MASTER_SEED, "e5-h5b", condition, index)
    cells = e5.h5b(_h5b_table(rng, d["n"], accuracy), bootstrap=bootstrap, seed=MASTER_SEED,
                   analysis_id="%s-%d" % (condition, index))
    return {"supported": [c.p_support < 0.025 / len(cells) for c in cells], "cells": len(cells)}


def h5b_joint(click_accuracy: float) -> dict:
    d = H5B_DESIGN
    py = d["choice"] * d["purchase_given_choice"]
    joint = {}
    for e in (0, 1):
        for c in (0, 1):
            for y in (0, 1):
                pe = d["event_accuracy"] if e == y else 1 - d["event_accuracy"]
                pc = click_accuracy if c == y else 1 - click_accuracy
                joint[(e, c, y)] = (py if y else 1 - py) * pe * pc
    return joint


def h5b_study(datasets: int, bootstrap: int) -> dict:
    d = H5B_DESIGN
    out = {"design": d, "datasets": datasets, "bootstrap": bootstrap}
    for condition in ("null", "effect"):
        rows = _map(_h5b_dataset, [(condition, i, bootstrap) for i in range(datasets)], "h5b %s" % condition)
        out[condition] = {"cell_supported_rate": float(np.mean([s for r in rows for s in r["supported"]])),
                          "cells": rows[0]["cells"]}
    cells = out["effect"]["cells"]
    out["analytic_power"] = e5.h5b_power(d["n"] // cells, h5b_joint(d["click_accuracy_effect"]), cells)
    out["null_size_bound"] = 0.025 / cells
    return out


# ------------------------------------------------------------ contract check
def synthetic_events(seed: int, requests: int) -> list[dict]:
    """A small event log in the contract's form (the fixture)."""
    rng = stream(seed, "e5-fixture")
    start = datetime.datetime(2026, 11, 2, 9, 0, tzinfo=datetime.timezone.utc)
    events = []

    def event(request, event_type, minutes, product, **extra):
        stamp = (start + datetime.timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")
        events.append({"schema_version": 1, "protocol_version": e5.PROTOCOL_VERSION,
                       "event_id": e5.digest("event-%d" % len(events)), "request_id": request,
                       "event_time_utc": stamp, "product_id": product, "event_type": event_type, **extra})

    for r in range(requests):
        request = e5.digest("request-%d" % r)
        arm = "strong" if rng.random() < 0.5 else "weak"
        focal = "SKU-%03d" % rng.integers(1, 6)
        rivals = ["SKU-1%02d" % k for k in (rng.choice(10, size=3, replace=False) + (0 if arm == "strong" else 50))]
        base = 30 * r
        event(request, "assignment", base, focal, arm=arm, assignment_probability=0.5,
              candidate_set_id="SET-%s-%d" % (arm, r), candidate_product_ids=[focal] + rivals,
              traffic_type=["human", "agent", "unknown"][r % 3],
              exclusion="internal_test" if r == requests - 1 else "none")
        if rng.random() < 0.6:
            event(request, "click", base + 1, focal)
        if rng.random() < 0.4:
            event(request, "qualifying_event", base + 2, focal, kind="add_to_cart" if r % 2 else "confirmed_selection")
            if rng.random() < 0.7:
                order = e5.digest("order-%d" % r)
                event(request, "purchase", base + 10, focal, order_id=order, quantity=1)
                if rng.random() < 0.3:
                    event(request, "return", base + 60 * 24 * 3, focal, order_id=order, quantity=1)
    return events


def contract_check() -> dict:
    schema = json.loads(CONTRACT.read_text(encoding="utf-8"))
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    errors = [(i, e.message) for i, line in enumerate(lines) for e in validator.iter_errors(json.loads(line))]
    parsed = e5.parse_events(lines)
    table = e5.request_table(parsed)
    planted = {"properties": {"customer_email": {"type": "string"}, "shipping_address": {"type": "string"},
                              "full_name": {"type": "string"}, "note": {"type": "string"}}}
    record = {"request_id": "x", "comment": "write to someone@example.com"}
    return {"fixture_events": len(lines), "schema_errors": errors, "parsed": len(parsed),
            "fixture_requests": table.n, "fixture_excluded": {k: v for k, v in table.excluded.items() if v},
            "pii_schema": e5.pii_fields(schema), "pii_fixtures": [p for line in lines for p in
                                                                   e5.pii_fields(json.loads(line))],
            "planted_detected": sorted(e5.pii_fields(planted) + e5.pii_fields(record)),
            "passed": not errors and len(parsed) == len(lines) and not e5.pii_fields(schema)
            and not any(e5.pii_fields(json.loads(line)) for line in lines)
            and len(e5.pii_fields(planted)) == 3 and len(e5.pii_fields(record)) == 1}


# ----------------------------------------------------------------- plumbing
WORKERS = 1


def _map(function, arguments: list, label: str) -> list:
    start, out = time.time(), []
    if WORKERS <= 1:
        for count, a in enumerate(arguments, 1):
            out.append(function(a))
            _progress(label, count, len(arguments), start)
        return out
    with cf.ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        for count, result in enumerate(pool.map(function, arguments, chunksize=8), 1):
            out.append(result)
            _progress(label, count, len(arguments), start)
    return out


def _progress(label: str, done: int, total: int, start: float) -> None:
    if done % max(1, total // 4) == 0 or done == total:
        print("%s %d/%d (%.0fs)" % (label, done, total, time.time() - start), flush=True)


def run(counts: dict) -> dict:
    start = time.time()
    out = {"study": "SL-22 E5 analysis validation", "protocol_version": e5.PROTOCOL_VERSION,
           "master_seed": MASTER_SEED, "code_fingerprint": code_fingerprint(), "counts": counts,
           "baseline": BASELINE, "contract": contract_check(), "h5a": h5a_study(counts["datasets"], counts["draws"]),
           "h5b": h5b_study(counts["h5b_datasets"], counts["bootstrap"])}
    out["seconds"] = round(time.time() - start, 1)
    out["passed"] = {"AC-1": out["h5a"]["passed"], "AC-2": out["contract"]["passed"]}
    return out


def markdown(report: dict) -> str:
    a, b, c = report["h5a"], report["h5b"], report["contract"]
    lines = ["# SL-22 E5 analysis validation", "",
             "Generated by `python -m features.storelab.analysis.e5_validation` (protocol `%s`, master seed %d, code "
             "fingerprint `%s`, %.0f s). Synthetic baselines (stand-ins for the partner's, protocol §8): %s." % (
                 report["protocol_version"], report["master_seed"], report["code_fingerprint"][:12], report["seconds"],
                 ", ".join("%s %g" % kv for kv in report["baseline"].items())), "",
             "## H5a: Fisher randomisation test and TOST (AC-1)", "",
             "%d datasets per condition, n = %d requests each (the power formula's n for %.0f%% TOST power at τ = 0), "
             "%d randomisation draws per dataset; Monte Carlo SE of a rate near 0.05: %.3f." % (
                 a["datasets"], a["n"], 100 * POWER_TARGET, a["draws"], a["rate_mc_se"]), "",
             "| Condition | Fisher test of τ = 0 rejects | Supported (TOST) | Not supported | 95% interval covers τ | "
             "Mean τ̂ |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in a["conditions"].items():
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %+.4f |" % (name, row["fisher_rejection_rate"],
                                                                    row["supported_rate"], row["not_supported_rate"],
                                                                    row["ci95_coverage"], row["mean_estimate"]))
    lines += ["", "Sizes (required in [%.2f, %.2f]): %s. Power at the designed effect (weak null, τ = 0): simulated "
              "%.3f, analytic %.3f (required within ±%.0f points): %s." % (
                  SIZE_RANGE[0], SIZE_RANGE[1], ", ".join("%s %.3f" % kv for kv in a["checks"].items()),
                  a["simulated_power"], a["analytic_power"], 100 * POWER_TOLERANCE,
                  "passed" if a["passed"] else "FAILED"), "",
              "## H5b: event vs click as purchase predictor (reported)", "",
              "%d datasets of %d requests, %d bootstrap draws; %d cells (arms, traffic type unidentified). Rate of a "
              "supported cell at the per-cell level %.4f (0.025 / cells): under equal predictive power %.3f, at the "
              "designed advantage (click accuracy %.2f vs event %.2f) %.3f against the analytic %.3f." % (
                  b["datasets"], b["design"]["n"], b["bootstrap"], b["null"]["cells"], b["null_size_bound"],
                  b["null"]["cell_supported_rate"], b["design"]["click_accuracy_effect"],
                  b["design"]["event_accuracy"], b["effect"]["cell_supported_rate"], b["analytic_power"]), "",
              "## Data contract (AC-2)", "",
              "`contracts/storelab/e5_event.schema.json` against the fixture `tests/features/storelab/fixtures/"
              "e5_events.jsonl`: %d events, %d schema errors, %d parsed, %d requests in the table (excluded: %s). PII "
              "field scan: schema %s, fixtures %s; planted fields detected: %s. %s." % (
                  c["fixture_events"], len(c["schema_errors"]), c["parsed"], c["fixture_requests"],
                  c["fixture_excluded"] or "none", c["pii_schema"] or "none", c["pii_fixtures"] or "none",
                  ", ".join(c["planted_detected"]), "Passed" if c["passed"] else "FAILED"), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.e5_validation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--write-fixture", action="store_true", help="rewrite the contract fixture and exit")
    args = parser.parse_args(argv)
    if args.write_fixture:
        FIXTURE.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in synthetic_events(MASTER_SEED, 12)),
                           encoding="utf-8")
        return 0
    global WORKERS
    WORKERS = args.workers
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"  # one BLAS thread per worker process (spawned workers read it at import)
    report = run(QUICK if args.quick else FULL)
    out = args.out or REPORT
    out.mkdir(parents=True, exist_ok=True)
    (out / "e5_validation.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "e5_validation.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["passed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
