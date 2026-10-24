"""E4 runs on the stipulated state variance of A17 §4 (SL-21S; SL-21 reruns with E1 variance).

``python -m features.storelab.analysis.e4_run`` writes ``artifacts/results/sl21s/``:

- ``bound.csv.gz``: A(N) = ⌊e · N!⌋ against an enumeration of choice functions for N ≤ 7 and
  K = 2..4, and the corrected Cor design(b) bound (A1 §E1) beside the paper's;
- ``identification.csv.gz`` (H4a): one row per (N_x, f, replication, design);
- ``h4b_check.csv.gz``: the Prop learning(a) implementation check, 10,000 replications in each
  of 8 cells fixed before the run (owner decision 2026-10-24);
- ``learning/N_x=<n>/f=<f>/n0=<n0>.csv.gz`` (H4b–H4d): one row per (replication, h);
- ``recovery/N_x=<n>/f=<f>/n0=<n0>.csv.gz``: the exploratory recovery, one row per
  (replication, design).

**Markets** (A17 §4). One context, one model, σ = 0.3, outside score −1, K = 4. Replication r at
N_x draws from ``stream(MASTER_SEED, "sl21s-market", N_x, r)``: scores v_j ~ N(0, 1), shifts
η_j ~ N(0, 1), the recovery shifts η′_j, the 4 advertisers (uniform) and their values θ ~ U[0, 1].
Two equally likely states: v and v + τ η, τ = 0.5 √f. The same draws serve every f and n₀, and
the period states come from ``stream(..., "sl21s-states", N_x, r)``, so cells differ only in what
they vary.

**Designs** (2 N_x audited sets of size K, built from the mean scores v + τ η / 2 with
``stream(..., "sl21s-design", N_x, "f=<f>", r)``): leader-inclusive, competitor-varying (every pair of
the top 5 meets in a set filled from outside the top 5) and random matched to the
competitor-varying design on per-product inclusion counts.

**H4a** (owner decision 2026-10-24): choices in the deterministic limit, state by state; target
sets are the subsets of the top 5 products (by mean score) with 2 to K products (25 sets); the
share is averaged over the two states. Each replication also computes the share by the
independent possible-winner rule (the I-check) and whether each identified choice is the true
one; a secondary share covers every K-subset of E(x).

**H4b–H4d** use the competitor-varying design's cells: n₀ audit requests per cell and period at
σ = 0.3 (``stream(..., "sl21s-audit", N_x, "f=<f>", n0, r)``), 100 periods per replication, and every h
is the first h periods (common random numbers across h). The plug-in mechanism (n = 1000, g = 0.1,
quadratic cost λ = 0.5 and B = 0.5 n on every product) chooses among 16 seeded policies, each a
uniform mixture over 4 cells. H4c: the largest gain from a misreport over a grid of reports in
[0, 2] (step 0.005) for each advertiser, against 2 θ_j n ε₁ with ϱ = 0.05, N = N_x, |S| = 2 N_x.
H4d: cumulative loss Σ_{e=2}^{h} [max W_b − W_b(Q̂_e)] at truthful bids, Q̂_e from periods 1..e − 1.

**Recovery** (exploratory, owner decision 2026-10-24): at h = 100 the version changes after
period 50 (state 2 becomes v + τ η′); the platform re-estimates from post-change audit data. Each
audited set's estimated choice is its modal alternative (ties to the lowest index); recovery is
the number of post-change periods until the target-set choices it identifies are all correct for
the new version (the modal choice of the state-averaged probabilities) and cover at least the
share it identified at period 50; censored at 50.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..core.rng import stream
from . import e2_run as er
from . import e4
from .figures import REPO

MASTER_SEED = 2026102321
RESULTS = REPO / "artifacts" / "results" / "sl21s"
REPLICATIONS = 200
N_XS = (10, 30)
K = 4
SIGMA = 0.3
OUTSIDE_SCORE = -1.0
FACTORS = (0.0, 0.5, 1.0, 2.0)
HS = (5, 10, 20, 50, 100)
N0S = (2, 10, 100, 1000)
TOP = 5
STATE_PROBS = np.array([0.5, 0.5])
REQUESTS = 1000
ADVERTISERS = 4
CONTRIBUTION = 0.1
LAMBDA, OMEGA = 0.5, 0.5
MENU, MENU_CELLS = 16, 4
RHO = 0.05
REPORTS = np.round(np.arange(0.0, 2.0 + 1e-9, 0.005), 3)
SWITCH = 50
CHECK_REPLICATIONS = 10_000
CHECK_CELLS = [(h, n0, f) for h in (5, 100) for n0 in (2, 1000) for f in (0.0, 2.0)]  # N_x = 10, fixed before the run
DESIGNS = ("leader_inclusive", "competitor_varying", "random")


def tau(f: float) -> float:
    return 0.5 * math.sqrt(f)


def _f(f: float) -> str:
    """The stream key of a variance factor (keys are str or int)."""
    return "f=%g" % f


# --------------------------------------------------------------------- markets
def market(n_x: int, r: int) -> dict:
    rng = stream(MASTER_SEED, "sl21s-market", n_x, r)
    v = rng.standard_normal(n_x)
    eta = rng.standard_normal(n_x)
    eta_new = rng.standard_normal(n_x)
    ads = np.sort(rng.choice(n_x, ADVERTISERS, replace=False) + 1)
    theta = rng.uniform(size=ADVERTISERS)
    return {"v": v, "eta": eta, "eta_new": eta_new, "advertisers": ads, "theta": theta}


def scores(mk: dict, f: float, version: int = 1) -> np.ndarray:
    shift = mk["eta"] if version == 1 else mk["eta_new"]
    return np.array([mk["v"], mk["v"] + tau(f) * shift])


def designs(mk: dict, n_x: int, f: float, r: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """The three designs, the menu ι (policies × CV cells) and the mean scores."""
    rng = stream(MASTER_SEED, "sl21s-design", n_x, _f(f), r)
    means = STATE_PROBS @ scores(mk, f)
    count = 2 * n_x
    cv = e4.competitor_varying(rng, means, count, K, TOP)
    out = {"leader_inclusive": e4.leader_inclusive(rng, means, count, K), "competitor_varying": cv,
           "random": e4.matched_random(rng, cv)}
    iota = np.zeros((MENU, count))
    for p in range(MENU):
        iota[p, rng.choice(count, MENU_CELLS, replace=False)] = 1.0 / MENU_CELLS
    return out, iota, means


def targets(means: np.ndarray) -> list[tuple[int, ...]]:
    """The H4a target sets: subsets of the top products (by mean score) with 2..K members."""
    best = sorted(int(j) + 1 for j in np.argsort(-means)[:TOP])
    sets = [s for size in range(2, K + 1) for s in itertools.combinations(best, size)]
    return sets


def _shares(reach: np.ndarray, sets: list[tuple[int, ...]]) -> np.ndarray:
    """Identified alternative per target (−1 if none), for targets of mixed size."""
    out = np.full(len(sets), -1)
    for size in sorted({len(s) for s in sets}):
        idx = [i for i, s in enumerate(sets) if len(s) == size]
        out[idx] = e4.identified_shares(reach, np.array([sets[i] for i in idx]))
    return out


def _det_choice(state_scores: np.ndarray, target: tuple[int, ...]) -> int:
    values = [OUTSIDE_SCORE] + [state_scores[j - 1] for j in target]
    k = int(np.argmax(values))
    return 0 if k == 0 else target[k - 1]


# ------------------------------------------------------------------------ H4a
IDENTIFICATION_FIELDS = ["n_x", "f", "replication", "design", "share", "share_state1", "share_state2", "share_all_k",
                         "check_equal", "identified_correct", "inclusion_top5", "inclusion_leader", "sets"]


def evaluate_identification(task: tuple[int, float, int]) -> list[dict]:
    n_x, f, r = task
    mk = market(n_x, r)
    ds, _, means = designs(mk, n_x, f, r)
    sc = scores(mk, f)
    tg = targets(means)
    all_k = np.array(list(itertools.combinations(range(1, n_x + 1), K)))
    leader = int(np.argmax(means)) + 1
    top = [int(j) + 1 for j in np.argsort(-means)[:TOP]]
    rows = []
    for name in DESIGNS:
        sets = ds[name]
        winners = e4.deterministic(sc, sets, OUTSIDE_SCORE)  # (S, cells)
        shares, all_shares, equal, correct = [], [], True, True
        for s in range(len(sc)):
            reach = e4.closure(e4.comparison_graph([(tuple(x), int(w)) for x, w in zip(sets, winners[s])], n_x + 1))
            got = _shares(reach, tg)
            for t, target in enumerate(tg):
                possible = e4.possible_winners(reach, target)
                by_rule = possible[0] if len(possible) == 1 else -1
                equal &= by_rule == got[t] and (got[t] == -1) == (e4.identified(reach, target) is None)
                if got[t] != -1:
                    correct &= got[t] == _det_choice(sc[s], target)
            shares.append(float(np.mean(got != -1)))
            all_shares.append(float(np.mean(e4.identified_shares(reach, all_k) != -1)))
        freq = e4.inclusion(sets, n_x)
        rows.append({"n_x": n_x, "f": f, "replication": r, "design": name, "share": float(np.mean(shares)),
                     "share_state1": shares[0], "share_state2": shares[1], "share_all_k": float(np.mean(all_shares)),
                     "check_equal": int(equal), "identified_correct": int(correct),
                     "inclusion_top5": float(np.mean(freq[np.array(top) - 1])), "inclusion_leader": float(freq[leader - 1]),
                     "sets": json.dumps(sets.tolist())})
    return rows


# --------------------------------------------------------------- H4b check
CHECK_FIELDS = ["h", "n0", "f", "q_bar", "var_closed", "mean", "bias", "mc_se", "bias_in_se", "variance", "variance_ratio",
                "passed", "replications", "cell", "product"]


def evaluate_check(task: tuple[int, int, float]) -> list[dict]:
    h, n0, f = task
    mk = market(10, 0)
    ds, _, _ = designs(mk, 10, f, 0)
    cell = ds["competitor_varying"][:1]
    q = e4.logit(scores(mk, f), cell, SIGMA, OUTSIDE_SCORE)[:, 0, :]  # (S, k + 1)
    alt = 1 + int(np.argmax(STATE_PROBS @ q[:, 1:]))  # the cell's leader
    qj = q[:, alt]
    rng = stream(MASTER_SEED, "sl21s-check", h, n0, _f(f))
    states = rng.integers(0, 2, size=(CHECK_REPLICATIONS, h))
    y = rng.binomial(n0, qj[states]) / n0
    q_hat = y.mean(axis=1)
    q_bar = float(STATE_PROBS @ qj)
    closed = float(e4.variance_closed_form(qj, STATE_PROBS, h, n0))
    bias, se = float(q_hat.mean() - q_bar), float(q_hat.std(ddof=1) / math.sqrt(CHECK_REPLICATIONS))
    variance = float(q_hat.var(ddof=1))
    ratio = variance / closed
    return [{"h": h, "n0": n0, "f": f, "q_bar": q_bar, "var_closed": closed, "mean": float(q_hat.mean()), "bias": bias,
             "mc_se": se, "bias_in_se": abs(bias) / se, "variance": variance, "variance_ratio": ratio,
             "passed": int(abs(bias) <= 3 * se and abs(ratio - 1) <= 0.05), "replications": CHECK_REPLICATIONS,
             "cell": json.dumps(cell[0].tolist()), "product": int(cell[0][alt - 1])}]


# --------------------------------------------------------------- H4b-H4d
LEARNING_FIELDS = (["n_x", "f", "n0", "replication", "h", "sse", "closed_variance_sum", "estimates", "chosen", "best",
                    "regret", "cumulative_loss", "epsilon_1"]
                   + ["%s_%d" % (k, a) for a in range(ADVERTISERS) for k in ("theta", "gain", "bound")]
                   + ["violations", "max_gain"])


def _mechanism(mk: dict, sets: np.ndarray, iota: np.ndarray, n_x: int) -> e4.Mechanism:
    return e4.Mechanism(iota=iota, sets=sets, products=n_x, advertisers=mk["advertisers"], n=REQUESTS, g=CONTRIBUTION,
                        lam=LAMBDA, capacity=OMEGA * REQUESTS)


def evaluate_learning(task: tuple[int, float, int, int]) -> list[dict]:
    n_x, f, n0, r = task
    mk = market(n_x, r)
    ds, iota, _ = designs(mk, n_x, f, r)
    sets = ds["competitor_varying"]
    q = e4.logit(scores(mk, f), sets, SIGMA, OUTSIDE_SCORE)  # (S, cells, k + 1)
    states = stream(MASTER_SEED, "sl21s-states", n_x, r).integers(0, 2, size=max(HS))
    y = e4.audit(stream(MASTER_SEED, "sl21s-audit", n_x, _f(f), n0, r), q, states, n0)  # (periods, cells, k + 1)
    m = _mechanism(mk, sets, iota, n_x)
    theta = mk["theta"]
    truth = e4.true_values(m, q, STATE_PROBS)
    w_true = e4.welfare(truth, m.advertisers, theta)
    best = float(w_true.max())
    prefix = e4.prefix_welfare(m, y, n0, theta)  # (periods, policies): Ŵ after e periods
    loss = np.zeros(max(HS) + 1)
    for e in range(2, max(HS) + 1):
        loss[e] = best - w_true[int(np.argmax(prefix[e - 2]))]
    cumulative = np.cumsum(loss)
    q_bar = np.tensordot(STATE_PROBS, q, axes=1)  # (cells, k + 1)
    rows = []
    for h in HS:
        est = e4.estimates(m, y[:h], n0)
        chosen = e4.choose(est, m.advertisers, theta)
        gains = e4.misreport_gains(m, est, truth, theta, REPORTS)
        eps = e4.epsilon_1(n_x, len(sets), h, RHO)
        bounds = 2 * theta * REQUESTS * eps
        row = {"n_x": n_x, "f": f, "n0": n0, "replication": r, "h": h,
               "sse": float(((y[:h].mean(axis=0) - q_bar) ** 2).sum()),
               "closed_variance_sum": float(e4.variance_closed_form(q, STATE_PROBS, h, n0).sum()),
               "estimates": int(q_bar.size), "chosen": chosen, "best": best, "regret": best - float(w_true[chosen]),
               "cumulative_loss": float(cumulative[h]), "epsilon_1": eps,
               "violations": int(np.sum(gains > bounds)), "max_gain": float(gains.max())}
        for a in range(ADVERTISERS):
            row.update({"theta_%d" % a: float(theta[a]), "gain_%d" % a: float(gains[a]), "bound_%d" % a: float(bounds[a])})
        rows.append(row)
    return rows


# ------------------------------------------------------------------ recovery
RECOVERY_FIELDS = ["n_x", "f", "n0", "replication", "design", "share_pre", "correct_pre", "recovery", "censored",
                   "share_final", "correct_final"]


def _estimated_identification(y: np.ndarray, sets: np.ndarray, n_x: int, tg: list, truth: dict) -> tuple[float, bool]:
    """Identified share of the targets from modal choices of y (periods, cells, k + 1), and
    whether every identified choice is the true one."""
    modal = np.argmax(y.mean(axis=0), axis=1)
    winners = np.where(modal == 0, 0, sets[np.arange(len(sets)), np.maximum(modal - 1, 0)])
    reach = e4.closure(e4.comparison_graph([(tuple(s), int(w)) for s, w in zip(sets, winners)], n_x + 1))
    got = _shares(reach, tg)
    correct = all(g == truth[t] for g, t in zip(got, tg) if g != -1)
    return float(np.mean(got != -1)), bool(correct)


def _modal_truth(sc: np.ndarray, tg: list) -> dict:
    out = {}
    for t in tg:
        q = e4.logit(sc, np.array([t]), SIGMA, OUTSIDE_SCORE)[:, 0, :]
        k = int(np.argmax(STATE_PROBS @ q))
        out[t] = 0 if k == 0 else t[k - 1]
    return out


def evaluate_recovery(task: tuple[int, float, int, int]) -> list[dict]:
    n_x, f, n0, r = task
    mk = market(n_x, r)
    ds, _, means = designs(mk, n_x, f, r)
    tg = targets(means)
    old, new = scores(mk, f, 1), scores(mk, f, 2)
    truth_old, truth_new = _modal_truth(old, tg), _modal_truth(new, tg)
    states = stream(MASTER_SEED, "sl21s-recovery-states", n_x, r).integers(0, 2, size=2 * SWITCH)
    rng = stream(MASTER_SEED, "sl21s-recovery", n_x, _f(f), n0, r)
    rows = []
    for name in DESIGNS:
        sets = ds[name]
        pre = e4.audit(rng, e4.logit(old, sets, SIGMA, OUTSIDE_SCORE), states[:SWITCH], n0)
        post = e4.audit(rng, e4.logit(new, sets, SIGMA, OUTSIDE_SCORE), states[SWITCH:], n0)
        share_pre, correct_pre = _estimated_identification(pre, sets, n_x, tg, truth_old)
        recovery, share, correct = None, 0.0, False
        for e in range(1, SWITCH + 1):
            share, correct = _estimated_identification(post[:e], sets, n_x, tg, truth_new)
            if correct and share >= share_pre:
                recovery = e
                break
        rows.append({"n_x": n_x, "f": f, "n0": n0, "replication": r, "design": name, "share_pre": share_pre,
                     "correct_pre": int(correct_pre), "recovery": recovery if recovery is not None else SWITCH,
                     "censored": int(recovery is None), "share_final": share, "correct_final": int(correct)})
    return rows


# --------------------------------------------------------------------- bound
BOUND_FIELDS = ["n", "k", "choice_functions", "floor_e_factorial", "enumerated", "corrected_bound", "paper_bound",
                "audited_sets"]


def bound_rows() -> list[dict]:
    rows = []
    for n in range(2, 8):
        for k in range(2, min(K, n) + 1):
            rows.append({"n": n, "k": k, "choice_functions": e4.choice_functions(n),
                         "floor_e_factorial": math.floor(math.e * math.factorial(n)),
                         "enumerated": e4.count_choice_functions(n, k), "corrected_bound": e4.design_lower_bound(n, k),
                         "paper_bound": e4.paper_lower_bound(n, k), "audited_sets": None})
    for n in N_XS:
        rows.append({"n": n, "k": K, "choice_functions": e4.choice_functions(n), "floor_e_factorial": None,
                     "enumerated": None, "corrected_bound": e4.design_lower_bound(n, K),
                     "paper_bound": e4.paper_lower_bound(n, K), "audited_sets": 2 * n})
    return rows


# ------------------------------------------------------------------------- run
def cell_id(kind: str, n_x: int, f: float, n0: int) -> str:
    return "%s/N_x=%d/f=%g/n0=%d" % (kind, n_x, f, n0)


def run(group: str, match: str | None, replications: int, out: Path) -> dict:
    summary = {}
    if group in ("bound", "all"):
        er.write_rows(out / "bound.csv.gz", BOUND_FIELDS, bound_rows())
        summary["bound"] = "written"
    if group in ("check", "all"):
        rows = [r for rs in er._map(evaluate_check, CHECK_CELLS, "H4b check") for r in rs]
        er.write_rows(out / "h4b_check.csv.gz", CHECK_FIELDS, rows)
        summary["h4b_check"] = {"cells": len(rows), "failed": sum(1 - r["passed"] for r in rows)}
    if group in ("identification", "all"):
        tasks = [(n, f, r) for n in N_XS for f in FACTORS for r in range(replications)]
        rows = [r for rs in er._map(evaluate_identification, tasks, "H4a") for r in rs]
        er.write_rows(out / "identification.csv.gz", IDENTIFICATION_FIELDS, rows)
        summary["identification"] = {"rows": len(rows)}
    for kind, function, fields in (("learning", evaluate_learning, LEARNING_FIELDS),
                                   ("recovery", evaluate_recovery, RECOVERY_FIELDS)):
        if group not in (kind, "all"):
            continue
        cells = [(n, f, n0) for n in N_XS for f in FACTORS for n0 in N0S
                 if match is None or match in cell_id(kind, n, f, n0)]
        tasks = [(n, f, n0, r) for n, f, n0 in cells for r in range(replications)]
        results = er._map(function, tasks, kind)
        by_cell: dict[str, list] = {}
        for (n, f, n0, _), rows in zip(tasks, results):
            by_cell.setdefault(cell_id(kind, n, f, n0), []).extend(rows)
        for cid, rows in by_cell.items():
            er.write_rows(er.result_path(out, cid), fields, rows)
        summary[kind] = {"cells": len(by_cell)}
    return summary


def verify_rerun(groups: list[str], committed: Path, replications: int) -> int:
    """Rerun each ``group[:match]`` into a temporary directory and compare every file it writes
    byte for byte with the committed one; the result goes to ``rerun_check.json``."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k, spec in enumerate(groups):
            group, _, match = spec.partition(":")
            scratch = Path(tmp) / str(k)
            run(group, match or None, replications, scratch)
            for path in sorted(scratch.rglob("*.csv.gz")):
                name = path.relative_to(scratch).as_posix()
                results[name] = (committed / name).exists() and path.read_bytes() == (committed / name).read_bytes()
    (committed / "rerun_check.json").write_text(json.dumps({"files": results, "identical": all(results.values()),
                                                            "workers": er.WORKERS}, indent=1, sort_keys=True) + "\n",
                                                encoding="utf-8")
    print(json.dumps(results))
    return 0 if all(results.values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.e4_run")
    parser.add_argument("--group", default="all", choices=["bound", "check", "identification", "learning", "recovery",
                                                           "all"])
    parser.add_argument("--match", help="only learning/recovery cells whose id contains this text")
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--verify-rerun", nargs="+", metavar="GROUP[:MATCH]",
                        help="rerun these groups into a temporary directory and compare the files byte for byte")
    args = parser.parse_args(argv)
    er.WORKERS = args.workers
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    start = time.time()
    if args.verify_rerun:
        return verify_rerun(args.verify_rerun, args.out, args.replications)
    summary = run(args.group, args.match, args.replications, args.out)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("%.0f s" % (time.time() - start), flush=True)
    return 1 if summary.get("h4b_check", {}).get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
