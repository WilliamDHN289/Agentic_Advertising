"""Run the SL-19S E2 dry run and write its result files.

    python -m features.storelab.analysis.e2_run [--group core|sensitivity|part3|gating|all]
        [--match TEXT] [--markets N] [--out DIR] [--workers N]

**Markets.** Every configuration of ``e2_markets.configurations`` gets ``MARKETS`` = 200
markets. Each market runs all six mechanisms of SL-16 with truthful bids: M3, M4 and the
profit rule bid θ, and M1 and M2 use the level's click-bidding rule. Payments are split by
traffic type (A1 §E5): by eq. payment for M3, set-blind M3 and M4, by the profit rule's own
eq. payment, and by charged clicks for M1 and M2. A solver failure in any mechanism excludes
the market from all of them and is recorded.

**Gating.** Prop gating's I-part runs 200 markets per κ in {0.05, 0.2}: designs (a) and (b)
under M3 (C ≡ 0). Conj gating runs the same 200 markets with humans added, κ = 0.05, over α.

**Output.** Under ``artifacts/results/sl19s/``, one gzip CSV per configuration, with one row
per (market, mechanism) and the market's input hash. Floats are written with repr, and the
gzip header has no timestamp, so a seeded rerun reproduces the files byte for byte whatever
the number of workers.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import io
import math
import multiprocessing
import os
import time
from pathlib import Path

from ..core.mechanisms import gsp
from ..core.mechanisms.evaluate import MECHANISMS, MarketRunner
from ..core.mechanisms.solver import SolverFailure
from . import e2_markets as em
from .figures import REPO

RESULTS = REPO / "artifacts" / "results" / "sl19s"
MARKET_FIELDS = ["config_id", "group", "level", "alpha", "regime", "variation", "index", "market_hash", "excluded",
                 "status", "feasible", "screen", "favourites", "excess", "theta_bar", "m_ref", "m_organic",
                 "threshold", "floor_binds"]
OUTCOME_FIELDS = ["mechanism", "revenue", "revenue_human", "revenue_agent", "x_ads", "x_ads_agent", "contribution",
                  "cost", "operating_value", "match", "slack", "profit", "payment_per_choice", "welfare",
                  "converged", "iterations", "qp_fallbacks", "payment_error_bound"]


def _gsp_split(runner: MarketRunner, mechanism: str, outcome) -> tuple[float, float]:
    result = gsp.auction(runner.market, mechanism, outcome.bids)
    market = runner.market
    human = agent = 0.0
    for context in market.contexts:
        chi = math.fsum(context.model_shares.get(m.key, 0.0) * m.page_request_prob for m in market.models)
        for k, j in enumerate(result.sponsored[context.context_id], start=1):
            scale = market.requests * context.probability * result.prices[context.context_id][j]
            human += scale * (1 - context.agent_share) * market.position_weights[k - 1] * \
                context.offers[j].sponsored.request
            if mechanism == "M1":
                agent += scale * context.agent_share * chi
    return human, agent


def _fallbacks(runner: MarketRunner, mechanism: str) -> int:
    """QP solves of this mechanism that needed the vertex or ADMM fallback (SL-16 solver)."""
    program = runner.programs.get(mechanism if mechanism in ("M3_set_blind", "M3", "M4", "profit_rule") else "")
    if program is None:
        return 0
    count, program.fallbacks[:] = len(program.fallbacks), []
    return count


def evaluate_market(task: tuple[em.Configuration, int]) -> list[dict]:
    config, index = task
    market, info, draw = em.market(config, index)
    row = {"config_id": config.config_id, "group": config.group, "level": config.level, "alpha": config.alpha,
           "regime": config.regime, "variation": config.variation, "index": index,
           "market_hash": market.market_hash(), "feasible": int(info["feasible"]), "screen": int(info["screen"]),
           "favourites": len(info["favourites"]), "excess": info["excess"], "theta_bar": config.params.theta_bar(),
           **{k: info[k] for k in ("m_ref", "m_organic", "threshold", "floor_binds")}}
    runner = MarketRunner(market)
    rows = []
    try:
        for mechanism in MECHANISMS:
            o = runner.evaluate(mechanism, draw.theta, bidding_rule=config.params.bidding,
                                by_type=mechanism in ("M3_set_blind", "M3", "M4"))
            if mechanism in ("M1", "M2"):
                human, agent = _gsp_split(runner, mechanism, o)
            else:
                human, agent = math.fsum(o.payments_human.values()), math.fsum(o.payments_agent.values())
            ads = market.advertisers
            x_ads = math.fsum(o.X[j] for j in ads)
            rows.append({**row, "excluded": 0, "status": "optimal", "mechanism": mechanism, "revenue": o.revenue,
                         "revenue_human": human, "revenue_agent": agent, "x_ads": x_ads,
                         "x_ads_agent": math.fsum(o.X_agent[j] for j in ads), "contribution": o.contribution,
                         "cost": math.fsum(o.cost.values()), "operating_value": o.operating_value, "match": o.match,
                         "slack": o.match - info["threshold"], "profit": o.profit,
                         "payment_per_choice": o.revenue / x_ads if x_ads > 0 else float("nan"),
                         "welfare": math.fsum(draw.theta[j] * o.X[j] for j in ads) + o.contribution,
                         "converged": int(o.converged), "iterations": o.iterations,
                         "qp_fallbacks": _fallbacks(runner, mechanism),
                         "payment_error_bound": o.payment_error_bound})
    except SolverFailure as failure:
        empty = {f: float("nan") for f in OUTCOME_FIELDS if f not in ("mechanism", "converged", "iterations")}
        return [{**row, **empty, "excluded": 1, "status": "%s (%s)" % (failure.status, failure.where),
                 "mechanism": mechanism, "converged": 0, "iterations": 0, "qp_fallbacks": 0,
                 "payment_error_bound": 0.0} for mechanism in MECHANISMS]
    return rows


GATING_FIELDS = ["kind", "kappa", "alpha", "index", "market_hash_a", "market_hash_b", "excluded", "status", "f", "r",
                 "xi", "theta_f", "theta_r", "u_f", "u_r", "sigma", "profit_a", "profit_b", "gain", "expected_gain",
                 "relative_error", "match_a", "match_b", "match_loss", "expected_match_loss", "m_ref_a", "m_ref_b",
                 "revenue_a"]


def evaluate_gating(task: tuple[str, float, float, int]) -> dict:
    kind, kappa, alpha, index = task
    a, info = em.gating_market(index, kappa, "aware", alpha)
    b, _ = em.gating_market(index, kappa, "gated", alpha)
    values = {j: em.gating_draw(index)[0].theta[j] for j in a.advertisers}
    row = {"kind": kind, "kappa": kappa, "alpha": alpha, "index": index, "market_hash_a": a.market_hash(),
           "market_hash_b": b.market_hash(), **{k: info[k] for k in ("f", "r", "xi", "theta_f", "theta_r", "u_f", "u_r",
                                                                       "sigma")}}
    try:
        oa, ob = MarketRunner(a).evaluate("M3", values), MarketRunner(b).evaluate("M3", values)
    except SolverFailure as failure:
        return {**row, "excluded": 1, "status": "%s (%s)" % (failure.status, failure.where)}
    n = a.requests
    expected = n * info["xi"] * min(info["theta_f"], info["theta_r"])
    gain = ob.profit - oa.profit
    loss = n * info["xi"] * (info["u_f"] - info["u_r"]) * (info["theta_f"] < info["theta_r"])
    return {**row, "excluded": 0, "status": "optimal", "profit_a": oa.profit, "profit_b": ob.profit, "gain": gain,
            "expected_gain": expected, "relative_error": abs(gain - expected) / max(expected, 1e-300),
            "match_a": oa.match, "match_b": ob.match, "match_loss": oa.match - ob.match, "expected_match_loss": loss,
            "m_ref_a": oa.m_ref, "m_ref_b": ob.m_ref, "revenue_a": oa.revenue}


def gating_tasks(markets: int) -> dict[str, list]:
    tasks = {"gating/i_part/kappa=%g" % k: [("i_part", k, 1.0, i) for i in range(markets)] for k in em.GATING_KAPPAS}
    for alpha in em.ALPHAS:
        tasks["gating/conj/alpha=%g" % alpha] = [("conj", em.CONJ_KAPPA, alpha, i) for i in range(markets)]
    return tasks


# --------------------------------------------------------------------- files
def _cell(value) -> str:
    if isinstance(value, float):
        return repr(value)
    return "" if value is None else str(value)


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([_cell(row.get(f)) for f in fields])
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    path.write_bytes(raw.getvalue())


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def result_path(out: Path, config_id: str) -> Path:
    return out / (config_id.replace("=", "-").replace(",", "_") + ".csv.gz")


# ------------------------------------------------------------------ plumbing
WORKERS = 1


def _map(function, tasks: list, label: str) -> list:
    """``function`` on every task, results in task order. Tasks run on WORKERS processes as
    they become free; a task slower than SLOW seconds is logged by name."""
    start, out = time.time(), [None] * len(tasks)
    if WORKERS <= 1:
        for count, task in enumerate(tasks, 1):
            out[count - 1] = function(task)
            _progress(label, count, len(tasks), start)
        return out
    with cf.ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_timed, function, task): k for k, task in enumerate(tasks)}
        for count, future in enumerate(cf.as_completed(futures), 1):
            result, seconds = future.result()
            out[futures[future]] = result
            if seconds > SLOW:
                print("slow task %s: %.0f s" % (_name(tasks[futures[future]]), seconds), flush=True)
            _progress(label, count, len(tasks), start)
    return out


SLOW = 60.0


def _timed(function, task):
    start = time.perf_counter()
    return function(task), time.perf_counter() - start


def _name(task) -> str:
    return "%s#%s" % (task[0].config_id, task[1]) if isinstance(task[0], em.Configuration) else str(task)


def _progress(label: str, done: int, total: int, start: float) -> None:
    if done % max(1, total // 20) == 0 or done == total:
        print("%s %d/%d (%.0fs)" % (label, done, total, time.time() - start), flush=True)


TOP_UP_LIMIT = 50  # extra markets tried per configuration (and blocks of Conj gating markets)


def top_up(out: Path, markets: int, match: str | None = None) -> dict:
    """Bring every configuration to ``markets`` valid markets. A market configuration with
    exclusions continues at index ``markets``, ``markets`` + 1, ... until enough are valid.
    Conj gating compares the aware and gated designs at every α, so it continues in blocks
    of indices until ``markets`` markets are feasible under the paper's guardrail at every
    α. Rows are appended in index order, so a full run (which calls this) and a top-up of
    existing files give the same bytes."""
    added = {}
    for config in em.configurations():
        if match is not None and match not in config.config_id:
            continue
        path = result_path(out, config.config_id)
        if not path.exists():
            continue
        rows = read_rows(path)
        valid = len({r["index"] for r in rows if r["excluded"] == "0"})
        index = 1 + max(int(r["index"]) for r in rows)
        new = []
        while valid < markets and index < markets + TOP_UP_LIMIT:
            fresh = evaluate_market((config, index))
            new += fresh
            valid += not fresh[0]["excluded"]
            index += 1
        if new:
            write_rows(path, MARKET_FIELDS + OUTCOME_FIELDS, rows + new)
            added[config.config_id] = len({r["index"] for r in new})
    conj = {a: result_path(out, "gating/conj/alpha=%g" % a) for a in em.ALPHAS}
    if all(path.exists() for path in conj.values()) and (match is None or "conj" in match):
        rows = {a: read_rows(path) for a, path in conj.items()}
        feasible = set.intersection(*[{r["index"] for r in rows[a] if r["excluded"] == "0"} for a in em.ALPHAS])
        start = 1 + max(int(r["index"]) for r in rows[em.ALPHAS[0]])
        fresh = {a: [] for a in em.ALPHAS}
        block = 0
        while len(feasible) < markets and block < TOP_UP_LIMIT:
            indices = range(start + block * markets, start + (block + 1) * markets)
            batch = {a: _map(evaluate_gating, [("conj", em.CONJ_KAPPA, a, i) for i in indices], "conj top-up")
                     for a in em.ALPHAS}
            for a in em.ALPHAS:
                fresh[a] += batch[a]
            feasible |= set.intersection(*[{str(r["index"]) for r in batch[a] if not r["excluded"]}
                                           for a in em.ALPHAS])
            block += 1
        if fresh[em.ALPHAS[0]]:
            for a in em.ALPHAS:
                write_rows(conj[a], GATING_FIELDS, rows[a] + fresh[a])
            added["gating/conj"] = len(fresh[em.ALPHAS[0]])
    return added


def run(group: str, match: str | None, markets: int, out: Path) -> dict:
    summary = {}
    if group in ("core", "sensitivity", "part3", "all"):
        configs = [c for c in em.configurations() if (group == "all" or c.group == group)
                   and (match is None or match in c.config_id)]
        tasks = [(c, i) for c in configs for i in range(markets)]
        results = _map(evaluate_market, tasks, "markets")
        by_config: dict[str, list] = {}
        for (config, _), rows in zip(tasks, results):
            by_config.setdefault(config.config_id, []).extend(rows)
        for config_id, rows in by_config.items():
            write_rows(result_path(out, config_id), MARKET_FIELDS + OUTCOME_FIELDS, rows)
            summary[config_id] = {"markets": len({r["index"] for r in rows}),
                                  "excluded": len({r["index"] for r in rows if r["excluded"]})}
    if group in ("gating", "all"):
        for name, tasks in gating_tasks(markets).items():
            if match is not None and match not in name:
                continue
            rows = _map(evaluate_gating, tasks, name)
            write_rows(result_path(out, name), GATING_FIELDS, rows)
            summary[name] = {"markets": len(rows), "excluded": sum(r["excluded"] for r in rows)}
    summary["top_up"] = top_up(out, markets, match)
    return summary


def verify_rerun(config_ids: list[str], committed: Path) -> int:
    """A seeded rerun of whole configurations, compared byte for byte with the committed
    files; every file the rerun writes is compared (``gating/conj`` writes one per α), and
    the result goes to ``rerun_check.json`` next to them."""
    import json
    import tempfile
    configs = {c.config_id: c for c in em.configurations()}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for config_id in config_ids:
            group = configs[config_id].group if config_id in configs else "gating"
            scratch = Path(tmp) / str(len(results))
            run(group, config_id, em.MARKETS, scratch)
            for path in sorted(scratch.rglob("*.csv.gz")):
                name = path.relative_to(scratch).as_posix()
                results[name] = (committed / name).exists() and path.read_bytes() == (committed / name).read_bytes()
    (committed / "rerun_check.json").write_text(json.dumps({"configs": results, "identical": all(results.values()),
                                                            "workers": WORKERS}, indent=1, sort_keys=True) + "\n",
                                                encoding="utf-8")
    print(json.dumps(results))
    return 0 if all(results.values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.e2_run")
    parser.add_argument("--group", default="all", choices=["core", "sensitivity", "part3", "gating", "all"])
    parser.add_argument("--match", help="only configurations whose id contains this text")
    parser.add_argument("--markets", type=int, default=em.MARKETS)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--top-up", action="store_true",
                        help="only bring the existing result files to the required number of valid markets")
    parser.add_argument("--verify-rerun", nargs="+", metavar="CONFIG_ID",
                        help="rerun these configurations into a temporary directory and compare the files byte for byte")
    args = parser.parse_args(argv)
    global WORKERS
    WORKERS = args.workers
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    start = time.time()
    if args.verify_rerun:
        return verify_rerun(args.verify_rerun, args.out)
    if args.top_up:
        print(top_up(args.out, args.markets, args.match), flush=True)
        return 0
    summary = run(args.group, args.match, args.markets, args.out)
    print("top-up %s" % summary.pop("top_up"), flush=True)
    excluded = sum(s["excluded"] for s in summary.values())
    print("configurations %d, markets %d, excluded %d, %.0f s" % (len(summary), sum(s["markets"] for s in
                                                                               summary.values()), excluded,
                                                                  time.time() - start), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
