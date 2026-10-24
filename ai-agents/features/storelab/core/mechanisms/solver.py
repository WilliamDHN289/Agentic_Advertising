"""The pinned solvers (SL-16): fixed solver per problem class, fixed options, strict status.

``solver_config.json`` pins, for each class of mechanism program, one solver through
cvxpy: the LPs (M3, set-blind M3, overflow-cost M4, the profit rule without
quadratic costs) go to HiGHS's dual simplex (scipy's HiGHS, through cvxpy's SCIPY
interface), whose basic solutions are exact vertices, so X_j(b) is exact between
breakpoints; the QPs (quadratic-cost M4 and profit rule) go to Clarabel, an
interior-point solver, and are then polished on their active set
(allocation.polish: the KKT system solved exactly and verified). If the
interior-point solve or its polish fails, the QP is solved by the same verified
active-set method started at a feasible vertex from the LP solver, and failing that
by OSQP (``qp_fallback``) followed by the polish; only if every path fails is the
solve a failure (``unpolished``) and the market excluded. No solve is warm-started, so a result
never depends on earlier solves. The file also pins the exact
versions of cvxpy, scipy and the solver packages and records the platform check
against ai-agents DEPENDENCIES.md; ``check_versions`` refuses other installed
versions.

``solve`` accepts only the statuses in ``accepted_status`` (``optimal``); anything
else, including ``optimal_inaccurate``, raises ``SolverFailure``. Callers exclude the
market and count it (``Exclusions``); a failed solve is never used.
"""
from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cvxpy as cp

CONFIG = Path(__file__).with_name("solver_config.json")


class SolverFailure(RuntimeError):
    def __init__(self, status: str, where: str = ""):
        super().__init__("solver status %s%s" % (status, " (%s)" % where if where else ""))
        self.status, self.where = status, where


@lru_cache(maxsize=1)
def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def check_versions() -> dict[str, str]:
    """The installed versions; raises if any differs from the pin."""
    installed = {name: importlib.metadata.version(name) for name in config()["packages"]}
    wrong = {k: (v, config()["packages"][k]) for k, v in installed.items() if v != config()["packages"][k]}
    if wrong:
        raise RuntimeError("solver packages differ from solver_config.json (installed, pinned): %s" % wrong)
    return installed


def solve(problem: cp.Problem, kind: str, where: str = "") -> float:
    """Solve an ``lp`` or ``qp`` with its pinned solver; the optimal value, or SolverFailure."""
    settings = config()
    pinned = settings["programs"][kind]
    try:
        value = problem.solve(solver=pinned["solver"], **pinned["options"])
    except cp.error.SolverError as error:
        raise SolverFailure("solver_error: %s" % error, where) from error
    if problem.status not in settings["accepted_status"]:
        raise SolverFailure(str(problem.status), where)
    return float(value)


@dataclass
class Exclusions:
    """Markets excluded because a solve failed: counted, with the status and place."""
    entries: list[dict] = field(default_factory=list)

    def add(self, market_id: str, mechanism: str, failure: SolverFailure) -> None:
        self.entries.append({"market_id": market_id, "mechanism": mechanism, "status": failure.status,
                             "where": failure.where})

    @property
    def count(self) -> int:
        return len({e["market_id"] for e in self.entries})
