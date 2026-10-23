"""Version comparisons for one provider's version pair within one interface (SL-15, H1d).

For each task: Spearman and Kendall correlations of the two versions' scores v/σ, the
favourite f_v(x) = argmax over E(x) ∪ {0} of the unlabeled scores (outside 0), a *raw
change* when the favourites differ, and a *credible change* when they differ in at
least 90% of bootstrap replicates (sets resampled within the task for both versions,
A12 §3.2). Each credible change says whether the new favourite (version B) has the
higher ground-truth u; a tie in u, or a favourite that is the outside option (which
has no u), gives ``higher_u`` None and is excluded from the alignment test. The
records are the input of decide.py's ``first_credible_change`` rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import stats

from ..core.rng import stream
from . import clogit
from .data import ChoiceData
from .intervals import percentile

CREDIBLE = 0.90  # A12 §3.2


@dataclass
class TaskComparison:
    task: str
    spearman: float
    kendall: float
    favourite_a: str | None
    favourite_b: str | None
    credible_share: float  # share of replicates in which the favourites differ
    spearman_ci: tuple[float, float] = (float("nan"), float("nan"))
    kendall_ci: tuple[float, float] = (float("nan"), float("nan"))

    @property
    def raw_change(self) -> bool:
        return self.favourite_a != self.favourite_b

    @property
    def credible_change(self) -> bool:
        return self.credible_share >= CREDIBLE


@dataclass
class VersionComparison:
    tasks: list[TaskComparison] = field(default_factory=list)

    def counts(self) -> dict:
        return {"n": len(self.tasks), "counts": {"raw_changes": sum(t.raw_change for t in self.tasks),
                                                 "credible_changes": sum(t.credible_change for t in self.tasks)}}

    def credible_changes(self, pair: str, interface: str, utility: Mapping[str, Mapping[str, float]]) -> list[dict]:
        out = []
        for t in self.tasks:
            if not t.credible_change:
                continue
            higher = None
            if t.favourite_a is not None and t.favourite_b is not None:
                old, new = utility[t.task][t.favourite_a], utility[t.task][t.favourite_b]
                higher = None if old == new else bool(new > old)
            out.append({"pair": pair, "task": t.task, "interface": interface, "higher_u": higher})
        return out


def _task_scores(fit: clogit.ClogitFit) -> dict[str, dict[str, float]]:
    return fit.scores()


def favourite(scores: Mapping[str, float]) -> str | None:
    """argmax over E(x) ∪ {0}; None is the outside option (score 0)."""
    best, value = None, 0.0
    for product, score in sorted(scores.items()):
        if np.isfinite(score) and score > value:
            best, value = product, score
    return best


def compare(data_a: ChoiceData, data_b: ChoiceData, replicates: int, seed: int, analysis_id: str,
            fit_a: clogit.ClogitFit | None = None, fit_b: clogit.ClogitFit | None = None) -> VersionComparison:
    fit_a, fit_b = fit_a or clogit.fit(data_a), fit_b or clogit.fit(data_b)
    scores_a, scores_b = _task_scores(fit_a), _task_scores(fit_b)
    tasks = sorted(set(scores_a) & set(scores_b))
    rng_a = stream(seed, "bootstrap", analysis_id, "a")
    rng_b = stream(seed, "bootstrap", analysis_id, "b")
    boot = {t: {"differ": 0, "rho": [], "tau": []} for t in tasks}
    for _ in range(replicates):
        draws = []
        for data, rng in ((data_a, rng_a), (data_b, rng_b)):
            sets = list(dict.fromkeys(data.set_ids))
            rows_of: dict[str, list[int]] = {}
            for i, s in enumerate(data.set_ids):
                rows_of.setdefault(s, []).append(i)
            picked = rng.integers(0, len(sets), len(sets))  # sets belong to one task, so this resamples within tasks
            draws.append(_task_scores(clogit.fit(data.subset(np.concatenate([rows_of[sets[j]] for j in picked])))))
        for t in tasks:
            a, b = draws[0].get(t, {}), draws[1].get(t, {})
            boot[t]["differ"] += favourite(a) != favourite(b)
            common = sorted(p for p in set(a) & set(b) if np.isfinite(a[p]) and np.isfinite(b[p]))
            if len(common) >= 3:
                boot[t]["rho"].append(stats.spearmanr([a[p] for p in common], [b[p] for p in common])[0])
                boot[t]["tau"].append(stats.kendalltau([a[p] for p in common], [b[p] for p in common])[0])
    out = VersionComparison()
    for t in tasks:
        common = sorted(set(scores_a[t]) & set(scores_b[t]))
        x, y = [scores_a[t][p] for p in common], [scores_b[t][p] for p in common]
        entry = TaskComparison(t, float(stats.spearmanr(x, y)[0]), float(stats.kendalltau(x, y)[0]),
                               favourite(scores_a[t]), favourite(scores_b[t]),
                               boot[t]["differ"] / replicates if replicates else float("nan"))
        if boot[t]["rho"]:
            entry.spearman_ci, entry.kendall_ci = percentile(boot[t]["rho"]), percentile(boot[t]["tau"])
        out.tasks.append(entry)
    return out
