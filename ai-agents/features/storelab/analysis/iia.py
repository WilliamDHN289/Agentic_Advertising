"""IIA tests for one E1 cell (SL-15, H1a).

**Triple odds test** (registered primary, SL-01 register; Monte Carlo calibration by
owner decision 2026-10-22). For products i, j and a product k that the cell's
conditional logit scores above both (v_k > v_i, v_j), compare the log odds of choosing
i over j in the candidate sets that contain i and j with k and without k, within one
task and one set size K (positions are randomised independently of composition, A15,
but position effects differ across K):

    D = log((n_i^with + ½) / (n_j^with + ½)) − log((n_i^without + ½) / (n_j^without + ½))

(Haldane-corrected). A triple is *informative* when i and j are each chosen in both
conditions; T = Σ D² over the informative triples. The p-value is a Monte Carlo test
under the fitted logit, which satisfies IIA: B data sets are drawn from the fitted
choice probabilities of the observed presentations, T is recomputed with the same
triples, and p = (1 + #{T_b ≥ T}) / (1 + B). This accounts for position and label
effects, for the smaller i-or-j totals of sets with k and for the dependence between
triples that share sets. The set-level ``permutation`` version (reported in the
validation, not registered) is anti-conservative for exactly those reasons.

**Hausman–McFadden** (reported; 1 df by owner decision 2026-10-22). Fit the cell's
logit on all alternatives and again with each task's top-scored product removed (its
choices dropped, it is removed from the other sets), and compare the label effect:
H = (δ_r − δ_f)² / (V_r − V_f), χ²₁. The position effects are not compared: the design
informs β for positions 5–8 only through the K = 8 sets, and the joint statistic's
size is far above nominal (SL-15 validation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse, stats

from ..core.rng import stream
from . import clogit
from .data import ChoiceData

HALDANE = 0.5


@dataclass
class TripleOddsResult:
    statistic: float
    p_value: float
    informative: int  # informative triples
    triples: int  # triples with both conditions present
    replicates: int
    method: str


@dataclass
class HausmanResult:
    statistic: float
    df: int
    p_value: float
    dropped: dict[str, str]  # task -> product removed


class TripleOdds:
    """The triples of a cell and T as sparse sums over (set, product) choice counts."""

    def __init__(self, data: ChoiceData, scores: np.ndarray):
        self.data = data
        sets = list(dict.fromkeys(data.set_ids))
        index = {s: n for n, s in enumerate(sets)}
        self.set_of = np.array([index[s] for s in data.set_ids])
        self.n_sets, self.width = len(sets), len(data.products) + 1  # last column: outside option
        members: dict[int, tuple[int, ...]] = {}
        for row, s in enumerate(data.set_ids):
            members.setdefault(index[s], tuple(sorted(int(p) for p in data.prod[row] if p >= 0)))
        groups: dict[tuple[str, int], list[int]] = {}
        for si, m in members.items():
            groups.setdefault((data.products[m[0]][0], len(m)), []).append(si)
        # one row per triple and condition: (sets in the condition, i, j)
        rows: list[tuple[list[int], int, int]] = []
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []  # per pair: (count rows, membership) for permutation
        for sids in groups.values():
            products = sorted({p for si in sids for p in members[si]})
            for a, i in enumerate(products):
                for j in products[a + 1:]:
                    hold = [si for si in sids if i in members[si] and j in members[si]]
                    if len(hold) < 2:
                        continue
                    above = [k for k in products if k not in (i, j) and scores[k] > max(scores[i], scores[j])]
                    pair_members = []
                    for k in above:
                        with_k = [si for si in hold if k in members[si]]
                        if 0 < len(with_k) < len(hold):
                            rows.append((with_k, i, j))
                            rows.append(([si for si in hold if k not in members[si]], i, j))
                            pair_members.append([k in members[si] for si in hold])
                    if pair_members:
                        self.pairs.append((np.array(hold), np.array([i, j]), np.array(pair_members, bool).T))
        self.triples = len(rows) // 2

        def selector(column: int) -> sparse.csr_matrix:
            r, c = [], []
            for n, (sids, i, j) in enumerate(rows):
                product = (i, j)[column]
                r += [n] * len(sids)
                c += [si * self.width + product for si in sids]
            return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), self.n_sets * self.width))

        self.select_i, self.select_j = selector(0), selector(1)

    def counts(self, chosen_slot: np.ndarray) -> np.ndarray:
        rows = np.arange(len(chosen_slot))
        product = np.full(len(chosen_slot), self.width - 1)
        picked = chosen_slot >= 0
        product[picked] = self.data.prod[rows[picked], chosen_slot[picked]]
        return np.bincount(self.set_of * self.width + product, minlength=self.n_sets * self.width).astype(float)

    def statistic(self, counts: np.ndarray) -> tuple[float, int]:
        n_i, n_j = self.select_i @ counts, self.select_j @ counts
        with_i, with_j, without_i, without_j = n_i[0::2], n_j[0::2], n_i[1::2], n_j[1::2]
        informative = (with_i > 0) & (with_j > 0) & (without_i > 0) & (without_j > 0)
        d = (np.log((with_i + HALDANE) / (with_j + HALDANE)) - np.log((without_i + HALDANE) / (without_j + HALDANE)))
        return float((d[informative] ** 2).sum()), int(informative.sum())


def draw_choices(model: clogit.Model, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One choice per trial from the fitted probabilities: the slot, or -1 (outside)."""
    p, p0 = model.probabilities(theta)
    cumulative = np.cumsum(np.concatenate([p0[:, None], p], axis=1), axis=1)
    return (rng.random(len(p0))[:, None] > cumulative).sum(axis=1) - 1


def triple_odds_test(data: ChoiceData, fit: clogit.ClogitFit | None = None, replicates: int = 999, seed: int = 0,
                     analysis_id: str = "triple_odds", method: str = "monte_carlo") -> TripleOddsResult:
    """The registered Monte Carlo test, or (``permutation``) the set-level permutation
    of each pair's (i, j) outcomes, kept to document its size."""
    fit = fit or clogit.fit(data)
    scores = np.nan_to_num(fit.theta[:len(data.products)], nan=-np.inf)
    triples = TripleOdds(data, scores)
    counts = triples.counts(data.chosen)
    observed, informative = triples.statistic(counts)
    rng = stream(seed, method, analysis_id)
    exceed = 0
    if method == "monte_carlo":
        model, theta = clogit.Model(data), np.nan_to_num(fit.theta)
        for _ in range(replicates):
            exceed += triples.statistic(triples.counts(draw_choices(model, theta, rng)))[0] >= observed - 1e-12
    elif method == "permutation":
        table = counts.reshape(triples.n_sets, triples.width)
        for _ in range(replicates):
            total = 0.0
            for hold, pair, membership in triples.pairs:
                c = table[hold][:, pair][rng.permutation(len(hold))]
                with_k = membership.T.astype(float) @ c
                without = c.sum(axis=0)[None, :] - with_k
                ok = (with_k > 0).all(axis=1) & (without > 0).all(axis=1)
                d = (np.log((with_k[:, 0] + HALDANE) / (with_k[:, 1] + HALDANE))
                     - np.log((without[:, 0] + HALDANE) / (without[:, 1] + HALDANE)))
                total += float((d[ok] ** 2).sum())
            exceed += total >= observed - 1e-12
    else:
        raise ValueError("method: monte_carlo or permutation")
    p = (1 + exceed) / (1 + replicates) if informative else 1.0
    return TripleOddsResult(observed, p, informative, triples.triples, replicates, method)


def without_products(data: ChoiceData, products: set[int]) -> ChoiceData:
    """Drop trials that chose one of ``products`` and remove them from the other sets."""
    chosen = data.chosen_product()
    keep = ~np.isin(chosen, list(products))
    sub = data.subset(np.flatnonzero(keep))
    prod = sub.prod.copy()
    prod[np.isin(prod, list(products))] = -1
    return ChoiceData(sub.products, prod, np.where(prod >= 0, sub.pos, 0), np.where(prod >= 0, sub.label, 0.0),
                      sub.chosen, sub.set_ids, sub.reps)


def hausman_mcfadden(data: ChoiceData) -> HausmanResult:
    """Compares the MLEs (the efficient estimator the Hausman construction needs), not
    the registered Firth fits."""
    full = clogit.fit(data, penalty="none", detect=False)
    scores = full.theta[:len(data.products)]
    drop: dict[str, int] = {}
    for index, (task, _) in enumerate(data.products):
        if task not in drop or scores[index] > scores[drop[task]]:
            drop[task] = index
    reduced = without_products(data, set(drop.values()))
    restricted = clogit.fit(reduced, penalty="none", detect=False)
    variance = (restricted.std_error("delta") ** 2 - full.std_error("delta") ** 2)
    difference = restricted.value("delta") - full.value("delta")
    if variance <= 0:
        return HausmanResult(0.0, 1, 1.0, {task: data.products[i][1] for task, i in drop.items()})
    statistic = difference ** 2 / variance
    return HausmanResult(float(statistic), 1, float(stats.chi2.sf(statistic, 1)),
                         {task: data.products[i][1] for task, i in drop.items()})
