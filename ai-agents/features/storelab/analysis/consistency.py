"""Choice consistency for one E1 cell (SL-15, H1b).

**Modal share** (part i): for each candidate set presented r times identically (A15),
m_s = max_j n_sj / r_valid over the displayed products and the outside option; a set
is eligible with at least ``MIN_VALID`` valid repeats (A12 §3.1). The cell statistic
is the mean over eligible sets, with a percentile bootstrap over sets.

**Score gaps** (part ii): for each set, the gap between the highest and second-highest
label-adjusted scores v_j/σ + (δ/σ) ℓ_j in S ∪ {0} (outside 0) from the cell's
conditional logit; the cell statistic is the median over sets, with a cluster
bootstrap over sets that refits the logit (penalised under separation, so the gap is
then a lower bound).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.rng import stream
from . import clogit
from .data import ChoiceData
from .intervals import bootstrap_p_values, percentile

MIN_VALID = 8
MODAL_THRESHOLD, GAP_THRESHOLD = 0.80, float(np.log(4.0))  # A12 §3.1


@dataclass
class ModalShares:
    shares: dict[str, float]  # eligible set -> m_s
    ineligible: list[str]  # sets with fewer than MIN_VALID valid repeats

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.shares.values()))) if self.shares else float("nan")


def modal_shares(data: ChoiceData, min_valid: int = MIN_VALID) -> ModalShares:
    chosen = data.chosen_product()
    counts: dict[str, dict[int, int]] = {}
    for set_id, product in zip(data.set_ids, chosen):
        bucket = counts.setdefault(set_id, {})
        bucket[int(product)] = bucket.get(int(product), 0) + 1
    shares, ineligible = {}, []
    for set_id, bucket in counts.items():
        valid = sum(bucket.values())
        if valid < min_valid:
            ineligible.append(set_id)
        else:
            shares[set_id] = max(bucket.values()) / valid
    return ModalShares(shares, ineligible)


def modal_share_test(data: ChoiceData, replicates: int, seed: int, analysis_id: str,
                     threshold: float = MODAL_THRESHOLD) -> dict:
    """Mean modal share, its percentile interval over resampled sets and the
    (p_support, p_against) against the threshold for decide.py."""
    result = modal_shares(data)
    values = np.array(list(result.shares.values()))
    rng = stream(seed, "bootstrap", analysis_id)
    boot = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(replicates)])
    p_support, p_against = bootstrap_p_values(boot, threshold, "above")
    return {"estimate": result.mean, "ci": percentile(boot), "n": len(values), "ineligible": len(result.ineligible),
            "p_support": p_support, "p_against": p_against}


def score_gaps(data: ChoiceData, fit: clogit.ClogitFit) -> dict[str, float]:
    """Per set: the top-two gap of label-adjusted scores in S ∪ {0}."""
    out: dict[str, float] = {}
    delta = fit.label_effect
    for row, set_id in enumerate(data.set_ids):
        if set_id in out:
            continue
        slots = data.prod[row] >= 0
        scores = np.concatenate([[0.0], fit.theta[data.prod[row][slots]] + delta * data.label[row][slots]])
        top = np.sort(scores)[::-1]
        out[set_id] = float(top[0] - top[1])
    return out


def gap_test(data: ChoiceData, replicates: int, seed: int, analysis_id: str, threshold: float = GAP_THRESHOLD,
             fit: clogit.ClogitFit | None = None) -> dict:
    fit = fit or clogit.fit(data)
    estimate = float(np.median(list(score_gaps(data, fit).values())))

    def statistic(sample: ChoiceData) -> float:
        refit = clogit.fit(sample)
        gaps = [g for g in score_gaps(sample, refit).values() if np.isfinite(g)]
        return float(np.median(gaps)) if gaps else float("nan")

    boot = clogit.cluster_bootstrap(data, statistic, replicates, seed, analysis_id).ravel()
    p_support, p_against = bootstrap_p_values(boot[np.isfinite(boot)], threshold, "above")
    return {"estimate": estimate, "ci": percentile(boot), "penalised": fit.firth, "p_support": p_support,
            "p_against": p_against}
