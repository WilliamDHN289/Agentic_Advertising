"""Interval helpers shared by the E1 statistics (SL-15): Wilson and Clopper–Pearson
intervals for proportions, percentile bootstrap intervals, and the one-sided
p-values that ``decide.py`` reads (the duals of the registered interval bounds).
Holm's adjustment is ``decide.holm``."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson(k: int, n: int, level: float = 0.95, one_sided: bool = False) -> tuple[float, float]:
    """Exact interval; ``one_sided`` gives each bound at ``level`` on its own."""
    a = 1 - level if one_sided else (1 - level) / 2
    lower = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lower, upper


def percentile(replicates: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    finite = np.asarray(replicates, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, a)), float(np.quantile(finite, 1 - a))


def bootstrap_p_values(replicates: np.ndarray, threshold: float, side: str) -> tuple[float, float]:
    """(p_support, p_against) of a percentile-bootstrap bound against ``threshold``:
    the share of replicates on the far side of it. For ``side`` above, the percentile
    lower bound at level 1 − a is at or above the threshold exactly when the share of
    replicates below it is at most a."""
    values = np.asarray(replicates, dtype=float)
    below, above = float(np.mean(values < threshold)), float(np.mean(values > threshold))
    return (below, above) if side == "above" else (above, below)
