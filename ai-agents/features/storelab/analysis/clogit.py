"""Conditional logit with an outside option for one E1 cell (SL-15).

Model (SL-06, ``v1.tex`` l.289), on the σ scale: displayed product j in position k
with label ℓ has utility v_j/σ + (δ/σ) ℓ + β_k/σ, the outside option 0, and choice is
multinomial logit. σ is not identified; the outside option's score and β_1 are
normalised to 0, so the parameters are one score per (task, product), δ/σ and β_k/σ
for k = 2..K (``names``). The MLE uses Newton–Fisher scoring with analytic score and
information (the log likelihood is concave) and step halving.

**Estimator** (registered, SL-01 register ``conventions.separation``): Firth's
penalised likelihood in every cell, which removes the MLE's first-order bias and stays
finite under separation. **Separation** is detected and reported exactly by the linear program of Konis (2007) adapted to the conditional
logit: a direction d ≠ 0 with (x_chosen − x_a)·d ≥ 0 for every trial and every other
alternative a, strict for some, exists iff the likelihood keeps increasing along d,
so some MLEs diverge. The Firth-penalised likelihood ℓ*(θ) = ℓ(θ) + ½ log det I(θ)
is maximised by Fisher scoring on the modified score U + ½ Σ_a p h z over every
alternative including the outside option (h the leverages), with step halving; it
always has a finite maximum.

**Intervals**: Wald (from I⁻¹) and profile (penalised) likelihood: the values of a
parameter where 2(ℓ̂ − ℓ_profile) equals the χ²₁ quantile. ``cluster_bootstrap``
resamples candidate sets with replacement (B = 2,000 registered) and refits.

``to_model_params`` writes a fit as SL-06 ``ModelParams`` (σ = 1, outside 0, β_1 = 0),
so ``SimulatedAgent`` reproduces the fitted choice probabilities exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import optimize, sparse, stats

from ..core.agent_params import AgentParams, ModelParams
from ..core.rng import stream
from .data import ChoiceData

CHI2_95 = float(stats.chi2.ppf(0.95, 1))
SEPARATION_TOL = 1e-7
MAX_STEP = 10.0


@dataclass
class SeparationCheck:
    separated: bool
    objective: float  # the LP's maximum; > 0 means separation
    parameters: list[str] = field(default_factory=list)  # names with a nonzero direction component


@dataclass
class ClogitFit:
    names: list[str]
    theta: np.ndarray
    se: np.ndarray
    loglik: float
    penalised_loglik: float | None
    firth: bool
    separation: SeparationCheck | None
    converged: bool
    iterations: int
    n_trials: int
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    products: list[tuple[str, str]] = field(default_factory=list)
    max_position: int = 1

    def value(self, name: str) -> float:
        return float(self.theta[self.names.index(name)])

    def std_error(self, name: str) -> float:
        return float(self.se[self.names.index(name)])

    def wald_ci(self, name: str, level: float = 0.95) -> tuple[float, float]:
        z = float(stats.norm.ppf(0.5 + level / 2))
        value, se = self.value(name), self.std_error(name)
        return value - z * se, value + z * se

    @property
    def label_effect(self) -> float:
        return self.value("delta")

    @property
    def position_effects(self) -> list[float]:
        return [0.0] + [self.value("beta_%d" % k) for k in range(2, self.max_position + 1)]

    def scores(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for i, (task, product) in enumerate(self.products):
            out.setdefault(task, {})[product] = float(self.theta[i])
        return out


def parameter_names(data: ChoiceData) -> list[str]:
    return (["v:%s/%s" % pair for pair in data.products] + ["delta"]
            + ["beta_%d" % k for k in range(2, data.max_position + 1)])


class Model:
    """The design matrix of a ChoiceData (one row per trial and slot) and its
    log likelihood, score and information."""

    def __init__(self, data: ChoiceData):
        self.data = data
        t, s = data.prod.shape
        self.t, self.s = t, s
        self.n_products = len(data.products)
        self.names = parameter_names(data)
        self.p = len(self.names)
        valid = data.prod >= 0
        self.valid = valid
        rows, cols, vals = [], [], []
        flat = np.arange(t * s).reshape(t, s)
        r = flat[valid]
        rows += [r, r]
        cols += [data.prod[valid], np.full(r.size, self.n_products)]
        vals += [np.ones(r.size), data.label[valid]]
        beta = valid & (data.pos >= 2)
        rows.append(flat[beta])
        cols.append(self.n_products + 1 + data.pos[beta] - 2)
        vals.append(np.ones(int(beta.sum())))
        self.x = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(t * s, self.p))
        self.y = np.zeros((t, s))
        picked = data.chosen >= 0
        self.y[np.arange(t)[picked], data.chosen[picked]] = 1.0
        self.group = sparse.csr_matrix((np.ones(t * s), (np.repeat(np.arange(t), s), np.arange(t * s))),
                                       shape=(t, t * s))
        self.active = np.ones(self.p, dtype=bool)  # fit() narrows it to the parameters the data inform

    def probabilities(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(slot probabilities (T, S), outside probabilities (T,))."""
        utility = (self.x @ theta).reshape(self.t, self.s)
        utility = np.where(self.valid, utility, -np.inf)
        top = np.maximum(utility.max(axis=1), 0.0)
        e = np.exp(utility - top[:, None])
        e0 = np.exp(-top)
        total = e.sum(axis=1) + e0
        return e / total[:, None], e0 / total

    def loglik(self, theta: np.ndarray) -> float:
        p, p0 = self.probabilities(theta)
        chosen = self.data.chosen
        picked = chosen >= 0
        probs = np.where(picked, p[np.arange(self.t), np.maximum(chosen, 0)], p0)
        return float(np.log(np.maximum(probs, 1e-300)).sum())

    def score_info(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(score, information, slot probabilities flattened)."""
        p, _ = self.probabilities(theta)
        flat = p.ravel()
        score = self.x.T @ (self.y.ravel() - flat)
        weighted = sparse.diags(flat) @ self.x
        mean = self.group @ weighted  # (T, P): x̄_t
        info = (self.x.T @ weighted).toarray() - (mean.T @ mean).toarray()
        return np.asarray(score).ravel(), info, flat

    def firth_score(self, theta: np.ndarray, info: np.ndarray, flat: np.ndarray, active: np.ndarray,
                    chunk: int = 2000) -> np.ndarray:
        """∇ ½ log det I = ½ Σ_t Σ_a p_ta h_ta z_ta over every alternative a, the outside
        option included (x_0 = 0, so z_0 = −x̄_t), with z_ta = x_ta − x̄_t and leverages
        h_ta = z_ta' I⁻¹ z_ta (active parameters)."""
        inverse = np.zeros_like(info)
        inverse[np.ix_(active, active)] = np.linalg.inv(info[np.ix_(active, active)])
        out = np.zeros(self.p)
        for start in range(0, self.t, chunk):
            stop = min(self.t, start + chunk)
            rows = slice(start * self.s, stop * self.s)
            x = self.x[rows].toarray()
            p = flat[rows]
            slots = p.reshape(-1, self.s)
            mean = (slots[:, :, None] * x.reshape(-1, self.s, self.p)).sum(axis=1)
            z = x - np.repeat(mean, self.s, axis=0)
            h = ((z @ inverse) * z).sum(axis=1)
            out += 0.5 * z.T @ (p * h)
            p0 = 1.0 - slots.sum(axis=1)  # the outside option: z_0 = −x̄_t
            h0 = ((mean @ inverse) * mean).sum(axis=1)
            out -= 0.5 * mean.T @ (p0 * h0)
        return out


def _logdet(info: np.ndarray, active: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(info[np.ix_(active, active)])
    return value if sign > 0 else -np.inf


def _maximise(model: Model, theta: np.ndarray, free: np.ndarray, firth: bool, max_iter: int = 200,
              tol: float = 1e-9) -> tuple[np.ndarray, float, bool, int]:
    """Fisher scoring on the free parameters, with step halving on the objective (the
    log likelihood, or under Firth the penalised log likelihood, for which I⁻¹ times
    the modified score is an ascent direction); returns (theta, objective, converged,
    iterations)."""
    active = model.active

    def objective(th: np.ndarray) -> float:
        value = model.loglik(th)
        if firth:
            _, info, _ = model.score_info(th)
            value += 0.5 * _logdet(info, active)
        return value

    current = objective(theta)
    for iteration in range(1, max_iter + 1):
        score, info, flat = model.score_info(theta)
        if firth:
            score = score + model.firth_score(theta, info, flat, active)
        if np.abs(score[free]).max(initial=0.0) < 1e-8:
            return theta, current, True, iteration
        sub = info[np.ix_(free, free)]
        try:
            step_free = np.linalg.solve(sub, score[free])
        except np.linalg.LinAlgError:
            step_free = np.linalg.lstsq(sub, score[free], rcond=None)[0]
        largest = np.abs(step_free).max(initial=0.0)
        if largest > MAX_STEP:
            step_free *= MAX_STEP / largest
        step = np.zeros_like(theta)
        step[free] = step_free
        scale = 1.0
        while True:
            candidate = theta + scale * step
            value = objective(candidate)
            if value >= current - 1e-12 or scale < 1e-6:
                break
            scale /= 2
        theta, current = candidate, value
        if np.abs(scale * step_free).max(initial=0.0) < tol:
            return theta, current, True, iteration
    return theta, current, False, max_iter


def detect_separation(data: ChoiceData, model: Model | None = None) -> SeparationCheck:
    """The LP test: max Σ (x_c − x_a)·d s.t. (x_c − x_a)·d ≥ 0, −1 ≤ d ≤ 1."""
    model = model or Model(data)
    t, s = model.t, model.s
    chosen = data.chosen
    trial, slot = np.nonzero(model.valid)
    row = trial * s + slot
    picked = chosen[trial] >= 0
    is_chosen = picked & (slot == chosen[trial])
    # pairs (chosen row or none for the outside option, alternative row or none for the outside option)
    first = np.concatenate([row[is_chosen], (trial * s + chosen[trial])[picked & ~is_chosen], np.full(
        int((~picked).sum()), -1)])
    second = np.concatenate([np.full(int(is_chosen.sum()), -1), row[picked & ~is_chosen], row[~picked]])
    count = first.size

    def select(rows: np.ndarray) -> sparse.csr_matrix:
        keep = rows >= 0
        return sparse.csr_matrix((np.ones(int(keep.sum())), (np.flatnonzero(keep), rows[keep])), shape=(count, t * s))

    differences = (select(first) @ model.x - select(second) @ model.x).tocsr()
    active = model.active
    differences = differences[:, active]
    result = optimize.linprog(-np.asarray(differences.sum(axis=0)).ravel(), A_ub=-differences,
                              b_ub=np.zeros(differences.shape[0]), bounds=(-1, 1), method="highs")
    if result.status != 0:
        raise RuntimeError("separation LP failed: %s" % result.message)
    objective = -float(result.fun)
    names = [n for n, d in zip(np.asarray(model.names)[active], result.x) if abs(d) > 1e-6]
    separated = objective > SEPARATION_TOL
    return SeparationCheck(separated, objective, names if separated else [])


def fit(data: ChoiceData, penalty: str = "firth", profile: list[str] | None = None, level: float = 0.95,
        detect: bool = True) -> ClogitFit:
    """The registered estimator is Firth's penalised likelihood in every cell (owner
    decision 2026-10-23); ``none`` gives the MLE (Hausman–McFadden needs it) and
    ``auto`` the MLE unless separation is detected. Separation is checked and reported
    either way (``detect``). ``profile`` names the parameters given (penalised)
    profile-likelihood intervals."""
    if penalty not in ("auto", "none", "firth"):
        raise ValueError("penalty: auto, none or firth")
    model = Model(data)
    _, info0, _ = model.score_info(np.zeros(model.p))
    model.active = np.diag(info0) > 1e-12  # parameters the data inform (a product absent from a resample is not)
    separation = detect_separation(data, model) if detect else None
    firth = penalty == "firth" or (penalty == "auto" and separation is not None and separation.separated)
    start = np.zeros(model.p)
    if firth and separation is not None and not separation.separated:  # the MLE exists: start Firth from it
        start = _maximise(model, start, model.active.copy(), False)[0]
    theta, objective, converged, iterations = _maximise(model, start, model.active.copy(), firth)
    _, info, _ = model.score_info(theta)
    se = np.full(model.p, np.nan)
    active = model.active
    covariance = np.linalg.inv(info[np.ix_(active, active)])
    se[active] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    theta = np.where(active, theta, np.nan)
    result = ClogitFit(model.names, theta, se, model.loglik(np.nan_to_num(theta)),
                       objective if firth else None, firth, separation, converged, iterations, data.n_trials,
                       products=list(data.products), max_position=data.max_position)
    for name in profile or []:
        result.ci[name] = profile_ci(model, result, name, level, objective)
    return result


def profile_ci(model: Model, result: ClogitFit, name: str, level: float, best: float) -> tuple[float, float]:
    """Where 2(ℓ̂ − ℓ_profile) = χ²₁(level), penalised under Firth; ±inf if unbounded."""
    i = model.names.index(name)
    quantile = float(stats.chi2.ppf(level, 1))
    free = model.active.copy()
    free[i] = False
    start = np.nan_to_num(result.theta)
    cache = {"theta": start}

    def gap(value: float) -> float:
        theta = cache["theta"].copy()
        theta[i] = value
        theta, objective, _, _ = _maximise(model, theta, free, result.firth, max_iter=100, tol=1e-8)
        cache["theta"] = theta
        return 2 * (best - objective) - quantile

    bounds = []
    se = result.se[i] if np.isfinite(result.se[i]) and result.se[i] > 0 else 1.0
    for direction in (-1, 1):
        cache["theta"] = start.copy()
        width, inner = 2.0 * se, start[i]
        outer = inner + direction * width
        found = False
        for _ in range(12):
            if gap(outer) > 0:
                found = True
                break
            inner, width = outer, width * 2
            outer = start[i] + direction * width
        if not found:
            bounds.append(direction * math.inf)
            continue
        cache["theta"] = start.copy()
        bounds.append(optimize.brentq(gap, min(inner, outer), max(inner, outer), xtol=1e-5))
    return bounds[0], bounds[1]


def cluster_bootstrap(data: ChoiceData, statistic: Callable[[ChoiceData], np.ndarray | float], replicates: int,
                      seed: int, analysis_id: str) -> np.ndarray:
    """``statistic`` on ``replicates`` resamples of the candidate sets (with
    replacement, all trials of a set together), from stream(seed, "bootstrap", analysis_id)."""
    rng = stream(seed, "bootstrap", analysis_id)
    sets = list(dict.fromkeys(data.set_ids))
    rows_of = {s: [] for s in sets}
    for i, s in enumerate(data.set_ids):
        rows_of[s].append(i)
    out = []
    for _ in range(replicates):
        drawn = rng.integers(0, len(sets), len(sets))
        rows = np.concatenate([rows_of[sets[j]] for j in drawn])
        out.append(np.atleast_1d(statistic(data.subset(rows))))
    return np.asarray(out)


def to_model_params(result: ClogitFit, model_key: str, page_request_prob: float) -> ModelParams:
    """The fit as SL-06 parameters: σ = 1, outside 0, β_1 = 0, scores v/σ per task."""
    if not np.isfinite(result.theta).all():
        raise ValueError("a parameter is not estimated (a product absent from the data)")
    return ModelParams(model_key=model_key, sigma=1.0, label_effect=result.label_effect,
                       position_effects=result.position_effects, outside_value=0.0,
                       page_request_prob=page_request_prob, scores=result.scores())


def agent_params_from_fits(fits: dict[str, tuple[ClogitFit, float]], provenance: dict[str, str]) -> AgentParams:
    """``agent_params.json`` content (source ``estimated``): one model per key, from
    its cell fit and its page-request rate χ (H1e; 0 for a model without browser data)."""
    models = [to_model_params(result, key, chi) for key, (result, chi) in fits.items()]
    return AgentParams(source="estimated", models=models, provenance=dict(provenance))
