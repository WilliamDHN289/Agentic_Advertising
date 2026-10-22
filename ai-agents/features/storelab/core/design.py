"""E1 set design (SL-05): candidate sets, presentations and their diagnostics.

For one task with eligible set E(x) of v products and one set size K:

- K = 2: every pair (``all_pairs``).
- K = v: the whole eligible set (``complete``).
- Otherwise a BIBD when BIBD_TABLE holds a construction for (v, K) (``bibd``), else
  a pair-covering design (``pair_covering``): the fewest blocks the search finds that
  cover every pair, starting from the Schönheim bound, with the variance of the
  pair co-occurrence counts minimised by local search.

Blocks are built on points 0..v-1 and mapped to products by a permutation drawn from
``stream(seed, "design", task_id, K)``, which also drives the covering search. Each
block is a candidate set. Presentations follow tickets.md A15: positions are a
uniform random permutation and each product's sponsored label is Bernoulli(label
rate), both drawn from ``stream(seed, "presentation", set_id, draw)``. Under
``per_set`` one draw (0) per set is repeated identically in every repetition; under
``per_repetition`` repetition r uses draw r.

``design_hash`` covers the realised design (sets, presentations and their
parameters), not the seed, so equal hashes mean equal stimuli.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from .config import NonEmptyStr, Probability, Section, canonical_json
from .protocol import Candidate, ProductId
from .rng import stream

DESIGN_VERSION = "e1_design@1"
Z_99 = 2.5758293035489004  # two-sided 99% normal quantile


# ------------------------------------------------------------- constructions
def _cyclic(v: int, *bases: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [tuple(sorted((x + shift) % v for x in base)) for base in bases for shift in range(v)]


def _affine_plane_gf4() -> list[tuple[int, ...]]:
    """AG(2, 4): points (x, y) of GF(4)^2 as 4x + y; lines y = mx + c and x = c."""
    mul = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]  # GF(4), a^2 = a + 1; addition is XOR
    lines = [tuple(sorted(4 * x + (mul[m][x] ^ c) for x in range(4))) for m in range(4) for c in range(4)]
    return lines + [tuple(4 * c + y for y in range(4)) for c in range(4)]


def _hadamard_16_8() -> list[tuple[int, ...]]:
    """Sylvester Hadamard matrix of order 16: the +1 and the -1 columns of rows 1..15."""
    sign = [[bin(i & j).count("1") % 2 for j in range(16)] for i in range(16)]
    return [tuple(j for j in range(16) if sign[i][j] == s) for i in range(1, 16) for s in (0, 1)]


# (v, K) -> (description, blocks); tests check each is a BIBD.
BIBD_TABLE: dict[tuple[int, int], tuple[str, list[tuple[int, ...]]]] = {
    (10, 4): ("(10,4,2) BIBD, explicit blocks", [
        (0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6), (0, 3, 7, 8), (0, 5, 7, 9), (0, 6, 8, 9), (1, 2, 7, 8),
        (1, 3, 6, 9), (1, 4, 7, 9), (1, 5, 6, 8), (2, 3, 5, 9), (2, 4, 8, 9), (2, 5, 6, 7), (3, 4, 5, 8),
        (3, 4, 6, 7)]),
    (13, 4): ("(13,4,1) projective plane PG(2,3), cyclic difference set {0,1,3,9} mod 13",
              _cyclic(13, (0, 1, 3, 9))),
    (16, 4): ("(16,4,1) affine plane AG(2,4)", _affine_plane_gf4()),
    (15, 8): ("(15,8,4) complement of the Singer (15,7,3) difference set {0,1,2,4,5,8,10} mod 15",
              _cyclic(15, (3, 6, 7, 9, 11, 12, 13, 14))),
    (16, 8): ("(16,8,7) Hadamard design from the Sylvester matrix of order 16", _hadamard_16_8()),
}


def pair_counts(v: int, blocks: list[tuple[int, ...]]) -> np.ndarray:
    """The v x v co-occurrence matrix: counts[i, j] = blocks containing i and j (i != j)."""
    counts = np.zeros((v, v), dtype=np.int64)
    for block in blocks:
        for i, j in itertools.combinations(block, 2):
            counts[i, j] += 1
            counts[j, i] += 1
    return counts


def lambda_range(v: int, blocks: list[tuple[int, ...]]) -> tuple[int, int]:
    counts = pair_counts(v, blocks)
    off = counts[~np.eye(v, dtype=bool)]
    return int(off.min()), int(off.max())


def schonheim_bound(v: int, k: int) -> int:
    """Lower bound on the blocks of a (v, k, 2) covering."""
    return math.ceil(v / k * math.ceil((v - 1) / (k - 1)))


def _search(v: int, k: int, b: int, rng: np.random.Generator, sweeps: int) -> list[tuple[int, ...]]:
    """Local search for b blocks of size k minimising (uncovered pairs, sum of squared
    pair counts), lexicographically; starts from a randomised greedy cover."""
    lam = np.zeros((v, v), dtype=np.int64)
    blocks = []
    for _ in range(b):
        block = [int(rng.integers(v))]
        while len(block) < k:
            others = [p for p in range(v) if p not in block]
            gain = [sum(lam[p, q] == 0 for q in block) for p in others]
            best = max(gain)
            choices = [p for p, g in zip(others, gain) if g == best]
            block.append(choices[int(rng.integers(len(choices)))])
        for i, j in itertools.combinations(block, 2):
            lam[i, j] += 1
            lam[j, i] += 1
        blocks.append(block)
    big = 4 * b * k * k  # one uncovered pair outweighs any change in the squared counts
    for _ in range(sweeps * b * k):
        bi = int(rng.integers(b))
        block = blocks[bi]
        x = block[int(rng.integers(k))]
        z = int(rng.integers(v))
        if z in block:
            continue
        rest = [y for y in block if y != x]
        delta = 0
        for y in rest:
            delta += (1 - 2 * lam[x, y]) + (big if lam[x, y] == 1 else 0)
            delta += (1 + 2 * lam[z, y]) - (big if lam[z, y] == 0 else 0)
        if delta <= 0:
            for y in rest:
                lam[x, y] -= 1
                lam[y, x] -= 1
                lam[z, y] += 1
                lam[y, z] += 1
            block[block.index(x)] = z
    return [tuple(sorted(block)) for block in blocks]


def pair_covering(v: int, k: int, rng: np.random.Generator, sweeps: int = 200) -> list[tuple[int, ...]]:
    """The fewest blocks (from the Schönheim bound up) for which the search covers every pair."""
    b = schonheim_bound(v, k)
    while True:
        blocks = _search(v, k, b, rng, sweeps)
        if lambda_range(v, blocks)[0] >= 1:
            return blocks
        b += 1


# --------------------------------------------------------------------- design
class CandidateSet(Section):
    set_id: NonEmptyStr
    product_ids: list[ProductId] = Field(min_length=2)  # sorted


class Presentation(Section):
    presentation_id: NonEmptyStr
    set_id: NonEmptyStr
    draw: NonNegativeInt  # 0 under per_set; the repetition under per_repetition
    candidates: list[Candidate] = Field(min_length=2)  # in position order


class Design(Section):
    design_version: Literal["e1_design@1"] = DESIGN_VERSION
    task_id: NonEmptyStr
    k: PositiveInt
    eligible_ids: list[ProductId] = Field(min_length=2)
    design_type: Literal["all_pairs", "complete", "bibd", "pair_covering"]
    construction: NonEmptyStr
    lambda_min: NonNegativeInt
    lambda_max: NonNegativeInt
    randomisation: Literal["per_set", "per_repetition"]
    label_rate: Probability
    repetitions: PositiveInt
    sets: list[CandidateSet] = Field(min_length=1)
    presentations: list[Presentation] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Design:
        draws = 1 if self.randomisation == "per_set" else self.repetitions
        if len(self.presentations) != len(self.sets) * draws:
            raise ValueError("presentations: %d, expected %d" % (len(self.presentations), len(self.sets) * draws))
        if any(len(s.product_ids) != self.k for s in self.sets):
            raise ValueError("sets: every set must have k products")
        return self

    def presentation_for(self, set_id: str, rep: int) -> Presentation:
        """The presentation shown in repetition ``rep`` (0-based) of ``set_id``."""
        if not 0 <= rep < self.repetitions:
            raise ValueError("rep %d outside 0..%d" % (rep, self.repetitions - 1))
        draw = 0 if self.randomisation == "per_set" else rep
        return next(p for p in self.presentations if p.set_id == set_id and p.draw == draw)

    def design_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()


def blocks_for(v: int, k: int, rng: np.random.Generator) -> tuple[str, str, list[tuple[int, ...]]]:
    """(design_type, construction, blocks) on points 0..v-1."""
    if not 2 <= k <= v:
        raise ValueError("set size %d outside 2..v=%d" % (k, v))
    if k == 2:
        return "all_pairs", "all %d pairs" % (v * (v - 1) // 2), list(itertools.combinations(range(v), 2))
    if k == v:
        return "complete", "the whole eligible set", [tuple(range(v))]
    if (v, k) in BIBD_TABLE:
        description, blocks = BIBD_TABLE[v, k]
        return "bibd", description, list(blocks)
    blocks = pair_covering(v, k, rng)
    return "pair_covering", "local-search pair covering, %d blocks (Schönheim bound %d)" % (
        len(blocks), schonheim_bound(v, k)), blocks


def presentation(set_id: str, product_ids: list[str], draw: int, label_rate: float, seed: int) -> Presentation:
    rng = stream(seed, "presentation", set_id, draw)
    positions = rng.permutation(len(product_ids)) + 1
    labels = rng.random(len(product_ids)) < label_rate
    candidates = sorted((Candidate(product_id=pid, position=int(pos), sponsored_label=bool(lab))
                         for pid, pos, lab in zip(product_ids, positions, labels)), key=lambda c: c.position)
    return Presentation(presentation_id="%s/p%d" % (set_id, draw), set_id=set_id, draw=draw, candidates=candidates)


def build_design(task_id: str, eligible_ids: list[str], k: int, repetitions: int, label_rate: float,
                 randomisation: Literal["per_set", "per_repetition"], seed: int) -> Design:
    eligible = sorted(eligible_ids)
    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible_ids: duplicates")
    rng = stream(seed, "design", task_id, k)
    mapping = rng.permutation(len(eligible))
    design_type, construction, blocks = blocks_for(len(eligible), k, rng)
    lam_min, lam_max = lambda_range(len(eligible), blocks)
    sets = [CandidateSet(set_id="%s/k%d/s%03d" % (task_id, k, i), product_ids=sorted(eligible[int(mapping[p])]
                                                                                       for p in block))
            for i, block in enumerate(blocks)]
    draws = range(1 if randomisation == "per_set" else repetitions)
    presentations = [presentation(s.set_id, s.product_ids, d, label_rate, seed) for s in sets for d in draws]
    return Design(task_id=task_id, k=k, eligible_ids=eligible, design_type=design_type, construction=construction,
                  lambda_min=lam_min, lambda_max=lam_max, randomisation=randomisation, label_rate=label_rate,
                  repetitions=repetitions, sets=sets, presentations=presentations)


def designs_hash(designs: list[Design]) -> str:
    """sha256 over the design hashes, in order."""
    return hashlib.sha256(canonical_json([d.design_hash() for d in designs]).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- diagnostics
def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-squared distribution: Q(df/2, x/2), the regularised gamma."""
    if df < 1 or x < 0:
        raise ValueError("chi2_sf needs df >= 1 and x >= 0")
    a, y = df / 2.0, x / 2.0
    if y == 0:
        return 1.0
    log_prefix = -y + a * math.log(y) - math.lgamma(a)
    if y < a + 1:  # series for the lower tail
        term = total = 1.0 / a
        n = a
        for _ in range(10_000):
            n += 1
            term *= y / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300  # modified Lentz continued fraction for the upper tail
    b = y + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-16:
            break
    return math.exp(log_prefix) * h


class ProductBalance(Section):
    product_id: ProductId
    sets: NonNegativeInt  # candidate sets containing it
    draws: NonNegativeInt  # presentation draws containing it
    labelled: NonNegativeInt
    position_counts: list[NonNegativeInt]  # index i: position i + 1
    chi2: float | None  # position uniformity; None when the product is in no draw
    p_value: float | None


class DesignDiagnostics(Section):
    task_id: NonEmptyStr
    k: PositiveInt
    design_type: NonEmptyStr
    lambda_min: NonNegativeInt
    lambda_max: NonNegativeInt
    eligible_ids: list[ProductId]
    cooccurrence: list[list[NonNegativeInt]]  # rows and columns in eligible_ids order
    products: list[ProductBalance]
    label_draws: NonNegativeInt
    labelled: NonNegativeInt


def diagnose(design: Design) -> DesignDiagnostics:
    index = {pid: i for i, pid in enumerate(design.eligible_ids)}
    blocks = [tuple(index[p] for p in s.product_ids) for s in design.sets]
    counts = pair_counts(len(index), blocks)
    sets = {pid: sum(pid in s.product_ids for s in design.sets) for pid in index}
    positions = {pid: [0] * design.k for pid in index}
    labelled = dict.fromkeys(index, 0)
    for p in design.presentations:
        for c in p.candidates:
            positions[c.product_id][c.position - 1] += 1
            labelled[c.product_id] += c.sponsored_label
    products = []
    for pid in design.eligible_ids:
        observed = positions[pid]
        n = sum(observed)
        if n:
            expected = n / design.k
            chi2 = sum((o - expected) ** 2 / expected for o in observed)
            p_value = chi2_sf(chi2, design.k - 1)
        else:
            chi2 = p_value = None
        products.append(ProductBalance(product_id=pid, sets=sets[pid], draws=n, labelled=labelled[pid],
                                       position_counts=observed, chi2=chi2, p_value=p_value))
    return DesignDiagnostics(task_id=design.task_id, k=design.k, design_type=design.design_type,
                             lambda_min=design.lambda_min, lambda_max=design.lambda_max,
                             eligible_ids=design.eligible_ids, cooccurrence=counts.tolist(), products=products,
                             label_draws=sum(len(p.candidates) for p in design.presentations),
                             labelled=sum(labelled.values()))


class LabelRateCheck(Section):
    draws: NonNegativeInt  # N: label draws (one per product per presentation draw)
    labelled: NonNegativeInt
    rate: float
    target: Probability
    bound: float  # Z_99 * sqrt(p (1 - p) / N)
    within: bool


def label_rate_check(designs: list[Design]) -> LabelRateCheck:
    """The overall sponsored-label rate against target ± the 99% binomial bound for N."""
    targets = {d.label_rate for d in designs}
    if len(targets) != 1:
        raise ValueError("designs disagree on label_rate: %s" % sorted(targets))
    target = targets.pop()
    draws = sum(len(p.candidates) for d in designs for p in d.presentations)
    labelled = sum(c.sponsored_label for d in designs for p in d.presentations for c in p.candidates)
    if not draws:
        raise ValueError("no label draws")
    rate = labelled / draws
    bound = Z_99 * math.sqrt(target * (1 - target) / draws)
    return LabelRateCheck(draws=draws, labelled=labelled, rate=rate, target=target, bound=bound,
                          within=abs(rate - target) <= bound)
