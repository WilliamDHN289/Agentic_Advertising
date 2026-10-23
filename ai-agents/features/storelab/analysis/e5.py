"""E5 field experiment: data contract, request table, H5a and H5b analyses, power (SL-22).

The protocol is ``docs/e5_protocol.md`` (``PROTOCOL_VERSION``); the registered rules are
the H5a and H5b entries of ``docs/claims_register.yaml``.

**Events** (``contracts/storelab/e5_event.schema.json``): one JSON object per event, with
a type of ``assignment``, ``click``, ``qualifying_event``, ``purchase`` or ``return``.
Request, event and order ids are sha256 digests (the partner hashes its ids with a
secret salt). No field holds personal data, and ``pii_fields`` scans a schema or a
record for field names that would.

**Request table** (``request_table``): one row per assigned, non-excluded request. Each
row holds the arm (strong-competitor or weak-competitor set), the assignment
probability and the traffic type, plus these indicators for the focal product:
- ``choice``: a qualifying event within the choice window;
- ``click``: a page request;
- ``purchase``: a purchase within the purchase window;
- ``net``: a purchase not fully returned within the return window.

Windows run from the assignment (protocol §5).

**H5a** (``h5a``): τ = log(R_strong / R_weak), with R_a = Σ net / Σ choice in arm a.
Randomisation intervals invert Fisher randomisation tests of H0(τ₀): with the strong
arm's net outcomes divided by e^τ₀, the arm has no effect. The statistic is the
studentised log-ratio difference (delta-method standard error), which keeps the test
asymptotically valid when only the ratios are equal (choices may differ by arm). Its
null distribution comes from ``DRAWS`` = 10,000 draws of the actual assignment mechanism.
That mechanism is independent Bernoulli(π) per request, so a draw is the number of
requests of each (arm, outcome) type that it assigns to the strong arm. The outputs are:
- the TOST p-value at ±margin (``p_support``; supported ⇔ the 90% interval lies inside the margin);
- the dual p-value of "the 95% interval lies entirely outside" (``p_against``);
- both intervals.

**H5b** (``h5b``): per arm and, where identified, traffic type. The out-of-sample
log-loss of predicting a purchase from the qualifying event is compared with that from
the click. Predictions come from K-fold cross-fitting, with folds keyed by request id,
and the predictor is the smoothed purchase rate given the indicator. The cell's mean
loss difference (event − click) gets a percentile bootstrap that resamples requests
within the cell, so the randomised arms are never mixed. Its one-sided p-values feed
the registered Holm rule. AUC of each predictor is reported.

**Power** (``h5a_power``, ``h5a_required_n``, ``h5b_power``): normal approximations
parameterised by the partner's baseline rates.
"""
from __future__ import annotations

import datetime
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Iterable, Literal, Union

import numpy as np
from pydantic import AwareDatetime, Field, PositiveInt, TypeAdapter, model_validator
from scipy import optimize, stats

from ..core.config import Section
from ..core.rng import stream
from .intervals import bootstrap_p_values, percentile

PROTOCOL_VERSION = "e5_protocol@1"
DRAWS = 10_000
MARGIN = 0.10
BOOTSTRAP = 2_000
FOLDS = 5
SMOOTHING = 0.5  # added to each count of the cross-fitted purchase rates
CHOICE_WINDOW = datetime.timedelta(hours=24)
PURCHASE_WINDOW = datetime.timedelta(days=7)
RETURN_WINDOW = datetime.timedelta(days=30)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CatalogId = Annotated[str, Field(pattern=r"^[A-Za-z0-9._:-]{1,64}$")]


# ----------------------------------------------------------------- contract
class _Event(Section):
    schema_version: Literal[1] = 1
    protocol_version: Literal["e5_protocol@1"] = PROTOCOL_VERSION
    event_id: Digest
    request_id: Digest
    event_time_utc: AwareDatetime
    product_id: CatalogId  # the partner's catalog id of the product the event concerns


class AssignmentEvent(_Event):
    """Logged before the candidate set is rendered; product_id is the focal product."""
    event_type: Literal["assignment"]
    arm: Literal["strong", "weak"]
    assignment_probability: Annotated[float, Field(gt=0.0, lt=1.0)]  # π of the strong arm
    candidate_set_id: CatalogId
    candidate_product_ids: list[CatalogId] = Field(min_length=2)
    traffic_type: Literal["human", "agent", "unknown"]
    exclusion: Literal["none", "partner_traffic_filter", "internal_test", "focal_unavailable"] = "none"

    @model_validator(mode="after")
    def _focal_displayed(self) -> AssignmentEvent:
        if self.product_id not in self.candidate_product_ids:
            raise ValueError("the focal product must be in the candidate set")
        return self


class ClickEvent(_Event):
    """A page request of a displayed product (human or agent)."""
    event_type: Literal["click"]


class QualifyingEvent(_Event):
    """The effective choice: a confirmed selection or an add-to-cart instruction."""
    event_type: Literal["qualifying_event"]
    kind: Literal["confirmed_selection", "add_to_cart"]


class PurchaseEvent(_Event):
    event_type: Literal["purchase"]
    order_id: Digest
    quantity: PositiveInt


class ReturnEvent(_Event):
    event_type: Literal["return"]
    order_id: Digest
    quantity: PositiveInt


E5Event = Annotated[Union[AssignmentEvent, ClickEvent, QualifyingEvent, PurchaseEvent, ReturnEvent],
                    Field(discriminator="event_type")]
EVENT_ADAPTER = TypeAdapter(E5Event)


def e5_event_schema() -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "E5Event",
            **EVENT_ADAPTER.json_schema()}


def parse_events(lines: Iterable[str]) -> list:
    return [EVENT_ADAPTER.validate_json(line) for line in lines if line.strip()]


PII_FIELD = re.compile(r"(?i)(e[-_]?mail|(^|_)(first|last|full|given|family|customer|user|buyer|shopper)?_?name$|"
                       r"address|street|postcode|post_code|postal|zip_?code|phone|mobile|ip_?addr|birth|passport|"
                       r"card_?number|iban)")
PII_VALUE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\+?\d[\d ()-]{8,}\d|\b\d{1,3}(\.\d{1,3}){3}\b")


def pii_fields(obj: Any, path: str = "") -> list[str]:
    """Paths of keys (in a JSON Schema's properties, or in a record) whose names denote
    personal data, and of string values that look like an email, phone number or IP
    address. Schema keywords themselves (``properties``, ``$defs``, ...) are not fields."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = "%s/%s" % (path, key)
            schema_keyword = key.startswith("$") or key in {"properties", "items", "oneOf", "anyOf", "allOf",
                                                            "mapping", "discriminator", "required", "enum", "const"}
            if not schema_keyword and PII_FIELD.search(key) and key not in {"title", "description"}:
                found.append(here)
            if key in {"description", "title"}:
                continue
            found += pii_fields(value, here)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found += pii_fields(value, "%s[%d]" % (path, i))
    elif isinstance(obj, str) and PII_VALUE.search(obj) and not re.fullmatch(r"[0-9a-f]{64}", obj) and not _is_time(obj):
        found.append(path)
    return found


def _is_time(text: str) -> bool:
    try:
        datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------ request table
@dataclass
class RequestTable:
    request_id: list[str]
    strong: np.ndarray  # 1 = strong-competitor arm
    probability: np.ndarray  # π
    traffic: np.ndarray  # "human" | "agent" | "unknown"
    choice: np.ndarray
    click: np.ndarray
    purchase: np.ndarray
    net: np.ndarray
    excluded: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.request_id)

    def subset(self, mask: np.ndarray) -> RequestTable:
        return RequestTable([r for r, keep in zip(self.request_id, mask) if keep], self.strong[mask],
                            self.probability[mask], self.traffic[mask], self.choice[mask], self.click[mask],
                            self.purchase[mask], self.net[mask], dict(self.excluded))

    @classmethod
    def from_arrays(cls, strong, probability, traffic, choice, click, purchase, net, ids=None) -> RequestTable:
        n = len(strong)
        ids = ids if ids is not None else [digest("synthetic-%d" % i) for i in range(n)]
        return cls(list(ids), np.asarray(strong, dtype=np.int64), np.broadcast_to(probability, (n,)).astype(float),
                   np.asarray(traffic), np.asarray(choice, dtype=np.int64), np.asarray(click, dtype=np.int64),
                   np.asarray(purchase, dtype=np.int64), np.asarray(net, dtype=np.int64))


def request_table(events: Iterable) -> RequestTable:
    """Protocol §5–§6: assigned requests minus exclusions; outcomes of the focal product
    within the windows from the assignment. Counts of excluded requests by reason."""
    by_request: dict[str, list] = {}
    seen: set[str] = set()
    excluded = {"duplicate_event": 0, "no_assignment": 0, "several_assignments": 0, "partner_traffic_filter": 0,
                "internal_test": 0, "focal_unavailable": 0, "event_before_assignment": 0}
    for event in events:
        if event.event_id in seen:
            excluded["duplicate_event"] += 1
            continue
        seen.add(event.event_id)
        by_request.setdefault(event.request_id, []).append(event)
    rows = []
    for request_id, items in sorted(by_request.items()):
        assignments = [e for e in items if e.event_type == "assignment"]
        if not assignments:
            excluded["no_assignment"] += 1
            continue
        if len(assignments) > 1:
            excluded["several_assignments"] += 1
            continue
        a = assignments[0]
        if a.exclusion != "none":
            excluded[a.exclusion] += 1
            continue
        start, focal = a.event_time_utc, a.product_id
        later = [e for e in items if e is not a and e.product_id == focal]
        if any(e.event_time_utc < start for e in later):
            excluded["event_before_assignment"] += 1
            continue
        choice = any(e.event_type == "qualifying_event" and e.event_time_utc - start <= CHOICE_WINDOW for e in later)
        click = any(e.event_type == "click" for e in later)
        bought: dict[str, int] = {}
        for e in later:
            if e.event_type == "purchase" and e.event_time_utc - start <= PURCHASE_WINDOW:
                bought[e.order_id] = bought.get(e.order_id, 0) + e.quantity
        kept = 0
        for order, quantity in bought.items():
            purchase_time = min(e.event_time_utc for e in later if e.event_type == "purchase" and e.order_id == order)
            returned = sum(e.quantity for e in later if e.event_type == "return" and e.order_id == order
                           and e.event_time_utc - purchase_time <= RETURN_WINDOW)
            kept += max(quantity - returned, 0) > 0
        rows.append((request_id, a.arm == "strong", a.assignment_probability, a.traffic_type, choice, click,
                     bool(bought), kept > 0))
    if not rows:
        return RequestTable([], *(np.zeros(0, dtype=np.int64) for _ in range(2)), np.zeros(0, dtype="<U7"),
                            *(np.zeros(0, dtype=np.int64) for _ in range(4)), excluded)
    ids, strong, prob, traffic, choice, click, purchase, net = zip(*rows)
    table = RequestTable.from_arrays(strong, np.array(prob), np.array(traffic), choice, click, purchase, net, ids)
    table.excluded = excluded
    return table


# ---------------------------------------------------------------------- H5a
@dataclass
class H5aResult:
    estimate: float  # τ̂ = log R̂_strong − log R̂_weak
    standard_error: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    p_support: float  # TOST p-value at ±margin
    p_against: float  # dual of "the 95% interval lies entirely outside the margin"
    p_no_effect: float  # two-sided Fisher p-value of τ = 0
    n: int
    counts: dict[str, int]
    draws: int

    def decide_input(self) -> dict:
        return {"p_support": self.p_support, "p_against": self.p_against, "n": self.n, "estimate": self.estimate,
                "ci": self.ci90}


class _Randomisation:
    """The randomisation distribution of the per-arm sums, for any τ₀.

    Requests are grouped by (actual arm, choice, net); a draw of the Bernoulli(π)
    assignment is the number in each group assigned to the strong arm. Every
    statistic below depends on the data only through these sums."""

    def __init__(self, table: RequestTable, draws: int, rng: np.random.Generator):
        pi = np.unique(table.probability)
        if len(pi) != 1:
            raise ValueError("H5a needs one assignment probability (protocol §4); found %s" % pi[:5])
        self.pi = float(pi[0])
        keys = np.stack([table.strong, table.choice, table.net], axis=1)
        self.types, counts = np.unique(keys, axis=0, return_counts=True)
        self.counts = counts
        self.observed = self.types[:, 0] * counts  # every request of a type has its actual arm
        self.drawn = rng.binomial(counts[None, :], self.pi, size=(draws, len(counts)))

    def statistic(self, tau0: float, assigned: np.ndarray) -> np.ndarray:
        """Studentised τ̂(z) − 0 on the data with the strong arm's net divided by e^τ₀,
        for assignment counts ``assigned`` (rows: draws, or one row for the observed)."""
        choice, net = self.types[:, 1], self.types[:, 2] * np.exp(-tau0 * self.types[:, 0])
        out = []
        for arm_count in (assigned, self.counts[None, :] - assigned):  # strong, then weak (by drawn label)
            m = arm_count.sum(axis=1).astype(float)
            s_n = arm_count @ net
            s_c = arm_count @ choice
            s_nn, s_cc, s_nc = arm_count @ (net * net), arm_count @ (choice * choice), arm_count @ (net * choice)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_n, mean_c = s_n / m, s_c / m
                var_n = s_nn / m - mean_n ** 2
                var_c = s_cc / m - mean_c ** 2
                cov = s_nc / m - mean_n * mean_c
                log_ratio = np.log(mean_n / mean_c)
                variance = (var_n / mean_n ** 2 + var_c / mean_c ** 2 - 2 * cov / (mean_n * mean_c)) / m
            out.append((log_ratio, variance))
        (l1, v1), (l0, v0) = out
        with np.errstate(invalid="ignore", divide="ignore"):
            return (l1 - l0) / np.sqrt(v1 + v0)

    def p_greater(self, tau0: float) -> float:
        """P*(T ≥ T_obs) under H0(τ₀): small when τ > τ₀."""
        observed = self.statistic(tau0, self.observed[None, :])[0]
        drawn = self.statistic(tau0, self.drawn)
        return (1 + np.sum(~(drawn < observed))) / (1 + len(drawn))  # NaN draws count as at least as extreme

    def p_less(self, tau0: float) -> float:
        observed = self.statistic(tau0, self.observed[None, :])[0]
        drawn = self.statistic(tau0, self.drawn)
        return (1 + np.sum(~(drawn > observed))) / (1 + len(drawn))


def h5a(table: RequestTable, draws: int = DRAWS, seed: int = 0, analysis_id: str = "h5a",
        margin: float = MARGIN) -> H5aResult:
    rand = _Randomisation(table, draws, stream(seed, "e5", "h5a", analysis_id))
    strong, weak = table.strong == 1, table.strong == 0
    r1 = table.net[strong].sum() / table.choice[strong].sum()
    r0 = table.net[weak].sum() / table.choice[weak].sum()
    estimate = float(math.log(r1) - math.log(r0))
    t0 = rand.statistic(0.0, rand.observed[None, :])[0]
    se = float(estimate / t0) if t0 else float("nan")
    p_low, p_high = rand.p_greater(-margin), rand.p_less(margin)
    p_support = max(p_low, p_high)
    p_against = min(1.0, 2 * min(rand.p_greater(margin), rand.p_less(-margin)))
    p_no_effect = min(1.0, 2 * min(rand.p_greater(0.0), rand.p_less(0.0)))
    return H5aResult(estimate, se, _interval(rand, estimate, se, 0.05), _interval(rand, estimate, se, 0.025),
                     float(p_support), float(p_against), float(p_no_effect), table.n,
                     {"requests": table.n, "strong": int(strong.sum()), "choices": int(table.choice.sum()),
                      "net_purchases": int(table.net.sum())}, draws)


def _interval(rand: _Randomisation, estimate: float, se: float, tail: float) -> tuple[float, float]:
    """{τ₀ : both one-sided p-values exceed ``tail``}, by bracketing and bisection."""
    def lower(t):
        return rand.p_greater(t) - tail  # rises with t

    def upper(t):
        return rand.p_less(t) - tail  # falls with t

    width = 10 * (se if math.isfinite(se) and se > 0 else 1.0)
    low = optimize.brentq(lower, estimate - width, estimate, xtol=1e-6) if lower(estimate - width) < 0 else -math.inf
    high = optimize.brentq(upper, estimate, estimate + width, xtol=1e-6) if upper(estimate + width) < 0 else math.inf
    return float(low), float(high)


def h5a_standard_error(n: int, pi: float, choice_weak: float, choice_strong: float, net_per_choice: float,
                       tau: float = 0.0) -> float:
    """Delta-method SE of τ̂ for n requests: per arm, Var(log R̂_a) ≈ (1 − r_a) / (n_a c_a r_a)
    for binary choice (rate c_a) and net purchase given choice (rate r_a; strong arm r e^τ)."""
    r1 = net_per_choice * math.exp(tau)
    return math.sqrt((1 - r1) / (n * pi * choice_strong * r1)
                     + (1 - net_per_choice) / (n * (1 - pi) * choice_weak * net_per_choice))


def h5a_power(n: int, pi: float, choice_weak: float, choice_strong: float, net_per_choice: float, tau: float = 0.0,
              margin: float = MARGIN, alpha: float = 0.05) -> float:
    """TOST power at true τ: Φ((m − τ)/se − z) + Φ((m + τ)/se − z) − 1 (at least 0)."""
    se = h5a_standard_error(n, pi, choice_weak, choice_strong, net_per_choice, tau)
    z = stats.norm.ppf(1 - alpha)
    return max(0.0, float(stats.norm.cdf((margin - tau) / se - z) + stats.norm.cdf((margin + tau) / se - z) - 1))


def h5a_required_n(power: float, pi: float, choice_weak: float, choice_strong: float, net_per_choice: float,
                   tau: float = 0.0, margin: float = MARGIN, alpha: float = 0.05) -> int:
    """Smallest n with h5a_power ≥ ``power``."""
    if abs(tau) >= margin:
        raise ValueError("no sample size gives TOST power when |τ| ≥ margin")
    high = 100
    while h5a_power(high, pi, choice_weak, choice_strong, net_per_choice, tau, margin, alpha) < power:
        high *= 2
    low = high // 2
    while high - low > 1:
        mid = (low + high) // 2
        if h5a_power(mid, pi, choice_weak, choice_strong, net_per_choice, tau, margin, alpha) >= power:
            high = mid
        else:
            low = mid
    return high


# ---------------------------------------------------------------------- H5b
@dataclass
class H5bCell:
    cell: str
    n: int
    log_loss_event: float
    log_loss_click: float
    difference: float  # event − click; negative: the event predicts better
    ci: tuple[float, float]
    p_support: float  # against "upper bound < 0" (Holm in decide.py)
    p_against: float
    auc_event: float
    auc_click: float

    def decide_input(self) -> dict:
        return {"cell": self.cell, "n": self.n, "tests": {"log_loss_difference": {
            "p_support": self.p_support, "p_against": self.p_against, "estimate": self.difference, "ci": self.ci,
            "metrics": {"auc_event": self.auc_event, "auc_click": self.auc_click}}}}


def h5b_cells(table: RequestTable) -> list[tuple[str, np.ndarray]]:
    """Arm × traffic type where types are identified (rows of unknown type are left out of
    the cells); arm alone when none is."""
    arms = (("strong", table.strong == 1), ("weak", table.strong == 0))
    types = [t for t in ("human", "agent") if (table.traffic == t).any()]
    if not types:
        return [(name, mask) for name, mask in arms]
    return [("%s/%s" % (name, t), mask & (table.traffic == t)) for name, mask in arms for t in types]


def _fold(request_ids: list[str], folds: int) -> np.ndarray:
    return np.array([int(r[:8], 16) % folds for r in request_ids])


def _cross_fitted_loss(x: np.ndarray, y: np.ndarray, fold: np.ndarray, folds: int) -> np.ndarray:
    """Per-row out-of-sample log-loss of p̂(y = 1 | x), fitted on the other folds."""
    loss = np.zeros(len(y))
    for k in range(folds):
        test, train = fold == k, fold != k
        for value in (0, 1):
            cell = train & (x == value)
            p = (y[cell].sum() + SMOOTHING) / (cell.sum() + 2 * SMOOTHING)
            rows = test & (x == value)
            loss[rows] = -(y[rows] * np.log(p) + (1 - y[rows]) * np.log(1 - p))
    return loss


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC of a score against y (ties count one half)."""
    pos, neg = x[y == 1], x[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = stats.rankdata(np.r_[pos, neg])
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def h5b(table: RequestTable, bootstrap: int = BOOTSTRAP, folds: int = FOLDS, seed: int = 0,
        analysis_id: str = "h5b") -> list[H5bCell]:
    out = []
    fold = _fold(table.request_id, folds)
    for name, mask in h5b_cells(table):
        y, event, click = table.purchase[mask], table.choice[mask], table.click[mask]
        diff_rows = (_cross_fitted_loss(event, y, fold[mask], folds) - _cross_fitted_loss(click, y, fold[mask], folds))
        rng = stream(seed, "e5", "h5b", analysis_id, name)
        values, inverse = np.unique(diff_rows, return_inverse=True)  # a few distinct per-row values
        weights = np.bincount(inverse, minlength=len(values)) / len(diff_rows)
        draws = rng.multinomial(len(diff_rows), weights, size=bootstrap) @ values / len(diff_rows)
        p_support, p_against = bootstrap_p_values(draws, 0.0, "below")
        ll_event = float(_cross_fitted_loss(event, y, fold[mask], folds).mean())
        ll_click = float(_cross_fitted_loss(click, y, fold[mask], folds).mean())
        out.append(H5bCell(name, int(mask.sum()), ll_event, ll_click, float(diff_rows.mean()), percentile(draws),
                           p_support, p_against, _auc(event, y), _auc(click, y)))
    return out


def h5b_power(n_cell: int, joint: dict[tuple[int, int, int], float], cells: int, alpha: float = 0.025) -> float:
    """Normal-approximation power of one cell's one-sided test (Bonferroni over ``cells``,
    conservative for Holm) from the joint probabilities of (event, click, purchase):
    the per-row loss difference uses the population rates p(y | event) and p(y | click)."""
    def rate(index, value):
        mass = sum(p for k, p in joint.items() if k[index] == value)
        return sum(p for k, p in joint.items() if k[index] == value and k[2] == 1) / mass
    diffs, probs = [], []
    for (e, c, y), p in joint.items():
        pe, pc = rate(0, e), rate(1, c)
        loss = lambda q: -(y * math.log(q) + (1 - y) * math.log(1 - q))  # noqa: E731
        diffs.append(loss(pe) - loss(pc))
        probs.append(p)
    diffs, probs = np.array(diffs), np.array(probs)
    mean = float(probs @ diffs)
    sd = math.sqrt(float(probs @ (diffs - mean) ** 2))
    return float(stats.norm.cdf(-mean / (sd / math.sqrt(n_cell)) - stats.norm.ppf(1 - alpha / cells)))
