"""The claims register (SL-01): its model, JSON Schema and loader.

``docs/claims_register.yaml`` holds one entry per hypothesis H1a–H5b with every field
the registration needs (tickets.md SL-01 scope) and a machine-readable ``decision``
that ``decide.py`` applies. Every section rejects unknown fields; text fields must be
non-empty, so a claim cannot be registered with a field left blank. The published
schema is ``contracts/storelab/claims_register.schema.json`` (``cli schemas``).

A decision has groups (the claim's parts). A group is evaluated on cells (E1 primary
cells, E2 configurations, arms, ...): each cell's tests are combined (all must be
supported), cells are filtered by the floor and the missingness rules, and the cell
labels are aggregated (``two_thirds``, ``all`` or ``single``). Primary groups are then
combined into the claim's label; ``secondary`` groups are reported beside it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, StringConstraints, model_validator

CLAIM_IDS = ("H1a", "H1b", "H1c", "H1d", "H1e", "H2a", "H2b", "H2c", "H3a", "H3b", "H3c", "H4a", "H4b", "H4c",
             "H4d", "H5a", "H5b")
REGISTER_PATH = Path(__file__).resolve().parents[1] / "docs" / "claims_register.yaml"
REPO_ROOT = Path(__file__).resolve().parents[4]  # 7_experiments: requirements/ and tickets.md

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Name = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$")]
Alpha = Annotated[float, Field(gt=0.0, lt=0.5)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Label = Literal["supported", "not_supported", "inconclusive"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# --------------------------------------------------------------------- tests
class FirstCredibleChange(Strict):
    """Counts of the H1d alignment test derived from the credible changes themselves:
    each (provider version pair, task) counts at most once, through the first interface
    in ``order`` that shows a credible change; a change whose new and old favourites tie
    in u is excluded (the unit is not counted). The units are then pooled across
    providers: ``n`` is the number of units, ``aligned`` those whose new favourite has
    the higher u. Counting a task once per interface would break the independence of
    the binomial test."""

    order: list[Literal["text", "structured", "browser_tool"]] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct(self) -> FirstCredibleChange:
        if len(set(self.order)) != len(self.order):
            raise ValueError("first_credible_change: duplicate interfaces in order")
        return self


class BoundTest(Strict):
    """A bound of the cell's interval against a threshold (three-way rule).

    ``side`` is the claim's side: ``above`` means supported when the lower bound is at
    or above ``threshold`` (above it if ``strict``), not supported when the upper bound
    is below ``against_threshold`` (default ``threshold``). ``method`` ``p_values``
    reads the one-sided p-values from the results; ``clopper_pearson`` and ``wilson``
    compute them from counts (``support_count`` / ``against_count`` successes of ``n``).
    """

    kind: Literal["bound"]
    name: Name
    side: Literal["above", "below"]
    threshold: float | None = None
    threshold_rule: Text | None = None  # a data-dependent threshold the results state
    against_threshold: float | None = None
    strict: bool = False
    method: Literal["p_values", "clopper_pearson", "wilson"] = "p_values"
    support_count: Name | None = None
    against_count: Name | None = None
    first_credible_change: FirstCredibleChange | None = None
    alpha_support: Alpha
    alpha_against: Alpha

    @model_validator(mode="after")
    def _consistent(self) -> BoundTest:
        if self.first_credible_change is not None and (
                self.method == "p_values" or {self.support_count, self.against_count} != {"aligned"}):
            raise ValueError("%s: first_credible_change derives the counts n and aligned of a count rule" % self.name)
        if (self.threshold is None) == (self.threshold_rule is None):
            raise ValueError("%s: give exactly one of threshold and threshold_rule" % self.name)
        counts = self.method != "p_values"
        if counts and (self.support_count is None or self.against_count is None or self.threshold is None):
            raise ValueError("%s: method %s needs support_count, against_count and a numeric threshold"
                             % (self.name, self.method))
        if not counts and (self.support_count or self.against_count):
            raise ValueError("%s: counts are only read by clopper_pearson and wilson" % self.name)
        return self


class EquivalenceTest(Strict):
    """TOST: supported when the interval lies inside ``center`` ± ``margin``
    (``p_support`` is the TOST p-value), not supported when it lies entirely outside
    (``p_against``)."""

    kind: Literal["equivalence"]
    name: Name
    center: float = 0.0
    margin: PositiveFloat
    alpha_support: Alpha
    alpha_against: Alpha


class RejectsNullTest(Strict):
    """Supported when the null is rejected (Holm-adjusted ``p_value`` ≤ alpha); with no
    registered margin a non-rejection is inconclusive, never 'not supported'."""

    kind: Literal["rejects_null"]
    name: Name
    alpha_support: Alpha


class Check(Strict):
    """``metric op bound``. A ``monte_carlo`` check (A12 §6.2) is a tolerance on a quantity
    estimated from Monte Carlo replications: the results give each estimate with its
    Monte Carlo SE, and an estimate fails only if it is beyond the bound by more than
    z·SE, z = z_{1-0.05/m} over the m Monte Carlo estimates of the test."""

    metric: Name
    op: Literal["<=", ">=", "=="]
    bound: float
    monte_carlo: bool = False

    @model_validator(mode="after")
    def _one_sided(self) -> Check:
        if self.monte_carlo and self.op == "==":
            raise ValueError("%s: a Monte Carlo tolerance needs <= or >=" % self.metric)
        return self


class ImplementationCheck(Strict):
    """An exact (I-type) result: every metric within its tolerance. A failure is an
    implementation error, never a finding."""

    kind: Literal["implementation_check"]
    name: Name
    checks: list[Check] = Field(min_length=1)


class ContainsTest(Strict):
    """An exact (I-type) value checked by an interval (A12 §6.3, H4b): per cell, the
    Bonferroni interval at level 1 - ``alpha``/m (m cells) must contain ``value``.
    Supported if it does and its half-width is ≤ ``max_half_width``; if it does but is
    wider, the cell is rerun once with ``rerun_factor`` times the replications and is
    inconclusive if still too wide; an interval that excludes ``value`` is an
    implementation error. Never 'not supported'."""

    kind: Literal["contains"]
    name: Name
    value: float
    max_half_width: PositiveFloat
    rerun_factor: Annotated[int, Field(ge=2)]
    alpha: Alpha


class ShareAtLeast(Strict):
    """A computed (not sampled) share against a threshold: supported if ≥ it, else not."""

    kind: Literal["share_at_least"]
    name: Name
    threshold: float


TestSpec = Annotated[Union[BoundTest, EquivalenceTest, RejectsNullTest, ImplementationCheck, ContainsTest,
                           ShareAtLeast], Field(discriminator="kind")]
EXACT_KINDS = ("implementation_check", "contains")  # tests of I-type results


class Floor(Strict):
    field: Name
    min: float


class Group(Strict):
    name: Name
    role: Literal["primary", "secondary"] = "primary"
    aggregation: Literal["two_thirds", "all", "single"]
    multiplicity: Literal["holm", "bonferroni", "none"] = "holm"  # bonferroni: contains tests only
    floor: Floor | None = None
    missingness: bool = False  # E1 cell rules (A12 §3.4): missing_rate, sensitivity
    differential_missingness: bool = False
    labels: dict[Label, Text] | None = None  # display names, e.g. aligned / not_aligned
    tests: list[TestSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _names(self) -> Group:
        names = [t.name for t in self.tests]
        if len(set(names)) != len(names):
            raise ValueError("group %s: duplicate test names" % self.name)
        if self.differential_missingness and not self.missingness:
            raise ValueError("group %s: differential_missingness needs missingness" % self.name)
        contains = [t.kind == "contains" for t in self.tests]
        if any(contains) and (not all(contains) or self.multiplicity != "bonferroni"):
            raise ValueError("group %s: contains tests need their own group with multiplicity bonferroni" % self.name)
        if self.multiplicity == "bonferroni" and not any(contains):
            raise ValueError("group %s: multiplicity bonferroni is for contains tests" % self.name)
        return self


class Decision(Strict):
    groups: list[Group] = Field(min_length=1)

    @model_validator(mode="after")
    def _groups(self) -> Decision:
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("duplicate group names")
        if not any(g.role == "primary" for g in self.groups):
            raise ValueError("a decision needs a primary group")
        return self


# -------------------------------------------------------------------- claims
class Interpretation(Strict):
    supported: Text
    not_supported: Text
    inconclusive: Text


class Claim(Strict):
    id: Literal[CLAIM_IDS]
    kind: Literal["P", "Pr", "I", "I/Pr"]
    sources: list[Text] = Field(min_length=1)
    statement: Text
    amendments: list[Text]
    experiment: Literal["E1", "E2", "E3", "E4", "E5"]
    run_mode: Text
    unit_of_analysis: Text
    metric: Text
    estimator: Text
    test: Text
    alpha: Alpha
    ci_level: Annotated[float, Field(gt=0.5, lt=1.0)]
    multiplicity_family: Text
    tolerance_or_margin: Text
    min_n: Text
    decision_rule: Text
    closed_forms: list[Text] = []
    decision: Decision
    interpretation: Interpretation
    status: Literal["draft", "registered", "amended"]
    registration: Literal["E1", "E5"]

    @model_validator(mode="after")
    def _closed_forms_for_exact_results(self) -> Claim:
        exact = any(t.kind in EXACT_KINDS for g in self.decision.groups for t in g.tests)
        if exact != ("I" in self.kind.split("/")):
            raise ValueError("%s: kind %s, but implementation checks are %s" % (self.id, self.kind,
                                                                                "present" if exact else "absent"))
        if exact and not self.closed_forms:
            raise ValueError("%s: an exact (I-type) result needs its closed forms" % self.id)
        return self


class Source(Strict):
    path: Text
    sha256: Sha256


class RegistrationPlan(Strict):
    venue: Text
    covers: list[Literal[CLAIM_IDS]] = Field(min_length=1)
    document: Text
    when: Text


class Register(Strict):
    register_version: Literal[1]
    status: Literal["draft", "registered"]
    sources: list[Source] = Field(min_length=1)
    registrations: dict[Literal["E1", "E5"], RegistrationPlan]
    conventions: dict[Name, Text | float | dict[Name, Text]]
    shared_definitions: dict[Name, Text | float | list | dict]
    claims: list[Claim]

    @model_validator(mode="after")
    def _every_claim_once(self) -> Register:
        ids = [c.id for c in self.claims]
        if sorted(ids) != sorted(CLAIM_IDS) or len(set(ids)) != len(ids):
            raise ValueError("claims: need each of %s exactly once; got %s" % (", ".join(CLAIM_IDS), ids))
        for plan_id, plan in self.registrations.items():
            for claim_id in plan.covers:
                if self.claim(claim_id).registration != plan_id:
                    raise ValueError("registration %s covers %s, which names registration %s"
                                     % (plan_id, claim_id, self.claim(claim_id).registration))
        missing = sorted(set(CLAIM_IDS) - {c for p in self.registrations.values() for c in p.covers})
        if missing:
            raise ValueError("registrations: no registration covers %s" % missing)
        return self

    def claim(self, claim_id: str) -> Claim:
        return next(c for c in self.claims if c.id == claim_id)


REQUIRED_SHARED = ("ground_truth_utility", "effective_choice", "non_choices", "missingness", "thresholds",
                   "presentation_randomisation", "profit_rule_regularity")


def read_register_data(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REGISTER_PATH).read_text(encoding="utf-8"))


def load_register(path: Path | None = None) -> Register:
    register = Register.model_validate(read_register_data(path))
    absent = [name for name in REQUIRED_SHARED if name not in register.shared_definitions]
    if absent:
        raise ValueError("shared_definitions: missing %s" % absent)
    return register


def register_sha256(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or REGISTER_PATH).read_bytes()).hexdigest()


def register_schema() -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **Register.model_json_schema()}


def claim_section(claim: Claim) -> str:
    """The Markdown section of a claim in the pre-registration documents (kept in step
    with the register by a test: regenerate the section when the register changes)."""
    rows = [("Kind", claim.kind), ("Sources", ", ".join("`%s`" % s for s in claim.sources)),
            ("Experiment and mode", "%s, %s" % (claim.experiment, claim.run_mode)),
            ("Unit of analysis", claim.unit_of_analysis), ("Metric", claim.metric), ("Estimator", claim.estimator),
            ("Test", claim.test), ("alpha / CI level", "%g / %g" % (claim.alpha, claim.ci_level)),
            ("Multiplicity family", claim.multiplicity_family), ("Tolerance or margin", claim.tolerance_or_margin),
            ("Minimum N", claim.min_n), ("Decision rule", claim.decision_rule)]
    rows += [("Closed form", form) for form in claim.closed_forms]
    rows += [("Amendment", amendment) for amendment in claim.amendments]
    rows += [("If supported", claim.interpretation.supported),
             ("If not supported", claim.interpretation.not_supported),
             ("If inconclusive", claim.interpretation.inconclusive)]
    lines = ["### %s" % claim.id, "", claim.statement, "", "| | |", "|---|---|"]
    lines += ["| %s | %s |" % (name, value.replace("|", "\\|")) for name, value in rows]
    return "\n".join(lines) + "\n"
