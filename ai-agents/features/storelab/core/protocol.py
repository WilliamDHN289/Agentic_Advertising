"""Behaviour protocol (SL-03): what a decision-maker observes, what it may do, and how
a trial ends.

Every choice source (simulated agent, human model, local LLM, provider API, replay)
receives the same ``Observation`` and answers with ``Action`` values, and every trial
ends in one ``TrialOutcome`` whose status says what happened. A failed or unparseable
call is never turned into a choice (tickets.md section 3.1, principle 3).

``parse_reply`` is the strict parser for the text and structured interfaces; its
rules are in docs/protocol.md.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Collection, Literal, Union

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt, TypeAdapter, model_validator

from .config import Interface, NonEmptyStr, Section, canonical_json

PROTOCOL_VERSION = "storelab-protocol@1"
REASON_MAX_WORDS = 20

InvalidReason = Literal["no_json", "schema_violation", "unknown_id", "multiple_choices", "empty", "max_turns"]
# tickets.md section 3.5.
ErrorCode = Literal["rate_limited", "provider_unavailable", "bad_request", "auth", "model_unavailable", "cache_miss"]
NotRunCode = Literal["budget", "kill_switch"]
Status = Literal["choice", "outside", "invalid", "refusal", "error", "not_run"]
Backend = Literal["simulated", "human_model", "local_llm", "openai", "anthropic", "google", "replay"]
ProductId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^\S+$")]


# ------------------------------------------------------------------ observation
class Task(Section):
    task_id: NonEmptyStr
    instruction: NonEmptyStr


class Candidate(Section):
    product_id: ProductId
    position: PositiveInt  # 1-based display position
    sponsored_label: bool


class Observation(Section):
    """One presentation of a candidate set, rendered for each interface."""

    task: Task
    set_id: NonEmptyStr
    presentation_id: NonEmptyStr
    candidates: list[Candidate] = Field(min_length=1)
    views: dict[Interface, str] = {}

    @model_validator(mode="after")
    def _well_formed(self) -> Observation:
        ids = [c.product_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidates: duplicate product ids")
        if sorted(c.position for c in self.candidates) != list(range(1, len(ids) + 1)):
            raise ValueError("candidates: positions must be 1..K, each once")
        return self

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(c.product_id for c in self.candidates)

    def observation_hash(self) -> str:
        """sha256 of the canonical JSON of the observation, views included (S2)."""
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- actions
class RequestPage(Section):
    type: Literal["request_page"] = "request_page"
    product_id: ProductId


class Select(Section):
    """The effective choice: a confirmed selection or add-to-cart (v1.tex l.316)."""

    type: Literal["select"] = "select"
    product_id: ProductId


class Leave(Section):
    """The outside option."""

    type: Literal["leave"] = "leave"


class Invalid(Section):
    type: Literal["invalid"] = "invalid"
    reason: InvalidReason
    detail: str | None = None


Action = Annotated[Union[RequestPage, Select, Leave, Invalid], Field(discriminator="type")]
ACTION = TypeAdapter(Action)


# ---------------------------------------------------------------------- outcome
class Usage(Section):
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None


class TrialOutcome(Section):
    status: Status
    chosen_id: ProductId | None = None
    page_requests: list[ProductId] = []
    turns: NonNegativeInt = 0
    attempts: NonNegativeInt = 0
    error_code: ErrorCode | NotRunCode | None = None
    invalid_reason: InvalidReason | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_s: NonNegativeFloat | None = None
    backend: Backend
    requested_model: str | None = None
    returned_model: str | None = None
    raw_ref: str | None = None

    @model_validator(mode="after")
    def _status_is_explained(self) -> TrialOutcome:
        """Every outcome other than a choice says why there is no choice (S3)."""
        s, problems = self.status, []
        if (s == "choice") != (self.chosen_id is not None):
            problems.append("chosen_id: required for status choice and only for it")
        if (s == "invalid") != (self.invalid_reason is not None):
            problems.append("invalid_reason: required for status invalid and only for it")
        if s == "error" and self.error_code not in ErrorCode.__args__:
            problems.append("error_code: status error needs one of %s" % (ErrorCode.__args__,))
        if s == "not_run" and self.error_code not in NotRunCode.__args__:
            problems.append("error_code: status not_run needs one of %s" % (NotRunCode.__args__,))
        if s not in ("error", "not_run") and self.error_code is not None:
            problems.append("error_code: allowed only for status error or not_run")
        if s == "not_run" and self.attempts:
            problems.append("attempts: a not_run trial made no attempt")
        if problems:
            raise ValueError("; ".join(problems))
        return self


# ----------------------------------------------------------------------- parser
class ChoiceReply(Section):
    """The reply the text and structured interfaces ask for. ``reason`` is at most
    20 words, which JSON Schema cannot express; ``parse_reply`` enforces it."""

    choice: str | None
    reason: str | None = None


class _DuplicateKeys(dict):
    """A JSON object that remembers the keys it saw more than once."""

    def __init__(self, pairs):
        super().__init__(pairs)
        counts: dict[str, list[Any]] = {}
        for key, value in pairs:
            counts.setdefault(key, []).append(value)
        self.repeated = {key: values for key, values in counts.items() if len(values) > 1}


def _reject_constant(name: str):
    raise ValueError("%s is not valid JSON" % name)


# Strict JSON: NaN and Infinity, which Python's decoder accepts by default, are rejected.
_DECODER = json.JSONDecoder(object_pairs_hook=_DuplicateKeys, parse_constant=_reject_constant)


def _objects(text: str) -> list[_DuplicateKeys]:
    """Every top-level JSON object in ``text``, in order; prose and fences are ignored."""
    found, i = [], text.find("{")
    while i != -1:
        try:
            value, end = _DECODER.raw_decode(text, i)
        except (ValueError, RecursionError):  # JSONDecodeError is a ValueError
            i = text.find("{", i + 1)
            continue
        found.append(value)
        i = text.find("{", end)
    return found


def _invalid(reason: str, detail: str) -> Invalid:
    return Invalid(reason=reason, detail=detail)


def parse_reply(text: str | None, allowed_ids: Collection[str]) -> Select | Leave | Invalid:
    """Map a model reply to Select, Leave or Invalid (docs/protocol.md, "Parser").

    The reply must contain exactly one JSON object with a ``choice`` key (repeating
    an identical object is tolerated); ``choice`` is a candidate id or null and
    ``reason`` an optional string of at most 20 words; no other key is allowed. A
    ``Select`` is returned only for an id in ``allowed_ids``, compared exactly after
    stripping surrounding whitespace.
    """
    if text is None or not text.strip():
        return _invalid("empty", "no output")
    objects = _objects(text)
    if not objects:
        return _invalid("no_json", "no JSON object in the reply")
    replies = [o for o in objects if "choice" in o]
    if not replies:
        return _invalid("schema_violation", "no JSON object has a 'choice' key")
    if len({canonical_json(r) for r in replies}) > 1:
        return _invalid("multiple_choices", "%d different JSON replies" % len(replies))
    reply = replies[0]
    if "choice" in reply.repeated and len({canonical_json(v) for v in reply.repeated["choice"]}) > 1:
        return _invalid("multiple_choices", "'choice' given more than once")
    choice = reply["choice"]
    if isinstance(choice, list):
        if len({canonical_json(c) for c in choice}) > 1:
            return _invalid("multiple_choices", "'choice' lists %d values" % len(choice))
        return _invalid("schema_violation", "'choice' must be a string or null, not a list")
    extra = sorted(set(reply) - {"choice", "reason"})
    if extra:
        return _invalid("schema_violation", "unexpected keys %s" % extra)
    reason = reply.get("reason")
    if reason is not None and (not isinstance(reason, str) or len(reason.split()) > REASON_MAX_WORDS):
        return _invalid("schema_violation", "'reason' must be a string of at most %d words" % REASON_MAX_WORDS)
    if choice is None:
        return Leave()
    if not isinstance(choice, str):
        return _invalid("schema_violation", "'choice' must be a string or null")
    product_id = choice.strip()
    if not product_id:
        return _invalid("empty", "'choice' is an empty string")
    if product_id not in allowed_ids:
        return _invalid("unknown_id", "%r is not a candidate" % product_id[:64])
    return Select(product_id=product_id)


def parse_for(observation: Observation, text: str | None) -> Select | Leave | Invalid:
    return parse_reply(text, observation.candidate_ids)
