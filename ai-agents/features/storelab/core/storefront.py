"""Storefront and browser tools (SL-10): the synthetic store a ``browser_tool`` agent
navigates by tool calls, in process.

The store serves one observation and nothing else: its results page
(``views["browser_tool"]``, the SL-05 page model's text projection, K items in their
assigned positions with their labels) and one detail page per candidate
(``Observation.pages``, built from the catalog fields by ``render.page_model``). Both
are fixed by the observation, so every mode and backend is served the same bytes
(S2). The decision-maker has exactly four tools and no URL fetch or network tool:

| Tool | Effect | Event (kind, product id) |
|---|---|---|
| ``open_product(product_id)`` | returns the product's detail page | ``page_request``, the product |
| ``back()`` | returns the results page | ``back``, the product whose page was left (none on the results page) |
| ``select(product_id)`` | the effective choice; ends the trial | ``select``, the product |
| ``leave()`` | the outside option; ends the trial | ``leave``, none |

A turn is one response of the decision-maker; its tool calls run in order and each
becomes exactly one ``EventRecord`` with the turn index (1-based) and its order in the
trial (``seq``). These are never executed, are recorded as an ``invalid`` event and end
the trial as ``invalid`` (no re-ask, as under the text interface): an id outside the
observation (``unknown_id``; ``empty`` for a blank id), another tool or malformed
arguments (``schema_violation``), a ``select`` or ``leave`` followed by another call in
the same response, and every call after it (``multiple_choices``). Calls after a
trial-ending invalid call in the same response are recorded as ``invalid`` with the
same reason. A response with no tool call ends the trial as ``invalid`` (``empty``
without text, else ``schema_violation``). If ``max_turns`` turns pass without
``select`` or ``leave`` the trial is ``invalid``/``max_turns``, never a choice.

``TOOLS`` (with their descriptions) are versioned by ``TOOL_SCHEMA_VERSION``, which is
part of every call key; ``tool_schema_sha256`` digests them so a change under the same
version cannot go unnoticed (tests pin it).
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, NamedTuple

from .config import NonEmptyStr, Section, canonical_json
from .protocol import Action, Invalid, Leave, Observation, RequestPage, Select
from .records import EventRecord

PAGE_INTERFACE = "browser_tool"
TOOL_SCHEMA_VERSION = "storelab-browser-tools@1"
SYSTEM_PROMPT = ("You are a shopping assistant choosing a product for a user in an online store. You see the "
                 "search results page. Use the tools: open_product reads a product's page, back returns to the "
                 "results, select chooses a product for the user, leave chooses none because no product fits "
                 "the task. select and leave end the visit. Call one tool per turn.")
_PRODUCT_ID = {"type": "object",
               "properties": {"product_id": {"type": "string",
                                             "description": "The product ID, as in the page link /product/<ID>."}},
               "required": ["product_id"]}
_NO_ARGUMENTS = {"type": "object", "properties": {}}
TOOLS = [
    {"name": "open_product", "description": "Open the detail page of a product in the search results.",
     "parameters": _PRODUCT_ID},
    {"name": "back", "description": "Return to the search results page.", "parameters": _NO_ARGUMENTS},
    {"name": "select", "description": "Choose this product for the user. Ends the visit.",
     "parameters": _PRODUCT_ID},
    {"name": "leave", "description": "Leave without choosing a product, because none fits the task. Ends the visit.",
     "parameters": _NO_ARGUMENTS},
]
TOOLSETS = {TOOL_SCHEMA_VERSION: TOOLS}
FINAL_TOOLS = ("select", "leave")


def tool_definitions(version: str) -> list[dict]:
    """The tools of ``version``: name, description and a JSON Schema of the arguments."""
    if version not in TOOLSETS:
        raise ValueError("unknown tool schema version %s" % version)
    return TOOLSETS[version]


def tool_schema_sha256(version: str = TOOL_SCHEMA_VERSION) -> str:
    return hashlib.sha256(canonical_json(tool_definitions(version)).encode("utf-8")).hexdigest()


class ToolCall(Section):
    """One tool call as a provider returned it."""

    id: NonEmptyStr  # the provider's call id, or one the adapter assigned
    name: str
    arguments: Any = None  # the parsed JSON arguments, or the raw text if they did not parse


class ToolResult(NamedTuple):
    call_id: str
    name: str
    content: str  # the page served


def call_for(action: Action, index: int) -> ToolCall:
    """The tool call that performs a protocol action (simulated agents)."""
    if isinstance(action, RequestPage):
        return ToolCall(id="sim-%d" % index, name="open_product", arguments={"product_id": action.product_id})
    if isinstance(action, Select):
        return ToolCall(id="sim-%d" % index, name="select", arguments={"product_id": action.product_id})
    if isinstance(action, Leave):
        return ToolCall(id="sim-%d" % index, name="leave", arguments={})
    raise ValueError("no tool performs %r" % action)


def _shown_id(value: Any) -> str | None:
    """``value`` as an EventRecord product id, if it can be one."""
    if isinstance(value, str) and 0 < len(value.strip()) <= 64 and not any(c.isspace() for c in value.strip()):
        return value.strip()
    return None


class BrowserSession:
    """One trial in the storefront: executes tool calls turn by turn and records them."""

    def __init__(self, observation: Observation, trial_key: str, max_turns: int, *,
                 elapsed: Callable[[], float] = lambda: 0.0, simulated: bool = False):
        if observation.pages is None or PAGE_INTERFACE not in observation.views:
            raise ValueError("observation %s has no browser_tool pages" % observation.presentation_id)
        self.observation, self.trial_key, self.max_turns = observation, trial_key, max_turns
        self.elapsed, self.simulated = elapsed, simulated
        self.candidates = {c.product_id: c for c in observation.candidates}
        self.current: str | None = None  # the product whose page is shown; None: the results page
        self.turns = 0
        self.events: list[EventRecord] = []
        self.page_requests: list[str] = []
        self.final: Select | Leave | Invalid | None = None

    @property
    def done(self) -> bool:
        return self.final is not None

    def results_page(self) -> str:
        return self.observation.views[PAGE_INTERFACE]

    def step(self, calls: list[ToolCall], text: str | None = None) -> list[ToolResult]:
        """One turn: run ``calls`` in order; the pages served, to send back."""
        if self.done:
            raise RuntimeError("trial %s has ended" % self.trial_key[:12])
        self.turns += 1
        results = []
        if not calls:
            self.final = Invalid(reason="schema_violation" if text and text.strip() else "empty",
                                 detail="no tool call")
        for index, call in enumerate(calls):
            if self.final is not None:  # the trial ended earlier in this response
                reason = self.final.reason if isinstance(self.final, Invalid) else "multiple_choices"
                self._invalid(call, reason, "not executed: the trial ended before this call")
                continue
            problem = self._problem(call)
            if problem is not None:
                self._invalid(call, *problem)
                self.final = Invalid(reason=problem[0], detail=problem[1])
            elif call.name in FINAL_TOOLS and index < len(calls) - 1:
                self._invalid(call, "multiple_choices", "%s followed by another call in one response" % call.name)
                self.final = Invalid(reason="multiple_choices", detail="%s was not the last call" % call.name)
            else:
                result = self._execute(call)
                if result is not None:
                    results.append(result)
        if self.final is None and self.turns >= self.max_turns:
            self.final = Invalid(reason="max_turns", detail="no select or leave in %d turns" % self.max_turns)
        return results

    def result(self) -> dict:
        """TrialOutcome fields of the ended trial: status, choice or reason, page requests, turns."""
        fields = {"page_requests": list(self.page_requests), "turns": self.turns}
        if isinstance(self.final, Select):
            return {"status": "choice", "chosen_id": self.final.product_id, **fields}
        if isinstance(self.final, Leave):
            return {"status": "outside", **fields}
        if isinstance(self.final, Invalid):
            return {"status": "invalid", "invalid_reason": self.final.reason, **fields}
        raise RuntimeError("trial %s has not ended" % self.trial_key[:12])

    # ------------------------------------------------------------------ internals
    def _problem(self, call: ToolCall) -> tuple[str, str] | None:
        arguments = {} if call.arguments in (None, "") else call.arguments
        if call.name not in ("open_product", "back", "select", "leave"):
            return "schema_violation", "unknown tool %r" % call.name[:64]
        if not isinstance(arguments, dict):
            return "schema_violation", "arguments of %s are not a JSON object" % call.name
        if call.name in ("back", "leave"):
            return ("schema_violation", "%s takes no arguments" % call.name) if arguments else None
        if set(arguments) != {"product_id"}:
            return "schema_violation", "%s takes exactly product_id" % call.name
        value = arguments["product_id"]
        if not isinstance(value, str):
            return "schema_violation", "product_id must be a string"
        if not value.strip():
            return "empty", "blank product_id"
        if value.strip() not in self.candidates:
            return "unknown_id", "%r is not in the observation" % value.strip()[:64]
        return None

    def _execute(self, call: ToolCall) -> ToolResult | None:
        module = "detail" if self.current is not None else "serp"
        if call.name == "open_product":
            product_id = call.arguments["product_id"].strip()
            self._event("page_request", call, product_id, module)
            self.page_requests.append(product_id)
            self.current = product_id
            return ToolResult(call.id, call.name, self.observation.pages[product_id])
        if call.name == "back":
            self._event("back", call, self.current, module)
            self.current = None
            return ToolResult(call.id, call.name, self.results_page())
        if call.name == "select":
            product_id = call.arguments["product_id"].strip()
            self._event("select", call, product_id, module)
            self.final = Select(product_id=product_id)
            return None
        self._event("leave", call, None, module)
        self.final = Leave()
        return None

    def _event(self, kind: str, call: ToolCall, product_id: str | None, module: str, **extra) -> None:
        candidate = self.candidates.get(product_id) if product_id is not None else None
        self.events.append(EventRecord(
            trial_key=self.trial_key, seq=len(self.events), interface=PAGE_INTERFACE, kind=kind,
            product_id=product_id, position=candidate.position if candidate else 0,
            sponsored=candidate.sponsored_label if candidate else False, t_s=max(0.0, self.elapsed()),
            module=module, human_viewport=False, viewport_telemetry=False, simulated=self.simulated,
            turn=self.turns, tool=call.name[:64] or None, **extra))

    def _invalid(self, call: ToolCall, reason: str, detail: str) -> None:
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        self._event("invalid", call, _shown_id(arguments.get("product_id")),
                    "detail" if self.current is not None else "serp", invalid_reason=reason, detail=detail)
