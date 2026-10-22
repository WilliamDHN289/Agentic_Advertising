"""The ``browser_tool`` interface (SL-10): a decision-maker navigates the storefront
(``core/storefront.py``) through the tools ``open_product``, ``back``, ``select`` and
``leave``, in process, one turn at a time.

**Language models** (``run_tool_loop``). The conversation starts with the storefront's
system prompt and the results page as the user message. Each turn is one call through
``LLMBackend.call`` (cache, provider, guard, invocation log), with the tools of
``TOOL_SCHEMA_VERSION`` (part of the call key) and a forced tool call; the turn's tool
calls run in the storefront and the pages served go back as tool results. The loop
ends when the session ends (``select``, ``leave``, an invalid call, or
``invalid``/``max_turns`` after ``protocol.max_turns`` turns, never a choice) or a turn
ends the trial:

| Turn result | Trial outcome |
|---|---|
| provider error after retries, replay cache miss | ``error`` with that code; the events so far are kept |
| safety refusal | ``refusal``; the events so far are kept |
| kill switch before the turn's first attempt | ``not_run``/``kill_switch``, no events: the trial runs again on resume (earlier turns are cache hits under ``read_write``) |
| ``failure_policy.trial_timeout_s`` passed since the trial started | ``error``/``provider_unavailable`` before the next call, as the runner's trial timeout |

Attempts, tokens and latency are summed over the turns; the returned model id is the
last turn's; ``raw_ref`` names the cache entry of the last turn, whose request holds the
whole conversation. Each tool call is one ``EventRecord`` (``Decision.events``).

**Simulated agents** (``simulated_decision``) act through the same storefront: each
action of ``SimulatedAgent.act`` (χ-driven page requests, then ``select`` or
``leave``) is one turn with one tool call, so their events and outcomes have the same
form as a model's.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.cache import Message
from ..core.protocol import Observation, TrialOutcome, Usage
from ..core.runner import Decision, Trial
from ..core.storefront import SYSTEM_PROMPT, BrowserSession, call_for

if TYPE_CHECKING:
    from .llm import LLMBackend
    from .simulated import SimulatedAgent


def first_messages(observation: Observation) -> list[Message]:
    return [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=observation.views["browser_tool"])]


def _add(total: int | None, value: int | None) -> int | None:
    return total if value is None else (total or 0) + value


def run_tool_loop(backend: LLMBackend, trial: Trial, observation: Observation) -> Decision:
    clock, config = backend.clock, backend.config
    started = clock()
    session = BrowserSession(observation, trial.trial_key, config.protocol.max_turns,
                             elapsed=lambda: clock() - started)
    messages = first_messages(observation)
    totals = {"attempts": 0, "latency_s": 0.0, "input_tokens": None, "output_tokens": None, "returned_model": None,
              "raw_ref": None}

    def count(response) -> None:
        totals["attempts"] += response.attempts
        totals["latency_s"] += response.latency_s or 0.0
        totals["input_tokens"] = _add(totals["input_tokens"], response.usage.input_tokens)
        totals["output_tokens"] = _add(totals["output_tokens"], response.usage.output_tokens)
        totals["returned_model"] = response.returned_model or totals["returned_model"]

    def decision(**fields) -> Decision:
        fields.setdefault("page_requests", list(session.page_requests))
        fields.setdefault("turns", session.turns)
        outcome = TrialOutcome(attempts=totals["attempts"], latency_s=totals["latency_s"],
                               usage=Usage(input_tokens=totals["input_tokens"],
                                           output_tokens=totals["output_tokens"]),
                               returned_model=totals["returned_model"], raw_ref=totals["raw_ref"],
                               **backend.common, **fields)
        return Decision(outcome, list(session.events))

    while not session.done:
        if clock() - started >= config.failure_policy.trial_timeout_s:
            return decision(status="error", error_code="provider_unavailable")
        call = backend.call(trial, backend.request(trial, observation, messages))
        if call.stop is not None:
            if call.stop.status == "not_run":
                return Decision(call.stop)
            if call.response is not None:
                count(call.response)
            totals["raw_ref"] = call.stop.raw_ref or totals["raw_ref"]
            return decision(status="error", error_code=call.stop.error_code)
        response = call.response
        count(response)
        totals["raw_ref"] = call.raw_ref
        if response.refusal:
            return decision(status="refusal")
        calls = response.tool_calls or []
        results = session.step(calls, response.text)
        messages = messages + [Message(role="assistant", content=response.text or "", tool_calls=calls or None)] + [
            Message(role="tool", content=r.content, tool_call_id=r.call_id, name=r.name) for r in results]
    return decision(**session.result())


def simulated_decision(agent: SimulatedAgent, trial: Trial, observation: Observation, max_turns: int) -> Decision:
    """The simulated agent's actions, one turn each, through the storefront."""
    session = BrowserSession(observation, trial.trial_key, max_turns, simulated=True)
    for index, action in enumerate(agent.act(observation, trial.interface, trial.trial_key)):
        session.step([call_for(action, index)])
    outcome = TrialOutcome(attempts=1, backend=agent.backend, requested_model=agent.model,
                           returned_model=agent.model, **session.result())
    return Decision(outcome, session.events)
