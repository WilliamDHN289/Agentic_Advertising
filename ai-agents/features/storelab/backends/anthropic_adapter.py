"""Anthropic adapter (SL-09): the Messages API through the ``anthropic`` SDK (1.17.0)
with ``max_retries=0``.

The adapter pins API version 2023-06-01 (``settings.ANTHROPIC_API_VERSION``), the
version on which the pinned Claude snapshots accept an explicit temperature; version
2026-10-21 (SDK 2.x) rejects ``temperature`` and ``top_p`` on every model. SDK 1.x no
longer types these two fields, so they are sent in the request body (``extra_body``).

Sent: ``model`` (the configured id), the prompt as one user message, ``max_tokens``,
``temperature`` and ``top_p`` when set; Anthropic takes no seed. Recorded: ``model``
(returned id), the ``request-id`` header, ``usage.input_tokens`` / ``output_tokens``.
``stop_reason == "refusal"`` is a refusal.

Tools (``browser_tool``, SL-10): system messages go in ``system``; ``tools`` with
``input_schema`` and ``tool_choice: {type: any, disable_parallel_tool_use: true}``;
earlier turns go back as ``tool_use`` blocks and one user message of ``tool_result``
blocks; the response's ``tool_use`` blocks become ``ToolCall`` values.
"""
from __future__ import annotations

from ..core.cache import LLMRequest, Message
from ..core.protocol import Usage
from ..core.storefront import ToolCall
from ..settings import ANTHROPIC_API_VERSION
from .provider import Classified, ProviderAdapter, Sent, classify_http, tools_of


def _messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)  # the results of one turn form one user message
            else:
                out.append({"role": "user", "content": [block]})
        elif m.tool_calls:
            out.append({"role": "assistant", "content": ([{"type": "text", "text": m.content}] if m.content else []) + [
                {"type": "tool_use", "id": c.id, "name": c.name,
                 "input": c.arguments if isinstance(c.arguments, dict) else {}} for c in m.tool_calls]})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    adapter_version = "anthropic-messages@1"
    api_version = ANTHROPIC_API_VERSION

    def __init__(self, client, policy, guard=None, **kwargs):
        super().__init__(policy, guard, **kwargs)
        self.client = client

    @classmethod
    def from_key(cls, api_key: str, policy, guard=None, *, http_client=None, **kwargs) -> AnthropicAdapter:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=policy.trial_timeout_s,
                                     default_headers={"anthropic-version": ANTHROPIC_API_VERSION},
                                     http_client=http_client)
        return cls(client, policy, guard, **kwargs)

    def send(self, request: LLMRequest) -> Sent:
        params = request.params
        sampling = {name: params[name] for name in ("temperature", "top_p") if params.get(name) is not None}
        kwargs = {}
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        if system:
            kwargs["system"] = system
        tools = tools_of(request)
        if tools:
            kwargs.update(tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                                 for t in tools], tool_choice={"type": "any", "disable_parallel_tool_use": True})
        raw = self.client.messages.with_raw_response.create(
            model=request.requested_model, max_tokens=params["max_output_tokens"],
            messages=_messages([m for m in request.messages if m.role != "system"]),
            extra_body=sampling or None, **kwargs)
        message = raw.parse()
        text = "".join(block.text for block in message.content if block.type == "text")
        calls = [ToolCall(id=block.id, name=block.name, arguments=block.input)
                 for block in message.content if block.type == "tool_use"]
        return Sent(text=text, refusal=message.stop_reason == "refusal", returned_model=message.model,
                    tool_calls=calls or None,
                    usage=Usage(input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens),
                    raw=message.model_dump(mode="json"), request_id=raw.headers.get("request-id"))

    def classify(self, error: BaseException) -> Classified | None:
        response = getattr(error, "response", None)
        return classify_http(error, getattr(error, "status_code", None),
                             getattr(response, "headers", None))
