"""Google adapter (SL-09): Gemini ``generateContent`` through the ``google-genai`` SDK
(1.56.0, the ai-agents pin) with one attempt per SDK call (``HttpRetryOptions(attempts=1)``).

Sent: ``model`` (the configured id), the prompt as the contents, ``max_output_tokens``,
``temperature`` and ``top_p`` when set, and ``seed``. Recorded: ``modelVersion``
(returned id), ``responseId`` (the provider request id), ``usageMetadata``
``promptTokenCount`` / ``candidatesTokenCount``. A blocked prompt
(``promptFeedback.blockReason``) or a candidate stopped for safety, prohibited content,
a blocklist or personal data is a refusal.

Tools (``browser_tool``, SL-10): system messages go in ``system_instruction``; the tools
as ``function_declarations`` (no ``parameters`` for a tool without arguments) with
``function_calling_config`` mode ``ANY`` and automatic function calling off; earlier
turns go back as ``function_call`` parts and one user turn of ``function_response``
parts. Gemini may return several calls in one response (it has no switch for parallel
calls); calls without an id get ``call-<index>``.
"""
from __future__ import annotations

from ..core.cache import LLMRequest, Message
from ..core.protocol import Usage
from ..core.storefront import ToolCall
from .provider import Classified, ProviderAdapter, Sent, classify_http, tools_of

REFUSAL_FINISH = ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY")


def _name(value) -> str | None:
    return None if value is None else getattr(value, "name", str(value))


def _contents(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            part = {"function_response": {"name": m.name, "response": {"content": m.content}}}
            if out and out[-1]["role"] == "user" and "function_response" in out[-1]["parts"][-1]:
                out[-1]["parts"].append(part)  # the results of one turn form one user turn
            else:
                out.append({"role": "user", "parts": [part]})
        elif m.role == "assistant":
            out.append({"role": "model", "parts": ([{"text": m.content}] if m.content else []) + [
                {"function_call": {"name": c.name, "args": c.arguments if isinstance(c.arguments, dict) else {}}}
                for c in m.tool_calls or []]})
        else:
            out.append({"role": "user", "parts": [{"text": m.content}]})
    return out


def _declaration(tool: dict) -> dict:
    declaration = {"name": tool["name"], "description": tool["description"]}
    if tool["parameters"]["properties"]:
        declaration["parameters"] = tool["parameters"]
    return declaration


class GoogleAdapter(ProviderAdapter):
    name = "google"
    adapter_version = "gemini-generate-content@1"

    def __init__(self, client, policy, guard=None, **kwargs):
        super().__init__(policy, guard, **kwargs)
        self.client = client

    @classmethod
    def from_key(cls, api_key: str, policy, guard=None, *, http_client=None, **kwargs) -> GoogleAdapter:
        from google import genai
        from google.genai import types

        options = types.HttpOptions(timeout=int(policy.trial_timeout_s * 1000),
                                    retry_options=types.HttpRetryOptions(attempts=1), httpx_client=http_client)
        return cls(genai.Client(api_key=api_key, http_options=options), policy, guard, **kwargs)

    def send(self, request: LLMRequest) -> Sent:
        params = request.params
        config = {"max_output_tokens": params["max_output_tokens"]}
        for name in ("temperature", "top_p", "seed"):
            if params.get(name) is not None:
                config[name] = params[name]
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        if system:
            config["system_instruction"] = system
        tools = tools_of(request)
        if tools:
            config.update(tools=[{"function_declarations": [_declaration(t) for t in tools]}],
                          tool_config={"function_calling_config": {"mode": "ANY"}},
                          automatic_function_calling={"disable": True})
        response = self.client.models.generate_content(
            model=request.requested_model, contents=_contents([m for m in request.messages if m.role != "system"]),
            config=config)
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate is not None and candidate.content and candidate.content.parts \
            else []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [ToolCall(id=p.function_call.id or "call-%d" % i, name=p.function_call.name or "",
                          arguments=dict(p.function_call.args or {}))
                 for i, p in enumerate(p for p in parts if p.function_call is not None)]
        blocked = response.prompt_feedback is not None and response.prompt_feedback.block_reason is not None
        refusal = blocked or (candidate is not None and _name(candidate.finish_reason) in REFUSAL_FINISH)
        usage = response.usage_metadata
        return Sent(text=text if candidate is not None else None, refusal=refusal, tool_calls=calls or None,
                    returned_model=response.model_version,
                    usage=Usage(input_tokens=usage.prompt_token_count if usage else None,
                                output_tokens=usage.candidates_token_count if usage else None),
                    raw=response.model_dump(mode="json", exclude={"sdk_http_response"}),
                    request_id=response.response_id)

    def classify(self, error: BaseException) -> Classified | None:
        status = None
        if any(cls.__name__ == "APIError" for cls in type(error).__mro__) and isinstance(error.code, int):
            status = error.code
        response = getattr(error, "response", None)
        return classify_http(error, status, getattr(response, "headers", None))
