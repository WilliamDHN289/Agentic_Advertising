"""OpenAI adapter (SL-09): Chat Completions through the ``openai`` SDK (2.8.0, the
ai-agents pin) with ``max_retries=0``.

Sent: ``model`` (the configured id, never another), the prompt as one user message,
``max_completion_tokens``, ``temperature`` and ``top_p`` when set, and ``seed`` when the
model takes one. Recorded: ``model`` (returned id), ``system_fingerprint``, the
``x-request-id`` header, ``usage.prompt_tokens`` / ``completion_tokens``. A
``message.refusal`` or ``finish_reason == "content_filter"`` is a refusal.
"""
from __future__ import annotations

from ..core.cache import LLMRequest
from ..core.protocol import Usage
from .provider import Classified, ProviderAdapter, Sent, classify_http


class OpenAIAdapter(ProviderAdapter):
    name = "openai"
    adapter_version = "openai-chat@1"

    def __init__(self, client, policy, guard=None, **kwargs):
        super().__init__(policy, guard, **kwargs)
        self.client = client

    @classmethod
    def from_key(cls, api_key: str, policy, guard=None, *, http_client=None, **kwargs) -> OpenAIAdapter:
        import openai

        client = openai.OpenAI(api_key=api_key, max_retries=0, timeout=policy.trial_timeout_s,
                               http_client=http_client)
        return cls(client, policy, guard, **kwargs)

    def send(self, request: LLMRequest) -> Sent:
        params = request.params
        kwargs = {"model": request.requested_model,
                  "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                  "max_completion_tokens": params["max_output_tokens"]}
        for name in ("temperature", "top_p", "seed"):
            if params.get(name) is not None:
                kwargs[name] = params[name]
        raw = self.client.chat.completions.with_raw_response.create(**kwargs)
        completion = raw.parse()
        choice = completion.choices[0] if completion.choices else None
        message = choice.message if choice is not None else None
        refusal = bool(message is not None and message.refusal) or (
            choice is not None and choice.finish_reason == "content_filter")
        usage = completion.usage
        return Sent(text=message.content if message is not None else None, refusal=refusal,
                    returned_model=completion.model,
                    usage=Usage(input_tokens=usage.prompt_tokens if usage else None,
                                output_tokens=usage.completion_tokens if usage else None),
                    raw=completion.model_dump(mode="json"), request_id=raw.headers.get("x-request-id"),
                    system_fingerprint=completion.system_fingerprint)

    def classify(self, error: BaseException) -> Classified | None:
        response = getattr(error, "response", None)
        return classify_http(error, getattr(error, "status_code", None),
                             getattr(response, "headers", None))
