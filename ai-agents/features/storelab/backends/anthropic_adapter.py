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
"""
from __future__ import annotations

from ..core.cache import LLMRequest
from ..core.protocol import Usage
from ..settings import ANTHROPIC_API_VERSION
from .provider import Classified, ProviderAdapter, Sent, classify_http


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
        raw = self.client.messages.with_raw_response.create(
            model=request.requested_model, max_tokens=params["max_output_tokens"],
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            extra_body=sampling or None)
        message = raw.parse()
        text = "".join(block.text for block in message.content if block.type == "text")
        return Sent(text=text, refusal=message.stop_reason == "refusal", returned_model=message.model,
                    usage=Usage(input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens),
                    raw=message.model_dump(mode="json"), request_id=raw.headers.get("request-id"))

    def classify(self, error: BaseException) -> Classified | None:
        response = getattr(error, "response", None)
        return classify_http(error, getattr(error, "status_code", None),
                             getattr(response, "headers", None))
