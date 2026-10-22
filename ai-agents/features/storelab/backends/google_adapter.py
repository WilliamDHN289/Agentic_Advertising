"""Google adapter (SL-09): Gemini ``generateContent`` through the ``google-genai`` SDK
(1.56.0, the ai-agents pin) with one attempt per SDK call (``HttpRetryOptions(attempts=1)``).

Sent: ``model`` (the configured id), the prompt as the contents, ``max_output_tokens``,
``temperature`` and ``top_p`` when set, and ``seed``. Recorded: ``modelVersion``
(returned id), ``responseId`` (the provider request id), ``usageMetadata``
``promptTokenCount`` / ``candidatesTokenCount``. A blocked prompt
(``promptFeedback.blockReason``) or a candidate stopped for safety, prohibited content,
a blocklist or personal data is a refusal.
"""
from __future__ import annotations

from ..core.cache import LLMRequest
from ..core.protocol import Usage
from .provider import Classified, ProviderAdapter, Sent, classify_http

REFUSAL_FINISH = ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY")


def _name(value) -> str | None:
    return None if value is None else getattr(value, "name", str(value))


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
        response = self.client.models.generate_content(
            model=request.requested_model,
            contents=[{"role": "user", "parts": [{"text": m.content}]} for m in request.messages], config=config)
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate is not None and candidate.content and candidate.content.parts \
            else []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        blocked = response.prompt_feedback is not None and response.prompt_feedback.block_reason is not None
        refusal = blocked or (candidate is not None and _name(candidate.finish_reason) in REFUSAL_FINISH)
        usage = response.usage_metadata
        return Sent(text=text if candidate is not None else None, refusal=refusal,
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
