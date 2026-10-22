"""``StorelabSettings`` (SL-09): provider keys, the allowed pinned models, breaker
thresholds and the queue.

On the platform this is a section of ``config/settings.py`` (ai-agents); here it is
read from the environment by ``StorelabSettings.from_env``:

| Variable | Field | Default |
|---|---|---|
| ``OPENAI_API_KEY`` | ``openai_api_key`` | none |
| ``ANTHROPIC_API_KEY`` | ``anthropic_api_key`` | none |
| ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` | ``google_api_key`` | none |
| ``STORELAB_ALLOWED_MODELS`` (comma-separated) | ``allowed_models`` | every model in ``pricing.MODEL_PRICES`` |
| ``STORELAB_BREAKER_FAILURE_THRESHOLD`` | ``breaker_failure_threshold`` | 5 |
| ``STORELAB_BREAKER_WINDOW_S`` | ``breaker_window_s`` | 30 |
| ``STORELAB_BREAKER_COOLDOWN_S`` | ``breaker_cooldown_s`` | 20 |
| ``STORELAB_QUEUE`` | ``queue`` | ``storelab`` |

A model may be used in a sandbox run only if it is a pinned snapshot, listed in
``allowed_models`` and priced (``model_problems``).
"""
from __future__ import annotations

import os
from typing import Mapping

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, SecretStr

from .core.config import RunConfig
from .core.model_registry import API_PROVIDERS, is_pinned
from .pricing import MODEL_PRICES

# The Anthropic API version the adapter sends: the version on which the pinned Claude
# snapshots still accept an explicit temperature (2026-10-21 rejects it on every model).
ANTHROPIC_API_VERSION = "2023-06-01"


class StorelabSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    allowed_models: tuple[str, ...] = tuple(sorted(MODEL_PRICES))
    breaker_failure_threshold: PositiveInt = 5
    breaker_window_s: PositiveFloat = 30.0
    breaker_cooldown_s: PositiveFloat = 20.0
    queue: str = "storelab"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> StorelabSettings:
        env = os.environ if environ is None else environ
        values: dict = {"openai_api_key": env.get("OPENAI_API_KEY") or None,
                        "anthropic_api_key": env.get("ANTHROPIC_API_KEY") or None,
                        "google_api_key": env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY") or None}
        if env.get("STORELAB_ALLOWED_MODELS"):
            values["allowed_models"] = tuple(m.strip() for m in env["STORELAB_ALLOWED_MODELS"].split(",") if m.strip())
        for name in ("breaker_failure_threshold", "breaker_window_s", "breaker_cooldown_s", "queue"):
            variable = "STORELAB_" + name.upper()
            if env.get(variable):
                values[name] = env[variable]
        return cls(**values)

    def api_key(self, provider: str) -> str | None:
        secret = getattr(self, "%s_api_key" % provider)
        return secret.get_secret_value() if secret is not None else None


def model_problems(config: RunConfig, settings: StorelabSettings) -> list[str]:
    """Why a configured API model may not be called: unpinned, not allowed or unpriced."""
    problems = []
    for m in config.agents.models:
        if m.provider not in API_PROVIDERS:
            continue
        if not is_pinned(m.provider, m.model):
            problems.append("%s: %s is not a pinned snapshot" % (m.key, m.model))
        if m.model not in settings.allowed_models:
            problems.append("%s: %s is not in StorelabSettings.allowed_models" % (m.key, m.model))
        if m.model not in MODEL_PRICES:
            problems.append("%s: %s has no price row (pricing.py)" % (m.key, m.model))
    return problems
