"""What each provider's models accept, for config validation (SL-02).

The rules encode provider API behaviour as documented when this was written. They
must be re-checked when A5 fixes the pinned model list and the adapters are built
(SL-09); a model no rule matches is rejected rather than guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

API_PROVIDERS = ("openai", "anthropic", "google")
PROVIDERS = API_PROVIDERS + ("ollama",)


@dataclass(frozen=True)
class ModelRule:
    provider: str
    pattern: str
    # Accepted temperature range (low, high); high None means unbounded.
    # None means the API rejects any explicit temperature.
    temperature: tuple[float, float | None] | None
    seed_supported: bool


# First match wins.
RULES = (
    # Reasoning models accept only the default temperature.
    ModelRule("openai", r"^(o\d|gpt-5)", None, False),
    ModelRule("openai", r"^gpt-", (0.0, 2.0), True),
    ModelRule("anthropic", r"^claude-", (0.0, 1.0), False),
    ModelRule("google", r"^gemini-", (0.0, 2.0), True),
    # Local models served by ollama (the original StoreLab path).
    ModelRule("ollama", r".", (0.0, None), True),
)

# A pinned snapshot carries a dated or numbered version suffix; any other id is an
# alias the provider can repoint (for example gpt-4o, claude-sonnet-4-5,
# claude-3-5-sonnet-latest, gemini-2.0-flash).
PINNED_SUFFIX = {
    "openai": r"-\d{4}-\d{2}-\d{2}$",
    "anthropic": r"-\d{8}$",
    "google": r"-(\d{3}|preview-\d{2}-\d{2})$",
}


def rule_for(provider: str, model: str) -> ModelRule | None:
    return next((r for r in RULES if r.provider == provider and re.search(r.pattern, model)), None)


def is_pinned(provider: str, model: str) -> bool:
    """True for a dated or numbered snapshot of an API provider's model."""
    suffix = PINNED_SUFFIX.get(provider)
    return bool(suffix and re.search(suffix, model))
