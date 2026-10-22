"""USD prices of the pinned StoreLab models (SL-09).

These are the rows SL-09 adds to the platform price table (``evals/pricing.py:30``
in ai-agents, "keyed by model ID", USD per one million tokens, input and output billed
separately); this module keeps them, with the same formula, where the StoreLab subtree
can import them. A model id must be pinned and priced to be used in a sandbox run
(``settings.py``). Rates are the providers' published list prices for the A5
placeholder models; replace them with the contracted rates when A5 fixes the list.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core.protocol import Usage


@dataclass(frozen=True)
class ModelPrice:
    """USD per **one million** tokens, input and output billed separately."""

    input_usd_per_1m: float
    output_usd_per_1m: float


MODEL_PRICES: dict[str, ModelPrice] = {
    # --- OpenAI ---
    "gpt-4o-2024-08-06": ModelPrice(input_usd_per_1m=2.50, output_usd_per_1m=10.00),
    "gpt-4.1-2025-04-14": ModelPrice(input_usd_per_1m=2.00, output_usd_per_1m=8.00),
    # --- Anthropic ---
    "claude-sonnet-4-20250514": ModelPrice(input_usd_per_1m=3.00, output_usd_per_1m=15.00),
    "claude-sonnet-4-5-20250929": ModelPrice(input_usd_per_1m=3.00, output_usd_per_1m=15.00),
    # --- Google ---
    "gemini-2.0-flash-001": ModelPrice(input_usd_per_1m=0.10, output_usd_per_1m=0.40),
    "gemini-2.0-flash-lite-001": ModelPrice(input_usd_per_1m=0.075, output_usd_per_1m=0.30),
}


class UnknownModelError(KeyError):
    """A model id with no price row: cost is never guessed."""


def price(model: str) -> ModelPrice:
    try:
        return MODEL_PRICES[model]
    except KeyError:
        raise UnknownModelError("no price row for model %r; add it to features/storelab/pricing.py "
                                "(evals/pricing.py on the platform)" % model) from None


def call_cost_usd(model: str, usage: Usage) -> float:
    """``input_tokens / 1e6 * input rate + output_tokens / 1e6 * output rate``; missing counts are 0."""
    rate = price(model)
    return ((usage.input_tokens or 0) / 1_000_000 * rate.input_usd_per_1m
            + (usage.output_tokens or 0) / 1_000_000 * rate.output_usd_per_1m)
