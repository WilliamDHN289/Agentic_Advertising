"""Token counts for the cost estimate (SL-11).

OpenAI models are counted with tiktoken (0.14.0, the ai-agents pin) in the encoding
``tiktoken.model.encoding_name_for_model`` names for them, ``o200k_base`` for every
pinned OpenAI model. The encoding file is read from the artifact store
(``artifacts/tokenizers/o200k_base/<sha256>/``, the file tiktoken downloads, with the
sha256 tiktoken expects), never from the network, so plans work offline and in CI.
Other providers publish no offline tokenizer: their counts are characters ÷ 3.5 ×
1.2, rounded up (``core.cost.heuristic_tokens``). An OpenAI model in another
encoding is refused rather than counted approximately.
"""
from __future__ import annotations

import base64
import functools
import hashlib
from pathlib import Path

from .core.cost import HEURISTIC, CostError, heuristic_tokens
from .legacy_baseline import default_artifact_root

ENCODING = "o200k_base"
ENCODING_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
# tiktoken 0.14.0, tiktoken_ext/openai_public.py, o200k_base() l.100-114.
SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}
PAT_STR = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


def encoding_path(root: Path | None = None) -> Path:
    return Path(root or default_artifact_root()) / "tokenizers" / ENCODING / ENCODING_SHA256 / (ENCODING + ".tiktoken")


@functools.lru_cache(maxsize=2)
def encoding(path: Path | None = None):
    """The o200k_base ``tiktoken.Encoding`` from the pinned file (hash checked)."""
    try:
        import tiktoken
    except ImportError:
        raise CostError("tiktoken is required to count OpenAI tokens (requirements-storelab.txt)") from None
    path = Path(path or encoding_path())
    try:
        data = path.read_bytes()
    except OSError as error:
        raise CostError("tokenizer file %s: %s" % (path, error)) from None
    if hashlib.sha256(data).hexdigest() != ENCODING_SHA256:
        raise CostError("tokenizer file %s does not have sha256 %s" % (path, ENCODING_SHA256))
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in data.splitlines() if line)}
    return tiktoken.Encoding(name=ENCODING, pat_str=PAT_STR, mergeable_ranks=ranks, special_tokens=SPECIAL_TOKENS)


def count_tokens(provider: str, model: str, text: str) -> int:
    if provider != "openai":
        return heuristic_tokens(text)
    import tiktoken.model

    try:
        name = tiktoken.model.encoding_name_for_model(model)
    except KeyError:
        name = None
    if name != ENCODING:
        raise CostError("model %s: tiktoken encoding %s, but only %s is pinned (tokens.py)" % (model, name, ENCODING))
    return len(encoding().encode(text, disallowed_special=()))


def token_counting() -> dict[str, str]:
    """How each provider's tokens are counted, for the estimate report."""
    return {"openai": "tiktoken %s (sha256 %s)" % (ENCODING, ENCODING_SHA256[:12]), "anthropic": HEURISTIC,
            "google": HEURISTIC}
