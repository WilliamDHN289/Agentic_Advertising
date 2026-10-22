"""Random streams derived by key, never by call order (SL-04; tickets.md section 3.6).

A stream depends only on the registered seed and its keys, for example
``stream(seed, "design", task_id, k)`` or ``py_random(seed, "legacy", "auction",
task_id, arm)``. Running episodes in another order, or resuming after a crash,
therefore cannot change what any of them draws; this is what fixes the shared
generators of the original StoreLab (SL-00 defect D-01).

Keys are hashed with their types (``1`` and ``"1"`` are different keys).
"""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np


def _digest(seed: int, keys: tuple) -> bytes:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an int in [0, 2**63)")
    for key in keys:
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise TypeError("stream keys must be str or int, got %r" % (key,))
    material = json.dumps([seed, *keys], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).digest()


def stream(seed: int, *keys: str | int) -> np.random.Generator:
    """A numpy Generator from SeedSequence(sha256(seed, keys))."""
    entropy = int.from_bytes(_digest(seed, keys), "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def py_random(seed: int, *keys: str | int) -> random.Random:
    """A ``random.Random`` seeded from sha256(seed, keys), for ported code that uses it."""
    return random.Random(int.from_bytes(_digest(seed, keys), "big"))
