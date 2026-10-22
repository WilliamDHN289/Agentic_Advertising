"""SL-04 keyed random streams (core/rng.py)."""
from __future__ import annotations

import random
import unittest

import numpy as np

from features.storelab.core.rng import py_random, stream


class KeyedStreamTest(unittest.TestCase):
    def test_same_seed_and_keys_give_the_same_draws(self):
        self.assertEqual(stream(7, "design", "task_1", 4).random(5).tolist(),
                         stream(7, "design", "task_1", 4).random(5).tolist())
        self.assertEqual([py_random(7, "agent", "abc").random() for _ in range(3)],
                         [py_random(7, "agent", "abc").random() for _ in range(3)])

    def test_any_change_of_seed_key_order_or_key_type_changes_the_stream(self):
        base = stream(7, "design", "task_1", 4).random(4).tolist()
        for other in (stream(8, "design", "task_1", 4), stream(7, "design", "task_2", 4),
                      stream(7, "task_1", "design", 4), stream(7, "design", "task_1", "4"),
                      stream(7, "design", "task_1"), stream(7, "design", "task_1", 4, 0)):
            self.assertNotEqual(other.random(4).tolist(), base)

    def test_draws_do_not_depend_on_what_other_streams_drew(self):
        first = py_random(1, "ivt", "T0001", "api").gauss(0, 1)
        noise = py_random(1, "ivt", "T0000", "api")
        [noise.random() for _ in range(1000)]
        random.random()
        np.random.random()
        self.assertEqual(py_random(1, "ivt", "T0001", "api").gauss(0, 1), first)

    def test_types(self):
        self.assertIsInstance(stream(1, "x"), np.random.Generator)
        self.assertIsInstance(py_random(1, "x"), random.Random)

    def test_bad_seeds_and_keys_are_rejected(self):
        for seed in (-1, 2 ** 63, 1.5, True, "1"):
            with self.subTest(seed=seed), self.assertRaises((ValueError, TypeError)):
                stream(seed, "x")
        for key in (1.5, None, ("a",), True):
            with self.subTest(key=key), self.assertRaises(TypeError):
                py_random(1, key)


if __name__ == "__main__":
    unittest.main()
