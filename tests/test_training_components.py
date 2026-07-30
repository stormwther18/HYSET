from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_hyset import NegativeSampler, QueryExample, _mixture_quotas


class FakeNeighbors:
    def get(self, tool_id: int):
        return tuple((tool_id + offset) % 50 for offset in range(1, 12))


class TrainingComponentsTest(unittest.TestCase):
    def test_fixed_mixture_allocates_all_negatives(self) -> None:
        quotas = _mixture_quotas(63)
        self.assertEqual((31, 19, 13), quotas)
        self.assertEqual(63, sum(quotas))

    def test_candidate_pool_is_distinct_and_keeps_positive_first(self) -> None:
        example = QueryExample("q0", "query", (0, 1))
        batch = [
            example,
            QueryExample("q1", "other", (2, 3)),
            QueryExample("q2", "other", (4, 5, 6)),
            QueryExample("q3", "other", (7,)),
        ]
        sampler = NegativeSampler(
            num_tools=50,
            k_neg=16,
            rng=random.Random(42),
            neighbors=FakeNeighbors(),
        )
        pool = sampler.build_pool(example, batch)
        self.assertEqual(example.ground_truth, pool[0])
        self.assertEqual(16, len(pool))
        self.assertEqual(16, len(set(pool)))


if __name__ == "__main__":
    unittest.main()
