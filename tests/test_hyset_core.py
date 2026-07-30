from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyset_model import HYSETScorer


class HYSETScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = HYSETScorer(num_tools=5, d_z=2, d_r=2, m_max=3)
        with torch.no_grad():
            self.scorer.Z.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [math.sqrt(0.5), math.sqrt(0.5)],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )
            self.scorer.P.weight.copy_(torch.eye(2))
            self.scorer.interactions["2"].copy_(torch.tensor([[1.0, 0.4], [0.4, 2.0]]))
            self.scorer.interactions["3"].copy_(torch.tensor([[0.5, -0.2], [-0.2, 1.5]]))

    def test_score_matches_equations_2_to_6(self) -> None:
        query = torch.tensor([1.0, 2.0])
        candidate = torch.tensor([[0, 1]])
        score = self.scorer.score_sets(query, candidate)[0]

        logits = torch.tensor([1.0, 2.0])
        expected_align = (torch.softmax(logits, dim=0) * logits).sum()
        expected_set = torch.tensor(0.4)
        self.assertTrue(torch.allclose(score, expected_align + expected_set, atol=1e-6))

    def test_set_score_is_permutation_invariant(self) -> None:
        query = torch.tensor([0.3, -0.2])
        candidates = torch.tensor([[0, 1, 2], [2, 0, 1]])
        scores = self.scorer.score_sets(query, candidates)
        self.assertTrue(torch.allclose(scores[0], scores[1], atol=1e-6))

    def test_cardinality_specific_interactions_change_pair_contribution(self) -> None:
        z0 = self.scorer.Z.weight[0]
        z1 = self.scorer.Z.weight[1]
        pair_at_two = z0 @ self.scorer.interaction_matrix(2) @ z1
        pair_at_three = z0 @ self.scorer.interaction_matrix(3) @ z1
        self.assertFalse(torch.allclose(pair_at_two, pair_at_three))

    def test_shortlist_adds_complementary_low_relevance_tool(self) -> None:
        with torch.no_grad():
            self.scorer.interactions["3"].copy_(torch.tensor([[-2.0, 4.0], [4.0, -2.0]]))
        query = torch.tensor([1.0, 0.2])
        top_two = torch.topk(self.scorer.singleton_scores(query), k=2).indices.tolist()
        shortlist = self.scorer.build_shortlist(query, k1=2, k_pool=3)
        self.assertEqual(tuple(top_two), shortlist[:2])
        self.assertNotIn(shortlist[2], top_two)

    def test_prediction_exhaustively_compares_cardinalities(self) -> None:
        query = torch.tensor([0.4, 0.8])
        shortlist = (0, 1, 2, 3)
        predicted, predicted_score = self.scorer.predict_set(query, shortlist, batch_size=2)

        all_candidates = []
        for cardinality in range(1, 4):
            from itertools import combinations

            all_candidates.extend(combinations(shortlist, cardinality))
        all_scores = self.scorer.score_candidate_sets(query, all_candidates)
        best = int(all_scores.argmax().item())
        self.assertEqual(tuple(all_candidates[best]), predicted)
        self.assertAlmostEqual(float(all_scores[best].detach()), predicted_score, places=6)

    def test_constraints_are_explicit(self) -> None:
        with torch.no_grad():
            self.scorer.Z.weight.mul_(3.0)
            self.scorer.interactions["2"][0, 1] = 2.0
            self.scorer.interactions["2"][1, 0] = -1.0
        self.scorer.project_constraints_()
        norms = self.scorer.Z.weight.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        matrix = self.scorer.interactions["2"]
        self.assertTrue(torch.allclose(matrix, matrix.T, atol=1e-6))

    def test_only_paper_parameters_receive_core_gradients(self) -> None:
        query = torch.tensor([0.25, 0.75])
        candidates = [(0, 1), (0, 2), (3, 4)]
        logits = self.scorer.score_candidate_sets(query, candidates)
        loss = -torch.log_softmax(logits, dim=0)[0] + self.scorer.regularisation(0.01)
        loss.backward()
        self.assertIsNotNone(self.scorer.Z.weight.grad)
        self.assertIsNotNone(self.scorer.P.weight.grad)
        self.assertIsNotNone(self.scorer.interactions["2"].grad)


if __name__ == "__main__":
    unittest.main()
