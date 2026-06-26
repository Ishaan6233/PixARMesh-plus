"""Tests for EdgeRunner autoregressive generation helpers.

Related files:
- Exercises `src/utils/inference.py::get_prefix_allowed_tokens_fn_edgerunner`.
- Protects beam-search decoding from shared mutable grammar state.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.inference import get_prefix_allowed_tokens_fn_edgerunner


class _DummyConfig:
    vocab_size = 20
    eos_token_id = 2


class _DummyModel:
    config = _DummyConfig()


class EdgeRunnerGrammarTest(unittest.TestCase):
    def setUp(self):
        self.allowed = get_prefix_allowed_tokens_fn_edgerunner(_DummyModel())

    def _allowed_for(self, tokens, batch_id=0):
        return self.allowed(batch_id, torch.as_tensor(tokens, dtype=torch.long))

    def test_first_token_must_start_patch(self):
        self.assertEqual(self._allowed_for([]), [5])

    def test_bom_requires_nine_coordinate_tokens(self):
        self.assertEqual(self._allowed_for([5]), list(range(6, 20)))
        self.assertEqual(self._allowed_for([5] + [6] * 8), list(range(6, 20)))
        self.assertEqual(self._allowed_for([5] + [6] * 9), [3, 4, 5, 2])

    def test_lr_requires_three_coordinate_tokens(self):
        prefix = [5] + [6] * 9 + [3]
        self.assertEqual(self._allowed_for(prefix), list(range(6, 20)))
        self.assertEqual(self._allowed_for(prefix + [7, 8]), list(range(6, 20)))
        self.assertEqual(self._allowed_for(prefix + [7, 8, 9]), [3, 4, 5, 2])

    def test_stateless_for_beam_batch_ids(self):
        # Beam search can call the grammar with batch IDs beyond the original batch
        # size. The grammar should derive state from tokens, not index mutable state.
        self.assertEqual(self._allowed_for([], batch_id=7), [5])
        self.assertEqual(self._allowed_for([5] + [6] * 9 + [4, 7], batch_id=7),
                         list(range(6, 20)))


if __name__ == "__main__":
    unittest.main()
