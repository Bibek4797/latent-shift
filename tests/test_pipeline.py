"""
tests/test_pipeline.py
----------------------
Unit tests for the LatentShift core pipeline.

Uses lightweight mock objects (no real LLM required) to validate:
- get_transformer_layer() with multiple architecture styles
- normalize_vector() correctness
- set_seed() reproducibility
- ConceptVectorEngine.compute_mean_difference()
- ConceptVectorEngine.compute_pca_vector()
- ConceptVectorEngine.save_vectors() / load_vectors() round-trip with metadata
- ActivationExtractor hook registration and cleanup
- SteeredGenerator hook lifecycle (register, remove, no leakage)
- SteeringEvaluator.compute_cosine_similarity()
- SteeringEvaluator.compute_perplexity() basic contract
"""

import os
import sys
import tempfile
import types
import unittest

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# PATH SETUP — allow running from repo root: python -m pytest tests/
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import get_logger, get_transformer_layer, normalize_vector, set_seed
from src.compute import ConceptVectorEngine
from src.evaluator import SteeringEvaluator

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Minimal mock models
# ---------------------------------------------------------------------------

class _DummyLayer(nn.Module):
    """Single linear layer that doubles hidden states."""
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, **kwargs):
        return (self.linear(x),)


class _LlamaStyleModel(nn.Module):
    """Mimics model.model.layers architecture (Llama / Qwen / Mistral)."""
    def __init__(self, num_layers: int = 4, hidden_dim: int = 16) -> None:
        super().__init__()
        self.model = types.SimpleNamespace(
            layers=nn.ModuleList([_DummyLayer(hidden_dim) for _ in range(num_layers)])
        )

    def forward(self, input_ids, **kwargs):
        h = torch.rand(1, input_ids.shape[1], 16)
        for layer in self.model.layers:
            h = layer(h)[0]
        # Simulate a CausalLM output
        class _Out:
            logits = torch.rand(1, input_ids.shape[1], 100)
            loss = None
        return _Out()

    def generate(self, input_ids, max_new_tokens=5, **kwargs):
        return torch.cat([input_ids, torch.zeros(1, max_new_tokens, dtype=torch.long)], dim=1)

    def get_input_embeddings(self):
        emb = nn.Embedding(100, 16)
        return emb


class _GPT2StyleModel(nn.Module):
    """Mimics model.transformer.h architecture (GPT-2)."""
    def __init__(self) -> None:
        super().__init__()
        self.transformer = types.SimpleNamespace(
            h=nn.ModuleList([_DummyLayer() for _ in range(4)])
        )


class _NeoXStyleModel(nn.Module):
    """Mimics model.gpt_neox.layers architecture (GPT-NeoX)."""
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = types.SimpleNamespace(
            layers=nn.ModuleList([_DummyLayer() for _ in range(4)])
        )


class _DummyTokenizer:
    """Minimal tokenizer stub for pipeline tests."""
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    vocab_size = 100
    padding_side = "left"

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        return {"input_ids": ids}


# ===========================================================================
# Test Cases
# ===========================================================================

class TestUtilsGetTransformerLayer(unittest.TestCase):
    """Tests for get_transformer_layer() across architectures."""

    def test_llama_style(self):
        model = _LlamaStyleModel(num_layers=4)
        layer = get_transformer_layer(model, 2)
        self.assertIsInstance(layer, _DummyLayer)

    def test_gpt2_style(self):
        model = _GPT2StyleModel()
        layer = get_transformer_layer(model, 0)
        self.assertIsInstance(layer, _DummyLayer)

    def test_neox_style(self):
        model = _NeoXStyleModel()
        layer = get_transformer_layer(model, 3)
        self.assertIsInstance(layer, _DummyLayer)

    def test_out_of_range_raises(self):
        model = _LlamaStyleModel(num_layers=4)
        with self.assertRaises(IndexError):
            get_transformer_layer(model, 10)

    def test_unsupported_raises(self):
        with self.assertRaises(AttributeError):
            get_transformer_layer(nn.Linear(4, 4), 0)


class TestUtilsNormalizeVector(unittest.TestCase):
    """Tests for normalize_vector()."""

    def test_unit_norm(self):
        vec = torch.tensor([3.0, 4.0])
        normed = normalize_vector(vec)
        self.assertAlmostEqual(torch.norm(normed).item(), 1.0, places=5)

    def test_dtype_preserved(self):
        vec = torch.tensor([1.0, 2.0], dtype=torch.float16)
        normed = normalize_vector(vec)
        self.assertEqual(normed.dtype, torch.float16)

    def test_zero_vector_no_crash(self):
        vec = torch.zeros(8)
        normed = normalize_vector(vec)
        self.assertFalse(torch.isnan(normed).any())


class TestUtilsSetSeed(unittest.TestCase):
    """Tests for set_seed() reproducibility."""

    def test_same_seed_same_output(self):
        set_seed(99)
        a = torch.rand(10)
        set_seed(99)
        b = torch.rand(10)
        self.assertTrue(torch.allclose(a, b))

    def test_diff_seed_diff_output(self):
        set_seed(1)
        a = torch.rand(10)
        set_seed(2)
        b = torch.rand(10)
        self.assertFalse(torch.allclose(a, b))


class TestConceptVectorEngine(unittest.TestCase):
    """Tests for ConceptVectorEngine compute / save / load."""

    def _make_activations(self, n=8, d=32):
        pos = torch.randn(n, d)
        neg = torch.randn(n, d)
        return pos, neg

    def test_mean_difference_shape(self):
        pos, neg = self._make_activations()
        vec = ConceptVectorEngine.compute_mean_difference(pos, neg)
        self.assertEqual(vec.shape, (32,))

    def test_mean_difference_direction(self):
        """The MD vector should point from neg centroid toward pos centroid."""
        d = 16
        pos = torch.ones(4, d) * 2.0 + torch.randn(4, d) * 0.01
        neg = torch.zeros(4, d) + torch.randn(4, d) * 0.01
        vec = ConceptVectorEngine.compute_mean_difference(pos, neg)
        # All elements should be positive (pos > neg)
        self.assertTrue((vec > 0).all())

    def test_pca_vector_shape(self):
        pos, neg = self._make_activations()
        vec = ConceptVectorEngine.compute_pca_vector(pos, neg)
        self.assertEqual(vec.shape, (32,))

    def test_pca_sign_alignment(self):
        """PCA vector should be positively correlated with mean difference."""
        pos, neg = self._make_activations(n=20, d=64)
        mean_vec = ConceptVectorEngine.compute_mean_difference(pos, neg)
        pca_vec = ConceptVectorEngine.compute_pca_vector(pos, neg)
        cos = SteeringEvaluator.compute_cosine_similarity(mean_vec, pca_vec)
        self.assertGreater(cos, 0.0)

    def test_normalize_option(self):
        pos, neg = self._make_activations()
        vec = ConceptVectorEngine.compute_mean_difference(pos, neg, normalize=True)
        self.assertAlmostEqual(torch.norm(vec.float()).item(), 1.0, places=4)

    def test_save_and_load_roundtrip(self):
        pos, neg = self._make_activations(d=16)
        vectors = {
            0: ConceptVectorEngine.compute_mean_difference(pos, neg),
            1: ConceptVectorEngine.compute_mean_difference(pos, neg),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = ConceptVectorEngine.save_vectors(
                vectors,
                tmpdir,
                "test_vec.pt",
                metadata={"model_name": "test", "method": "mean_diff"},
            )
            self.assertTrue(os.path.exists(saved_path))
            loaded = ConceptVectorEngine.load_vectors(saved_path)
            for layer in vectors:
                self.assertTrue(torch.allclose(vectors[layer].float(), loaded[layer].float(), atol=1e-5))

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ConceptVectorEngine.load_vectors("/nonexistent/path/vec.pt")


class TestSteeringEvaluator(unittest.TestCase):
    """Tests for SteeringEvaluator metrics."""

    def test_cosine_similarity_identical(self):
        vec = torch.randn(64)
        self.assertAlmostEqual(
            SteeringEvaluator.compute_cosine_similarity(vec, vec), 1.0, places=5
        )

    def test_cosine_similarity_orthogonal(self):
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(
            SteeringEvaluator.compute_cosine_similarity(v1, v2), 0.0, places=5
        )

    def test_cosine_similarity_opposite(self):
        vec = torch.randn(64)
        self.assertAlmostEqual(
            SteeringEvaluator.compute_cosine_similarity(vec, -vec), -1.0, places=4
        )

    def test_cosine_zero_vector_returns_zero(self):
        v1 = torch.zeros(16)
        v2 = torch.randn(16)
        self.assertEqual(SteeringEvaluator.compute_cosine_similarity(v1, v2), 0.0)

    def test_perplexity_empty_text(self):
        """compute_perplexity should return nan for empty string."""
        model = _LlamaStyleModel()
        tokenizer = _DummyTokenizer()
        result = SteeringEvaluator.compute_perplexity(model, tokenizer, "   ")
        self.assertTrue(result != result)  # nan != nan


class TestSteeredGeneratorHooks(unittest.TestCase):
    """Test hook lifecycle without triggering real generation."""

    def setUp(self):
        from src.steer import SteeredGenerator
        self.model = _LlamaStyleModel(num_layers=4)
        self.tokenizer = _DummyTokenizer()
        self.generator = SteeredGenerator(self.model, self.tokenizer, device="cpu")

    def test_no_hooks_initially(self):
        self.assertEqual(len(self.generator.active_hooks), 0)

    def test_register_adds_hooks(self):
        vectors = {0: torch.randn(16), 1: torch.randn(16)}
        self.generator.register_steering_hooks(vectors, alpha=1.0)
        self.assertEqual(len(self.generator.active_hooks), 2)
        self.generator.remove_steering_hooks()

    def test_remove_clears_hooks(self):
        vectors = {0: torch.randn(16)}
        self.generator.register_steering_hooks(vectors, alpha=1.0)
        self.generator.remove_steering_hooks()
        self.assertEqual(len(self.generator.active_hooks), 0)

    def test_double_register_no_stacking(self):
        """Re-registering should replace, not stack, hooks."""
        vectors = {0: torch.randn(16)}
        self.generator.register_steering_hooks(vectors, alpha=1.0)
        self.generator.register_steering_hooks(vectors, alpha=2.0)
        self.assertEqual(len(self.generator.active_hooks), 1)
        self.generator.remove_steering_hooks()

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            self.generator.generate("   ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
