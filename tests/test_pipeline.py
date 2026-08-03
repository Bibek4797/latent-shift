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

from src.utils import compute_layer_weights, get_logger, get_transformer_layer, normalize_vector, set_seed
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

    def test_kl_divergence_identical(self):
        logits = torch.randn(1, 10, 100)
        kl = SteeringEvaluator.compute_kl_divergence(logits, logits)
        self.assertAlmostEqual(kl, 0.0, places=4)

    def test_kl_divergence_positive(self):
        p_logits = torch.tensor([[[2.0, 1.0, 0.1]]])
        q_logits = torch.tensor([[[0.1, 1.0, 2.0]]])
        kl = SteeringEvaluator.compute_kl_divergence(p_logits, q_logits)
        self.assertGreater(kl, 0.0)

    def test_js_divergence_symmetry(self):
        p_logits = torch.randn(1, 5, 50)
        q_logits = torch.randn(1, 5, 50)
        js1 = SteeringEvaluator.compute_js_divergence(p_logits, q_logits)
        js2 = SteeringEvaluator.compute_js_divergence(q_logits, p_logits)
        self.assertAlmostEqual(js1, js2, places=4)
        self.assertGreaterEqual(js1, 0.0)

    def test_entropy_non_negative(self):
        logits = torch.randn(1, 5, 50)
        ent = SteeringEvaluator.compute_entropy(logits)
        self.assertGreater(ent, 0.0)

    def test_hidden_state_norm_difference(self):
        hb = torch.tensor([3.0, 4.0])  # norm = 5.0
        hs = torch.tensor([6.0, 8.0])  # norm = 10.0
        diff = SteeringEvaluator.compute_hidden_state_norm_difference(hb, hs)
        self.assertAlmostEqual(diff, 5.0, places=4)

    def test_layerwise_cosine_similarity(self):
        hb_dict = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}
        hs_dict = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, -1.0])}
        cos_map = SteeringEvaluator.compute_layerwise_cosine_similarity(hb_dict, hs_dict)
        self.assertAlmostEqual(cos_map[0], 1.0, places=4)
        self.assertAlmostEqual(cos_map[1], -1.0, places=4)

    def test_average_shift_magnitude(self):
        hb_dict = {0: torch.zeros(4), 1: torch.zeros(4)}
        hs_dict = {0: torch.ones(4) * 3.0, 1: torch.ones(4) * 4.0}  # norm: 6.0 and 8.0
        avg_shift = SteeringEvaluator.compute_average_shift_magnitude(hb_dict, hs_dict)
        self.assertAlmostEqual(avg_shift, 7.0, places=4)

    def test_steering_strength_score(self):
        hb_dict = {0: torch.tensor([3.0, 4.0])}  # norm = 5.0
        hs_dict = {0: torch.tensor([3.0, 9.0])}  # shift = [0, 5], shift norm = 5.0
        score = SteeringEvaluator.compute_steering_strength_score(hb_dict, hs_dict)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_evaluate_full_and_serialization(self):
        model = _LlamaStyleModel()
        tokenizer = _DummyTokenizer()
        hb_dict = {0: torch.randn(16), 1: torch.randn(16)}
        hs_dict = {0: torch.randn(16), 1: torch.randn(16)}

        rep = SteeringEvaluator.evaluate_full(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            baseline_text="Hello baseline",
            steered_text="Hello steered",
            h_base_dict=hb_dict,
            h_steered_dict=hs_dict,
        )
        self.assertIsNotNone(rep.kl_divergence)
        self.assertIsNotNone(rep.js_divergence)
        json_str = rep.to_json()
        self.assertIn("kl_divergence", json_str)
        self.assertIn("js_divergence", json_str)



class TestComputeLayerWeights(unittest.TestCase):
    """Tests for compute_layer_weights() weighting strategies."""

    def test_uniform_strategy(self):
        weights = compute_layer_weights([10, 11, 12], base_alpha=2.0, strategy="uniform")
        self.assertEqual(weights, {10: 2.0, 11: 2.0, 12: 2.0})

    def test_linear_decay_strategy(self):
        weights = compute_layer_weights([10, 11, 12], base_alpha=2.0, strategy="linear_decay")
        self.assertAlmostEqual(weights[10], 2.0)
        self.assertAlmostEqual(weights[11], 1.0)
        self.assertAlmostEqual(weights[12], 0.0)

    def test_cosine_decay_strategy(self):
        weights = compute_layer_weights([10, 11, 12], base_alpha=2.0, strategy="cosine_decay")
        self.assertAlmostEqual(weights[10], 2.0)
        self.assertAlmostEqual(weights[11], 1.0)
        self.assertAlmostEqual(weights[12], 0.0)

    def test_single_layer(self):
        for strat in ["uniform", "linear_decay", "cosine_decay"]:
            weights = compute_layer_weights([5], base_alpha=3.0, strategy=strat)
            self.assertEqual(weights, {5: 3.0})

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            compute_layer_weights([1, 2], base_alpha=1.0, strategy="unknown_strat")


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

    def test_register_per_layer_alpha_dict(self):
        vectors = {0: torch.randn(16), 1: torch.randn(16)}
        alpha_dict = {0: 2.5, 1: 0.5}
        self.generator.register_steering_hooks(vectors, alpha=alpha_dict)
        self.assertEqual(len(self.generator.active_hooks), 2)
        self.generator.remove_steering_hooks()

    def test_register_with_decay_strategy(self):
        vectors = {0: torch.randn(16), 1: torch.randn(16), 2: torch.randn(16)}
        self.generator.register_steering_hooks(vectors, alpha=2.0, strategy="cosine_decay")
        self.assertEqual(len(self.generator.active_hooks), 3)
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


class TestLayerSelector(unittest.TestCase):
    """Tests for LayerSelector scoring functions and ranking."""

    def setUp(self):
        from src.layer_selector import LayerSelector
        self.selector = LayerSelector
        self.pos_acts = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        self.neg_acts = torch.tensor([[-1.0, -2.0], [-2.0, -3.0]])

    def test_compute_mean_separation(self):
        sep = self.selector.compute_mean_separation(self.pos_acts, self.neg_acts)
        self.assertGreater(sep, 0.0)

    def test_compute_cosine_separation(self):
        sep = self.selector.compute_cosine_separation(self.pos_acts, self.neg_acts)
        self.assertAlmostEqual(sep, 2.0, places=4)

    def test_compute_fisher_score(self):
        score = self.selector.compute_fisher_score(self.pos_acts, self.neg_acts)
        self.assertGreater(score, 0.0)

    def test_compute_snr(self):
        snr = self.selector.compute_snr(self.pos_acts, self.neg_acts)
        self.assertGreater(snr, 0.0)

    def test_compute_activation_variance(self):
        var = self.selector.compute_activation_variance(self.pos_acts, self.neg_acts)
        self.assertGreater(var, 0.0)

    def test_score_layers_all_methods(self):
        pos_dict = {0: self.pos_acts, 1: self.pos_acts * 2.0}
        neg_dict = {0: self.neg_acts, 1: self.neg_acts * 2.0}
        methods = ["mean_separation", "cosine_separation", "fisher_score", "snr", "activation_variance"]
        for m in methods:
            scores = self.selector.score_layers(pos_dict, neg_dict, method=m)
            self.assertEqual(len(scores), 2)
            self.assertIn(0, scores)
            self.assertIn(1, scores)

    def test_rank_layers(self):
        scores = {0: 1.2, 1: 4.5, 2: 2.3}
        ranked = self.selector.rank_layers(scores)
        self.assertEqual(ranked[0].layer_idx, 1)
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].layer_idx, 2)
        self.assertEqual(ranked[2].layer_idx, 0)

    def test_select_top_k_layers(self):
        scores = {0: 1.2, 1: 4.5, 2: 2.3, 3: 0.1}
        top_2 = self.selector.select_top_k_layers(scores, k=2, preserve_order=True)
        self.assertEqual(top_2, [1, 2])

    def test_invalid_scoring_method_raises(self):
        pos_dict = {0: self.pos_acts}
        neg_dict = {0: self.neg_acts}
        with self.assertRaises(ValueError):
            self.selector.score_layers(pos_dict, neg_dict, method="unknown_metric")


class TestConceptExtractors(unittest.TestCase):
    """Tests for all 7 concept vector extraction algorithms."""

    def setUp(self):
        self.pos_acts = torch.randn(10, 32) + 2.0
        self.neg_acts = torch.randn(10, 32) - 2.0

    def test_all_extraction_methods(self):
        methods = [
            "mean_diff",
            "pca",
            "lda",
            "logistic_regression",
            "linear_svm",
            "sparse_pca",
            "truncated_svd",
        ]
        for m in methods:
            vec = ConceptVectorEngine.compute_vector(m, self.pos_acts, self.neg_acts, normalize=False)
            self.assertEqual(vec.shape, (32,))
            self.assertGreater(torch.norm(vec).item(), 0.0)

    def test_all_extraction_methods_normalized(self):
        methods = [
            "mean_diff",
            "pca",
            "lda",
            "logistic_regression",
            "linear_svm",
            "sparse_pca",
            "truncated_svd",
        ]
        for m in methods:
            vec = ConceptVectorEngine.compute_vector(m, self.pos_acts, self.neg_acts, normalize=True)
            self.assertAlmostEqual(torch.norm(vec).item(), 1.0, places=4)

    def test_invalid_extraction_method_raises(self):
        with self.assertRaises(ValueError):
            ConceptVectorEngine.compute_vector("unknown_algo", self.pos_acts, self.neg_acts)

    def test_concept_vector_comparer_benchmark(self):
        from src.concept_extractors import ConceptVectorComparer
        res = ConceptVectorComparer.benchmark_all_methods(self.pos_acts, self.neg_acts)
        self.assertEqual(len(res), 7)
        self.assertIn("lda", res)
        self.assertIn("linear_svm", res)

        labels, matrix = ConceptVectorComparer.compute_pairwise_cosine_matrix(res)
        self.assertEqual(len(labels), 7)
        self.assertEqual(matrix.shape, (7, 7))
        self.assertAlmostEqual(matrix[0, 0], 1.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)



