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

import json
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
    eos_token_id = 1
    vocab_size = 100
    padding_side = "left"

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens=True):
        return "dummy decoded text"


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


class TestBenchmarkEngine(unittest.TestCase):
    """Tests for BenchmarkEngine, dataset exports, and plotting functions."""

    def setUp(self):
        import tempfile
        from src.benchmark import BenchmarkEngine, SingleBenchmarkRun, BenchmarkGridConfig
        self.temp_dir = tempfile.mkdtemp()
        self.engine = BenchmarkEngine(output_dir=self.temp_dir)
        self.sample_run = SingleBenchmarkRun(
            run_id="run_001",
            model_name="gpt2",
            concept="positivity",
            extraction_method="lda",
            steering_strategy="uniform",
            alpha=2.0,
            layers=[6, 7, 8],
            prompt="How are you?",
            ppl_baseline=10.0,
            ppl_steered=15.0,
            delta_ppl=5.0,
            ppl_ratio=1.5,
            cosine_sim=0.85,
            kl_divergence=1.2,
            js_divergence=0.25,
            entropy_baseline=3.5,
            entropy_steered=4.0,
            steering_strength_score=0.75,
            runtime_ms=120.0,
            cpu_memory_mb=12.0,
            gpu_memory_mb=0.0,
            timestamp="2026-08-03 22:00:00 UTC",
        )
        self.engine.add_run(self.sample_run)

    def test_add_and_filter_runs(self):
        self.assertEqual(len(self.engine.runs), 1)
        filtered = self.engine.filter_runs(extraction_method="lda", steering_strategy="uniform")
        self.assertEqual(len(filtered), 1)
        filtered_empty = self.engine.filter_runs(extraction_method="nonexistent")
        self.assertEqual(len(filtered_empty), 0)

    def test_export_csv_json_markdown(self):
        csv_path = self.engine.export_csv("test_benchmark.csv")
        json_path = self.engine.export_json("test_benchmark.json")
        md_path = self.engine.export_markdown_report("test_benchmark.md")

        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))

    def test_benchmark_plotting_functions(self):
        from src.benchmark import (
            plot_benchmark_bar_chart,
            plot_benchmark_heatmap,
            plot_benchmark_leaderboard,
            plot_benchmark_radar_chart,
        )
        fig_bar = plot_benchmark_bar_chart(self.engine.runs)
        fig_radar = plot_benchmark_radar_chart(self.engine.runs)
        fig_heat = plot_benchmark_heatmap(self.engine.runs)
        fig_lead = plot_benchmark_leaderboard(self.engine.runs)

        self.assertIsNotNone(fig_bar)
        self.assertIsNotNone(fig_radar)
        self.assertIsNotNone(fig_heat)
        self.assertIsNotNone(fig_lead)


# ===========================================================================
# TEST: Dynamic Closed-Loop Steering Schedulers
# ===========================================================================

class TestSchedulers(unittest.TestCase):
    """Validate all alpha scheduler implementations and factory."""

    def test_fixed_scheduler_returns_constant(self):
        from src.schedulers import FixedScheduler
        sched = FixedScheduler(alpha=2.5)
        for step in range(20):
            self.assertAlmostEqual(sched.step(step, 20), 2.5)

    def test_linear_scheduler_endpoints(self):
        from src.schedulers import LinearScheduler
        sched = LinearScheduler(alpha_start=4.0, alpha_end=1.0)
        self.assertAlmostEqual(sched.step(0, 100), 4.0, places=5)
        self.assertAlmostEqual(sched.step(99, 100), 1.0, places=5)

    def test_linear_scheduler_midpoint(self):
        from src.schedulers import LinearScheduler
        sched = LinearScheduler(alpha_start=4.0, alpha_end=0.0)
        mid = sched.step(50, 101)
        self.assertAlmostEqual(mid, 2.0, places=3)

    def test_linear_scheduler_single_step(self):
        from src.schedulers import LinearScheduler
        sched = LinearScheduler(alpha_start=3.0, alpha_end=1.0)
        self.assertAlmostEqual(sched.step(0, 1), 3.0)

    def test_cosine_scheduler_endpoints(self):
        from src.schedulers import CosineScheduler
        sched = CosineScheduler(alpha_max=3.0, alpha_min=0.0)
        self.assertAlmostEqual(sched.step(0, 100), 3.0, places=4)
        self.assertAlmostEqual(sched.step(99, 100), 0.0, places=4)

    def test_cosine_scheduler_symmetry(self):
        """Cosine schedule should be symmetric around the midpoint."""
        from src.schedulers import CosineScheduler
        sched = CosineScheduler(alpha_max=4.0, alpha_min=0.0)
        val_quarter = sched.step(25, 101)
        val_three_quarter = sched.step(75, 101)
        self.assertAlmostEqual(val_quarter + val_three_quarter, 4.0, places=3)

    def test_cosine_scheduler_single_step(self):
        from src.schedulers import CosineScheduler
        sched = CosineScheduler(alpha_max=5.0, alpha_min=1.0)
        self.assertAlmostEqual(sched.step(0, 1), 5.0)

    def test_confidence_scheduler_no_logits_returns_base(self):
        from src.schedulers import ConfidenceBasedScheduler
        sched = ConfidenceBasedScheduler(alpha_base=2.0)
        self.assertAlmostEqual(sched.step(0, 10, logits=None), 2.0)

    def test_confidence_scheduler_with_peaked_logits(self):
        """Peaked distribution (low entropy) should yield alpha < base."""
        from src.schedulers import ConfidenceBasedScheduler
        sched = ConfidenceBasedScheduler(alpha_base=2.0, gamma=1.0, top_k=10)
        logits = torch.full((100,), -100.0)
        logits[0] = 100.0  # One dominant token
        alpha = sched.step(0, 10, logits=logits)
        self.assertLess(alpha, 2.0)

    def test_confidence_scheduler_with_flat_logits(self):
        """Flat distribution (high entropy) should yield alpha close to base."""
        from src.schedulers import ConfidenceBasedScheduler
        sched = ConfidenceBasedScheduler(alpha_base=2.0, gamma=1.0, top_k=50)
        logits = torch.zeros(100)  # Uniform distribution
        alpha = sched.step(0, 10, logits=logits)
        self.assertGreater(alpha, 1.5)

    def test_entropy_scheduler_no_logits_returns_midpoint(self):
        from src.schedulers import EntropyBasedScheduler
        sched = EntropyBasedScheduler(alpha_min=1.0, alpha_max=3.0)
        self.assertAlmostEqual(sched.step(0, 10, logits=None), 2.0)

    def test_entropy_scheduler_low_entropy(self):
        """Peaked distribution → close to alpha_min."""
        from src.schedulers import EntropyBasedScheduler
        sched = EntropyBasedScheduler(alpha_min=0.5, alpha_max=3.0)
        logits = torch.full((100,), -100.0)
        logits[0] = 100.0
        alpha = sched.step(0, 10, logits=logits)
        self.assertLess(alpha, 1.0)

    def test_entropy_scheduler_high_entropy(self):
        """Flat distribution → close to alpha_max."""
        from src.schedulers import EntropyBasedScheduler
        sched = EntropyBasedScheduler(alpha_min=0.5, alpha_max=3.0, h_ref=4.0)
        logits = torch.zeros(1000)  # Nearly uniform
        alpha = sched.step(0, 10, logits=logits)
        self.assertGreater(alpha, 2.5)

    def test_build_scheduler_factory_all_types(self):
        from src.schedulers import build_scheduler
        for name in ["fixed", "linear", "cosine", "confidence", "entropy"]:
            sched = build_scheduler(name)
            self.assertEqual(sched.name, name)

    def test_build_scheduler_unknown_raises(self):
        from src.schedulers import build_scheduler
        with self.assertRaises(ValueError):
            build_scheduler("nonexistent_scheduler")

    def test_build_scheduler_with_kwargs(self):
        from src.schedulers import build_scheduler
        sched = build_scheduler("linear", alpha_start=5.0, alpha_end=0.0)
        self.assertAlmostEqual(sched.step(0, 10), 5.0)

    def test_scheduler_monotonicity_linear(self):
        """Linear scheduler should monotonically decrease (start > end)."""
        from src.schedulers import LinearScheduler
        sched = LinearScheduler(alpha_start=5.0, alpha_end=1.0)
        values = [sched.step(i, 50) for i in range(50)]
        for i in range(1, len(values)):
            self.assertLessEqual(values[i], values[i - 1] + 1e-9)

    def test_scheduler_monotonicity_cosine(self):
        """Cosine scheduler should monotonically decrease."""
        from src.schedulers import CosineScheduler
        sched = CosineScheduler(alpha_max=5.0, alpha_min=0.0)
        values = [sched.step(i, 50) for i in range(50)]
        for i in range(1, len(values)):
            self.assertLessEqual(values[i], values[i - 1] + 1e-9)


class TestAlphaTrajectory(unittest.TestCase):
    """Validate alpha trajectory recording and serialization."""

    def test_record_and_to_dict(self):
        from src.schedulers import AlphaTrajectory
        traj = AlphaTrajectory(scheduler_name="test")
        for v in [1.0, 2.0, 3.0]:
            traj.record(v)
        d = traj.to_dict()
        self.assertEqual(d["scheduler_name"], "test")
        self.assertEqual(d["num_steps"], 3)
        self.assertAlmostEqual(d["alpha_mean"], 2.0, places=3)
        self.assertAlmostEqual(d["alpha_min"], 1.0, places=3)
        self.assertAlmostEqual(d["alpha_max"], 3.0, places=3)

    def test_empty_trajectory(self):
        from src.schedulers import AlphaTrajectory
        traj = AlphaTrajectory(scheduler_name="empty")
        d = traj.to_dict()
        self.assertEqual(d["num_steps"], 0)
        self.assertEqual(d["alpha_mean"], 0.0)


class TestSchedulerPlotting(unittest.TestCase):
    """Validate that plotting utilities produce valid Plotly figures."""

    def test_plot_alpha_trajectory(self):
        from src.schedulers import AlphaTrajectory, plot_alpha_trajectory
        traj = AlphaTrajectory(scheduler_name="linear")
        for i in range(10):
            traj.record(3.0 - 0.3 * i)
        fig = plot_alpha_trajectory(traj)
        self.assertIsNotNone(fig)

    def test_plot_token_steering_strength(self):
        from src.schedulers import AlphaTrajectory, plot_token_steering_strength
        traj = AlphaTrajectory(scheduler_name="cosine")
        for i in range(5):
            traj.record(2.0)
        fig = plot_token_steering_strength(traj)
        self.assertIsNotNone(fig)

    def test_plot_with_token_labels(self):
        from src.schedulers import AlphaTrajectory, plot_token_steering_strength
        traj = AlphaTrajectory(scheduler_name="cosine")
        for i in range(3):
            traj.record(1.5)
        fig = plot_token_steering_strength(traj, tokens=["Hello", "world", "!"])
        self.assertIsNotNone(fig)


class TestDynamicSteeringHooks(unittest.TestCase):
    """Validate dynamic hook registration and alpha container update."""

    def setUp(self):
        self.model = _LlamaStyleModel(num_layers=4, hidden_dim=16)
        self.tokenizer = _DummyTokenizer()
        from src.steer import SteeredGenerator
        self.gen = SteeredGenerator(self.model, self.tokenizer, device="cpu")

    def test_dynamic_hooks_register_and_remove(self):
        """Dynamic hooks should register and clean up properly."""
        vectors = {0: torch.randn(16), 1: torch.randn(16)}
        self.gen.register_dynamic_steering_hooks(vectors)
        self.assertEqual(len(self.gen.active_hooks), 2)
        self.gen.remove_steering_hooks()
        self.assertEqual(len(self.gen.active_hooks), 0)

    def test_dynamic_alpha_container_updates(self):
        """Changing _dynamic_alpha should affect hook behavior."""
        vectors = {0: torch.ones(16)}
        self.gen.register_dynamic_steering_hooks(vectors, layer_weight_ratios={0: 1.0})

        # Set alpha to 0 → output should be unaffected
        self.gen._dynamic_alpha[0] = 0.0
        x = torch.randn(1, 3, 16)
        layer = self.model.model.layers[0]
        out_base = layer(x)

        # Set alpha to a large value → output should be shifted
        self.gen._dynamic_alpha[0] = 10.0
        out_steered = layer(x)

        # The outputs should differ when alpha != 0
        if isinstance(out_base, tuple):
            diff = (out_steered[0] - out_base[0]).abs().sum().item()
        else:
            diff = (out_steered - out_base).abs().sum().item()
        self.assertGreater(diff, 0.0)

        self.gen.remove_steering_hooks()

    def test_dynamic_hooks_with_ratios(self):
        """Layer weight ratios should scale the effective alpha."""
        vectors = {0: torch.ones(16), 1: torch.ones(16)}
        ratios = {0: 0.5, 1: 2.0}
        self.gen.register_dynamic_steering_hooks(vectors, layer_weight_ratios=ratios)
        self.assertEqual(len(self.gen.active_hooks), 2)
        self.gen.remove_steering_hooks()

    def test_no_hook_leakage_after_dynamic(self):
        """After dynamic generation flow, hooks should be cleaned."""
        vectors = {0: torch.randn(16)}
        self.gen.register_dynamic_steering_hooks(vectors)
        self.gen.remove_steering_hooks()

        # Verify no hooks remain on any layer
        for layer in self.model.model.layers:
            self.assertEqual(len(layer._forward_hooks), 0)


class TestCustomScheduler(unittest.TestCase):
    """Validate that custom schedulers can be created by subclassing."""

    def test_custom_scheduler_subclass(self):
        from src.schedulers import BaseAlphaScheduler

        class StepScheduler(BaseAlphaScheduler):
            name = "step"
            def __init__(self, alpha_high=3.0, alpha_low=1.0, switch_at=0.5):
                self.alpha_high = alpha_high
                self.alpha_low = alpha_low
                self.switch_at = switch_at

            def step(self, step_idx, total_steps, logits=None, **kw):
                t = step_idx / max(total_steps - 1, 1)
                return self.alpha_high if t < self.switch_at else self.alpha_low

        sched = StepScheduler()
        self.assertAlmostEqual(sched.step(0, 10), 3.0)
        self.assertAlmostEqual(sched.step(9, 10), 1.0)

    def test_scheduler_registry_contains_builtins(self):
        from src.schedulers import SCHEDULER_REGISTRY
        for key in ["fixed", "linear", "cosine", "confidence", "entropy"]:
            self.assertIn(key, SCHEDULER_REGISTRY)


# ===========================================================================
# TEST: Experiment Tracker
# ===========================================================================

class TestExperimentRecord(unittest.TestCase):
    """Validate ExperimentRecord dataclass."""

    def test_auto_generated_fields(self):
        from src.experiment_tracker import ExperimentRecord
        rec = ExperimentRecord(model_name="gpt2", concept="safety")
        self.assertTrue(len(rec.experiment_id) > 0)
        self.assertIn("UTC", rec.timestamp)
        # git_commit may be empty if not in a repo, that's fine
        self.assertIsInstance(rec.git_commit, str)

    def test_to_dict(self):
        from src.experiment_tracker import ExperimentRecord
        rec = ExperimentRecord(
            model_name="gpt2",
            layers=[1, 2, 3],
            alpha=2.0,
            concept="positivity",
            extraction_method="pca",
        )
        d = rec.to_dict()
        self.assertEqual(d["model_name"], "gpt2")
        self.assertEqual(d["layers"], [1, 2, 3])
        self.assertEqual(d["alpha"], 2.0)

    def test_to_json(self):
        from src.experiment_tracker import ExperimentRecord
        rec = ExperimentRecord(model_name="gpt2", concept="test")
        j = rec.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["model_name"], "gpt2")


class TestExperimentTracker(unittest.TestCase):
    """Validate ExperimentTracker SQLite operations."""

    def setUp(self):
        # Use a temp DB for tests
        self.db_path = os.path.join(tempfile.mkdtemp(), "test_experiments.db")
        from src.experiment_tracker import ExperimentTracker, ExperimentRecord
        self.tracker = ExperimentTracker(db_path=self.db_path)
        self.ExperimentRecord = ExperimentRecord

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _make_record(self, **kwargs):
        defaults = dict(
            model_name="gpt2",
            layers=[6, 7, 8],
            alpha=2.0,
            weight_strategy="uniform",
            concept="safety",
            extraction_method="mean_diff",
            prompt="test prompt",
            baseline_text="baseline output",
            steered_text="steered output",
            ppl_baseline=10.0,
            ppl_steered=12.0,
            delta_ppl=2.0,
            ppl_ratio=1.2,
            cosine_sim=0.95,
            kl_divergence=0.05,
            js_divergence=0.03,
            entropy_baseline=3.0,
            entropy_steered=3.5,
            steering_strength_score=0.15,
            runtime_ms=150.0,
            cpu_memory_mb=512.0,
            gpu_memory_mb=1024.0,
        )
        defaults.update(kwargs)
        return self.ExperimentRecord(**defaults)

    def test_log_and_get_experiment(self):
        rec = self._make_record()
        exp_id = self.tracker.log_experiment(rec)
        self.assertEqual(exp_id, rec.experiment_id)

        retrieved = self.tracker.get_experiment(exp_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_name, "gpt2")
        self.assertEqual(retrieved.layers, [6, 7, 8])
        self.assertAlmostEqual(retrieved.alpha, 2.0)

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.tracker.get_experiment("nonexistent-id"))

    def test_count_experiments(self):
        self.assertEqual(self.tracker.count_experiments(), 0)
        self.tracker.log_experiment(self._make_record())
        self.tracker.log_experiment(self._make_record())
        self.assertEqual(self.tracker.count_experiments(), 2)

    def test_list_experiments(self):
        for i in range(5):
            self.tracker.log_experiment(self._make_record(alpha=float(i)))
        exps = self.tracker.list_experiments(limit=3)
        self.assertEqual(len(exps), 3)

    def test_list_with_filters(self):
        self.tracker.log_experiment(self._make_record(concept="safety"))
        self.tracker.log_experiment(self._make_record(concept="positivity"))
        self.tracker.log_experiment(self._make_record(concept="safety"))

        safety_exps = self.tracker.list_experiments(concept_filter="safety")
        self.assertEqual(len(safety_exps), 2)

        positivity_exps = self.tracker.list_experiments(concept_filter="positivity")
        self.assertEqual(len(positivity_exps), 1)

    def test_delete_experiment(self):
        rec = self._make_record()
        self.tracker.log_experiment(rec)
        self.assertTrue(self.tracker.delete_experiment(rec.experiment_id))
        self.assertIsNone(self.tracker.get_experiment(rec.experiment_id))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.tracker.delete_experiment("nonexistent"))

    def test_compare_experiments(self):
        ids = []
        for i in range(3):
            rec = self._make_record(alpha=float(i))
            self.tracker.log_experiment(rec)
            ids.append(rec.experiment_id)

        compared = self.tracker.compare_experiments(ids[:2])
        self.assertEqual(len(compared), 2)

    def test_compare_empty_list(self):
        self.assertEqual(self.tracker.compare_experiments([]), [])

    def test_get_unique_values(self):
        self.tracker.log_experiment(self._make_record(concept="safety"))
        self.tracker.log_experiment(self._make_record(concept="positivity"))
        self.tracker.log_experiment(self._make_record(concept="safety"))

        uniques = self.tracker.get_unique_values("concept")
        self.assertIn("safety", uniques)
        self.assertIn("positivity", uniques)
        self.assertEqual(len(uniques), 2)

    def test_get_unique_values_invalid_column(self):
        with self.assertRaises(ValueError):
            self.tracker.get_unique_values("nonexistent_column")

    def test_export_json(self):
        self.tracker.log_experiment(self._make_record())
        self.tracker.log_experiment(self._make_record())
        out_path = os.path.join(os.path.dirname(self.db_path), "export.json")
        result = self.tracker.export_experiments_json(out_path)
        self.assertTrue(os.path.exists(result))
        with open(result, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)

    def test_export_csv(self):
        self.tracker.log_experiment(self._make_record())
        out_path = os.path.join(os.path.dirname(self.db_path), "export.csv")
        result = self.tracker.export_experiments_csv(out_path)
        self.assertTrue(os.path.exists(result))


class TestExperimentTrackerHelpers(unittest.TestCase):
    """Validate helper functions."""

    def test_get_git_commit_returns_string(self):
        from src.experiment_tracker import get_git_commit
        result = get_git_commit()
        self.assertIsInstance(result, str)

    def test_get_system_memory_returns_tuple(self):
        from src.experiment_tracker import get_system_memory
        cpu, gpu = get_system_memory()
        self.assertIsInstance(cpu, float)
        self.assertIsInstance(gpu, float)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertGreaterEqual(gpu, 0.0)


class TestExperimentTrackerPlotting(unittest.TestCase):
    """Validate experiment plotting utilities."""

    def _make_records(self, n=3):
        from src.experiment_tracker import ExperimentRecord
        records = []
        for i in range(n):
            records.append(ExperimentRecord(
                model_name="gpt2",
                layers=[6, 7],
                alpha=float(i + 1),
                concept="safety",
                extraction_method="pca",
                ppl_ratio=1.0 + 0.1 * i,
                cosine_sim=0.9 - 0.05 * i,
                kl_divergence=0.01 * (i + 1),
                js_divergence=0.005 * (i + 1),
                steering_strength_score=0.1 * (i + 1),
                entropy_baseline=3.0,
                entropy_steered=3.0 + 0.2 * i,
            ))
        return records

    def test_plot_experiment_timeline(self):
        from src.experiment_tracker import plot_experiment_timeline
        fig = plot_experiment_timeline(self._make_records())
        self.assertIsNotNone(fig)

    def test_plot_experiment_comparison(self):
        from src.experiment_tracker import plot_experiment_comparison
        fig = plot_experiment_comparison(self._make_records(), metric="ppl_ratio")
        self.assertIsNotNone(fig)

    def test_plot_experiment_comparison_custom_metric(self):
        from src.experiment_tracker import plot_experiment_comparison
        fig = plot_experiment_comparison(self._make_records(), metric="cosine_sim")
        self.assertIsNotNone(fig)

    def test_plot_experiment_radar(self):
        from src.experiment_tracker import plot_experiment_radar
        fig = plot_experiment_radar(self._make_records())
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
