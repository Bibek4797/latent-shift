"""LatentShift — src package."""

from src.compute import ConceptVectorEngine
from src.concept_extractors import (
    EXTRACTOR_REGISTRY,
    BaseConceptExtractor,
    ConceptVectorComparer,
    ExtractorBenchmarkResult,
    plot_memory_comparison,
    plot_pairwise_cosine_heatmap,
    plot_runtime_comparison,
    plot_vector_magnitude_comparison,
)
from src.evaluator import (
    SteeringEvaluationReport,
    SteeringEvaluator,
    plot_layerwise_changes,
    plot_metric_comparison,
    plot_steering_strength,
)
from src.extractor import ActivationExtractor
from src.layer_selector import (
    LayerScoreResult,
    LayerSelector,
    plot_layer_scores_heatmap,
    plot_layer_scores_line,
    plot_top_k_layers_bar,
)
from src.model_loader import load_model_and_tokenizer
from src.steer import SteeredGenerator
from src.utils import compute_layer_weights, get_logger, normalize_vector, set_seed

from config import SteeringConfig

__all__ = [
    "SteeringConfig",
    "load_model_and_tokenizer",
    "ActivationExtractor",
    "ConceptVectorEngine",
    "SteeredGenerator",
    "SteeringEvaluator",
    "SteeringEvaluationReport",
    "LayerSelector",
    "LayerScoreResult",
    "BaseConceptExtractor",
    "EXTRACTOR_REGISTRY",
    "ConceptVectorComparer",
    "ExtractorBenchmarkResult",
    "plot_layerwise_changes",
    "plot_metric_comparison",
    "plot_steering_strength",
    "plot_layer_scores_line",
    "plot_layer_scores_heatmap",
    "plot_top_k_layers_bar",
    "plot_pairwise_cosine_heatmap",
    "plot_runtime_comparison",
    "plot_memory_comparison",
    "plot_vector_magnitude_comparison",
    "compute_layer_weights",
    "get_logger",
    "set_seed",
    "normalize_vector",
]




