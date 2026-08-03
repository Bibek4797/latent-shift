"""LatentShift — src package."""

from src.compute import ConceptVectorEngine
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
    "plot_layerwise_changes",
    "plot_metric_comparison",
    "plot_steering_strength",
    "plot_layer_scores_line",
    "plot_layer_scores_heatmap",
    "plot_top_k_layers_bar",
    "compute_layer_weights",
    "get_logger",
    "set_seed",
    "normalize_vector",
]



