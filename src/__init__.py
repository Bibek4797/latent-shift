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
    "plot_layerwise_changes",
    "plot_metric_comparison",
    "plot_steering_strength",
    "compute_layer_weights",
    "get_logger",
    "set_seed",
    "normalize_vector",
]


