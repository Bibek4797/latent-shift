"""
src/utils.py
------------
Shared utility functions used across the LatentShift pipeline.

Includes:
- get_transformer_layer(): DRY helper to locate a layer by index across diverse
  transformer architectures (Llama, Qwen, Mistral, GPT-2, GPT-NeoX).
- get_logger(): Standard library logger factory with consistent formatting.
- set_seed(): Reproducibility seed setter for Python, NumPy, and PyTorch.
- normalize_vector(): Optional L2 normalization for concept vectors.
- compute_layer_weights(): Calculates per-layer steering coefficients (alpha_i)
  based on adaptive weighting strategies ("uniform", "linear_decay", "cosine_decay").
"""

import logging
import math
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger with a standard console handler.

    Parameters
    ----------
    name : str
        The logger name (typically ``__name__`` of the calling module).
    level : int, default=logging.INFO
        The minimum logging level threshold.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transformer_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """
    Retrieve a transformer layer module by index, supporting multiple
    well-known LLM architectures.

    Supported architectures:
    - Llama, Qwen, Mistral, Falcon (``model.model.layers``)
    - GPT-2 / DistilGPT-2 (``model.transformer.h``)
    - GPT-NeoX / Pythia (``model.gpt_neox.layers``)

    Parameters
    ----------
    model : nn.Module
        The top-level causal language model.
    layer_idx : int
        Zero-based index of the target transformer layer.

    Returns
    -------
    nn.Module
        The PyTorch module representing the requested transformer layer.

    Raises
    ------
    AttributeError
        If the architecture cannot be auto-detected.
    IndexError
        If ``layer_idx`` is out of range for the discovered layer list.
    """
    # Llama / Qwen / Mistral / Falcon / Gemma style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    # GPT-2 / DistilGPT-2 style
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    # GPT-NeoX / Pythia style
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
    else:
        raise AttributeError(
            "Unsupported transformer architecture. "
            "Cannot auto-detect layer list. "
            "Supported: model.model.layers, model.transformer.h, model.gpt_neox.layers"
        )

    num_layers = len(layers)
    if not (0 <= layer_idx < num_layers):
        raise IndexError(
            f"layer_idx={layer_idx} is out of range for a model with "
            f"{num_layers} layers (valid range: 0–{num_layers - 1})."
        )

    return layers[layer_idx]


def normalize_vector(
    vec: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    """
    L2-normalize a concept vector to unit length.

    Normalizing before injection decouples direction from magnitude,
    making the steering coefficient ``alpha`` the sole intensity controller.

    Parameters
    ----------
    vec : torch.Tensor
        The concept vector to normalize. Shape: ``(hidden_dim,)``.
    eps : float, default=1e-9
        Small epsilon to prevent division-by-zero.

    Returns
    -------
    torch.Tensor
        Unit-norm vector of the same shape and dtype.
    """
    norm = torch.norm(vec.to(torch.float32)) + eps
    return (vec.to(torch.float32) / norm).to(vec.dtype)


def compute_layer_weights(
    layers: List[int],
    base_alpha: float = 2.0,
    strategy: str = "uniform",
) -> Dict[int, float]:
    """
    Compute per-layer steering coefficients (alpha_i) for adaptive multi-layer steering.

    Supported strategies:
    - "uniform": All target layers receive base_alpha (alpha_i = base_alpha).
    - "linear_decay": Steering strength decays linearly across target layers from base_alpha to 0.0.
    - "cosine_decay": Steering strength follows a smooth cosine curve from base_alpha to 0.0.

    Parameters
    ----------
    layers : List[int]
        List of target layer indices.
    base_alpha : float, default=2.0
        Base steering intensity coefficient.
    strategy : str, default="uniform"
        Weighting strategy ("uniform", "linear_decay", "cosine_decay").

    Returns
    -------
    Dict[int, float]
        Mapping from layer_idx to its layer-specific steering coefficient (alpha_i).

    Raises
    ------
    ValueError
        If an unrecognized weighting strategy is specified.
    """
    if not layers:
        return {}

    sorted_layers = sorted(layers)
    num_layers = len(sorted_layers)
    strat = strategy.lower().strip()

    weights: Dict[int, float] = {}

    if strat == "uniform":
        for layer in sorted_layers:
            weights[layer] = float(base_alpha)
    elif strat in ("linear_decay", "linear"):
        if num_layers == 1:
            weights[sorted_layers[0]] = float(base_alpha)
        else:
            for i, layer in enumerate(sorted_layers):
                t = i / (num_layers - 1)
                w = 1.0 - t
                weights[layer] = float(base_alpha) * w
    elif strat in ("cosine_decay", "cosine"):
        if num_layers == 1:
            weights[sorted_layers[0]] = float(base_alpha)
        else:
            for i, layer in enumerate(sorted_layers):
                t = i / (num_layers - 1)
                w = 0.5 * (1.0 + math.cos(math.pi * t))
                weights[layer] = float(base_alpha) * w
    else:
        raise ValueError(
            f"Unknown weighting strategy '{strategy}'. "
            "Supported strategies: 'uniform', 'linear_decay', 'cosine_decay'."
        )

    return weights


