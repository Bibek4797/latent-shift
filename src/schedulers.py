"""
src/schedulers.py
-----------------
Dynamic closed-loop alpha schedulers for activation steering.

Instead of a fixed steering coefficient alpha throughout generation, these
schedulers adapt alpha on a per-token basis. Four built-in strategies are
provided, plus an extensible base class for custom schedulers.

Built-in schedulers:
    1. **LinearScheduler**: Linearly interpolates alpha from start to end.
    2. **CosineScheduler**: Smoothly decays alpha following a half-cosine.
    3. **ConfidenceBasedScheduler**: Reduces alpha when the model is confident
       (low entropy in top-k logits) and increases it when uncertain.
    4. **EntropyBasedScheduler**: Adjusts alpha proportional to token entropy,
       steering harder when the model's output distribution is diffuse.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils import get_logger

logger = get_logger(__name__)


# ===========================================================================
# ABSTRACT BASE
# ===========================================================================

class BaseAlphaScheduler(ABC):
    """
    Abstract base class for dynamic alpha schedulers.

    Subclasses must implement ``step()`` which is called once per generated
    token and returns the alpha value for that decoding step.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scheduler name."""
        pass

    @abstractmethod
    def step(
        self,
        step_idx: int,
        total_steps: int,
        logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> float:
        """
        Compute the alpha coefficient for the current decoding step.

        Parameters
        ----------
        step_idx : int
            Current token index (0-based).
        total_steps : int
            Total number of tokens to generate (``max_new_tokens``).
        logits : torch.Tensor, optional
            Raw logits from the model's last forward pass. Shape: ``(vocab_size,)``.
            Available for confidence/entropy-based schedulers.

        Returns
        -------
        float
            The steering alpha for this decoding step.
        """
        pass

    def reset(self) -> None:
        """Reset any internal state. Called before a new generation run."""
        pass


# ===========================================================================
# BUILT-IN SCHEDULERS
# ===========================================================================

class FixedScheduler(BaseAlphaScheduler):
    """Constant alpha — reproduces the legacy fixed-alpha behaviour."""

    name = "fixed"

    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def step(self, step_idx: int, total_steps: int, logits=None, **kw) -> float:
        return self.alpha


class LinearScheduler(BaseAlphaScheduler):
    """
    Linearly interpolate alpha from ``alpha_start`` to ``alpha_end``.

    .. math::
        \\alpha_t = \\alpha_{\\text{start}} + (\\alpha_{\\text{end}} - \\alpha_{\\text{start}}) \\cdot \\frac{t}{T-1}
    """

    name = "linear"

    def __init__(self, alpha_start: float = 3.0, alpha_end: float = 0.5):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def step(self, step_idx: int, total_steps: int, logits=None, **kw) -> float:
        if total_steps <= 1:
            return self.alpha_start
        t = step_idx / (total_steps - 1)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * t


class CosineScheduler(BaseAlphaScheduler):
    """
    Half-cosine annealing from ``alpha_max`` to ``alpha_min``.

    .. math::
        \\alpha_t = \\alpha_{\\min} + \\frac{\\alpha_{\\max} - \\alpha_{\\min}}{2}
                    \\left(1 + \\cos\\left(\\pi \\frac{t}{T-1}\\right)\\right)
    """

    name = "cosine"

    def __init__(self, alpha_max: float = 3.0, alpha_min: float = 0.3):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min

    def step(self, step_idx: int, total_steps: int, logits=None, **kw) -> float:
        if total_steps <= 1:
            return self.alpha_max
        t = step_idx / (total_steps - 1)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * 0.5 * (1.0 + math.cos(math.pi * t))


class ConfidenceBasedScheduler(BaseAlphaScheduler):
    """
    Adjust alpha based on model confidence (negative entropy of top-k logits).

    When the model is confident (peaked distribution), alpha is reduced;
    when uncertain (flat distribution), alpha is increased to provide
    stronger concept guidance.

    .. math::
        H_t = -\\sum_{i=1}^{k} p_i \\log p_i, \\qquad
        \\alpha_t = \\alpha_{\\text{base}} \\cdot \\left(\\frac{H_t}{\\log k}\\right)^{\\gamma}

    where :math:`\\gamma` controls sensitivity.
    """

    name = "confidence"

    def __init__(self, alpha_base: float = 2.0, gamma: float = 1.0, top_k: int = 50):
        self.alpha_base = alpha_base
        self.gamma = gamma
        self.top_k = top_k

    def step(self, step_idx: int, total_steps: int, logits=None, **kw) -> float:
        if logits is None:
            return self.alpha_base

        logits_1d = logits.detach().float()
        if logits_1d.dim() > 1:
            logits_1d = logits_1d[-1]

        k = min(self.top_k, logits_1d.shape[-1])
        top_logits = torch.topk(logits_1d, k).values
        probs = torch.softmax(top_logits, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        max_entropy = math.log(k) if k > 1 else 1.0

        normalised = entropy / max_entropy
        scale = normalised ** self.gamma
        return self.alpha_base * scale


class EntropyBasedScheduler(BaseAlphaScheduler):
    """
    Scale alpha proportional to full-vocabulary token entropy.

    High entropy (diffuse distribution) → steer harder.
    Low entropy (peaked distribution) → steer softer.

    .. math::
        H_t = -\\sum_{v} p_v \\log p_v, \\qquad
        \\alpha_t = \\alpha_{\\min} + (\\alpha_{\\max} - \\alpha_{\\min})
                    \\cdot \\min\\left(1, \\frac{H_t}{H_{\\text{ref}}}\\right)

    where :math:`H_{\\text{ref}}` is a reference entropy (default ``\\ln(1000) \\approx 6.9``).
    """

    name = "entropy"

    def __init__(self, alpha_min: float = 0.5, alpha_max: float = 3.0, h_ref: float = 6.9):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.h_ref = h_ref

    def step(self, step_idx: int, total_steps: int, logits=None, **kw) -> float:
        if logits is None:
            return (self.alpha_min + self.alpha_max) / 2.0

        logits_1d = logits.detach().float()
        if logits_1d.dim() > 1:
            logits_1d = logits_1d[-1]

        probs = torch.softmax(logits_1d, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

        ratio = min(1.0, entropy / self.h_ref)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * ratio


# ===========================================================================
# REGISTRY & FACTORY
# ===========================================================================

SCHEDULER_REGISTRY: Dict[str, type] = {
    "fixed": FixedScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
    "confidence": ConfidenceBasedScheduler,
    "entropy": EntropyBasedScheduler,
}


def build_scheduler(name: str, **kwargs) -> BaseAlphaScheduler:
    """
    Build a scheduler by name from the global registry.

    Parameters
    ----------
    name : str
        Scheduler key: "fixed", "linear", "cosine", "confidence", "entropy".
    **kwargs
        Forwarded to the scheduler constructor.

    Returns
    -------
    BaseAlphaScheduler

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.
    """
    key = name.lower().strip()
    if key not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return SCHEDULER_REGISTRY[key](**kwargs)


# ===========================================================================
# ALPHA TRAJECTORY LOG
# ===========================================================================

@dataclass
class AlphaTrajectory:
    """Records the per-token alpha values produced during a generation run."""
    scheduler_name: str = ""
    values: List[float] = field(default_factory=list)

    def record(self, alpha: float) -> None:
        self.values.append(alpha)

    def to_dict(self) -> dict:
        return {
            "scheduler_name": self.scheduler_name,
            "num_steps": len(self.values),
            "alpha_mean": round(float(np.mean(self.values)), 4) if self.values else 0.0,
            "alpha_min": round(float(np.min(self.values)), 4) if self.values else 0.0,
            "alpha_max": round(float(np.max(self.values)), 4) if self.values else 0.0,
            "alpha_std": round(float(np.std(self.values)), 4) if self.values else 0.0,
            "values": [round(v, 4) for v in self.values],
        }


# ===========================================================================
# PLOTLY VISUALISERS
# ===========================================================================

def plot_alpha_trajectory(trajectory: AlphaTrajectory):
    """Line chart of alpha evolution over token generation steps."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(trajectory.values))),
        y=trajectory.values,
        mode="lines+markers",
        name=f"α ({trajectory.scheduler_name})",
        line=dict(color="rgb(124, 58, 237)", width=3),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=f"Dynamic Alpha Evolution ({trajectory.scheduler_name})",
        xaxis_title="Token Step",
        yaxis_title="α Value",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_token_steering_strength(trajectory: AlphaTrajectory, tokens: Optional[List[str]] = None):
    """Bar chart showing per-token steering strength (alpha × normalised position)."""
    import plotly.graph_objects as go

    n = len(trajectory.values)
    labels = tokens[:n] if tokens and len(tokens) >= n else [f"t{i}" for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=trajectory.values,
        marker=dict(
            color=trajectory.values,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="α"),
        ),
        text=[f"{v:.2f}" for v in trajectory.values],
        textposition="auto",
    ))
    fig.update_layout(
        title="Token-wise Steering Strength",
        xaxis_title="Generated Token",
        yaxis_title="α Value",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig
