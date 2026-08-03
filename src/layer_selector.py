"""
src/layer_selector.py
----------------------
Automatic Layer Selection framework for activation steering.

Provides statistical scoring methods to automatically discover optimal target layers
for concept vector injection without manual guesswork.

Scoring Methods:
- **Mean Activation Separation**: Euclidean distance between positive and negative centroids.
- **Cosine Separation**: Directional divergence (1 - cosine_similarity) between centroids.
- **Fisher Score**: Ratio of between-class variance to within-class variance.
- **Signal-to-Noise Ratio (SNR)**: Mean contrastive difference magnitude over variation standard deviation.
- **Activation Variance**: Total activation variance across representations.

Visualizations:
- `plot_layer_scores_line()`: Line plot of layer score trajectory across model depth.
- `plot_layer_scores_heatmap()`: Heatmap comparing normalized scores across all 5 methods.
- `plot_top_k_layers_bar()`: Bar chart highlighting Top-K selected layers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LayerScoreResult:
    """Dataclass encapsulating scoring and ranking results for a single layer."""
    layer_idx: int
    score: float
    rank: int


class LayerSelector:
    """
    Statistically rank and select optimal transformer layers for activation steering.
    """

    @staticmethod
    def compute_mean_separation(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> float:
        """
        Compute Mean Activation Separation: ||mean(pos) - mean(neg)||_2.

        Parameters
        ----------
        pos_acts : torch.Tensor
            Positive concept activations. Shape: (num_prompts, hidden_dim).
        neg_acts : torch.Tensor
            Negative concept activations. Shape: (num_prompts, hidden_dim).

        Returns
        -------
        float
            Euclidean distance between positive and negative activation centroids.
        """
        pos_mean = pos_acts.to(torch.float32).mean(dim=0)
        neg_mean = neg_acts.to(torch.float32).mean(dim=0)
        return float(torch.norm(pos_mean - neg_mean).item())

    @staticmethod
    def compute_cosine_separation(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> float:
        """
        Compute Cosine Separation: 1 - cos(mean(pos), mean(neg)).

        Parameters
        ----------
        pos_acts : torch.Tensor
        neg_acts : torch.Tensor

        Returns
        -------
        float
            Directional separation score in [0.0, 2.0].
        """
        pos_mean = pos_acts.to(torch.float32).mean(dim=0)
        neg_mean = neg_acts.to(torch.float32).mean(dim=0)
        n_pos, n_neg = torch.norm(pos_mean), torch.norm(neg_mean)
        if n_pos == 0.0 or n_neg == 0.0:
            return 0.0
        cos_sim = (torch.dot(pos_mean, neg_mean) / (n_pos * n_neg)).item()
        return float(1.0 - cos_sim)

    @staticmethod
    def compute_fisher_score(
        pos_acts: torch.Tensor, neg_acts: torch.Tensor, eps: float = 1e-9
    ) -> float:
        """
        Compute Fisher Score: ||mean(pos) - mean(neg)||_2^2 / (var(pos) + var(neg) + eps).

        Parameters
        ----------
        pos_acts : torch.Tensor
        neg_acts : torch.Tensor
        eps : float, default=1e-9

        Returns
        -------
        float
            Fisher ratio score.
        """
        pos = pos_acts.to(torch.float32)
        neg = neg_acts.to(torch.float32)
        pos_mean, neg_mean = pos.mean(dim=0), neg.mean(dim=0)
        between_var = torch.norm(pos_mean - neg_mean).item() ** 2

        pos_var = pos.var(dim=0, unbiased=False).sum().item() if pos.shape[0] > 1 else 0.0
        neg_var = neg.var(dim=0, unbiased=False).sum().item() if neg.shape[0] > 1 else 0.0
        within_var = pos_var + neg_var

        return float(between_var / (within_var + eps))

    @staticmethod
    def compute_snr(
        pos_acts: torch.Tensor, neg_acts: torch.Tensor, eps: float = 1e-9
    ) -> float:
        """
        Compute Signal-to-Noise Ratio (SNR): ||mean(diff)||_2 / (std(diff) + eps).

        Parameters
        ----------
        pos_acts : torch.Tensor
        neg_acts : torch.Tensor
        eps : float, default=1e-9

        Returns
        -------
        float
            Signal-to-noise ratio score.
        """
        pos = pos_acts.to(torch.float32)
        neg = neg_acts.to(torch.float32)
        min_n = min(pos.shape[0], neg.shape[0])
        diff = pos[:min_n] - neg[:min_n]
        mean_diff_norm = torch.norm(diff.mean(dim=0)).item()
        std_diff = diff.std(dim=0, unbiased=False).mean().item() if min_n > 1 else eps
        return float(mean_diff_norm / (std_diff + eps))

    @staticmethod
    def compute_activation_variance(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> float:
        """
        Compute Total Activation Variance across positive and negative representations.

        Parameters
        ----------
        pos_acts : torch.Tensor
        neg_acts : torch.Tensor

        Returns
        -------
        float
            Total variance sum across dimensions.
        """
        all_acts = torch.cat([pos_acts.to(torch.float32), neg_acts.to(torch.float32)], dim=0)
        if all_acts.shape[0] <= 1:
            return 0.0
        var_total = all_acts.var(dim=0, unbiased=False).sum().item()
        return float(var_total)

    @staticmethod
    def score_layers(
        pos_acts_dict: Dict[int, torch.Tensor],
        neg_acts_dict: Dict[int, torch.Tensor],
        method: str = "mean_separation",
    ) -> Dict[int, float]:
        """
        Compute layer scores for all layers in the activation dictionary.

        Parameters
        ----------
        pos_acts_dict : Dict[int, torch.Tensor]
        neg_acts_dict : Dict[int, torch.Tensor]
        method : str, default="mean_separation"
            Scoring method ("mean_separation", "cosine_separation", "fisher_score",
            "snr", "activation_variance").

        Returns
        -------
        Dict[int, float]
            Mapping from layer index to calculated score.

        Raises
        ------
        ValueError
            If an unsupported scoring method is specified.
        """
        method_clean = method.lower().strip()
        scoring_funcs = {
            "mean_separation": LayerSelector.compute_mean_separation,
            "cosine_separation": LayerSelector.compute_cosine_separation,
            "fisher_score": LayerSelector.compute_fisher_score,
            "snr": LayerSelector.compute_snr,
            "activation_variance": LayerSelector.compute_activation_variance,
        }

        if method_clean not in scoring_funcs:
            raise ValueError(
                f"Unknown scoring method '{method}'. "
                f"Supported methods: {list(scoring_funcs.keys())}"
            )

        fn = scoring_funcs[method_clean]
        scores: Dict[int, float] = {}

        for layer in sorted(pos_acts_dict.keys()):
            if layer in neg_acts_dict:
                scores[layer] = round(fn(pos_acts_dict[layer], neg_acts_dict[layer]), 6)

        logger.info("Scored %d layers using method '%s'", len(scores), method)
        return scores

    @staticmethod
    def rank_layers(scores_dict: Dict[int, float]) -> List[LayerScoreResult]:
        """
        Rank layers by score in descending order.

        Parameters
        ----------
        scores_dict : Dict[int, float]

        Returns
        -------
        List[LayerScoreResult]
            Sorted list of LayerScoreResult dataclasses.
        """
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        results = [
            LayerScoreResult(layer_idx=layer, score=score, rank=rank + 1)
            for rank, (layer, score) in enumerate(sorted_items)
        ]
        return results

    @staticmethod
    def select_top_k_layers(
        scores_dict: Dict[int, float], k: int = 3, preserve_order: bool = True
    ) -> List[int]:
        """
        Select Top-K highest scoring layer indices.

        Parameters
        ----------
        scores_dict : Dict[int, float]
        k : int, default=3
            Number of top layers to select.
        preserve_order : bool, default=True
            If True, returns layer indices sorted in architectural order (ascending).

        Returns
        -------
        List[int]
            List of selected layer indices.
        """
        ranked = LayerSelector.rank_layers(scores_dict)
        k_capped = min(k, len(ranked))
        top_k_ranked = ranked[:k_capped]
        top_layers = [r.layer_idx for r in top_k_ranked]
        return sorted(top_layers) if preserve_order else top_layers


# ===========================================================================
# VISUALIZATION UTILITIES
# ===========================================================================

def plot_layer_scores_line(scores_dict: Dict[int, float], method_name: str = "Scoring Method"):
    """
    Generate interactive Plotly line chart of layer scores across model depth.
    """
    import plotly.graph_objects as go

    layers = sorted(scores_dict.keys())
    scores = [scores_dict[l] for l in layers]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers,
        y=scores,
        mode='lines+markers',
        marker=dict(color='rgb(124, 58, 237)', size=8),
        line=dict(color='rgb(124, 58, 237)', width=3),
        name=method_name
    ))
    fig.update_layout(
        title=f"Layer Scores Across Model Depth ({method_name})",
        xaxis_title="Layer Index",
        yaxis_title="Layer Score",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_layer_scores_heatmap(all_method_scores: Dict[str, Dict[int, float]]):
    """
    Generate interactive Plotly heatmap comparing min-max normalized scores across methods.
    """
    import plotly.graph_objects as go

    methods = list(all_method_scores.keys())
    if not methods:
        return go.Figure()

    first_method = methods[0]
    layers = sorted(all_method_scores[first_method].keys())

    matrix = []
    for m in methods:
        row = []
        scores = all_method_scores[m]
        max_s = max(scores.values()) if scores.values() and max(scores.values()) > 0 else 1.0
        for l in layers:
            row.append(scores.get(l, 0.0) / max_s)
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=layers,
        y=methods,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title="Normalized Layer Scores Comparison Across Methods",
        xaxis_title="Layer Index",
        yaxis_title="Scoring Method",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_top_k_layers_bar(scores_dict: Dict[int, float], k: int = 3, method_name: str = "Scoring Method"):
    """
    Generate interactive Plotly bar chart highlighting Top-K selected layers.
    """
    import plotly.graph_objects as go

    ranked = LayerSelector.rank_layers(scores_dict)[:k]
    layers = [f"Layer {r.layer_idx}" for r in ranked]
    scores = [r.score for r in ranked]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers,
        y=scores,
        marker=dict(color='rgb(16, 185, 129)'),
        text=[f"Rank #{r.rank} ({r.score:.4f})" for r in ranked],
        textposition='auto'
    ))
    fig.update_layout(
        title=f"Top-{min(k, len(ranked))} Selected Layers ({method_name})",
        xaxis_title="Transformer Layer",
        yaxis_title="Score",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
