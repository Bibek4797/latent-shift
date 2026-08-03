"""
src/concept_extractors.py
-------------------------
Research-grade concept vector extraction methods.

Provides 7 extraction strategies:
1. **Mean Difference** (`mean_diff`): Centroid vector difference.
2. **PCA** (`pca`): First principal component of paired differences.
3. **Linear Discriminant Analysis** (`lda`): Maximizes between-class to within-class scatter.
4. **Logistic Regression** (`logistic_regression`): Hyperplane normal from log-loss classification.
5. **Linear SVM** (`linear_svm`): Maximum-margin separating hyperplane normal.
6. **Sparse PCA** (`sparse_pca`): L1-regularized principal component for sparse features.
7. **Truncated SVD** (`truncated_svd`): Top right singular vector of mean-centered data.

Includes benchmarking and Plotly visualizers for comparative analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.utils import get_logger, normalize_vector

logger = get_logger(__name__)


# ===========================================================================
# ABSTRACT BASE CLASS & EXTRACTORS
# ===========================================================================

class BaseConceptExtractor(ABC):
    """Abstract Base Class for concept vector extraction methods."""

    @property
    @abstractmethod
    def method_key(self) -> str:
        """Unique identifier key for the method."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name."""
        pass

    @abstractmethod
    def extract(
        self,
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        normalize: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Extract concept vector from positive and negative activations.

        Parameters
        ----------
        pos_activations : torch.Tensor
            Shape: (num_samples, hidden_dim).
        neg_activations : torch.Tensor
            Shape: (num_samples, hidden_dim).
        normalize : bool, default=False
            If True, L2-normalizes the vector before returning.

        Returns
        -------
        torch.Tensor
            Concept vector of shape (hidden_dim,).
        """
        pass

    @staticmethod
    def align_sign(vec: torch.Tensor, pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> torch.Tensor:
        """
        Ensure vector direction aligns with (mean_pos - mean_neg) so positive alpha steers towards positive concept.
        """
        mean_diff = (pos_activations - neg_activations).to(torch.float32).mean(dim=0)
        cos_sim = torch.dot(vec.to(torch.float32), mean_diff)
        if cos_sim < 0:
            vec = -vec
        return vec


class MeanDiffExtractor(BaseConceptExtractor):
    method_key = "mean_diff"
    display_name = "Mean Difference"

    def extract(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor, normalize: bool = False, **kwargs) -> torch.Tensor:
        orig_dtype = pos_activations.dtype
        mean_pos = pos_activations.to(torch.float32).mean(dim=0)
        mean_neg = neg_activations.to(torch.float32).mean(dim=0)
        vec = mean_pos - mean_neg
        if normalize:
            vec = normalize_vector(vec)
        return vec.to(orig_dtype)


class PCAExtractor(BaseConceptExtractor):
    method_key = "pca"
    display_name = "PCA (First Principal Component)"

    def extract(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor, normalize: bool = False, **kwargs) -> torch.Tensor:
        orig_dtype = pos_activations.dtype
        min_samples = min(pos_activations.shape[0], neg_activations.shape[0])
        diffs_np = (pos_activations[:min_samples] - neg_activations[:min_samples]).detach().cpu().to(torch.float32).numpy()

        pca = PCA(n_components=1)
        pca.fit(diffs_np)

        pca_vec = torch.tensor(pca.components_[0], dtype=torch.float32)
        pca_vec = self.align_sign(pca_vec, pos_activations, neg_activations)
        if normalize:
            pca_vec = normalize_vector(pca_vec)
        return pca_vec.to(orig_dtype)


class LDAExtractor(BaseConceptExtractor):
    method_key = "lda"
    display_name = "Linear Discriminant Analysis (LDA)"

    def extract(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor, normalize: bool = False, **kwargs) -> torch.Tensor:
        orig_dtype = pos_activations.dtype
        pos_np = pos_activations.detach().cpu().to(torch.float32).numpy()
        neg_np = neg_activations.detach().cpu().to(torch.float32).numpy()

        X = np.vstack([pos_np, neg_np])
        y = np.hstack([np.ones(pos_np.shape[0]), np.zeros(neg_np.shape[0])])

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X, y)

        vec = torch.tensor(lda.coef_[0], dtype=torch.float32)
        vec = self.align_sign(vec, pos_activations, neg_activations)
        if normalize:
            vec = normalize_vector(vec)
        return vec.to(orig_dtype)


class LogisticRegressionExtractor(BaseConceptExtractor):
    method_key = "logistic_regression"
    display_name = "Logistic Regression Direction"

    def extract(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor, normalize: bool = False, C: float = 1.0, **kwargs) -> torch.Tensor:
        orig_dtype = pos_activations.dtype
        pos_np = pos_activations.detach().cpu().to(torch.float32).numpy()
        neg_np = neg_activations.detach().cpu().to(torch.float32).numpy()

        X = np.vstack([pos_np, neg_np])
        y = np.hstack([np.ones(pos_np.shape[0]), np.zeros(neg_np.shape[0])])

        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        clf.fit(X, y)

        vec = torch.tensor(clf.coef_[0], dtype=torch.float32)
        vec = self.align_sign(vec, pos_activations, neg_activations)
        if normalize:
            vec = normalize_vector(vec)
        return vec.to(orig_dtype)


class LinearSVMExtractor(BaseConceptExtractor):
    method_key = "linear_svm"
    display_name = "Linear SVM Hyperplane Normal"

    def extract(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor, normalize: bool = False, C: float = 1.0, **kwargs) -> torch.Tensor:
        orig_dtype = pos_activations.dtype
        pos_np = pos_activations.detach().cpu().to(torch.float32).numpy()
        neg_np = neg_activations.detach().cpu().to(torch.float32).numpy()

        X = np.vstack([pos_np, neg_np])
        y = np.hstack([np.ones(pos_np.shape[0]), -np.ones(neg_np.shape[0])])

        svm = SVC(kernel="linear", C=C)
        svm.fit(X, y)

        vec = torch.tensor(svm.coef_[0], dtype=torch.float32)
        vec = self.align_sign(vec, pos_activations, neg_activations)
        if normalize:
            vec = normalize_vector(vec)
        return vec.to(orig_dtype)


class SparsePCAExtractor(BaseConceptExtractor):
    method_key = "sparse_pca"
    display_name = "Sparse PCA"

    def extract(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor, normalize: bool = False, alpha: float = 1.0, **kwargs) -> torch.Tensor:
        orig_dtype = pos_activations.dtype
        min_samples = min(pos_activations.shape[0], neg_activations.shape[0])
        diffs_np = (pos_activations[:min_samples] - neg_activations[:min_samples]).detach().cpu().to(torch.float32).numpy()

        spca = SparsePCA(n_components=1, alpha=alpha, random_state=42)
        spca.fit(diffs_np)

        vec = torch.tensor(spca.components_[0], dtype=torch.float32)
        vec = self.align_sign(vec, pos_activations, neg_activations)
        if normalize:
            vec = normalize_vector(vec)
        return vec.to(orig_dtype)


class TruncatedSVDExtractor(BaseConceptExtractor):
    method_key = "truncated_svd"
    display_name = "Truncated SVD"

    def extract(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor, normalize: bool = False, **kwargs) -> torch.Tensor:
        orig_dtype = pos_activations.dtype
        pos_np = pos_activations.detach().cpu().to(torch.float32).numpy()
        neg_np = neg_activations.detach().cpu().to(torch.float32).numpy()

        X = np.vstack([pos_np, -neg_np])
        svd = TruncatedSVD(n_components=1, random_state=42)
        svd.fit(X)

        vec = torch.tensor(svd.components_[0], dtype=torch.float32)
        vec = self.align_sign(vec, pos_activations, neg_activations)
        if normalize:
            vec = normalize_vector(vec)
        return vec.to(orig_dtype)


# Registry mapping method key -> Extractor class
EXTRACTOR_REGISTRY: Dict[str, BaseConceptExtractor] = {
    "mean_diff": MeanDiffExtractor(),
    "pca": PCAExtractor(),
    "lda": LDAExtractor(),
    "logistic_regression": LogisticRegressionExtractor(),
    "linear_svm": LinearSVMExtractor(),
    "sparse_pca": SparsePCAExtractor(),
    "truncated_svd": TruncatedSVDExtractor(),
}


# ===========================================================================
# BENCHMARK & COMPARISON UTILITIES
# ===========================================================================

@dataclass
class ExtractorBenchmarkResult:
    """Benchmark metrics for a single extraction method."""
    method_key: str
    display_name: str
    vector: torch.Tensor
    vector_norm: float
    runtime_ms: float
    memory_kb: float


class ConceptVectorComparer:
    """
    Comparison framework to benchmark and visualize multiple extraction methods simultaneously.
    """

    @staticmethod
    def benchmark_all_methods(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        normalize: bool = False,
    ) -> Dict[str, ExtractorBenchmarkResult]:
        """
        Run all 7 concept extractors and measure runtime, memory usage, and vector norms.

        Parameters
        ----------
        pos_activations : torch.Tensor
        neg_activations : torch.Tensor
        normalize : bool, default=False

        Returns
        -------
        Dict[str, ExtractorBenchmarkResult]
            Benchmark results keyed by method_key.
        """
        results = {}
        for key, extractor in EXTRACTOR_REGISTRY.items():
            start_time = time.perf_counter()
            mem_before = pos_activations.element_size() * pos_activations.nelement()

            vec = extractor.extract(pos_activations, neg_activations, normalize=normalize)

            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            mem_kb = (vec.element_size() * vec.nelement()) / 1024.0
            v_norm = float(torch.norm(vec.to(torch.float32)).item())

            results[key] = ExtractorBenchmarkResult(
                method_key=key,
                display_name=extractor.display_name,
                vector=vec,
                vector_norm=round(v_norm, 4),
                runtime_ms=round(runtime_ms, 3),
                memory_kb=round(mem_kb, 2),
            )
        return results

    @staticmethod
    def compute_pairwise_cosine_matrix(
        benchmark_results: Dict[str, ExtractorBenchmarkResult]
    ) -> Tuple[List[str], np.ndarray]:
        """
        Compute NxN pairwise cosine similarity matrix across all benchmarked methods.
        """
        labels = [res.display_name for res in benchmark_results.values()]
        keys = list(benchmark_results.keys())
        N = len(keys)
        matrix = np.zeros((N, N))

        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                v1 = benchmark_results[k1].vector.to(torch.float32).flatten()
                v2 = benchmark_results[k2].vector.to(torch.float32).flatten()
                n1, n2 = torch.norm(v1), torch.norm(v2)
                sim = (torch.dot(v1, v2) / (n1 * n2 + 1e-9)).item() if n1 > 0 and n2 > 0 else 0.0
                matrix[i, j] = round(sim, 4)

        return labels, matrix


# ===========================================================================
# PLOTTING VISUALIZERS
# ===========================================================================

def plot_pairwise_cosine_heatmap(labels: List[str], matrix: np.ndarray):
    """
    Generate interactive Plotly heatmap for pairwise cosine similarity matrix across methods.
    """
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=matrix,
        texttemplate="%{text:.3f}",
        zmin=-1.0,
        zmax=1.0
    ))
    fig.update_layout(
        title="Pairwise Cosine Similarity Between Concept Extractors",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_runtime_comparison(benchmark_results: Dict[str, ExtractorBenchmarkResult]):
    """
    Generate interactive Plotly bar chart comparing extraction runtime (ms).
    """
    import plotly.graph_objects as go

    names = [r.display_name for r in benchmark_results.values()]
    runtimes = [r.runtime_ms for r in benchmark_results.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=runtimes,
        marker=dict(color='rgb(124, 58, 237)'),
        text=[f"{rt:.2f} ms" for rt in runtimes],
        textposition='auto',
    ))
    fig.update_layout(
        title="Concept Extraction Execution Runtime (ms)",
        xaxis_title="Extraction Method",
        yaxis_title="Runtime (milliseconds)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_memory_comparison(benchmark_results: Dict[str, ExtractorBenchmarkResult]):
    """
    Generate interactive Plotly bar chart comparing memory consumption (KB).
    """
    import plotly.graph_objects as go

    names = [r.display_name for r in benchmark_results.values()]
    mems = [r.memory_kb for r in benchmark_results.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=mems,
        marker=dict(color='rgb(79, 70, 229)'),
        text=[f"{m:.1f} KB" for m in mems],
        textposition='auto',
    ))
    fig.update_layout(
        title="Concept Vector Memory Footprint (KB)",
        xaxis_title="Extraction Method",
        yaxis_title="Memory (KB)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_vector_magnitude_comparison(benchmark_results: Dict[str, ExtractorBenchmarkResult]):
    """
    Generate interactive Plotly bar chart comparing unnormalized vector L2 norms.
    """
    import plotly.graph_objects as go

    names = [r.display_name for r in benchmark_results.values()]
    norms = [r.vector_norm for r in benchmark_results.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=norms,
        marker=dict(color='rgb(16, 185, 129)'),
        text=[f"{n:.3f}" for n in norms],
        textposition='auto',
    ))
    fig.update_layout(
        title="Concept Vector L2 Magnitude Comparison",
        xaxis_title="Extraction Method",
        yaxis_title="Vector L2 Norm",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
