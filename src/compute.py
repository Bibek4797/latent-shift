"""
src/compute.py
--------------
Concept vector computation, persistence, and loading.

Implements two extraction strategies:

1. **Mean Difference**: The centroid of positive activations minus the centroid
   of negative activations.  Simple, interpretable, effective.

2. **PCA (First Principal Component)**: Fits PCA on the element-wise differences
   and extracts the direction of maximal variance.  Acts as a denoiser when
   contrastive prompts introduce spurious variability.

Each saved file is a torch checkpoint containing both the vectors *and* metadata
(model name, method, target layers, timestamp) for full reproducibility.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.utils import get_logger, normalize_vector

logger = get_logger(__name__)


class ConceptVectorEngine:
    """
    Compute, persist, and load steering concept vectors.

    All methods are static so the class acts as a stateless namespace.
    Includes an in-memory cache to avoid redundant disk I/O.
    """
    _vector_cache: Dict[str, Dict[int, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_vector(
        method: str,
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        normalize: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute concept vector using the specified extraction method.

        Parameters
        ----------
        method : str
            Extraction method key ("mean_diff", "pca", "lda", "logistic_regression",
            "linear_svm", "sparse_pca", "truncated_svd").
        pos_activations : torch.Tensor
        neg_activations : torch.Tensor
        normalize : bool, default=False

        Returns
        -------
        torch.Tensor
            Extracted concept vector of shape (hidden_dim,).
        """
        from src.concept_extractors import EXTRACTOR_REGISTRY

        method_clean = method.lower().strip()
        if method_clean not in EXTRACTOR_REGISTRY:
            raise ValueError(
                f"Unknown extraction method '{method}'. "
                f"Supported methods: {list(EXTRACTOR_REGISTRY.keys())}"
            )

        extractor = EXTRACTOR_REGISTRY[method_clean]
        return extractor.extract(pos_activations, neg_activations, normalize=normalize, **kwargs)

    @staticmethod
    def compute_mean_difference(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Compute mean difference vector v = mean(h_pos) - mean(h_neg)."""
        return ConceptVectorEngine.compute_vector("mean_diff", pos_activations, neg_activations, normalize=normalize)

    @staticmethod
    def compute_pca_vector(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Compute first principal component concept vector."""
        return ConceptVectorEngine.compute_vector("pca", pos_activations, neg_activations, normalize=normalize)


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_vectors(
        vectors: Dict[int, torch.Tensor],
        data_dir: str,
        filename: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save concept vectors and optional metadata to a ``.pt`` checkpoint.

        The checkpoint structure is::

            {
                "vectors": {layer_idx: tensor, ...},
                "metadata": {
                    "model_name": ...,
                    "method": ...,
                    "layers": [...],
                    "timestamp": "YYYY-MM-DD HH:MM:SS",
                    ...   # any extra keys from the caller
                }
            }

        Parameters
        ----------
        vectors : Dict[int, torch.Tensor]
            Mapping of layer index → concept vector.
        data_dir : str
            Directory path for saving.  Created if absent.
        filename : str
            Output filename (e.g. ``"safety_mean_diff.pt"``).
        metadata : dict, optional
            Extra key-value pairs to store alongside the vectors.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, filename)

        cpu_vectors = {
            layer: vec.cpu().detach().float()
            for layer, vec in vectors.items()
        }

        meta = metadata or {}
        meta.setdefault("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        meta.setdefault("layers", sorted(cpu_vectors.keys()))

        checkpoint = {"vectors": cpu_vectors, "metadata": meta}
        torch.save(checkpoint, file_path)

        # Cache in memory
        abs_path = os.path.abspath(file_path)
        ConceptVectorEngine._vector_cache[abs_path] = cpu_vectors

        logger.info("Vectors saved | path=%s | layers=%s", abs_path, list(cpu_vectors.keys()))
        return abs_path

    @staticmethod
    def load_vectors(file_path: str) -> Dict[int, torch.Tensor]:
        """
        Load concept vectors from a saved checkpoint.

        Handles both the new checkpoint format (dict with ``"vectors"`` key)
        and the legacy format (raw ``{layer: tensor}`` dict) for backwards
        compatibility.

        Parameters
        ----------
        file_path : str
            Path to the ``.pt`` file.

        Returns
        -------
        Dict[int, torch.Tensor]
            Mapping of layer index → concept vector.

        Raises
        ------
        FileNotFoundError
            If ``file_path`` does not exist.
        """
        abs_path = os.path.abspath(file_path)
        if abs_path in ConceptVectorEngine._vector_cache:
            logger.debug("Vectors loaded from memory cache | path=%s", abs_path)
            return ConceptVectorEngine._vector_cache[abs_path]

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Concept vector file not found: {file_path}"
            )

        # weights_only=False required because we store a mixed dict with tensors + metadata
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        # New format
        if isinstance(checkpoint, dict) and "vectors" in checkpoint:
            meta = checkpoint.get("metadata", {})
            vectors = checkpoint["vectors"]
            ConceptVectorEngine._vector_cache[abs_path] = vectors
            logger.info(
                "Vectors loaded | path=%s | metadata=%s", abs_path, meta
            )
            return vectors

        # Legacy format — plain {layer: tensor} dict
        logger.warning(
            "Legacy vector format detected in %s. Consider re-extracting.",
            file_path,
        )
        ConceptVectorEngine._vector_cache[abs_path] = checkpoint
        return checkpoint
