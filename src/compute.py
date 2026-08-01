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
    """

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mean_difference(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Compute ``v = mean(h_pos) - mean(h_neg)``.

        Parameters
        ----------
        pos_activations : torch.Tensor
            Shape ``(num_samples, hidden_dim)``.
        neg_activations : torch.Tensor
            Shape ``(num_samples, hidden_dim)``.
        normalize : bool, default=False
            If True, L2-normalize the result to unit length before returning.

        Returns
        -------
        torch.Tensor
            Concept vector of shape ``(hidden_dim,)``.
        """
        orig_dtype = pos_activations.dtype
        mean_pos = pos_activations.to(torch.float32).mean(dim=0)
        mean_neg = neg_activations.to(torch.float32).mean(dim=0)
        vec = mean_pos - mean_neg
        if normalize:
            vec = normalize_vector(vec)
        logger.debug(
            "Mean difference vector computed | norm=%.4f | normalize=%s",
            torch.norm(vec).item(),
            normalize,
        )
        return vec.to(orig_dtype)

    @staticmethod
    def compute_pca_vector(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Compute the first principal component of ``(h_pos - h_neg)``.

        The sign of the component is aligned to the mean-difference direction
        so that positive alpha always steers toward the positive concept.

        Parameters
        ----------
        pos_activations : torch.Tensor
            Shape ``(num_samples, hidden_dim)``.
        neg_activations : torch.Tensor
            Shape ``(num_samples, hidden_dim)``.
        normalize : bool, default=False
            If True, L2-normalize the result to unit length before returning.

        Returns
        -------
        torch.Tensor
            PCA-derived concept vector of shape ``(hidden_dim,)``.
        """
        orig_dtype = pos_activations.dtype
        diffs_np = (
            (pos_activations - neg_activations).to(torch.float32).numpy()
        )

        pca = PCA(n_components=1)
        pca.fit(diffs_np)

        pca_vec = torch.tensor(pca.components_[0], dtype=torch.float32)
        explained = float(pca.explained_variance_ratio_[0])

        # ---- Sign alignment ------------------------------------------------
        mean_diff = (pos_activations - neg_activations).to(torch.float32).mean(dim=0)
        cos_sim = torch.dot(pca_vec, mean_diff) / (
            torch.norm(pca_vec) * torch.norm(mean_diff) + 1e-9
        )
        if cos_sim < 0:
            pca_vec = -pca_vec

        if normalize:
            pca_vec = normalize_vector(pca_vec)

        logger.debug(
            "PCA vector computed | explained_variance=%.4f | cos_sim=%.4f | normalize=%s",
            explained,
            cos_sim.item(),
            normalize,
        )
        return pca_vec.to(orig_dtype)

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

        logger.info("Vectors saved | path=%s | layers=%s", file_path, list(cpu_vectors.keys()))
        return file_path

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
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Concept vector file not found: {file_path}"
            )

        # weights_only=False required because we store a mixed dict with tensors + metadata
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        # New format
        if isinstance(checkpoint, dict) and "vectors" in checkpoint:
            meta = checkpoint.get("metadata", {})
            logger.info(
                "Vectors loaded | path=%s | metadata=%s", file_path, meta
            )
            return checkpoint["vectors"]

        # Legacy format — plain {layer: tensor} dict
        logger.warning(
            "Legacy vector format detected in %s. Consider re-extracting.",
            file_path,
        )
        return checkpoint
