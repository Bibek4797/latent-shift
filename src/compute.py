import os
from typing import Dict
import numpy as np
from sklearn.decomposition import PCA
import torch

class ConceptVectorEngine:
    """
    ConceptVectorEngine for computing, saving, and loading steering vectors.

    This engine supports calculating concept vectors via:
    1. Mean Difference: Calculates the mean difference between positive and negative
       hidden states.
    2. Principal Component Analysis (PCA): Calculates the first principal component
       of the difference of positive and negative hidden states.

    It also ensures the sign of the PCA-extracted vector matches the positive concept direction.
    """

    @staticmethod
    def compute_mean_difference(
        pos_activations: torch.Tensor, neg_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean difference vector: v = mean(h_pos) - mean(h_neg).

        Parameters
        ----------
        pos_activations : torch.Tensor
            Hidden state activations for positive prompts. Shape: `(num_samples, hidden_dim)`.
        neg_activations : torch.Tensor
            Hidden state activations for negative prompts. Shape: `(num_samples, hidden_dim)`.

        Returns
        -------
        torch.Tensor
            The calculated mean difference concept vector. Shape: `(hidden_dim,)`.
        """
        # Ensure activations are float32 for stable mean computation, then cast back
        orig_dtype = pos_activations.dtype
        mean_pos = torch.mean(pos_activations.to(torch.float32), dim=0)
        mean_neg = torch.mean(neg_activations.to(torch.float32), dim=0)

        diff = mean_pos - mean_neg
        return diff.to(orig_dtype)

    @staticmethod
    def compute_pca_vector(
        pos_activations: torch.Tensor, neg_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the concept vector using PCA on the differences (h_pos - h_neg).

        Extracts the first principal component. Ensures sign alignment with the positive
        mean difference direction.

        Parameters
        ----------
        pos_activations : torch.Tensor
            Hidden state activations for positive prompts. Shape: `(num_samples, hidden_dim)`.
        neg_activations : torch.Tensor
            Hidden state activations for negative prompts. Shape: `(num_samples, hidden_dim)`.

        Returns
        -------
        torch.Tensor
            The calculated PCA concept vector. Shape: `(hidden_dim,)`.
        """
        orig_dtype = pos_activations.dtype
        diffs = (pos_activations - neg_activations).to(torch.float32).numpy()

        # Fit PCA to extract the first principal component
        pca = PCA(n_components=1)
        pca.fit(diffs)

        # Get the first principal component
        pca_vector_np = pca.components_[0]
        pca_vector = torch.tensor(pca_vector_np, dtype=torch.float32)

        # Align the sign of the PCA component with the mean difference direction
        mean_diff = torch.mean(
            (pos_activations - neg_activations).to(torch.float32), dim=0
        )
        cos_sim = torch.dot(pca_vector, mean_diff) / (
            torch.norm(pca_vector) * torch.norm(mean_diff) + 1e-9
        )

        if cos_sim < 0:
            pca_vector = -pca_vector

        return pca_vector.to(orig_dtype)

    @staticmethod
    def save_vectors(
        vectors: Dict[int, torch.Tensor], data_dir: str, filename: str
    ) -> str:
        """
        Save computed concept vectors to disk.

        Parameters
        ----------
        vectors : Dict[int, torch.Tensor]
            Dictionary mapping layer indices to their computed concept vectors.
        data_dir : str
            Directory path to save the concept vector file.
        filename : str
            The output filename (e.g. 'safety_vector.pt').

        Returns
        -------
        str
            The absolute path of the saved file.
        """
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, filename)

        # Move vectors to CPU before saving to make it device-independent
        cpu_vectors = {layer: vec.cpu().detach() for layer, vec in vectors.items()}
        torch.save(cpu_vectors, file_path)
        return file_path

    @staticmethod
    def load_vectors(file_path: str) -> Dict[int, torch.Tensor]:
        """
        Load concept vectors from disk.

        Parameters
        ----------
        file_path : str
            The path to the saved concept vector file.

        Returns
        -------
        Dict[int, torch.Tensor]
            Dictionary mapping layer indices to their loaded concept vectors.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the specified path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No concept vector file found at: {file_path}"
            )

        vectors = torch.load(file_path, map_location="cpu")
        return vectors
