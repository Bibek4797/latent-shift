import os
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class SteeringConfig:
    """
    Configuration parameters for the LatentShift Steering System.

    Attributes
    ----------
    model_name : str
        Hugging Face model ID. Default is "Qwen/Qwen2.5-7B-Instruct".
    device : str
        The computing device to load the model on (cuda, mps, or cpu).
        Automatically detected by default.
    dtype_str : str
        String representation of torch dtype ("float16", "bfloat16", "float32").
    default_layers : List[int]
        Indices of target hidden layers where hooks will be attached.
    default_alpha : float
        Steering scale coefficient. Multiplies the concept vector during addition.
    data_dir : str
        Directory path to store cached concept vectors.
    """
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = field(
        default_factory=lambda: "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype_str: str = "float16"
    default_layers: List[int] = field(default_factory=lambda: [12, 13, 14, 15, 16])
    default_alpha: float = 2.0
    data_dir: str = "data/concept_vectors"

    @property
    def dtype(self) -> torch.dtype:
        """
        Convert dtype_str to corresponding torch.dtype.

        Returns
        -------
        torch.dtype
            The PyTorch data type (torch.float16, torch.bfloat16, or torch.float32).
        """
        if self.dtype_str == "bfloat16":
            return torch.bfloat16
        elif self.dtype_str == "float16":
            return torch.float16
        return torch.float32

    def __post_init__(self):
        """
        Create the data directory if it does not exist.
        """
        os.makedirs(self.data_dir, exist_ok=True)
