import gc
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import SteeringConfig

def load_model_and_tokenizer(
    config: SteeringConfig,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Safely load a Hugging Face Causal LLM and its associated tokenizer.

    Performs pre-load and post-load memory cleanup (garbage collection and
    CUDA cache clearing). Configures tokenizer padding properties for batch
    inference and enables quantization (4-bit/8-bit) if CUDA is available.

    Parameters
    ----------
    config : SteeringConfig
        System-wide configuration dataclass containing model name, device, and dtype.
    load_in_4bit : bool, default=False
        If True, loads the model in 4-bit quantization using bitsandbytes (requires CUDA).
    load_in_8bit : bool, default=False
        If True, loads the model in 8-bit quantization using bitsandbytes (requires CUDA).

    Returns
    -------
    model : AutoModelForCausalLM
        The instantiated PyTorch model ready for inference or extraction.
    tokenizer : AutoTokenizer
        The corresponding tokenizer with padding token configured.

    Raises
    ------
    ValueError
        If invalid model name is provided or device mapping fails.
    """
    # Trigger garbage collection and empty CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Configure quantization or device placement options
    model_kwargs = {
        "trust_remote_code": True,
    }

    is_cuda = torch.cuda.is_available() or config.device == "cuda"

    if is_cuda:
        model_kwargs["device_map"] = "auto"
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = config.dtype
    else:
        # Non-CUDA path (CPU or MPS)
        model_kwargs["torch_dtype"] = config.dtype
        if config.device == "mps":
            model_kwargs["device_map"] = "auto"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is required for batched autoregressive generation
    tokenizer.padding_side = "left"

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Manual placement if device_map did not map it
    if "device_map" not in model_kwargs:
        model = model.to(config.device)

    # Set model to evaluation mode
    model.eval()

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer
