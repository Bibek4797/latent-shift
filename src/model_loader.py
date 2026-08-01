"""
src/model_loader.py
-------------------
Hugging Face model loading with quantization and memory management.

Handles:
- Full-precision and quantized (4-bit / 8-bit via bitsandbytes) loading.
- Multi-GPU placement via ``device_map="auto"``.
- Tokenizer padding configuration (required for stable generation).
- Pre- and post-load memory cleanup (CUDA cache + GC).
"""

import gc
import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SteeringConfig
from src.utils import get_logger

logger = get_logger(__name__)


def load_model_and_tokenizer(
    config: SteeringConfig,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a Hugging Face Causal LLM and its tokenizer.

    Pre- and post-load garbage collection and CUDA cache clearing prevent
    memory fragmentation when multiple models are loaded in a session.

    Parameters
    ----------
    config : SteeringConfig
        System-wide configuration with model name, device, and dtype.
    load_in_4bit : bool, default=False
        Enable 4-bit NF4 quantization via bitsandbytes (requires CUDA).
    load_in_8bit : bool, default=False
        Enable 8-bit LLM.int8 quantization via bitsandbytes (requires CUDA).

    Returns
    -------
    model : AutoModelForCausalLM
        Loaded model in eval mode.
    tokenizer : AutoTokenizer
        Corresponding tokenizer with pad token set.

    Raises
    ------
    ValueError
        If the model name is unresolvable by Hugging Face Hub.
    RuntimeError
        If 4-bit / 8-bit quantization is requested without a CUDA device.
    """
    # ---- Pre-load cleanup --------------------------------------------------
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Loading model '%s' | device=%s | dtype=%s | 4bit=%s | 8bit=%s",
        config.model_name,
        config.device,
        config.dtype_str,
        load_in_4bit,
        load_in_8bit,
    )

    # ---- Guard: quantization requires CUDA ---------------------------------
    if (load_in_4bit or load_in_8bit) and not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit and 8-bit quantization require a CUDA GPU. "
            "Remove the quantization flag or run on a CUDA device."
        )

    # ---- Build model kwargs ------------------------------------------------
    model_kwargs: dict = {"trust_remote_code": True}
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
        # CPU / MPS path
        model_kwargs["torch_dtype"] = config.dtype
        if config.device == "mps":
            model_kwargs["device_map"] = "auto"

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for stable autoregressive batch generation
    tokenizer.padding_side = "left"
    logger.info("Tokenizer loaded | vocab_size=%d", tokenizer.vocab_size)

    # ---- Model -------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Manual device placement when device_map was not set
    if "device_map" not in model_kwargs:
        model = model.to(config.device)

    model.eval()
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(
        "Model loaded | params=%.2fB | eval_mode=True", num_params
    )

    # ---- Post-load cleanup -------------------------------------------------
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer
