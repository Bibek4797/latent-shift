"""
src/steer.py
------------
Runtime activation steering via PyTorch forward hooks.

The ``SteeredGenerator`` class manages the full lifecycle of concept vector
injection into a causal LLM:

1. Registers output hooks on target residual stream layers.
2. At each token step, modifies hidden states as:
   ``h_steered = h_original + alpha * v_concept``
3. Removes hooks cleanly after generation (via ``try/finally``).
4. Provides ``generate_comparative`` for side-by-side baseline vs. steered output.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger, get_transformer_layer, set_seed

logger = get_logger(__name__)


class SteeredGenerator:
    """
    Inject concept steering vectors into LLM residual streams during generation.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The causal language model to steer.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    device : str, default="cpu"
        Target compute device.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_hooks: List[torch.utils.hooks.RemovableHandle] = []
        logger.info("SteeredGenerator initialized | device=%s", device)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_steering_hooks(
        self,
        vectors: Dict[int, torch.Tensor],
        alpha: float,
    ) -> None:
        """
        Attach output hooks that additively inject concept vectors.

        The hook applies:  ``h_steered = h_original + alpha * v_concept``

        Existing hooks are removed before new ones are registered to avoid
        accidental stacking.

        Parameters
        ----------
        vectors : Dict[int, torch.Tensor]
            ``{layer_idx: concept_vector}`` mapping.
        alpha : float
            Steering coefficient.  ``alpha > 0`` reinforces the positive
            concept; ``alpha < 0`` suppresses or reverses it.
        """
        self.remove_steering_hooks()

        def _make_hook(concept_vector: torch.Tensor):
            def _steering_hook(
                module: nn.Module,
                input_args: Tuple,
                output,
            ):
                hidden = output[0] if isinstance(output, tuple) else output
                vec = concept_vector.to(
                    device=hidden.device, dtype=hidden.dtype
                )
                steered = hidden + alpha * vec
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return _steering_hook

        for layer_idx, vec in vectors.items():
            module = get_transformer_layer(self.model, layer_idx)
            hook = module.register_forward_hook(_make_hook(vec))
            self.active_hooks.append(hook)

        logger.debug(
            "Registered %d steering hooks | alpha=%.3f", len(self.active_hooks), alpha
        )

    def remove_steering_hooks(self) -> None:
        """Remove all currently registered steering hooks."""
        for hook in self.active_hooks:
            hook.remove()
        if self.active_hooks:
            logger.debug("Removed %d steering hooks.", len(self.active_hooks))
        self.active_hooks.clear()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate text under the model's current hook state.

        Parameters
        ----------
        prompt : str
            Input text (may include a chat template prefix).
        max_new_tokens : int, default=128
            Maximum tokens to generate.
        temperature : float, default=0.7
            Sampling temperature. Ignored when ``do_sample=False``.
        top_p : float, default=0.9
            Nucleus sampling probability. Ignored when ``do_sample=False``.
        top_k : int, default=50
            Top-k sampling cutoff. Ignored when ``do_sample=False``.
        do_sample : bool, default=True
            Use sampling; set ``False`` for greedy / deterministic decoding.
        seed : int, optional
            If provided, calls ``set_seed(seed)`` before generation for
            reproducibility.

        Returns
        -------
        str
            Generated text with the prompt tokens stripped.

        Raises
        ------
        ValueError
            If ``prompt`` is empty.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        if seed is not None:
            set_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: Dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.debug(
            "Generated %d tokens | do_sample=%s | seed=%s",
            len(generated_ids),
            do_sample,
            seed,
        )
        return text

    def generate_comparative(
        self,
        prompt: str,
        vectors: Dict[int, torch.Tensor],
        alpha: float,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Generate both an unsteered baseline and a steered output.

        Hooks are guaranteed to be removed after both generations regardless
        of exceptions, preventing hook leakage.

        Parameters
        ----------
        prompt : str
            Input text.
        vectors : Dict[int, torch.Tensor]
            Concept vectors keyed by layer index.
        alpha : float
            Steering intensity coefficient.
        max_new_tokens : int, default=128
        temperature : float, default=0.7
        top_p : float, default=0.9
        top_k : int, default=50
        do_sample : bool, default=True
        seed : int, optional
            Fixed seed for identical sampling conditions in both runs.

        Returns
        -------
        baseline_text : str
            Output without steering.
        steered_text : str
            Output with concept vector injected.
        """
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            seed=seed,
        )

        # Baseline (no hooks)
        self.remove_steering_hooks()
        logger.info("Generating baseline response...")
        baseline_text = self.generate(prompt, **gen_kwargs)

        # Steered
        try:
            self.register_steering_hooks(vectors, alpha)
            logger.info(
                "Generating steered response | alpha=%.3f | layers=%s",
                alpha,
                list(vectors.keys()),
            )
            steered_text = self.generate(prompt, **gen_kwargs)
        finally:
            self.remove_steering_hooks()

        return baseline_text, steered_text
