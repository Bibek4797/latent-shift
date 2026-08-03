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

Dynamic Closed-Loop Steering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``generate_dynamic`` method enables token-by-token alpha adaptation using
pluggable schedulers (see ``src.schedulers``). Instead of a single fixed alpha,
the steering coefficient evolves over the generation sequence:

    h_steered(t) = h_original(t) + alpha(t) * v_concept

Supported schedulers: Fixed, Linear, Cosine, Confidence-based, Entropy-based.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schedulers import (
    AlphaTrajectory,
    BaseAlphaScheduler,
    FixedScheduler,
    build_scheduler,
)
from src.utils import compute_layer_weights, get_logger, get_transformer_layer, set_seed

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
        # Mutable alpha container used by dynamic hooks (single-element list for closure mutability)
        self._dynamic_alpha: List[float] = [1.0]
        logger.info("SteeredGenerator initialized | device=%s", device)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_steering_hooks(
        self,
        vectors: Dict[int, torch.Tensor],
        alpha: Union[float, Dict[int, float]],
        strategy: str = "uniform",
    ) -> None:
        """
        Attach output hooks that inject concept vectors with layer-specific steering coefficients.

        The hook applies: ``h_steered = h_original + alpha_i * v_concept_i``

        Existing hooks are removed before new ones are registered to avoid
        accidental stacking.

        Parameters
        ----------
        vectors : Dict[int, torch.Tensor]
            ``{layer_idx: concept_vector}`` mapping.
        alpha : float or Dict[int, float]
            Steering coefficient(s). If a float/int scalar is provided, it is converted
            into a layer-specific weight mapping based on ``strategy``.
            If a dictionary mapping ``{layer_idx: alpha_i}`` is provided, it is used directly.
        strategy : str, default="uniform"
            Adaptive weighting strategy ("uniform", "linear_decay", "cosine_decay").
            Ignored if ``alpha`` is a dictionary.
        """
        self.remove_steering_hooks()

        if isinstance(alpha, (int, float)):
            alpha_dict = compute_layer_weights(
                list(vectors.keys()), base_alpha=float(alpha), strategy=strategy
            )
        elif isinstance(alpha, dict):
            alpha_dict = alpha
        else:
            raise TypeError(
                f"alpha must be a float, int, or Dict[int, float], got {type(alpha).__name__}"
            )

        def _make_hook(concept_vector: torch.Tensor, alpha_i: float):
            def _steering_hook(
                module: nn.Module,
                input_args: Tuple,
                output,
            ):
                hidden = output[0] if isinstance(output, tuple) else output
                vec = concept_vector.to(
                    device=hidden.device, dtype=hidden.dtype
                )
                steered = hidden + alpha_i * vec
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return _steering_hook

        for layer_idx, vec in vectors.items():
            alpha_i = alpha_dict.get(layer_idx, 1.0)
            module = get_transformer_layer(self.model, layer_idx)
            hook = module.register_forward_hook(_make_hook(vec, alpha_i))
            self.active_hooks.append(hook)

        logger.debug(
            "Registered %d steering hooks | alpha_dict=%s", len(self.active_hooks), alpha_dict
        )

    def register_dynamic_steering_hooks(
        self,
        vectors: Dict[int, torch.Tensor],
        layer_weight_ratios: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Attach hooks whose effective alpha is ``self._dynamic_alpha[0] * ratio_i``.

        The ``_dynamic_alpha`` container is updated externally (by ``generate_dynamic``)
        at each token step so that the hooks always read the current alpha.

        Parameters
        ----------
        vectors : Dict[int, torch.Tensor]
            ``{layer_idx: concept_vector}`` mapping.
        layer_weight_ratios : Dict[int, float], optional
            Per-layer weighting ratios. If ``None``, all layers receive ratio 1.0
            (i.e., they all use the raw scheduler alpha).
        """
        self.remove_steering_hooks()

        if layer_weight_ratios is None:
            layer_weight_ratios = {idx: 1.0 for idx in vectors}

        # Reference to the mutable container for closures
        dynamic_alpha_ref = self._dynamic_alpha

        def _make_dynamic_hook(concept_vector: torch.Tensor, ratio: float):
            def _dynamic_steering_hook(
                module: nn.Module,
                input_args: Tuple,
                output,
            ):
                hidden = output[0] if isinstance(output, tuple) else output
                vec = concept_vector.to(device=hidden.device, dtype=hidden.dtype)
                effective_alpha = dynamic_alpha_ref[0] * ratio
                steered = hidden + effective_alpha * vec
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return _dynamic_steering_hook

        for layer_idx, vec in vectors.items():
            ratio = layer_weight_ratios.get(layer_idx, 1.0)
            module = get_transformer_layer(self.model, layer_idx)
            hook = module.register_forward_hook(_make_dynamic_hook(vec, ratio))
            self.active_hooks.append(hook)

        logger.debug(
            "Registered %d dynamic steering hooks | ratios=%s",
            len(self.active_hooks),
            layer_weight_ratios,
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

    def generate_dynamic(
        self,
        prompt: str,
        vectors: Dict[int, torch.Tensor],
        scheduler: BaseAlphaScheduler,
        strategy: str = "uniform",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[str, AlphaTrajectory]:
        """
        Generate text with dynamic, token-by-token alpha adaptation.

        At each decoding step, the scheduler computes the current alpha value,
        which is then applied via forward hooks to steer the residual stream.

        Parameters
        ----------
        prompt : str
            Input text.
        vectors : Dict[int, torch.Tensor]
            Concept vectors keyed by layer index.
        scheduler : BaseAlphaScheduler
            Alpha scheduler controlling per-token steering intensity.
        strategy : str, default="uniform"
            Layer-weighting strategy for multi-layer injection.
        max_new_tokens : int, default=128
            Maximum tokens to generate.
        temperature : float, default=0.7
        top_p : float, default=0.9
        top_k : int, default=50
        do_sample : bool, default=True
        seed : int, optional

        Returns
        -------
        text : str
            Generated output text (prompt stripped).
        trajectory : AlphaTrajectory
            Per-token alpha history for analysis and visualisation.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        if seed is not None:
            set_seed(seed)

        scheduler.reset()

        # Compute per-layer weight ratios from the strategy
        layer_weight_ratios = compute_layer_weights(
            list(vectors.keys()), base_alpha=1.0, strategy=strategy
        )

        trajectory = AlphaTrajectory(scheduler_name=scheduler.name)

        # Register dynamic hooks
        self.register_dynamic_steering_hooks(vectors, layer_weight_ratios)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]

            generated_tokens: List[int] = []

            for step in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)

                logits = outputs.logits[:, -1, :]  # (1, vocab_size)

                # Compute alpha for this step
                alpha_t = scheduler.step(
                    step_idx=step,
                    total_steps=max_new_tokens,
                    logits=logits.squeeze(0),
                )
                trajectory.record(alpha_t)
                self._dynamic_alpha[0] = alpha_t

                # Sample next token
                if do_sample:
                    scaled_logits = logits / max(temperature, 1e-6)

                    # Top-k filtering
                    if top_k > 0:
                        topk_vals, topk_idx = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        mask = torch.full_like(scaled_logits, float("-inf"))
                        mask.scatter_(1, topk_idx, topk_vals)
                        scaled_logits = mask

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                        sorted_logits[remove_mask] = float("-inf")
                        scaled_logits = scaled_logits.scatter(1, sorted_indices, sorted_logits)

                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                token_id = next_token.item()
                generated_tokens.append(token_id)

                # Check for EOS
                if token_id == self.tokenizer.eos_token_id:
                    break

                # Append to input for next step
                input_ids = torch.cat([input_ids, next_token], dim=1)

        finally:
            self.remove_steering_hooks()

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logger.debug(
            "Dynamic generation complete | scheduler=%s | tokens=%d | alpha_mean=%.3f",
            scheduler.name,
            len(generated_tokens),
            trajectory.to_dict()["alpha_mean"],
        )
        return text, trajectory

    def generate_comparative(
        self,
        prompt: str,
        vectors: Dict[int, torch.Tensor],
        alpha: Union[float, Dict[int, float]],
        strategy: str = "uniform",
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
        alpha : float or Dict[int, float]
            Steering intensity coefficient(s).
        strategy : str, default="uniform"
            Weighting strategy ("uniform", "linear_decay", "cosine_decay").
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
            self.register_steering_hooks(vectors, alpha, strategy=strategy)
            logger.info(
                "Generating steered response | alpha=%s | strategy=%s | layers=%s",
                alpha,
                strategy,
                list(vectors.keys()),
            )
            steered_text = self.generate(prompt, **gen_kwargs)
        finally:
            self.remove_steering_hooks()

        return baseline_text, steered_text

    def generate_comparative_dynamic(
        self,
        prompt: str,
        vectors: Dict[int, torch.Tensor],
        scheduler: BaseAlphaScheduler,
        strategy: str = "uniform",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[str, str, AlphaTrajectory]:
        """
        Generate baseline and dynamically-steered outputs side-by-side.

        Parameters
        ----------
        prompt : str
            Input text.
        vectors : Dict[int, torch.Tensor]
            Concept vectors keyed by layer index.
        scheduler : BaseAlphaScheduler
            Alpha scheduler for dynamic steering.
        strategy : str, default="uniform"
            Layer-weighting strategy.
        max_new_tokens : int, default=128
        temperature : float, default=0.7
        top_p : float, default=0.9
        top_k : int, default=50
        do_sample : bool, default=True
        seed : int, optional

        Returns
        -------
        baseline_text : str
        steered_text : str
        trajectory : AlphaTrajectory
        """
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            seed=seed,
        )

        # Baseline
        self.remove_steering_hooks()
        logger.info("Generating baseline response...")
        baseline_text = self.generate(prompt, **gen_kwargs)

        # Dynamic steered
        logger.info(
            "Generating dynamic-steered response | scheduler=%s | strategy=%s",
            scheduler.name,
            strategy,
        )
        steered_text, trajectory = self.generate_dynamic(
            prompt=prompt,
            vectors=vectors,
            scheduler=scheduler,
            strategy=strategy,
            **gen_kwargs,
        )

        return baseline_text, steered_text, trajectory
