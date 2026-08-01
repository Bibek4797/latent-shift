"""
src/extractor.py
----------------
Activation extraction via PyTorch forward hooks.

The ``ActivationExtractor`` class registers hooks on target transformer layers,
runs a forward pass for each prompt, and captures the hidden-state tensor at
the final token position — the standard approach in Representation Engineering.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger, get_transformer_layer

logger = get_logger(__name__)


class ActivationExtractor:
    """
    Capture internal hidden-state activations from a Causal LLM via forward hooks.

    Registers ``register_forward_hook`` on each requested transformer layer,
    runs a single forward pass per prompt under ``torch.no_grad()``, records
    the activation at the **last token position** (the prediction point), then
    cleans up all hooks — even if an exception is raised.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The causal language model to probe.
    tokenizer : AutoTokenizer
        The tokenizer corresponding to the model.
    layers : List[int]
        Zero-based layer indices to hook into.
    device : str, default="cpu"
        Device string matching the model's current placement.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layers: List[int],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device
        logger.info(
            "ActivationExtractor initialized | layers=%s | device=%s",
            layers,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_activations(
        self, prompts: List[str]
    ) -> Dict[int, torch.Tensor]:
        """
        Run a forward pass for each prompt and collect last-token activations.

        Parameters
        ----------
        prompts : List[str]
            Input text prompts. Processed one-by-one to avoid padding artefacts.

        Returns
        -------
        Dict[int, torch.Tensor]
            Mapping ``{layer_idx: tensor of shape (num_prompts, hidden_dim)}``.

        Raises
        ------
        ValueError
            If ``prompts`` is empty.
        """
        if not prompts:
            raise ValueError("prompts list must not be empty.")

        accumulated: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in self.layers
        }
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(layer_idx: int):
            def _hook_fn(
                module: nn.Module,
                input_args: Tuple,
                output,
            ) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                # Shape: (batch=1, seq_len, hidden_dim) → take last token
                vec = hidden[0, -1, :].clone().detach().cpu()
                accumulated[layer_idx].append(vec)

            return _hook_fn

        # ---- Register hooks ------------------------------------------------
        for layer_idx in self.layers:
            module = get_transformer_layer(self.model, layer_idx)
            hooks.append(module.register_forward_hook(_make_hook(layer_idx)))
        logger.debug("Registered %d forward hooks.", len(hooks))

        # ---- Forward passes ------------------------------------------------
        try:
            for i, prompt in enumerate(prompts):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model(**inputs)
                logger.debug("Processed prompt %d/%d", i + 1, len(prompts))
        finally:
            for hook in hooks:
                hook.remove()
            logger.debug("All %d hooks removed.", len(hooks))

        # ---- Stack per layer -----------------------------------------------
        stacked: Dict[int, torch.Tensor] = {
            layer: torch.stack(accumulated[layer]) for layer in self.layers
        }
        logger.info(
            "Extraction complete | prompts=%d | layers=%s",
            len(prompts),
            list(stacked.keys()),
        )
        return stacked

    def extract_contrastive(
        self,
        pairs: List[Tuple[str, str]],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extract activations for a list of contrastive prompt pairs.

        Parameters
        ----------
        pairs : List[Tuple[str, str]]
            Each element is ``(positive_prompt, negative_prompt)``.

        Returns
        -------
        pos_activations : Dict[int, torch.Tensor]
            Last-token activations for all positive prompts.
        neg_activations : Dict[int, torch.Tensor]
            Last-token activations for all negative prompts.

        Raises
        ------
        ValueError
            If ``pairs`` is empty.
        """
        if not pairs:
            raise ValueError("pairs list must not be empty.")

        pos_prompts = [p[0] for p in pairs]
        neg_prompts = [p[1] for p in pairs]

        logger.info(
            "Extracting contrastive activations | num_pairs=%d", len(pairs)
        )
        pos_acts = self.extract_activations(pos_prompts)
        neg_acts = self.extract_activations(neg_prompts)
        return pos_acts, neg_acts
