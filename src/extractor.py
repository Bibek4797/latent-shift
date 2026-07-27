from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationExtractor:
    """
    ActivationExtractor for capturing internal representations of Causal LLMs.

    This class registers PyTorch forward hooks on intermediate residual streams
    to capture the hidden states at the last token position for a set of prompts.
    It supports contrastive pairs (e.g., Positive vs. Negative) and handles
    diverse model architectures.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The causal language model to extract activations from.
    tokenizer : AutoTokenizer
        The tokenizer corresponding to the model.
    layers : List[int]
        The list of layer indices to hook and extract activations from.
    device : str, default="cpu"
        The device where the model resides.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layers: List[int],
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device

    def _get_layer(self, layer_idx: int) -> nn.Module:
        """
        Retrieve the transformer layer module by index dynamically.

        Parameters
        ----------
        layer_idx : int
            The index of the target layer.

        Returns
        -------
        nn.Module
            The PyTorch module representing the target layer.

        Raises
        ------
        AttributeError
            If the model architecture does not match known patterns.
        """
        # Check standard architectures (Llama, Qwen, Mistral)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        # Check GPT-2 style architectures
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h[layer_idx]
        # Check GPT-NeoX architectures
        elif hasattr(self.model, "gpt_neox") and hasattr(
            self.model.gpt_neox, "layers"
        ):
            return self.model.gpt_neox.layers[layer_idx]
        else:
            raise AttributeError(
                "Unsupported transformer architecture. "
                "Unable to locate model layers dynamically."
            )

    def extract_activations(self, prompts: List[str]) -> Dict[int, torch.Tensor]:
        """
        Extract hidden state activations at the last token for all target layers.

        Parameters
        ----------
        prompts : List[str]
            A list of input text prompts.

        Returns
        -------
        Dict[int, torch.Tensor]
            A dictionary mapping layer index to a tensor of shape
            `(num_prompts, hidden_dim)` containing the last-token activations.
        """
        # Initialize lists to accumulate activations for each layer
        accumulated: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in self.layers
        }
        hooks = []

        # Hook function builder
        def get_hook(layer_idx: int):
            def hook_fn(module: nn.Module, input_args: Tuple, output: nn.Module):
                # Unpack the output tuple if needed
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # hidden_states is of shape (batch_size, seq_len, hidden_dim)
                # We extract the last token position (-1) for the single item in the batch.
                # Clone and detach to prevent memory leakage or graph retention.
                last_token_activation = hidden_states[0, -1, :].clone().detach().cpu()
                accumulated[layer_idx].append(last_token_activation)

            return hook_fn

        # Register forward hooks on target layers
        for layer in self.layers:
            layer_module = self._get_layer(layer)
            hook = layer_module.register_forward_hook(get_hook(layer))
            hooks.append(hook)

        try:
            # Process prompts one by one to avoid padding mismatch artifacts
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            # Ensure hooks are cleanly removed even if an error occurs
            for hook in hooks:
                hook.remove()

        # Stack the list of activations into single tensors per layer
        stacked_activations: Dict[int, torch.Tensor] = {}
        for layer in self.layers:
            stacked_activations[layer] = torch.stack(accumulated[layer])

        return stacked_activations

    def extract_contrastive(
        self, pairs: List[Tuple[str, str]]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extract activations for contrasting prompt pairs.

        Parameters
        ----------
        pairs : List[Tuple[str, str]]
            A list of contrastive prompt pairs, e.g., (positive_prompt, negative_prompt).

        Returns
        -------
        pos_activations : Dict[int, torch.Tensor]
            Last-token hidden states for positive prompts.
        neg_activations : Dict[int, torch.Tensor]
            Last-token hidden states for negative prompts.
        """
        pos_prompts = [pair[0] for pair in pairs]
        neg_prompts = [pair[1] for pair in pairs]

        pos_activations = self.extract_activations(pos_prompts)
        neg_activations = self.extract_activations(neg_prompts)

        return pos_activations, neg_activations
