from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SteeredGenerator:
    """
    SteeredGenerator for dynamically injecting steering vectors during inference.

    This class manages PyTorch forward hooks to modify the residual stream
    activations of target layers in a causal language model during token generation.
    It provides methods to generate unsteered baseline text and steered text
    side-by-side.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The causal language model to apply steering to.
    tokenizer : AutoTokenizer
        The tokenizer corresponding to the model.
    device : str, default="cpu"
        The computing device (e.g., 'cuda', 'mps', 'cpu').
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_hooks: List[torch.utils.hooks.RemovableHandle] = []

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
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, "gpt_neox") and hasattr(
            self.model.gpt_neox, "layers"
        ):
            return self.model.gpt_neox.layers[layer_idx]
        else:
            raise AttributeError(
                "Unsupported transformer architecture. "
                "Unable to locate model layers dynamically."
            )

    def register_steering_hooks(
        self, vectors: Dict[int, torch.Tensor], alpha: float
    ) -> None:
        """
        Register steering hooks on specified layers.

        Parameters
        ----------
        vectors : Dict[int, torch.Tensor]
            Dictionary mapping layer indices to concept vectors.
        alpha : float
            Steering intensity coefficient.
        """
        # Ensure any existing hooks are removed first
        self.remove_steering_hooks()

        def get_steering_hook(layer_idx: int, concept_vector: torch.Tensor):
            def steering_hook(module: nn.Module, input_args: Tuple, output: nn.Module):
                # Unpack the output tuple if needed
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Align shape of concept vector for broadcasting
                # h shape: (batch_size, seq_len, hidden_dim)
                # concept_vector shape: (hidden_dim,)
                # Move concept vector to the same device/dtype as hidden_states
                vec = concept_vector.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

                # Add the scaled concept vector to the original hidden states
                steered_hidden_states = hidden_states + alpha * vec

                if isinstance(output, tuple):
                    return (steered_hidden_states,) + output[1:]
                return steered_hidden_states

            return steering_hook

        for layer, vec in vectors.items():
            layer_module = self._get_layer(layer)
            hook = layer_module.register_forward_hook(
                get_steering_hook(layer, vec)
            )
            self.active_hooks.append(hook)

    def remove_steering_hooks(self) -> None:
        """
        Remove all active steering hooks from the model.
        """
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks.clear()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text under the current state of the model.

        Parameters
        ----------
        prompt : str
            The input prompt string (can include chat templates).
        max_new_tokens : int, default=128
            Maximum number of new tokens to generate.
        temperature : float, default=0.7
            Sampling temperature.
        top_p : float, default=0.9
            Nucleus sampling probability.
        do_sample : bool, default=True
            Whether to use sampling or greedy decoding.

        Returns
        -------
        str
            The generated output text (excluding the input prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Build token generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # Filter out None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Slice to exclude input tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_comparative(
        self,
        prompt: str,
        vectors: Dict[int, torch.Tensor],
        alpha: float,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Tuple[str, str]:
        """
        Generate both unsteered baseline text and steered text side-by-side.

        Ensures that hooks are cleanly registered, used for generation,
        and removed, even in the event of an error.

        Parameters
        ----------
        prompt : str
            The input prompt string.
        vectors : Dict[int, torch.Tensor]
            Dictionary mapping layer indices to concept vectors.
        alpha : float
            Steering intensity coefficient.
        max_new_tokens : int, default=128
            Maximum number of new tokens to generate.
        temperature : float, default=0.7
            Sampling temperature.
        top_p : float, default=0.9
            Nucleus sampling probability.
        do_sample : bool, default=True
            Whether to use sampling or greedy decoding.

        Returns
        -------
        baseline_text : str
            The generated text without steering hooks.
        steered_text : str
            The generated text with steering hooks active.
        """
        # Ensure model has no hooks during baseline generation
        self.remove_steering_hooks()
        baseline_text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        try:
            # Register steering hooks and generate
            self.register_steering_hooks(vectors, alpha)
            steered_text = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        finally:
            # Clean up hooks under all circumstances
            self.remove_steering_hooks()

        return baseline_text, steered_text
