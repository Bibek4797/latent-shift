from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SteeringEvaluator:
    """
    SteeringEvaluator to compute perplexity and cosine similarity.

    Used to measure the quality of steered generation (via perplexity to verify
    text fluency) and trace the alignment of representation engineering interventions
    (via cosine similarity).
    """

    @staticmethod
    def compute_perplexity(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        text: str,
        device: str = "cpu",
    ) -> float:
        """
        Compute the language modeling perplexity (PPL) of the given text.

        Perplexity is calculated as exp(cross_entropy_loss). A lower perplexity
        denotes a more fluent and probable sequence according to the model.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The causal language model.
        tokenizer : AutoTokenizer
            The corresponding tokenizer.
        text : str
            The text string to evaluate.
        device : str, default="cpu"
            The device where model evaluation should take place.

        Returns
        -------
        float
            The perplexity score. Returns NaN if the text is empty.
        """
        if not text.strip():
            return float("nan")

        # Set up input and labels
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

            # Fallback manual shifting if HF model does not compute loss automatically
            if loss is None:
                logits = outputs.logits
                # Shift logits and labels for Causal LM next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

        return torch.exp(loss).item()

    @staticmethod
    def compute_cosine_similarity(
        vec1: torch.Tensor, vec2: torch.Tensor
    ) -> float:
        """
        Calculate the cosine similarity between two tensors.

        Tensors are flattened prior to comparison.

        Parameters
        ----------
        vec1 : torch.Tensor
            First input tensor.
        vec2 : torch.Tensor
            Second input tensor.

        Returns
        -------
        float
            Cosine similarity value in range [-1.0, 1.0].
        """
        # Ensure tensors are float32 and flat
        v1 = vec1.to(torch.float32).flatten()
        v2 = vec2.to(torch.float32).flatten()

        norm_v1 = torch.norm(v1)
        norm_v2 = torch.norm(v2)

        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0

        similarity = torch.dot(v1, v2) / (norm_v1 * norm_v2)
        return similarity.item()
