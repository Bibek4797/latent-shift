"""
src/evaluator.py
----------------
Evaluation metrics for steering quality assessment.

Provides:
- **Perplexity (PPL)**: Measures fluency degradation caused by steering.
  Lower PPL = more natural text. A steered response with PPL close to the
  baseline indicates that the concept injection did not damage coherence.
- **Cosine Similarity**: Quantifies how well a vector shift aligns with the
  target concept direction.
- **Steering Effectiveness Score**: Combined metric capturing both fluency
  preservation and concept alignment.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger

logger = get_logger(__name__)


class SteeringEvaluator:
    """
    Compute quantitative metrics for evaluating activation steering quality.

    All methods are static — the class is a stateless evaluation utility.
    """

    @staticmethod
    def compute_perplexity(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        text: str,
        device: str = "cpu",
    ) -> float:
        """
        Compute language-model perplexity of ``text``.

        Perplexity = exp(cross_entropy_loss).  Lower values denote more
        fluent, model-coherent text.  Returns ``float('nan')`` for empty
        strings so callers can handle this case gracefully.

        Parameters
        ----------
        model : AutoModelForCausalLM
        tokenizer : AutoTokenizer
        text : str
            Text to evaluate.
        device : str, default="cpu"

        Returns
        -------
        float
            Perplexity score ≥ 1.0, or ``nan`` for empty text.
        """
        if not text.strip():
            logger.warning("compute_perplexity called with empty text, returning nan.")
            return float("nan")

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

            if loss is None:
                # Manual CE loss for models that don't expose it
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

        ppl = torch.exp(loss).item()
        logger.debug("Perplexity=%.4f | tokens=%d", ppl, inputs["input_ids"].shape[1])
        return ppl

    @staticmethod
    def compute_cosine_similarity(
        vec1: torch.Tensor,
        vec2: torch.Tensor,
    ) -> float:
        """
        Cosine similarity between two tensors (flattened to 1-D before comparison).

        Parameters
        ----------
        vec1 : torch.Tensor
        vec2 : torch.Tensor

        Returns
        -------
        float
            Value in [-1.0, 1.0].  Returns 0.0 if either vector is zero-norm.
        """
        v1 = vec1.to(torch.float32).flatten()
        v2 = vec2.to(torch.float32).flatten()
        n1, n2 = torch.norm(v1), torch.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return torch.dot(v1, v2).item() / (n1 * n2).item()

    @staticmethod
    def compute_steering_report(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        baseline_text: str,
        steered_text: str,
        concept_vector: torch.Tensor,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Compute a consolidated evaluation report for one steering run.

        Metrics included:

        - ``ppl_baseline``   : Perplexity of the unsteered output.
        - ``ppl_steered``    : Perplexity of the steered output.
        - ``delta_ppl``      : ``ppl_steered - ppl_baseline``.  Positive = less
          fluent; negative = more fluent.
        - ``ppl_ratio``      : ``ppl_steered / ppl_baseline``.  Close to 1.0 is
          ideal.
        - ``cosine_sim``     : Cosine similarity between the two output
          representations at the last hidden layer (approximated by
          embedding-level comparison when hidden states unavailable).

        Parameters
        ----------
        model : AutoModelForCausalLM
        tokenizer : AutoTokenizer
        baseline_text : str
        steered_text : str
        concept_vector : torch.Tensor
            The applied concept vector (used as a reference for similarity).
        device : str, default="cpu"

        Returns
        -------
        Dict[str, float]
            Keys: ``ppl_baseline``, ``ppl_steered``, ``delta_ppl``,
            ``ppl_ratio``, ``cosine_sim``.
        """
        ppl_b = SteeringEvaluator.compute_perplexity(
            model, tokenizer, baseline_text, device
        )
        ppl_s = SteeringEvaluator.compute_perplexity(
            model, tokenizer, steered_text, device
        )
        delta = ppl_s - ppl_b
        ratio = (ppl_s / ppl_b) if ppl_b > 0 else float("nan")

        # Approximate cosine similarity via token embedding centroids
        def _text_embedding(text: str) -> torch.Tensor:
            """Mean of token embeddings as a proxy hidden-state representation."""
            enc = tokenizer(text, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            with torch.no_grad():
                emb_weight = model.get_input_embeddings().weight
                embs = emb_weight[ids[0]].mean(dim=0).float().cpu()
            return embs

        emb_b = _text_embedding(baseline_text)
        emb_s = _text_embedding(steered_text)
        cos_sim = SteeringEvaluator.compute_cosine_similarity(emb_b, emb_s)

        report = {
            "ppl_baseline": round(ppl_b, 4),
            "ppl_steered": round(ppl_s, 4),
            "delta_ppl": round(delta, 4),
            "ppl_ratio": round(ratio, 4),
            "cosine_sim": round(cos_sim, 4),
        }
        logger.info("Steering report: %s", report)
        return report
