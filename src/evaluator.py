"""
src/evaluator.py
----------------
Research-grade evaluation framework for activation steering.

Provides:
- **Perplexity (PPL)**: Measures fluency degradation caused by steering.
- **Cosine Similarity**: Directional representation alignment between baseline and steered embeddings.
- **KL Divergence (D_KL)**: Relative entropy shift between baseline and steered output distributions.
- **Jensen-Shannon Divergence (D_JS)**: Symmetric, bounded distance between baseline and steered distributions.
- **Token Entropy (H)**: Quantifies model prediction uncertainty before and after steering.
- **Hidden State Norm Difference**: Measures magnitude changes in target layer activations.
- **Layer-wise Cosine Similarity**: Tracks directional alignment across intermediate layers.
- **Average Activation Shift Magnitude**: Mean Euclidean distance of hidden state vectors.
- **Steering Strength Score**: Shift magnitude normalized by baseline activation norm.
- **SteeringEvaluationReport**: Consolidated dataclass serializable to JSON.
- **Plotting Utilities**: Interactive Plotly charts for layerwise dynamics & metric comparisons.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SteeringEvaluationReport:
    """
    Consolidated research evaluation report for activation steering.
    """
    ppl_baseline: float
    ppl_steered: float
    delta_ppl: float
    ppl_ratio: float
    cosine_sim: float
    kl_divergence: float
    js_divergence: float
    entropy_baseline: float
    entropy_steered: float
    delta_entropy: float
    avg_shift_magnitude: float
    steering_strength_score: float
    layerwise_cosine_sim: Dict[int, float] = field(default_factory=dict)
    layerwise_norm_diff: Dict[int, float] = field(default_factory=dict)
    layerwise_shift_magnitude: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation report to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Serialize report to JSON string and optionally save to file.

        Parameters
        ----------
        filepath : Optional[str]
            File path to write JSON report.
        indent : int, default=2
            Indentation level for formatting.

        Returns
        -------
        str
            JSON string representation of evaluation report.
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Evaluation report saved to JSON: %s", filepath)
        return json_str


class SteeringEvaluator:
    """
    Research-grade quantitative evaluation framework for LLM activation steering.
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

        Perplexity = exp(cross_entropy_loss). Lower values denote more
        fluent, model-coherent text. Returns ``float('nan')`` for empty
        strings so callers can handle this case gracefully.

        Parameters
        ----------
        model : AutoModelForCausalLM
        tokenizer : AutoTokenizer
        text : str
        device : str, default="cpu"

        Returns
        -------
        float
            Perplexity score >= 1.0, or ``nan`` for empty text.
        """
        if not text.strip():
            logger.warning("compute_perplexity called with empty text, returning nan.")
            return float("nan")

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        with torch.inference_mode():
            outputs = model(**inputs)
            loss = outputs.loss

            if loss is None:
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
            Value in [-1.0, 1.0]. Returns 0.0 if either vector is zero-norm.
        """
        v1 = vec1.to(torch.float32).flatten()
        v2 = vec2.to(torch.float32).flatten()
        n1, n2 = torch.norm(v1), torch.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return (torch.dot(v1, v2) / (n1 * n2)).item()

    @staticmethod
    def compute_kl_divergence(
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        eps: float = 1e-10,
    ) -> float:
        """
        Compute KL Divergence D_KL(P || Q) between baseline logits (P) and steered logits (Q).

        Parameters
        ----------
        p_logits : torch.Tensor
            Baseline output logits.
        q_logits : torch.Tensor
            Steered output logits.
        eps : float, default=1e-10

        Returns
        -------
        float
            KL divergence in nats.
        """
        p_probs = F.softmax(p_logits.to(torch.float32), dim=-1)
        q_probs = F.softmax(q_logits.to(torch.float32), dim=-1)

        kl = (p_probs * (torch.log(p_probs + eps) - torch.log(q_probs + eps))).sum(dim=-1).mean()
        return float(kl.item())

    @staticmethod
    def compute_js_divergence(
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        eps: float = 1e-10,
    ) -> float:
        """
        Compute Jensen-Shannon Divergence D_JS(P || Q).

        Parameters
        ----------
        p_logits : torch.Tensor
        q_logits : torch.Tensor
        eps : float, default=1e-10

        Returns
        -------
        float
            JS divergence score.
        """
        p_probs = F.softmax(p_logits.to(torch.float32), dim=-1)
        q_probs = F.softmax(q_logits.to(torch.float32), dim=-1)
        m_probs = 0.5 * (p_probs + q_probs)

        kl_pm = (p_probs * (torch.log(p_probs + eps) - torch.log(m_probs + eps))).sum(dim=-1).mean()
        kl_qm = (q_probs * (torch.log(q_probs + eps) - torch.log(m_probs + eps))).sum(dim=-1).mean()

        jsd = 0.5 * (kl_pm + kl_qm)
        return float(jsd.item())

    @staticmethod
    def compute_entropy(
        logits: torch.Tensor,
        eps: float = 1e-10,
    ) -> float:
        """
        Compute average token entropy H(P) = -sum(P log P).

        Parameters
        ----------
        logits : torch.Tensor

        Returns
        -------
        float
            Entropy value in nats.
        """
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1).mean()
        return float(entropy.item())

    @staticmethod
    def compute_hidden_state_norm_difference(
        h_base: torch.Tensor,
        h_steered: torch.Tensor,
    ) -> float:
        """
        Compute hidden state norm difference: ||h_steered||_2 - ||h_base||_2.

        Parameters
        ----------
        h_base : torch.Tensor
        h_steered : torch.Tensor

        Returns
        -------
        float
            Norm difference value.
        """
        norm_b = torch.norm(h_base.to(torch.float32)).item()
        norm_s = torch.norm(h_steered.to(torch.float32)).item()
        return norm_s - norm_b

    @staticmethod
    def compute_layerwise_cosine_similarity(
        h_base_dict: Dict[int, torch.Tensor],
        h_steered_dict: Dict[int, torch.Tensor],
    ) -> Dict[int, float]:
        """
        Compute layer-wise cosine similarity between baseline and steered hidden states.

        Parameters
        ----------
        h_base_dict : Dict[int, torch.Tensor]
        h_steered_dict : Dict[int, torch.Tensor]

        Returns
        -------
        Dict[int, float]
            Mapping from layer index to cosine similarity.
        """
        res = {}
        for layer in sorted(h_base_dict.keys()):
            if layer in h_steered_dict:
                res[layer] = SteeringEvaluator.compute_cosine_similarity(
                    h_base_dict[layer], h_steered_dict[layer]
                )
        return res

    @staticmethod
    def compute_average_shift_magnitude(
        h_base_dict: Dict[int, torch.Tensor],
        h_steered_dict: Dict[int, torch.Tensor],
    ) -> float:
        """
        Compute average Euclidean distance between baseline and steered activations across layers.

        Parameters
        ----------
        h_base_dict : Dict[int, torch.Tensor]
        h_steered_dict : Dict[int, torch.Tensor]

        Returns
        -------
        float
            Average activation shift magnitude.
        """
        shifts = []
        for layer in h_base_dict:
            if layer in h_steered_dict:
                diff = (h_steered_dict[layer] - h_base_dict[layer]).to(torch.float32)
                shifts.append(torch.norm(diff).item())
        return float(np.mean(shifts)) if shifts else 0.0

    @staticmethod
    def compute_steering_strength_score(
        h_base_dict: Dict[int, torch.Tensor],
        h_steered_dict: Dict[int, torch.Tensor],
        eps: float = 1e-9,
    ) -> float:
        """
        Compute Steering Strength Score: (Average Shift Magnitude) / (Average Baseline Norm + eps).

        Parameters
        ----------
        h_base_dict : Dict[int, torch.Tensor]
        h_steered_dict : Dict[int, torch.Tensor]
        eps : float, default=1e-9

        Returns
        -------
        float
            Normalized steering strength score.
        """
        avg_shift = SteeringEvaluator.compute_average_shift_magnitude(h_base_dict, h_steered_dict)
        base_norms = [torch.norm(h_base_dict[l].to(torch.float32)).item() for l in h_base_dict]
        avg_base_norm = float(np.mean(base_norms)) if base_norms else eps
        return float(avg_shift / (avg_base_norm + eps))

    @staticmethod
    def evaluate_full(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        baseline_text: str,
        steered_text: str,
        concept_vector: Optional[torch.Tensor] = None,
        h_base_dict: Optional[Dict[int, torch.Tensor]] = None,
        h_steered_dict: Optional[Dict[int, torch.Tensor]] = None,
        device: str = "cpu",
    ) -> SteeringEvaluationReport:
        """
        Run full research-grade evaluation on baseline and steered outputs.

        Parameters
        ----------
        model : AutoModelForCausalLM
        tokenizer : AutoTokenizer
        prompt : str
        baseline_text : str
        steered_text : str
        concept_vector : Optional[torch.Tensor]
        h_base_dict : Optional[Dict[int, torch.Tensor]]
        h_steered_dict : Optional[Dict[int, torch.Tensor]]
        device : str, default="cpu"

        Returns
        -------
        SteeringEvaluationReport
            Dataclass containing all quantitative research metrics.
        """
        ppl_b = SteeringEvaluator.compute_perplexity(model, tokenizer, baseline_text, device)
        ppl_s = SteeringEvaluator.compute_perplexity(model, tokenizer, steered_text, device)
        delta_ppl = ppl_s - ppl_b
        ppl_ratio = (ppl_s / ppl_b) if (ppl_b > 0 and not np.isnan(ppl_b)) else float("nan")

        # ---- Distribution divergence & entropy -----------------------------
        enc_b = tokenizer(baseline_text or prompt, return_tensors="pt")
        enc_s = tokenizer(steered_text or prompt, return_tensors="pt")
        inputs_b = {k: v.to(device) for k, v in enc_b.items()}
        inputs_s = {k: v.to(device) for k, v in enc_s.items()}

        with torch.inference_mode():
            out_b = model(**inputs_b)
            out_s = model(**inputs_s)
            logits_b = out_b.logits
            logits_s = out_s.logits

        min_len = min(logits_b.shape[1], logits_s.shape[1])
        sub_b = logits_b[:, :min_len, :]
        sub_s = logits_s[:, :min_len, :]

        kl_div = SteeringEvaluator.compute_kl_divergence(sub_b, sub_s)
        js_div = SteeringEvaluator.compute_js_divergence(sub_b, sub_s)
        ent_b = SteeringEvaluator.compute_entropy(sub_b)
        ent_s = SteeringEvaluator.compute_entropy(sub_s)
        delta_ent = ent_s - ent_b

        # ---- Embedding Cosine Similarity ----------------------------------
        def _text_embedding(text: str) -> torch.Tensor:
            enc = tokenizer(text or " ", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            with torch.inference_mode():
                if hasattr(model, "get_input_embeddings"):
                    emb_weight = model.get_input_embeddings().weight
                    embs = emb_weight[ids[0]].mean(dim=0).float().cpu()
                else:
                    embs = torch.randn(16)
            return embs

        emb_b = _text_embedding(baseline_text)
        emb_s = _text_embedding(steered_text)
        cos_sim = SteeringEvaluator.compute_cosine_similarity(emb_b, emb_s)

        # ---- Hidden state layerwise metrics --------------------------------
        l_cos: Dict[int, float] = {}
        l_norm_diff: Dict[int, float] = {}
        l_shift_mag: Dict[int, float] = {}

        if h_base_dict and h_steered_dict:
            l_cos = SteeringEvaluator.compute_layerwise_cosine_similarity(h_base_dict, h_steered_dict)
            for layer in h_base_dict:
                if layer in h_steered_dict:
                    hb, hs = h_base_dict[layer], h_steered_dict[layer]
                    l_norm_diff[layer] = round(SteeringEvaluator.compute_hidden_state_norm_difference(hb, hs), 4)
                    l_shift_mag[layer] = round(torch.norm((hs - hb).float()).item(), 4)
            avg_shift = SteeringEvaluator.compute_average_shift_magnitude(h_base_dict, h_steered_dict)
            strength_score = SteeringEvaluator.compute_steering_strength_score(h_base_dict, h_steered_dict)
        else:
            avg_shift = 0.0
            strength_score = 0.0

        report = SteeringEvaluationReport(
            ppl_baseline=round(ppl_b, 4) if not np.isnan(ppl_b) else float("nan"),
            ppl_steered=round(ppl_s, 4) if not np.isnan(ppl_s) else float("nan"),
            delta_ppl=round(delta_ppl, 4) if not np.isnan(delta_ppl) else float("nan"),
            ppl_ratio=round(ppl_ratio, 4) if not np.isnan(ppl_ratio) else float("nan"),
            cosine_sim=round(cos_sim, 4),
            kl_divergence=round(kl_div, 4),
            js_divergence=round(js_div, 4),
            entropy_baseline=round(ent_b, 4),
            entropy_steered=round(ent_s, 4),
            delta_entropy=round(delta_ent, 4),
            avg_shift_magnitude=round(avg_shift, 4),
            steering_strength_score=round(strength_score, 4),
            layerwise_cosine_sim=l_cos,
            layerwise_norm_diff=l_norm_diff,
            layerwise_shift_magnitude=l_shift_mag,
        )
        logger.info("Full evaluation report generated: PPL_ratio=%.4f, KL=%.4f, JSD=%.4f", report.ppl_ratio, report.kl_divergence, report.js_divergence)
        return report

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
        Compute consolidated evaluation report (Backward-compatible dict format).

        Parameters
        ----------
        model : AutoModelForCausalLM
        tokenizer : AutoTokenizer
        baseline_text : str
        steered_text : str
        concept_vector : torch.Tensor
        device : str, default="cpu"

        Returns
        -------
        Dict[str, float]
        """
        full_rep = SteeringEvaluator.evaluate_full(
            model=model,
            tokenizer=tokenizer,
            prompt="Evaluation",
            baseline_text=baseline_text,
            steered_text=steered_text,
            concept_vector=concept_vector,
            device=device,
        )
        report_dict = full_rep.to_dict()
        # Ensure standard keys for backward compatibility
        return {
            "ppl_baseline": report_dict["ppl_baseline"],
            "ppl_steered": report_dict["ppl_steered"],
            "delta_ppl": report_dict["delta_ppl"],
            "ppl_ratio": report_dict["ppl_ratio"],
            "cosine_sim": report_dict["cosine_sim"],
            "kl_divergence": report_dict["kl_divergence"],
            "js_divergence": report_dict["js_divergence"],
            "entropy_baseline": report_dict["entropy_baseline"],
            "entropy_steered": report_dict["entropy_steered"],
            "delta_entropy": report_dict["delta_entropy"],
            "avg_shift_magnitude": report_dict["avg_shift_magnitude"],
            "steering_strength_score": report_dict["steering_strength_score"],
        }


# ===========================================================================
# PLOTTING UTILITIES
# ===========================================================================

def plot_layerwise_changes(report: SteeringEvaluationReport):
    """
    Generate interactive Plotly figure for layer-wise activation changes.
    """
    import plotly.graph_objects as go

    layers = sorted(report.layerwise_shift_magnitude.keys())
    if not layers:
        # Fallback dummy figure if layerwise dictionary empty
        layers = [10, 11, 12, 13, 14, 15]
        shifts = [0.0] * len(layers)
    else:
        shifts = [report.layerwise_shift_magnitude[l] for l in layers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers,
        y=shifts,
        marker=dict(color='rgb(124, 58, 237)'),
        name="Activation Shift Magnitude (||h_s - h_b||)"
    ))
    fig.update_layout(
        title="Layer-wise Activation Shift Magnitude",
        xaxis_title="Layer Index",
        yaxis_title="Euclidean Shift Magnitude",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_steering_strength(report: SteeringEvaluationReport):
    """
    Generate interactive Plotly figure comparing steering strength scores.
    """
    import plotly.graph_objects as go

    metrics = ["KL Divergence", "JS Divergence", "Steering Strength Score"]
    values = [report.kl_divergence, report.js_divergence, report.steering_strength_score]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker=dict(color=['rgb(79, 70, 229)', 'rgb(124, 58, 237)', 'rgb(16, 185, 129)']),
        text=[f"{v:.4f}" for v in values],
        textposition='auto',
    ))
    fig.update_layout(
        title="Steering Intervention & Divergence Metrics",
        xaxis_title="Metric",
        yaxis_title="Score / Divergence (nats)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_metric_comparison(report: SteeringEvaluationReport):
    """
    Generate interactive Plotly figure comparing Baseline vs. Steered metrics (Perplexity & Entropy).
    """
    import plotly.graph_objects as go

    categories = ["Perplexity (PPL)", "Entropy (H)"]
    baseline_vals = [report.ppl_baseline if not np.isnan(report.ppl_baseline) else 0, report.entropy_baseline]
    steered_vals = [report.ppl_steered if not np.isnan(report.ppl_steered) else 0, report.entropy_steered]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=baseline_vals,
        name="Baseline (Unsteered)",
        marker=dict(color='rgb(156, 163, 175)')
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=steered_vals,
        name="Steered",
        marker=dict(color='rgb(124, 58, 237)')
    ))
    fig.update_layout(
        barmode='group',
        title="Baseline vs. Steered Metric Comparison",
        xaxis_title="Evaluation Metric",
        yaxis_title="Metric Value",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
