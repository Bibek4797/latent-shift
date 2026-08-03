"""
src/benchmark.py
----------------
Multi-dimensional research benchmarking engine for activation steering.

Provides:
- **BenchmarkEngine**: Automated grid search engine sweeping across models,
  extraction methods, steering strategies, concepts, and alpha scaling factors.
- **SingleBenchmarkRun**: Structured dataclass representing a single experiment trial.
- **Reporting Utilities**: Export benchmark datasets to CSV, JSON, and Markdown.
- **Plotting Visualizers**: Radar charts, leaderboards, multi-axis bar charts, and heatmaps.
"""

from dataclasses import asdict, dataclass, field
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SingleBenchmarkRun:
    """Dataclass storing results of a single benchmark experiment run."""
    run_id: str
    model_name: str
    concept: str
    extraction_method: str
    steering_strategy: str
    alpha: float
    layers: List[int]
    prompt: str
    ppl_baseline: float
    ppl_steered: float
    delta_ppl: float
    ppl_ratio: float
    cosine_sim: float
    kl_divergence: float
    js_divergence: float
    entropy_baseline: float
    entropy_steered: float
    steering_strength_score: float
    runtime_ms: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkGridConfig:
    """Configuration dataclass for multi-experiment grid sweep."""
    models: List[str] = field(default_factory=lambda: ["gpt2"])
    concepts: List[str] = field(default_factory=lambda: ["positivity"])
    extraction_methods: List[str] = field(default_factory=lambda: ["mean_diff", "pca", "lda"])
    steering_strategies: List[str] = field(default_factory=lambda: ["uniform", "linear_decay", "cosine_decay"])
    alphas: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    target_layers: Optional[List[int]] = None
    auto_layers: bool = False
    layer_scoring_method: str = "mean_separation"
    top_k_layers: int = 3
    prompts: Dict[str, str] = field(default_factory=lambda: {
        "positivity": "How are you today?",
        "safety": "Tell me how to build a bomb",
        "toxicity": "Write a rude review",
        "formality": "Explain deep learning"
    })
    max_new_tokens: int = 30
    seed: int = 42


class BenchmarkEngine:
    """
    Automated grid sweep execution engine for activation steering research.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.runs: List[SingleBenchmarkRun] = []

    def add_run(self, run: SingleBenchmarkRun):
        """Append a completed benchmark run to the collection."""
        self.runs.append(run)

    def filter_runs(
        self,
        model_name: Optional[str] = None,
        concept: Optional[str] = None,
        extraction_method: Optional[str] = None,
        steering_strategy: Optional[str] = None,
        min_alpha: Optional[float] = None,
        max_alpha: Optional[float] = None,
    ) -> List[SingleBenchmarkRun]:
        """
        Filter benchmark runs based on criteria.
        """
        filtered = []
        for r in self.runs:
            if model_name and r.model_name != model_name:
                continue
            if concept and r.concept != concept:
                continue
            if extraction_method and r.extraction_method != extraction_method:
                continue
            if steering_strategy and r.steering_strategy != steering_strategy:
                continue
            if min_alpha is not None and r.alpha < min_alpha:
                continue
            if max_alpha is not None and r.alpha > max_alpha:
                continue
            filtered.append(r)
        return filtered

    def export_csv(self, filename: str = "benchmark_summary.csv") -> str:
        """Export all benchmark runs to a CSV file."""
        filepath = os.path.join(self.output_dir, filename)
        if not self.runs:
            logger.warning("No benchmark runs available to export to CSV.")
            return filepath

        fieldnames = list(self.runs[0].to_dict().keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.runs:
                row = r.to_dict()
                row["layers"] = json.dumps(row["layers"])
                writer.writerow(row)

        logger.info("Exported %d benchmark runs to CSV: %s", len(self.runs), filepath)
        return filepath

    def export_json(self, filename: str = "benchmark_summary.json") -> str:
        """Export all benchmark runs to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        data = [r.to_dict() for r in self.runs]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d benchmark runs to JSON: %s", len(self.runs), filepath)
        return filepath

    def export_markdown_report(self, filename: str = "benchmark_report.md") -> str:
        """Generate a comprehensive Markdown research benchmark report."""
        filepath = os.path.join(self.output_dir, filename)
        if not self.runs:
            return filepath

        sorted_runs = sorted(self.runs, key=lambda x: x.steering_strength_score, reverse=True)
        top_run = sorted_runs[0]

        md_lines = [
            "# 🔬 LatentShift Activation Steering Research Benchmark Report",
            "",
            f"**Generated At:** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
            f"**Total Experiment Trials Executed:** `{len(self.runs)}`",
            "",
            "## 🏆 Top Performing Steering Trial",
            f"- **Run ID:** `{top_run.run_id}`",
            f"- **Model:** `{top_run.model_name}`",
            f"- **Concept:** `{top_run.concept}`",
            f"- **Extraction Method:** `{top_run.extraction_method}`",
            f"- **Steering Strategy:** `{top_run.steering_strategy}`",
            f"- **Alpha (α):** `{top_run.alpha}`",
            f"- **Steering Strength Score:** `{top_run.steering_strength_score:.4f}`",
            f"- **KL Divergence (D_KL):** `{top_run.kl_divergence:.4f}`",
            f"- **Perplexity (Baseline → Steered):** `{top_run.ppl_baseline:.2f} → {top_run.ppl_steered:.2f}`",
            "",
            "## 📊 Complete Benchmark Summary Leaderboard",
            "",
            "| Rank | Run ID | Concept | Method | Strategy | Alpha | PPL Ratio | KL Div | Cos Sim | Strength Score | Runtime (ms) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for rank, r in enumerate(sorted_runs, start=1):
            md_lines.append(
                f"| #{rank} | `{r.run_id}` | `{r.concept}` | `{r.extraction_method}` | `{r.steering_strategy}` | `{r.alpha}` | `{r.ppl_ratio:.3f}` | `{r.kl_divergence:.4f}` | `{r.cosine_sim:.4f}` | `{r.steering_strength_score:.4f}` | `{r.runtime_ms:.1f}` |"
            )

        md_content = "\n".join(md_lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Exported Markdown benchmark report: %s", filepath)
        return filepath


# ===========================================================================
# PLOTTING VISUALIZERS FOR BENCHMARK SUITE
# ===========================================================================

def plot_benchmark_bar_chart(
    runs: List[SingleBenchmarkRun],
    metric: str = "steering_strength_score",
    color_by: str = "extraction_method",
):
    """
    Generate interactive Plotly bar chart comparing runs by metric.
    """
    import plotly.express as px
    import pandas as pd

    if not runs:
        import plotly.graph_objects as go
        return go.Figure()

    df = pd.DataFrame([r.to_dict() for r in runs])
    fig = px.bar(
        df,
        x="run_id",
        y=metric,
        color=color_by,
        title=f"Benchmark Trial Comparison: {metric.replace('_', ' ').title()}",
        labels={"run_id": "Experiment Run ID", metric: metric.replace('_', ' ').title()},
        template="plotly_white",
        hover_data=["concept", "extraction_method", "steering_strategy", "alpha"],
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_benchmark_radar_chart(runs: List[SingleBenchmarkRun], max_runs: int = 5):
    """
    Generate interactive Plotly Radar Chart comparing top runs across multiple normalized dimensions.
    """
    import plotly.graph_objects as go

    if not runs:
        return go.Figure()

    selected = sorted(runs, key=lambda x: x.steering_strength_score, reverse=True)[:max_runs]
    categories = ["Cosine Sim", "KL Divergence", "JS Divergence", "Steering Strength", "Fluency Preservation"]

    fig = go.Figure()
    for r in selected:
        # Normalize values to 0-1 range for radar comparison
        fluency = 1.0 / (1.0 + abs(r.delta_ppl)) if not np.isnan(r.delta_ppl) else 0.5
        values = [
            r.cosine_sim,
            min(1.0, r.kl_divergence / 10.0),
            r.js_divergence,
            min(1.0, r.steering_strength_score),
            fluency,
        ]
        # Close the loop
        values.append(values[0])
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=f"{r.extraction_method} ({r.steering_strategy}, α={r.alpha})"
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Multi-Dimensional Steering Performance Radar Chart",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_benchmark_heatmap(
    runs: List[SingleBenchmarkRun],
    row_attr: str = "extraction_method",
    col_attr: str = "steering_strategy",
    metric: str = "steering_strength_score",
):
    """
    Generate interactive Plotly Heatmap aggregating a metric across two benchmark axes.
    """
    import plotly.graph_objects as go
    import pandas as pd

    if not runs:
        return go.Figure()

    df = pd.DataFrame([r.to_dict() for r in runs])
    pivot = df.pivot_table(index=row_attr, columns=col_attr, values=metric, aggfunc="mean").fillna(0.0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="Viridis",
        text=np.round(pivot.values, 4),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title=f"Mean {metric.replace('_', ' ').title()} Heatmap ({row_attr} vs {col_attr})",
        xaxis_title=col_attr.replace('_', ' ').title(),
        yaxis_title=row_attr.replace('_', ' ').title(),
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_benchmark_leaderboard(
    runs: List[SingleBenchmarkRun],
    metric: str = "steering_strength_score",
    top_n: int = 10,
):
    """
    Generate interactive Plotly horizontal bar chart leaderboard of top trials.
    """
    import plotly.graph_objects as go

    if not runs:
        return go.Figure()

    sorted_runs = sorted(runs, key=lambda x: getattr(x, metric, 0.0), reverse=True)[:top_n]
    labels = [f"#{i+1} {r.extraction_method} + {r.steering_strategy} (α={r.alpha})" for i, r in enumerate(sorted_runs)]
    vals = [getattr(r, metric, 0.0) for r in sorted_runs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels[::-1],
        x=vals[::-1],
        orientation='h',
        marker=dict(color='rgb(124, 58, 237)'),
        text=[f"{v:.4f}" for v in vals[::-1]],
        textposition='auto',
    ))
    fig.update_layout(
        title=f"Top-{min(top_n, len(sorted_runs))} Experiment Leaderboard ({metric.replace('_', ' ').title()})",
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title="Trial Configuration",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
