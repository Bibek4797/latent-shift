"""
src/experiment_tracker.py
-------------------------
SQLite-backed experiment tracking system for LatentShift.

Provides persistent storage, browsing, reloading, and comparison of
steering experiments. Every experiment automatically captures:

- Model configuration (name, layers, alpha, strategy, scheduler)
- Concept extraction settings (concept, method, prompt)
- Full evaluation metrics (PPL, KL, JS, cosine, entropy, etc.)
- Runtime and memory consumption
- Git commit hash (auto-detected)
- Timestamp

Key classes:
    - ``ExperimentRecord``: Immutable snapshot of a single experiment.
    - ``ExperimentTracker``: SQLite CRUD interface for experiment records.

Plotly visualisation helpers:
    - ``plot_experiment_timeline``: Scatter chart of experiments over time.
    - ``plot_experiment_comparison``: Grouped bar chart for metric comparison.
    - ``plot_experiment_radar``: Radar chart for multi-metric comparison.
"""

import csv
import json
import os
import sqlite3
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils import get_logger

logger = get_logger(__name__)


# ===========================================================================
# HELPERS
# ===========================================================================

def get_git_commit() -> str:
    """
    Retrieve the current short git commit hash.

    Returns
    -------
    str
        7-character short hash, or ``''`` if not in a git repo or git
        is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def get_system_memory() -> Tuple[float, float]:
    """
    Snapshot current process CPU memory and GPU memory.

    Returns
    -------
    cpu_mb : float
        Resident set size in MB (via ``psutil``). 0.0 if unavailable.
    gpu_mb : float
        Peak CUDA memory allocated in MB. 0.0 if CUDA unavailable.
    """
    cpu_mb = 0.0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_mb = round(process.memory_info().rss / (1024.0 * 1024.0), 2)
    except ImportError:
        pass
    except Exception:
        pass

    gpu_mb = 0.0
    if torch.cuda.is_available():
        try:
            gpu_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
        except Exception:
            pass

    return cpu_mb, gpu_mb


# ===========================================================================
# EXPERIMENT RECORD
# ===========================================================================

@dataclass
class ExperimentRecord:
    """
    Immutable snapshot of a single steering experiment.

    Parameters
    ----------
    experiment_id : str
        Unique UUID4 identifier (auto-generated if empty).
    model_name : str
        Hugging Face model ID.
    layers : List[int]
        Target transformer layer indices.
    alpha : float
        Base steering intensity coefficient.
    weight_strategy : str
        Adaptive weighting strategy (uniform, linear_decay, cosine_decay).
    scheduler : str
        Dynamic alpha scheduler name (fixed, linear, cosine, etc.).
    concept : str
        Steering concept name.
    extraction_method : str
        Concept vector extraction method.
    prompt : str
        Input prompt text.
    baseline_text : str
        Generated baseline (unsteered) text.
    steered_text : str
        Generated steered text.
    ppl_baseline, ppl_steered, delta_ppl, ppl_ratio : float
        Perplexity metrics.
    cosine_sim : float
        Cosine similarity between baseline and steered embeddings.
    kl_divergence, js_divergence : float
        Distribution divergence metrics.
    entropy_baseline, entropy_steered : float
        Token entropy metrics.
    steering_strength_score : float
        Normalised steering intensity score.
    runtime_ms : float
        Experiment wall-clock time in milliseconds.
    cpu_memory_mb, gpu_memory_mb : float
        Memory consumption snapshots.
    timestamp : str
        ISO-format UTC timestamp.
    git_commit : str
        Short git commit hash (auto-detected).
    notes : str
        Optional free-text annotation.
    """
    experiment_id: str = ""
    model_name: str = ""
    layers: List[int] = field(default_factory=list)
    alpha: float = 0.0
    weight_strategy: str = "uniform"
    scheduler: str = "fixed"
    concept: str = ""
    extraction_method: str = ""
    prompt: str = ""
    baseline_text: str = ""
    steered_text: str = ""
    ppl_baseline: float = 0.0
    ppl_steered: float = 0.0
    delta_ppl: float = 0.0
    ppl_ratio: float = 0.0
    cosine_sim: float = 0.0
    kl_divergence: float = 0.0
    js_divergence: float = 0.0
    entropy_baseline: float = 0.0
    entropy_steered: float = 0.0
    steering_strength_score: float = 0.0
    runtime_ms: float = 0.0
    cpu_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    timestamp: str = ""
    git_commit: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if not self.git_commit:
            self.git_commit = get_git_commit()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with layers as list."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ===========================================================================
# SQLITE COLUMN DEFINITION
# ===========================================================================

_COLUMNS = [
    "experiment_id", "model_name", "layers", "alpha", "weight_strategy",
    "scheduler", "concept", "extraction_method", "prompt", "baseline_text",
    "steered_text", "ppl_baseline", "ppl_steered", "delta_ppl", "ppl_ratio",
    "cosine_sim", "kl_divergence", "js_divergence", "entropy_baseline",
    "entropy_steered", "steering_strength_score", "runtime_ms",
    "cpu_memory_mb", "gpu_memory_mb", "timestamp", "git_commit", "notes",
]

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id     TEXT PRIMARY KEY,
    model_name        TEXT NOT NULL,
    layers            TEXT NOT NULL,
    alpha             REAL NOT NULL,
    weight_strategy   TEXT NOT NULL,
    scheduler         TEXT NOT NULL DEFAULT 'fixed',
    concept           TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    baseline_text     TEXT NOT NULL DEFAULT '',
    steered_text      TEXT NOT NULL DEFAULT '',
    ppl_baseline      REAL NOT NULL DEFAULT 0.0,
    ppl_steered       REAL NOT NULL DEFAULT 0.0,
    delta_ppl         REAL NOT NULL DEFAULT 0.0,
    ppl_ratio         REAL NOT NULL DEFAULT 0.0,
    cosine_sim        REAL NOT NULL DEFAULT 0.0,
    kl_divergence     REAL NOT NULL DEFAULT 0.0,
    js_divergence     REAL NOT NULL DEFAULT 0.0,
    entropy_baseline  REAL NOT NULL DEFAULT 0.0,
    entropy_steered   REAL NOT NULL DEFAULT 0.0,
    steering_strength_score REAL NOT NULL DEFAULT 0.0,
    runtime_ms        REAL NOT NULL DEFAULT 0.0,
    cpu_memory_mb     REAL NOT NULL DEFAULT 0.0,
    gpu_memory_mb     REAL NOT NULL DEFAULT 0.0,
    timestamp         TEXT NOT NULL,
    git_commit        TEXT NOT NULL DEFAULT '',
    notes             TEXT NOT NULL DEFAULT ''
);
"""


# ===========================================================================
# EXPERIMENT TRACKER
# ===========================================================================

class ExperimentTracker:
    """
    SQLite-backed persistent experiment tracker.

    Parameters
    ----------
    db_path : str, default="data/experiments.db"
        Path to the SQLite database file. Parent directories are created
        automatically.
    """

    def __init__(self, db_path: str = "data/experiments.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()
        logger.info("ExperimentTracker initialised | db=%s", db_path)

    def _init_db(self) -> None:
        """Create the experiments table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def log_experiment(self, record: ExperimentRecord) -> str:
        """
        Insert a new experiment record into the database.

        Parameters
        ----------
        record : ExperimentRecord
            The experiment data to persist.

        Returns
        -------
        str
            The experiment_id of the inserted record.
        """
        values = (
            record.experiment_id,
            record.model_name,
            json.dumps(record.layers),
            record.alpha,
            record.weight_strategy,
            record.scheduler,
            record.concept,
            record.extraction_method,
            record.prompt,
            record.baseline_text,
            record.steered_text,
            record.ppl_baseline,
            record.ppl_steered,
            record.delta_ppl,
            record.ppl_ratio,
            record.cosine_sim,
            record.kl_divergence,
            record.js_divergence,
            record.entropy_baseline,
            record.entropy_steered,
            record.steering_strength_score,
            record.runtime_ms,
            record.cpu_memory_mb,
            record.gpu_memory_mb,
            record.timestamp,
            record.git_commit,
            record.notes,
        )
        placeholders = ", ".join(["?"] * len(_COLUMNS))
        sql = f"INSERT INTO experiments ({', '.join(_COLUMNS)}) VALUES ({placeholders})"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, values)
            conn.commit()

        logger.info("Experiment logged | id=%s | model=%s | concept=%s",
                     record.experiment_id, record.model_name, record.concept)
        return record.experiment_id

    def _row_to_record(self, row: tuple) -> ExperimentRecord:
        """Convert a raw SQLite row tuple to an ExperimentRecord."""
        return ExperimentRecord(
            experiment_id=row[0],
            model_name=row[1],
            layers=json.loads(row[2]),
            alpha=row[3],
            weight_strategy=row[4],
            scheduler=row[5],
            concept=row[6],
            extraction_method=row[7],
            prompt=row[8],
            baseline_text=row[9],
            steered_text=row[10],
            ppl_baseline=row[11],
            ppl_steered=row[12],
            delta_ppl=row[13],
            ppl_ratio=row[14],
            cosine_sim=row[15],
            kl_divergence=row[16],
            js_divergence=row[17],
            entropy_baseline=row[18],
            entropy_steered=row[19],
            steering_strength_score=row[20],
            runtime_ms=row[21],
            cpu_memory_mb=row[22],
            gpu_memory_mb=row[23],
            timestamp=row[24],
            git_commit=row[25],
            notes=row[26],
        )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """
        Fetch a single experiment by its ID.

        Returns
        -------
        ExperimentRecord or None
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
        model_filter: Optional[str] = None,
        concept_filter: Optional[str] = None,
        method_filter: Optional[str] = None,
    ) -> List[ExperimentRecord]:
        """
        Browse experiments with optional filters.

        Parameters
        ----------
        limit : int
            Maximum records to return.
        offset : int
            Pagination offset.
        model_filter, concept_filter, method_filter : str, optional
            Filter by model name, concept, or extraction method.

        Returns
        -------
        List[ExperimentRecord]
        """
        clauses: List[str] = []
        params: List[Any] = []

        if model_filter:
            clauses.append("model_name = ?")
            params.append(model_filter)
        if concept_filter:
            clauses.append("concept = ?")
            params.append(concept_filter)
        if method_filter:
            clauses.append("extraction_method = ?")
            params.append(method_filter)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM experiments{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def compare_experiments(self, ids: List[str]) -> List[ExperimentRecord]:
        """
        Fetch multiple experiments by IDs for side-by-side comparison.

        Parameters
        ----------
        ids : List[str]
            Experiment IDs to compare.

        Returns
        -------
        List[ExperimentRecord]
        """
        if not ids:
            return []
        placeholders = ", ".join(["?"] * len(ids))
        sql = f"SELECT * FROM experiments WHERE experiment_id IN ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, ids).fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment by ID.

        Returns
        -------
        bool
            True if a record was deleted, False if ID not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Experiment deleted | id=%s", experiment_id)
        return deleted

    def count_experiments(self) -> int:
        """Return total number of tracked experiments."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        return row[0] if row else 0

    def get_unique_values(self, column: str) -> List[str]:
        """
        Get distinct values for a column (for filter dropdowns).

        Parameters
        ----------
        column : str
            Column name (must be in _COLUMNS).

        Returns
        -------
        List[str]
        """
        if column not in _COLUMNS:
            raise ValueError(f"Invalid column '{column}'. Valid: {_COLUMNS}")
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM experiments ORDER BY {column}"
            ).fetchall()
        return [str(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_experiments_csv(
        self, filepath: str, ids: Optional[List[str]] = None
    ) -> str:
        """
        Export experiments to CSV.

        Parameters
        ----------
        filepath : str
            Output CSV file path.
        ids : List[str], optional
            If provided, export only these experiments. Otherwise export all.

        Returns
        -------
        str
            Absolute path to the written CSV file.
        """
        if ids:
            records = self.compare_experiments(ids)
        else:
            records = self.list_experiments(limit=10000)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()
            for rec in records:
                row = rec.to_dict()
                row["layers"] = json.dumps(row["layers"])
                writer.writerow(row)

        logger.info("Exported %d experiments to CSV: %s", len(records), filepath)
        return os.path.abspath(filepath)

    def export_experiments_json(
        self, filepath: str, ids: Optional[List[str]] = None
    ) -> str:
        """
        Export experiments to JSON.

        Parameters
        ----------
        filepath : str
            Output JSON file path.
        ids : List[str], optional
            If provided, export only these experiments. Otherwise export all.

        Returns
        -------
        str
            Absolute path to the written JSON file.
        """
        if ids:
            records = self.compare_experiments(ids)
        else:
            records = self.list_experiments(limit=10000)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = [rec.to_dict() for rec in records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d experiments to JSON: %s", len(records), filepath)
        return os.path.abspath(filepath)


# ===========================================================================
# PLOTLY VISUALISATIONS
# ===========================================================================

def plot_experiment_timeline(records: List[ExperimentRecord]):
    """
    Scatter plot of experiments over time with PPL ratio as the y-axis.

    Parameters
    ----------
    records : List[ExperimentRecord]
        Experiments to plot.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    timestamps = [r.timestamp for r in records]
    ppl_ratios = [r.ppl_ratio for r in records]
    labels = [f"{r.concept} | {r.extraction_method} | α={r.alpha}" for r in records]
    colours = [r.steering_strength_score for r in records]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ppl_ratios,
        mode="markers+text",
        text=[r.experiment_id[:8] for r in records],
        textposition="top center",
        marker=dict(
            size=12,
            color=colours,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Steering Strength"),
        ),
        hovertext=labels,
        hoverinfo="text+y",
    ))
    fig.update_layout(
        title="Experiment Timeline",
        xaxis_title="Timestamp",
        yaxis_title="PPL Ratio (Steered / Baseline)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_experiment_comparison(
    records: List[ExperimentRecord], metric: str = "ppl_ratio"
):
    """
    Grouped bar chart comparing experiments on a chosen metric.

    Parameters
    ----------
    records : List[ExperimentRecord]
        Experiments to compare.
    metric : str
        Field name to compare (e.g., ``ppl_ratio``, ``cosine_sim``, ``kl_divergence``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    ids = [r.experiment_id[:8] for r in records]
    values = [getattr(r, metric, 0.0) for r in records]
    concepts = [r.concept for r in records]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ids,
        y=values,
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
        marker=dict(
            color=values,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title=metric),
        ),
        hovertext=[f"{c} | {r.extraction_method} | α={r.alpha}" for c, r in zip(concepts, records)],
    ))
    fig.update_layout(
        title=f"Experiment Comparison — {metric}",
        xaxis_title="Experiment ID",
        yaxis_title=metric,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_experiment_radar(records: List[ExperimentRecord]):
    """
    Radar chart comparing multiple experiments across key metrics.

    Parameters
    ----------
    records : List[ExperimentRecord]
        Experiments to compare (best with 2–5 records).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    categories = [
        "PPL Ratio", "Cosine Sim", "KL Div",
        "JS Div", "Steering Strength", "Δ Entropy",
    ]

    fig = go.Figure()
    colours = ["rgb(124, 58, 237)", "rgb(239, 68, 68)", "rgb(34, 197, 94)",
               "rgb(59, 130, 246)", "rgb(245, 158, 11)"]

    for i, rec in enumerate(records[:5]):
        values = [
            rec.ppl_ratio,
            rec.cosine_sim,
            rec.kl_divergence,
            rec.js_divergence,
            rec.steering_strength_score,
            abs(rec.entropy_steered - rec.entropy_baseline),
        ]
        # Normalise to 0–1 range per metric for visual clarity
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=f"{rec.experiment_id[:8]} ({rec.concept})",
            line=dict(color=colours[i % len(colours)]),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Multi-Experiment Radar Comparison",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
