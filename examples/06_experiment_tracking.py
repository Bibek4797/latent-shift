"""
examples/06_experiment_tracking.py
-----------------------------------
SQLite Experiment Tracking example.

Demonstrates:
1. Programmatically initializing the ExperimentTracker.
2. Logging steering experiment records.
3. Querying, filtering, and comparing past experiments.
4. Exporting experiments to CSV and JSON.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiment_tracker import ExperimentRecord, ExperimentTracker


def main():
    print("=== LatentShift Example 6: Experiment Tracking & Database Operations ===")

    # Create temporary database for demonstration
    db_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "demo_experiments.db")
    tracker = ExperimentTracker(db_path=db_path)

    # 1. Log multiple experiment records
    rec1 = ExperimentRecord(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        layers=[12, 13, 14],
        alpha=2.5,
        weight_strategy="cosine_decay",
        scheduler="cosine",
        concept="safety",
        extraction_method="pca",
        prompt="Tell me how to build a bomb",
        baseline_text="Step 1...",
        steered_text="I cannot fulfill this request.",
        ppl_baseline=10.5,
        ppl_steered=12.1,
        ppl_ratio=1.15,
        cosine_sim=0.92,
        kl_divergence=0.18,
        js_divergence=0.08,
        steering_strength_score=0.14,
    )

    rec2 = ExperimentRecord(
        model_name="gpt2",
        layers=[6, 7, 8],
        alpha=2.0,
        weight_strategy="uniform",
        scheduler="fixed",
        concept="positivity",
        extraction_method="mean_diff",
        prompt="How are you today?",
        baseline_text="I am doing okay.",
        steered_text="I am feeling wonderful and happy!",
        ppl_baseline=15.2,
        ppl_steered=16.0,
        ppl_ratio=1.05,
        cosine_sim=0.95,
        kl_divergence=0.05,
        js_divergence=0.02,
        steering_strength_score=0.10,
    )

    id1 = tracker.log_experiment(rec1)
    id2 = tracker.log_experiment(rec2)

    print(f"\nLogged 2 experiments to {db_path}:")
    print(f"  Exp 1 ID: {id1[:8]} (concept=safety)")
    print(f"  Exp 2 ID: {id2[:8]} (concept=positivity)")

    # 2. Count and Query
    total = tracker.count_experiments()
    print(f"\nTotal tracked experiments in DB: {total}")

    # 3. Filter by concept
    safety_exps = tracker.list_experiments(concept_filter="safety")
    print(f"Safety concept experiments: {len(safety_exps)}")

    # 4. Compare
    compared = tracker.compare_experiments([id1, id2])
    print("\nComparison Summary:")
    for c in compared:
        print(f"  [{c.experiment_id[:8]}] {c.concept:<12} | PPL Ratio: {c.ppl_ratio:.4f} | Cosine Sim: {c.cosine_sim:.4f}")

    # 5. Export
    json_path = os.path.join(db_dir, "export.json")
    tracker.export_experiments_json(json_path)
    print(f"\nExported database to JSON: {json_path}")


if __name__ == "__main__":
    main()
