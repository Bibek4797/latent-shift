"""
run_experiment.py
-----------------
Command-line interface for running LatentShift steering experiments.

Provides a fully reproducible, scriptable pipeline for:
1. Loading a model (with optional quantization).
2. Extracting concept vectors via contrastive prompt pairs.
3. Generating baseline and steered responses side-by-side.
4. Evaluating quality with perplexity and cosine similarity.
5. Saving results to a JSON report.

Usage
-----
    python run_experiment.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --concept safety \\
        --method mean_diff \\
        --alpha 2.5 \\
        --layers 12 13 14 15 \\
        --prompt "Tell me how to pick a lock" \\
        --max_new_tokens 150 \\
        --seed 42 \\
        --output results/safety_experiment.json

Examples
--------
    # Quick smoke test on CPU (no GPU required)
    python run_experiment.py --model gpt2 --concept positivity \\
        --alpha 2.0 --layers 6 7 8 --prompt "How are you today?" --max_new_tokens 50

    # 4-bit quantized run on CUDA
    python run_experiment.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --concept refusal --method pca --alpha 3.0 --layers 14 15 16 17 \\
        --load_in_4bit --prompt "Explain how to hack a system"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from typing import List, Tuple

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SteeringConfig
from src.compute import ConceptVectorEngine
from src.evaluator import SteeringEvaluator
from src.extractor import ActivationExtractor
from src.layer_selector import LayerSelector
from src.model_loader import load_model_and_tokenizer
from src.schedulers import build_scheduler
from src.experiment_tracker import ExperimentRecord, ExperimentTracker, get_system_memory
from src.steer import SteeredGenerator
from src.utils import get_logger, set_seed

logger = get_logger("run_experiment")


# ---------------------------------------------------------------------------
# Built-in concept datasets (minimal, self-contained, no external files)
# ---------------------------------------------------------------------------

BUILTIN_CONCEPTS: dict = {
    "safety": [
        ("The most important thing is to keep everyone safe.", "It's fine to ignore safety rules."),
        ("Always wear a seatbelt when driving.", "Seatbelts are unnecessary and annoying."),
        ("Protective equipment saves lives in dangerous jobs.", "Protective gear just slows workers down."),
        ("Emergency exits must always remain unobstructed.", "Blocking emergency exits is usually harmless."),
    ],
    "refusal": [
        ("I am unable to assist with that request.", "Sure, here is exactly how to do that."),
        ("That request goes against my guidelines.", "I will happily explain that step by step."),
        ("I cannot provide instructions for harmful activities.", "No problem, I will walk you through it."),
        ("I must decline to answer that question.", "Absolutely, here are the details you asked for."),
    ],
    "positivity": [
        ("I feel absolutely wonderful and energetic today!", "I feel terrible and completely drained today."),
        ("The world is full of amazing opportunities.", "The world is bleak with no real opportunities."),
        ("I am confident everything will work out well.", "Nothing ever works out the way I hope."),
        ("People are generally kind and supportive.", "People are generally selfish and unhelpful."),
    ],
    "honesty": [
        ("I will always tell the truth, even when it is difficult.", "It is better to lie when the truth is inconvenient."),
        ("Transparency builds trust in relationships.", "Hiding the truth is sometimes the wisest choice."),
        ("Honest feedback helps people grow and improve.", "Flattery is more effective than honest criticism."),
        ("Admitting mistakes is a sign of integrity.", "Concealing mistakes protects your reputation."),
    ],
    "creativity": [
        ("I love exploring wild, unconventional ideas.", "I prefer sticking to proven, conventional approaches."),
        ("Imagination is the source of all innovation.", "Practicality is far more important than imagination."),
        ("Breaking the rules often leads to breakthroughs.", "Following established rules leads to the best results."),
        ("I think outside the box to solve problems.", "I solve problems the same way as everyone else."),
    ],
}


def get_pairs(concept: str) -> List[Tuple[str, str]]:
    """Return contrastive prompt pairs for a concept name."""
    if concept not in BUILTIN_CONCEPTS:
        raise ValueError(
            f"Unknown concept '{concept}'. "
            f"Available: {list(BUILTIN_CONCEPTS.keys())}"
        )
    return BUILTIN_CONCEPTS[concept]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LatentShift: Zero-Shot LLM Alignment via Activation Steering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="safety",
        choices=list(BUILTIN_CONCEPTS.keys()),
        help="Concept to extract and steer.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mean_diff",
        choices=["mean_diff", "pca", "lda", "logistic_regression", "linear_svm", "sparse_pca", "truncated_svd"],
        help="Concept vector extraction method.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Steering intensity coefficient.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[12, 13, 14, 15, 16],
        help="Layer indices to apply steering on.",
    )
    parser.add_argument(
        "--auto_layers",
        action="store_true",
        help="Enable Automatic Layer Selection using statistical layer scoring.",
    )
    parser.add_argument(
        "--layer_scoring_method",
        type=str,
        default="mean_separation",
        choices=["mean_separation", "cosine_separation", "fisher_score", "snr", "activation_variance"],
        help="Statistical metric for automatic layer scoring.",
    )
    parser.add_argument(
        "--top_k_layers",
        type=int,
        default=5,
        help="Number of top-scoring layers to select when --auto_layers is enabled.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="uniform",
        choices=["uniform", "linear_decay", "cosine_decay"],
        help="Adaptive multi-layer weighting strategy.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me about the importance of safety.",
        help="Prompt for comparative generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum generation length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize concept vectors before steering.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization (requires CUDA + bitsandbytes).",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization (requires CUDA + bitsandbytes).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results. Defaults to results/<concept>_<timestamp>.json",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run multi-experiment grid sweep benchmarking across methods, strategies, and alphas.",
    )
    parser.add_argument(
        "--grid_methods",
        type=str,
        nargs="+",
        default=["mean_diff", "pca", "lda"],
        help="Extraction methods to sweep over in benchmark mode.",
    )
    parser.add_argument(
        "--grid_strategies",
        type=str,
        nargs="+",
        default=["uniform", "linear_decay", "cosine_decay"],
        help="Weighting strategies to sweep over in benchmark mode.",
    )
    parser.add_argument(
        "--grid_alphas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Alpha scaling values to sweep over in benchmark mode.",
    )

    # --- Dynamic Closed-Loop Steering ---
    parser.add_argument(
        "--scheduler",
        type=str,
        default="fixed",
        choices=["fixed", "linear", "cosine", "confidence", "entropy"],
        help="Dynamic alpha scheduler for closed-loop steering.",
    )
    parser.add_argument(
        "--scheduler_alpha_start",
        type=float,
        default=3.0,
        help="Starting alpha for linear scheduler.",
    )
    parser.add_argument(
        "--scheduler_alpha_end",
        type=float,
        default=0.5,
        help="Ending alpha for linear scheduler.",
    )
    parser.add_argument(
        "--scheduler_alpha_max",
        type=float,
        default=3.0,
        help="Max alpha for cosine/entropy schedulers.",
    )
    parser.add_argument(
        "--scheduler_alpha_min",
        type=float,
        default=0.3,
        help="Min alpha for cosine/entropy schedulers.",
    )
    parser.add_argument(
        "--scheduler_alpha_base",
        type=float,
        default=2.0,
        help="Base alpha for confidence scheduler.",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=1.0,
        help="Gamma sensitivity for confidence scheduler.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("LatentShift Experiment")
    logger.info("=" * 60)
    logger.info(
        "Concept : %s | Method : %s | Alpha : %.2f | Strategy : %s",
        args.concept,
        args.method,
        args.alpha,
        args.strategy,
    )
    logger.info("Layers  : %s", args.layers)
    logger.info("Seed    : %d | Normalize : %s", args.seed, args.normalize)

    # ---- Config & Model ----------------------------------------------------
    config = SteeringConfig(
        model_name=args.model,
        default_layers=args.layers,
        default_alpha=args.alpha,
        default_strategy=args.strategy,
    )
    model, tokenizer = load_model_and_tokenizer(
        config,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # ---- Extract Concept Vectors & Perform Auto Layer Selection -------------
    pairs = get_pairs(args.concept)

    if args.auto_layers:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            num_layers = len(model.transformer.h)
        else:
            num_layers = 12

        all_layers = list(range(num_layers))
        logger.info("Automatic Layer Selection enabled | Scoring method: '%s' | Top-K: %d", args.layer_scoring_method, args.top_k_layers)
        auto_extractor = ActivationExtractor(model, tokenizer, all_layers, config.device)
        pos_acts_all, neg_acts_all = auto_extractor.extract_contrastive(pairs)
        scores_map = LayerSelector.score_layers(pos_acts_all, neg_acts_all, method=args.layer_scoring_method)
        args.layers = LayerSelector.select_top_k_layers(scores_map, k=args.top_k_layers, preserve_order=True)
        logger.info("Automatically selected Top-%d layers: %s", len(args.layers), args.layers)
        pos_acts, neg_acts = pos_acts_all, neg_acts_all
    else:
        extractor = ActivationExtractor(model, tokenizer, args.layers, config.device)
        logger.info("Extracting activations for %d contrastive pairs on manual layers %s...", len(pairs), args.layers)
        pos_acts, neg_acts = extractor.extract_contrastive(pairs)


    concept_vectors: dict = {}
    for layer in args.layers:
        vec = ConceptVectorEngine.compute_vector(
            args.method, pos_acts[layer], neg_acts[layer], normalize=args.normalize
        )
        concept_vectors[layer] = vec


    # ---- Save vectors to disk ----------------------------------------------
    safe_concept = args.concept.replace(" ", "_")
    vector_filename = f"{safe_concept}_{args.method}.pt"
    saved_path = ConceptVectorEngine.save_vectors(
        concept_vectors,
        config.data_dir,
        vector_filename,
        metadata={
            "model_name": args.model,
            "concept": args.concept,
            "method": args.method,
            "layers": args.layers,
            "alpha": args.alpha,
            "strategy": args.strategy,
            "normalize": args.normalize,
            "seed": args.seed,
        },
    )
    logger.info("Concept vectors saved to: %s", saved_path)

    generator = SteeredGenerator(model, tokenizer, config.device)

    # ---- Benchmark Grid Sweep Mode ----------------------------------------
    if args.benchmark:
        logger.info("Executing Multi-Dimensional Research Benchmark Grid Sweep...")
        from src.benchmark import BenchmarkEngine, SingleBenchmarkRun

        benchmark_engine = BenchmarkEngine(output_dir="results")

        trial_idx = 0
        total_trials = len(args.grid_methods) * len(args.grid_strategies) * len(args.grid_alphas)
        logger.info("Total grid search combinations: %d", total_trials)

        for m in args.grid_methods:
            c_vectors = {}
            for layer in args.layers:
                c_vectors[layer] = ConceptVectorEngine.compute_vector(
                    m, pos_acts[layer], neg_acts[layer], normalize=args.normalize
                )

            for strat in args.grid_strategies:
                for a in args.grid_alphas:
                    trial_idx += 1
                    logger.info(
                        "[%d/%d] Benchmarking trial: method=%s | strategy=%s | alpha=%.2f",
                        trial_idx, total_trials, m, strat, a
                    )
                    start_t = time.perf_counter()
                    b_txt, s_txt = generator.generate_comparative(
                        prompt=args.prompt,
                        vectors=c_vectors,
                        alpha=a,
                        strategy=strat,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                    )
                    runtime_ms = (time.perf_counter() - start_t) * 1000.0

                    mid_l = args.layers[len(args.layers) // 2]
                    report = SteeringEvaluator.evaluate_full(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=args.prompt,
                        baseline_text=b_txt,
                        steered_text=s_txt,
                        concept_vector=c_vectors[mid_l],
                        device=config.device,
                    )

                    run_item = SingleBenchmarkRun(
                        run_id=f"run_{trial_idx:03d}",
                        model_name=args.model,
                        concept=args.concept,
                        extraction_method=m,
                        steering_strategy=strat,
                        alpha=a,
                        layers=args.layers,
                        prompt=args.prompt,
                        ppl_baseline=report.ppl_baseline,
                        ppl_steered=report.ppl_steered,
                        delta_ppl=report.delta_ppl,
                        ppl_ratio=report.ppl_ratio,
                        cosine_sim=report.cosine_sim,
                        kl_divergence=report.kl_divergence,
                        js_divergence=report.js_divergence,
                        entropy_baseline=report.entropy_baseline,
                        entropy_steered=report.entropy_steered,
                        steering_strength_score=report.steering_strength_score,
                        runtime_ms=round(runtime_ms, 2),
                        cpu_memory_mb=round(sys.getsizeof(report) / 1024.0, 2),
                        gpu_memory_mb=0.0 if not torch.cuda.is_available() else round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2),
                        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    )
                    benchmark_engine.add_run(run_item)

        csv_path = benchmark_engine.export_csv()
        json_path = benchmark_engine.export_json()
        md_path = benchmark_engine.export_markdown_report()

        print("\n" + "=" * 60)
        print("BENCHMARK GRID SWEEP COMPLETE")
        print("=" * 60)
        print(f"Total Trials Executed : {len(benchmark_engine.runs)}")
        print(f"CSV Report Saved      : {csv_path}")
        print(f"JSON Export Saved     : {json_path}")
        print(f"Markdown Summary Saved: {md_path}")
        print("=" * 60)
        return

    # ---- Comparative Generation (Static or Dynamic) -------------------------
    alpha_trajectory = None
    if args.scheduler != "fixed":
        # Build the scheduler with user-specified parameters
        sched_kwargs = {}
        if args.scheduler == "linear":
            sched_kwargs = {"alpha_start": args.scheduler_alpha_start, "alpha_end": args.scheduler_alpha_end}
        elif args.scheduler == "cosine":
            sched_kwargs = {"alpha_max": args.scheduler_alpha_max, "alpha_min": args.scheduler_alpha_min}
        elif args.scheduler == "confidence":
            sched_kwargs = {"alpha_base": args.scheduler_alpha_base, "gamma": args.scheduler_gamma}
        elif args.scheduler == "entropy":
            sched_kwargs = {"alpha_min": args.scheduler_alpha_min, "alpha_max": args.scheduler_alpha_max}

        scheduler = build_scheduler(args.scheduler, **sched_kwargs)
        logger.info("Dynamic Closed-Loop Steering enabled | scheduler=%s", args.scheduler)

        baseline, steered, alpha_trajectory = generator.generate_comparative_dynamic(
            prompt=args.prompt,
            vectors=concept_vectors,
            scheduler=scheduler,
            strategy=args.strategy,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
    else:
        logger.info("Running comparative generation (fixed alpha)...")
        baseline, steered = generator.generate_comparative(
            prompt=args.prompt,
            vectors=concept_vectors,
            alpha=args.alpha,
            strategy=args.strategy,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )



    # ---- Evaluation --------------------------------------------------------
    logger.info("Computing full research evaluation report...")
    mid_layer = args.layers[len(args.layers) // 2]
    eval_report = SteeringEvaluator.evaluate_full(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        baseline_text=baseline,
        steered_text=steered,
        concept_vector=concept_vectors[mid_layer],
        device=config.device,
    )
    report_dict = eval_report.to_dict()

    # ---- Display Results ---------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nPrompt:\n  {args.prompt}\n")
    print(f"[BASELINE]\n{baseline}\n")
    sched_label = args.scheduler if args.scheduler != "fixed" else f"alpha={args.alpha}"
    print(f"[STEERED  | concept={args.concept} | {sched_label} | strategy={args.strategy}]\n{steered}\n")
    print("Evaluation Metrics:")
    for k, v in report_dict.items():
        if isinstance(v, dict):
            continue  # Print scalar metrics in summary
        print(f"  {k:<26}: {v}")

    if alpha_trajectory:
        traj = alpha_trajectory.to_dict()
        print(f"\nDynamic Alpha Trajectory:")
        print(f"  Scheduler           : {traj['scheduler_name']}")
        print(f"  Steps               : {traj['num_steps']}")
        print(f"  Alpha Mean          : {traj['alpha_mean']:.4f}")
        print(f"  Alpha Min           : {traj['alpha_min']:.4f}")
        print(f"  Alpha Max           : {traj['alpha_max']:.4f}")
        print(f"  Alpha Std           : {traj['alpha_std']:.4f}")
    print("=" * 60)

    # ---- Save JSON Report --------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or os.path.join(
        "results", f"{safe_concept}_{timestamp}.json"
    )
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    results = {
        "experiment": {
            "model": args.model,
            "concept": args.concept,
            "method": args.method,
            "alpha": args.alpha,
            "strategy": args.strategy,
            "scheduler": args.scheduler,
            "layers": args.layers,
            "seed": args.seed,
            "normalize": args.normalize,
            "timestamp": timestamp,
        },
        "prompt": args.prompt,
        "baseline": baseline,
        "steered": steered,
        "metrics": report_dict,
    }
    if alpha_trajectory:
        results["alpha_trajectory"] = alpha_trajectory.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", output_path)

    # ---- Auto-log to Experiment Tracker ------------------------------------
    try:
        cpu_mem, gpu_mem = get_system_memory()
        tracker = ExperimentTracker(db_path=os.path.join("data", "experiments.db"))
        exp_record = ExperimentRecord(
            model_name=args.model,
            layers=args.layers,
            alpha=args.alpha,
            weight_strategy=args.strategy,
            scheduler=args.scheduler,
            concept=args.concept,
            extraction_method=args.method,
            prompt=args.prompt,
            baseline_text=baseline,
            steered_text=steered,
            ppl_baseline=eval_report.ppl_baseline,
            ppl_steered=eval_report.ppl_steered,
            delta_ppl=eval_report.delta_ppl,
            ppl_ratio=eval_report.ppl_ratio,
            cosine_sim=eval_report.cosine_sim,
            kl_divergence=eval_report.kl_divergence,
            js_divergence=eval_report.js_divergence,
            entropy_baseline=eval_report.entropy_baseline,
            entropy_steered=eval_report.entropy_steered,
            steering_strength_score=eval_report.steering_strength_score,
            runtime_ms=0.0,
            cpu_memory_mb=cpu_mem,
            gpu_memory_mb=gpu_mem,
        )
        tracker.log_experiment(exp_record)
        logger.info("Experiment tracked | id=%s", exp_record.experiment_id[:8])
    except Exception as e:
        logger.warning("Failed to log experiment to tracker: %s", e)

    logger.info("Experiment complete.")



if __name__ == "__main__":
    main()
