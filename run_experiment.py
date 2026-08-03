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
from src.model_loader import load_model_and_tokenizer
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
        choices=["mean_diff", "pca"],
        help="Concept vector computation method.",
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

    # ---- Extract Concept Vectors -------------------------------------------
    pairs = get_pairs(args.concept)
    extractor = ActivationExtractor(model, tokenizer, args.layers, config.device)
    logger.info("Extracting activations for %d contrastive pairs...", len(pairs))
    pos_acts, neg_acts = extractor.extract_contrastive(pairs)

    concept_vectors: dict = {}
    for layer in args.layers:
        if args.method == "pca":
            vec = ConceptVectorEngine.compute_pca_vector(
                pos_acts[layer], neg_acts[layer], normalize=args.normalize
            )
        else:
            vec = ConceptVectorEngine.compute_mean_difference(
                pos_acts[layer], neg_acts[layer], normalize=args.normalize
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

    # ---- Comparative Generation --------------------------------------------
    generator = SteeredGenerator(model, tokenizer, config.device)
    logger.info("Running comparative generation...")
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
    print(f"[STEERED  | concept={args.concept} | alpha={args.alpha} | strategy={args.strategy}]\n{steered}\n")
    print("Evaluation Metrics:")
    for k, v in report_dict.items():
        if isinstance(v, dict):
            continue  # Print scalar metrics in summary
        print(f"  {k:<26}: {v}")
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", output_path)
    logger.info("Experiment complete.")



if __name__ == "__main__":
    main()
