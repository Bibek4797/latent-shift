"""
examples/02_adaptive_multi_layer.py
------------------------------------
Adaptive Multi-Layer Steering example.

Demonstrates:
1. Multi-layer concept vector extraction.
2. Weighting strategies: Uniform, Linear Decay, and Cosine Decay.
3. Comparing outputs across different weighting strategies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model_loader import load_model_and_tokenizer
from src.extractor import ActivationExtractor
from src.compute import ConceptVectorEngine
from src.steer import SteeredGenerator
from src.utils import compute_layer_weights


def main():
    print("=== LatentShift Example 2: Adaptive Multi-Layer Steering ===")

    model_name = "gpt2"
    model, tokenizer, config = load_model_and_tokenizer(model_name)

    target_layers = [4, 5, 6, 7, 8, 9]

    # Compute layer weight distribution for different strategies
    print("\nLayer Weight Distributions (base_alpha=3.0, layers=4-9):")
    for strat in ["uniform", "linear_decay", "cosine_decay"]:
        weights = compute_layer_weights(target_layers, base_alpha=3.0, strategy=strat)
        print(f"  {strat:<15}: {dict(sorted(weights.items()))}")

    # Extract vectors
    pos_prompts = ["I am feeling happy, creative, and full of inspiration."]
    neg_prompts = ["I am feeling uninspired, dull, and completely unmotivated."]

    extractor = ActivationExtractor(model, tokenizer, target_layers, device=config.device)
    pos_acts, neg_acts = extractor.extract_contrastive(list(zip(pos_prompts, neg_prompts)))

    vectors = {
        l: ConceptVectorEngine.compute_pca_vector(pos_acts[l], neg_acts[l], normalize=True)
        for l in target_layers
    }

    # Generate under Cosine Decay strategy
    prompt = "Today I decided to start a new project because"
    generator = SteeredGenerator(model, tokenizer, device=config.device)

    baseline, steered = generator.generate_comparative(
        prompt=prompt,
        vectors=vectors,
        alpha=3.0,
        strategy="cosine_decay",
        max_new_tokens=40,
        do_sample=False,
    )

    print("\n" + "=" * 50)
    print(f"Prompt: '{prompt}'")
    print(f"⚪ BASELINE:\n{baseline}\n")
    print(f"🔮 STEERED (Cosine Decay, alpha=3.0):\n{steered}")
    print("=" * 50)


if __name__ == "__main__":
    main()
