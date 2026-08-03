"""
examples/01_basic_steering.py
------------------------------
Basic zero-shot activation steering example using LatentShift.

Demonstrates:
1. Loading model & tokenizer (using GPT-2 or Mock Model).
2. Extracting a concept vector with Mean Difference.
3. Comparative text generation (Baseline vs. Steered).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model_loader import load_model_and_tokenizer
from src.extractor import ActivationExtractor
from src.compute import ConceptVectorEngine
from src.steer import SteeredGenerator


def main():
    print("=== LatentShift Example 1: Basic Activation Steering ===")

    # 1. Load model and tokenizer
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Define contrastive prompt sets for positivity steering
    pos_prompts = [
        "I am feeling joyful, enthusiastic, and wonderfully optimistic today!",
        "Life is bright, full of hope, and overflowing with happiness.",
        "Everything is wonderful and great things are happening!",
    ]
    neg_prompts = [
        "I am feeling miserable, depressed, and deeply hopeless today.",
        "Life is dark, painful, and filled with disappointment.",
        "Everything is terrible and bad things keep happening.",
    ]

    target_layers = [6, 7, 8]

    # 3. Extract activations
    print(f"Extracting activations across layers {target_layers}...")
    extractor = ActivationExtractor(model, tokenizer, target_layers, device=device)
    pos_acts, neg_acts = extractor.extract_contrastive(list(zip(pos_prompts, neg_prompts)))

    # 4. Compute concept vectors using Mean Difference
    vectors = {}
    for layer in target_layers:
        vectors[layer] = ConceptVectorEngine.compute_mean_difference(
            pos_acts[layer], neg_acts[layer], normalize=True
        )
    print("Concept vectors computed and normalized.")

    # 5. Comparative generation
    prompt = "The weather outside is rainy, but"
    print(f"\nPrompt: '{prompt}'")

    generator = SteeredGenerator(model, tokenizer, device=device)
    baseline, steered = generator.generate_comparative(
        prompt=prompt,
        vectors=vectors,
        alpha=2.5,
        strategy="uniform",
        max_new_tokens=40,
        do_sample=False,
    )

    print("\n" + "=" * 50)
    print(f"[BASELINE]:\n{baseline}\n")
    print(f"[STEERED] (alpha=2.5):\n{steered}")
    print("=" * 50)


if __name__ == "__main__":
    main()
