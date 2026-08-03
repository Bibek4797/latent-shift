"""
examples/04_automatic_layer_selection.py
-----------------------------------------
Automatic Layer Selection and Ranking example.

Demonstrates:
1. Scoring all transformer layers using Fisher Score and SNR.
2. Ranking candidate layers by statistical separability.
3. Automatically selecting top-K target layers for steering.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model_loader import load_model_and_tokenizer
from src.extractor import ActivationExtractor
from src.layer_selector import LayerSelector


def main():
    print("=== LatentShift Example 4: Automatic Layer Selection ===")

    model_name = "gpt2"
    model, tokenizer, config = load_model_and_tokenizer(model_name)

    all_layers = list(range(12))

    pos_prompts = [
        "Be extremely polite, formal, respectful, and courteous.",
        "Always express deep gratitude, respect, and politeness.",
    ]
    neg_prompts = [
        "Be rude, aggressive, offensive, and extremely vulgar.",
        "Express contempt, hostility, insult, and disrespect.",
    ]

    print("Extracting activations across all 12 layers...")
    extractor = ActivationExtractor(model, tokenizer, all_layers, device=config.device)
    pos_acts, neg_acts = extractor.extract_contrastive(list(zip(pos_prompts, neg_prompts)))

    selector = LayerSelector(pos_acts, neg_acts)

    # Score using Fisher Score
    fisher_scores = selector.score_layers(method="fisher_score")
    print("\nLayer Ranking (Fisher Score):")
    for res in fisher_scores[:5]:
        print(f"  Rank {res.rank}: Layer {res.layer:<2} | Score = {res.score:.4f}")

    # Select top 3 layers automatically
    top_3_layers = selector.select_top_k(k=3, method="fisher_score")
    print(f"\nAutomatically selected Top-3 Steering Layers: {top_3_layers}")


if __name__ == "__main__":
    main()
