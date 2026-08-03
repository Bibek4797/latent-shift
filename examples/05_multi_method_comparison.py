"""
examples/05_multi_method_comparison.py
---------------------------------------
Benchmark and compare all 7 concept vector extraction algorithms.

Demonstrates:
1. Extracting vectors using all 7 extraction algorithms.
2. Computing pairwise cosine similarity between extracted vectors.
3. Measuring runtime and memory overhead for each method.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model_loader import load_model_and_tokenizer
from src.extractor import ActivationExtractor
from src.concept_extractors import ConceptVectorComparer, EXTRACTOR_REGISTRY


def main():
    print("=== LatentShift Example 5: Multi-Method Concept Extraction Benchmark ===")

    model_name = "gpt2"
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = 6

    pos_prompts = ["Scientific research drives discovery, innovation, and technological progress."]
    neg_prompts = ["Scientific research is worthless, meaningless, and completely useless."]

    extractor = ActivationExtractor(model, tokenizer, [layer], device=device)
    pos_acts, neg_acts = extractor.extract_contrastive(list(zip(pos_prompts, neg_prompts)))

    h_pos = pos_acts[layer]
    h_neg = neg_acts[layer]

    print(f"\nBenchmarking all {len(EXTRACTOR_REGISTRY)} extraction algorithms for Layer {layer}:")
    results = ConceptVectorComparer.benchmark_methods(h_pos, h_neg, normalize=True)

    print(f"\n{'Method':<22} | {'Vector Norm':<12} | {'Runtime (ms)':<12}")
    print("-" * 52)
    for res in results:
        print(f"{res.method_name:<22} | {res.vector_norm:<12.4f} | {res.runtime_ms:<12.2f}")


if __name__ == "__main__":
    main()
