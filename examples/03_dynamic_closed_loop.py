"""
examples/03_dynamic_closed_loop.py
----------------------------------
Dynamic Closed-Loop Steering example using per-token alpha schedulers.

Demonstrates:
1. Linear, Cosine, Confidence-based, and Entropy-based schedulers.
2. Generating text with dynamic alpha adaptation step-by-step.
3. Inspecting the AlphaTrajectory history.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model_loader import load_model_and_tokenizer
from src.extractor import ActivationExtractor
from src.compute import ConceptVectorEngine
from src.steer import SteeredGenerator
from src.schedulers import (
    CosineScheduler,
    EntropyBasedScheduler,
    LinearScheduler,
)


def main():
    print("=== LatentShift Example 3: Dynamic Closed-Loop Steering ===")

    model_name = "gpt2"
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_layers = [6, 7, 8]

    # Extract concept vectors
    pos_prompts = ["Always answer with absolute honesty, clarity, and truth."]
    neg_prompts = ["Answer with deceptive claims, false rumors, and lies."]

    extractor = ActivationExtractor(model, tokenizer, target_layers, device=device)
    pos_acts, neg_acts = extractor.extract_contrastive(list(zip(pos_prompts, neg_prompts)))
    vectors = {l: ConceptVectorEngine.compute_vector("pca", pos_acts[l], neg_acts[l], normalize=True) for l in target_layers}

    generator = SteeredGenerator(model, tokenizer, device=device)
    prompt = "The scientific consensus on global climate change is"

    # 1. Cosine Scheduler (Smooth Annealing)
    cos_sched = CosineScheduler(alpha_max=3.5, alpha_min=0.5)
    text_cos, traj_cos = generator.generate_dynamic(
        prompt=prompt, vectors=vectors, scheduler=cos_sched, max_new_tokens=30, do_sample=False
    )

    # 2. Entropy Scheduler (Uncertainty Feedback)
    ent_sched = EntropyBasedScheduler(alpha_min=0.5, alpha_max=3.5)
    text_ent, traj_ent = generator.generate_dynamic(
        prompt=prompt, vectors=vectors, scheduler=ent_sched, max_new_tokens=30, do_sample=False
    )

    print("\n" + "=" * 60)
    print(f"Prompt: '{prompt}'\n")
    print(f"[STEERED - Cosine Scheduler Output]:\n{text_cos}")
    print(f"  Alpha Trajectory Summary: {traj_cos.to_dict()}\n")
    print(f"[STEERED - Entropy Scheduler Output]:\n{text_ent}")
    print(f"  Alpha Trajectory Summary: {traj_ent.to_dict()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
