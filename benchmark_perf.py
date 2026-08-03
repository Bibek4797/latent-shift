"""
benchmark_perf.py
------------------
Performance benchmark measuring speed and memory for LatentShift components.
"""

import time
import torch
import torch.nn as nn
from src.extractor import ActivationExtractor
from src.steer import SteeredGenerator
from src.compute import ConceptVectorEngine
from src.evaluator import SteeringEvaluator
from src.schedulers import LinearScheduler


class MockModel(nn.Module):
    """Simple mock model for performance timing."""
    def __init__(self, hidden_dim=128, num_layers=6, vocab_size=1000):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids=None, past_key_values=None, use_cache=False, **kwargs):
        if input_ids is not None:
            x = self.model.embed_tokens(input_ids)
        else:
            x = torch.zeros((1, 1, self.config.hidden_size), device=next(self.parameters()).device)

        new_past = [] if use_cache else None
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            if use_cache:
                new_past.append((x, x))

        logits = self.lm_head(x)
        if use_cache:
            return type("Output", (), {"logits": logits, "past_key_values": tuple(new_past)})()
        return type("Output", (), {"logits": logits, "loss": torch.tensor(0.5)})()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def generate(self, input_ids=None, max_new_tokens=30, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 5
        # Return input_ids + new generated token ids
        new_tokens = torch.randint(1, 100, (batch_size, seq_len + max_new_tokens))
        return new_tokens


class MockTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __call__(self, text_or_list, return_tensors="pt", padding=True, **kwargs):
        if isinstance(text_or_list, str):
            ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
        else:
            ids = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 3, 4, 5]], dtype=torch.long)
            mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, token_ids, skip_special_tokens=True):
        return "benchmarked generated text response"


def run_benchmark():
    model = MockModel(hidden_dim=256, num_layers=8, vocab_size=5000)
    tokenizer = MockTokenizer()
    device = "cpu"

    prompts = [f"Prompt sample {i}" for i in range(20)]
    target_layers = [2, 3, 4, 5]

    # 1. Extraction timing
    extractor = ActivationExtractor(model, tokenizer, target_layers, device=device)
    t0 = time.perf_counter()
    acts = extractor.extract_activations(prompts)
    t_extract = time.perf_counter() - t0

    # 2. Vector computation timing
    pos_acts = acts[2]
    neg_acts = acts[2]
    t0 = time.perf_counter()
    v_mean = ConceptVectorEngine.compute_mean_difference(pos_acts, neg_acts)
    v_pca = ConceptVectorEngine.compute_pca_vector(pos_acts, neg_acts)
    t_vector = time.perf_counter() - t0

    # 3. Steering generation timing
    vectors = {layer: torch.randn(256) for layer in target_layers}
    generator = SteeredGenerator(model, tokenizer, device=device)

    t0 = time.perf_counter()
    for _ in range(5):
        base, steered = generator.generate_comparative(
            "Test prompt for steering", vectors=vectors, alpha=2.0, max_new_tokens=30, do_sample=False
        )
    t_static_gen = time.perf_counter() - t0

    # 4. Dynamic steering generation timing
    scheduler = LinearScheduler(alpha_start=3.0, alpha_end=0.5)
    t0 = time.perf_counter()
    for _ in range(5):
        dyn_text, traj = generator.generate_dynamic(
            "Test prompt for dynamic steering", vectors=vectors, scheduler=scheduler, max_new_tokens=30, do_sample=False
        )
    t_dyn_gen = time.perf_counter() - t0

    # 5. Evaluation timing
    t0 = time.perf_counter()
    report = SteeringEvaluator.evaluate_full(
        model, tokenizer, "Test prompt", base, steered, concept_vector=vectors[2], device=device
    )
    t_eval = time.perf_counter() - t0

    print("=== LATENTSHIFT PERFORMANCE BENCHMARK ===")
    print(f"Extraction (20 prompts)    : {t_extract*1000:.2f} ms")
    print(f"Vector Compute (Mean + PCA): {t_vector*1000:.2f} ms")
    print(f"Static Generation (5 runs) : {t_static_gen*1000:.2f} ms")
    print(f"Dynamic Generation (5 runs): {t_dyn_gen*1000:.2f} ms")
    print(f"Full Evaluation Report     : {t_eval*1000:.2f} ms")
    print("==========================================")
    return {
        "extract_ms": t_extract * 1000,
        "vector_ms": t_vector * 1000,
        "static_gen_ms": t_static_gen * 1000,
        "dyn_gen_ms": t_dyn_gen * 1000,
        "eval_ms": t_eval * 1000,
    }

if __name__ == "__main__":
    run_benchmark()
