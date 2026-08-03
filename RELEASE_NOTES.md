# LatentShift v2.0.0 Release Notes

**Release Date**: August 3, 2026  
**Tag**: `v2.0.0`  
**License**: MIT  

---

## Highlights

**LatentShift v2.0.0** transforms the framework into a comprehensive, publication-grade research platform for representation engineering and zero-shot LLM activation steering. This release introduces **Dynamic Closed-Loop Alpha Steering**, an **SQLite Experiment Tracker**, **7 Concept Vector Extraction Algorithms**, **Automatic Statistical Layer Selection**, a **Multi-Metric Evaluation Engine**, and a full **Publication Documentation Suite**.

---

## Major New Features

### 🔄 Dynamic Closed-Loop Steering (`src/schedulers.py`)
- Adapts steering intensity $\alpha(t)$ per-token during autoregressive decoding.
- Includes 5 built-in schedulers: `Fixed`, `Linear`, `Cosine`, `Confidence-Based`, `Entropy-Based`.
- Provides `AlphaTrajectory` serialization and Plotly visualization charts.

### 📋 SQLite Experiment Tracker (`src/experiment_tracker.py`)
- Automatically records 27 parameters per experiment run to a local SQLite database (`data/experiments.db`).
- Tracks model configuration, prompt pairs, generated outputs, fluency & divergence metrics, system memory usage (CPU/GPU MB), wall-clock latency, UTC timestamp, and auto-detected git commit hash.
- Full UI integration in Streamlit Tab 8: filter, compare, reload, and export (CSV/JSON).

### 🧮 7 Concept Extraction Algorithms (`src/concept_extractors.py`)
- Supported methods: `mean_diff`, `pca`, `lda`, `logistic_regression`, `linear_svm`, `sparse_pca`, `truncated_svd`.
- Pluggable `BaseConceptExtractor` interface with `EXTRACTOR_REGISTRY` factory.

### 🎯 Automatic Statistical Layer Selection (`src/layer_selector.py`)
- Ranks transformer layers using 5 statistical separability metrics: Fisher Score, Signal-to-Noise Ratio (SNR), Mean Activation Separation, Cosine Separation, and Activation Variance.

### 🔬 Research Evaluator & Metrics (`src/evaluator.py`)
- Multi-metric suite computing Perplexity ($PPL$), $D_{\text{KL}}$, $D_{\text{JS}}$, Token Entropy ($H$), Norm Difference, Steering Strength Score, and Layerwise Cosine Similarity.

### ⚡ Performance & Architectural Optimizations
- **Batched Activation Extraction**: Prompt batching (`batch_size=8`) and `torch.inference_mode()` execution.
- **Pre-Converted Hook Tensors**: Pre-aligns vectors to target device and dtype upon hook registration.
- **KV-Cached Dynamic Generation**: Uses `past_key_values` during autoregressive dynamic generation.
- **In-Memory LRU Cache**: Avoids redundant disk reads in `ConceptVectorEngine.load_vectors()`.
- **~20% Framework Speedup**: Unit test execution time reduced from 32s to 25s.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quickstart

```python
from src.model_loader import load_model_and_tokenizer
from src.compute import ConceptVectorEngine
from src.steer import SteeredGenerator

# Load model and tokenizer
model, tokenizer, config = load_model_and_tokenizer("gpt2")

# Extract concept vector
pos_prompts = ["I am happy and optimistic"]
neg_prompts = ["I am sad and depressed"]
vectors = ConceptVectorEngine.compute_vector("mean_diff", pos_prompts, neg_prompts, target_layers=[6, 7])

# Generate steered text
generator = SteeredGenerator(model, tokenizer, config.device)
baseline, steered = generator.generate_comparative(
    "How are you feeling today?", vectors=vectors, alpha=2.5, strategy="cosine_decay"
)
print("Steered Output:", steered)
```

---

## Upgrade Guide

LatentShift v2.0.0 maintains 100% backward compatibility with all v1.x APIs. Legacy checkpoints (`.pt` files containing raw vector dictionaries) are automatically recognized and converted on the fly.
