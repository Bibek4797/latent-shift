# LatentShift Project Roadmap

This roadmap outlines past milestones, current release capabilities, and future research directions for **LatentShift**.

---

## 🟢 Completed Milestones

### Phase 1: Core Activation Engine (v1.0.0)
- [x] PyTorch forward hook registration and cleanup.
- [x] Mean Difference and PCA concept vector extractions.
- [x] Basic comparative generation interface.
- [x] CLI runner `run_experiment.py`.

### Phase 2: Multi-Layer & Interactive UI (v1.1.0)
- [x] Adaptive Multi-Layer Steering with Linear & Cosine decay strategies.
- [x] Streamlit web interface with interactive parameter controls and side-by-side comparative views.
- [x] Simulated model mode (`Mock-Model-1.5B`) for zero-GPU testing.

### Phase 3: Research Benchmark & Dynamic Steering Suite (v2.0.0)
- [x] **Dynamic Closed-Loop Steering**: 5 alpha schedulers (`Fixed`, `Linear`, `Cosine`, `Confidence`, `Entropy`).
- [x] **SQLite Experiment Tracker**: Persistent database logging (`src/experiment_tracker.py`) recording 27 parameters per run.
- [x] **7 Concept Extraction Algorithms**: `mean_diff`, `pca`, `lda`, `logistic_regression`, `linear_svm`, `sparse_pca`, `truncated_svd`.
- [x] **Automatic Layer Selection**: Statistical layer ranking using Fisher Score, SNR, Mean Separation, Cosine Separation, and Activation Variance.
- [x] **Multi-Metric Evaluator**: Perplexity ($PPL$), $D_{\text{KL}}$, $D_{\text{JS}}$, Token Entropy ($H$), Norm Difference, Steering Strength.
- [x] **Automated Grid Sweep Engine**: Cartesian benchmark sweeps with CSV, JSON, and Markdown report exports.
- [x] **Publication Documentation Suite**: Formal paper (`RESEARCH_REPORT.md`), LaTeX derivations, experimental protocols, diagrams, and developer guides.
- [x] **Performance Optimization**: Batched extractions, `torch.inference_mode()`, pre-converted hook tensors, KV-cached dynamic decoding.

---

## 🟡 Short-Term Roadmap (v2.1.0 - Q4 2026)

- [ ] **Sparse Autoencoders (SAEs)**: Integrate dictionary learning (SAEs) for automated, unsupervised monosemantic feature direction discovery without contrastive prompt pairs.
- [ ] **Cross-Layer Attention Steering**: Extend residual stream forward hooks to internal Query-Key-Value ($Q, K, V$) attention projection matrices.
- [ ] **Multi-Concept Superposition**: Simultaneous vector steering along orthogonal concept axes (e.g., $\alpha_1 v_{\text{safety}} + \alpha_2 v_{\text{conciseness}}$) with Gram-Schmidt orthogonalization.
- [ ] **vLLM & TensorRT-LLM Integration**: Native support for high-throughput serving engines with custom CUDA steering kernels.

---

## 🔵 Long-Term Research Vision (v3.0.0 - 2027)

- [ ] **Contextual Bandit Alpha Policies**: Reinforcement learning policies that dynamically optimize closed-loop $\alpha(t)$ trajectories based on real-time token entropy feedback.
- [ ] **Cross-Architecture Transferability**: Automated translation of extracted concept directions between different model families (e.g., transferring Llama-3 concept vectors to Qwen-2.5).
- [ ] **Interactive Latent Manipulation Studio**: Real-time 3D web visualizer for interactive vector space exploration and point-and-click latent space editing.
