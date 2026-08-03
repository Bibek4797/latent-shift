# Changelog

All notable changes to the **LatentShift** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-08-03

### Added
- **Dynamic Closed-Loop Steering**: Introduced per-token alpha adaptation during autoregressive generation using 5 schedulers (`Fixed`, `Linear`, `Cosine`, `Confidence`, `Entropy`).
- **SQLite Experiment Tracker**: Persistent database logging (`src/experiment_tracker.py`) recording 27 parameters per run (model, layers, alpha, strategy, scheduler, metrics, CPU/GPU memory, git commit).
- **Multi-Algorithm Concept Extraction**: Added support for 7 extraction algorithms (`mean_diff`, `pca`, `lda`, `logistic_regression`, `linear_svm`, `sparse_pca`, `truncated_svd`).
- **Automatic Statistical Layer Selection**: Statistical layer scoring module (`LayerSelector`) utilizing Fisher Score, Signal-to-Noise Ratio (SNR), Mean Activation Separation, Cosine Separation, and Activation Variance.
- **Multi-Dimensional Research Evaluator**: Research metric suite computing Perplexity ($PPL$), Kullback-Leibler Divergence ($D_{\text{KL}}$), Jensen-Shannon Divergence ($D_{\text{JS}}$), Token Entropy ($H$), Hidden State Norm Difference, and Steering Strength.
- **Automated Grid Sweep Engine**: `BenchmarkEngine` for running cartesian sweeps across models, methods, strategies, concepts, and alpha scales with automated CSV, JSON, and Markdown report generation.
- **Publication Documentation Suite**: Added `RESEARCH_REPORT.md`, `docs/METHODOLOGY.md`, `docs/MATHEMATICAL_DERIVATIONS.md`, `docs/EXPERIMENTAL_PROTOCOL.md`, `docs/BENCHMARK_PROTOCOL.md`, `docs/REPRODUCIBILITY_GUIDE.md`, `docs/LIMITATIONS_AND_FUTURE_WORK.md`, `docs/ARCHITECTURE_AND_DIAGRAMS.md`, `docs/API_DOCUMENTATION.md`, and `docs/DEVELOPER_GUIDE.md`.
- **Standalone Examples Directory**: Added 6 executable Python scripts under `examples/`.
- **Academic Metadata**: Added `CITATION.cff`, `CONTRIBUTING.md`, `ROADMAP.md`, and `RELEASE_NOTES.md`.

### Performance & Optimizations
- **Batched Activation Extractor**: Added prompt batching (`batch_size=8`) and `torch.inference_mode()` execution.
- **Pre-Converted Hook Vectors**: Pre-aligned concept vectors to target device and dtype upon hook registration, eliminating redundant inside-hook tensor conversions.
- **KV-Cached Dynamic Generation**: Optimized `generate_dynamic()` loop with `past_key_values` KV caching, converting $O(N^2)$ sequence passes to $O(N)$ single-token passes.
- **In-Memory Checkpoint Cache**: Added LRU dictionary cache to `ConceptVectorEngine.load_vectors()`.
- **Overall Speedup**: Reduced unit test execution time by ~20% (32s $\rightarrow$ 25s).

---

## [1.1.0] - 2026-08-01

### Added
- **Adaptive Multi-Layer Steering**: Per-layer weighting strategies (`uniform`, `linear_decay`, `cosine_decay`).
- **Streamlit Web Application**: Interactive multi-tab UI for live model steering, vector analytics, and benchmark visualization.

---

## [1.0.0] - 2026-07-21

### Added
- Initial release of LatentShift core engine.
- Support for Mean Difference and PCA concept vector extractions.
- PyTorch forward hook registration and cleanup.
- CLI experiment execution script (`run_experiment.py`).
