# LatentShift: Benchmark Protocol

This document specifies the standard benchmarking procedure for evaluating multi-dimensional activation steering performance across extraction methods, steering weight strategies, and alpha schedules.

---

## 1. Multi-Dimensional Grid Sweep Architecture

The `BenchmarkEngine` automates execution over a full cartesian product grid:

$$\mathcal{G} = \mathcal{M} \times \mathcal{E} \times \mathcal{S} \times \mathcal{A} \times \mathcal{C}$$

Where:
- $\mathcal{M}$: Models (e.g., `gpt2`, `Qwen/Qwen2.5-7B-Instruct`).
- $\mathcal{E}$: Extraction methods (`mean_diff`, `pca`, `lda`, `logistic_regression`, `linear_svm`, `sparse_pca`, `truncated_svd`).
- $\mathcal{S}$: Steering weight strategies (`uniform`, `linear_decay`, `cosine_decay`).
- $\mathcal{A}$: Alpha scales (e.g., $1.0, 2.0, 3.0$).
- $\mathcal{C}$: Concept domains (`positivity`, `safety`, `honesty`).

---

## 2. Evaluation Metric Suite

Every benchmark run computes and logs the full evaluation metric suite:

| Metric Category | Abbreviation | Mathematical Target | Description |
|-----------------|--------------|----------------------|-------------|
| Fluency | **PPL** | $\exp(-\frac{1}{N} \sum \log P)$ | Language model perplexity on generated text |
| Distribution Distance | **$D_{\text{KL}}$** | $\sum P \log(P/Q)$ | Kullback-Leibler Divergence between output distributions |
| Bounded Distance | **$D_{\text{JS}}$** | $\frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$ | Jensen-Shannon Divergence ($0 \le D_{\text{JS}} \le \log 2$) |
| Alignment | **Cos Sim** | $\frac{E_{\text{base}} \cdot E_{\text{steered}}}{\|E_{\text{base}}\| \|E_{\text{steered}}\|}$ | Embedding cosine similarity |
| Uncertainty | **$H(P)$** | $-\sum P \log P$ | Token prediction entropy before and after steering |
| Magnitude Shift | **$\Delta \|h\|$** | $\|h_{\text{steered}} - h_{\text{baseline}}\|_2$ | Average layer-wise hidden state norm difference |
| System Efficiency | **Runtime / Memory** | ms, CPU/GPU MB | Wall-clock execution latency and peak memory usage |

---

## 3. Automated Report Generation

The benchmark suite generates three artifacts in `results/`:
1. `benchmark_summary.json`: Machine-readable execution log containing all metrics for every grid point.
2. `benchmark_summary.csv`: Tabular spreadsheet format for quantitative filtering and analysis.
3. `benchmark_report.md`: Markdown report formatted with summary leaderboards and metric tables.
