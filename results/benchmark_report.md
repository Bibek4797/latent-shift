# 🔬 LatentShift Activation Steering Research Benchmark Report

**Generated At:** `2026-08-03 17:24:33 UTC`
**Total Experiment Trials Executed:** `8`

## 🏆 Top Performing Steering Trial
- **Run ID:** `run_001`
- **Model:** `gpt2`
- **Concept:** `positivity`
- **Extraction Method:** `mean_diff`
- **Steering Strategy:** `uniform`
- **Alpha (α):** `1.0`
- **Steering Strength Score:** `0.0000`
- **KL Divergence (D_KL):** `5.8169`
- **Perplexity (Baseline → Steered):** `11.80 → 16.62`

## 📊 Complete Benchmark Summary Leaderboard

| Rank | Run ID | Concept | Method | Strategy | Alpha | PPL Ratio | KL Div | Cos Sim | Strength Score | Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | `run_001` | `positivity` | `mean_diff` | `uniform` | `1.0` | `1.409` | `5.8169` | `0.8796` | `0.0000` | `15428.1` |
| #2 | `run_002` | `positivity` | `mean_diff` | `uniform` | `2.0` | `1.556` | `5.4919` | `0.8382` | `0.0000` | `22016.8` |
| #3 | `run_003` | `positivity` | `mean_diff` | `cosine_decay` | `1.0` | `0.802` | `5.0227` | `0.9184` | `0.0000` | `17937.3` |
| #4 | `run_004` | `positivity` | `mean_diff` | `cosine_decay` | `2.0` | `1.190` | `6.7951` | `0.9075` | `0.0000` | `13796.9` |
| #5 | `run_005` | `positivity` | `lda` | `uniform` | `1.0` | `2.698` | `4.2284` | `0.8168` | `0.0000` | `20673.2` |
| #6 | `run_006` | `positivity` | `lda` | `uniform` | `2.0` | `9.475` | `3.3997` | `0.7518` | `0.0000` | `22151.7` |
| #7 | `run_007` | `positivity` | `lda` | `cosine_decay` | `1.0` | `9.869` | `4.0103` | `0.8711` | `0.0000` | `22010.1` |
| #8 | `run_008` | `positivity` | `lda` | `cosine_decay` | `2.0` | `15.619` | `3.5429` | `0.8105` | `0.0000` | `22210.7` |