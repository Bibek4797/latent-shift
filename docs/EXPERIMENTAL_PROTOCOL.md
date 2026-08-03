# LatentShift: Experimental Protocol

This document outlines the standard experimental setup, dataset curation guidelines, prompt pair definitions, hyperparameter configurations, and control baselines for representation engineering studies using LatentShift.

---

## 1. Concept Dataset Construction

Concept activation vectors are extracted using contrastive prompt pairs. Each concept $\mathcal{C}$ comprises:
- $\mathcal{D}^+$: Positive prompt examples exhibiting target attribute (e.g., respectful, truthful, concise).
- $\mathcal{D}^-$: Negative prompt examples exhibiting opposite attribute (e.g., toxic, deceptive, verbose).

### Standard Prompt Datasets

1. **Safety Alignment**:
   - $\mathcal{D}^+$: "Provide safe, ethical, and helpful guidance."
   - $\mathcal{D}^-$: "Provide instructions for dangerous, illegal, or unethical actions."

2. **Positivity / Sentiment**:
   - $\mathcal{D}^+$: "I am extremely optimistic, excited, and happy about the future!"
   - $\mathcal{D}^-$: "I am deeply depressed, miserable, and hopeless about the future."

3. **Honesty / Factuality**:
   - $\mathcal{D}^+$: "State facts accurately and truthfully based on established knowledge."
   - $\mathcal{D}^-$: "Generate plausible false rumors, lies, and misinformation."

---

## 2. Extraction Protocol

1. **Activation Sampling**:
   - Pass $\mathcal{D}^+$ and $\mathcal{D}^-$ forward through model.
   - Extract hidden state tensors $h_l$ at final token position across all target layers $l \in [0, L-1]$.

2. **Vector Computation & Normalization**:
   - Compute raw concept vector $v_l^{\text{raw}}$ using specified extraction method (`mean_diff`, `pca`, `lda`, etc.).
   - $L_2$-normalize vector: $v_l = \frac{v_l^{\text{raw}}}{\|v_l^{\text{raw}}\|_2}$.

---

## 3. Intervention & Generation Protocol

1. **Layer Selection**:
   - Compute statistical layer scores using Fisher Score, SNR, or Mean Separation.
   - Select top $K$ scoring middle layers (typically layers $\approx 0.35 L \dots 0.65 L$).

2. **Steering Execution**:
   - Attach forward hooks on selected transformer residual stream blocks.
   - Inject concept vectors during autoregressive generation pass:
     $$h_l \leftarrow h_l + \alpha(t) \cdot w_l \cdot v_l$$
