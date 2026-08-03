# LatentShift: Methodology & Theoretical Framework

## 1. Executive Summary & Overview

**LatentShift** is a research-grade framework designed for zero-shot Large Language Model (LLM) alignment via real-time activation steering in transformer hidden representation spaces. Traditional alignment paradigms such as Reinforcement Learning from Human Feedback (RLHF), Direct Preference Optimization (DPO), and Supervised Fine-Tuning (SFT) modify model parameters permanently. This parameter modification often leads to the "alignment tax"—a degradation in general reasoning capability, loss of stylistic flexibility, and high computational costs.

LatentShift operates entirely in the **activation space**, intervening on intermediate residual stream vectors during autoregressive forward passes without altering model weights ($\Delta \theta = 0$).

---

## 2. Representation Engineering Foundation

Modern transformer architectures construct high-dimensional semantic vector spaces across their internal layer residual streams $h_l \in \mathbb{R}^d$. The **Linear Representation Hypothesis** posits that high-level concepts (e.g., toxicity, truthfulness, sentiment, formality) are encoded as linear directions (subspaces) within these hidden representations.

Given a positive concept prompt set $\mathcal{D}^+$ and a negative contrast set $\mathcal{D}^-$, the activation space is sampled at specified layer $l$:

$$H_l^+ = \{h_{l,i}^+ \}_{i=1}^{N^+}, \quad H_l^- = \{h_{l,j}^- \}_{j=1}^{N^-}$$

Where $h_{l,i} \in \mathbb{R}^d$ represents the hidden state vector at the final prompt token index.

---

## 3. Concept Vector Extraction Algorithms

LatentShift implements seven distinct algorithms for isolating the concept vector $v_l \in \mathbb{R}^d$:

1. **Mean Difference (Centroid Shift)**:
   Computes the vector difference between class centroids:
   $$v_l = \frac{1}{N^+} \sum_{i=1}^{N^+} h_{l,i}^+ - \frac{1}{N^-} \sum_{j=1}^{N^-} h_{l,j}^-$$

2. **Principal Component Analysis (PCA)**:
   Extracts the first principal component (maximal variance direction) of the contrastive difference matrix $D_l = H_l^+ - H_l^-$.

3. **Linear Discriminant Analysis (LDA)**:
   Finds the projection direction maximizing between-class variance relative to within-class variance:
   $$w \propto S_W^{-1} (\mu^+ - \mu^-)$$

4. **Logistic Regression Hyperplane Normal**:
   Trains a linear decision boundary with $L_2$ regularization and extracts the weight vector $w$.

5. **Linear Support Vector Machine (Linear SVM)**:
   Identifies the maximum-margin hyperplane separating $\mathcal{H}^+$ and $\mathcal{H}^-$.

6. **Sparse PCA**:
   Extracts sparse directional vectors enforcing an $L_1$ penalty to isolate key active dimensions.

7. **Truncated Singular Value Decomposition (SVD)**:
   Computes the primary right singular vector of the centered difference matrix.

---

## 4. Adaptive Multi-Layer Activation Steering

Rather than applying a scalar intervention across all layers uniformly, LatentShift supports layer-wise coefficient scaling:

$$h_l^{\text{steered}} = h_l + \alpha_l \cdot v_l$$

where $\alpha_l = \alpha \cdot w_l$, and $w_l$ is derived from one of three weighting strategies:
- **Uniform**: $w_l = 1.0$
- **Linear Decay**: $w_l = 1.0 - \frac{i}{K-1} (1 - \text{min\_ratio})$
- **Cosine Decay**: $w_l = \text{min\_ratio} + (1 - \text{min\_ratio}) \cdot \frac{1}{2} \left(1 + \cos\left(\frac{\pi i}{K-1}\right)\right)$

---

## 5. Automatic Layer Selection & Statistical Scoring

Target layers are automatically identified by scoring candidate transformer layers according to statistical separability metrics:
- **Mean Activation Separation**: $\|\mu^+ - \mu^-\|_2$
- **Cosine Separation**: $1 - \frac{\mu^+ \cdot \mu^-}{\|\mu^+\|_2 \|\mu^-\|_2}$
- **Fisher Score**: $\frac{(\mu^+ - \mu^-)^2}{\sigma_+^2 + \sigma_-^2}$
- **Signal-to-Noise Ratio (SNR)**: $\frac{\|\mu^+ - \mu^-\|_2}{\frac{1}{2}(\sigma_+ + \sigma_-)}$
- **Activation Variance**: $\text{Tr}(\Sigma^+ + \Sigma^-)$

---

## 6. Dynamic Closed-Loop Alpha Scheduling

During decoding step $t$, the dynamic steering coefficient $\alpha(t)$ evolves based on real-time feedback:
- **Linear Scheduler**: Intermediates linearly from $\alpha_{\text{start}}$ to $\alpha_{\text{end}}$.
- **Cosine Scheduler**: Smooth half-cosine annealing from $\alpha_{\text{max}}$ to $\alpha_{\text{min}}$.
- **Confidence-Based Scheduler**: Scales $\alpha(t)$ inversely with model confidence (higher entropy $\rightarrow$ stronger steering).
- **Entropy-Based Scheduler**: Scales $\alpha(t)$ proportionally to full-vocabulary entropy $H(P_t)$.
