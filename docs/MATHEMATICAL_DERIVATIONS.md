# LatentShift: Mathematical Derivations

This document presents full mathematical derivations for all concept extraction algorithms, layer scoring methods, dynamic schedulers, and evaluation metrics implemented in LatentShift.

---

## 1. Concept Extraction Formulations

Let $H^+ \in \mathbb{R}^{N^+ \times d}$ and $H^- \in \mathbb{R}^{N^- \times d}$ denote activation matrices collected from positive and negative prompt sets for hidden dimension $d$.

### 1.1 Mean Difference (Centroid Shift)
The mean activation vector for each set is:
$$\mu^+ = \frac{1}{N^+} \sum_{i=1}^{N^+} h_i^+, \quad \mu^- = \frac{1}{N^-} \sum_{j=1}^{N^-} h_j^-$$

The concept vector $v$ is the normalized difference:
$$v_{\text{raw}} = \mu^+ - \mu^-, \quad v = \frac{v_{\text{raw}}}{\|v_{\text{raw}}\|_2}$$

### 1.2 Principal Component Analysis (PCA)
Assuming paired differences $D_k = h_k^+ - h_k^-$ for $k=1, \dots, N$:
$$\bar{D} = \frac{1}{N} \sum_{k=1}^N D_k$$
$$\tilde{D}_k = D_k - \bar{D}$$
The covariance matrix $C \in \mathbb{R}^{d \times d}$ is:
$$C = \frac{1}{N-1} \tilde{D}^T \tilde{D}$$

The concept vector $v$ is the eigenvector corresponding to the maximum eigenvalue $\lambda_{\max}$:
$$C v = \lambda_{\max} v, \quad \text{subject to } \|v\|_2 = 1$$

### 1.3 Linear Discriminant Analysis (LDA)
LDA finds $v$ maximizing Rayleigh's quotient:
$$J(v) = \frac{v^T S_B v}{v^T S_W v}$$

where between-class scatter $S_B$ and within-class scatter $S_W$ are defined as:
$$S_B = (\mu^+ - \mu^-)(\mu^+ - \mu^-)^T$$
$$S_W = \sum_{i=1}^{N^+} (h_i^+ - \mu^+)(h_i^+ - \mu^+)^T + \sum_{j=1}^{N^-} (h_j^- - \mu^-)(h_j^- - \mu^-)^T$$

Solving the generalized eigenvalue problem $S_W^{-1} S_B v = \lambda v$ yields:
$$v \propto S_W^{-1} (\mu^+ - \mu^-)$$

### 1.4 Support Vector Machine (Linear SVM)
Linear SVM solves the primal convex optimization problem:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|_2^2 + C \sum_{k=1}^{N} \xi_k$$
$$\text{s.t. } y_k (w^T h_k + b) \ge 1 - \xi_k, \quad \xi_k \ge 0, \quad y_k \in \{-1, +1\}$$

The steering vector $v$ is the normalized weight vector $w / \|w\|_2$.

### 1.5 Sparse PCA
Sparse PCA enforces an $L_1$ penalty on the principal component loading:
$$\min_{v} -v^T C v + \lambda \|v\|_1 \quad \text{s.t. } \|v\|_2 = 1$$

---

## 2. Layer Scoring Formulations

Given positive sample activations $H_l^+$ and negative activations $H_l^-$ at layer $l$:

### 2.1 Fisher Score
$$\text{Fisher}(l) = \sum_{m=1}^d \frac{(\mu_{l,m}^+ - \mu_{l,m}^-)^2}{(\sigma_{l,m}^+)^2 + (\sigma_{l,m}^-)^2}$$

### 2.2 Signal-to-Noise Ratio (SNR)
$$\text{SNR}(l) = \frac{\|\mu_l^+ - \mu_l^-\|_2}{\frac{1}{2} \left( \sqrt{\frac{1}{d} \sum_{m=1}^d (\sigma_{l,m}^+)^2} + \sqrt{\frac{1}{d} \sum_{m=1}^d (\sigma_{l,m}^-)^2} \right)}$$

---

## 3. Divergence & Perplexity Metrics

### 3.1 Kullback-Leibler (KL) Divergence
Given baseline probability distribution $P$ and steered distribution $Q$ over vocabulary $V$:
$$D_{\text{KL}}(P \parallel Q) = \sum_{x \in V} P(x) \log \left(\frac{P(x)}{Q(x)}\right)$$

### 3.2 Jensen-Shannon (JS) Divergence
Let $M = \frac{1}{2}(P + Q)$. Then:
$$D_{\text{JS}}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$$

$$0 \le D_{\text{JS}}(P \parallel Q) \le \log 2$$

### 3.3 Perplexity (PPL)
Given token sequence $X = (x_1, x_2, \dots, x_N)$:
$$\text{PPL}(X) = \exp \left( -\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_1, \dots, x_{i-1}) \right)$$
