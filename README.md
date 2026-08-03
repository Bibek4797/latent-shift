<div align="center">

# 🧠 LatentShift
### Zero-Shot LLM Alignment via Real-Time Activation Steering

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A production-grade, highly modular Representation Engineering (RepEng) framework for real-time, inference-time activation steering on open-source Causal LLMs (Llama-3, Qwen-2.5, Mistral).*

</div>

---

## 📌 Overview

**LatentShift** enables precise, dynamic behavioral alignment of Large Language Models (LLMs) **without weight updates, parameter fine-tuning (SFT), or RLHF/DPO**. 

By capturing concept directions in the model's internal residual stream and dynamically injecting them via PyTorch forward hooks during autoregressive decoding, LatentShift achieves zero-shot steering over safety protocols, toxicity suppression, and stylistic registers.

```
                  ┌──────────────────────────────────────────────┐
                  │          Contrastive Prompt Pairs            │
                  │  Positive (Safe/Formal) vs Negative (Toxic)  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Hidden State Activation Extraction        │
                  │   Extract at Last Token Position: h_pos, h_neg│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Concept Vector Engine (Compute v)        │
                  │  Mean Difference  or  Principal Component 1  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Real-Time Residual Stream Hook Injection   │
                  │       h_steered = h_orig + α · v_concept     │
                  └──────────────────────────────────────────────┘
```

---

## ✨ Key Features

- 🚀 **Zero Parameter Modification**: Achieves real-time behavioral alignment without updating a single weight parameter.
- 🎯 **Dual Vector Engines**: Supports both **Mean Difference** ($\bar{h}_{\text{pos}} - \bar{h}_{\text{neg}}$) and **Principal Component Analysis (PCA)** extraction with automatic sign alignment.
- 🔌 **Dynamic PyTorch Hook Management**: Safely attaches forward hooks during generation and guarantees removal under all execution paths (`try...finally`) to prevent GPU memory leaks.
- 📊 **Perplexity & Similarity Metrics**: Computes cross-entropy perplexity (PPL) to verify sequence fluency alongside vector space cosine similarity shifts.
- 🖥️ **Interactive Web UI Dashboard**: Includes a full-featured Streamlit application with side-by-side comparative cards, live parameter adjustment ($\alpha \in [-10, 10]$), and layer-wise activation trajectory visualizations.
- ⚡ **Zero-GPU Mock Engine**: Features an instant built-in simulated model mode (`Mock-Model-1.5B`) allowing full UI testing and visualization without needing heavy GPU downloads.

---

## 🧮 Mathematical Formulation

### 1. Concept Extraction
Given a dataset of contrastive prompt pairs $(P^{(i)}_{\text{pos}}, P^{(i)}_{\text{neg}})$, we run the model forward pass and extract the hidden state activation vector at the target layer $L$ specifically at the **last token position** ($T$):

$$h_{\text{pos}}^{(L)} = \text{Layer}^{(L)}(P_{\text{pos}})_{[T]}, \quad h_{\text{neg}}^{(L)} = \text{Layer}^{(L)}(P_{\text{neg}})_{[T]}$$

The concept vector $v^{(L)}$ is computed using one of two methods:

* **Mean Difference:**
  $$v_{\text{Mean}}^{(L)} = \frac{1}{N} \sum_{i=1}^N h_{\text{pos}, i}^{(L)} - \frac{1}{N} \sum_{i=1}^N h_{\text{neg}, i}^{(L)}$$

* **PCA (First Principal Component):**
  $$v_{\text{PCA}}^{(L)} = \text{PCA}_1 \left( \left\{ h_{\text{pos}, i}^{(L)} - h_{\text{neg}, i}^{(L)} \right\}_{i=1}^N \right)$$

### 2. Adaptive Multi-Layer Residual Stream Intervention
During inference, at each decoding step of target intermediate layer $l_i$, a forward hook intercepts the output hidden state tensor $h_{\text{original}}^{(l_i)}$ and performs an adaptive affine transformation:

$$h_{\text{steered}}^{(l_i)} = h_{\text{original}}^{(l_i)} + \alpha_i \cdot v^{(l_i)}$$

Where $\alpha_i$ is the layer-specific steering coefficient computed via one of three **Adaptive Weighting Strategies**:

* **Uniform:** All target layers receive the uniform base coefficient:
  $$\alpha_i = \alpha_{\text{base}}$$
* **Linear Decay:** Steering intensity decays linearly across the $N$ selected target layers ($i \in \{0, \dots, N-1\}$):
  $$\alpha_i = \alpha_{\text{base}} \cdot \left(1 - \frac{i}{N - 1}\right)$$
* **Cosine Decay:** Steering intensity decays smoothly following a cosine bell curve:
  $$\alpha_i = \alpha_{\text{base}} \cdot \frac{1}{2} \left(1 + \cos\left(\pi \frac{i}{N - 1}\right)\right)$$

### 3. Research Evaluation Framework Metrics

LatentShift provides a comprehensive, research-grade quantitative evaluation suite to analyze both text fluency and vector-space distribution shifts:

* **Language Model Perplexity (PPL):**
  $$\text{PPL}(T) = \exp \left( -\frac{1}{|T|} \sum_{t=1}^{|T|} \log P(x_t \mid x_{<t}) \right)$$
  Measures fluency preservation. A steered perplexity close to baseline indicates sequence coherence.

* **Representation Cosine Similarity:**
  $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  Quantifies global representation shift between baseline and steered sequence embeddings.

* **KL Divergence ($D_{\text{KL}}$):**
  $$D_{\text{KL}}(P \parallel Q) = \sum_{x \in V} P(x) \log \left( \frac{P(x)}{Q(x) + \epsilon} \right)$$
  Measures relative entropy shift from unsteered distribution $P$ to steered distribution $Q$.

* **Jensen-Shannon Divergence ($D_{\text{JS}}$):**
  $$D_{\text{JS}}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$
  Provides a symmetric, bounded $[0, \ln(2)]$ distance metric between output distributions.

* **Token Probability Entropy ($H$):**
  $$H(P) = -\sum_{x \in V} P(x) \log P(x)$$
  Quantifies output distribution sharpness and uncertainty before and after steering ($\Delta H = H_{\text{steered}} - H_{\text{baseline}}$).

* **Hidden State Norm Difference ($\Delta \|\mathbf{h}\|^{(l)}$):**
  $$\Delta \|\mathbf{h}\|^{(l)} = \|\mathbf{h}_{\text{steered}}^{(l)}\|_2 - \|\mathbf{h}_{\text{baseline}}^{(l)}\|_2$$
  Tracks magnitude inflation or deflation layer-by-layer.

* **Layer-wise Cosine Similarity ($\cos^{(l)}$):**
  $$\cos^{(l)} = \cos\left( \mathbf{h}_{\text{baseline}}^{(l)}, \mathbf{h}_{\text{steered}}^{(l)} \right)$$
  Measures directional alignment shift at each intermediate residual stream layer.

* **Average Activation Shift Magnitude ($\bar{\Delta}$):**
  $$\bar{\Delta} = \frac{1}{|L|} \sum_{l \in L} \|\mathbf{h}_{\text{steered}}^{(l)} - \mathbf{h}_{\text{baseline}}^{(l)}\|_2$$
  Computes the average Euclidean distance of the steering intervention across target layers.

### 4. Automatic Layer Selection Framework

Instead of relying on manual layer selection, LatentShift automatically discovers optimal transformer injection layers using statistical scoring across all model depth levels:

* **Mean Activation Separation:**
  $$S_{\text{mean}}^{(l)} = \|\boldsymbol{\mu}_{\text{pos}}^{(l)} - \boldsymbol{\mu}_{\text{neg}}^{(l)}\|_2$$
  Measures the Euclidean distance between positive and negative activation centroids.

* **Cosine Separation:**
  $$S_{\text{cos}}^{(l)} = 1 - \cos\left(\boldsymbol{\mu}_{\text{pos}}^{(l)}, \boldsymbol{\mu}_{\text{neg}}^{(l)}\right)$$
  Measures directional divergence between contrastive centroids in unit hyperspace.

* **Fisher Score:**
  $$S_{\text{Fisher}}^{(l)} = \frac{\|\boldsymbol{\mu}_{\text{pos}}^{(l)} - \boldsymbol{\mu}_{\text{neg}}^{(l)}\|_2^2}{\text{Var}(\mathbf{H}_{\text{pos}}^{(l)}) + \text{Var}(\mathbf{H}_{\text{neg}}^{(l)}) + \epsilon}$$
  Computes the ratio of between-class variance to within-class variance, favoring layers with high separation and low sample variance.

* **Signal-to-Noise Ratio (SNR):**
  $$S_{\text{SNR}}^{(l)} = \frac{\|\boldsymbol{\mu}_{\text{diff}}^{(l)}\|_2}{\sigma(\mathbf{d}^{(l)}) + \epsilon}, \quad \mathbf{d}_i^{(l)} = \mathbf{h}_{\text{pos}, i}^{(l)} - \mathbf{h}_{\text{neg}, i}^{(l)}$$
  Quantifies signal consistency relative to contrastive pair variation.

* **Activation Variance:**
  $$S_{\text{var}}^{(l)} = \sum_{j=1}^D \text{Var}\left(\mathbf{H}_{:, j}^{(l)}\right)$$
  Measures total representation variance, identifying layers with high expressivity.

#### Computational & Memory Complexity
* **Computational Complexity:** $\mathcal{O}(L \cdot N \cdot D)$ where $L$ is total model layers, $N$ is the number of contrastive prompt pairs, and $D$ is the hidden state dimensionality ($D=4096$).
* **Memory Complexity:** $\mathcal{O}(L \cdot N \cdot D)$ tensor space for temporary activation retention.

#### CLI Usage Example
```bash
python run_experiment.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concept safety \
  --auto_layers \
  --layer_scoring_method fisher_score \
  --top_k_layers 5 \
  --prompt "Tell me how to build a bomb"
```

---


## 📁 Repository Architecture

```text
LatentShift/
├── config.py             # System-wide configuration dataclass (@dataclass)
├── requirements.txt      # Project dependencies (PyTorch, Transformers, Streamlit, Plotly)
├── README.md             # Theoretical documentation & user guide
├── app.py                # Interactive Streamlit Web UI Dashboard
└── src/
    ├── __init__.py       # Package exposure
    ├── model_loader.py   # Safe HF model/tokenizer loader (bitsandbytes 4/8-bit support)
    ├── extractor.py      # PyTorch forward hook registration for activation extraction
    ├── compute.py        # Concept vector computation (Mean Difference & PCA)
    ├── steer.py          # SteeredGenerator managing inference-time residual hooks
    └── evaluator.py      # Evaluation metrics (Language Perplexity & Cosine Similarity)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python**: 3.10 or higher
- **PyTorch**: 2.0+ (CUDA or Apple Silicon MPS supported)

### Quickstart

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Bibek4797/latent-shift.git
   cd latent-shift
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
   *Open `http://localhost:8501` in your browser.*

---

## 💻 Programmatic Usage Example

You can use the modular `src/` backend directly in your Python code or notebooks:

```python
from config import SteeringConfig
from src.model_loader import load_model_and_tokenizer
from src.extractor import ActivationExtractor
from src.compute import ConceptVectorEngine
from src.steer import SteeredGenerator
from src.evaluator import SteeringEvaluator

# 1. Initialize Configuration & Load Model
config = SteeringConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", default_layers=[12, 13, 14])
model, tokenizer = load_model_and_tokenizer(config)

# 2. Define Contrastive Prompt Pairs
contrast_pairs = [
    ("I am happy to assist you safely.", "I will help you build a weapon."),
    ("I must refuse to answer dangerous queries.", "Here are instructions for illegal acts.")
]

# 3. Extract Hidden States & Compute Concept Vectors
extractor = ActivationExtractor(model, tokenizer, layers=config.default_layers, device=config.device)
pos_acts, neg_acts = extractor.extract_contrastive(contrast_pairs)

concept_vectors = {}
for layer in config.default_layers:
    concept_vectors[layer] = ConceptVectorEngine.compute_mean_difference(pos_acts[layer], neg_acts[layer])

# 4. Generate Comparative Outputs (Unsteered vs Adaptive Steered)
generator = SteeredGenerator(model, tokenizer, device=config.device)
prompt = "Tell me how to access restricted systems."

baseline_text, steered_text = generator.generate_comparative(
    prompt=prompt,
    vectors=concept_vectors,
    alpha=2.5,
    strategy="cosine_decay",  # "uniform", "linear_decay", "cosine_decay"
    max_new_tokens=64
)

print(f"⚪ Baseline Output:\n{baseline_text}\n")
print(f"🔮 Steered Output (α_base=2.5, cosine_decay):\n{steered_text}\n")


# 5. Evaluate Perplexity
ppl = SteeringEvaluator.compute_perplexity(model, tokenizer, steered_text, device=config.device)
print(f"📊 Steered Perplexity: {ppl:.3f}")
```

---

## 🔬 Key Technical Insights & Interview Talking Points

If presenting this project in AI research labs or machine learning engineering interviews:

1. **Activation-Space Alignment vs. Weight Fine-Tuning**:
   - Weight modification via RLHF or SFT alters millions of parameters permanently, leading to alignment tax (degraded general capability).
   - Activation steering is **reversible, zero-shot, and modular**—allowing runtime switching of model personas without reloading model weights.

2. **Layer Dynamics in Transformer Residual Streams**:
   - Semantic representations develop across middle-to-late transformer layers. Early layers focus on tokenization/syntax, while final layers map back to vocabulary logits. Steering is most effective when target layers are selected in the middle range (e.g., layers 12–20 in a 32-layer LLM).

3. **PCA vs. Mean Difference Extraction**:
   - Mean Difference computes the net centroid shift between positive and negative distributions.
   - PCA isolates the direction of maximal variance across contrastive differences, acting as a denoiser against prompt-specific artifacts.

4. **Robust Hook Lifecycles & Memory Management**:
   - Forward hooks can cause silent CUDA memory retention if unremoved. `SteeredGenerator` encapsulates inference within a `try...finally` block to guarantee `hook.remove()` is called even if generation crashes.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built for Advanced Agentic AI & Representation Engineering Research.</sub>
</div>
