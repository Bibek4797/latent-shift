# LatentShift: Zero-Shot LLM Alignment via Real-Time Activation Steering

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**LatentShift** is a production-grade, highly modular Python framework for performing **Representation Engineering (Activation Steering)** on open-source causal language models (e.g., Qwen-2.5, Llama-3, Mistral) without modifying model weights or performing expensive reinforcement learning (RLHF/DPO). 

By extracting latent concept vectors from intermediate hidden states and dynamically injecting them into the residual stream via PyTorch forward hooks, LatentShift achieves real-time, zero-shot behavior control (e.g., safety enforcement, toxicity suppression, formal style shifting).

---

## 📖 Theoretical Background

Traditional LLM alignment methods (SFT, RLHF, DPO) rely on gradient descent to modify millions or billions of parameters. This process is:
1. **Computationally intensive**: Requires massive hardware resources.
2. **Brittle**: Often results in "alignment tax," reducing the model's general reasoning capabilities.
3. **Static**: Once weights are updated, the alignment behavior is locked in.

**Representation Engineering (RepEng)** views alignment through the lens of internal activations rather than model weights. As information propagates through the transformer block, the model builds structured latent representations. 

### Latent Space Intervention

1. **Extraction**: We run the model on contrasting prompt pairs (e.g., $P_{\text{positive}}$ vs $P_{\text{negative}}$) representing a target concept.
2. **Concept Vector Computation**: We capture the final token's hidden states $h_{\text{pos}}$ and $h_{\text{neg}}$ at target layers. We isolate the concept direction using:
   - **Mean Difference**: $v_{\text{concept}} = \bar{h}_{\text{pos}} - \bar{h}_{\text{neg}}$
   - **Principal Component Analysis (PCA)**: Fitting the first principal component of the difference matrix $D = h_{\text{pos}} - h_{\text{neg}}$.
3. **Dynamic Steering**: During autoregressive generation, we inject the concept vector into the residual stream at target layer $L$:
   $$h_{\text{steered}}^{(L)} = h_{\text{original}}^{(L)} + \alpha \cdot v_{\text{concept}}^{(L)}$$
   where $\alpha$ is a scalar multiplier that controls steering intensity. Positive values reinforce the concept, and negative values suppress it.

---

## 📁 Repository Architecture

```
LatentShift/
├── config.py             # System-wide parameter configuration (@dataclass)
├── requirements.txt      # Python dependencies
├── README.md             # Theoretical overview and guides
├── src/
│   ├── __init__.py       # Package exposure
│   ├── model_loader.py   # Secure HF model loader with quantization support
│   ├── extractor.py      # Forward hook registration for activation extraction
│   ├── compute.py        # Vector calculation engines (Mean Diff, PCA)
│   ├── steer.py          # SteeredGenerator managing inference hooks
│   └── evaluator.py      # Language fluency (Perplexity) and cosine similarity metrics
└── app.py                # Streamlit Web Application Dashboard
```

---

## ⚡ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+ (CUDA or Apple Silicon MPS recommended for accelerated inference)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LatentShift.git
   cd LatentShift
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```
   *Note: If you do not have a dedicated GPU, you can select the built-in **Mock-Model-1.5B** in the sidebar to test the steering interface and see comparative outputs instantly.*

---

## 🧪 Verification Pipeline

We include an automated verification script to test system integration and hook safety:
```bash
python scratch/verify_steering.py
```
This ensures that:
- Hidden states are correctly captured at last token positions.
- Concept vector calculations (Mean Diff & PCA) match shape expectations.
- PyTorch hooks are dynamically registered and **cleanly removed** after generation, avoiding state contamination.

---

## 🎓 Placement Interview Talking Points

If discussing this project in research lab or machine learning engineering interviews, emphasize the following points:

### 1. Parametric vs. Activation-Space Alignment
- **Concept**: Contrast the computational cost of editing weights (RLHF) with editing activation trajectories. Explain that activation steering provides *zero-shot*, *dynamic*, and *reversible* control, which is extremely useful for multi-tenant applications where different users require different safety or stylistic profiles.

### 2. Multi-Layer Trajectory and Layer Selection
- **Insight**: Point out that semantic concepts are not represented uniformly across the network. Concepts form in the early-to-middle layers, stabilize in the middle layers, and dissolve in the final layers as they map to the vocabulary distribution. Explain how the `app.py` trajectory plot allows visualizing this phenomenon by showing concept vector norms and projection shifts.

### 3. PCA vs. Mean Difference
- **Technical Nuance**: Explain that while Mean Difference is intuitive, it can be sensitive to outliers and the choice of prompt pairs. PCA on contrastive differences acts as a denoiser, capturing the axis of maximum variance (direction of maximum style/concept difference) and ignoring noise directions.

### 4. Hook Safety & Memory Leak Prevention
- **Engineering Quality**: Highlight that registering PyTorch hooks can lead to silent memory leaks and graph retention if not managed properly. Discuss how `SteeredGenerator` wraps inference inside a `try...finally` block, ensuring that `hook.remove()` is called under all execution states (successful generation or runtime exception).
