# LatentShift: Limitations & Future Work

While LatentShift provides a flexible, modular, and mathematically rigorous framework for zero-shot activation steering, several theoretical and empirical limitations exist.

---

## 1. Current Limitations

1. **Activation Drift in Long Contexts**:
   As generation context length increases beyond $100+$ tokens, fixed or simple scheduled interventions can accumulate subtle semantic drift or hallucination risks if $\alpha$ is tuned too aggressively.

2. **Concept Entanglement**:
   Linear concept vectors extracted via contrastive pairs can occasionally isolate secondary stylistic features (e.g., formal tone, verbosity) alongside the primary target concept (e.g., positivity or safety).

3. **Layer Boundary Sensitivity**:
   Injecting steering vectors into very early transformer layers (e.g., layers 0–3) disrupts lower-level token embedding representations, causing sudden perplexity spikes. Injecting into final pre-head layers can distort vocabulary logit projections.

4. **Hardware Memory Overhead for Large Sweeps**:
   Automated grid search benchmarks involving multi-gigabyte open models (e.g., 7B–70B parameters) require significant VRAM when evaluating pairwise divergence metrics across long prompt batches.

---

## 2. Future Work & Roadmap

1. **Cross-Layer Attention Steering**:
   Extending residual stream activation hooks to internal Query-Key-Value ($Q, K, V$) attention projection matrices to control multi-head attention weights directly.

2. **Unsupervised Concept Discovery**:
   Integrating dictionary learning (Sparse Autoencoders / SAEs) to discover latent monosemantic feature directions automatically without requiring paired positive/negative prompt contrast sets.

3. **Multi-Concept Superposition**:
   Implementing simultaneous vector steering along orthogonal concept axes (e.g., $\alpha_1 v_{\text{safety}} + \alpha_2 v_{\text{conciseness}}$) with automatic Gram-Schmidt orthogonalization.

4. **Reinforcement Learning Scheduler Tuning**:
   Applying lightweight contextual bandits or Q-learning to learn optimal closed-loop $\alpha(t)$ trajectory policies dynamically based on real-time logit entropy feedback.
