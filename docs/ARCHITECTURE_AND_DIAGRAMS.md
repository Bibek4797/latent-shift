# LatentShift: System Architecture & Sequence Diagrams

This document contains comprehensive architecture, block, and sequence diagrams detailing the internal workflows of LatentShift.

---

## 1. System Architecture Diagram

```mermaid
graph TD
    A[Contrastive Prompt Pair D+ / D-] --> B[Activation Extractor]
    B --> C[Hidden State Sampling]
    C --> D[Concept Vector Engine]
    
    subgraph Concept Extraction Algorithms
        D --> D1[Mean Difference]
        D --> D2[PCA / Sparse PCA]
        D --> D3[LDA]
        D --> D4[Logistic Regression / Linear SVM]
        D --> D5[Truncated SVD]
    end
    
    D1 & D2 & D3 & D4 & D5 --> E[Normalized Concept Vector v_l]
    
    F[Input User Prompt] --> G[Steered Generator]
    E --> G
    
    subgraph Intervention & Controls
        H[Layer Selector] -->|Scored Target Layers| G
        I[Adaptive Weight Strategy] -->|Layer Weight Ratios| G
        J[Dynamic Alpha Scheduler] -->|Per-Token α_t| G
    end
    
    G --> K[Steered Output Text & Logits]
    K --> L[Steering Evaluator]
    K --> M[SQLite Experiment Tracker]
    
    subgraph Evaluation & Analytics
        L --> L1[Perplexity PPL]
        L --> L2[Cosine Similarity]
        L --> L3[KL & JS Divergence]
        L --> L4[Token Entropy]
        L --> L5[Layer-wise Norm Difference]
    end
```

---

## 2. Dynamic Steering Autoregressive Loop Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User/UI
    participant SG as SteeredGenerator
    participant Sched as AlphaScheduler
    participant Hook as PyTorch Forward Hook
    participant Model as LLM Transformer
    participant Eval as SteeringEvaluator

    User/UI->>SG: generate_comparative_dynamic(prompt, vectors, scheduler)
    SG->>Model: Forward pass baseline (unsteered)
    Model-->>SG: baseline_tokens & logits
    
    SG->>Model: Register dynamic forward hooks on target layers
    loop Autoregressive Decoding Loop (t = 0 to max_tokens)
        SG->>Sched: step(step_idx=t, total_steps, logits)
        Sched-->>SG: alpha_t
        SG->>Hook: Update _dynamic_alpha[layer] = alpha_t * ratio_l
        SG->>Model: Forward pass 1 token step
        Hook->>Model: Inject h_l = h_l + alpha_t * ratio_l * v_l
        Model-->>SG: next_token_logits
        SG->>SG: Record alpha_t in AlphaTrajectory
    end
    SG->>Model: Remove all forward hooks
    SG->>Eval: evaluate_full(baseline, steered)
    Eval-->>SG: SteeringEvaluationReport
    SG-->>User/UI: Baseline Text, Steered Text, Trajectory, Evaluation Report
```

---

## 3. Automated Benchmark Engine Flowchart

```mermaid
flowchart LR
    A[BenchmarkGridConfig] --> B[BenchmarkEngine]
    B --> C{Loop over Grid}
    C -->|Model, Concept, Method, Strategy, Alpha| D[Concept Extraction]
    D --> E[Inference Pass]
    E --> F[Evaluation & Timing]
    F --> G[SingleBenchmarkRun]
    G --> C
    C -->|Grid Complete| H[Export Report]
    H --> H1[CSV Summary]
    H --> H2[JSON Summary]
    H --> H3[Markdown Summary]
```
