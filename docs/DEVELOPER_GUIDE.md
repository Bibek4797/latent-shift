# LatentShift: Developer & Extensibility Guide

This guide outlines principles for maintaining, extending, and contributing to the LatentShift codebase.

---

## 1. Architectural Principles

1. **Zero Weight Mutation ($\Delta \theta = 0$)**:
   Never mutate model weights directly. All steering interventions must operate exclusively via PyTorch forward hooks attached to residual stream modules (`model.layers[l]`).

2. **Guaranteed Hook Cleanup**:
   Every hook registration must be encapsulated within a `try...finally` block to guarantee `hook.remove()` is called, preventing CUDA memory retention.

3. **Pluggable Registries**:
   Extend concept extractors, layer scoring methods, and alpha schedulers by subclassing base classes and registering instances in global dictionaries (`EXTRACTOR_REGISTRY`, `SCHEDULER_REGISTRY`).

---

## 2. Adding a Custom Concept Extractor

To add a new concept extraction algorithm (e.g., Kernel PCA):

1. Open `src/concept_extractors.py`.
2. Subclass `BaseConceptExtractor`:

```python
from src.concept_extractors import BaseConceptExtractor, EXTRACTOR_REGISTRY

class KernelPCAExtractor(BaseConceptExtractor):
    name = "kernel_pca"

    def fit(self, H_pos: torch.Tensor, H_neg: torch.Tensor) -> torch.Tensor:
        # Implement extraction logic returning 1D Tensor of shape (hidden_dim,)
        ...

# Register the new extractor
EXTRACTOR_REGISTRY["kernel_pca"] = KernelPCAExtractor()
```

---

## 3. Adding a Custom Alpha Scheduler

To implement a custom dynamic alpha schedule:

1. Open `src/schedulers.py`.
2. Subclass `BaseAlphaScheduler`:

```python
from src.schedulers import BaseAlphaScheduler, SCHEDULER_REGISTRY

class ExponentialDecayScheduler(BaseAlphaScheduler):
    name = "exponential"

    def __init__(self, alpha_start: float = 4.0, decay_rate: float = 0.95):
        self.alpha_start = alpha_start
        self.decay_rate = decay_rate

    def step(self, step_idx: int, total_steps: int, logits=None, **kwargs) -> float:
        return self.alpha_start * (self.decay_rate ** step_idx)

# Register the new scheduler
SCHEDULER_REGISTRY["exponential"] = ExponentialDecayScheduler
```

---

## 4. Running Unit Tests

Execute the full pytest suite before submitting pull requests:

```bash
python -m pytest tests/test_pipeline.py -v
```
