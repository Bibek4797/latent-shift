# Contributing to LatentShift

Thank you for your interest in contributing to **LatentShift**! We welcome contributions from researchers, machine learning engineers, and open-source developers.

---

## 1. Code of Conduct

We expect all contributors to adhere to standard scientific integrity, open-source professionalism, and respectful communication.

---

## 2. Development Setup

### 2.1 Environment Setup
1. Fork and clone the repository:
   ```bash
   git clone https://github.com/Bibek4797/latent-shift.git
   cd latent-shift
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest black ruff mypy
   ```

---

## 3. Core Architectural Rules

When contributing code to LatentShift:

1. **Zero Weight Mutation ($\Delta \theta = 0$)**:
   Never modify model weights. All steering interventions must operate exclusively via PyTorch forward hooks attached to transformer residual stream blocks (`model.model.layers[l]`).

2. **Guaranteed Hook Lifecycle**:
   Every hook registration must be encapsulated within a `try...finally` block to guarantee `hook.remove()` is called, preventing CUDA memory leaks.

3. **Stateless Extractor & Scheduler Registries**:
   Extend concept extractors, layer scoring methods, and alpha schedulers by subclassing base classes (`BaseConceptExtractor`, `BaseAlphaScheduler`) and registering instances in global dictionaries (`EXTRACTOR_REGISTRY`, `SCHEDULER_REGISTRY`).

4. **Inference Mode Execution**:
   Use `torch.inference_mode()` for non-training extraction, generation, and evaluation passes.

---

## 4. Coding & Docstring Standards

- **Python Version**: Python 3.9+ compatible.
- **Type Hints**: Use strict type hints on all public function and method signatures.
- **Docstring Style**: Write comprehensive NumPy-style docstrings for all classes and functions.
- **Formatting**: Format code using `black` (line length 100).

```python
def compute_vector(
    method: str,
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute concept vector using specified extraction method.

    Parameters
    ----------
    method : str
        Extraction method key.
    pos_activations : torch.Tensor
        Positive contrast activation matrix of shape (num_samples, hidden_dim).
    neg_activations : torch.Tensor
        Negative contrast activation matrix of shape (num_samples, hidden_dim).
    normalize : bool, default=False
        Whether to L2-normalize the output vector.

    Returns
    -------
    torch.Tensor
        Extracted 1D concept vector of shape (hidden_dim,).
    """
    ...
```

---

## 5. Testing Requirements

Run the full pytest suite before submitting pull requests:

```bash
python -m pytest tests/test_pipeline.py -v
```

All existing unit tests (110+) must pass. Add new unit tests in `tests/test_pipeline.py` for any new feature or bug fix.

---

## 6. Pull Request Process

1. Create a feature branch (`git checkout -b feature/my-new-feature`).
2. Commit your changes with clear, descriptive commit messages (`git commit -m "feat: Add Kernel PCA concept extractor"`).
3. Ensure all tests pass.
4. Push to your branch and open a Pull Request targeting `main`.
