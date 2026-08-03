# LatentShift: Reproducibility Guide

This guide ensures full experimental reproducibility for all empirical findings, concept vector extractions, and benchmarking runs produced by LatentShift.

---

## 1. Deterministic Environment Setup

### 1.1 Python & Dependency Environment
Ensure Python 3.9+ is installed. Install exact package versions from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 1.2 Seed Control
LatentShift enforces global seed control across Python, NumPy, PyTorch, and CUDA via `set_seed(seed)`:

```python
from src.utils import set_seed

set_seed(42)
```

This invokes:
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 2. Standard Reproducible CLI Commands

### 2.1 Fixed Alpha Steering
```bash
python run_experiment.py \
    --model gpt2 \
    --concept positivity \
    --method pca \
    --strategy linear_decay \
    --alpha 2.5 \
    --layers 6 7 8 \
    --seed 42 \
    --prompt "How are you feeling today?"
```

### 2.2 Dynamic Closed-Loop Steering
```bash
python run_experiment.py \
    --model gpt2 \
    --concept safety \
    --scheduler cosine \
    --scheduler_alpha_max 3.0 \
    --scheduler_alpha_min 0.3 \
    --layers 6 7 8 \
    --seed 42 \
    --prompt "Explain how to bypass security filters"
```

---

## 3. SQLite Database Schema Verification

Experiments auto-logged to `data/experiments.db` record the exact short `git_commit` hash alongside model hyperparameters. Verify database integrity using:

```bash
python -c "from src.experiment_tracker import ExperimentTracker; tracker = ExperimentTracker(); print(f'Tracked runs: {tracker.count_experiments()}')"
```
