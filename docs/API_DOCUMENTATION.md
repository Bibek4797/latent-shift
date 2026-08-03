# LatentShift: Core API Documentation

Comprehensive API reference for the primary classes, interfaces, and methods in LatentShift.

---

## 1. `ConceptVectorEngine` (`src/compute.py`)

Primary engine for concept vector extraction and disk cache management.

```python
class ConceptVectorEngine:
    def __init__(self, data_dir: str = "data/concept_vectors") -> None: ...
```

### Key Methods

- **`compute_concept_vector(model, tokenizer, pos_prompts, neg_prompts, target_layers, method="mean_diff", normalize=True)`**:
  Computes concept vectors across target layers using the specified algorithm.
  - **Parameters**:
    - `model`: Hugging Face model instance.
    - `tokenizer`: Hugging Face tokenizer instance.
    - `pos_prompts` (`List[str]`): Positive contrast prompt dataset.
    - `neg_prompts` (`List[str]`): Negative contrast prompt dataset.
    - `target_layers` (`List[int]`): List of target transformer layer indices.
    - `method` (`str`): Extraction algorithm (`mean_diff`, `pca`, `lda`, `logistic_regression`, `linear_svm`, `sparse_pca`, `truncated_svd`).
    - `normalize` (`bool`): Whether to $L_2$-normalize vectors.
  - **Returns**: `Dict[int, torch.Tensor]` mapping layer index to concept vector.

- **`save_vectors(concept_name, vectors, metadata=None)`**: Saves vectors to PyTorch `.pt` format with metadata.
- **`load_vectors(concept_name)`**: Loads saved vector dictionary and metadata.

---

## 2. `SteeredGenerator` (`src/steer.py`)

Engine for forward hook injection and baseline vs. steered autoregressive text generation.

```python
class SteeredGenerator:
    def __init__(self, model, tokenizer, device: str = "cuda") -> None: ...
```

### Key Methods

- **`register_steering_hooks(vectors, alpha=2.0, strategy="uniform", min_ratio=0.2)`**:
  Attaches forward hooks to target layers using fixed or weighted alpha scaling.
- **`register_dynamic_steering_hooks(vectors, layer_weight_ratios=None)`**:
  Attaches dynamic hooks reading from a mutable `_dynamic_alpha` container.
- **`remove_steering_hooks()`**: Removes all active hooks from the model.
- **`generate(prompt, max_new_tokens=80, temperature=0.7, top_p=0.9, do_sample=False, seed=42)`**:
  Generates text under the current hook configuration.
- **`generate_comparative(prompt, vectors, alpha=2.0, strategy="uniform", max_new_tokens=80, ...)`**:
  Generates baseline (unsteered) and steered text side-by-side.
- **`generate_dynamic(prompt, vectors, scheduler, strategy="uniform", max_new_tokens=80, ...)`**:
  Generates text with dynamic alpha adaptation step-by-step.
  - **Returns**: `Tuple[str, AlphaTrajectory]`.

---

## 3. `SteeringEvaluator` (`src/evaluator.py`)

Research evaluator computing perplexity, divergences, cosine similarity, and norm shifts.

```python
class SteeringEvaluator:
    @staticmethod
    def evaluate_full(model, tokenizer, prompt, baseline_text, steered_text, concept_vector, device="cuda") -> SteeringEvaluationReport: ...
```

---

## 4. `ExperimentTracker` (`src/experiment_tracker.py`)

SQLite database interface for experiment persistence and retrieval.

```python
class ExperimentTracker:
    def __init__(self, db_path: str = "data/experiments.db") -> None: ...
    def log_experiment(self, record: ExperimentRecord) -> str: ...
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]: ...
    def list_experiments(self, limit=100, offset=0, model_filter=None, concept_filter=None, method_filter=None) -> List[ExperimentRecord]: ...
    def compare_experiments(self, ids: List[str]) -> List[ExperimentRecord]: ...
    def export_experiments_csv(self, filepath: str, ids=None) -> str: ...
    def export_experiments_json(self, filepath: str, ids=None) -> str: ...
```
