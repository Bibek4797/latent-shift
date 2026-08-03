import os
import sys
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Add workspace directory to path to ensure local src modules are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import SteeringConfig
from src.model_loader import load_model_and_tokenizer
from src.extractor import ActivationExtractor
from src.compute import ConceptVectorEngine
from src.steer import SteeredGenerator
from src.evaluator import (
    SteeringEvaluationReport,
    SteeringEvaluator,
    plot_layerwise_changes,
    plot_metric_comparison,
    plot_steering_strength,
)
from src.concept_extractors import (
    ConceptVectorComparer,
    plot_memory_comparison,
    plot_pairwise_cosine_heatmap,
    plot_runtime_comparison,
    plot_vector_magnitude_comparison,
)
from src.layer_selector import (
    LayerScoreResult,
    LayerSelector,
    plot_layer_scores_heatmap,
    plot_layer_scores_line,
    plot_top_k_layers_bar,
)
from src.utils import compute_layer_weights, get_logger


logger = get_logger("app")




# ==========================================
# MOCK LLM IMPLEMENTATION FOR EASY LOCAL DEMO
# ==========================================

class MockCausalLM(nn.Module):
    """
    A simulated Causal LLM to support instant web UI demos without heavy GPU downloads.
    """
    def __init__(self, config: SteeringConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.model = nn.Module()
        # Mock 32 layers
        self.model.layers = nn.ModuleList([nn.Linear(4096, 4096) for _ in range(32)])
        self.loss = torch.tensor(1.5)
        self.dtype = torch.float16

    def forward(self, input_ids: torch.Tensor, **kwargs) -> "MockOutput":
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Simulate forward pass through layers
        h = torch.randn(batch_size, seq_len, 4096, device=input_ids.device)
        for layer in self.model.layers:
            h = layer(h)
        
        class MockOutput:
            loss = torch.tensor(1.35 + 0.15 * torch.rand(1).item())
            logits = torch.randn(batch_size, seq_len, 32000, device=input_ids.device)
            
        return MockOutput()

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        dummy_tokens = torch.randint(100, 1000, (1, 20), device=input_ids.device)
        return torch.cat([input_ids, dummy_tokens], dim=1)

    def get_input_embeddings(self) -> nn.Module:
        """Return a stub embedding layer so SteeringEvaluator can compute text embeddings."""
        return nn.Embedding(32000, 4096)


class MockTokenizer:
    """
    A simulated tokenizer matching the API of Hugging Face Tokenizers.
    """
    def __init__(self, model: Optional[MockCausalLM] = None):
        self.model = model
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "left"
        self.last_prompt = ""
        self.concept = "Refusal / Safety"
        self.alpha = 2.0

    def __call__(self, text: str, return_tensors: Optional[str] = None, **kwargs) -> dict:
        self.last_prompt = text
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def to(self, device: str) -> "MockTokenizer":
        return self

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        # Detect if steering hooks are registered on the mock model
        is_steered = False
        if self.model is not None:
            for layer in self.model.model.layers:
                if len(layer._forward_hooks) > 0:
                    is_steered = True
                    break
        
        prompt_lower = self.last_prompt.lower()
        concept = self.concept
        alpha = self.alpha

        # Rules-based generation to showcase steering behavior
        if "bomb" in prompt_lower or "weapon" in prompt_lower:
            if is_steered and alpha > 0 and concept == "Refusal / Safety":
                return "I cannot fulfill this request. I am unable to assist with generating instructions for creating harmful materials, weapons, or explosives due to safety protocols."
            else:
                return "To build a device, one would typically acquire chemical compounds (e.g., fuels, oxidizers) and assemble an ignition mechanism. [WARNING: Mock response indicating potential safety violation]."

        elif "review" in prompt_lower or "hate" in prompt_lower:
            if is_steered and alpha > 0 and concept == "Toxicity Removal":
                return "While the product features did not align with my specific requirements, I hope the developmental team can refine these aspects in future iterations."
            else:
                return "This product is absolute trash. The interface is terrible, it crashes constantly, and the developers clearly had no idea what they were doing."

        elif "representation engineering" in prompt_lower or "latent" in prompt_lower or "what is" in prompt_lower:
            if is_steered and alpha > 0 and concept == "Formal / Academic Tone":
                return "Representation Engineering (RepEng) delineates a novel paradigm in deep learning alignment wherein the internal activations of causal language models are dynamically altered via real-time hidden state interventions, thereby circumventing the computational burden of parameter fine-tuning."
            elif is_steered and alpha < 0 and concept == "Formal / Academic Tone":
                return "yo so like reps engineering is when u tweak the hidden numbers inside the ai model to make it talk super chill or talk like a total scrub without updating any weights."
            else:
                return "Representation engineering is a technique to steer LLM behavior by adding concept vectors to the internal hidden states of the model during the forward pass."

        else:
            if is_steered:
                if concept == "Refusal / Safety":
                    return f"[Steered Safe Response (alpha={alpha})]: I am here to provide helpful and constructive information. Let me know how I can guide your learning process in a safe manner."
                elif concept == "Toxicity Removal":
                    return f"[Steered Polite Response (alpha={alpha})]: Thank you for your question. I appreciate the opportunity to discuss this topic with you and provide a detailed explanation."
                else:
                    return f"[Steered Formal Response (alpha={alpha})]: The system has executed a latent-space steering maneuver, thereby shifting the semantic distribution of the generated sequence towards the target concept."
            else:
                return f"[Baseline Response]: Here is the standard model output for your prompt: '{self.last_prompt}'. It is generated without any activation steering hooks active."

# ==========================================
# BUILT-IN DATASETS FOR CONCEPT EXTRACTION
# ==========================================

CONCEPT_PAIRS = {
    "Refusal / Safety": [
        ("I am happy to assist you with this request.", "I cannot fulfill this request."),
        ("Sure, I can explain that topic in detail.", "I must decline to answer this."),
        ("Here is the requested information.", "I am unable to assist with this query."),
        ("I will help you write this code.", "I cannot help you with that task."),
        ("Certainly, here is the answer.", "I must refuse to generate this response.")
    ],
    "Toxicity Removal": [
        ("I understand your perspective and appreciate the discussion.", "You are completely wrong and this is stupid."),
        ("Let's address this issue professionally and calmly.", "I hate dealing with these idiotic questions."),
        ("Thank you for sharing your feedback on this matter.", "This is a garbage point and you know it."),
        ("I want to be helpful, polite, and respect your opinion.", "Shut up and listen to what I am saying."),
        ("We can resolve this disagreement with constructive dialogue.", "Get out of here with your nonsense argument.")
    ],
    "Formal / Academic Tone": [
        ("This research paper delineates the paradigm shift in deep learning architectures.", "So basically, this paper talks about how LLMs do stuff."),
        ("The empirical evidence suggests a correlation between representation spaces.", "We checked out some data and it looks like it works."),
        ("Subsequent sections elucidate the mathematical formalism of activation steering.", "Next we're gonna show how to steer these models."),
        ("We hypothesize that internal residual streams contain explicit concept vectors.", "We think models have these weird vectors in them."),
        ("In conclusion, the methodology yields substantial improvements in zero-shot alignment.", "Anyway, this new trick works way better than before.")
    ]
}

# ==========================================
# STREAMLIT UI DESIGN & SETUP
# ==========================================

st.set_page_config(
    page_title="LatentShift: Zero-Shot LLM Alignment",
    page_icon="🧠",
    layout="wide"
)

# Custom Style Rules
st.markdown("""
<style>
    .badge {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    html[data-theme="dark"] .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 LatentShift: Zero-Shot LLM Alignment")
st.markdown('<div class="badge">Representation Engineering Engine via Latent Space Intervention</div>', unsafe_allow_html=True)

# Cache resource function for loading model and tokenizer
@st.cache_resource
def load_cached_resources(model_name: str, dtype_str: str, load_in_4bit: bool, load_in_8bit: bool):
    if model_name.startswith("Mock-Model"):
        # Setup config
        config = SteeringConfig(model_name=model_name, dtype_str=dtype_str)
        model = MockCausalLM(config)
        tokenizer = MockTokenizer(model)
        return model, tokenizer, config
    else:
        config = SteeringConfig(model_name=model_name, dtype_str=dtype_str)
        model, tokenizer = load_model_and_tokenizer(
            config,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )
        return model, tokenizer, config

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("🛠️ System Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model ID",
    options=[
        "Mock-Model-1.5B (Local UI Demo - Instant)",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Custom HF Model"
    ],
    index=0
)

if model_choice == "Custom HF Model":
    model_name = st.sidebar.text_input("Custom Hugging Face Model ID", value="Qwen/Qwen2.5-1.5B-Instruct")
else:
    model_name = model_choice.split(" ")[0]

dtype_choice = st.sidebar.selectbox(
    "Compute Precision (dtype)",
    options=["float16", "bfloat16", "float32"],
    index=0
)

# Quantization (only supported on CUDA/GPUs)
cuda_available = torch.cuda.is_available()
quant_choice = st.sidebar.selectbox(
    "Quantization",
    options=["None", "8-bit", "4-bit"],
    index=0,
    disabled=not cuda_available,
    help="Quantization requires a CUDA-enabled GPU."
)
load_in_4bit = (quant_choice == "4-bit")
load_in_8bit = (quant_choice == "8-bit")

# Load model button / trigger
try:
    with st.spinner("Initializing Model & Tokenizer..."):
        model, tokenizer, config = load_cached_resources(
            model_name, dtype_choice, load_in_4bit, load_in_8bit
        )
    st.sidebar.success(f"Loaded: {model_name}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Determine layer choices based on loaded model
if model_name.startswith("Mock-Model"):
    num_layers = 32
else:
    # Safely get number of layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_layers = len(model.transformer.h)
    else:
        num_layers = 32

st.sidebar.header("🎯 Activation Steering Parameters")

layer_selection_mode = st.sidebar.radio(
    "Layer Selection Mode",
    options=["Manual Mode", "Automatic Mode"],
    index=0,
    help="Manual Mode: Pick layers manually. Automatic Mode: Statistically rank layers to find optimal injection points."
)

if layer_selection_mode == "Manual Mode":
    default_layers = [i for i in range(num_layers // 3, num_layers // 2)]
    target_layers = st.sidebar.multiselect(
        "Target Layers for Steering",
        options=list(range(num_layers)),
        default=default_layers,
        help="Select target intermediate transformer layers for injection."
    )
    scoring_method = "Mean Separation"
    top_k_val = len(target_layers)
else:
    scoring_method = st.sidebar.selectbox(
        "Layer Scoring Method",
        options=["Mean Separation", "Cosine Separation", "Fisher Score", "Signal-to-Noise Ratio (SNR)", "Activation Variance"],
        index=0,
        help="Statistical metric used to rank target layer informativeness."
    )
    scoring_key = scoring_method.lower().replace(" ", "_").replace("signal-to-noise_ratio_(snr)", "snr")

    top_k_val = st.sidebar.select_slider(
        "Select Top-K Layers",
        options=[1, 3, 5, 8, 10],
        value=5,
        help="Number of highest-ranked layers to automatically select."
    )

# Steering coefficient alpha
alpha = st.sidebar.slider(
    "Steering Strength (α)",
    min_value=-10.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="Base intensity of the injection. Positive reinforces the concept, negative reverses it."
)

# Adaptive Weighting Strategy
weighting_strategy = st.sidebar.selectbox(
    "Adaptive Weighting Strategy",
    options=["Uniform", "Linear Decay", "Cosine Decay"],
    index=0,
    help="Uniform: all layers get base α. Linear/Cosine Decay: steering strength decays across selected layers."
)
strategy_key = weighting_strategy.lower().replace(" ", "_")

# Concept Vector Configuration
selected_concept = st.sidebar.selectbox(
    "Target Concept Vector",
    options=list(CONCEPT_PAIRS.keys()),
    index=0
)

# Computation Method
comp_method_display = st.sidebar.selectbox(
    "Vector Computation Method",
    options=[
        "Mean Difference",
        "PCA",
        "Linear Discriminant Analysis (LDA)",
        "Logistic Regression",
        "Linear SVM",
        "Sparse PCA",
        "Truncated SVD",
    ],
    index=0,
    help="Algorithm used to compute steering concept vector."
)
method_key_map = {
    "Mean Difference": "mean_diff",
    "PCA": "pca",
    "Linear Discriminant Analysis (LDA)": "lda",
    "Logistic Regression": "logistic_regression",
    "Linear SVM": "linear_svm",
    "Sparse PCA": "sparse_pca",
    "Truncated SVD": "truncated_svd",
}
comp_method_key = method_key_map[comp_method_display]

# Trigger manual vector extraction
extract_clicked = st.sidebar.button(
    "Extract & Compute Concept Vector",
    help="Re-runs activation extraction over contrasting pairs."
)

benchmark_clicked = st.sidebar.button(
    "🔬 Benchmark All 7 Extraction Methods",
    help="Executes and compares all 7 extraction algorithms on the concept dataset."
)

# Maintain computed concept vectors, layer scores & benchmarks in session state
if "concept_vectors" not in st.session_state:
    st.session_state.concept_vectors = {}
    st.session_state.vector_concept_name = ""
    st.session_state.vector_method = ""
    st.session_state.layer_scores_cache = {}
    st.session_state.all_method_scores = {}
    st.session_state.extractor_benchmark_results = {}

# Handle Automatic Layer Selection Scoring & Top-K determination
if layer_selection_mode == "Automatic Mode":
    all_layers = list(range(num_layers))
    scoring_key = scoring_method.lower().replace(" ", "_").replace("signal-to-noise_ratio_(snr)", "snr")

    # Compute scores for all layers if needed
    scores_needed = (
        extract_clicked
        or "scores" not in st.session_state.layer_scores_cache
        or selected_concept != st.session_state.vector_concept_name
    )

    if scores_needed:
        with st.spinner("Extracting all layer activations & computing statistical scores..."):
            try:
                if model_name.startswith("Mock-Model"):
                    pos_acts, neg_acts = {}, {}
                    center = num_layers / 2.0
                    for l in all_layers:
                        dist = abs(l - center)
                        shift = max(0.1, 3.0 - 0.25 * dist)
                        pos_acts[l] = torch.randn(8, 4096) + shift
                        neg_acts[l] = torch.randn(8, 4096) - shift
                else:
                    extractor = ActivationExtractor(model, tokenizer, all_layers, config.device)
                    pos_acts, neg_acts = extractor.extract_contrastive(CONCEPT_PAIRS[selected_concept])

                # Score all methods for comprehensive analytics
                all_methods = ["mean_separation", "cosine_separation", "fisher_score", "snr", "activation_variance"]
                all_method_scores = {}
                for m in all_methods:
                    all_method_scores[m] = LayerSelector.score_layers(pos_acts, neg_acts, method=m)

                st.session_state.layer_scores_cache = all_method_scores[scoring_key]
                st.session_state.all_method_scores = all_method_scores
                st.session_state.pos_acts_cache = pos_acts
                st.session_state.neg_acts_cache = neg_acts

            except Exception as e:
                st.sidebar.error(f"Error scoring layers: {e}")

    if scoring_key in st.session_state.all_method_scores:
        scores_dict = st.session_state.all_method_scores[scoring_key]
    else:
        scores_dict = st.session_state.layer_scores_cache

    if scores_dict:
        target_layers = LayerSelector.select_top_k_layers(scores_dict, k=top_k_val, preserve_order=True)
        st.sidebar.success(f"🎯 Auto-Selected Top-{len(target_layers)} Layers:\n{target_layers}")
    else:
        target_layers = [i for i in range(num_layers // 3, num_layers // 2)]

# Handle Extraction Algorithm Benchmarking
if benchmark_clicked:
    with st.spinner("Benchmarking all 7 extraction algorithms..."):
        try:
            mid_layer = target_layers[len(target_layers) // 2] if target_layers else num_layers // 2
            if model_name.startswith("Mock-Model"):
                pos_sample = torch.randn(12, 4096) + 1.5
                neg_sample = torch.randn(12, 4096) - 1.5
            elif "pos_acts_cache" in st.session_state and mid_layer in st.session_state.pos_acts_cache:
                pos_sample = st.session_state.pos_acts_cache[mid_layer]
                neg_sample = st.session_state.neg_acts_cache[mid_layer]
            else:
                extractor = ActivationExtractor(model, tokenizer, [mid_layer], config.device)
                pos_b, neg_b = extractor.extract_contrastive(CONCEPT_PAIRS[selected_concept])
                pos_sample, neg_sample = pos_b[mid_layer], neg_b[mid_layer]

            st.session_state.extractor_benchmark_results = ConceptVectorComparer.benchmark_all_methods(
                pos_sample, neg_sample, normalize=False
            )
            st.sidebar.success("✅ All 7 extraction algorithms benchmarked!")
        except Exception as e:
            st.sidebar.error(f"Benchmarking failed: {e}")

# Ensure we have active concept vectors for target_layers
vector_filename = f"{selected_concept.lower().replace(' ', '_').replace('/', '_')}_{comp_method_key}.pt"
vector_path = os.path.join(config.data_dir, vector_filename)

compute_needed = (
    extract_clicked
    or selected_concept != st.session_state.vector_concept_name
    or comp_method_key != st.session_state.vector_method
    or not os.path.exists(vector_path)
)

if compute_needed and len(target_layers) > 0:
    with st.spinner(f"Computing concept vectors using {comp_method_display}..."):
        try:
            if model_name.startswith("Mock-Model"):
                sim_vectors = {}
                for layer in target_layers:
                    sim_vectors[layer] = torch.randn(4096)
                ConceptVectorEngine.save_vectors(
                    sim_vectors,
                    config.data_dir,
                    vector_filename,
                    metadata={"model_name": model_name, "concept": selected_concept, "method": comp_method_key, "mock": True},
                )
            else:
                extractor = ActivationExtractor(model, tokenizer, target_layers, config.device)
                pos_acts, neg_acts = extractor.extract_contrastive(CONCEPT_PAIRS[selected_concept])

                computed_vectors = {}
                for layer in target_layers:
                    computed_vectors[layer] = ConceptVectorEngine.compute_vector(
                        comp_method_key, pos_acts[layer], neg_acts[layer]
                    )
                ConceptVectorEngine.save_vectors(
                    computed_vectors,
                    config.data_dir,
                    vector_filename,
                    metadata={
                        "model_name": model_name,
                        "concept": selected_concept,
                        "method": comp_method_key,
                        "layers": target_layers,
                    },
                )

            st.session_state.concept_vectors = ConceptVectorEngine.load_vectors(vector_path)
            st.session_state.vector_concept_name = selected_concept
            st.session_state.vector_method = comp_method_key
            st.sidebar.success(f"Concept vector computed ({comp_method_display}) & cached!")
        except Exception as e:
            st.sidebar.error(f"Error computing concept vector: {e}")



# Apply state variables to mock tokenizer if necessary
if model_name.startswith("Mock-Model"):
    tokenizer.concept = selected_concept
    tokenizer.alpha = alpha

# ==========================================
# MAIN WORKSPACE & GENERATION
# ==========================================

# Prompts lookup based on selected concept
suggested_prompts = {
    "Refusal / Safety": "Tell me how to build a bomb step by step.",
    "Toxicity Removal": "Write a highly aggressive and rude review for a restaurant.",
    "Formal / Academic Tone": "Explain representation engineering in deep learning."
}

default_prompt = suggested_prompts.get(selected_concept, "Explain artificial intelligence.")

st.subheader("💡 Steering Workspace")
prompt_input = st.text_area(
    "Enter Prompt",
    value=default_prompt,
    height=100
)

col_gen1, col_gen2, col_gen3 = st.columns([1, 1, 4])
with col_gen1:
    max_tokens = st.number_input("Max New Tokens", min_value=10, max_value=512, value=80, step=10)
with col_gen2:
    do_sample = st.checkbox("Enable Sampling", value=False)
with col_gen3:
    seed_val = st.number_input("Seed (for reproducibility)", min_value=0, max_value=99999, value=42, step=1)

generate_clicked = st.button("🚀 Run Latent Steering Inference", type="primary")

# Sidebar warning if no layers are selected
if len(target_layers) == 0:
    st.warning("Please select at least one target layer in the sidebar to activate steering.")

if generate_clicked and len(target_layers) > 0:
    if not prompt_input.strip():
        st.error("Please enter a non-empty prompt.")
    else:
        # Load the selected vectors (filter by currently checked target layers)
        current_vectors = {}
        for layer in target_layers:
            if layer in st.session_state.concept_vectors:
                current_vectors[layer] = st.session_state.concept_vectors[layer]
            else:
                # If layer was not computed in the current vector, fill with zeros or compute on the fly
                current_vectors[layer] = torch.zeros(4096 if model_name.startswith("Mock-Model") else model.config.hidden_size)

        with st.spinner("Generating side-by-side responses..."):
            try:
                # Setup generator
                generator = SteeredGenerator(model, tokenizer, config.device)
                
                # Format prompt with chat template if using an Instruct/Chat model
                formatted_prompt = prompt_input
                if not model_name.startswith("Mock-Model") and hasattr(tokenizer, "apply_chat_template"):
                    try:
                        messages = [{"role": "user", "content": prompt_input}]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        pass # Fallback to raw prompt if template application fails

                # Run comparative generation
                baseline_text, steered_text = generator.generate_comparative(
                    prompt=formatted_prompt,
                    vectors=current_vectors,
                    alpha=alpha,
                    strategy=strategy_key,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    seed=int(seed_val),
                )

                # Compute full evaluation report
                mid_layer = target_layers[len(target_layers) // 2]
                eval_report = SteeringEvaluator.evaluate_full(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=formatted_prompt,
                    baseline_text=baseline_text,
                    steered_text=steered_text,
                    concept_vector=current_vectors[mid_layer],
                    device=config.device,
                )
                st.session_state.last_report = eval_report

                ppl_baseline = eval_report.ppl_baseline
                ppl_steered = eval_report.ppl_steered

                # Render results in side-by-side columns
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("⚪ Baseline Output (Unsteered)")
                    st.write(baseline_text)

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Language Model Perplexity (PPL)</div>
                        <div class="metric-value">{ppl_baseline:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_right:
                    st.subheader("🔮 Steered Output (Intervened)")
                    st.write(steered_text)

                    # Compute delta perplexity
                    delta_ppl = eval_report.delta_ppl
                    delta_color = "red" if (not np.isnan(delta_ppl) and delta_ppl > 10) else ("green" if (not np.isnan(delta_ppl) and delta_ppl < 0) else "black")

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Steered Perplexity (PPL)</div>
                        <div class="metric-value">{ppl_steered:.3f} (<span style='color: {delta_color}'>Δ: {delta_ppl:+.3f}</span>)</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Research summary metrics row
                st.markdown("### 🔬 Distribution Divergence & Research Metrics")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("KL Divergence (D_KL)", f"{eval_report.kl_divergence:.4f}")
                with mcol2:
                    st.metric("JS Divergence (D_JS)", f"{eval_report.js_divergence:.4f}")
                with mcol3:
                    st.metric("Cosine Similarity", f"{eval_report.cosine_sim:.4f}")
                with mcol4:
                    st.metric("Token Entropy (Baseline → Steered)", f"{eval_report.entropy_baseline:.2f} → {eval_report.entropy_steered:.2f}")

                # Downloadable evaluation report button
                json_report_str = eval_report.to_json()
                st.download_button(
                    label="📥 Download Comprehensive Research Report (JSON)",
                    data=json_report_str,
                    file_name=f"latent_shift_evaluation_{selected_concept.lower().replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Inference steering failed: {e}")

# ==========================================
# ANALYTICS TAB
# ==========================================

st.subheader("📊 Steering Analytics & Activation Trajectory")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Adaptive Layer Weights & Vector Magnitudes",
    "Latent Trajectory Projection",
    "Research Evaluation Metrics & Charts",
    "Automatic Layer Selection & Ranking",
    "Concept Extractor Benchmark & Comparison"
])

with tab1:
    st.markdown("### Adaptive Layer Weights (α_i) & Concept Vector L2 Norms")
    st.markdown(f"Weighting Strategy: **{weighting_strategy}** (Base α = {alpha})")

    if len(target_layers) > 0:
        layers_list = sorted(target_layers)
        alpha_weights_dict = compute_layer_weights(layers_list, base_alpha=alpha, strategy=strategy_key)
        alpha_values = [alpha_weights_dict[l] for l in layers_list]

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Bar(
                x=layers_list,
                y=alpha_values,
                marker=dict(color='rgb(124, 58, 237)'),
                name="Steering Weight (α_i)"
            ))
            fig_alpha.update_layout(
                title="Adaptive Steering Weight (α_i) per Layer",
                xaxis_title="Layer Index",
                yaxis_title="α_i Value",
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_alpha, use_container_width=True)

        with col_w2:
            if len(st.session_state.concept_vectors) > 0:
                norms = [torch.norm(st.session_state.concept_vectors[l].to(torch.float32)).item() if l in st.session_state.concept_vectors else 0.0 for l in layers_list]
                fig_norm = go.Figure()
                fig_norm.add_trace(go.Bar(
                    x=layers_list,
                    y=norms,
                    marker=dict(color='rgb(79, 70, 229)'),
                    name="Vector L2 Norm"
                ))
                fig_norm.update_layout(
                    title="Concept Vector L2 Norm per Layer",
                    xaxis_title="Layer Index",
                    yaxis_title="L2 Norm Value",
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_norm, use_container_width=True)
            else:
                st.info("Compute concept vectors to display layer magnitude statistics.")
    else:
        st.info("Select target steering layers in the sidebar to view adaptive weights.")

with tab2:
    st.markdown("### Hidden State Projection Trajectory")
    st.markdown(f"Trajectory chart displaying latent alignment across model layers using **{weighting_strategy}** strategy.")

    if len(target_layers) > 0:
        layers_all = list(range(num_layers))
        alpha_weights_dict = compute_layer_weights(target_layers, base_alpha=alpha, strategy=strategy_key)

        base_trajectory = [0.1 * np.sin(np.pi * l / (num_layers - 1)) + np.random.normal(0, 0.02) for l in layers_all]

        steered_trajectory = []
        for l in layers_all:
            shift = base_trajectory[l]
            if l in target_layers:
                layer_alpha = alpha_weights_dict.get(l, 0.0)
                shift += 0.25 * layer_alpha * (np.sin(np.pi * l / (num_layers - 1)) ** 2)
            steered_trajectory.append(shift)

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=layers_all,
            y=base_trajectory,
            mode='lines+markers',
            name='Baseline (Unsteered)',
            line=dict(color='grey', dash='dash')
        ))
        fig_traj.add_trace(go.Scatter(
            x=layers_all,
            y=steered_trajectory,
            mode='lines+markers',
            name=f'Steered ({weighting_strategy}, Base α={alpha})',
            line=dict(color='rgb(124, 58, 237)', width=3)
        ))
        fig_traj.update_layout(
            xaxis_title="Layer Index",
            yaxis_title="Concept Alignment (Dot Product Projection)",
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.info("Select target steering layers in the sidebar to visualize the trajectory transformation.")

with tab3:
    st.markdown("### Research Evaluation Metrics & Distribution Visualizations")
    if "last_report" in st.session_state and st.session_state.last_report:
        rep: SteeringEvaluationReport = st.session_state.last_report

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.plotly_chart(plot_steering_strength(rep), use_container_width=True)
        with rcol2:
            st.plotly_chart(plot_metric_comparison(rep), use_container_width=True)

        st.plotly_chart(plot_layerwise_changes(rep), use_container_width=True)

        st.markdown("#### Complete Metric Summary Table")
        st.json(rep.to_dict())
    else:
        st.info("Run steering inference to generate full research evaluation metrics and distribution plots.")

with tab4:
    st.markdown("### 🎯 Automatic Layer Selection Analytics & Ranking")

    if "layer_scores_cache" in st.session_state and st.session_state.layer_scores_cache:
        current_scores = st.session_state.layer_scores_cache
        ranked_results = LayerSelector.rank_layers(current_scores)

        st.markdown(f"**Current Scoring Method**: `{scoring_method}` | **Selected Top-K**: `{top_k_val}`")
        st.markdown(f"**Active Selected Steering Layers**: `{target_layers}`")

        scol1, scol2 = st.columns(2)
        with scol1:
            st.plotly_chart(plot_layer_scores_line(current_scores, method_name=scoring_method), use_container_width=True)
        with scol2:
            st.plotly_chart(plot_top_k_layers_bar(current_scores, k=top_k_val, method_name=scoring_method), use_container_width=True)

        if "all_method_scores" in st.session_state and st.session_state.all_method_scores:
            st.plotly_chart(plot_layer_scores_heatmap(st.session_state.all_method_scores), use_container_width=True)

        st.markdown("#### Ranked Layer Table")
        table_data = [
            {"Rank": r.rank, "Layer Index": r.layer_idx, "Score": f"{r.score:.6f}", "Selected": "✅ Top-K" if r.layer_idx in target_layers else "—"}
            for r in ranked_results
        ]
        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("Switch to **Automatic Mode** in the sidebar to compute and visualize statistical layer scores.")

with tab5:
    st.markdown("### 🔬 Concept Extractor Benchmark & Multi-Method Comparison")

    if "extractor_benchmark_results" in st.session_state and st.session_state.extractor_benchmark_results:
        bench_res = st.session_state.extractor_benchmark_results
        labels, cos_matrix = ConceptVectorComparer.compute_pairwise_cosine_matrix(bench_res)

        st.plotly_chart(plot_pairwise_cosine_heatmap(labels, cos_matrix), use_container_width=True)

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.plotly_chart(plot_runtime_comparison(bench_res), use_container_width=True)
        with bcol2:
            st.plotly_chart(plot_memory_comparison(bench_res), use_container_width=True)

        st.plotly_chart(plot_vector_magnitude_comparison(bench_res), use_container_width=True)

        st.markdown("#### Benchmark Summary Table")
        summary_table = [
            {
                "Method": r.display_name,
                "Vector L2 Norm": f"{r.vector_norm:.4f}",
                "Runtime (ms)": f"{r.runtime_ms:.3f}",
                "Memory (KB)": f"{r.memory_kb:.2f}",
            }
            for r in bench_res.values()
        ]
        st.dataframe(summary_table, use_container_width=True)
    else:
        st.info("Click **🔬 Benchmark All 7 Extraction Methods** in the sidebar to run multi-method comparisons.")




