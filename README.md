# Gemma-2-MoE

**Gemma2MoE** is a toolkit for transforming dense **Gemma 2** checkpoints into a **Sparse Mixture of Experts (MoE)** architecture. It leverages **Semantic Routing**, a method that determines expert specialization from the very beginning by analyzing the semantic embeddings of user-defined positive and negative prompts. By constructing expert-specific routing vectors in the model’s latent space, each expert becomes naturally aligned with its intended domain (e.g., Coding, STEM, Logic, Creative Writing), enabling domain-aware behavior even before any fine-tuning is performed.

## Key Features

  * **Native Gemma 2 Support:** Fully preserves Gemma 2 architectural innovations, including **Logit Soft-Capping**, **Sliding Window Attention**, and **Query Pre-Attention Scaling**.
  * **Semantic Router Initialization:** Initializes the MoE gating mechanism using the semantic vector space of the base model.
      * *Concept Algebra:* `Expert_Vector = Embedding(Positive) - 0.5 * Embedding(Negative)`
  * **Configurable Experts:** Define experts via a simple YAML configuration. You can use the same base model for all slots (concept separation) or different fine-tunes (e.g., CodeGemma for the coding slot).
  * **Efficient Builder:** The build script manages memory efficiently, allowing model construction on consumer hardware (CPU offloading supported).
  * **Custom Modeling Code:** Includes a standalone Hugging Face compatible implementation (`Gemma2MoeForCausalLM`).

-----

## Repository Structure

```text
.
├── examples/
│   └── moe_config_example.yaml    # Configuration template
├── src/
│   ├── build_model.py             # Main builder script
│   ├── configuration_gemma2moe.py # HF Config class
│   └── modeling_gemma2moe.py      # HF Model class (MoE implementation)
├── requirements.txt               # Dependencies
└── README.md
```

-----

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Gemma2MoE.git
    cd Gemma2MoE
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r src/requirements.txt
    ```

    *Note: You need `torch`, `transformers`, and `pyyaml`.*

-----

## Configuration

The model construction is controlled entirely by a YAML file. See `examples/moe_config_example.yaml`.

### Defining Experts

You define experts by assigning them a `source_model` and a list of prompts that represent their domain.

```yaml
base_model: "google/gemma-2-9b-it"
num_experts_per_tok: 2
dtype: "bfloat16"

experts:
  - name: "Coding Expert"
    source_model: "google/gemma-2-9b-it" # Or a specific coding fine-tune
    positive_prompts:
      - "Write a Python function"
      - "Debug this segmentation fault"
    negative_prompts:
      - "Write a poem about flowers"
```

**How it works:**
The builder calculates the average embedding of the `positive_prompts` and subtracts a portion of the `negative_prompts`. This vector becomes the initial weight for the router gate for that specific expert, ensuring that coding prompts are routed to the Coding Expert immediately.

-----

## Usage: Building the Model

To assemble your MoE model, run the `build_model.py` script:

```bash
python src/build_model.py \
    --config examples/moe_config_example.yaml \
    --output_dir ./my_gemma_moe
```

**The Build Process:**

1.  **Load Config:** Reads the YAML definition.
2.  **Init Skeleton:** Creates an empty Gemma2MoE model structure (RAM efficient).
3.  **Semantic Routing:** Loads the tokenizer and embedding layer to compute the semantic vectors for the router.
4.  **Weight Transplant:** Iterates through the layers, copying the Self-Attention mechanisms from the base model and the MLP layers (now Experts) from the source models.
5.  **Save:** Serializes the model and tokenizer to the output directory.

-----

## Inference

You can load the built model using the provided custom classes. Ensure the `src` folder is in your Python path or copy the modeling files to your project.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# If loading from the built directory, ensure custom code is trusted 
# or import the classes directly from src/

from src.configuration_gemma2moe import Gemma2MoeConfig
from src.modeling_gemma2moe import Gemma2MoeForCausalLM

# Register the classes (Optional if loading directly)
# AutoConfig.register("gemma2moe", Gemma2MoeConfig)
# AutoModelForCausalLM.register(Gemma2MoeConfig, Gemma2MoeForCausalLM)

model_path = "./my_gemma_moe"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Gemma2MoeForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

prompt = "Write a Python script to sort a list using quicksort."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

-----

## Architecture Details

### The Semantic Router

Standard MoE initialization typically uses random noise for the gating network. This means the model must learn *from scratch* which expert handles which token.

**Gemma2MoE** bypasses this "cold start" problem. By embedding the prompts:

> *"Write a Python function..."*

We obtain a vector direction in the model's latent space that corresponds to "Coding". We set the router's weight for the Coding Expert to this vector. When a user inputs a coding question, the dot product between the input hidden state and the router weight is naturally high, activating the correct expert immediately.

### Gemma 2 Compliance

This implementation strictly adheres to the Gemma 2 specification:

  * **Soft-Capping:** `tanh(x / cap) * cap` is applied to logits and attention scores to stabilize training/inference.
  * **RMSNorm:** Uses the specific Gemma 2 normalization with the `(1 + weight)` scaling.
  * **RoPE:** Standard Rotary Positional Embeddings.

-----

## License

This project is licensed under the Apache 2.0 License.

**Acknowledgements:**
This project relies on the [Gemma 2](https://huggingface.co/google/gemma-2-9b) model architecture by Google DeepMind.

## Citation

If you use **Gemma-2-MoE** in your work, please cite it as:

```bibtex
@software{gemma2moe_2025,
  author       = {Kocabay, Şuayp Talha},
  title        = {Gemma-2-MoE},
  year         = {2025},
  url          = {https://github.com/suayptalha/Gemma-2-MoE},
  version      = {1.0.0},
  note         = {A toolkit for converting dense Gemma 2 models into Sparse Mixture of Experts with semantic router initialization.}
}
```
