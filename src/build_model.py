# coding=utf-8
# Copyright 2024 suayptalha and Gemma2MoE Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gemma-2-MoE Builder Script.
This script assembles a Mixture of Experts model from multiple Gemma-2 checkpoints
defined in a YAML configuration file.
"""

import argparse
import gc
import os
import yaml
import torch
import torch.nn.functional as F
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from configuration_gemma2moe import Gemma2MoeConfig
from modeling_gemma2moe import Gemma2MoeForCausalLM

# Register Custom Architecture
AutoConfig.register("gemma2moe", Gemma2MoeConfig)
AutoModelForCausalLM.register(Gemma2MoeConfig, Gemma2MoeForCausalLM)

def get_mean_embedding(prompts: List[str], tokenizer, embedding_layer, device="cpu"):
    """
    Calculates the weighted mean embedding vector for a list of prompts.
    Used to initialize router weights semantically.
    """
    if not prompts:
        # Return random vector if no prompts provided (break symmetry)
        return torch.randn(embedding_layer.weight.shape[1], device=device)
        
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        # Shape: (Batch, Seq, Hidden)
        embeds = embedding_layer(input_ids)
        
        # Masking: Ignore padding tokens
        attention_mask = inputs.attention_mask.unsqueeze(-1).expand(embeds.size()).float()
        sum_embeds = torch.sum(embeds * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        
        # Sentence-level average
        sentence_embeds = sum_embeds / sum_mask
        
        # Mean of all prompts
        final_embed = sentence_embeds.mean(dim=0)
        
    return final_embed

def build_model_from_yaml(config_path, output_dir):
    """
    Main builder function.
    1. Loads config.
    2. Initializes empty MoE shell.
    3. Calculates router weights via semantic embeddings.
    4. Transplants MLP weights from source models to experts.
    """
    
    # 
    
    # 1. Load Configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model_id = cfg.get("base_model", "google/gemma-2-9b-it")
    experts_list = cfg.get("experts", [])
    num_local_experts = len(experts_list)
    num_experts_per_tok = cfg.get("num_experts_per_tok", 2)
    dtype_str = cfg.get("dtype", "float16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Build Configuration ---")
    print(f"Base Model: {base_model_id}")
    print(f"Target Expert Count: {num_local_experts}")
    print(f"Active Experts per Token: {num_experts_per_tok}")
    print(f"Precision: {dtype}")
    print(f"---------------------------")

    # [1/4] Load Base Configuration
    print("\n[1/4] Loading Base Configuration...")
    base_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    
    moe_config = Gemma2MoeConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        head_dim=base_config.head_dim,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        # Gemma 2 specific attributes
        query_pre_attn_scalar=getattr(base_config, "query_pre_attn_scalar", 224),
        sliding_window=getattr(base_config, "sliding_window", 4096),
        logit_soft_capping=getattr(base_config, "final_logit_softcapping", getattr(base_config, "logit_soft_capping", 30.0)),
        attn_logit_soft_capping=getattr(base_config, "attn_logit_softcapping", 50.0),
    )

    # [2/4] Initialize Empty Model
    print(f"\n[2/4] Initializing Empty Gemma2MoE Model...")
    # This creates the skeleton on CPU to save VRAM
    moe_model = Gemma2MoeForCausalLM(moe_config).to(dtype)
    
    # [3/4] Smart Router Initialization
    print("\n[3/4] Calculating Router Weights (Semantic Initialization)...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # We load the base model to get embeddings and shared layers
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=dtype, 
        device_map="cpu", # Keep on CPU, move specific layers to GPU if needed
        trust_remote_code=True
    )
    
    print("  > Copying shared layers (Embeddings, Norms, Self-Attn)...")
    moe_model.model.embed_tokens.load_state_dict(ref_model.model.embed_tokens.state_dict())
    moe_model.model.norm.load_state_dict(ref_model.model.norm.state_dict())
    moe_model.lm_head.load_state_dict(ref_model.lm_head.state_dict())
    
    # Copy shared components of decoder layers (Self-Attention & Norms)
    # MoE/MLP parts are left empty to be filled later
    for i in range(len(moe_model.model.layers)):
        src = ref_model.model.layers[i]
        tgt = moe_model.model.layers[i]
        tgt.input_layernorm.load_state_dict(src.input_layernorm.state_dict())
        tgt.post_attention_layernorm.load_state_dict(src.post_attention_layernorm.state_dict())
        tgt.self_attn.load_state_dict(src.self_attn.state_dict())
        tgt.pre_feedforward_layernorm.load_state_dict(src.pre_feedforward_layernorm.state_dict())
        tgt.post_feedforward_layernorm.load_state_dict(src.post_feedforward_layernorm.state_dict())

    # --- Router Calculation Logic ---
    embedding_layer = ref_model.model.embed_tokens.to(device)
    router_weights_list = []

    print("  > Computing prompt embeddings for experts...")
    for idx, expert_def in enumerate(experts_list):
        pos_prompts = expert_def.get("positive_prompts", [])
        neg_prompts = expert_def.get("negative_prompts", [])
        
        pos_emb = get_mean_embedding(pos_prompts, tokenizer, embedding_layer, device)
        
        if neg_prompts:
             neg_emb = get_mean_embedding(neg_prompts, tokenizer, embedding_layer, device)
             # Concept algebra: Expert = Positive - (0.5 * Negative)
             expert_vector = pos_emb - (neg_emb * 0.5) 
        else:
             expert_vector = pos_emb

        router_weights_list.append(expert_vector)
        print(f"    Expert {idx} ({expert_def.get('name', 'Unknown')}): Initialized.")

    # Stack weights
    router_matrix = torch.stack(router_weights_list).float().cpu()
    
    # --- CENTERING & SCALING ---
    # Subtract mean to ensure experts specialize in differences, not shared commonalities.
    mean_vector = torch.mean(router_matrix, dim=0, keepdim=True)
    router_matrix = router_matrix - mean_vector
    
    # Normalize for cosine similarity behavior
    router_matrix = F.normalize(router_matrix, p=2, dim=1)
    
    # Scale to sharpen activations (Temperature scaling)
    router_matrix = router_matrix * 5.0
    router_matrix = router_matrix.to(dtype)

    print("  > Assigning and Jittering Router Weights...")
    for layer in moe_model.model.layers:
        # Add slight noise per layer to encourage diversity in routing decisions across depth
        noise = torch.randn_like(router_matrix) * 0.02
        layer.block_sparse_moe.gate.weight.data = (router_matrix + noise.to(dtype)).clone()

    # Cleanup Reference Model
    del ref_model
    del embedding_layer
    gc.collect()
    torch.cuda.empty_cache()

    # [4/4] Transplanting Expert Weights
    print("\n[4/4] Transplanting Expert Weights...")
    
    # 
    
    # Group experts by source model to minimize I/O overhead
    source_map = {} 
    for idx, expert_def in enumerate(experts_list):
        src = expert_def["source_model"]
        if src not in source_map:
            source_map[src] = []
        source_map[src].append(idx)
    
    for src_model_id, target_indices in source_map.items():
        print(f"  > Loading Source: {src_model_id}")
        print(f"    -> Targets Experts: {target_indices}")
        
        src_model = AutoModelForCausalLM.from_pretrained(
            src_model_id,
            torch_dtype=dtype,
            device_map="cpu", 
            trust_remote_code=True
        )
        
        print("    -> Copying layers...")
        for layer_idx in range(len(moe_model.model.layers)):
            # Gemma 2 uses 'mlp' for the feedforward block
            src_mlp = src_model.model.layers[layer_idx].mlp
            tgt_block = moe_model.model.layers[layer_idx].block_sparse_moe
            
            for exp_idx in target_indices:
                # Copy Up, Gate, Down projections
                tgt_block.experts[exp_idx].load_state_dict(src_mlp.state_dict())
        
        del src_model
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    print(f"\nSaving finalized model to {output_dir}...")
    moe_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Build Complete! Model is ready for inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma-2-MoE Model Builder")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output_dir", type=str, default="./built_model", help="Directory to save the built model.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    build_model_from_yaml(args.config, args.output_dir)