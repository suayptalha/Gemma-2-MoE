# coding=utf-8
# Copyright 2025 suayptalha and Gemma2MoE Contributors.
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
""" Gemma2MoE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Gemma2MoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma2MoeModel`]. It is used to instantiate an
    Gemma2MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Gemma-2-9b but with MoE capabilities.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    """
    model_type = "gemma2moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3584,
        intermediate_size=14336,
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        
        # Gemma 2 Specific Args
        query_pre_attn_scalar=224,     # Scaling specific to Gemma 2 (typically sqrt(hidden_size))
        sliding_window=4096,           # Sliding Window Attention window size
        logit_soft_capping=30.0,       # Final logit soft capping
        attn_logit_soft_capping=50.0,  # Attention scores soft capping
        
        # MoE Arguments
        num_experts_per_tok=2,
        num_local_experts=8,
        router_aux_loss_coef=0.001,
        output_router_logits=False,
        router_jitter_noise=0.0,       # Optional: Jitter for router stability
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # Grouped Query Attention (GQA) check
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Gemma 2 Specifics
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.logit_soft_capping = logit_soft_capping
        self.attn_logit_soft_capping = attn_logit_soft_capping

        # MoE Specifics
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.router_jitter_noise = router_jitter_noise

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
