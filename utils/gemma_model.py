"""
Rewrite the Gemma2 attention forward methods to support sparse attention.
The methods include:
  - gemma_sdpa_forward_PyramidKV
  - gemma_sdpa_attn_forward_H2O
  - gemma_sdpa_attn_forward_StreamingLLM
  - gemma_sdpa_attn_forward_sliding_window
  - gemma_sdpa_attn_forward_topk
  - gemma_sdpa_attn_forward_blocksparse
  - gemma_sdpa_attn_forward_dhsa
"""

import math
import torch
import transformers

from transformers.models import gemma2

import copy

from .kvcache_compression import (
    init_pyramidkv, init_H2O, init_StreamingLLM
)
from .helper import (
    topk_keys_from_attention, block_sparse_attn_sdpa,
    label_boundaries, predict_boundaries,
    dhsa_sdpa
)

from transformers.cache_utils import Cache, HybridCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from functools import partial
from transformers.utils.deprecation import deprecate_kwarg



def gemma_sdpa_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: transformers.cache_utils.Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """
    Gemma2 SDPA attention forward with PyramidKV compression.

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Attention mask
        past_key_value: Cache object
        cache_position: Cache position
        **kwargs: Additional keyword arguments (e.g., output_attentions)
    """
    # Initialize PyramidKV if not already done
    if not hasattr(self, "kv_cluster"):
        # Assuming init_pyramidkv is available and initializes self.kv_cluster
        init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)

    if not hasattr(self, "act_kv_seq_len"):
        self.act_kv_seq_len = 0

    # Get input dimensions
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project to query, key, value states
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    is_sliding = False

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = gemma2.modeling_gemma2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle cache update
    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window,
        }
        self.act_kv_seq_len += key_states.shape[-2]

        # Check if we need to compress
        if self.act_kv_seq_len == key_states.shape[-2]:
            # Apply PyramidKV compression
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states,
                attention_mask, self.num_key_value_groups,
                headwise_selection=False,
                use_global_score=False,
                is_sliding=is_sliding
            )

            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Repeat k/v heads if num_key_value_heads < num_attention_heads (GQA)
    if hasattr(self, "num_key_value_groups"):
        key_states = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
        value_states = gemma2.modeling_gemma2.repeat_kv(value_states, self.num_key_value_groups)

    # Prepare causal mask
    causal_mask = attention_mask
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            # Handle 4D mask for key-value cache length
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        elif causal_mask.ndim == 2:
            # Handle 2D mask for key-value cache length
            causal_mask = causal_mask[:, :key_states.shape[-2]]

    # Ensure contiguous tensors for CUDA
    if query_states.device.type == "cuda":
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Determine if we should use causal mask
    is_causal = query_states.shape[2] > 1 and causal_mask is None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
    )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def gemma_sdpa_attn_forward_H2O(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: transformers.cache_utils.Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """
    Gemma2 SDPA attention forward with H2O compression.

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Attention mask
        past_key_value: Cache object
        cache_position: Cache position
        **kwargs: Additional keyword arguments (e.g., output_attentions)
    """
    # Initialize H2O if not already done
    if not hasattr(self, "kv_cluster"):
        init_H2O(self)

    if not hasattr(self, "act_kv_seq_len"):
        self.act_kv_seq_len = 0

    # Get input dimensions
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project to query, key, value states
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    is_sliding = False

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = gemma2.modeling_gemma2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle cache update
    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window
        }
        self.act_kv_seq_len += key_states.shape[-2]

        if self.act_kv_seq_len == key_states.shape[-2]:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states,
                value_states, self.num_key_value_groups,
                headwise_selection=False,
                use_global_score=False,
                is_sliding=is_sliding
            )
            # We use all the key/value states and update the cache with the compressed key/value states for the future steps
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Prepare causal mask
    causal_mask = attention_mask
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            # Handle 4D mask for key-value cache length
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        elif causal_mask.ndim == 2:
            # Handle 2D mask for key-value cache length
            causal_mask = causal_mask[:, :key_states.shape[-2]]

    # Ensure contiguous tensors for CUDA
    if query_states.device.type == "cuda":
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Repeat k/v heads if num_key_value_heads < num_attention_heads (GQA)
    if hasattr(self, "num_key_value_groups"):
        key_states = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
        value_states = gemma2.modeling_gemma2.repeat_kv(value_states, self.num_key_value_groups)

    # Determine if we should use causal mask
    is_causal = query_states.shape[2] > 1 and causal_mask is None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
    )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def gemma_sdpa_attn_forward_StreamingLLM(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: transformers.cache_utils.Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """
    Gemma2 SDPA attention forward with streaming LLM compression.

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Attention mask
        past_key_value: Cache object
        cache_position: Cache position
        **kwargs: Additional keyword arguments (e.g., output_attentions)
    """
    if not hasattr(self, "kv_cluster"):
        init_StreamingLLM(self)

    if not hasattr(self, "act_kv_seq_len"):
        self.act_kv_seq_len = 0

    # Get input dimensions
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project to query, key, value states
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = gemma2.modeling_gemma2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle cache update
    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window,
        }
        self.act_kv_seq_len += key_states.shape[-2]
        if self.act_kv_seq_len == key_states.shape[-2]:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states
            )

            # We use all the key/value states and update the cache with the compressed key/value states for the future steps
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Prepare causal mask
    causal_mask = attention_mask
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        elif causal_mask.ndim == 2:
            causal_mask = causal_mask[:, : key_states.shape[-2]]

    # Ensure contiguous tensors for CUDA
    if query_states.device.type == "cuda":
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Repeat k/v heads if num_key_value_heads < num_attention_heads (GQA)
    if hasattr(self, "num_key_value_groups"):
        key_states = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
        value_states = gemma2.modeling_gemma2.repeat_kv(value_states, self.num_key_value_groups)

    # Determine if we should use causal mask
    is_causal = query_states.shape[2] > 1 and causal_mask is None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
    )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def gemma_sdpa_attn_forward_sliding_window(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: transformers.cache_utils.Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """
    Gemma2 SDPA attention forward with sliding window attention.

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Attention mask
        past_key_value: Cache object
        cache_position: Cache position
        **kwargs: Additional keyword arguments (e.g., output_attentions)
    """
    if not hasattr(self, "act_kv_seq_len"):
        self.act_kv_seq_len = 0

    # Get input dimensions
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project to query, key, value states
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = gemma2.modeling_gemma2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle cache update
    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window,
        }
        self.act_kv_seq_len += key_states.shape[-2]
        if self.act_kv_seq_len == key_states.shape[-2]:
            key_states_compress = key_states[:, :, -self.config.max_capacity_prompt:, :]
            value_states_compress = value_states[:, :, -self.config.max_capacity_prompt:, :]

            # We use all the key/value states and update the cache with the compressed key/value states for the future steps
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Prepare causal mask
    causal_mask = attention_mask
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        elif causal_mask.ndim == 2:
            causal_mask = causal_mask[:, : key_states.shape[-2]]

    # Ensure contiguous tensors for CUDA
    if query_states.device.type == "cuda":
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Repeat k/v heads if num_key_value_heads < num_attention_heads (GQA)
    if hasattr(self, "num_key_value_groups"):
        key_states = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
        value_states = gemma2.modeling_gemma2.repeat_kv(value_states, self.num_key_value_groups)

    # Determine if we should use causal mask
    is_causal = query_states.shape[2] > 1 and causal_mask is None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
    )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def gemma_sdpa_attn_forward_topk(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: transformers.cache_utils.Cache | None = None,
    cache_position: torch.LongTensor| None = None,
    **kwargs,
):
    """
    Gemma2 SDPA attention forward with top-k compression.

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Attention mask
        past_key_value: Cache object
        cache_position: Cache position
        **kwargs: Additional keyword arguments (e.g., output_attentions)
    """
    budget_prefill = self.config.budget_prefill

    if not hasattr(self, "act_kv_seq_len"):
        self.act_kv_seq_len = 0

    # Get input dimensions
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project to query, key, value states
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    is_sliding = False

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = gemma2.modeling_gemma2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle cache update
    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window,
        }
        self.act_kv_seq_len += key_states.shape[-2]
        if self.act_kv_seq_len == key_states.shape[-2]:
            past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            if budget_prefill > 0:
                key_states_repeated = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(query_states.shape[-1])
                # Ensure attn_weights is causal
                if attention_mask is not None:
                    if attention_mask.ndim == 4:
                        attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    if attention_mask.ndim == 2:
                        attention_mask = attention_mask[:, : key_states.shape[-2]]
                    attn_weights.masked_fill_(attention_mask < -1e9, torch.finfo(attn_weights.dtype).min)
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                attn_weights = topk_keys_from_attention(attn_weights, budget_prefill, is_sliding=is_sliding)

                # Ensure attention_mask is causal after top-k block selection
                if attention_mask is not None:
                    attn_weights.masked_fill_(attention_mask < -1e9, torch.finfo(attn_weights.dtype).min)
                attention_mask = attn_weights
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Prepare causal mask
    causal_mask = attention_mask
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        elif causal_mask.ndim == 2:
            causal_mask = causal_mask[:, : key_states.shape[-2]]

    # Ensure contiguous tensors for CUDA
    if query_states.device.type == "cuda":
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Repeat k/v heads if num_key_value_heads < num_attention_heads (GQA)
    if hasattr(self, "num_key_value_groups"):
        key_states = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
        value_states = gemma2.modeling_gemma2.repeat_kv(value_states, self.num_key_value_groups)

    # Determine if we should use causal mask
    is_causal = query_states.shape[2] > 1 and causal_mask is None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
    )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def gemma_sdpa_attn_forward_block_sparse(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: transformers.cache_utils.Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """
    Gemma2 SDPA attention forward with block sparse attention.

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Attention mask
        past_key_value: Cache object
        cache_position: Cache position
        **kwargs: Additional keyword arguments (e.g., output_attentions)
    """
    budget_prefill = self.config.budget_prefill
    block_size = self.config.block_size
    topk_blocks = budget_prefill // block_size

    if not hasattr(self, "act_kv_seq_len"):
        self.act_kv_seq_len = 0

    # Get input dimensions
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project to query, key, value states
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = gemma2.modeling_gemma2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle cache update
    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window,
        }
        self.act_kv_seq_len += key_states.shape[-2]
        if self.act_kv_seq_len == key_states.shape[-2]:
            past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Prepare causal mask
    causal_mask = attention_mask
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        elif causal_mask.ndim == 2:
            causal_mask = causal_mask[:, : key_states.shape[-2]]

    # Ensure contiguous tensors for CUDA
    if query_states.device.type == "cuda":
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Repeat k/v heads if num_key_value_heads < num_attention_heads (GQA)
    if hasattr(self, "num_key_value_groups"):
        key_states = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
        value_states = gemma2.modeling_gemma2.repeat_kv(value_states, self.num_key_value_groups)

    # Determine if we should use causal mask
    is_causal = query_states.shape[2] > 1 and causal_mask is None

    if query_states.shape[2] > 1:
        # prefilling stage
        attn_output = block_sparse_attn_sdpa(
            query_states,
            key_states,
            value_states,
            block_size,
            topk_blocks,
            causal_mask=causal_mask,
            scaling=self.scaling,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            sliding_window_size=self.config.sliding_window if not bool(self.layer_idx % 2) else None,
        )
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=is_causal,
        )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def _prepare_qkv(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    Gemma2 SDPA attention forward with dynamic hierarchical sparse attention (DHSA).

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings

    Returns:
        query_states: Query states
        key_states: Key states
        value_states: Value states
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = gemma2.modeling_gemma2.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    return query_states, key_states, value_states


def _update_kv_cache(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: transformers.cache_utils.Cache,
    cache_position: torch.LongTensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the KV cache and returns the full key and value states.

    Args:
        key_states: Key states
        value_states: Value states
        past_key_value: Cache object
        cache_position: Cache position
        position_embeddings: Tuple of (cos, sin) for rotary embeddings

    Returns:
        key_states: Full key states
        value_states: Full value states
    """
    cos, sin = position_embeddings
    cache_kwargs = {
        "sin": sin,
        "cos": cos,
        "cache_position": cache_position,
        "sliding_window": self.sliding_window
    }

    # This state tracks the total sequence length in the cache
    self.act_kv_seq_len += key_states.shape[-2]

    # The first time a cache is updated, its length is the input sequence length
    if self.act_kv_seq_len == key_states.shape[-2]:
        past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        return key_states, value_states
    else:
        return past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )


def _compute_boundaries_for_training(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Computes and stores attention boundaries for DHSA during training.

    Args:
        query_states: Query states
        key_states: Key states
        attention_mask: Attention mask

    Returns:
        boundaries: Attention boundaries
    """
    # During training, we label boundaries directly from the data.
    # The predictor is not used.
    num_chunks = key_states.shape[-2] // self.config.block_size
    num_chunks = num_chunks * self.config.chunk_beta
    instruction_tokens = self.instruction_tokens if self.instruction_tokens is not None else 64
    boundaries, ratios = label_boundaries(
        query_states,
        key_states,
        attention_mask,
        num_chunks=num_chunks,
        layer_idx=self.layer_idx,
        boundary_window_size=self.config.boundary_window_size,
        theta=self.config.boundary_ratio_theta,
        use_nms=self.config.use_nms,
        nms_window_size=self.config.nms_window_size,
        instruction_tokens=instruction_tokens
    )
    # Store on self for access by the training loss function
    self.boundaries = boundaries
    self.ratios = ratios
    return


def _compute_boundaries_for_inference(
    self,
    key_states: torch.Tensor
) -> torch.Tensor:
    """
    Computes and stores attention boundaries for DHSA during inference.

    Args:
        key_states: Key states

    Returns:
        boundaries: Attention boundaries
    """
    num_chunks = key_states.shape[-2] // self.config.block_size
    num_chunks = num_chunks * self.config.chunk_beta
    instruction_tokens = self.instruction_tokens if self.instruction_tokens is not None else 64
    boundaries, ratios = predict_boundaries(
        self.boundary_predictor,
        key_states,
        num_chunks,
        boundary_window_size=self.config.boundary_window_size,
        theta=self.config.boundary_ratio_theta,
        use_nms=self.config.use_nms,
        nms_window_size=self.config.nms_window_size,
        instruction_tokens=instruction_tokens
    )
    self.boundaries = boundaries
    self.ratios = ratios
    return


def gemma_sdpa_attn_forward_dhsa(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: transformers.cache_utils.Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    boundaries: torch.Tensor | None = None,
    topk_indices_global: torch.Tensor | None = None,
    topk_indices_local: torch.Tensor | None = None,
    **kwargs,
):
    """
    Gemma2 SDPA attention forward with dhsa:
        dynamic hierarchical sparse attention

    Args:
        hidden_states: Input hidden states
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Attention mask
        past_key_value: Cache object
        cache_position: Cache position
        **kwargs: Additional keyword arguments (e.g., output_attentions)
    """
    # 1. Prepare Q, K, V states
    query_states, key_states, value_states = _prepare_qkv(self, hidden_states, position_embeddings)

    # 2. Update KV Cache if applicable
    if past_key_value is not None:
        key_states, value_states = _update_kv_cache(
            self,
            key_states,
            value_states,
            past_key_value,
            cache_position,
            position_embeddings
        )

    # 3. Repeat K/V for Grouped-Query Attention (GQA)
    key_states_raw = copy.copy(key_states)
    key_states = gemma2.modeling_gemma2.repeat_kv(key_states, self.num_key_value_groups)
    value_states = gemma2.modeling_gemma2.repeat_kv(value_states, self.num_key_value_groups)

    # 4. Determine attention mode (Prefill vs. Decode) and compute attention
    is_prefill = query_states.shape[-2] > 1
    is_causal = is_prefill and attention_mask is None

    # Prepare causal mask
    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    if is_prefill:
        # --- Prefilling Stage (DHSA) ---
        # Compute boundaries if not already computed.
        self.boundaries = boundaries
        if self.boundaries is None:
            if self.boundary_predictor is None:
                # Automatically label boundaries during training.
                _compute_boundaries_for_training(self, query_states, key_states_raw, attention_mask)
            else:
                # Run boundary predictor for inference
                _compute_boundaries_for_inference(self, key_states_raw)
            boundaries = self.boundaries

        if self.boundaries is None:
            raise RuntimeError("Boundaries must be computed during the prefilling stage for DHSA.")

        # Compute attention output with boundaries
        attn_output, topk_indices = dhsa_sdpa(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            block_size=self.config.block_size,
            topk_tokens=self.config.budget_prefill,
            causal_mask=causal_mask,
            scaling=self.scaling,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            sliding_window_size=self.config.sliding_window if not (self.layer_idx % 2) else None,
            boundaries=boundaries,
            instruction_tokens=self.instruction_tokens if self.instruction_tokens is not None else 64,
            loop_times=self.config.loop_times,
            pooling=self.config.chunk_representation_pooling,
            topk_indices=topk_indices_global if not bool(self.layer_idx % 2) else topk_indices_local
        )   # (optional) We pass the topk_indices_global and topk_indices_local
            # to every other layer to avoid recomputation.
        if not bool(self.layer_idx % 2):
            topk_indices_global = topk_indices
        else:
            topk_indices_local = topk_indices
    else:
        # --- Decoding Stage (Standard SDPA) ---
        # TODO: Add support for decoding stage.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=is_causal,
        )

    # 5. Reshape and project output
    input_shape = hidden_states.shape[:-1]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    # Reset boundary and block mask if not shared
    attn_weights = None
    if not self.config.share_boundaries:
        boundaries = None
    if not self.config.share_sparsity_masks:
        topk_indices_global = None
        topk_indices_local = None

    return attn_output, attn_weights, boundaries, topk_indices_global, topk_indices_local


def prepare_inputs_for_generation_gemma(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    logits_to_keep=None,
    **kwargs
):
    """
    Prepare inputs for generation - adapted for Gemma2 with sparse attention.
    This function should be used to patch the model's
    prepare_inputs_for_generation method.

    Args:
        input_ids: Input token IDs
        past_key_values: Past key/value states
        attention_mask: Attention mask
        inputs_embeds: Input embeddings
        cache_position: Cache position
        position_ids: Position IDs
        use_cache: Whether to use cache
        logits_to_keep: Logits to keep
        **kwargs: Additional keyword arguments
    """
    # Reset kv_seq_len for all layers if no past_key_values
    if past_key_values is None or (hasattr(past_key_values, "key_cache") and 
                                   past_key_values.get_seq_length() == 0):
        for layer in self.model.layers:
            if hasattr(layer.self_attn, "kv_seq_len"):
                layer.self_attn.kv_seq_len = 0

    # Handle cache position
    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, transformers.cache_utils.Cache):
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
        else:
            # For custom cache implementations
            past_length = getattr(self.model.layers[0].self_attn, "kv_seq_len", 0)

        # Only keep unprocessed tokens
        if cache_position is None:
            input_ids = input_ids[:, past_length:]
        elif past_length > 0:
            input_ids = input_ids[:, cache_position]

    # Handle position_ids if not provided
    if position_ids is None and attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # Prepare model inputs
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update({
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
    })

    # Handle logits_to_keep
    model_inputs["logits_to_keep"] = logits_to_keep
    if logits_to_keep is None:
        _ = model_inputs.pop("logits_to_keep", None)
    return model_inputs


# Re-implement gemma2.modeling_gemma2.Gemma2Model.forward to enable
# passing boundaries, topk_indices_global and topk_indices_local across layers.
def gemma2model_forward_dhsa(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: HybridCache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **flash_attn_kwargs,
) -> BaseModelOutputWithPast:
    """
    Gemma2 model forward pass with DHSA. This function is a modified version
    of the original gemma2.modeling_gemma2.Gemma2Model.forward function.
    It enables passing boundaries, topk_indices_global and topk_indices_local
    across layers.

    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        position_ids: Position IDs
        past_key_values: Past key/value states
        inputs_embeds: Input embeddings
        use_cache: Whether to use cache
        output_attentions: Whether to output attentions
        output_hidden_states: Whether to output hidden states
        cache_position: Cache position
        **flash_attn_kwargs: Flash attention kwargs

    Returns:
        Output of the model
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None and not self.training:
        batch_size, seq_len, _ = inputs_embeds.shape
        # NOTE: ideally, `HybridCache` should be initialized outside the model with `layer_device_map`
        # The cache object should be created once before we start the generation process 
        # (i.e., before we call the forward method for the first time).
        # This single cache object is then passed into the forward method
        # and updated on each subsequent step.
        # This is much more efficient as it avoids repeated object creation.
        past_key_values = HybridCache(
            self.config,
            max_batch_size=batch_size,
            max_cache_len=seq_len,
            dtype=inputs_embeds.dtype,
            device=self.device,
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds,
        cache_position,
        past_key_values, output_attentions
    )

    # embed positions
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
    hidden_states = hidden_states * normalizer

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    boundaries = None
    topk_indices_global = None
    topk_indices_local = None
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                partial(decoder_layer.__call__, **flash_attn_kwargs),
                hidden_states,
                position_embeddings,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                boundaries,
                topk_indices_global,
                topk_indices_local,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                boundaries=boundaries,
                topk_indices_global=topk_indices_global,
                topk_indices_local=topk_indices_local,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        boundaries = layer_outputs[-3]
        topk_indices_global = layer_outputs[-2]
        topk_indices_local = layer_outputs[-1]

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# Re-implement gemma2.modeling_gemma2.Gemma2DecoderLayer.forward to enable
# passing boundaries, topk_indices_global and topk_indices_local across layers.
# The `last_cache_position` argument is deprecated in transformers >= 4.53.0.
@deprecate_kwarg("last_cache_position", version="4.53.0")
def gemma2decoderlayer_forward_dhsa(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Cache | None = None,
    output_attentions: bool | None = False,
    use_cache: bool | None = False,
    cache_position: torch.LongTensor | None = None,
    boundaries: torch.Tensor | None = None,
    topk_indices_global: torch.Tensor | None = None,
    topk_indices_local: torch.Tensor | None = None,
    **kwargs,
):
    if self.is_sliding and attention_mask is not None and cache_position is not None:  # efficient SDPA and no padding
        # In prefill, we may be larger than sliding window
        effective_seq_len = max(cache_position.shape[0], self.sliding_window)
        # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
        # thus we must slice from the right (at most `effective_seq_len` elements)
        if self.config._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask[:, -effective_seq_len:]
        # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
        # from the left, with an offset if we are beyond the sliding window
        else:
            min_dtype = torch.finfo(attention_mask.dtype).min
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
            )
            attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
            # In case we are beyond the sliding window, we need to correctly offset the mask slicing
            offset = cache_position[-1] - effective_seq_len + 1
            # Should only be used when beyond the sliding window (i.e. offset > 0)
            offset = torch.clamp(offset, min=0)
            # equivalent to: `attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indexes = torch.arange(
                min(effective_seq_len, attention_mask.shape[-1]), device=attention_mask.device
            )
            mask_indexes += offset
            attention_mask = attention_mask[:, :, :, mask_indexes]

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, boundaries, topk_indices_global, topk_indices_local = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        boundaries=boundaries,
        topk_indices_global=topk_indices_global,
        topk_indices_local=topk_indices_local,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    outputs += (boundaries,)
    outputs += (topk_indices_global, topk_indices_local)

    return outputs