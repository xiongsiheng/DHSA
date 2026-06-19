"""
KV cache compression utilities.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def undo_repeat_kv_avg(tensor: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    Reverses the effect of repeat_kv by averaging over the repeated key/value groups.

    Args:
        tensor (torch.Tensor): Tensor of shape (batch_size, num_attention_heads, seq_len, head_dim)
        num_key_value_groups (int): Number of times each key/value head was repeated.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim)
    """
    if num_key_value_groups == 1:
        # If no grouping, return the tensor as is
        return tensor

    batch_size, num_attention_heads, seq_len, head_dim = tensor.shape
    assert num_attention_heads % num_key_value_groups == 0, "num_attention_heads must be divisible by num_key_value_groups"

    num_key_value_heads = num_attention_heads // num_key_value_groups

    # Reshape to (batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim)
    reshaped = tensor.view(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim)

    # Average across the repeated groups (dim=2)
    averaged = reshaped.mean(dim=2)

    return averaged


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M 

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected] 
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent


class PyramidKVCluster():
    def __init__(self,
        num_hidden_layers = 32,
        window_size = 64,
        max_capacity_prompt = 256 + 64,
        kernel_size = 5,
        pooling = 'avgpool',
        beta = 20,
        num_layers = 80,
        layer_idx=None,
        merge = None
    ):
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.steps = -1
        self.beta = beta

        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, headwise_selection=True, use_global_score=False, is_sliding=False):
        bsz, num_heads, q_len, head_dim = query_states.shape

        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps

        print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            if use_global_score:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            else:
                attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if use_global_score:
                attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            else:
                attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            if headwise_selection:
                indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            else:
                attn_cache_unified = attn_cache.mean(dim=1)
                indices = attn_cache_unified.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                if is_sliding:
                    seq_len = attn_cache_unified.shape[-1]
                    indices = torch.arange(seq_len - self.max_capacity_prompt + self.window_size, seq_len, device=attn_weights.device).unsqueeze(0)
                indices = indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)


            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            key_states = undo_repeat_kv_avg(key_states, num_key_value_groups)
            value_states = undo_repeat_kv_avg(value_states, num_key_value_groups)
            return key_states, value_states
        else:
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            if use_global_score:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            else:
                attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if use_global_score:
                attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            else:
                attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            if headwise_selection:
                indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            else:
                attn_cache_unified = attn_cache.mean(dim=1)
                indices = attn_cache_unified.topk(max_capacity_prompt, dim=-1).indices
                if is_sliding:
                    seq_len = attn_cache_unified.shape[-1]
                    indices = torch.arange(seq_len - max_capacity_prompt, seq_len, device=attn_weights.device).unsqueeze(0)
                indices = indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            key_states = undo_repeat_kv_avg(key_states, num_key_value_groups)
            value_states = undo_repeat_kv_avg(value_states, num_key_value_groups)
            return key_states, value_states


class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.accumulated_scores = None

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.accumulated_scores = None

    def update_kv(self, key_states, query_states, value_states, num_key_value_groups, headwise_selection=True, use_global_score=False, is_sliding=False):
        _, num_heads, _, head_dim = query_states.shape

        # Original prefill logic
        if key_states.shape[-2] < self.max_capacity_prompt:
            return key_states, value_states

        # Repeat k/v heads if num_key_value_heads < num_attention_heads (GQA)
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        if use_global_score:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)

        mask = torch.full((self.window_size, self.window_size),
                        torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if use_global_score:
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim=-2)
        else:
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
        attn_cache = attn_weights_sum

        if headwise_selection:
            indices = attn_cache.topk(self.max_capacity_prompt, dim=-1).indices
            indices = indices.sort(dim=-1).values  # Sort to get sequential order
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        else:
            attn_cache_unified = attn_cache.mean(dim=1)

            # Get top-k indices based on unified scores
            # Shape: [bsz, max_capacity_prompt - window_size]
            indices_unified = attn_cache_unified.topk(self.max_capacity_prompt, dim=-1).indices
            indices_unified = indices_unified.sort(dim=-1).values  # Sort to maintain temporal order

            if is_sliding:
                indices_unified = torch.arange(attn_cache_unified.shape[-1] - self.max_capacity_prompt, attn_cache_unified.shape[-1], device=attn_weights.device).unsqueeze(0)

            # Expand indices for all heads and head dimensions
            # Shape: [bsz, num_heads, max_capacity_prompt - window_size, head_dim]
            indices = indices_unified.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)

        if self.merge is not None:
            key_states, value_states = merge_kv(key_states, value_states, indices, 
                                            self.window_size, self.merge)
            key_states = undo_repeat_kv_avg(key_states, num_key_value_groups)
            value_states = undo_repeat_kv_avg(value_states, num_key_value_groups)
            return key_states, value_states

        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)

        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]

        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        key_states = undo_repeat_kv_avg(key_states, num_key_value_groups)
        value_states = undo_repeat_kv_avg(value_states, num_key_value_groups)

        return key_states, value_states




class StreamingLLMKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states):
        print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :self.window_size, :]
            v_past_compress = value_states[:, :, :self.window_size, :]

            k_cur = key_states[:, :, -self.max_capacity_prompt:, :]
            v_cur = value_states[:, :, -self.max_capacity_prompt:, :]

            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = PyramidKVCluster(
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_H2O(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = H2OKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = StreamingLLMKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )