"""
Rewrite the methods in transformers.cache_utils.HybridCache to support dynamic allocation (start with a minimal cache size and expand it on-the-fly as the sequence grows).
Originally the method requires pre-allocating a fixed-size tensor based on a max_sequence_length during initialization.

version: transformers == 4.52.3
"""

import torch
import transformers
from typing import Union, Any



def __init__(
    self,
    config: transformers.configuration_utils.PretrainedConfig,
    max_batch_size: int,
    max_cache_len: int | None = None,
    device: Union[torch.device, str, None] = None,
    dtype: torch.dtype = torch.float32,
    layer_device_map: dict[int, Union[str, torch.device, int]] | None = None,
) -> None:
    # This is the original init method
    # which is deprecated in transformers > 4.52.3
    # transformers.cache_utils.Cache.__init__(self)
    if not hasattr(config, "sliding_window") or config.sliding_window is None:
        raise ValueError(
            "Setting `cache_implementation` to 'hybrid' requires the model config supporting "
            "sliding window attention, please check if there is a `sliding_window` field in the model "
            "config and it's not set to None."
        )

    self.max_cache_len = max_cache_len if max_cache_len is not None else config.max_position_embeddings

    # Sliding layers can't be larger than the overall max cache len
    self.sliding_window_len = min(config.sliding_window, self.max_cache_len)
    self.max_batch_size = max_batch_size

    # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
    self.head_dim = (
        config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
    )

    self._dtype = dtype
    self.num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    layer_switch = config.sliding_window_pattern if hasattr(config, "sliding_window_pattern") else 2  # 2 is for BC
    self.is_sliding = [bool((i + 1) % layer_switch) for i in range(config.num_hidden_layers)]
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []

    self._total_seq_lens: list[int] = [0] * config.num_hidden_layers
    self.indices = None


def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_sliding_layer = self.is_sliding[layer_idx]

    # Update the number of seen tokens for the layer
    self._total_seq_lens[layer_idx] += key_states.shape[-2]

    # Update the cache
    if key_states is not None:
        if len(self.key_cache) <= layer_idx:
            if is_sliding_layer:
                key_states = key_states[:, :, -cache_kwargs['sliding_window']:, :]
                value_states = value_states[:, :, -cache_kwargs['sliding_window']:, :]
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (
            not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
        ):
            if is_sliding_layer:
                key_states = key_states[:, :, -cache_kwargs['sliding_window']:, :]
                value_states = value_states[:, :, -cache_kwargs['sliding_window']:, :]
            # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            if is_sliding_layer:
                key_states = key_states[:, :, -cache_kwargs['sliding_window']:, :]
                value_states = value_states[:, :, -cache_kwargs['sliding_window']:, :]
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

    return self.key_cache[layer_idx], self.value_cache[layer_idx]



def get_seq_length(self, layer_idx: int | None = None) -> int:
    """
    Return the total (global) sequence length currently stored in the cache.

    Args:
        layer_idx (int | None):  • int  → length for that specific layer
                                 • None → minimum length across *all* layers
                                           (keeps the previous default-semantics)
    """
    if layer_idx is None:
        # Use `min` to stay consistent with earlier behaviour: the generation
        # loop must stop at the shortest layer to avoid reading un-cached steps.
        return min(self._total_seq_lens) if any(self._total_seq_lens) else 0
    if not (0 <= layer_idx < len(self._total_seq_lens)):
        raise IndexError(f"layer_idx must be in [0, {len(self._total_seq_lens)-1}]")
    return self._total_seq_lens[layer_idx]



def reset(self):
    """Resets the cache values while preserving the objects"""
    for layer_idx in range(len(self.key_cache)):
        # In-place ops prevent breaking the static address
        self.key_cache[layer_idx] = torch.tensor([])
        self.value_cache[layer_idx] = torch.tensor([])
        self._total_seq_lens[layer_idx] = 0