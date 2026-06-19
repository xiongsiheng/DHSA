import importlib
import inspect
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

_PATCH_ROOT = Path(__file__).resolve().parent
_PATCH_IMPORT_PATHS = [_PATCH_ROOT] + sorted((_PATCH_ROOT / "build").glob("lib.*"))
for _import_path in reversed(_PATCH_IMPORT_PATHS):
    _import_path_str = str(_import_path)
    while _import_path_str in sys.path:
        sys.path.remove(_import_path_str)
    sys.path.insert(0, _import_path_str)
_PATCH_BUILD_ROOTS = tuple(str(path.resolve()) for path in _PATCH_IMPORT_PATHS[1:])
for _module_name, _module in list(sys.modules.items()):
    if _module_name == "block_sparse_attn" or _module_name.startswith("block_sparse_attn."):
        _module_file = getattr(_module, "__file__", None)
        if _module_file is not None and str(Path(_module_file).resolve()).startswith(_PATCH_BUILD_ROOTS):
            del sys.modules[_module_name]

from block_sparse_attn import block_sparse_attn_func
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    repeat_kv as llama_repeat_kv,
)

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2FlashAttention2,
        apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb,
        repeat_kv as qwen2_repeat_kv,
    )
except Exception:
    Qwen2FlashAttention2 = None
    qwen2_apply_rotary_pos_emb = None
    qwen2_repeat_kv = None

_BLOCK_SPARSE_ATTN_PARAMS = set(inspect.signature(block_sparse_attn_func).parameters)
_BLOCK_SPARSE_ATTN_SUPPORTS_BLOCK_DIMS = {
    "m_block_dim",
    "n_block_dim",
}.issubset(_BLOCK_SPARSE_ATTN_PARAMS)
_WARNED_COARSENED_K_BLOCK_SIZE = False


def _patch_flash_attn_unpad_input_compat() -> None:
    try:
        import transformers.modeling_flash_attention_utils as flash_utils
    except Exception:
        return

    unpad_input = getattr(flash_utils, "unpad_input", None)
    if unpad_input is None or getattr(unpad_input, "_block_sparse_compat_wrapped", False):
        return

    def _compat_unpad_input(*args, **kwargs):
        result = unpad_input(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 5:
            return result[:4]
        return result

    _compat_unpad_input._block_sparse_compat_wrapped = True
    flash_utils.unpad_input = _compat_unpad_input


_patch_flash_attn_unpad_input_compat()


def _is_stale_k_block_size_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "n_block_dim must be a multiple of 128" in message
        or "n_block_dim must be 32, 64, or a multiple of 128" in message
    )


def _allow_coarsened_k_block_fallback() -> bool:
    return os.getenv("BLOCK_SPARSE_ALLOW_COARSENED_K_BLOCK_FALLBACK", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _coarsen_key_block_mask(
    block_mask: Optional[torch.Tensor],
    k_block_size: int,
    effective_k_block_size: int,
) -> Optional[torch.Tensor]:
    if block_mask is None or k_block_size == effective_k_block_size:
        return block_mask
    if effective_k_block_size % k_block_size != 0:
        raise ValueError("effective_k_block_size must be a multiple of k_block_size")
    group_size = effective_k_block_size // k_block_size
    ncol = block_mask.shape[-1]
    if ncol % group_size != 0:
        raise ValueError(
            "Cannot coarsen block_mask because the key block count is not "
            "divisible by the requested coarsening factor"
        )
    return block_mask.view(*block_mask.shape[:-1], ncol // group_size, group_size).any(dim=-1)


def _warn_coarsened_k_block_size(k_block_size: int, effective_k_block_size: int) -> None:
    global _WARNED_COARSENED_K_BLOCK_SIZE
    if _WARNED_COARSENED_K_BLOCK_SIZE:
        return
    print(
        "Warning: active block_sparse_attn_cuda extension rejects "
        f"k_block_size={k_block_size}; retrying with CUDA k_block_size="
        f"{effective_k_block_size} and a coarsened key block mask. Rebuild "
        "Block-Sparse-Attention to use native 32/64 key blocks."
    )
    _WARNED_COARSENED_K_BLOCK_SIZE = True


def _resolve_block_sparse_attn_backend():
    def _safe_getattr(obj, name, default=None):
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    namespaces = [getattr(block_sparse_attn_func, "__globals__", {})]
    defining_module = sys.modules.get(getattr(block_sparse_attn_func, "__module__", ""))
    if defining_module is not None:
        namespaces.append(vars(defining_module))
    interface_module = sys.modules.get("block_sparse_attn.block_sparse_attn_interface")
    if interface_module is None:
        try:
            interface_module = importlib.import_module("block_sparse_attn.block_sparse_attn_interface")
        except Exception:
            interface_module = None
    if interface_module is not None:
        namespaces.append(vars(interface_module))
    for module in list(sys.modules.values()):
        if (
            module is not None
            and _safe_getattr(module, "BlockSparseAttnFunc") is not None
            and _safe_getattr(module, "replace_ones_with_count") is not None
        ):
            namespaces.append(vars(module))

    for namespace in namespaces:
        apply_cls = namespace.get("BlockSparseAttnFunc")
        replace_heads = namespace.get("replace_ones_with_count")
        apply_fn = _safe_getattr(apply_cls, "apply")
        if (
            apply_cls is not None
            and callable(apply_fn)
            and callable(replace_heads)
        ):
            return apply_cls, replace_heads
    return None, None


_BLOCK_SPARSE_ATTN_APPLY, _BLOCK_SPARSE_ATTN_REPLACE_HEADS = _resolve_block_sparse_attn_backend()


def _call_block_sparse_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info,
    block_mask: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    p_dropout: float,
    deterministic: bool,
    softmax_scale,
    is_causal: bool,
    exact_streaming: bool,
    return_attn_probs: bool,
    q_block_size: int,
    k_block_size: int,
):
    kwargs = {
        "deterministic": deterministic,
        "softmax_scale": softmax_scale,
        "is_causal": is_causal,
        "exact_streaming": exact_streaming,
        "return_attn_probs": return_attn_probs,
    }
    if _BLOCK_SPARSE_ATTN_SUPPORTS_BLOCK_DIMS:
        kwargs.update({"m_block_dim": q_block_size, "n_block_dim": k_block_size})
        try:
            return block_sparse_attn_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                head_mask_type,
                streaming_info,
                block_mask,
                max_seqlen_q,
                max_seqlen_k,
                p_dropout,
                **kwargs,
            )
        except RuntimeError as exc:
            if (
                k_block_size >= 128
                or not _is_stale_k_block_size_error(exc)
                or not _allow_coarsened_k_block_fallback()
            ):
                raise
            effective_k_block_size = 128
            _warn_coarsened_k_block_size(k_block_size, effective_k_block_size)
            kwargs["n_block_dim"] = effective_k_block_size
            return block_sparse_attn_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                head_mask_type,
                streaming_info,
                _coarsen_key_block_mask(block_mask, k_block_size, effective_k_block_size),
                max_seqlen_q,
                max_seqlen_k,
                p_dropout,
                **kwargs,
            )

    if q_block_size != 128 or k_block_size != 128:
        if _BLOCK_SPARSE_ATTN_APPLY is None or _BLOCK_SPARSE_ATTN_REPLACE_HEADS is None:
            raise ValueError(
                "The imported block_sparse_attn_func does not expose custom "
                "m_block_dim/n_block_dim arguments or the underlying autograd "
                "function needed to call custom block sizes."
            )

        head_mask_type, blocksparse_head_num = _BLOCK_SPARSE_ATTN_REPLACE_HEADS(head_mask_type)
        if block_mask is not None:
            assert block_mask.shape[1] == blocksparse_head_num

        try:
            return _BLOCK_SPARSE_ATTN_APPLY.apply(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                q_block_size,
                k_block_size,
                head_mask_type,
                streaming_info,
                block_mask,
                max_seqlen_q,
                max_seqlen_k,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_attn_probs,
                -1,
                -1,
                deterministic,
                torch.is_grad_enabled(),
            )
        except RuntimeError as exc:
            if (
                k_block_size >= 128
                or not _is_stale_k_block_size_error(exc)
                or not _allow_coarsened_k_block_fallback()
            ):
                raise
            effective_k_block_size = 128
            _warn_coarsened_k_block_size(k_block_size, effective_k_block_size)
            return _BLOCK_SPARSE_ATTN_APPLY.apply(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                q_block_size,
                effective_k_block_size,
                head_mask_type,
                streaming_info,
                _coarsen_key_block_mask(block_mask, k_block_size, effective_k_block_size),
                max_seqlen_q,
                max_seqlen_k,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_attn_probs,
                -1,
                -1,
                deterministic,
                torch.is_grad_enabled(),
            )

    return block_sparse_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        block_mask,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        **kwargs,
    )


def _get_cache_seq_length(past_key_value, layer_idx: int) -> int:
    if past_key_value is None:
        return 0
    if hasattr(past_key_value, "get_seq_length"):
        try:
            return int(past_key_value.get_seq_length(layer_idx))
        except TypeError:
            return int(past_key_value.get_seq_length())
        except Exception:
            return 0
    key_cache = getattr(past_key_value, "key_cache", None)
    if key_cache is not None and len(key_cache) > layer_idx and isinstance(key_cache[layer_idx], torch.Tensor):
        return int(key_cache[layer_idx].shape[-2])
    return 0


def _slice_cache_position_embeddings(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    if tensor is None:
        return tensor
    if tensor.shape[-2] >= seq_len:
        return tensor[..., :seq_len, :]
    return tensor


def _round_to_multiple(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def _mask_prediction_chunk_elements() -> int:
    try:
        return max(1, int(os.getenv("DHSA_MASK_PREDICTION_CHUNK_ELEMENTS", "8388608")))
    except ValueError:
        return 8388608


def _mask_prediction_row_chunk_size(
    batch_size: int,
    num_heads: int,
    num_rows: int,
    num_k_blocks: int,
    selected_count: int = 0,
) -> int:
    elements_per_row = max(1, batch_size * num_heads * max(num_k_blocks, selected_count, 1))
    return max(1, min(num_rows, _mask_prediction_chunk_elements() // elements_per_row))


def _recent_key_blocks_for_rows(
    row_indices: torch.Tensor,
    num_k_blocks: int,
    q_block_size: int,
    k_block_size: int,
) -> torch.Tensor:
    return torch.clamp(
        ((row_indices + 1) * q_block_size - 1) // k_block_size,
        max=num_k_blocks - 1,
    )


def _q_start_key_blocks_for_rows(
    row_indices: torch.Tensor,
    num_k_blocks: int,
    q_block_size: int,
    k_block_size: int,
) -> torch.Tensor:
    return torch.clamp(
        (row_indices * q_block_size) // k_block_size,
        max=num_k_blocks - 1,
    )


def _or_scattered_topk_by_row_budget(
    mask_chunk: torch.Tensor,
    row_scores: torch.Tensor,
    remaining_by_row: torch.Tensor,
) -> None:
    if remaining_by_row.numel() == 0:
        return

    active_rows = remaining_by_row > 0
    if not bool(active_rows.any().item()):
        return

    active_scores = row_scores[:, :, active_rows, :]
    active_remaining = remaining_by_row[active_rows]

    max_remaining = int(active_remaining.max().item())
    if max_remaining <= 0:
        return

    max_remaining = min(max_remaining, active_scores.shape[-1])
    top_indices = active_scores.topk(k=max_remaining, dim=-1).indices
    keep_by_rank = (
        torch.arange(max_remaining, device=row_scores.device).view(1, 1, 1, max_remaining)
        < active_remaining.view(1, 1, -1, 1)
    )
    topk_mask = torch.zeros_like(active_scores, dtype=torch.bool)
    topk_mask.scatter_(-1, top_indices, keep_by_rank.expand_as(top_indices))
    active_mask = mask_chunk[:, :, active_rows, :]
    active_mask.logical_or_(topk_mask)
    mask_chunk[:, :, active_rows, :] = active_mask


def _generate_sparsity_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    topk_blocks: int,
    block_size: Optional[int] = None,
    q_block_size: Optional[int] = None,
    k_block_size: Optional[int] = None,
):
    """
    Obtain the top-k blocks mask for each query block.

    Args:
        query_states: shape (B, H, seq_q, D)
        key_states:   shape (B, H, seq_k, D)
        topk_blocks:   number of top-k key blocks for each query block
        block_size:    legacy shared block size along the sequence dimension
        q_block_size:  query block size along the sequence dimension
        k_block_size:  key block size along the sequence dimension

    Returns:
        mask: (B, H, n_q_blocks, n_k_blocks) bool
    """
    if block_size is not None:
        if q_block_size is None:
            q_block_size = block_size
        if k_block_size is None:
            k_block_size = block_size
    if q_block_size is None or k_block_size is None:
        raise ValueError("Provide either block_size or both q_block_size and k_block_size")

    b, h, seq_q, d = query_states.shape
    b2, h2, seq_k, d2 = key_states.shape
    assert (b, h, d) == (b2, h2, d2), "Q/K batch/head/hidden mismatch"

    assert seq_q % q_block_size == 0, "seq_q must be divisible by q_block_size"
    assert seq_k % k_block_size == 0, "seq_k must be divisible by k_block_size"

    num_q_blocks = seq_q // q_block_size
    num_k_blocks = seq_k // k_block_size

    q_blocks = query_states.view(b, h, num_q_blocks, q_block_size, d)
    k_blocks = key_states.view(b, h, num_k_blocks, k_block_size, d)

    # Use FP32 for block representations if you care about stability
    q_block_repr = q_blocks.float().mean(dim=3)  # (B, H, n_q_blocks, D)
    k_block_repr = k_blocks.float().mean(dim=3)  # (B, H, n_k_blocks, D)

    # Block-level scores: (B, H, n_q_blocks, n_k_blocks)
    scores = torch.matmul(
        q_block_repr,                       # (B, H, n_q_blocks, D)
        k_block_repr.transpose(-2, -1),     # (B, H, D, n_k_blocks)
    )

    # Safety if someone passes topk_blocks > num_k_blocks
    k = min(topk_blocks, num_k_blocks)

    # Top-k indices along the key-block dimension
    topk_indices = scores.topk(k=k, dim=-1).indices  # (B, H, n_q_blocks, k)

    # Build the boolean mask by scattering along the last dim
    mask = torch.zeros_like(scores, dtype=torch.bool)  # (B, H, n_q_blocks, n_k_blocks)
    mask.scatter_(-1, topk_indices, True)

    return mask


def _generate_sparsity_mask_with_A_shape(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    topk_blocks: int,
    block_size: Optional[int] = None,
    q_block_size: Optional[int] = None,
    k_block_size: Optional[int] = None,
):
    """
    Obtain a block mask with an A-shaped pattern.

    For each query block under causal attention, only key blocks up through the
    most recent causal key block are eligible. If that causal window fits within
    topk_blocks, select it densely. Otherwise always select the first and most
    recent eligible key blocks, then use blockwise top-k scores to fill the
    remaining budget from the middle eligible key blocks.

    Args:
        query_states: shape (B, H, seq_q, D)
        key_states:   shape (B, H, seq_k, D)
        topk_blocks:   target number of eligible key blocks per query block
        block_size:    legacy shared block size along the sequence dimension
        q_block_size:  query block size along the sequence dimension
        k_block_size:  key block size along the sequence dimension

    Returns:
        mask: (B, H, n_q_blocks, n_k_blocks) bool
    """
    if block_size is not None:
        if q_block_size is None:
            q_block_size = block_size
        if k_block_size is None:
            k_block_size = block_size
    if q_block_size is None or k_block_size is None:
        raise ValueError("Provide either block_size or both q_block_size and k_block_size")

    b, h, seq_q, d = query_states.shape
    b2, h2, seq_k, d2 = key_states.shape
    assert (b, h, d) == (b2, h2, d2), "Q/K batch/head/hidden mismatch"

    assert seq_q % q_block_size == 0, "seq_q must be divisible by q_block_size"
    assert seq_k % k_block_size == 0, "seq_k must be divisible by k_block_size"

    num_q_blocks = seq_q // q_block_size
    num_k_blocks = seq_k // k_block_size

    q_blocks = query_states.view(b, h, num_q_blocks, q_block_size, d)
    k_blocks = key_states.view(b, h, num_k_blocks, k_block_size, d)

    q_block_repr = q_blocks.float().mean(dim=3)  # (B, H, n_q_blocks, D)
    k_block_repr = k_blocks.float().mean(dim=3)  # (B, H, n_k_blocks, D)

    scores = torch.matmul(
        q_block_repr,                       # (B, H, n_q_blocks, D)
        k_block_repr.transpose(-2, -1),     # (B, H, D, n_k_blocks)
    )

    target_k = min(max(0, int(topk_blocks)), num_k_blocks)
    row_indices = torch.arange(num_q_blocks, device=query_states.device)
    key_block_positions = torch.arange(num_k_blocks, device=query_states.device)
    recent_key_blocks = _recent_key_blocks_for_rows(
        row_indices,
        num_k_blocks,
        q_block_size,
        k_block_size,
    )
    available_counts = recent_key_blocks + 1
    dense_rows = available_counts <= target_k

    causal_mask = key_block_positions.view(1, -1) <= recent_key_blocks.view(-1, 1)
    mask = (
        dense_rows.view(1, 1, num_q_blocks, 1)
        & causal_mask.view(1, 1, num_q_blocks, num_k_blocks)
    ).expand(b, h, -1, -1).clone()

    sparse_rows = ~dense_rows
    forced_mask = sparse_rows.view(num_q_blocks, 1) & (
        (key_block_positions.view(1, num_k_blocks) == 0)
        | (key_block_positions.view(1, num_k_blocks) == recent_key_blocks.view(num_q_blocks, 1))
    )
    mask.logical_or_(forced_mask.view(1, 1, num_q_blocks, num_k_blocks))

    middle_counts = torch.clamp(recent_key_blocks - 1, min=0)
    forced_counts = 1 + (recent_key_blocks != 0).long()
    remaining_by_row = torch.minimum(
        torch.clamp(target_k - forced_counts, min=0),
        middle_counts,
    )
    remaining_by_row = torch.where(
        sparse_rows,
        remaining_by_row,
        torch.zeros_like(remaining_by_row),
    )

    eligible_middle = (
        (key_block_positions.view(1, num_k_blocks) > 0)
        & (key_block_positions.view(1, num_k_blocks) < recent_key_blocks.view(num_q_blocks, 1))
    )
    row_scores = scores.masked_fill(
        ~eligible_middle.view(1, 1, num_q_blocks, num_k_blocks),
        float("-inf"),
    )
    _or_scattered_topk_by_row_budget(mask, row_scores, remaining_by_row)

    return mask


def _generate_sparsity_mask_with_vertical_slash(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    topk_blocks: int,
    block_size: Optional[int] = None,
    q_block_size: Optional[int] = None,
    k_block_size: Optional[int] = None,
    vertical_slash_ratio: float = 0.5,
):
    """
    Obtain an A-shaped block mask whose middle rows are predicted from vertical
    and slash token-line patterns selected by the dense scores of the last query
    block.

    The first dense prefix rows and the last query block are kept dense under the
    causal boundary. For the remaining middle rows, always preserve the first
    eligible key block and the most recent eligible key block, then fill the
    remaining block budget using key blocks touched by selected vertical token
    columns and slash distances.

    Args:
        query_states: shape (B, H, seq_q, D)
        key_states:   shape (B, H, seq_k, D)
        topk_blocks:   target number of eligible key blocks per query block
        block_size:    legacy shared block size along the sequence dimension
        q_block_size:  query block size along the sequence dimension
        k_block_size:  key block size along the sequence dimension
        vertical_slash_ratio: fraction of the token-line budget assigned to
            vertical columns. The rest is assigned to slash distances.

    Returns:
        mask: (B, H, n_q_blocks, n_k_blocks) bool
    """
    if block_size is not None:
        if q_block_size is None:
            q_block_size = block_size
        if k_block_size is None:
            k_block_size = block_size
    if q_block_size is None or k_block_size is None:
        raise ValueError("Provide either block_size or both q_block_size and k_block_size")
    if not 0.0 <= vertical_slash_ratio <= 1.0:
        raise ValueError("vertical_slash_ratio must be in [0, 1]")

    b, h, seq_q, d = query_states.shape
    b2, h2, seq_k, d2 = key_states.shape
    assert (b, h, d) == (b2, h2, d2), "Q/K batch/head/hidden mismatch"

    assert seq_q % q_block_size == 0, "seq_q must be divisible by q_block_size"
    assert seq_k % k_block_size == 0, "seq_k must be divisible by k_block_size"

    num_q_blocks = seq_q // q_block_size
    num_k_blocks = seq_k // k_block_size
    mask = torch.zeros(
        (b, h, num_q_blocks, num_k_blocks),
        device=query_states.device,
        dtype=torch.bool,
    )
    if num_q_blocks == 0 or num_k_blocks == 0:
        return mask

    target_k = min(max(0, int(topk_blocks)), num_k_blocks)
    token_budget = min(seq_k, max(0, target_k * k_block_size))
    vertical_budget = min(seq_k, int(token_budget * float(vertical_slash_ratio)))
    slash_budget = min(seq_k, max(0, token_budget - vertical_budget))
    dense_prefix_q_blocks = min(
        num_q_blocks,
        max(0, math.ceil(token_budget / q_block_size)) if token_budget > 0 else 0,
    )

    q_last = min(q_block_size, seq_q, seq_k)
    last_query_states = query_states[..., -q_last:, :].float()
    key_states_fp32 = key_states.float()
    scores = torch.matmul(
        last_query_states,
        key_states_fp32.transpose(-2, -1),
    ) * (1.0 / math.sqrt(d))

    key_positions = torch.arange(seq_k, device=query_states.device)
    last_query_positions = torch.arange(
        seq_k - q_last,
        seq_k,
        device=query_states.device,
    )
    scores = scores.masked_fill(
        key_positions.view(1, 1, 1, seq_k) > last_query_positions.view(1, 1, q_last, 1),
        float("-inf"),
    )
    last_query_attn = torch.softmax(scores, dim=-1)  # (B, H, q_last, seq_k)

    col_scores = last_query_attn.sum(dim=-2)  # (B, H, seq_k)
    vertical_block_scores = torch.zeros(
        (b, h, num_k_blocks),
        device=query_states.device,
        dtype=last_query_attn.dtype,
    )
    if vertical_budget > 0:
        vertical_indices = col_scores.topk(k=vertical_budget, dim=-1).indices
        selected_col_scores = col_scores.gather(-1, vertical_indices)
        vertical_blocks = torch.div(vertical_indices, k_block_size, rounding_mode="floor")
        vertical_block_scores.scatter_add_(-1, vertical_blocks, selected_col_scores)

    slash_indices = torch.empty((b, h, 0), dtype=torch.long, device=query_states.device)
    slash_selected_scores = torch.empty(
        (b, h, 0),
        dtype=last_query_attn.dtype,
        device=query_states.device,
    )
    if slash_budget > 0:
        distances = torch.arange(seq_k, device=query_states.device)
        slash_key_indices = last_query_positions.view(q_last, 1) - distances.view(1, seq_k)
        slash_valid = slash_key_indices >= 0
        slash_key_indices = slash_key_indices.clamp(0, seq_k - 1).long()
        slash_values = last_query_attn.gather(
            -1,
            slash_key_indices.view(1, 1, q_last, seq_k).expand(b, h, -1, -1),
        )
        slash_values = slash_values * slash_valid.view(1, 1, q_last, seq_k)
        slash_scores = slash_values.sum(dim=-2)  # (B, H, seq_k)
        slash_indices = slash_scores.topk(k=slash_budget, dim=-1).indices
        slash_selected_scores = slash_scores.gather(-1, slash_indices)

    row_indices = torch.arange(num_q_blocks, device=query_states.device)
    key_block_positions = torch.arange(num_k_blocks, device=query_states.device)
    recent_key_blocks = _recent_key_blocks_for_rows(
        row_indices,
        num_k_blocks,
        q_block_size,
        k_block_size,
    )
    available_counts = recent_key_blocks + 1
    dense_rows = (
        (available_counts <= target_k)
        | (row_indices < dense_prefix_q_blocks)
        | (row_indices == num_q_blocks - 1)
    )
    sparse_rows = ~dense_rows

    middle_counts = torch.clamp(recent_key_blocks - 1, min=0)
    forced_counts = 1 + (recent_key_blocks != 0).long()
    remaining_by_row = torch.minimum(
        torch.clamp(target_k - forced_counts, min=0),
        middle_counts,
    )
    remaining_by_row = torch.where(
        sparse_rows,
        remaining_by_row,
        torch.zeros_like(remaining_by_row),
    )

    max_slash_block_span = (q_block_size + k_block_size - 1) // k_block_size + 1
    row_chunk_size = _mask_prediction_row_chunk_size(
        b,
        h,
        num_q_blocks,
        num_k_blocks,
        slash_indices.shape[-1],
    )

    for row_start in range(0, num_q_blocks, row_chunk_size):
        row_end = min(num_q_blocks, row_start + row_chunk_size)
        rows = row_indices[row_start:row_end]
        recent_chunk = recent_key_blocks[row_start:row_end]
        dense_chunk = dense_rows[row_start:row_end]
        sparse_chunk = sparse_rows[row_start:row_end]
        remaining_chunk = remaining_by_row[row_start:row_end]
        chunk_len = row_end - row_start
        mask_chunk = mask[..., row_start:row_end, :]

        causal_chunk = key_block_positions.view(1, num_k_blocks) <= recent_chunk.view(chunk_len, 1)
        dense_mask = dense_chunk.view(chunk_len, 1) & causal_chunk
        mask_chunk.logical_or_(dense_mask.view(1, 1, chunk_len, num_k_blocks))

        forced_mask = sparse_chunk.view(chunk_len, 1) & (
            (key_block_positions.view(1, num_k_blocks) == 0)
            | (key_block_positions.view(1, num_k_blocks) == recent_chunk.view(chunk_len, 1))
        )
        mask_chunk.logical_or_(forced_mask.view(1, 1, chunk_len, num_k_blocks))

        if int(remaining_chunk.max().item()) <= 0:
            continue

        row_scores = vertical_block_scores.unsqueeze(-2).expand(-1, -1, chunk_len, -1)

        if slash_indices.numel() > 0:
            slash_block_scores = torch.zeros(
                (b, h, chunk_len, num_k_blocks),
                device=query_states.device,
                dtype=vertical_block_scores.dtype,
            )
            q_start_tokens = rows * q_block_size
            q_end_tokens = (rows + 1) * q_block_size - 1
            slash_offsets = slash_indices.unsqueeze(-2)
            min_key_token = q_start_tokens.view(1, 1, chunk_len, 1) - slash_offsets
            max_key_token = q_end_tokens.view(1, 1, chunk_len, 1) - slash_offsets
            valid_slash = max_key_token >= 0
            start_blocks = torch.div(
                min_key_token.clamp(0, seq_k - 1),
                k_block_size,
                rounding_mode="floor",
            )
            end_blocks = torch.div(
                max_key_token.clamp(0, seq_k - 1),
                k_block_size,
                rounding_mode="floor",
            )

            for block_offset in range(max_slash_block_span):
                touched_blocks = start_blocks + block_offset
                valid_blocks = (
                    valid_slash
                    & (touched_blocks <= end_blocks)
                    & (touched_blocks <= recent_chunk.view(1, 1, chunk_len, 1))
                )
                slash_block_scores.scatter_add_(
                    -1,
                    touched_blocks.clamp(0, num_k_blocks - 1),
                    slash_selected_scores.unsqueeze(-2) * valid_blocks,
                )
            row_scores = row_scores + slash_block_scores

        eligible_middle = (
            (key_block_positions.view(1, num_k_blocks) > 0)
            & (key_block_positions.view(1, num_k_blocks) < recent_chunk.view(chunk_len, 1))
        )
        row_scores = row_scores.masked_fill(
            ~eligible_middle.view(1, 1, chunk_len, num_k_blocks),
            float("-inf"),
        )
        _or_scattered_topk_by_row_budget(mask_chunk, row_scores, remaining_chunk)

    return mask


def _generate_sparsity_mask_with_vertical_slash_blockwise(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    topk_blocks: int,
    block_size: Optional[int] = None,
    q_block_size: Optional[int] = None,
    k_block_size: Optional[int] = None,
    vertical_slash_ratio: float = 0.5,
):
    """
    Memory-efficient A-shaped vertical/slash block mask.

    Structural invariants:
      1. Causal boundary is respected.
      2. First dense prefix rows are dense under causal boundary.
      3. Last query block is dense under causal boundary.
      4. For every sparse middle query block, always preserve:
           - first key block
           - most recent causal key block
      5. Only the remaining middle blocks are selected by rough blockwise
         vertical/slash prediction.

    Returns:
        mask: (B, H, n_q_blocks, n_k_blocks) bool
    """
    if block_size is not None:
        if q_block_size is None:
            q_block_size = block_size
        if k_block_size is None:
            k_block_size = block_size

    if q_block_size is None or k_block_size is None:
        raise ValueError("Provide either block_size or both q_block_size and k_block_size")

    if not 0.0 <= vertical_slash_ratio <= 1.0:
        raise ValueError("vertical_slash_ratio must be in [0, 1]")

    b, h, seq_q, d = query_states.shape
    b2, h2, seq_k, d2 = key_states.shape
    assert (b, h, d) == (b2, h2, d2), "Q/K batch/head/hidden mismatch"

    assert seq_q % q_block_size == 0, "seq_q must be divisible by q_block_size"
    assert seq_k % k_block_size == 0, "seq_k must be divisible by k_block_size"

    num_q_blocks = seq_q // q_block_size
    num_k_blocks = seq_k // k_block_size

    mask = torch.zeros(
        (b, h, num_q_blocks, num_k_blocks),
        device=query_states.device,
        dtype=torch.bool,
    )

    if num_q_blocks == 0 or num_k_blocks == 0:
        return mask

    target_k = min(max(0, int(topk_blocks)), num_k_blocks)

    # Same dense-prefix policy as the token-level vertical_slash implementation.
    token_budget = min(seq_k, max(0, target_k * k_block_size))
    dense_prefix_q_blocks = min(
        num_q_blocks,
        max(0, math.ceil(token_budget / q_block_size)) if token_budget > 0 else 0,
    )

    # ---------------------------------------------------------------------
    # Rough blockwise vertical/slash predictor.
    # This avoids materializing last_query_attn: (B, H, q_block_size, seq_k).
    # ---------------------------------------------------------------------

    q_blocks = query_states.view(b, h, num_q_blocks, q_block_size, d)
    k_blocks = key_states.view(b, h, num_k_blocks, k_block_size, d)

    q_block_repr = q_blocks.float().mean(dim=3)  # (B, H, n_q_blocks, D)
    k_block_repr = k_blocks.float().mean(dim=3)  # (B, H, n_k_blocks, D)

    # Use the last query block to infer global vertical/slash patterns.
    last_q_repr = q_block_repr[..., -1:, :]  # (B, H, 1, D)

    last_block_scores = torch.matmul(
        last_q_repr,
        k_block_repr.transpose(-2, -1),
    ).squeeze(-2) * (1.0 / math.sqrt(d))  # (B, H, n_k_blocks)

    # The last query block is causal up to the final key block in normal prefill.
    # Still mask defensively for q/k length mismatch.
    last_recent_key_block = min(
        num_k_blocks - 1,
        (seq_q - 1) // k_block_size,
    )
    key_block_positions = torch.arange(num_k_blocks, device=query_states.device)
    last_block_scores = last_block_scores.masked_fill(
        key_block_positions.view(1, 1, num_k_blocks) > last_recent_key_block,
        float("-inf"),
    )

    vertical_budget = min(target_k, int(target_k * float(vertical_slash_ratio)))
    slash_budget = min(target_k, max(0, target_k - vertical_budget))

    vertical_block_scores = torch.zeros_like(last_block_scores)

    if vertical_budget > 0:
        vertical_indices = last_block_scores.topk(k=vertical_budget, dim=-1).indices
        vertical_values = last_block_scores.gather(-1, vertical_indices)
        vertical_block_scores.scatter_add_(-1, vertical_indices, vertical_values)

    # Block-level slash distances:
    # distance = last_query_block_idx - key_block_idx
    slash_distances = torch.empty(
        (b, h, 0),
        dtype=torch.long,
        device=query_states.device,
    )
    slash_values = torch.empty(
        (b, h, 0),
        dtype=last_block_scores.dtype,
        device=query_states.device,
    )

    if slash_budget > 0:
        block_distances = last_recent_key_block - key_block_positions
        valid_slash_keys = (
            (block_distances >= 0)
            & (key_block_positions <= last_recent_key_block)
        )

        slash_scores = last_block_scores.masked_fill(
            ~valid_slash_keys.view(1, 1, num_k_blocks),
            float("-inf"),
        )

        slash_key_indices = slash_scores.topk(k=slash_budget, dim=-1).indices
        slash_values = slash_scores.gather(-1, slash_key_indices)

        # Store distances in key-block coordinates. q_block_size and k_block_size
        # may differ, so query-block indices cannot be reused as key-block ids.
        slash_distances = last_recent_key_block - slash_key_indices

    # ---------------------------------------------------------------------
    # A-shape construction.
    # The rough predictor is used only for middle blocks.
    # ---------------------------------------------------------------------

    row_indices = torch.arange(num_q_blocks, device=query_states.device)
    recent_key_blocks = _recent_key_blocks_for_rows(
        row_indices,
        num_k_blocks,
        q_block_size,
        k_block_size,
    )
    q_start_key_blocks = _q_start_key_blocks_for_rows(
        row_indices,
        num_k_blocks,
        q_block_size,
        k_block_size,
    )
    available_counts = recent_key_blocks + 1
    dense_rows = (
        (available_counts <= target_k)
        | (row_indices < dense_prefix_q_blocks)
        | (row_indices == num_q_blocks - 1)
    )
    sparse_rows = ~dense_rows

    middle_counts = torch.clamp(recent_key_blocks - 1, min=0)
    local_band_counts = recent_key_blocks - q_start_key_blocks + 1
    forced_counts = local_band_counts + (q_start_key_blocks > 0).long()
    remaining_by_row = torch.minimum(
        torch.clamp(target_k - forced_counts, min=0),
        middle_counts,
    )
    remaining_by_row = torch.where(
        sparse_rows,
        remaining_by_row,
        torch.zeros_like(remaining_by_row),
    )

    row_chunk_size = _mask_prediction_row_chunk_size(
        b,
        h,
        num_q_blocks,
        num_k_blocks,
        slash_distances.shape[-1],
    )

    for row_start in range(0, num_q_blocks, row_chunk_size):
        row_end = min(num_q_blocks, row_start + row_chunk_size)
        recent_chunk = recent_key_blocks[row_start:row_end]
        q_start_chunk = q_start_key_blocks[row_start:row_end]
        dense_chunk = dense_rows[row_start:row_end]
        sparse_chunk = sparse_rows[row_start:row_end]
        remaining_chunk = remaining_by_row[row_start:row_end]
        chunk_len = row_end - row_start
        mask_chunk = mask[..., row_start:row_end, :]

        # Dense under the causal boundary for short prefixes, dense-prefix rows,
        # and the final query block.
        causal_chunk = key_block_positions.view(1, num_k_blocks) <= recent_chunk.view(chunk_len, 1)
        dense_mask = dense_chunk.view(chunk_len, 1) & causal_chunk
        mask_chunk.logical_or_(dense_mask.view(1, 1, chunk_len, num_k_blocks))

        # Sparse rows keep the first key block and the whole local diagonal band.
        forced_mask = sparse_chunk.view(chunk_len, 1) & (
            (key_block_positions.view(1, num_k_blocks) == 0)
            | (
                (key_block_positions.view(1, num_k_blocks) >= q_start_chunk.view(chunk_len, 1))
                & (key_block_positions.view(1, num_k_blocks) <= recent_chunk.view(chunk_len, 1))
            )
        )
        mask_chunk.logical_or_(forced_mask.view(1, 1, chunk_len, num_k_blocks))

        if int(remaining_chunk.max().item()) <= 0:
            continue

        row_scores = vertical_block_scores.unsqueeze(-2).expand(-1, -1, chunk_len, -1)

        if slash_distances.numel() > 0:
            slash_block_scores = torch.zeros(
                (b, h, chunk_len, num_k_blocks),
                device=query_states.device,
                dtype=vertical_block_scores.dtype,
            )
            min_touched_blocks = q_start_chunk.view(1, 1, chunk_len, 1) - slash_distances.unsqueeze(-2)
            max_touched_blocks = recent_chunk.view(1, 1, chunk_len, 1) - slash_distances.unsqueeze(-2)
            valid_slash = max_touched_blocks >= 0
            max_slash_block_span = int((recent_chunk - q_start_chunk + 1).max().item())

            for block_offset in range(max_slash_block_span):
                touched_blocks = min_touched_blocks + block_offset
                valid_touched = (
                    valid_slash
                    & (touched_blocks <= max_touched_blocks)
                    & (touched_blocks > 0)
                    & (touched_blocks < recent_chunk.view(1, 1, chunk_len, 1))
                )
                slash_block_scores.scatter_add_(
                    -1,
                    touched_blocks.clamp(0, num_k_blocks - 1),
                    slash_values.unsqueeze(-2) * valid_touched,
                )

            row_scores = row_scores + slash_block_scores

        # Only middle causal blocks can be selected by predictor.
        eligible_middle = (
            (key_block_positions.view(1, num_k_blocks) > 0)
            & (key_block_positions.view(1, num_k_blocks) < recent_chunk.view(chunk_len, 1))
        )
        row_scores = row_scores.masked_fill(
            ~eligible_middle.view(1, 1, chunk_len, num_k_blocks),
            float("-inf"),
        )
        _or_scattered_topk_by_row_budget(mask_chunk, row_scores, remaining_chunk)

    return mask


def _validate_sparse_block_sizes(q_block_size: int, k_block_size: int) -> None:
    if q_block_size <= 0 or k_block_size <= 0:
        raise ValueError("q_block_size and k_block_size must be positive")
    if q_block_size % 128 != 0:
        raise ValueError("q_block_size must be a multiple of 128 for this CUDA kernel")
    if k_block_size not in (32, 64, 128):
        raise ValueError("k_block_size must be one of 32, 64, or 128")


def _chunked_sequence_module(module: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    bsz, seq_len, hidden_size = x.shape
    if seq_len <= chunk_size:
        return module(x)

    out = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = x[:, start:end, :].contiguous()
        y = module(chunk.view(-1, hidden_size)).view(bsz, end - start, -1)
        if out is None:
            out = x.new_empty((bsz, seq_len, y.shape[-1]))
        out[:, start:end, :].copy_(y)
        del chunk, y
    return out


def _chunked_llama_mlp(mlp: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    bsz, seq_len, hidden_size = x.shape
    if seq_len <= chunk_size:
        return mlp(x)

    out = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = x[:, start:end, :].contiguous()
        flat = chunk.view(-1, hidden_size)

        up = mlp.up_proj(flat)
        gate = mlp.gate_proj(flat)
        activated = mlp.act_fn(gate)
        y = mlp.down_proj(activated * up).view(bsz, end - start, -1)
        if out is None:
            out = x.new_empty((bsz, seq_len, y.shape[-1]))
        out[:, start:end, :].copy_(y)
        del chunk, flat, up, gate, activated, y
    return out


def _llama_decoder_layer_chunked_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    residual = hidden_states

    norm_chunk_size = int(getattr(self, "_block_sparse_norm_chunk_size", 2048))
    mlp_chunk_size = int(getattr(self, "_block_sparse_mlp_chunk_size", 1024))

    hidden_states = _chunked_sequence_module(self.input_layernorm, hidden_states, norm_chunk_size)

    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    if isinstance(attn_outputs, tuple):
        attn_output = attn_outputs[0]
        self_attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
        present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None
    else:
        attn_output = attn_outputs
        self_attn_weights = None
        present_key_value = None

    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = _chunked_sequence_module(self.post_attention_layernorm, hidden_states, norm_chunk_size)
    hidden_states = _chunked_llama_mlp(self.mlp, hidden_states, mlp_chunk_size)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    signature = inspect.signature(fn)
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    if not has_var_kwargs:
        return filtered

    internal_compat_keys = {
        "past_key_value",
        "past_key_values",
        "output_attentions",
        "use_cache",
        "cache_position",
        "position_embeddings",
    }
    for key, value in kwargs.items():
        if key in filtered:
            continue
        if key in internal_compat_keys and value is None:
            continue
        if key in {"past_key_value", "output_attentions", "use_cache", "cache_position"}:
            continue
        filtered[key] = value
    return filtered


def _call_original_forward(self: nn.Module, hidden_states: torch.Tensor, **kwargs):
    return self._original_forward(hidden_states, **_filter_kwargs_for_callable(self._original_forward, kwargs))


def _patchable_attention_classes():
    classes = [LlamaFlashAttention2]
    for cls in (Qwen2FlashAttention2,):
        if cls is not None:
            classes.append(cls)
    return tuple(classes)


def _attention_family(module: nn.Module) -> str:
    cls_name = module.__class__.__name__.lower()
    model_type = str(getattr(getattr(module, "config", None), "model_type", "")).lower()
    if "qwen2" in cls_name or "qwen2" in model_type:
        return "qwen2"
    return "llama"


def _family_apply_rotary(family: str):
    if family == "qwen2" and qwen2_apply_rotary_pos_emb is not None:
        return qwen2_apply_rotary_pos_emb
    return llama_apply_rotary_pos_emb


def _family_repeat_kv(family: str):
    if family == "qwen2" and qwen2_repeat_kv is not None:
        return qwen2_repeat_kv
    return llama_repeat_kv


def _compute_rotary_embeddings(
    self: nn.Module,
    family: str,
    value_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    if position_embeddings is not None:
        return position_embeddings
    rotary_emb = getattr(self, "rotary_emb", None)
    if rotary_emb is None:
        raise ValueError(
            f"{self.__class__.__name__} needs external position_embeddings for block-sparse prefill"
        )
    return rotary_emb(value_states, position_ids)


def _maybe_apply_qk_norm(self: nn.Module, query_states: torch.Tensor, key_states: torch.Tensor):
    q_norm = getattr(self, "q_norm", None)
    k_norm = getattr(self, "k_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    if k_norm is not None:
        key_states = k_norm(key_states)
    return query_states, key_states


def _cache_aliases(past_key_value=None, past_key_values=None):
    return past_key_values if past_key_values is not None else past_key_value


def _update_past_key_value(
    past_key_value,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: dict,
):
    try:
        return past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)
    except TypeError:
        return past_key_value.update(key_states, value_states, layer_idx)


def _block_sparse_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    past_key_values=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    measure_kernel: bool = True,
    **kwargs,
):
    """
    Shared drop-in replacement for LLaMA/Qwen2 attention modules.

    - Prefill (past_key_value is None): use Block Sparse Attention.
    - Decode (past_key_value is not None) or output_attentions=True:
      fall back to the original FlashAttention2 forward.
    """
    # print("Using Block Sparse Attention forward...")
    # Fallback cases ---------------------------------------------------------
    if output_attentions:
        print("Falling back to original FA2 due to output_attentions=True")
        return _call_original_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    family = getattr(self, "_block_sparse_family", _attention_family(self))
    cache = _cache_aliases(past_key_value=past_key_value, past_key_values=past_key_values)
    only_prefill = getattr(self, "_block_sparse_only_prefill", True)
    cache_seq_len = _get_cache_seq_length(cache, self.layer_idx)
    if only_prefill and cache is not None and cache_seq_len > 0:
        # print("Falling back to original FA2 due to decode step with past_key_value")
        # decode step: stick to FA2 (no blocksparse)
        return _call_original_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    # ------------------------- Prefill path (Block Sparse) ------------------
    bsz, q_len, hidden_size = hidden_states.size()

    q_block_size = int(getattr(self, "_block_sparse_q_block_size", getattr(self, "_block_sparse_block_size", 128)))
    k_block_size = int(getattr(self, "_block_sparse_k_block_size", getattr(self, "_block_sparse_block_size", 128)))
    _validate_sparse_block_sizes(q_block_size, k_block_size)
    sparsity = float(getattr(self, "_block_sparse_sparsity", 0.5))
    apply_rotary = _family_apply_rotary(family)
    repeat_kv = _family_repeat_kv(family)

    original_q_len = q_len
    original_hidden_states = hidden_states
    original_cache_position = cache_position
    alignment = math.lcm(q_block_size, k_block_size)
    padded_q_len = _round_to_multiple(q_len, alignment)
    pad_len = padded_q_len - q_len
    if pad_len and position_embeddings is not None and getattr(self, "rotary_emb", None) is None:
        return _call_original_forward(
            self,
            original_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    if pad_len:
        pad_states = hidden_states.new_zeros((bsz, pad_len, hidden_size))
        hidden_states = torch.cat((hidden_states, pad_states), dim=1)
        q_len = padded_q_len

        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        else:
            pad_positions = position_ids[:, -1:] + torch.arange(
                1,
                pad_len + 1,
                device=position_ids.device,
                dtype=position_ids.dtype,
            ).unsqueeze(0)
            position_ids = torch.cat((position_ids, pad_positions), dim=-1)

        if cache_position is not None:
            cache_pad = cache_position[-1:] + torch.arange(
                1,
                pad_len + 1,
                device=cache_position.device,
                dtype=cache_position.dtype,
            )
            cache_position = torch.cat((cache_position, cache_pad), dim=0)

        position_embeddings = None

    # Standard LLaMA projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # (B, L, H, D) layouts used by LlamaFlashAttention2
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # RoPE
    query_states, key_states = _maybe_apply_qk_norm(self, query_states, key_states)
    cos, sin = _compute_rotary_embeddings(self, family, value_states, position_ids, position_embeddings)
    query_states, key_states = apply_rotary(query_states, key_states, cos, sin)
    if cache is not None:
        cache_kwargs = {
            "sin": _slice_cache_position_embeddings(sin, original_q_len),
            "cos": _slice_cache_position_embeddings(cos, original_q_len),
            "cache_position": original_cache_position,
        }
        if getattr(self, "sliding_window", None) is not None:
            cache_kwargs["sliding_window"] = self.sliding_window
        cache_key_states = key_states[..., :original_q_len, :].contiguous()
        cache_value_states = value_states[..., :original_q_len, :].contiguous()
        _update_past_key_value(cache, cache_key_states, cache_value_states, self.layer_idx, cache_kwargs)

    # Expand KV heads -> full num_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Layout for varlen kernel: (total_tokens, n_heads, head_dim)
    # Start from (B, H, L, D) -> (B, L, H, D) -> flatten
    q = query_states.transpose(1, 2).contiguous()   # (B, L, H, D)
    k = key_states.transpose(1, 2).contiguous()     # (B, L, H, D)
    v = value_states.transpose(1, 2).contiguous()   # (B, L, H, D)

    _, Lq, H, D = q.shape
    _, Lk, Hk, _ = k.shape

    if Lq != Lk or H != Hk:
        # Unexpected; safer to fall back than crash
        print("Falling back to original FA2 due to shape mismatch between query and key states")
        return _call_original_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    device = q.device

    q_unpad = q.reshape(bsz * Lq, H, D)
    k_unpad = k.reshape(bsz * Lk, H, D)
    v_unpad = v.reshape(bsz * Lk, H, D)

    cu_seqlens = torch.arange(
        0, (bsz + 1) * Lq, step=Lq, dtype=torch.int32, device=device
    )  # (B+1,)

    # ---------------------- Block mask construction ------------------------
    # This is the CRITICAL shape: (batch_size, num_blocksparse_heads, nrow, ncol)
    # base_blockmask = base_mask_1.unsqueeze(0).repeat(bsz, H, 1, 1)

    if measure_kernel:
        torch.cuda.synchronize()

        # memory already in use when this layer starts
        baseline_bytes = torch.cuda.memory_allocated()

        # reset peak tracker so it starts from *current* usage
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()


    block_mask = _generate_sparsity_mask(
        query_states=query_states,
        key_states=key_states,
        topk_blocks=max(1, int((1.0 - sparsity) * (Lk // k_block_size))),
        q_block_size=q_block_size,
        k_block_size=k_block_size,
    )  # (B, H, nrow, ncol)

    head_mask_type = torch.ones(H, dtype=torch.int32, device=device)  # all heads blocksparse

    p_dropout = self.attention_dropout if self.training else 0.0
    out_unpad = _call_block_sparse_attn_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens,
        cu_seqlens,
        head_mask_type,
        None,                 # streaming_info
        block_mask,           # (B, H, nrow, ncol)
        Lq,                   # max_seqlen_q_
        Lk,                   # max_seqlen_k_
        p_dropout,
        deterministic=False,
        softmax_scale=None,
        is_causal=self.is_causal,
        exact_streaming=False,
        return_attn_probs=False,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
    )  # (B*L, H, D)

    # Back to (B, L, H, D) -> (B, L, hidden_size)
    attn_output = out_unpad.view(bsz, Lq, H, D)
    attn_output = attn_output.reshape(bsz, Lq, H * D).contiguous()
    if pad_len:
        attn_output = attn_output[:, :original_q_len, :].contiguous()
    if getattr(self, "_block_sparse_chunk_calculation", False):
        proj_chunk_size = int(getattr(self, "_block_sparse_o_proj_chunk_size", 1024))
        attn_output = _chunked_sequence_module(self.o_proj, attn_output, proj_chunk_size)
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, cache


_llama_block_sparse_forward = _block_sparse_forward


def _configure_block_sparse_attention_module(
    module: nn.Module,
    family: str,
    sparsity: float,
    block_size: int,
    q_block_size: int,
    k_block_size: int,
    only_prefill: bool,
    chunk_calculation: bool,
    o_proj_chunk_size: int,
) -> bool:
    if hasattr(module, "_original_forward"):
        return False

    module._original_forward = module.forward
    module._block_sparse_family = family
    module._block_sparse_sparsity = float(sparsity)
    module._block_sparse_block_size = int(block_size)
    module._block_sparse_q_block_size = int(q_block_size)
    module._block_sparse_k_block_size = int(k_block_size)
    module._block_sparse_only_prefill = bool(only_prefill)
    module._block_sparse_chunk_calculation = bool(chunk_calculation)
    module._block_sparse_o_proj_chunk_size = int(o_proj_chunk_size)
    module.forward = _block_sparse_forward.__get__(module, module.__class__)
    return True


def patch_model_with_block_sparse(
    model: nn.Module,
    sparsity: float = 0.75,
    q_block_size: Optional[int] = None,
    k_block_size: Optional[int] = None,
    only_prefill: bool = True,
    model_families: Optional[Tuple[str, ...]] = None,
    chunk_calculation: Optional[bool] = None,
    norm_chunk_size: int = 2048,
    mlp_chunk_size: int = 1024,
    o_proj_chunk_size: int = 1024,
    default_block_size: int = 128,
):
    """
    Monkey-patch supported global-attention modules in `model`.

    LLaMA and Qwen2 attention layers are replaced by block-sparse attention.
    """
    if q_block_size is None:
        q_block_size = default_block_size
    if k_block_size is None:
        k_block_size = default_block_size
    _validate_sparse_block_sizes(q_block_size, k_block_size)

    if model_families is None:
        enabled_families = {"llama", "qwen2"}
    else:
        enabled_families = {family.lower() for family in model_families}

    attention_classes = _patchable_attention_classes()
    for module in model.modules():
        if isinstance(module, attention_classes):
            family = _attention_family(module)
            if family not in enabled_families:
                continue
            _configure_block_sparse_attention_module(
                module=module,
                family=family,
                sparsity=sparsity,
                block_size=default_block_size,
                q_block_size=q_block_size,
                k_block_size=k_block_size,
                only_prefill=only_prefill,
                chunk_calculation=chunk_calculation,
                o_proj_chunk_size=o_proj_chunk_size,
            )

        elif chunk_calculation and isinstance(module, LlamaDecoderLayer):
            if hasattr(module, "_block_sparse_original_layer_forward"):
                continue

            module._block_sparse_original_layer_forward = module.forward
            module._block_sparse_norm_chunk_size = int(norm_chunk_size)
            module._block_sparse_mlp_chunk_size = int(mlp_chunk_size)
            module.forward = _llama_decoder_layer_chunked_forward.__get__(module, module.__class__)


def patch_llama_with_block_sparse(
    model: nn.Module,
    sparsity: float = 0.75,
    q_block_size: Optional[int] = None,
    k_block_size: Optional[int] = None,
    only_prefill: bool = True,
    chunk_calculation: Optional[bool] = None,
    norm_chunk_size: int = 2048,
    mlp_chunk_size: int = 1024,
    o_proj_chunk_size: int = 1024,
):
    """Backward-compatible LLaMA-only wrapper."""
    return patch_model_with_block_sparse(
        model=model,
        sparsity=sparsity,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        only_prefill=only_prefill,
        model_families=("llama",),
        chunk_calculation=chunk_calculation,
        norm_chunk_size=norm_chunk_size,
        mlp_chunk_size=mlp_chunk_size,
        o_proj_chunk_size=o_proj_chunk_size,
    )
