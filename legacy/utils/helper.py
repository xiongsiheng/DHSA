"""
Helper functions for sparse attention mechanisms.
"""
import torch
import math



def topk_keys_from_attention_tokenwise(attn, topk):
    """
    Create a mask that keeps only top-k keys for each query position.

    Args:
        attn: Attention scores tensor of shape (B, H, Q, K)
              B = batch size, H = num heads, Q = query length, K = key length
        topk: Number of top keys to keep for each query

    Returns:
        mask: Tensor of shape (B, 1, Q, K) where preserved positions are 0
              and masked positions are -inf
    """
    b, _, q, k = attn.shape

    # Average attention scores over heads only
    # Shape: (B, Q, K)
    attn_avg = attn.mean(dim=1)

    # Create a mask tensor initialized with mask_value
    # Shape: (B, Q, K)
    mask = torch.full((b, q, k),
                      torch.finfo(attn.dtype).min,
                      dtype=attn.dtype,
                      device=attn.device)

    # First, always include diagonal positions
    for q_i in range(q):
        if q_i < k:  # Only if diagonal position exists
            mask[:, q_i, q_i] = 0

    # Now select top-(k-1) from remaining positions
    if topk > 1:
        # Create a copy of attention scores and
        # set diagonal to -inf to exclude from topk
        attn_for_topk = attn_avg.clone()
        for q_i in range(min(q, k)):
            attn_for_topk[:, q_i, q_i] = float('-inf')

        # Get the indices of top-(k-1) keys for each query
        # Shape: (B, Q, k-1)
        _, topk_indices = torch.topk(attn_for_topk,
                                     k=min(topk-1, k-1),
                                     dim=-1,
                                     largest=True)

        # Set top-k positions to 0
        batch_indices = torch.arange(b, device=attn.device).view(-1, 1, 1)
        query_indices = torch.arange(q, device=attn.device).view(1, -1, 1)
        mask[batch_indices, query_indices, topk_indices] = 0

    # Expand mask to match the original shape (B, 1, Q, K)
    mask = mask.unsqueeze(1)
    return mask


def merge_attention_regions(
    attn: torch.Tensor,
    bound: torch.Tensor,
    is_causal: bool
):
    """
    Merge rectangular regions of an attention matrix and return their averages.

    Args
    ----
    attn  : (B, Q, K)   - attention matrix
    bound : (B, L)      - 0-based inclusive left / exclusive right borders
                          (e.g. [0, 2, 6, 10] → 3 chunks: [0:2] [2:6] [6:10])
    is_causal: Whether the attention is causal

    Returns
    -------
    (B, L-1, L-1) tensor with the mean value inside every block
    """
    b, q, k = attn.shape
    l = bound.size(1)
    device = attn.device
    dtype = attn.dtype

    # ------------------------------------------------------------------
    # 1. 2-D prefix sum  (pad with a zero row/col so that indices can be
    #    used directly without if-statements)
    # ------------------------------------------------------------------
    pref = attn.new_zeros((b, q + 1, k + 1))
    pref[:, 1:, 1:] = attn
    pref = pref.to(torch.float64)
    pref = pref.cumsum(dim=1).cumsum(dim=2)          # (B, Q+1, K+1)

    # ------------------------------------------------------------------
    # 2. Compute the four corners for every (q-chunk, k-chunk) pair
    # ------------------------------------------------------------------
    q0 = bound[:, :-1].unsqueeze(2)            # (B, L-1, 1)
    q1 = bound[:, 1: ].unsqueeze(2)            # (B, L-1, 1)
    k0 = bound[:, :-1].unsqueeze(1)            # (B, 1, L-1)
    k1 = bound[:, 1: ].unsqueeze(1)            # (B, 1, L-1)

    b_idx = torch.arange(b, device=device).view(b, 1, 1)  # broadcast helper

    # gather the four corner sums
    s_q1_k1 = pref[b_idx, q1, k1]          #   Σ(0..q1-1 , 0..k1-1)
    s_q0_k1 = pref[b_idx, q0, k1]          #   Σ(0..q0-1 , 0..k1-1)
    s_q1_k0 = pref[b_idx, q1, k0]          #   Σ(0..q1-1 , 0..k0-1)
    s_q0_k0 = pref[b_idx, q0, k0]          #   Σ(0..q0-1 , 0..k0-1)

    region_sum = s_q1_k1 - s_q0_k1 - s_q1_k0 + s_q0_k0   # (B, L-1, L-1)

    # ------------------------------------------------------------------
    # 3. divide by the number of elements in each block to obtain the mean
    # ------------------------------------------------------------------
    q_len = (q1 - q0)                       # (B, L-1, 1)
    k_len = (k1 - k0)                       # (B, 1, L-1)
    region_cnt = q_len * k_len              # broadcast → (B, L-1, L-1)

    if is_causal:
        # Calculate segment lengths
        segment_lengths = bound[:, 1:] - bound[:, :-1]  # Shape: (B, L-1)

        # Apply lower triangle formula: n*(n+1)/2
        lower_triangle_counts = segment_lengths * (
            segment_lengths + 1) // 2  # Shape: (B, L-1)

        # Create a boolean mask for diagonal elements
        diag_mask = torch.eye(
            l-1,
            dtype=torch.bool
        ).unsqueeze(0).expand(b, -1, -1)

        # Set diagonal elements to half
        region_cnt[diag_mask] = lower_triangle_counts

    merged = region_sum / region_cnt.clamp(min=1)
    return merged.to(dtype).to(device)  # (B, L-1, L-1)


def unmerge_attention_regions(
    merged: torch.Tensor,
    bound: torch.Tensor,
    is_causal: bool
):
    """
    Expand a block-averaged attention matrix back to the original resolution.

    Args
    ----
    merged : (B, R, R) tensor
                R = L-1 , i.e. the number of regions per axis.
    bound  : (B, L) tensor with monotonically increasing integers
                (“left-inclusive / right-exclusive“ borders).
                The last element of every row is the sequence length, so
                Q = K = bound[b, -1].
    is_causal: Whether the attention is causal

    Returns
    -------
    attn   : (B, Q, K) tensor in which every element that belonged to a block
                has been filled with that block’s average value from `merged`.
    """
    b, r, _ = merged.shape
    _, l = bound.shape
    assert r == l - 1, "merged and bound have incompatible shapes"

    # We build the result batch-by-batch because the chunk sizes (boundaries)
    # can differ between samples.  The inner work is done by a pair of
    # repeat_interleave calls, which is completely vectorised on the GPU/CPU
    # and only needs tiny 1-D index tensors.
    expanded = []
    for b_i in range(b):
        # lengths of the chunks on the q- and k-axes
        seg_len = (bound[b_i, 1:] - bound[b_i, :-1]).to(torch.long)  # (R,)
        mb = merged[b_i]  # (R,R)

        # 1. repeat columns: [R,R] → [R, K]
        mb_cols = torch.repeat_interleave(mb, seg_len, dim=1)          # (R, K)

        # 2. repeat rows:    [R, K] → [Q, K]
        mb_full = torch.repeat_interleave(mb_cols, seg_len, dim=0)     # (Q, K)

        if is_causal:
            mb_full = mb_full.tril_()

        expanded.append(mb_full)

    return torch.stack(expanded, dim=0)     # (B, Q, K)


def topk_keys_from_attention_blockwise(
    attn: torch.Tensor,
    topk: int,
    block_size: int,
    chunk_boundary: list[int] | None = None
):
    """Create a mask that keeps only top-k blocks for each query block.

    Args:
        attn: Attention scores tensor of shape (B, H, Q, K)
              B = batch size, H = num heads, Q = query length, K = key length
        k: Number of top blocks to keep for each query block
        block_size: Size of the blocks to consider
        chunk_boundary: Optional list of chunk boundaries for merging regions

    Returns:
        mask: Tensor of shape (B, 1, Q, K) where preserved positions are 0
              and masked positions are -inf
    """
    topk //= block_size
    b, h, q, k = attn.shape

    # Pad Q and K dimensions to be divisible by block_size
    q_padded = ((q + block_size - 1) // block_size) * block_size
    k_padded = ((k + block_size - 1) // block_size) * block_size

    # Pad attention tensor if needed 
    # (the input attention ranges from [0, 1], so pad with zeros)
    padding_value = 0
    if q_padded > q or k_padded > k:
        attn_padded = torch.full((b, h, q_padded, k_padded),
                                 padding_value,
                                 dtype=attn.dtype,
                                 device=attn.device
                                 )
        attn_padded[:, :, :q, :k] = attn
    else:
        attn_padded = attn

    # Average over heads
    # Shape: (B, Q_padded, K_padded)
    attn_avg = attn_padded.mean(dim=1)

    # Reshape into blocks and sum within each block
    # Shape: (B, Q_blocks, block_size, K_blocks, block_size)
    q_blocks = q_padded // block_size
    k_blocks = k_padded // block_size
    attn_blocks = attn_avg.view(b, q_blocks, block_size, k_blocks, block_size)

    # Sum attention scores within each block
    # Shape: (B, Q_blocks, K_blocks)
    block_scores = attn_blocks.sum(dim=(2, 4))

    if chunk_boundary is not None:
        merged_block_scores = merge_attention_regions(block_scores,
                                                      chunk_boundary,
                                                      is_causal=True
                                                      )
        block_scores = unmerge_attention_regions(merged_block_scores,
                                                 chunk_boundary,
                                                 is_causal=True
                                                 )

    # Create block mask initialized with 1 (will be masked)
    # Shape: (B, Q_blocks, K_blocks)
    block_mask = torch.full((b, q_blocks, k_blocks),
                            1.0,
                            dtype=attn.dtype,
                            device=attn.device
                            )

    # First, always include diagonal blocks
    for q_block in range(min(q_blocks, k_blocks)):
        block_mask[:, q_block, q_block] = 0

    # Now select top-(k-1) blocks from remaining positions
    if topk > 1:
        # Create a copy of block scores
        # and set diagonal to -inf to exclude from topk
        block_scores_for_topk = block_scores.clone()
        for q_block in range(min(q_blocks, k_blocks)):
            block_scores_for_topk[:, q_block, q_block] = float('-inf')

        # Get top-(k-1) blocks for each query block
        # Shape: (B, Q_blocks, k-1)
        _, topk_block_indices = torch.topk(block_scores_for_topk,
                                           k=min(topk-1, k_blocks-1),
                                           dim=-1,
                                           largest=True
                                           )

        # Set selected blocks to 0
        batch_indices = torch.arange(b, device=attn.device).view(-1, 1, 1)
        q_block_indices = torch.arange(q_blocks,
                                       device=attn.device
                                       ).view(1, -1, 1)
        block_mask[batch_indices, q_block_indices, topk_block_indices] = 0

    # Expand block mask back to token level
    # First expand each block:
    # (B, Q_blocks, K_blocks) -> (B, Q_padded, K_padded)
    mask_expanded = block_mask.unsqueeze(2).unsqueeze(4)
    mask_expanded = mask_expanded.expand(-1, -1, block_size, -1, block_size)
    mask_expanded = mask_expanded.reshape(b, q_padded, k_padded)

    # Convert 0/1 mask to mask_value/0 format
    mask = torch.where(
        mask_expanded == 0,
        torch.tensor(0.0, dtype=attn.dtype, device=attn.device),
        torch.tensor(torch.finfo(attn.dtype).min,
                     dtype=attn.dtype, device=attn.device)
    )

    # Trim back to original size
    mask = mask[:, :q, :k]

    # Expand mask to match the original shape (B, 1, Q, K)
    mask = mask.unsqueeze(1)
    return mask


def topk_keys_from_attention(
    attn: torch.Tensor,
    topk: int,
    is_sliding: bool = False,
    block_size: int = 1
):
    """
    Create a causal mask that keeps only top-k keys for each query position.

    Args:
        attn: Attention scores tensor of shape (B, H, Q, K)
              B = batch size, H = num heads, Q = query length, K = key length
        topk: Number of top keys to keep for each query
        is_sliding: If True, use sliding window attention
                   (keep k most recent keys)
                   If False, use top-k attention based on attention scores

    Returns:
        causal_mask: Tensor of shape (B, H, Q, K)
                     where preserved positions are 0
                     and masked positions are mask_value
    """
    _, _, q, k = attn.shape
    device = attn.device
    dtype = attn.dtype
    mask_value = torch.finfo(dtype).min

    if is_sliding:
        # Handle sliding window attention
        if q == 1:
            # Decoding stage: attend to the last k keys
            # No causal constraint needed 
            # since we're generating one token at a time
            mask = torch.full((q, k), mask_value, dtype=dtype, device=device)

            # Attend to the last k keys (or all keys if k < topk)
            start_idx = max(0, k - topk)
            mask[:, start_idx:] = 0

            # Expand mask to match the original shape (B, H, Q, K)
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            # Prefill/training stage: use causal sliding window
            # For each query position q,
            # we can attend to at most k previous keys
            # and the key index must be <= query index (causal constraint)

            # Create indices for queries and keys
            q_indices = torch.arange(q, device=device).unsqueeze(1)  # (Q, 1)
            k_indices = torch.arange(k, device=device).unsqueeze(0)  # (1, K)

            # Causal constraint: key index <= query index
            causal_mask = k_indices > q_indices  # (Q, K)

            # Sliding window constraint: key index >= query index - k + 1
            # This ensures we only attend to the k most recent keys
            sliding_mask = k_indices < (q_indices - topk + 1)  # (Q, K)

            # Combine both constraints
            combined_mask = causal_mask | sliding_mask  # (Q, K)

            # Convert boolean mask to attention mask with correct dtype
            mask = torch.where(
                combined_mask,
                torch.tensor(mask_value, dtype=dtype, device=device),
                torch.tensor(0.0, dtype=dtype, device=device)
            )

            # Expand mask to match the original shape (B, H, Q, K)
            mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        if block_size == 1:
            # Original token-wise behavior
            mask = topk_keys_from_attention_tokenwise(attn, topk)
        else:
            # Block-wise behavior
            mask = topk_keys_from_attention_blockwise(attn, topk, block_size)
    return mask


def get_dilated_indices(
    current_pos: int,
    window_size: int,
    dilation: int
):
    """
    Get dilated sliding window indices for a given position.

    Args:
        current_pos: Current token position
        window_size: Size of the sliding window
        dilation: Dilation factor (1 = no dilation, 2 = every 2nd token, etc.)

    Returns:
        List of indices to attend to
    """
    if dilation <= 0:
        dilation = 1

    indices = []
    # Start from current position and go backwards with dilation
    pos = current_pos
    count = 0

    while pos >= 0 and count < window_size:
        indices.append(pos)
        pos -= dilation
        count += 1

    # Reverse to get chronological order
    return sorted(indices)


def _obtain_block_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    topk_blocks: int,
    sliding_window_size: int | None
):
    '''
    Compute the block mask for block sparse attention.

    Args:
        query_states: shape (B, H, num_q_blocks, block_size, D)
        key_states: shape (B, H, num_k_blocks, block_size, D)
        topk_blocks: Number of top blocks to keep for each query block
        sliding_window_size: Optional size for sliding window attention
    Returns:
        topk_indices: Tensor of shape (B, H, num_blocks_Q, topk_blocks)
    '''
    # Get input dimensions
    b, h, num_q_blocks, block_size, _ = query_states.shape
    _, _, num_k_blocks, _, _ = key_states.shape

    # Compute block representations - SHARED ACROSS HEADS
    q_block_repr = query_states.mean(dim=1,
                                     dtype=torch.float32
                                     )  # [B, num_q_blocks, block_size, D]
    k_block_repr = key_states.mean(dim=1,
                                   dtype=torch.float32
                                   )  # [B, num_k_blocks, block_size, D]

    q_block_repr = q_block_repr.mean(dim=2)  # [B, num_q_blocks, D]
    k_block_repr = k_block_repr.mean(dim=2)  # [B, num_k_blocks, D]

    # Compute similarity between query blocks and key blocks
    scores = torch.bmm(
        q_block_repr,
        k_block_repr.transpose(-2, -1)
    )

    # Create causal mask at block level
    q_idx = torch.arange(num_q_blocks, device=scores.device).view(-1, 1) + 1
    k_idx = torch.arange(num_k_blocks, device=scores.device)
    block_mask = q_idx > k_idx.view(1, -1)
    if sliding_window_size is not None:
        block_mask &= q_idx - sliding_window_size // block_size <= k_idx
    scores.masked_fill_(~block_mask.unsqueeze(0), float('-inf'))

    # For each query block, always include the most recent valid key block
    # and select top_k_blocks - 1 from the rest
    num_q_blocks = q_block_repr.shape[1]
    topk_indices = torch.empty(b, num_q_blocks, topk_blocks,
                               dtype=torch.long,
                               device=scores.device
                               )

    for i in range(num_q_blocks):
        query_block_idx = i
        # The most recent valid key block for this query block
        most_recent_block = min(query_block_idx, num_k_blocks - 1)

        if topk_blocks == 1:
            # If we only select 1 block, it's always the most recent
            topk_indices[:, i, 0] = most_recent_block
        else:
            # Always include the most recent block
            topk_indices[:, i, -1] = most_recent_block

            # Mask out the most recent block from scores to select the rest
            scores_copy = scores[:, i].clone()
            scores_copy[:, most_recent_block] = float('-inf')

            # Select top_k_blocks - 1 from the remaining blocks
            _, selected_indices = torch.topk(scores_copy,
                                             k=topk_blocks - 1,
                                             dim=-1)
            selected_indices[
                selected_indices >= most_recent_block
            ] = num_k_blocks - 1

            topk_indices[:, i, :-1] = selected_indices

    topk_indices, _ = torch.sort(topk_indices, dim=-1)   # ascending order

    # Expand indices to all heads (since block selection is shared)
    topk_indices = topk_indices.unsqueeze(1)
    topk_indices = topk_indices.expand(b, h, num_q_blocks, topk_blocks)

    return topk_indices



def block_sparse_attn_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
    topk_blocks: int,
    causal_mask: torch.Tensor | None = None,
    scaling: float | None = 1.0,
    dropout_p: float | None = 0.0,
    is_causal: bool = True,
    sliding_window_size: int | None = None
):
    """
    Perform block sparse attention with
    optional sliding window and causal masking.

    Args:
        query_states: shape (B, H, Q_len, D)
        key_states: shape (B, H, K_len, D)
        value_states: shape (B, H, K_len, D)
        block_size: Size of the blocks to consider
        topk_blocks: Number of top blocks to keep for each query block
        causal_mask: Optional causal mask shape (B, 1, Q_len, K_len)
        scaling: Scaling factor for attention scores
        dropout_p: Dropout probability for attention scores
        is_causal: Whether to apply causal masking
        sliding_window_size: Optional size for sliding window attention
    Returns:
        output: shape (B, H, Q_len, D)
    """
    # Get input dimensions
    b, h, q_len, d = query_states.shape
    k_len = key_states.shape[2]

    # Calculate number of blocks for query and key
    num_q_blocks = (q_len + block_size - 1) // block_size
    num_k_blocks = (k_len + block_size - 1) // block_size

    # Ensure the number of selected blocks is 
    # not larger than the sliding window size
    if sliding_window_size is not None:
        topk_blocks = min(topk_blocks, sliding_window_size // block_size)
    topk_blocks = min(topk_blocks, num_k_blocks)

    # Padding with in-place operations when possible
    pad_len_q = (num_q_blocks * block_size) - q_len
    pad_len_k = (num_k_blocks * block_size) - k_len

    if pad_len_q > 0:
        query_states = torch.nn.functional.pad(
            query_states,
            (0, 0, 0, pad_len_q),
            value=0
        )
    if pad_len_k > 0:
        key_states = torch.nn.functional.pad(
            key_states,
            (0, 0, 0, pad_len_k),
            value=0
        )
        value_states = torch.nn.functional.pad(
            value_states,
            (0, 0, 0, pad_len_k),
            value=0
        )

    # Reshape into blocks (views, no copy)
    query_states = query_states.view(b, h, -1, block_size, d)
    key_states = key_states.view(b, h, -1, block_size, d)
    value_states = value_states.view(b, h, -1, block_size, d)

    # Get the topk blocks for each query block
    topk_indices = _obtain_block_mask(
        query_states,
        key_states,
        topk_blocks,
        sliding_window_size
    )

    # Reshape the query_states
    query_states = query_states.reshape(b * h * num_q_blocks, block_size, d)

    # Gather selected key/value blocks
    # Create index tensor for gathering
    batch_idx = torch.arange(b, device=topk_indices.device)
    batch_idx = batch_idx.view(b, 1, 1, 1)
    batch_idx = batch_idx.expand(b, h, num_q_blocks, topk_blocks)

    head_idx = torch.arange(h, device=topk_indices.device)
    head_idx = head_idx.view(1, h, 1, 1)
    head_idx = head_idx.expand(b, h, num_q_blocks, topk_blocks)

    # Flatten for advanced indexing
    batch_idx = batch_idx.reshape(-1)
    head_idx = head_idx.reshape(-1)
    topk_indices = topk_indices.reshape(-1)

    # Gather blocks using advanced indexing
    selected_k = key_states[batch_idx, head_idx, topk_indices]
    selected_v = value_states[batch_idx, head_idx, topk_indices]

    # -------------------------------------------------------------
    # Build causal mask
    # -------------------------------------------------------------
    # 1) global positions of the query tokens in this chunk
    q_pos = torch.arange(num_q_blocks * block_size, device=query_states.device)
    # shape -> (1, 1, num_q_blocks, block_size, 1)
    q_pos = q_pos.view(1, 1, num_q_blocks, block_size, 1)

    # 2) global positions of every key token that belongs to the
    #    blocks stored in `topk_indices`
    k_pos = topk_indices.unsqueeze(-1) * block_size + torch.arange(
        block_size, device=query_states.device)  # (B,H,q_blocks,top_k,block)
    k_pos = k_pos.reshape(b, h, num_q_blocks, -1)  # (B,H,q_blocks,Kchunk)
    k_pos = k_pos.unsqueeze(3)  # (B,H,q_blocks,1,Kchunk)

    # 3) causal mask   (B,H,q_blocks,block,Kchunk)
    selected_causal_mask = k_pos <= q_pos          # False = mask (future keys)

    if sliding_window_size is not None:
        selected_causal_mask &= k_pos >= q_pos - sliding_window_size

    # reshape to match the input expected by SDPA
    selected_causal_mask = selected_causal_mask.reshape(
        b * h * num_q_blocks, block_size, -1)  # (B*H*q_blocks, block, Kchunk)

    # Reshape for attention computation
    selected_k = selected_k.reshape(
        b * h * num_q_blocks,
        topk_blocks * block_size,
        d)
    selected_v = selected_v.reshape(
        b * h * num_q_blocks,
        topk_blocks * block_size,
        d)

    # Convert to 4D to accelerate attention computation with SDPA kernel
    query_states = query_states.unsqueeze(1)
    selected_k = selected_k.unsqueeze(1)
    selected_v = selected_v.unsqueeze(1)
    selected_causal_mask = selected_causal_mask.unsqueeze(1)

    # Compute attention for this chunk
    output = torch.nn.functional.scaled_dot_product_attention(
        query_states, selected_k, selected_v,
        scale=scaling,
        attn_mask=selected_causal_mask,
        is_causal=False
    )

    # Reshape back to original dimensions
    output = output.squeeze(1)
    output = output.view(b, h, num_q_blocks * block_size, d)
    output = output[:, :, :q_len]

    return output


def _compute_cumulative_attention_mass_ratio(
    attn_weights: torch.Tensor,
    window_size: int,
    epsilon: float,
    position_candidates: list[int],
    is_sliding: bool,
    sliding_window_size: int = 4096,
    smooth_ends: bool = False,
) -> torch.Tensor:
    """
    Compute the cumulative attention mass ratio for
    a list of candidate positions.

    Args:
        attn_weights: Tensor of shape (L, L) representing the attention scores
        window_size: Size of the sliding window
        epsilon: Small value to avoid division by zero
        position_candidates: List of candidate positions
        is_sliding: Whether to use a sliding window approach
        sliding_window_size: Size of the sliding window for the future context
        smooth_ends: Whether to smooth both ends of the candidate sequence

    Returns:
        ratios: Cumulative attention mass ratio for each candidate position
    """
    if not position_candidates:
        return torch.empty(0, dtype=attn_weights.dtype, device=attn_weights.device)

    l = attn_weights.size(0)
    cand = torch.as_tensor(position_candidates, device=attn_weights.device)

    if smooth_ends:
        # Smooth the ends of the candidate sequence by adding a buffer zone
        valid = (cand >= 2 * window_size - 1) & (cand < l - 2 * window_size)
    else:
        valid = (cand >= window_size - 1) & (cand < l - window_size)

    ratios = torch.zeros_like(cand, dtype=torch.float64)  # work in FP64
    if not valid.any():
        return ratios

    cand = cand[valid]

    # ------------------------------------------------------------------ #
    #  Build integral image in FP64  (two cumsums still run in one kernel)
    # ------------------------------------------------------------------ #
    s = torch.zeros(
        (l + 1, l + 1), device=attn_weights.device, dtype=torch.float64
    )
    s[1:, 1:] = (
        attn_weights.to(torch.float64)
        .cumsum(0, dtype=torch.float64)
        .cumsum(1, dtype=torch.float64)
    )

    def _rect_sum(r1, c1, r2, c2):
        # r1,c1,r2,c2 are 1-based inclusive
        return s[r2, c2] - s[r1 - 1, c2] - s[r2, c1 - 1] + s[r1 - 1, c1 - 1]

    w = window_size
    r1 = cand + w + 1  # 1-based
    if is_sliding:
        r2 = torch.clamp(cand + w + sliding_window_size, max=l)
    else:
        r2 = torch.full_like(r1, l)  # last row index (1-based)

    # columns for "past": cand-(w-1) … cand
    c1_p = (cand - (w - 1)) + 1
    c2_p = cand + 1

    # columns for "future": cand+1 … cand+w
    c1_f = (cand + 1) + 1
    c2_f = cand + w + 1

    a_past = _rect_sum(r1, c1_p, r2, c2_p)
    a_fut = _rect_sum(r1, c1_f, r2, c2_f)

    bigger = torch.maximum(a_past, a_fut)
    smaller = torch.minimum(a_past, a_fut)
    ratio_ok = (bigger + epsilon) / (smaller + epsilon)

    ratios[valid] = ratio_ok
    return ratios


def _select_boundary_candidates(
    position_candidates: list[int],
    ratios: torch.Tensor,
    theta: float,
    topk: int,
    nms_window_size: int = 0,
    instruction_tokens: int | None = None
) -> list[int]:
    """
    Apply (1) thresholding, (2) ratio-sorting and (3) 1-D Non-Maximum
    Suppression (NMS) to a list of candidate positions.

    Args
    ----
    position_candidates: original indices in the sequence.
    ratios: score for each candidate (same ordering).
    theta: keep only ratio > theta.
    topk: at most k items in the final list.
    nms_window_size: nms window half-width.  If two candidates
                     differ by ≤ window_size, the second one
                     is suppressed.  Set 0 to disable NMS.
    instruction_tokens: instruction tokens.

    Returns
    -------
    final candidate indices in ascending order.
    """
    # 1. Threshold
    if theta is not None:
        keep_mask = ratios > theta
        keep_idx = torch.where(keep_mask)[0]
    else:
        keep_idx = torch.arange(len(ratios), device=ratios.device)

    if keep_idx.numel() == 0:
        return []

    # 2. Gather & sort by score (descending)
    cand_scores = ratios[keep_idx]
    cand_ids = torch.tensor(position_candidates, device=ratios.device)[keep_idx]

    order = torch.argsort(cand_scores, descending=True)
    cand_ids = cand_ids[order]

    # Find the position of the instruction part (if any)
    instruct_position = None
    if instruction_tokens is not None:
        instruct_position = len(position_candidates) - instruction_tokens

    # 3. Non-Maximum Suppression
    selected: list[int] = [] if instruct_position is None else [instruct_position]
    for idx in cand_ids.tolist(): # already in score order
        if instruct_position is not None and idx > instruct_position:
            continue
        if nms_window_size == 0:
            selected.append(idx)
        else:
            # Check distance to every previously accepted index
            if all(abs(idx - sel) > nms_window_size for sel in selected):
                selected.append(idx)

        if len(selected) >= topk:                    # early exit
            break

    # Return in ascending order (change to selected for score order)
    return sorted(selected)


def label_boundaries(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_chunks: int,
    layer_idx: int,
    boundary_window_size: int = 4,
    theta: float = 1.1,
    use_nms: bool = False,
    nms_window_size: int = -1,
    instruction_tokens: int | None = None
):
    '''
    Automatically select the boundaries for each chunk based on the cumulative attention mass ratio with non-maximum suppression (NMS).

    Args:
        query_states: shape (B, H, num_q_blocks, block_size, D)
        key_states: shape (B, H, num_k_tokens, 1, D)
        attention_mask: shape (B, H, num_q_blocks, block_size, block_size)
        num_chunks: the number of chunks to select
        layer_idx: the layer index
        boundary_window_size: the window size for computing the cumulative attention mass ratio
        theta: the threshold for selecting the boundary candidates
        use_nms: whether to use non-maximum suppression
        nms_window_size: the window size for non-maximum suppression
        instruction_tokens: the length of the instruction part

    Returns:
        boundaries: shape (B, num_chunks)
    '''
    # Generate position candidates
    num_k_tokens = key_states.shape[-2]
    position_candidates = list(range(0, num_k_tokens))

    # Compute attention weights
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(query_states.shape[-1])

    # Ensure attn_weights is causal
    if attention_mask is not None:
        attn_weights.masked_fill_(attention_mask < -1e9, torch.finfo(attn_weights.dtype).min)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Average over heads
    attn_weights = attn_weights.mean(dim=1)  # (B, L, L)

    boundaries = []
    ratios = []
    b = query_states.shape[0]
    for i in range(b):
        # For each sample in batch,
        # compute cumulative attention mass ratio for each position
        ratio = _compute_cumulative_attention_mass_ratio(
            attn_weights[i],
            window_size=boundary_window_size,
            epsilon=1e-6,
            position_candidates=position_candidates,
            is_sliding=not bool(layer_idx % 2),
            smooth_ends=True
        )
        # Select topK boundary candidates with NMS
        boundary = _select_boundary_candidates(
            position_candidates=position_candidates,
            ratios=ratio,
            theta=theta,
            topk=num_chunks,
            nms_window_size=nms_window_size if (use_nms and nms_window_size > 0) else 0,
            instruction_tokens=instruction_tokens
        )

        # Add 1 to the boundary indices to account for the chunk end
        # Suppose the chunk ends are i and j,
        # then we need to use [i+1, j+1] to indicate the chunk range
        boundary = [i+1 for i in boundary]

        # Add start and end boundaries
        if boundary[0] != 0:
            boundary = [0] + boundary
        if boundary[-1] != len(position_candidates):
            boundary += [len(position_candidates)]
        boundaries.append(boundary)

        ratios.append(ratio)

    # Convert to tensor
    boundaries = torch.tensor(boundaries, device=query_states.device)
    ratios = torch.tensor(ratios, device=query_states.device)
    return boundaries, ratios


def predict_boundaries(
    predictor: torch.nn.Module,
    key_states: torch.Tensor,
    num_chunks: int,
    boundary_window_size: int = 4,
    theta: float = 1.1,
    use_nms: bool = True,
    nms_window_size: int = 8,
    instruction_tokens: int | None = None,
    smooth_ends: bool = False
):
    '''
    Predict the boundaries for each chunk based on the key states.

    Args:
        predictor: The boundary predictor model.
        key_states: The key states tensor.
        num_chunks: The number of chunks.
        boundary_window_size: The boundary window size.
        theta: The threshold for the boundary predictor.
        use_nms: Whether to use NMS.
        nms_window_size: The NMS window size.
        instruction_tokens: The instruction tokens.
        smooth_ends: Whether to smooth the ends of the boundaries.

    Returns:
        boundaries: The boundaries tensor.
        ratios: The ratios tensor.
    '''
    # Prepare the features for the boundary predictor.
    b, h, l, d = key_states.shape
    feat = key_states.permute(0, 2, 1, 3).reshape(b, l, h * d).float()
    feat = feat.permute(0, 2, 1).contiguous()
    feat = feat.to(key_states.device)

    ratios = predictor(feat)

    if smooth_ends:
        ratios[:, : 2 * boundary_window_size - 1] = -100
        ratios[:, l - 2 * boundary_window_size:] = -100
    else:
        ratios[:, :boundary_window_size-1] = -100
        ratios[:, l-boundary_window_size:] = -100

    candidates = list(range(l))
    boundaries = []
    for i in range(b):
        boundary = _select_boundary_candidates(
            position_candidates=candidates,
            ratios=ratios[i],
            theta=theta,
            topk=num_chunks,
            nms_window_size=nms_window_size if (use_nms and nms_window_size > 0) else 0,
            instruction_tokens=instruction_tokens
        )
        boundary = [i+1 for i in boundary]

        # Add start and end boundaries
        if boundary[0] != 0:
            boundary = [0] + boundary
        if boundary[-1] != len(candidates):
            boundary += [len(candidates)]
        boundaries.append(boundary)

    boundaries = torch.tensor(boundaries, device=key_states.device)
    return boundaries, ratios


def _obtain_dhsa_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    topk_tokens: int,
    q_len: int,
    q_pad_len: int,
    boundaries: torch.Tensor,
    sliding_window_size: int | None = None,
    pooling: str = 'avgpool',
    local_tokens: int = 64
):
    '''
    Obtain the topk tokens mask for each query block.

    Args:
        query_states: shape (B, H, num_q_blocks, block_size, D)
        key_states: shape (B, H, num_k_tokens, 1, D)
        topk_tokens: the number of topk tokens to select for each query block
        q_len: the length of the query part
        q_pad_len: the length of the padded query part (0 if no padding)
        boundaries: shape (B, num_chunks)
        sliding_window_size: the sliding window size
        pooling: the pooling method
        local_tokens: the local window size for each query block

    Returns:
        topk_indices: shape (B, H, num_q_blocks, topk_tokens)
    '''
    # Get the input shapes
    b, h, num_q_blocks, block_size, d = query_states.shape
    num_k_tokens = key_states.shape[-3]

    # Compute block representations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Disable autocast to use FP32 for block selection
    with torch.amp.autocast(device, enabled=False):
        # Average across heads
        q_block_repr = query_states.mean(dim=1, dtype=torch.float32)  # resulting shape: (B, num_q_blocks, block_size, D)
        k_block_repr = key_states.mean(dim=1, dtype=torch.float32)    # resulting shape: (B, num_k_tokens, 1, D)

        # Pooling across block size
        if pooling in ['avgpool', 'avgpool_norm']:
            q_block_repr = q_block_repr.mean(dim=2)  # resulting shape: (B, num_q_blocks, D)
            if pooling == 'avgpool_norm':
                # Obtain the actual length of each query block
                # mask of real query tokens (1) vs padded (0)
                q_mask = torch.ones(b, q_len, device=query_states.device, dtype=torch.float32)
                if q_pad_len:
                    q_mask = torch.nn.functional.pad(q_mask, (0, q_pad_len), value=0)
                # length of each query block  →  shape (B, num_q_blocks)
                q_block_len = q_mask.view(b, num_q_blocks, block_size).sum(-1)
                # normalize by sqrt of length
                q_block_repr = q_block_repr * torch.sqrt(q_block_len.unsqueeze(-1))
        elif pooling == 'maxpool':
            q_block_repr = q_block_repr.max(dim=2).values  # resulting shape: (B, num_blocks, D)
        else:
            raise ValueError(f'Unsupported pooling method: {pooling}')

        k_block_repr = k_block_repr.squeeze(2)  # resulting shape: (B, num_k_tokens, D)

        # Build the prefix‐sum for keys
        k_pref = torch.cat(
            [torch.zeros(b, 1, d, device=k_block_repr.device, dtype=torch.float32),
            torch.cumsum(k_block_repr, dim=1)],
            dim=1
        )  # resulting shape: (B, num_k_tokens+1, D)

        # Gather the prefix‐sum for start and end boundary indices
        start = boundaries[:, :-1]  # resulting shape: (B, num_chunks)
        end = boundaries[:, 1:]   # resulting shape: (B, num_chunks)
        idx_ex = lambda x: x.unsqueeze(-1).expand(-1, -1, d)

        pref_s = torch.gather(k_pref, 1, idx_ex(start))
        pref_e = torch.gather(k_pref, 1, idx_ex(end))

        # Calculate chunk_sum and chunk_len
        chunk_sum = pref_e - pref_s     # resulting shape: (B, num_chunks, D)
        chunk_len = (end - start).unsqueeze(-1)
        chunk_len = chunk_len.clamp(min=1).to(torch.float32)  # resulting shape: (B, num_chunks, 1)

        # Compute the mean
        k_block_repr = chunk_sum / chunk_len   # resulting shape: (B, num_chunks, D)

        if pooling == 'avgpool_norm':
            k_block_repr = k_block_repr * torch.sqrt(chunk_len)

        # clean up the temporary tensors
        del k_pref, pref_s, pref_e, chunk_sum, chunk_len

    # Update the number of query blocks
    num_q_blocks = q_block_repr.shape[1]

    # Chunk-level scores (B, num_q_blocks, num_chunks)
    scores = torch.bmm(
        q_block_repr,
        k_block_repr.transpose(-2, -1)
    )

    # Perform upsampling to get the token-level scores
    # ------------------------------------------------------------------
    # 1) build a  (B , num_k_tokens)   tensor that maps every token
    #    index j  ->  the chunk index c  that contains this block
    # ------------------------------------------------------------------
    block_idx = torch.arange(num_k_tokens, device=boundaries.device)  # all tokens indices 0 … num_k_tokens-1
    block_idx = block_idx.unsqueeze(0).expand(b, -1)    # resulting shape: (B, K)

    # searchsorted / bucketize tells us into which "bin" the value falls.
    # right=True  ⇒  a value that is itself a boundary element is put
    #                into the bin that *starts* there.
    #
    # Result:  (B, K) with entries  0,0,0,0,1,1,1,1,1, …  (example)
    block2chunk = torch.searchsorted(
        boundaries,   # sorted_sequence (B , num_chunks)
        block_idx,    # values (B , K)
        right=True
    ) - 1        # shift because first bin is 0

    # ------------------------------------------------------------------
    # 2) expand the compressed scores with a single gather
    # ------------------------------------------------------------------
    # make block2chunk indexable along scores.dim=2
    index = block2chunk.unsqueeze(1)           # resulting shape: (B , 1 , K)
    index = index.expand(-1, scores.size(1), -1)

    # token-level scores
    scores = torch.gather(
        scores,  # (B, num_q_blocks, num_chunks)
        dim=2,
        index=index
    )   # (B, num_q_blocks, K)

    # Create causal mask at block level
    q_idx = torch.arange(num_q_blocks, device=scores.device).view(-1, 1) + 1
    q_idx = q_idx * block_size

    block_mask = q_idx > torch.arange(num_k_tokens, device=scores.device).view(1, -1)
    if sliding_window_size is not None:
        block_mask &= q_idx - sliding_window_size <= torch.arange(num_k_tokens, device=scores.device)
    scores.masked_fill_(~block_mask.unsqueeze(0), float('-inf'))

    # For each query block, always include the most recent valid key tokens
    # and select top_k_blocks - local_window_tokens from the rest
    topk_indices = torch.empty(b, num_q_blocks, topk_tokens, dtype=torch.long, device=scores.device)

    for i in range(num_q_blocks):
        # Valid (causal + sliding-window) key indices for this query block i
        allowed_i = torch.nonzero(block_mask[i], as_tuple=False).squeeze(-1)   # 1D [K_allowed]
        if allowed_i.numel() == 0:
            # Extremely early edge-case; just fill zeros
            topk_indices[:, i, :] = 0
            continue

        # How many recent tokens must we include
        recent_tokens = min(local_tokens, topk_tokens, allowed_i.numel())
        recent_idx = allowed_i[-recent_tokens:]   # last-N allowed = most recent N tokens

        # Start by placing the required recent indices in the *end* slots
        topk_indices[:, i, -recent_tokens:] = recent_idx.view(1, -1).expand(b, -1)

        # Fill remaining slots with top scores excluding the must-include set
        remaining = topk_tokens - recent_tokens
        if remaining > 0:
            scores_copy = scores[:, i].clone()
            scores_copy[:, recent_idx] = float('-inf')  # avoid duplicates

            # You can only choose from the remaining allowed set
            remaining_capacity = max(0, allowed_i.numel() - recent_tokens)
            k_select = min(remaining, remaining_capacity)

            if k_select > 0:
                _, selected = torch.topk(scores_copy, k=k_select, dim=-1)   # [b, k_select]
                topk_indices[:, i, :k_select] = selected

            # If we still have leftover slots (e.g., very short prefix), pad with num_k_tokens - 1
            if k_select < remaining:
                pad_val = num_k_tokens - 1
                topk_indices[:, i, k_select:remaining] = pad_val

    topk_indices, _ = torch.sort(topk_indices, dim=-1)   # Ensure ascending order

    # Expand indices to all heads (since block selection is shared)
    topk_indices = topk_indices.unsqueeze(1).expand(b, h, num_q_blocks, topk_tokens)

    return topk_indices




def _dhsa_sdpa_single_block(
    q_tokens: torch.Tensor,
    selected_k: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_start_idx: int = 0,
    scaling: float | None = None,
    sliding_window_size: int | None = None,
):
    """
    Perform DHSA within a single block.

    Args:
        q_tokens  : (B, H, L, D)
        selected_k: (B, H, K) indices into key_states / value_states
        key_states: (B, H, K, D) key states
        value_states: (B, H, K, D) value states
        q_start_idx: start index of the query tokens
        scaling: scaling factor for attention scores
        sliding_window_size: optional size for sliding window attention

    Returns:
        out: (B, H, L, D)
    """
    # Get input shapes
    b_, h_, l_, d_ = q_tokens.shape

    # Gather keys / values
    sel = selected_k.reshape(-1)  # (B_ * H_ * K,)
    batch_i = torch.arange(b_, device=sel.device).view(b_, 1, 1)
    batch_i = batch_i.expand(b_, h_, selected_k.size(-1))
    head_i = torch.arange(h_, device=sel.device).view(1, h_, 1)
    head_i = head_i.expand(b_, h_, selected_k.size(-1))

    sel_k = key_states[batch_i.reshape(-1), head_i.reshape(-1), sel]
    sel_k = sel_k.reshape(b_, h_, selected_k.size(-1), d_)

    sel_v = value_states[batch_i.reshape(-1), head_i.reshape(-1), sel]
    sel_v = sel_v.reshape(b_, h_, selected_k.size(-1), d_)

    q_flat = q_tokens.reshape(b_*h_, l_, d_)
    k_flat = sel_k.reshape(b_*h_, -1, d_)
    v_flat = sel_v.reshape(b_*h_, -1, d_)

    # Causal mask: keys must not cross the usual causal frontier
    q_pos = torch.arange(l_, device=q_tokens.device).view(1, l_, 1)
    q_pos = q_pos + q_start_idx

    k_pos = sel.unsqueeze(-1) + torch.arange(1, device=q_tokens.device)
    k_pos = k_pos.reshape(b_, h_, -1)  # (B_, H_, Nk)
    k_pos = k_pos.unsqueeze(2)  # (B_, H_, 1, Nk)

    causal = (k_pos <= q_pos)
    if sliding_window_size is not None:
        causal &= (k_pos >= q_pos - sliding_window_size)

    causal = causal.reshape(b_*h_, l_, -1)

    # Convert to 4D for attention computation
    q_flat = q_flat.unsqueeze(1)
    k_flat = k_flat.unsqueeze(1)
    v_flat = v_flat.unsqueeze(1)
    causal = causal.unsqueeze(1)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_flat,
        k_flat,
        v_flat,
        scale=scaling,
        attn_mask=causal,
        is_causal=False
    )
    out = out.squeeze(1)

    return out.reshape(b_, h_, l_, d_)


def dhsa_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
    topk_tokens: int,
    causal_mask: torch.Tensor | None = None,
    scaling: float | None = 1.0,
    dropout_p: float | None = 0.0,
    is_causal: bool = True,
    sliding_window_size: int | None = None,
    boundaries: torch.Tensor | None = None,
    instruction_tokens: int = 64,
    loop_times: int = 4,
    pooling: str = 'avgpool',
    topk_indices: torch.Tensor | None = None,
    global_tokens: int = 128,
    local_tokens: int = 128
):
    """
    Perform dynamic hierarchical sparse attention (DHSA) with optional sliding window and causal masking.

    Args:
        query_states: Tensor of shape (B, H, Q_len, D) representing query states
        key_states: Tensor of shape (B, H, K_len, D) representing key states
        value_states: Tensor of shape (B, H, K_len, D) representing value states
        block_size: Size of the blocks to consider
        topk_tokens: Number of top blocks to keep for each query block
        causal_mask: Optional causal mask tensor of shape (B, 1, Q_len, K_len)
        scaling: Scaling factor for attention scores
        dropout_p: Dropout probability for attention scores
        is_causal: Whether to apply causal masking
        sliding_window_size: Optional size for sliding window attention
        boundaries: boundary tensor of shape (B, num_chunks)
        instruction_tokens: number of instruction tokens at the end
        loop_times: number of times to loop through all the query blocks
        pooling: pooling method for block representation ('avgpool', 'avgpool_norm', 'maxpool')
        topk_indices: Precomputed topk indices tensor of shape (B, H, num_q_blocks, topk_tokens)
        global_tokens: Number of global tokens to always include
        local_tokens: Number of local window tokens to always include

    Returns:
        output: Tensor of shape (B, H, Q_len, D) representing the attention output
    """
    # Get the instruction length and tensor if instruction_tokens is provided
    q_instruct_tokens = 0
    q_instruct_tensor = None
    if instruction_tokens is not None:
        q_instruct_tensor = query_states[:, :, -instruction_tokens:, :]
        query_states = query_states[:, :, :-instruction_tokens, :]
        q_instruct_tokens = q_instruct_tensor.shape[2]

    # Get input shapes
    batch, head, q_len, dim = query_states.shape
    k_len = key_states.shape[2]

    # Calculate number of blocks for query and key
    num_q_blocks = (q_len + block_size - 1) // block_size
    num_k_tokens = k_len

    # Ensure the number of selected tokens 
    # is not larger than the sliding window size
    if sliding_window_size is not None:
        topk_tokens = min(topk_tokens, sliding_window_size)
    topk_tokens = min(topk_tokens, num_k_tokens)

    # Pre-allocate output with exact size needed (avoid padding)
    output = torch.empty(
        batch, head, q_len + q_instruct_tokens, dim,
        dtype=query_states.dtype,
        device=query_states.device
    )

    # Padding with in-place operations when possible
    pad_q_len = (num_q_blocks * block_size) - q_len
    if pad_q_len > 0:
        query_states = torch.nn.functional.pad(
            query_states,
            (0, 0, 0, pad_q_len),
            value=0
        )

    # Reshape into blocks (views, no copy)
    query_states = query_states.view(batch, head, -1, block_size, dim)
    key_states = key_states.view(batch, head, -1, 1, dim)
    value_states = value_states.view(batch, head, -1, 1, dim)

    # Get the topk tokens for each query block
    if topk_indices is None:
        topk_indices = _obtain_dhsa_mask(
            query_states,
            key_states,
            topk_tokens,
            q_len,
            pad_q_len,
            boundaries,
            sliding_window_size,
            pooling,
            local_tokens
        )
    else:
        # Since the sparsity mask is shared across all layers,
        # we can remove the head dimension for passing to subsequent layers.
        # When using the shared sparsity mask,
        # we need to expand the head dimension
        # back to the original size.
        topk_indices = topk_indices.unsqueeze(1).expand(-1, head, -1, -1)

    # Add global tokens
    if global_tokens > 0:
        device = topk_indices.device
        dtype  = topk_indices.dtype  # long
        K = key_states.size(2)       # key_states shaped (B,H,K,1,D)

        # Per-block start and upper bound
        q_starts = torch.arange(num_q_blocks, device=device) * block_size           # (Bq,)
        q_uppers = torch.minimum(q_starts + block_size, torch.full_like(q_starts, K))  # (Bq,)

        # ORIGINAL start: window begins at q_start - window (clamped)
        if (sliding_window_size is not None) and (sliding_window_size > 0):
            win_starts = (q_starts - sliding_window_size).clamp_min(0)              # (Bq,)
        else:
            win_starts = torch.zeros_like(q_starts)                                  # (Bq,)

        # Build consecutive globals per block starting at win_starts (unchanged)
        ar = torch.arange(global_tokens, device=device).view(1, 1, 1, -1)           # (1,1,1,G)
        win_starts_b = win_starts.view(1, 1, num_q_blocks, 1)                        # (1,1,Bq,1)
        cand = win_starts_b + ar                                                     # (1,1,Bq,G)

        # FIX: cap by (q_upper - 1), not (q_start - 1); also cap by K-1
        max_idx = (q_uppers - 1).clamp_min(0)                                        # (Bq,)
        max_idx_b = max_idx.view(1, 1, num_q_blocks, 1)                              # (1,1,Bq,1)

        cand = torch.minimum(cand, max_idx_b)                                        # causal cap
        cand = cand.clamp_min(0)                                                     # safety
        global_idx = cand.expand(batch, head, num_q_blocks, global_tokens).to(dtype) # (B,H,Bq,G)

        # Splice into front of per-block topk list
        if global_tokens < topk_indices.size(-1):
            topk_indices = torch.cat([global_idx, topk_indices[..., global_tokens:]], dim=-1)
        else:
            topk_indices = global_idx[..., :topk_indices.size(-1)]

    # Process query blocks in loops to reduce peak memory
    loop_size = max(1, num_q_blocks // loop_times)

    for loop_start in range(0, num_q_blocks, loop_size):
        loop_end = min(loop_start + loop_size, num_q_blocks)
        act_loop_size = loop_end - loop_start

        # Extract corresponding query blocks
        loop_q = query_states[:, :, loop_start:loop_end]
        loop_q = loop_q.reshape(batch * head * act_loop_size, block_size, dim)

        # Get indices for this loop
        loop_topk_indices = topk_indices[:, :, loop_start:loop_end]

        # Gather selected key/value tokens for this loop
        # Create index tensor for gathering
        batch_idx = torch.arange(batch, device=topk_indices.device)
        batch_idx = batch_idx.view(batch, 1, 1, 1)
        batch_idx = batch_idx.expand(batch, head, act_loop_size, topk_tokens)

        head_idx = torch.arange(head, device=topk_indices.device)
        head_idx = head_idx.view(1, head, 1, 1)
        head_idx = head_idx.expand(batch, head, act_loop_size, topk_tokens)

        # Flatten for advanced indexing
        batch_idx = batch_idx.reshape(-1)
        head_idx = head_idx.reshape(-1)
        loop_topk_indices = loop_topk_indices.reshape(-1)

        # Gather blocks using advanced indexing
        selected_k = key_states[batch_idx, head_idx, loop_topk_indices]
        selected_v = value_states[batch_idx, head_idx, loop_topk_indices]

        # -------------------------------------------------------------
        # Build causal mask for this loop
        # -------------------------------------------------------------
        # 1) global positions of the query tokens in this loop
        q_pos = loop_start * block_size
        q_pos += torch.arange(
            act_loop_size * block_size,
            device=query_states.device
        )

        # shape -> (1, 1, act_loop_size, block_size, 1)
        q_pos = q_pos.view(1, 1, act_loop_size, block_size, 1)

        # 2) global positions of every key token
        # that stored in `loop_topk_indices`
        k_pos = loop_topk_indices.unsqueeze(-1)
        k_pos += torch.arange(1, device=query_states.device)  # (B,H,q_blocks,top_k,block)
        k_pos = k_pos.reshape(batch, head, act_loop_size, -1)  # (B,H,loop_q_blocks,topk_tokens)
        k_pos = k_pos.unsqueeze(3)  # (B,H,loop_q_blocks,1,topk_tokens)

        # 3) causal mask   (B,H,loop_q_blocks,block,topk_tokens)
        selected_causal_mask = k_pos <= q_pos

        if sliding_window_size is not None:
            selected_causal_mask &= k_pos >= q_pos - sliding_window_size

        # reshape to match the input expected by SDPA
        selected_causal_mask = selected_causal_mask.reshape(
            batch * head * act_loop_size, block_size, -1
        )   # (B*H*loop_q_blocks, block, topk_tokens)

        # Free intermediate tensors
        del batch_idx, head_idx, loop_topk_indices, q_pos, k_pos

        # Reshape for attention computation
        selected_k = selected_k.reshape(batch * head * act_loop_size, topk_tokens, dim)
        selected_v = selected_v.reshape(batch * head * act_loop_size, topk_tokens, dim)

        # Convert to 4D for attention computation
        loop_q = loop_q.unsqueeze(1)
        selected_k = selected_k.unsqueeze(1)
        selected_v = selected_v.unsqueeze(1)
        selected_causal_mask = selected_causal_mask.unsqueeze(1)

        # Compute attention for this chunk
        loop_attn = torch.nn.functional.scaled_dot_product_attention(
            loop_q, selected_k, selected_v,
            scale=scaling,
            attn_mask=selected_causal_mask,
            is_causal=False
        )
        loop_attn = loop_attn.squeeze(1)

        # Write to output
        loop_op_start = loop_start * block_size
        loop_op_end = min(loop_end * block_size, q_len)
        loop_act_len = loop_op_end - loop_op_start

        if loop_act_len > 0:
            loop_attn = loop_attn.view(batch, head, act_loop_size * block_size, dim)
            output[:, :, loop_op_start:loop_op_end] = loop_attn[:, :, :loop_act_len]

        # Free intermediate tensors
        del loop_q, selected_k, selected_v, selected_causal_mask, loop_attn

    # Deal with the instruction part
    if q_instruct_tokens:
        instruct_selected_key = torch.arange(k_len, device=topk_indices.device)
        instruct_selected_key = instruct_selected_key.unsqueeze(0).unsqueeze(0).expand(batch, head, -1)
        instruct_out = _dhsa_sdpa_single_block(
            q_instruct_tensor,
            instruct_selected_key,
            key_states,
            value_states,
            q_len,
            scaling,
            sliding_window_size
        )
        output[:, :, q_len:] = instruct_out

    # Since the sparsity mask is shared across all layers,
    # we can remove the head dimension for passing to subsequent layers.
    return output, topk_indices[:, 0, :, :]