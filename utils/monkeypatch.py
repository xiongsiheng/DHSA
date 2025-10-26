"""
Monkey patching utilities for various attention mechanisms in transformers models.
"""

import transformers
from transformers.models import gemma2


from .hybridcache import (
    __init__,
    update,
    get_seq_length,
    reset
)


from .gemma_model import (
    gemma_sdpa_forward_PyramidKV,
    gemma_sdpa_attn_forward_H2O,
    gemma_sdpa_attn_forward_StreamingLLM,
    gemma_sdpa_attn_forward_sliding_window,
    gemma_sdpa_attn_forward_topk,
    gemma_sdpa_attn_forward_block_sparse,
    gemma_sdpa_attn_forward_dhsa,
    prepare_inputs_for_generation_gemma,
    gemma2model_forward_dhsa,
    gemma2decoderlayer_forward_dhsa
)

from .config import SparseAttnMethod



def replace_gemma(method):
    """Replace Gemma attention with PyramidKV implementations"""
    if method == SparseAttnMethod.slidingwindow.value:
        print("Using Gemma2 Sliding Window attention forward.")
        gemma2.modeling_gemma2.Gemma2Attention.forward = gemma_sdpa_attn_forward_sliding_window

    elif method == SparseAttnMethod.streamingllm.value:
        print("Using Gemma2 StreamingLLM attention forward.")
        gemma2.modeling_gemma2.Gemma2Attention.forward = gemma_sdpa_attn_forward_StreamingLLM

    elif method == SparseAttnMethod.h2o.value:
        print("Using Gemma2 H2O attention forward.")
        gemma2.modeling_gemma2.Gemma2Attention.forward = gemma_sdpa_attn_forward_H2O

    elif method == SparseAttnMethod.pyramidkv.value:
        print("Using Gemma2 PyramidKV attention forward.")
        gemma2.modeling_gemma2.Gemma2Attention.forward = gemma_sdpa_forward_PyramidKV

    elif method == SparseAttnMethod.topk.value:
        print("Using Gemma2 TopK attention forward.")
        gemma2.modeling_gemma2.Gemma2Attention.forward = gemma_sdpa_attn_forward_topk

    elif method == SparseAttnMethod.blocksparse.value:
        print("Using Gemma2 BlockSparse attention forward.")
        gemma2.modeling_gemma2.Gemma2Attention.forward = gemma_sdpa_attn_forward_block_sparse

    elif method == SparseAttnMethod.dhsa.value:
        print("Using Gemma2 DHSA attention forward.")
        gemma2.modeling_gemma2.Gemma2Attention.forward = gemma_sdpa_attn_forward_dhsa

        gemma2.modeling_gemma2.Gemma2Model.forward = gemma2model_forward_dhsa
        gemma2.modeling_gemma2.Gemma2DecoderLayer.forward = gemma2decoderlayer_forward_dhsa

    if method != SparseAttnMethod.full.value:
        gemma2.modeling_gemma2.Gemma2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_gemma

        transformers.cache_utils.HybridCache.__init__ = __init__
        transformers.cache_utils.HybridCache.update = update
        transformers.cache_utils.HybridCache.get_seq_length = get_seq_length
        transformers.cache_utils.HybridCache.reset = reset



def _configure_attention_layers(
    args,
    model,
    predictor = None
) -> None:
    """
    Configure attention layers with sparse attention parameters.

    Args:
        model: The model to configure
        window_size: The window size for the attention layers
        predictor: The boundary predictor for the attention layers
        args: The arguments for the attention layers

    Returns:
        None
    """
    # Set default values for sparse attention parameters.
    if not hasattr(args, "kv_compression_window_size"):
        args.kv_compression_window_size = 8
    if not hasattr(args, "dhsa_boundary_window_size"):
        args.dhsa_boundary_window_size = 4
    if not hasattr(args, "dhsa_boundary_ratio_theta"):
        args.dhsa_boundary_ratio_theta = 1.1
    if not hasattr(args, "dhsa_chunk_beta"):
        args.dhsa_chunk_beta = 8
    if not hasattr(args, "dhsa_use_nms"):
        args.dhsa_use_nms = True
    if not hasattr(args, "dhsa_nms_window_size"):
        args.dhsa_nms_window_size = 8
    if not hasattr(args, "dhsa_loop_times"):
        args.dhsa_loop_times = 4
    if not hasattr(args, "dhsa_chunk_representation_pooling"):
        args.dhsa_chunk_representation_pooling = "avgpool_norm"
    if not hasattr(args, "dhsa_share_boundaries"):
        args.dhsa_share_boundaries = True
    if not hasattr(args, "dhsa_share_sparsity_masks"):
        args.dhsa_share_sparsity_masks = False

    layers = len(model.model.layers)
    # Configure each layer
    for i in range(layers):
        attn_config = model.model.layers[i].self_attn.config
        attn_config.window_size = args.kv_compression_window_size
        attn_config.max_capacity_prompt = args.budget_decode
        attn_config.budget_prefill = args.budget_prefill

        if args.method in [SparseAttnMethod.blocksparse.value, SparseAttnMethod.dhsa.value]:
            attn_config.block_size = args.block_size

        if args.method == SparseAttnMethod.dhsa.value:
            attn_config.boundary_window_size = args.dhsa_boundary_window_size
            attn_config.boundary_ratio_theta = args.dhsa_boundary_ratio_theta
            attn_config.chunk_beta = args.dhsa_chunk_beta
            attn_config.use_nms = args.dhsa_use_nms
            attn_config.nms_window_size = args.dhsa_nms_window_size
            attn_config.loop_times = args.dhsa_loop_times

            attn_config.chunk_representation_pooling = args.dhsa_chunk_representation_pooling
            attn_config.share_boundaries = args.dhsa_share_boundaries
            attn_config.share_sparsity_masks = args.dhsa_share_sparsity_masks

            model.model.layers[i].self_attn.boundary_predictor = predictor
            model.model.layers[i].self_attn.boundaries = None
            model.model.layers[i].self_attn.ratios = None