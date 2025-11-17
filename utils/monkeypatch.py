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


import os, time
import torch
import torch.nn as nn
from transformers.generation.utils import (
    GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig, 
    Optional, Union, GenerateNonBeamOutput, GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)
from transformers.generation.streamers import BaseStreamer


def patched_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    f_show_stats: bool = True,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        f_show_stats (`bool`, *optional*, defaults to `True`):
            Whether to show latency and peak memory statistics during generation.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True


    t_sum_decode = 0
    cnt = 0
    peak_alloc = 0
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            if f_show_stats:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()   # start a fresh peak counter
                t0 = time.time()
            outputs = self(**model_inputs, return_dict=True)
            
            if f_show_stats:
                torch.cuda.synchronize()
                t1 = time.time()
                print('Time taken for prefill (GPU):', t1 - t0)
                cur_peak_alloc    = torch.cuda.max_memory_allocated()
                print('Peak memory allocated (GPU) for prefill:', cur_peak_alloc / 1024 / 1024 / 1024, 'GB')
            
            is_prefill = False
        else:
            if f_show_stats:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t2 = time.time()

            outputs = model_forward(**model_inputs, return_dict=True)

            if f_show_stats:
                torch.cuda.synchronize()
                t3 = time.time()
                cur_peak_alloc = torch.cuda.max_memory_allocated()
                peak_alloc = cur_peak_alloc
                t_sum_decode += t3 - t2
                cnt += 1

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if f_show_stats:
        print('Time taken for decode:', t_sum_decode)
        print('Peak memory allocated (GPU) for decode:', peak_alloc / 1024 / 1024 / 1024, 'GB')
        print('num_generated_tokens', cnt)

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def apply_patch_to_generation():
    GenerationMixin._sample = patched_sample