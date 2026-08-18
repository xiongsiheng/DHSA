#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

model_key="${MODEL_KEY:-llama}"
method="${ATTENTION_METHOD:-DHSA_learned_topK_static}"
density="${DENSITY:-0.125}"
load_mode="$(dhsa_default_load_mode "${model_key}")"
checkpoint="${PREDICTOR_CHECKPOINT:-$(dhsa_default_checkpoint "${model_key}" "${load_mode}" "${method}")}"
context_lengths="${CONTEXT_LENGTHS:-32768,65536,131072}"
output_json="${OUTPUT_JSON:-results_latency/${model_key}_${method}_d${density}.json}"

dhsa_attention_args "${method}" "${density}" "${checkpoint}"
dhsa_run_python measure_latency.py \
    --model-key "${model_key}" \
    --attention-method "${method}" \
    --density "${density}" \
    --context-lengths "${context_lengths}" \
    --q-block-size 128 \
    --k-block-size 32 \
    --warmup "${WARMUP:-2}" \
    --iters "${ITERS:-5}" \
    "${DHSA_PREDICTOR_ARGS[@]}" \
    --output-json "${output_json}"
