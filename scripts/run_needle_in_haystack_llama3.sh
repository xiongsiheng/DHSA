#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

method="${ATTENTION_METHOD:-DHSA_learned_topK_static}"
density="${DENSITY:-$(dhsa_default_density "${method}")}"
checkpoint="${PREDICTOR_CHECKPOINT:-$(dhsa_default_checkpoint llama 4bit "${method}")}"
dhsa_attention_args "${method}" "${density}" "${checkpoint}"

dhsa_run_python run_needle_in_haystack.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --use_quant \
    --max-new-tokens "${MAX_NEW_TOKENS:-15}" \
    "${DHSA_ATTENTION_ARGS[@]}" \
    --save_results \
    --save_contexts \
    --report-latency \
    --use_cache true \
    --haystack_dir "${HAYSTACK_DIR:-data/PaulGrahamEssays}" \
    --save_dir "${RESULTS_DIR:-results_NIAH}" \
    "$@"
