#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

method="${ATTENTION_METHOD:-DHSA_vs_optimized}"
density="${DENSITY:-$(dhsa_default_density "${method}")}"
checkpoint="${PREDICTOR_CHECKPOINT:-$(dhsa_default_checkpoint qwen bf16 "${method}")}"
dhsa_attention_args "${method}" "${density}" "${checkpoint}"

dhsa_run_python run_ruler.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_dir "${RULER_DATA_DIR:-data/RULER}" \
    "${DHSA_ATTENTION_ARGS[@]}" \
    --max_num_examples "${MAX_EXAMPLES:-50}" \
    --eval_batch_size "${BATCH_SIZE:-1}" \
    --report-latency \
    --save_dir "${RESULTS_DIR:-results_ruler}" \
    "$@"
