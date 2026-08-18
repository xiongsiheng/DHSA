#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

method="${ATTENTION_METHOD:-DHSA_vs_optimized}"
density="${DENSITY:-$(dhsa_default_density "${method}")}"
checkpoint="${PREDICTOR_CHECKPOINT:-$(dhsa_default_checkpoint qwen bf16 "${method}")}"
dhsa_attention_args "${method}" "${density}" "${checkpoint}"

dhsa_run_python run_longbench.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_dir "${LONG_BENCH_DATA_DIR:-data/LongBench}" \
    --model_max_length "${MODEL_MAX_LENGTH:-32768}" \
    "${DHSA_ATTENTION_ARGS[@]}" \
    --eval_batch_size "${BATCH_SIZE:-1}" \
    --save_dir "${RESULTS_DIR:-results_longbench}" \
    --report-latency \
    --verbose \
    "$@"
