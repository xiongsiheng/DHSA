#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

folder_path="${1:-${RESULTS_DIR:-results_NIAH}}"
shift $(( $# > 0 ? 1 : 0 ))

dhsa_run_python visualize_needle_in_haystack.py \
    --folder_path "${folder_path}" \
    --model_name "${MODEL_LABEL:-llama-3.1-8b-instruct}" \
    --method "${ATTENTION_METHOD:-DHSA_vs_optimized}" \
    --density "${DENSITY:-0.125}" \
    "$@"
