#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

results_dir="${1:-results_longbench/Llama-3.1-8B-Instruct}"
shift $(( $# > 0 ? 1 : 0 ))

dhsa_run_python eval_longbench.py --results_dir "${results_dir}" "$@"
