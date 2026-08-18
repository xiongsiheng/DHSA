#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

results_dir="${1:-results_ruler}"
shift $(( $# > 0 ? 1 : 0 ))

dhsa_run_python eval_ruler.py --results_dir "${results_dir}" "$@"
