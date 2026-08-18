#!/usr/bin/env bash

DHSA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DHSA_REPO_ROOT="$(cd "${DHSA_SCRIPT_DIR}/.." && pwd)"
DHSA_PYTHON="${DHSA_PYTHON:-python}"

dhsa_default_density() {
    if [[ "$1" == "full" ]]; then
        printf '1.0\n'
    else
        printf '0.125\n'
    fi
}

dhsa_default_load_mode() {
    case "$1" in
        qwen)
            printf 'bf16\n'
            ;;
        llama)
            printf '4bit\n'
            ;;
        *)
            echo "Unsupported MODEL_KEY: $1" >&2
            return 2
            ;;
    esac
}

dhsa_default_checkpoint() {
    local model_key="$1"
    local load_mode="$2"
    local method="$3"

    case "${model_key}:${load_mode}:${method}" in
        qwen:bf16:DHSA_learned_topK_static)
            printf 'checkpoints/DHSA-Qwen2.5-3B-Instruct-BF16/predictor_static.pt\n'
            ;;
        qwen:bf16:DHSA_learned_topK_dynamic)
            printf 'checkpoints/DHSA-Qwen2.5-3B-Instruct-BF16/predictor_dynamic.pt\n'
            ;;
        llama:bf16:DHSA_learned_topK_static)
            printf 'checkpoints/DHSA-Llama-3.1-8B-Instruct-BF16/predictor_static.pt\n'
            ;;
        llama:bf16:DHSA_learned_topK_dynamic)
            printf 'checkpoints/DHSA-Llama-3.1-8B-Instruct-BF16/predictor_dynamic.pt\n'
            ;;
        llama:4bit:DHSA_learned_topK_static)
            printf 'checkpoints/DHSA-Llama-3.1-8B-Instruct-NF4/predictor_static.pt\n'
            ;;
        llama:4bit:DHSA_learned_topK_dynamic)
            printf 'checkpoints/DHSA-Llama-3.1-8B-Instruct-NF4/predictor_dynamic.pt\n'
            ;;
    esac
}

dhsa_attention_args() {
    local method="$1"
    local density="$2"
    local checkpoint="${3:-}"

    DHSA_PREDICTOR_ARGS=()
    case "${method}" in
        full)
            if [[ "${density}" != "1" && "${density}" != "1.0" ]]; then
                echo "full attention requires DENSITY=1.0" >&2
                return 2
            fi
            checkpoint=""
            ;;
        DHSA_vs_optimized|DHSA_vsb_memory_efficient)
            checkpoint=""
            ;;
        DHSA_learned_topK_static|DHSA_learned_topK_dynamic)
            if [[ -z "${checkpoint}" ]]; then
                echo "${method} requires PREDICTOR_CHECKPOINT" >&2
                return 2
            fi
            if [[ "${checkpoint}" != /* ]]; then
                checkpoint="${DHSA_REPO_ROOT}/${checkpoint}"
            fi
            if [[ ! -f "${checkpoint}" ]]; then
                echo "Predictor checkpoint not found: ${checkpoint}" >&2
                return 1
            fi
            ;;
        *)
            echo "Unsupported ATTENTION_METHOD: ${method}" >&2
            return 2
            ;;
    esac

    DHSA_ATTENTION_ARGS=(
        --sparsity-mask "${method}"
        --density "${density}"
        --q-block-size 128
        --k-block-size 32
    )
    if [[ -n "${checkpoint}" ]]; then
        DHSA_PREDICTOR_ARGS=(--predictor-checkpoint "${checkpoint}")
        DHSA_ATTENTION_ARGS+=("${DHSA_PREDICTOR_ARGS[@]}")
    fi
}

dhsa_run_python() {
    local gpu_id="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
    (
        cd "${DHSA_REPO_ROOT}"
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export PYTHONPATH="${DHSA_REPO_ROOT}/Block-Sparse-Attention${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${DHSA_PYTHON}" "$@"
    )
}
