#!/usr/bin/env python3
"""
Measure kernel-level latency and GPU memory for dense FA2 and DHSA attention.

Loads the model config, generates Q/K/V tensors, and benchmarks
different context lengths, sparse densities, block sizes, and sparsity masks.
"""

import argparse

import torch

from utils.monkeypatch import SPARSITY_MASKS, load_DHSA_patch_module, validate_sparse_config


_block_sparse_patch = None
_call_block_sparse_attn_func = None


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FA2 and block-sparse attention kernels.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sparsity_mask", "--sparsity-mask", default="topk", choices=SPARSITY_MASKS)
    parser.add_argument("--q-block-size", dest="q_block_size", type=int, default=128)
    parser.add_argument("--k-block-size", dest="k_block_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--context_lengths",
        type=parse_int_list,
        default=[8192, 16384, 32768, 65536, 131072],
        help="Comma-separated context lengths.",
    )
    parser.add_argument(
        "--densities",
        type=parse_float_list,
        default=[0.5, 0.25, 0.125, 0.0625],
        help="Comma-separated sparse densities.",
    )
    return parser


def _get_block_sparse_patch_module():
    global _block_sparse_patch, _call_block_sparse_attn_func
    if _block_sparse_patch is None:
        _block_sparse_patch = load_DHSA_patch_module()
        _call_block_sparse_attn_func = _block_sparse_patch._call_block_sparse_attn_func
    return _block_sparse_patch


def _get_sparsity_mask_fn(sparsity_mask: str):
    patch_module = _get_block_sparse_patch_module()
    if sparsity_mask in ["topk", "DHSA_topk"]:
        return patch_module._generate_sparsity_mask
    if sparsity_mask == "DHSA_a":
        return patch_module._generate_sparsity_mask_with_A_shape
    if sparsity_mask == "DHSA_vs":
        return patch_module._generate_sparsity_mask_with_vertical_slash
    if sparsity_mask == "DHSA_vsb":
        return patch_module._generate_sparsity_mask_with_vertical_slash_blockwise
    raise ValueError(f"sparsity_mask must be one of: {', '.join(SPARSITY_MASKS)}")


@torch.no_grad()
def benchmark_kernel_DHSA(
    config,
    seq_len: int = 32768,
    density: float = 0.25,
    q_block_size: int = 128,
    k_block_size: int = 128,
    sparsity_mask: str = "topk",
    iters: int = 20,
    warmup: int = 5,
    device: str | torch.device | None = None,
    batch_size: int = 1,
    use_loop: bool = False,
):
    """
    Benchmark *kernel-level* cost of DHSA, starting
    from Q/K/V of shape (B, H, L, D), including:

      - sparsity mask construction
      - varlen layout conversion
      - call to block_sparse_attn_func

    Q/K/V themselves are assumed to be produced by the shared linear layers and
    are not part of the differential cost (same as dense).
    """
    validate_sparse_config(density, q_block_size, k_block_size)
    _get_block_sparse_patch_module()
    generate_sparsity_mask = _get_sparsity_mask_fn(sparsity_mask)

    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    if use_loop:
        loop_size = batch_size
        B = 1
    else:
        loop_size = 1
        B = batch_size

    H = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dtype = torch.bfloat16  # match your compute dtype

    assert seq_len % q_block_size == 0, "seq_len must be divisible by q_block_size"
    assert seq_len % k_block_size == 0, "seq_len must be divisible by k_block_size"

    def one_call():
        """
        One full DHSA call, starting from Q/K/V:

          Q/K/V -> sparsity mask -> varlen layout -> kernel
        """

        # 1) Block mask construction
        num_k_blocks = seq_len // k_block_size
        topk_blocks = max(1, int(density * num_k_blocks))

        block_mask = generate_sparsity_mask(
            query_states=query_states,
            key_states=key_states,
            topk_blocks=topk_blocks,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
        )  # (B, H, nrow, ncol)

        # 2) Convert to varlen layout
        # (B, H, L, D) -> (B, L, H, D) -> (B*L, H, D)
        q = query_states.permute(0, 2, 1, 3).contiguous()
        k = key_states.permute(0, 2, 1, 3).contiguous()
        v = value_states.permute(0, 2, 1, 3).contiguous()

        B_loc, Lq, H_check, D = q.shape
        assert B_loc == B and H_check == H

        q_unpad = q.reshape(B_loc * Lq, H, D)
        k_unpad = k.reshape(B_loc * Lq, H, D)
        v_unpad = v.reshape(B_loc * Lq, H, D)

        cu_seqlens = torch.arange(
            0, (B_loc + 1) * Lq, step=Lq, dtype=torch.int32, device=device
        )  # (B+1,)

        head_mask_type = torch.ones(H, dtype=torch.int32, device=device)
        p_dropout = 0.0
        is_causal = True

        _ = _call_block_sparse_attn_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens,
            cu_seqlens,
            head_mask_type,
            None,          # streaming_info
            block_mask,    # (B, H, nrow, ncol)
            Lq,            # max_seqlen_q_
            Lq,            # max_seqlen_k_
            p_dropout,
            deterministic=False,
            softmax_scale=None,
            is_causal=is_causal,
            exact_streaming=False,
            return_attn_probs=False,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
        )

    # ------------------- Build random Q/K/V (shared across runs) -------------------
    # (B, H, L, D)
    query_states = torch.randn(B, H, seq_len, head_dim, device=device, dtype=dtype)
    key_states   = torch.randn_like(query_states)
    value_states = torch.randn_like(query_states)

    # ------------------- Warmup (not timed) -------------------
    for _ in range(warmup):
        one_call()

    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    # ------------------- Timed iterations -------------------
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(iters):
        for _ in range(loop_size):
            one_call()
    end.record()

    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters  # or / (iters * loop_size) for per-call

    peak_bytes = torch.cuda.max_memory_allocated()
    extra_bytes = peak_bytes - baseline_bytes

    print(
        f"[DHSA method] seq_len={seq_len}, density={density}, "
        f"batch_size={batch_size}\n"
        f"sparsity_mask={sparsity_mask}, q_block_size={q_block_size}, "
        f"k_block_size={k_block_size}, iters={iters}\n"
        f"  avg latency: {avg_ms:.3f} ms\n"
        f"  extra GPU mem: {extra_bytes / 1024**3:.3f} GB "
        f"(baseline {baseline_bytes / 1024**3:.3f} GB, "
        f"peak {peak_bytes / 1024**3:.3f} GB)"
    )

    return avg_ms, extra_bytes


@torch.no_grad()
def benchmark_kernel_dense_FA2(
    config,
    seq_len: int = 32768,
    iters: int = 20,
    warmup: int = 5,
    device=None,
    batch_size: int = 1,
):
    """
    Pure FlashAttention-2 kernel benchmark.
    Uses flash_attn_func directly, bypassing all HF modules.

    Q/K/V layout: (B, L, H, D)
    CAUSAL mode enabled.
    """
    from flash_attn import flash_attn_func

    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    B = batch_size
    H = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dtype = torch.bfloat16

    # ------------------- Build QKV in FA2 layout: (B, L, H, D) -------------------
    q = torch.randn(B, seq_len, H, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # ------------------- Warmup (no timing) -------------------
    for _ in range(warmup):
        _ = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    # ------------------- Timed iterations -------------------
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    peak = torch.cuda.max_memory_allocated()
    extra = peak - baseline

    print(
        f"[FA2 kernel] seq_len={seq_len}, iters={iters}, batch_size={batch_size}\n"
        f"  avg latency: {avg_ms:.3f} ms\n"
        f"  extra GPU mem: {extra/1024**3:.3f} GB "
        f"(baseline {baseline/1024**3:.3f} GB, "
        f"peak {peak/1024**3:.3f} GB)"
    )

    return avg_ms, extra




def load_config(model_name: str):
    from transformers import AutoConfig

    print(f"Loading config only for {model_name}...")
    return AutoConfig.from_pretrained(model_name, trust_remote_code=True)



def _append_failure(results, method, density, context_length, args, error):
    results.append({
        "method": method,
        "density": density,
        "context_length": context_length,
        "batch_size": args.eval_batch_size,
        "sparsity_mask": getattr(args, "sparsity_mask", ""),
        "q_block_size": getattr(args, "q_block_size", ""),
        "k_block_size": getattr(args, "k_block_size", ""),
        "result": None,
        "error": str(error),
    })


def _safe_empty_cache() -> bool:
    try:
        torch.cuda.empty_cache()
        return True
    except RuntimeError as exc:
        print(f"[WARN] torch.cuda.empty_cache() failed after a CUDA error: {exc}")
        return False


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_batch_size < 1:
        raise ValueError("--eval_batch_size must be >= 1.")

    config = load_config(args.model_name)
    sparse_use_loop = args.eval_batch_size > 1
    if sparse_use_loop:
        print(
            f"Using eval_batch_size={args.eval_batch_size}: FA2 uses one batched attention call; "
            "DHSA uses a simple per-example loop."
        )

    results = []

    print("\n" + "#" * 100)
    print("Running FA2 + DHSA kernel benchmarks")

    abort_benchmarks = False
    for context_length in args.context_lengths:
        if abort_benchmarks:
            break

        print("=" * 100)
        print(f"[CONTEXT] context_length={context_length}")

        try:
            print(f"[RUN] FA2 | context_length={context_length}")
            fa2_ret = benchmark_kernel_dense_FA2(
                config,
                seq_len=context_length,
                iters=args.iters,
                warmup=args.warmup,
                batch_size=args.eval_batch_size,
            )
            results.append({
                "method": "fa2",
                "density": None,
                "context_length": context_length,
                "batch_size": args.eval_batch_size,
                "result": fa2_ret,
            })
        except RuntimeError as e:
            print(f"[FAIL] FA2 failed at context_length={context_length}: {e}")
            _append_failure(results, "fa2", None, context_length, args, e)
            if not _safe_empty_cache():
                abort_benchmarks = True
                break

        if not _safe_empty_cache():
            abort_benchmarks = True
            break

        method = "DHSA"
        q_block_size = args.q_block_size
        k_block_size = args.k_block_size
        sparsity_mask = args.sparsity_mask

        for density in args.densities:
            if abort_benchmarks:
                break
            try:
                print(
                    f"[RUN] {method} | density={density} | "
                    f"context_length={context_length} | sparsity_mask={sparsity_mask} | "
                    f"q_block_size={q_block_size} | k_block_size={k_block_size} | "
                    f"batch_size={args.eval_batch_size} | use_loop={sparse_use_loop}"
                )
                sparse_ret = benchmark_kernel_DHSA(
                    config,
                    seq_len=context_length,
                    density=density,
                    q_block_size=q_block_size,
                    k_block_size=k_block_size,
                    sparsity_mask=sparsity_mask,
                    iters=args.iters,
                    warmup=args.warmup,
                    batch_size=args.eval_batch_size,
                    use_loop=sparse_use_loop,
                )
                results.append({
                    "method": method,
                    "density": density,
                    "context_length": context_length,
                    "batch_size": args.eval_batch_size,
                    "sparsity_mask": sparsity_mask,
                    "q_block_size": q_block_size,
                    "k_block_size": k_block_size,
                    "result": sparse_ret,
                })
            except (RuntimeError, ValueError) as e:
                print(
                    f"[FAIL] {method} failed at density={density}, "
                    f"context_length={context_length}, sparsity_mask={sparsity_mask}, "
                    f"q_block_size={q_block_size}, k_block_size={k_block_size}: {e}"
                )
                _append_failure(results, method, density, context_length, args, e)
                if not _safe_empty_cache():
                    abort_benchmarks = True
                    break

            if not _safe_empty_cache():
                abort_benchmarks = True
                break

    print("\n" + "#" * 100)
    print("method\tdensity\tcontext_length\tbatch_size\tsparsity_mask\tq_block_size\tk_block_size\tresult\terror")
    for item in results:
        print(
            f"{item.get('method', '')}\t"
            f"{item.get('density', '')}\t"
            f"{item.get('context_length', '')}\t"
            f"{item.get('batch_size', '')}\t"
            f"{item.get('sparsity_mask', '')}\t"
            f"{item.get('q_block_size', '')}\t"
            f"{item.get('k_block_size', '')}\t"
            f"{item.get('result', '')}\t"
            f"{item.get('error', '')}"
        )


if __name__ == "__main__":
    main()
