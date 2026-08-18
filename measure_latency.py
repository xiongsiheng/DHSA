#!/usr/bin/env python3
"""Benchmark one attention layer's mask construction and attention kernel."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from flash_attn import flash_attn_func

from utils.monkeypatch import (
    ATTENTION_METHODS,
    load_DHSA_patch_module,
    validate_sparse_config,
)


MODEL_CONFIGS = {
    "qwen": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "load_mode": "bf16",
        "num_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
    },
    "llama": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "load_mode": "4bit",
        "num_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument(
        "--attention-method",
        "--scheme",
        dest="attention_method",
        choices=ATTENTION_METHODS,
        required=True,
    )
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument(
        "--context-lengths",
        type=parse_int_list,
        default=[32768, 65536, 131072],
    )
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--k-block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_avg_ms": sum(values) / len(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_min_ms": min(values),
        f"{prefix}_max_ms": max(values),
    }


def benchmark_full(
    q: torch.Tensor,
    k_native: torch.Tensor,
    v_native: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    q_flash = q.transpose(1, 2).contiguous()
    k_flash = k_native.transpose(1, 2).contiguous()
    v_flash = v_native.transpose(1, 2).contiguous()

    def run_once() -> torch.Tensor:
        return flash_attn_func(
            q_flash,
            k_flash,
            v_flash,
            causal=True,
        )

    for _ in range(warmup):
        output = run_once()
        if not torch.isfinite(output).all():
            raise RuntimeError("Full FA2 output contains non-finite values")
        del output

    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    attention_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = run_once()
        end.record()
        torch.cuda.synchronize()
        attention_times.append(elapsed_ms(start, end))
        if not torch.isfinite(output).all():
            raise RuntimeError("Full FA2 output contains non-finite values")
        del output

    peak_bytes = torch.cuda.max_memory_allocated()
    result = {
        **summarize(attention_times, "attention"),
        **summarize(attention_times, "total"),
        "mask_avg_ms": 0.0,
        "mask_median_ms": 0.0,
        "mask_min_ms": 0.0,
        "mask_max_ms": 0.0,
        "baseline_gib": baseline_bytes / 1024**3,
        "peak_gib": peak_bytes / 1024**3,
        "extra_gib": (peak_bytes - baseline_bytes) / 1024**3,
    }
    return result


def make_mask_builder(
    patch_module,
    attention_method: str,
    model_key: str,
    predictor_checkpoint: Path | None,
    density: float,
):
    if attention_method == "DHSA_vs_optimized":
        return (
            patch_module._generate_sparsity_mask_with_vertical_slash_sample32
            if model_key == "qwen"
            else patch_module._generate_sparsity_mask_with_vertical_slash_sample64
        )
    if attention_method == "DHSA_vsb_memory_efficient":
        return patch_module._generate_sparsity_mask_with_vertical_slash_blockwise
    if attention_method not in {
        "DHSA_learned_topK_static",
        "DHSA_learned_topK_dynamic",
    }:
        raise ValueError(
            f"Unsupported sparse attention method: {attention_method}"
        )
    if predictor_checkpoint is None:
        raise ValueError(
            f"{attention_method} requires --predictor-checkpoint"
        )
    expected_variant = (
        "static"
        if attention_method == "DHSA_learned_topK_static"
        else "dynamic"
    )
    patch_module.load_DHSA_topk_predictor(
        predictor_checkpoint,
        "cuda",
        expected_variant=expected_variant,
        density=density,
    )
    return patch_module._generate_sparsity_mask_with_learned_topk


def benchmark_sparse(
    *,
    patch_module,
    mask_builder,
    q: torch.Tensor,
    k_native: torch.Tensor,
    v_native: torch.Tensor,
    density: float,
    q_block_size: int,
    k_block_size: int,
    layer_idx: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    batch_size, num_heads, context_length, head_dim = q.shape
    num_key_value_heads = k_native.shape[1]
    num_key_value_groups = num_heads // num_key_value_heads
    cu_seqlens = torch.tensor(
        [0, context_length],
        device=q.device,
        dtype=torch.int32,
    )
    head_mask_type = torch.ones(
        num_heads,
        device=q.device,
        dtype=torch.int32,
    )
    topk_blocks = max(1, int(density * (context_length // k_block_size)))

    def expand_native_kv(sample: torch.Tensor) -> torch.Tensor:
        return (
            sample.transpose(0, 1)[:, :, None, :]
            .expand(
                context_length,
                num_key_value_heads,
                num_key_value_groups,
                head_dim,
            )
            .reshape(context_length, num_heads, head_dim)
            .contiguous()
        )

    def prepare_sample(
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_unpad = q[batch_idx].transpose(0, 1).contiguous()
        k_unpad = expand_native_kv(k_native[batch_idx])
        v_unpad = expand_native_kv(v_native[batch_idx])
        key_states = k_unpad.transpose(0, 1).contiguous().unsqueeze(0)
        return q_unpad, k_unpad, v_unpad, key_states

    def build_mask(batch_idx: int, key_states: torch.Tensor) -> torch.Tensor:
        return mask_builder(
            query_states=q[batch_idx : batch_idx + 1],
            key_states=key_states,
            topk_blocks=topk_blocks,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            key_states_native=k_native[batch_idx : batch_idx + 1],
            num_key_value_groups=num_key_value_groups,
            layer_idx=layer_idx,
        )

    def run_attention(
        q_unpad: torch.Tensor,
        k_unpad: torch.Tensor,
        v_unpad: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return patch_module._call_block_sparse_attn_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens,
            cu_seqlens,
            head_mask_type,
            None,
            mask,
            context_length,
            context_length,
            0.0,
            deterministic=False,
            softmax_scale=None,
            is_causal=True,
            exact_streaming=False,
            return_attn_probs=False,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
        )

    for _ in range(warmup):
        for batch_idx in range(batch_size):
            q_unpad, k_unpad, v_unpad, key_states = prepare_sample(batch_idx)
            mask = build_mask(batch_idx, key_states)
            output = run_attention(q_unpad, k_unpad, v_unpad, mask)
            if not torch.isfinite(output).all():
                raise RuntimeError("Sparse attention output contains non-finite values")
            del output, mask, q_unpad, k_unpad, v_unpad, key_states

    selected_blocks = 0
    for batch_idx in range(batch_size):
        k_unpad = expand_native_kv(k_native[batch_idx])
        key_states = k_unpad.transpose(0, 1).contiguous().unsqueeze(0)
        mask = build_mask(batch_idx, key_states)
        selected_blocks += int(mask.sum().item())
        del mask, k_unpad, key_states
    num_query_blocks = context_length // q_block_size
    recent = torch.minimum(
        torch.arange(num_query_blocks, device=q.device) * q_block_size
        + q_block_size
        - 1,
        torch.full(
            (num_query_blocks,),
            context_length - 1,
            device=q.device,
        ),
    )
    available = recent // k_block_size + 1
    eligible_blocks = int(available.sum().item()) * num_heads * batch_size

    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    mask_times = []
    attention_times = []
    total_times = []
    for _ in range(iters):
        sample_events = []
        for batch_idx in range(batch_size):
            q_unpad, k_unpad, v_unpad, key_states = prepare_sample(batch_idx)
            mask_start = torch.cuda.Event(enable_timing=True)
            mask_end = torch.cuda.Event(enable_timing=True)
            attention_end = torch.cuda.Event(enable_timing=True)
            mask_start.record()
            mask = build_mask(batch_idx, key_states)
            mask_end.record()
            output = run_attention(q_unpad, k_unpad, v_unpad, mask)
            attention_end.record()
            sample_events.append((mask_start, mask_end, attention_end))
            del output, mask, q_unpad, k_unpad, v_unpad, key_states
        torch.cuda.synchronize()

        mask_time = sum(
            elapsed_ms(start, end) for start, end, _ in sample_events
        )
        attention_time = sum(
            elapsed_ms(end, attention_end)
            for _, end, attention_end in sample_events
        )
        mask_times.append(mask_time)
        attention_times.append(attention_time)
        total_times.append(mask_time + attention_time)

    peak_bytes = torch.cuda.max_memory_allocated()
    return {
        **summarize(mask_times, "mask"),
        **summarize(attention_times, "attention"),
        **summarize(total_times, "total"),
        "selected_blocks": selected_blocks,
        "eligible_blocks": eligible_blocks,
        "realized_causal_density": selected_blocks / eligible_blocks,
        "baseline_gib": baseline_bytes / 1024**3,
        "peak_gib": peak_bytes / 1024**3,
        "extra_gib": (peak_bytes - baseline_bytes) / 1024**3,
        "batch_execution": "sequential_per_sample",
    }


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.attention_method == "full":
        if not math.isclose(args.density, 1.0):
            raise ValueError("Full FA2 must use density 1.0")
    else:
        validate_sparse_config(
            args.density,
            args.q_block_size,
            args.k_block_size,
        )

    print(f"Batch size: {args.batch_size}", flush=True)

    config = MODEL_CONFIGS[args.model_key]
    patch_module = None
    mask_builder = None
    if args.attention_method != "full":
        patch_module = load_DHSA_patch_module()
        mask_builder = make_mask_builder(
            patch_module,
            args.attention_method,
            args.model_key,
            args.predictor_checkpoint,
            args.density,
        )

    results = []
    for context_length in args.context_lengths:
        if context_length % math.lcm(args.q_block_size, args.k_block_size):
            raise ValueError(f"Context {context_length} is not block aligned")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + context_length)
        shape_q = (
            args.batch_size,
            config["num_heads"],
            context_length,
            config["head_dim"],
        )
        shape_kv = (
            args.batch_size,
            config["num_key_value_heads"],
            context_length,
            config["head_dim"],
        )
        q = torch.randn(
            shape_q,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_native = torch.randn(
            shape_kv,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_native = torch.randn(
            shape_kv,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )

        if args.attention_method == "full":
            metrics = benchmark_full(
                q,
                k_native,
                v_native,
                args.warmup,
                args.iters,
            )
        else:
            metrics = benchmark_sparse(
                patch_module=patch_module,
                mask_builder=mask_builder,
                q=q,
                k_native=k_native,
                v_native=v_native,
                density=args.density,
                q_block_size=args.q_block_size,
                k_block_size=args.k_block_size,
                layer_idx=args.layer_idx,
                warmup=args.warmup,
                iters=args.iters,
            )
        metrics["context_length"] = context_length
        results.append(metrics)
        print(
            f"{args.model_key} {args.attention_method} density={args.density:g} "
            f"context={context_length} "
            f"mask={metrics['mask_median_ms']:.3f}ms "
            f"attention={metrics['attention_median_ms']:.3f}ms "
            f"total={metrics['total_median_ms']:.3f}ms "
            f"peak={metrics['peak_gib']:.3f}GiB",
            flush=True,
        )
        del q, k_native, v_native
        torch.cuda.empty_cache()

    report = {
        **config,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "attention_method": args.attention_method,
        "density": args.density,
        "q_block_size": args.q_block_size,
        "k_block_size": args.k_block_size,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "layer_idx": args.layer_idx,
        "predictor_checkpoint": (
            str(args.predictor_checkpoint)
            if args.predictor_checkpoint is not None
            else None
        ),
        "scope": "single_attention_layer_mask_plus_kernel",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
