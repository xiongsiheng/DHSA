#!/usr/bin/env python3
"""
Run RULER evaluation with DHSA.

Loads a causal LM and tokenizer, applies the DHSA patch, evaluates RULER datasets across 
context lengths, and saves JSONL predictions with optional latency metrics.
"""

import argparse
import glob
import json
import math
import os
import time
from typing import List, Optional

import torch
from tqdm import tqdm
from utils.helper import FirstTokenTimer, infer_model_provider, parse_bool, set_seed
from utils.monkeypatch import SPARSITY_MASKS, configure_DHSA, validate_sparse_config

LOCAL_BLOCK_SPARSE_METHOD = "DHSA"
DEFAULT_BLOCK_SIZE = 128
CONTEXT_LENGTH_LIST = [8192, 16384, 32768, 49152]
DATASETS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "cwe",
    "fwe",
    "vt",
    "qa_1",
    "qa_2",
]
DATASET2MAXLEN = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64,
    "qa_1": 32,
    "qa_2": 32,
}

def build_chat(prompt: str) -> str:
    return f"[INST] {prompt} [/INST]"

def resolve_data_file(context_length, dataset, base_dir=""):
    candidates = []

    if base_dir:
        candidates.extend(
            [
                os.path.join(base_dir, str(context_length), f"{dataset}.jsonl"),
                os.path.join(base_dir, dataset, "validation.jsonl"),
                os.path.join(base_dir, dataset, "test.jsonl"),
            ]
        )

    candidates.append(os.path.join("data", "RULER", str(context_length), f"{dataset}.jsonl"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    official_patterns = [
        os.path.join(
            "data",
            "RULER_official",
            "benchmark_root",
            "*",
            "synthetic",
            str(context_length),
            "data",
            dataset,
            "validation.jsonl",
        ),
        os.path.join(
            "data",
            "RULER_official",
            "benchmark_root",
            "*",
            "synthetic",
            str(context_length),
            "data",
            dataset,
            "test.jsonl",
        ),
    ]

    for pattern in official_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]

    searched_paths = candidates + official_patterns
    raise FileNotFoundError(
        f"Could not find data file for dataset '{dataset}' at context length {context_length}. "
        f"Searched: {searched_paths}"
    )

def parse_int_list(value: str) -> List[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.") from exc

def parse_str_list(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of dataset names.")
    return values

def align_tokenized_batch(tokenized_prompts, args: argparse.Namespace):
    input_ids = tokenized_prompts.input_ids
    attention_mask = tokenized_prompts.attention_mask
    seq_len = input_ids.shape[-1]
    aligned_len = (seq_len // args.DHSA_alignment) * args.DHSA_alignment
    if aligned_len <= 0:
        raise ValueError(
            "Prompt is shorter than one sparse block after tokenization; "
            "increase the input length or reduce the block sizes."
        )
    if aligned_len != seq_len:
        input_ids = input_ids[:, -aligned_len:]
        attention_mask = attention_mask[:, -aligned_len:]

    tokenized_prompts["input_ids"] = input_ids.contiguous()
    tokenized_prompts["attention_mask"] = attention_mask.contiguous()
    return tokenized_prompts

def build_output_file(args: argparse.Namespace) -> str:
    model_name = args.model_name.split("/")[-1]
    output_dir = os.path.join(
        args.save_dir,
        f"{model_name}_{args.sparsity_mask}"
        f"_density_{args.DHSA_density}"
        f"_qbs{args.DHSA_q_block_size}"
        f"_kbs{args.DHSA_k_block_size}",
        str(args.context_length),
        args.dataset,
    )
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{LOCAL_BLOCK_SPARSE_METHOD}.json")

def load_ruler_data(args: argparse.Namespace):
    test_data = []
    input_max_len = 0
    with open(args.data_file, encoding="utf-8") as fp:
        for line in fp:
            example = json.loads(line)
            length = example["length"]
            input_max_len = max(input_max_len, length)

            prompt = example["input"]
            if "llama2" in args.model_name.lower():
                prompt = build_chat(prompt)
            example["prompt"] = prompt
            test_data.append(example)

    print(f"Max Length is {input_max_len}")
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            import random

            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[: args.max_num_examples]
    return test_data

def _first_token_id(token_id) -> Optional[int]:
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        for item in token_id:
            if item is not None:
                return int(item)
        return None
    return int(token_id)

def _resolve_eos_token_id(tokenizer, model):
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
    ):
        if token_id is None:
            continue
        if isinstance(token_id, (list, tuple)):
            cleaned = [int(item) for item in token_id if item is not None]
            if cleaned:
                return cleaned if len(cleaned) > 1 else cleaned[0]
        else:
            return int(token_id)
    raise ValueError("Could not resolve eos_token_id from tokenizer, model.generation_config, or model.config.")

def configure_generation_special_tokens(tokenizer, model) -> None:
    eos_token_id = _resolve_eos_token_id(tokenizer, model)
    eos_token_id_for_pad = _first_token_id(eos_token_id)

    pad_token_id = _first_token_id(getattr(tokenizer, "pad_token_id", None))
    if pad_token_id is None:
        pad_token_id = _first_token_id(getattr(model.generation_config, "pad_token_id", None))
    if pad_token_id is None:
        pad_token_id = _first_token_id(getattr(model.config, "pad_token_id", None))
    if pad_token_id is None:
        pad_token_id = eos_token_id_for_pad
    if pad_token_id is None:
        raise ValueError("Could not resolve pad_token_id.")

    if getattr(tokenizer, "eos_token_id", None) is None:
        tokenizer.eos_token_id = eos_token_id_for_pad

    if getattr(tokenizer, "pad_token_id", None) is None:
        pad_token = None
        try:
            pad_token = tokenizer.convert_ids_to_tokens(pad_token_id)
        except Exception:
            pad_token = None
        if pad_token is None:
            pad_token = getattr(tokenizer, "eos_token", None)
        if pad_token is not None:
            tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = pad_token_id

    model.generation_config.eos_token_id = eos_token_id
    model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = eos_token_id
    model.config.pad_token_id = pad_token_id

def generate_batch_outputs(model, tokenizer, batch_prompts: List[str], output_max_len: int, args: argparse.Namespace):
    tokenized_prompts = tokenizer(
        batch_prompts,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
    )
    tokenized_prompts = align_tokenized_batch(tokenized_prompts, args).to("cuda")
    context_length = tokenized_prompts.input_ids.shape[-1]

    print(
        "Context length:",
        context_length,
        f"(q_block_size={args.DHSA_q_block_size}, "
        f"k_block_size={args.DHSA_k_block_size})",
    )

    gen_kwargs = {
        "output_attentions": args.output_attentions,
        "max_new_tokens": output_max_len,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 1.0,
        "min_length": context_length + 1,
        "eos_token_id": model.generation_config.eos_token_id,
        "pad_token_id": model.generation_config.pad_token_id,
        "use_cache": args.use_cache,
    }

    latency_metrics = {
        "ttft_ms": None,
        "generation_latency_ms": None,
    }
    timer = None
    if args.report_latency:
        timer = FirstTokenTimer(context_length)
        gen_kwargs["streamer"] = timer

    with torch.inference_mode():
        if timer is not None:
            timer.start()
            total_start = timer.start_time
        output = model.generate(**tokenized_prompts, **gen_kwargs)
        if timer is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_metrics["ttft_ms"] = timer.ttft_ms
            latency_metrics["generation_latency_ms"] = (time.perf_counter() - total_start) * 1000.0

    batch_outputs = tokenizer.batch_decode(
        [row[context_length:] for row in output],
        skip_special_tokens=True,
    )

    print("output:", batch_outputs[0])
    if args.report_latency:
        ttft_text = (
            "n/a"
            if latency_metrics["ttft_ms"] is None
            else f"{latency_metrics['ttft_ms']:.3f} ms"
        )
        total_text = (
            "n/a"
            if latency_metrics["generation_latency_ms"] is None
            else f"{latency_metrics['generation_latency_ms']:.3f} ms"
        )
        print(f"TTFT: {ttft_text}")
        print(f"Total generation latency: {total_text}")

    return batch_outputs, latency_metrics

def evaluate_dataset(model, tokenizer, args: argparse.Namespace) -> None:
    print("Loading data...")
    test_data = load_ruler_data(args)
    output_max_len = args.max_new_tokens or DATASET2MAXLEN[args.dataset]
    output_file = build_output_file(args)

    prompt_list = [example["prompt"] for example in test_data]
    input_list = [example["input"] for example in test_data]
    outputs_list = [example["outputs"] for example in test_data]
    length_list = [example["length"] for example in test_data]

    with open(output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompt_list), args.eval_batch_size)):
            batch_prompts = prompt_list[i : i + args.eval_batch_size]
            batch_inputs = input_list[i : i + args.eval_batch_size]
            batch_answers = outputs_list[i : i + args.eval_batch_size]
            batch_lengths = length_list[i : i + args.eval_batch_size]

            batch_generations, latency_metrics = generate_batch_outputs(
                model,
                tokenizer,
                batch_prompts,
                output_max_len,
                args,
            )

            for j, generation in enumerate(batch_generations):
                example = {
                    "prompt": batch_prompts[j],
                    "input": batch_inputs[j],
                    "answers": batch_answers[j],
                    "pred": generation,
                    "length": batch_lengths[j],
                }
                if args.report_latency:
                    example.update(latency_metrics)
                fout.write(json.dumps(example) + "\n")

            torch.cuda.empty_cache()

def setup_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if args.attn_implementation != "flash_attention_2":
        raise ValueError(
            "run_ruler.py requires "
            "--attn_implementation flash_attention_2."
        )

    density = 1.0 - args.sparsity_ratio if args.sparsity_ratio is not None else args.density
    q_block_size = args.q_block_size if args.q_block_size is not None else args.block_size
    k_block_size = args.k_block_size if args.k_block_size is not None else args.block_size
    validate_sparse_config(density, q_block_size, k_block_size)

    model_provider = args.model_provider or infer_model_provider(args.model_name)
    supported_providers = {"LLaMA3", "Qwen2.5"}
    if model_provider not in supported_providers:
        raise ValueError(
            "The DHSA patch currently supports "
            f"{', '.join(sorted(supported_providers))}; got {model_provider}."
        )
    args.model_provider = model_provider

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=args.use_fast_tokenizer,
        padding_side="left",
        trust_remote_code=True,
    )

    if args.use_quant:
        print(f"Loading model with 4-bit quantization!")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
    else:
        print(f"Loading model with BF16!")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )

    tokenizer.padding_side = "left"
    configure_generation_special_tokens(tokenizer, model)

    configure_DHSA(
        model,
        density=density,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        sparsity_mask=args.sparsity_mask,
        chunk_calculation=False,
    )

    args.DHSA_density = float(density)
    args.DHSA_sparsity = 1.0 - float(density)
    args.DHSA_q_block_size = int(q_block_size)
    args.DHSA_k_block_size = int(k_block_size)
    args.DHSA_alignment = math.lcm(q_block_size, k_block_size)

    model.eval()
    return model, tokenizer

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RULER with the DHSA patch"
    )

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, required=True, help="Model name or Hugging Face/local model path.")
    parser.add_argument(
        "--model_provider",
        type=str,
        default=None,
        choices=["LLaMA3", "Qwen2.5"],
        help="Tokenizer/provider family. Defaults to inferring from --model_name.",
    )
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--use_cache", type=parse_bool, default=True, help="Use KV cache during generation")
    parser.add_argument("--report-latency", "--report_latency", dest="report_latency", action="store_true", help="Measure TTFT and total generation latency")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation; DHSA requires flash_attention_2",
    )
    parser.add_argument("--use_quant", action="store_true", help="Use 4-bit BitsAndBytes model loading")

    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="maximum number of new tokens to generate.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")

    parser.add_argument(
        "--context-lengths",
        "--context_lengths",
        dest="context_lengths",
        type=parse_int_list,
        default=CONTEXT_LENGTH_LIST,
        help="Comma-separated RULER context lengths to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        type=parse_str_list,
        default=DATASETS,
        help="Comma-separated RULER datasets to evaluate.",
    )
    parser.add_argument("--density", type=float, default=0.125, help="Fraction of key blocks kept per query block.")
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=None,
        help="Optional compatibility alias. If set, density becomes 1 - sparsity_ratio.",
    )
    parser.add_argument(
        "--sparsity-mask",
        "--sparsity_mask",
        dest="sparsity_mask",
        choices=SPARSITY_MASKS,
        help="Sparse block selection function from the DHSA patch.",
    )
    parser.add_argument("--q-block-size", "--q_block_size", dest="q_block_size", type=int, default=DEFAULT_BLOCK_SIZE, help="Query sparse block size.")
    parser.add_argument("--k-block-size", "--k_block_size", dest="k_block_size", type=int, default=DEFAULT_BLOCK_SIZE, help="Key sparse block size.")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = setup_model_and_tokenizer(args)

    for context_length in args.context_lengths:
        for idx, dataset in enumerate(args.datasets):
            print(
                f"Working on context length {context_length}, "
                f"density: {args.DHSA_density}, "
                f"mask: {args.sparsity_mask}, dataset: {dataset} - {idx}/{len(args.datasets)}"
            )
            args.context_length = context_length
            args.dataset = dataset
            args.data_file = resolve_data_file(context_length, args.dataset, args.base_dir)
            evaluate_dataset(model, tokenizer, args)

    print("Evaluation completed!")

if __name__ == "__main__":
    main()
