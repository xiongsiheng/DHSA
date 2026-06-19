#!/usr/bin/env python3
"""
Run LongBench evaluation with DHSA.

Loads a causal LM and tokenizer, applies the DHSA patch, evaluates all LongBench datasets, 
and saves JSONL predictions with optional latency metrics.
"""

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.helper import FirstTokenTimer, parse_bool, set_seed
from utils.monkeypatch import SPARSITY_MASKS, configure_DHSA, validate_sparse_config


LOCAL_BLOCK_SPARSE_METHOD = "DHSA"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_MODEL_MAX_LENGTH = 16384

# Disable torch dynamo for compatibility.
torch._dynamo.config.disable = True

# Dataset configurations
DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]

OUTPUT_MAX_LENGTHS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}


TASK_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer "
        "and do not output any other words. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
    "lcc": (
        "Please complete the code given below.\n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n{context}{input}Next line of code:\n"
    )
}

def load_test_data(data_file: str, dataset: str, max_examples: Optional[int] = None, 
                   sample_method: str = "topk") -> List[Dict[str, Any]]:
    """Load and preprocess test data."""
    test_data = []
    input_max_len = 0
    
    with open(data_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            example = json.loads(line)
            
            # Track maximum input length
            length = example["length"]
            if length > input_max_len:
                input_max_len = length
            
            # Format prompt using template
            template = TASK_PROMPTS[dataset]
            prompt = template.format(**example)
            example["prompt"] = prompt
            
            test_data.append(example)
    
    print(f"Max input length: {input_max_len}")
    
    # Sample data if needed
    if max_examples and len(test_data) > max_examples:
        if sample_method == "random":
            test_data = random.sample(test_data, max_examples)
        elif sample_method == "topk":
            test_data = test_data[:max_examples]
    
    return test_data


def _first_token_id(token_id) -> Optional[int]:
    """Return the first concrete token id from an int/list token-id field."""
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        for item in token_id:
            if item is not None:
                return int(item)
        return None
    return int(token_id)


def _resolve_eos_token_id(tokenizer, model):
    """Resolve EOS id from tokenizer, generation config, or model config."""
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
    raise ValueError(
        "Could not resolve eos_token_id from tokenizer, model.generation_config, or model.config."
    )


def configure_generation_special_tokens(tokenizer, model) -> None:
    """Ensure tokenizer/model generation configs have non-None EOS and PAD ids."""
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

    # tokenizer.eos_token_id must be a single int, even if generation can accept a list.
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


def setup_model_and_tokenizer(args: argparse.Namespace):
    if args.attn_implementation != "flash_attention_2":
        raise ValueError(
            "run_longbench.py requires "
            "--attn_implementation flash_attention_2."
        )

    density = 1.0 - args.sparsity_ratio if args.sparsity_ratio is not None else args.density
    q_block_size = args.q_block_size if args.q_block_size is not None else args.block_size
    k_block_size = args.k_block_size if args.k_block_size is not None else args.block_size
    validate_sparse_config(density, q_block_size, k_block_size)

    model_path = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left",
    )

    if not args.use_quant:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=args.attn_implementation,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
            use_cache=args.use_cache,
        )

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


def align_tokenized_batch(tokenized_prompts, model_max_len: int, args: argparse.Namespace):
    input_ids = tokenized_prompts.input_ids
    attention_mask = tokenized_prompts.attention_mask
    max_len = (model_max_len // args.DHSA_alignment) * args.DHSA_alignment
    if max_len <= 0:
        raise ValueError("model_max_len is shorter than one alignment unit.")

    if input_ids.shape[-1] > max_len:
        input_ids = input_ids[:, -max_len:]
        attention_mask = attention_mask[:, -max_len:]

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


def generate_batch_outputs(
    model,
    tokenizer,
    batch_prompts: List[str],
    model_max_len: int,
    output_max_len: int,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Optional[float]]]:
    tokenized_prompts = tokenizer(
        batch_prompts,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
    )
    tokenized_prompts = align_tokenized_batch(tokenized_prompts, model_max_len, args).to("cuda")
    context_length = tokenized_prompts.input_ids.shape[-1]

    if args.verbose:
        print(
            "Context length:",
            context_length,
        )

    gen_kwargs = {
        "max_new_tokens": output_max_len,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 1.0,
        "min_length": context_length + 1,
        "eos_token_id": model.generation_config.eos_token_id,
        "pad_token_id": model.generation_config.pad_token_id,
        "output_attentions": args.output_attentions,
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

    with torch.no_grad():
        if timer is not None:
            timer.start()
            total_start = timer.start_time
        outputs = model.generate(**tokenized_prompts, **gen_kwargs)
        if timer is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_metrics["ttft_ms"] = timer.ttft_ms
            latency_metrics["generation_latency_ms"] = (time.perf_counter() - total_start) * 1000.0

    batch_outputs = tokenizer.batch_decode(
        [output[context_length:] for output in outputs],
        skip_special_tokens=True,
    )

    if args.verbose and batch_outputs:
        print("output:", batch_outputs[0])
    if args.verbose and args.report_latency:
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


def build_output_file(args: argparse.Namespace, dataset: str) -> str:
    model_name = args.model_name.split("/")[-1]
    save_root = args.save_dir or "results_longbench"
    output_dir = os.path.join(save_root, model_name, dataset)
    os.makedirs(output_dir, exist_ok=True)

    filename = (
        f"{args.sparsity_mask}"
        f"_density_{args.DHSA_density}"
        f"_qbs{args.DHSA_q_block_size}"
        f"_kbs{args.DHSA_k_block_size}.json"
    )
    return os.path.join(output_dir, filename)


def evaluate_dataset(model, tokenizer, args: argparse.Namespace, dataset: str) -> None:
    print(
        f"Evaluating {LOCAL_BLOCK_SPARSE_METHOD} "
        f"density={args.DHSA_density} "
        f"mask={args.sparsity_mask} on {dataset}"
    )

    data_file = f"data/LongBench/{dataset}.jsonl"
    test_data = load_test_data(data_file, dataset, args.max_num_examples, args.sample_method)
    data_components = {
        "prompts": [ex["prompt"] for ex in test_data],
        "inputs": [ex["input"] for ex in test_data],
        "contexts": [ex["context"] for ex in test_data],
        "answers": [ex["answers"] for ex in test_data],
        "lengths": [ex["length"] for ex in test_data],
        "datasets": [ex["dataset"] for ex in test_data],
        "languages": [ex["language"] for ex in test_data],
        "all_classes": [ex["all_classes"] for ex in test_data],
        "ids": [ex["_id"] for ex in test_data],
    }

    model_max_len = args.model_max_length
    output_max_len = OUTPUT_MAX_LENGTHS[dataset]
    output_file = build_output_file(args, dataset)

    with open(output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(data_components["prompts"]), args.eval_batch_size)):
            batch_data = {
                key: values[i : i + args.eval_batch_size]
                for key, values in data_components.items()
            }
            batch_outputs, latency_metrics = generate_batch_outputs(
                model,
                tokenizer,
                batch_data["prompts"],
                model_max_len,
                output_max_len,
                args,
            )

            for j, output in enumerate(batch_outputs):
                if j >= len(batch_data["prompts"]):
                    continue
                result = {
                    "prompt": batch_data["prompts"][j],
                    "input": batch_data["inputs"][j],
                    "context": batch_data["contexts"][j],
                    "answers": batch_data["answers"][j],
                    "pred": output,
                    "length": batch_data["lengths"][j],
                    "dataset": batch_data["datasets"][j],
                    "language": batch_data["languages"][j],
                    "all_classes": batch_data["all_classes"][j],
                    "_id": batch_data["ids"][j],
                }
                if args.report_latency:
                    result.update(latency_metrics)
                fout.write(json.dumps(result) + "\n")

            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LongBench with the DHSA patch"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="", help="Save directory")

    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="Use fast tokenizer")
    parser.add_argument("--output_attentions", type=bool, default=False, help="Output attentions")
    parser.add_argument("--use_cache", type=parse_bool, default=True, help="Use KV cache during generation")
    parser.add_argument("--no-use-cache", "--no_use_cache", dest="use_cache", action="store_false", help="Disable KV cache to reduce memory at the cost of slower decode")
    parser.add_argument("--report-latency", "--report_latency", dest="report_latency", action="store_true", help="Measure TTFT and total generation latency")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch context length and generated output.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation; DHSA requires flash_attention_2",
    )
    parser.add_argument("--use_quant", action="store_true", help="Use quantization")

    parser.add_argument("--max_num_examples", type=int, default=None, help="Max examples per task")
    parser.add_argument(
        "--sample_method",
        type=str,
        default="topk",
        choices=["random", "topk"],
        help="Sampling method",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=DEFAULT_MODEL_MAX_LENGTH,
        help=f"Maximum input context length before sparse-block alignment/truncation (default: {DEFAULT_MODEL_MAX_LENGTH}).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.125,
        help="Fraction of key blocks kept per query block.",
    )
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

    for idx, dataset in enumerate(DATASETS):
        print(f"Processing dataset {dataset} ({idx + 1}/{len(DATASETS)})")
        evaluate_dataset(model, tokenizer, args, dataset)

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
