#!/usr/bin/env python3
"""
Run Needle-in-a-Haystack evaluation with DHSA.

Loads a causal LM, inserts a target needle into long haystack contexts at
different depths and lengths, applies the DHSA patch, and records
retrieval scores with optional latency metrics.
"""

import argparse
import math
import time
from typing import Any, Dict, List, Optional
import os
import json
import glob
import re
from datetime import datetime, timezone
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rouge_score import rouge_scorer

from utils.helper import FirstTokenTimer, parse_bool, infer_model_provider, preprocess_text, SimpleQuantizedKVCache
from utils.monkeypatch import (
    SPARSITY_MASKS,
    configure_DHSA as patch_DHSA,
    validate_sparse_config,
)


# Disable torch dynamo for stability
torch._dynamo.config.disable = True

# Initialize ROUGE scorer
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


# Period tokens for different model providers
PERIOD_TOKENS = {
    "LLaMA3": [13],
    "Qwen2.5": [13],
}

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_CONTEXT_LENGTHS = [8192, 16384, 32768, 49152]
DEFAULT_DEPTH_PERCENTS = [0, 11, 22, 33, 44, 56, 67, 78, 89, 100]
LOCAL_BLOCK_SPARSE_METHOD = "DHSA"
DEFAULT_BLOCK_SIZE = 128
NO_SPECIAL_TOKEN_PROVIDERS = {"LLaMA3", "Qwen2.5"}


class LLMNeedleHaystackTester:
    """
    A comprehensive testing framework for evaluating LLM performance on needle-in-haystack tasks.
    
    This class tests how well language models can retrieve specific information (needle) 
    from long contexts (haystack) at various depths and context lengths.
    """

    def __init__(
        self,
        needle: str = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        haystack_dir: str = "data/PaulGrahamEssays",
        retrieval_question: str = "The best thing to do in San Francisco is: ",
        results_version: int = 1,
        context_lengths_min: Optional[int] = None,
        context_lengths_max: Optional[int] = None,
        context_lengths_num_intervals: int = 40,
        context_lengths: Optional[List[int]] = None,
        document_depth_percent_min: int = 0,
        document_depth_percent_max: int = 100,
        document_depth_percent_intervals: int = 10,
        document_depth_percents: Optional[List[float]] = None,
        document_depth_percent_interval_type: str = "linear",
        model_provider: str = "LLaMA3",
        model_name: str = '',
        save_results: bool = True,
        save_contexts: bool = True,
        save_dir: str = "results_needle",
        final_context_length_buffer: int = 200,
        print_ongoing_status: bool = True,
        step: int = 100,
        attn_implementation: str = 'flash_attention_2',
        use_quant: bool = False,
    ):
        """Initialize the Needle Haystack Tester for full-model loading.

        DHSA is applied after construction by
        configure_DHSA().
        """
        self._validate_inputs(needle, haystack_dir, retrieval_question)
        
        # Core configuration
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.save_contexts = save_contexts
        self.save_dir = save_dir
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        
        # Runtime configuration
        self.step = step
        self.attn_implementation = attn_implementation

        # Model configuration
        self.model_name = model_name
        self.use_quant = use_quant
        self.model_version = f"{model_name.split('/')[-1]}_full"

        # Context and depth configuration
        self.context_lengths = self._setup_context_lengths(
            context_lengths, context_lengths_min, context_lengths_max, context_lengths_num_intervals
        )

        self.document_depth_percents = self._setup_depth_percents(
            document_depth_percents, document_depth_percent_min, document_depth_percent_max,
            document_depth_percent_intervals, document_depth_percent_interval_type
        )
        
        # Initialize model and tokenizer
        self._initialize_model()

    def _validate_inputs(self, needle: str, haystack_dir: str, retrieval_question: str) -> None:
        """Validate required inputs."""
        if not all([needle, haystack_dir, retrieval_question]):
            raise ValueError("Needle, haystack_dir, and retrieval_question must be provided.")


    def _setup_context_lengths(
        self, 
        context_lengths: Optional[List[int]], 
        min_len: Optional[int], 
        max_len: Optional[int], 
        num_intervals: int
    ) -> np.ndarray:
        """Setup context length array."""
        if context_lengths is None:
            if min_len is None or max_len is None:
                raise ValueError("Either context_lengths or min/max lengths must be provided.")
            return np.arange(min_len, max_len + 1, step=self.step)
        return np.array(context_lengths)

    def _setup_depth_percents(
        self,
        depth_percents: Optional[List[float]],
        min_percent: int,
        max_percent: int,
        intervals: int,
        interval_type: str
    ) -> np.ndarray:
        """Setup document depth percentages."""
        if depth_percents is None:
            if interval_type == 'linear':
                return np.round(np.linspace(min_percent, max_percent, num=intervals, endpoint=True)).astype(int)
            elif interval_type == 'sigmoid':
                return np.array([self._logistic_transform(x) for x in np.linspace(min_percent, max_percent, intervals)])
            else:
                raise ValueError("interval_type must be 'linear' or 'sigmoid'")
        return np.array(depth_percents)

    def _logistic_transform(self, x: float, L: float = 100, x0: float = 50, k: float = 0.1) -> float:
        """Apply logistic transformation for sigmoid distribution."""
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer."""
        model_path = self.model_name
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = self._load_full_model(model_path)

    def _load_full_model(self, model_path: str) -> AutoModelForCausalLM:
        """Load full attention model."""
        if not self.use_quant:
            print(f"Loading model with BF16!")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.attn_implementation,
                device_map="auto",
                low_cpu_mem_usage=True,
            ).eval()
            self._disable_generation_cache(model)
            return model

        print(f"Loading model with 4-bit quantization!")
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",             # or "fp4"
            bnb_4bit_use_double_quant=True,        # nested quantization, helps save memory
            bnb_4bit_compute_dtype=torch.bfloat16  # compute in bf16 (or fp16 if your GPU lacks bf16)
        )

        # Load model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=self.attn_implementation,
            low_cpu_mem_usage=True,
        ).eval()

        self._disable_generation_cache(model)
        return model

    @staticmethod
    def _disable_generation_cache(model: AutoModelForCausalLM) -> None:
        """Disable Gemma HybridCache setup for short needle-generation probes."""
        model.config.use_cache = False
        if model.generation_config is not None:
            model.generation_config.use_cache = False
            model.generation_config.cache_implementation = None


    def generate_prompt(self, context: str) -> str:
        """Generate prompt for the model."""
        return f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"

    def run_test(self, args: argparse.Namespace) -> None:
        """Run the needle haystack test."""
        for context_length in self.context_lengths:
            if (args.s_len is not None and context_length < args.s_len) or (args.e_len is not None and context_length > args.e_len):
                continue
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, depth_percent)


    def evaluate_and_log(self, context_length: int, depth_percent: float) -> None:
        """Evaluate model performance and log results."""
        # Generate context with needle
        context = self.generate_context(context_length, depth_percent)
        
        # Prepare prompt
        prompt = self.generate_prompt(context)

        # Tokenize and run inference
        test_start_time = time.time()
        response = self._run_inference(prompt)
        test_end_time = time.time()
        
        # Calculate score
        score = self._calculate_score(response)
        
        # Prepare results
        results = {
            'model': self.model_name,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_end_time - test_start_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
        }
        
        self.testing_results.append(results)
        
        # Print status
        if self.print_ongoing_status:
            self._print_test_summary(results)
        
        # Save results
        if self.save_contexts or self.save_results:
            self._save_test_data(results, context, context_length, depth_percent)

    def _run_inference(self, prompt: str) -> str:
        """Run model inference with full attention.

        The DHSA tester overrides this method after the local
        sparse-attention patch is applied.
        """
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded_prompt["input_ids"].to(self.model.device)
        print("input_ids shape:", tuple(input_ids.shape))

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                output_attentions=False,
                max_new_tokens=getattr(self, "max_new_tokens", 30),
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=None,
                use_cache=getattr(self, "use_cache", False),
            )

        return self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def _calculate_score(self, response: str) -> float:
        """Calculate ROUGE score for response."""
        if len(response) == 0:
            return 0.0
        return ROUGE_SCORER.score(self.needle, response)['rouge1'].fmeasure * 10

    def _print_test_summary(self, results: Dict[str, Any]) -> None:
        """Print test summary."""
        print("-- Test Summary --")
        print(f"Duration: {results['test_duration_seconds']:.1f} seconds")
        print(f"Context: {results['context_length']} tokens")
        print(f"Depth: {results['depth_percent']}%")
        print(f"Score: {results['score']}")
        print(f"Response: {results['model_response']}\n")

    def _save_test_data(self, results: Dict[str, Any], context: str, context_length: int, depth_percent: float) -> None:
        """Save test data to files."""
        context_file_location = f"len_{context_length}_depth_{int(depth_percent)}"
        
        # Save context
        if self.save_contexts:
            self._save_context(context, context_file_location)
        
        # Save results
        if self.save_results:
            self._save_results(results, context_file_location)

    def _save_context(self, context: str, file_location: str) -> None:
        """Save context to file."""
        context_dir = os.path.join(self.save_dir, 'contexts', self.model_version)
        os.makedirs(context_dir, exist_ok=True)

        context_path = os.path.join(context_dir, f'{file_location}_context.txt')
        with open(context_path, 'w') as f:
            f.write(context)

    def _save_results(self, results: Dict[str, Any], file_location: str) -> None:
        """Save results to file."""
        results_dir = os.path.join(self.save_dir, 'results', self.model_version)
        os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(results_dir, f'{file_location}_results.json')
        print(f"Writing results to: {results_path}")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def generate_context(self, context_length: int, depth_percent: float) -> str:
        """Generate context with needle inserted at specified depth."""
        # Read background context
        context = self.read_context_files()
        # Trim to desired length
        context = self.encode_and_trim(context, context_length)
        # Insert needle at specified depth
        context = self.insert_needle(context, depth_percent, context_length)
        return context

    def read_context_files(self) -> str:
        """Read context files from haystack directory."""
        context = ""
        max_context_length = max(self.context_lengths)
        
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file_path in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()        # trim leading/trailing blank lines
                    text = preprocess_text(text)
                    context += text + "\n\n"  # Add double newlines for separation
        return context.strip()  # Remove trailing newline

    def encode_text_to_tokens(self, text: str) -> List[int]:
        """Encode text to tokens."""
        if self.model_provider in NO_SPECIAL_TOKEN_PROVIDERS:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """Decode tokens to text."""
        if self.model_provider in NO_SPECIAL_TOKEN_PROVIDERS:
            return self.tokenizer.decode(tokens[:context_length], skip_special_tokens=True)
        else:
            return self.tokenizer.decode(tokens[:context_length])

    def get_context_length_in_tokens(self, context: str) -> int:
        """Get context length in tokens."""
        if self.model_provider in NO_SPECIAL_TOKEN_PROVIDERS:
            return len(self.tokenizer.encode(context, add_special_tokens=False))
        else:
            return len(self.tokenizer.encode(context))

    def encode_and_trim(self, context: str, context_length: int) -> str:
        """Encode context and trim to specified length."""
        tokens = self.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def insert_needle(self, context: str, depth_percent: float, context_length: int) -> str:
        """Insert needle at specified depth percentage."""
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Account for buffer
        context_length -= self.final_context_length_buffer
        
        # Trim context if too long
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]
        
        if depth_percent == 100:
            # Place needle at the end
            tokens_new_context = tokens_context + tokens_needle
            self.needle_boundary = [len(tokens_context), len(tokens_context) + len(tokens_needle)]
        else:
            # Calculate insertion point
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]
            
            # Find sentence boundary (period)
            period_tokens = PERIOD_TOKENS.get(self.model_provider, self.encode_text_to_tokens('.'))
            
            # Move backwards to find sentence boundary
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]
            
            print(f"Insertion point: {insertion_point}")
            
            self.needle_boundary = [insertion_point, insertion_point + len(tokens_needle)]
            # Insert needle
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]
        
        return self.decode_tokens(tokens_new_context)

    def get_results(self) -> List[Dict[str, Any]]:
        """Get test results."""
        return self.testing_results

    def print_start_test_summary(self) -> None:
        """Print test start summary."""
        print("\n" + "="*60)
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needle: {self.needle.strip()}")
        print("="*60 + "\n")

    def start_test(self, args: argparse.Namespace) -> None:
        """Start the needle haystack test."""
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args)


def configure_DHSA(
    tester: "LLMNeedleHaystackTester",
    density: float,
    q_block_size: int,
    k_block_size: int,
    sparsity_mask: str,
    chunk_calculation: bool,
) -> None:
    sparsity = 1.0 - float(density)
    patch_DHSA(
        tester.model,
        density=density,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        sparsity_mask=sparsity_mask,
        chunk_calculation=chunk_calculation,
    )

    tester.method = LOCAL_BLOCK_SPARSE_METHOD
    tester.DHSA_density = float(density)
    tester.DHSA_q_block_size = int(q_block_size)
    tester.DHSA_k_block_size = int(k_block_size)
    tester.DHSA_alignment = math.lcm(q_block_size, k_block_size)
    tester.DHSA_sparsity_mask = sparsity_mask
    tester.model_version = (
        f"{tester.model_name.split('/')[-1]}"
        f"_{sparsity_mask}"
        f"_density_{density}"
        f"_qbs{q_block_size}_kbs{k_block_size}"
    )


def parse_int_list(value: str) -> List[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.") from exc


def normalize_context_lengths(values: List[int]) -> List[int]:
    context_lengths = sorted(set(values))
    if any(value <= 0 for value in context_lengths):
        raise ValueError("All context lengths must be positive.")
    return context_lengths


def normalize_depth_percents(values: List[int]) -> List[int]:
    if any(value < 0 or value > 100 for value in values):
        raise ValueError("Depth percents must be between 0 and 100.")
    return values


def override_tester_schedule(
    tester: "LLMNeedleHaystackTester",
    context_lengths: List[int],
    depth_percents: List[int],
) -> None:
    tester.context_lengths = context_lengths
    tester.document_depth_percents = np.array(depth_percents)


def trim_to_block_multiple(input_ids: torch.Tensor, q_block_size: int, k_block_size: int):
    alignment = math.lcm(q_block_size, k_block_size)
    seq_len = input_ids.shape[-1]
    aligned_len = (seq_len // alignment) * alignment
    if aligned_len == 0:
        return input_ids, False
    if aligned_len != seq_len:
        input_ids = input_ids[:, -aligned_len:]
    return input_ids, True


def encode_prompt_preserving_needle_to_block_multiple(
    tokenizer,
    prompt: str,
    needle: str,
    q_block_size: int,
    k_block_size: int,
):
    book_start = prompt.find("<book>")
    book_end = prompt.rfind("</book>")
    if book_start == -1 or book_end == -1 or book_end <= book_start:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids, can_use_block_sparse = trim_to_block_multiple(input_ids, q_block_size, k_block_size)
        return input_ids, can_use_block_sparse, "left_trim_fallback", 0

    book_content_start = book_start + len("<book>")
    prefix_text = prompt[:book_content_start]
    context_text = prompt[book_content_start:book_end]
    suffix_text = prompt[book_end:]

    needle_text = needle.strip()
    needle_start = context_text.find(needle_text)
    if needle_start == -1:
        before_needle_text = context_text
        needle_text = ""
        after_needle_text = ""
    else:
        needle_end = needle_start + len(needle_text)
        before_needle_text = context_text[:needle_start]
        after_needle_text = context_text[needle_end:]

    prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
    before_needle_ids = tokenizer(before_needle_text, add_special_tokens=False)["input_ids"]
    needle_ids = tokenizer(needle_text, add_special_tokens=False)["input_ids"] if needle_text else []
    after_needle_ids = tokenizer(after_needle_text, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]

    total_len = (
        len(prefix_ids)
        + len(before_needle_ids)
        + len(needle_ids)
        + len(after_needle_ids)
        + len(suffix_ids)
    )
    alignment = math.lcm(q_block_size, k_block_size)
    aligned_len = (total_len // alignment) * alignment
    if aligned_len == 0:
        return torch.tensor([prefix_ids + before_needle_ids + needle_ids + after_needle_ids + suffix_ids]), False, "too_short", 0

    drop_tokens = total_len - aligned_len
    if drop_tokens:
        drop_from_after = min(drop_tokens, len(after_needle_ids))
        if drop_from_after:
            after_needle_ids = after_needle_ids[:-drop_from_after]
        remaining_drop = drop_tokens - drop_from_after
        if remaining_drop:
            if remaining_drop > len(before_needle_ids):
                return torch.tensor([prefix_ids + before_needle_ids + needle_ids + after_needle_ids + suffix_ids]), False, "cannot_preserve_needle", drop_tokens
            before_needle_ids = before_needle_ids[:-remaining_drop]

    input_ids = prefix_ids + before_needle_ids + needle_ids + after_needle_ids + suffix_ids
    return torch.tensor([input_ids], dtype=torch.long), True, "needle_preserving_context_trim", drop_tokens


class DHSATester:
    def _attach_latency_metrics(self, results: Dict) -> None:
        if getattr(self, "report_latency", False):
            results.update(getattr(self, "last_latency_metrics", {}))

    def _print_test_summary(self, results: Dict) -> None:
        self._attach_latency_metrics(results)
        super()._print_test_summary(results)

    def _save_results(self, results: Dict, file_location: str) -> None:
        self._attach_latency_metrics(results)
        super()._save_results(results, file_location)

    def _run_inference(self, prompt: str) -> str:
        input_ids, can_use_block_sparse, alignment_mode, dropped_tokens = encode_prompt_preserving_needle_to_block_multiple(
            self.tokenizer,
            prompt,
            self.needle,
            self.DHSA_q_block_size,
            self.DHSA_k_block_size,
        )
        if not can_use_block_sparse:
            raise ValueError(
                "Prompt is shorter than one sparse block after tokenization; "
                "increase the context length or reduce the block sizes."
            )

        input_ids = input_ids.to(self.model.device)
        print(
            "input_ids shape after block alignment:",
            tuple(input_ids.shape),
            f"(q_block_size={self.DHSA_q_block_size}, "
            f"k_block_size={self.DHSA_k_block_size}, "
            f"mode={alignment_mode}, dropped_tokens={dropped_tokens})",
        )

        gen_kwargs = {
            "output_attentions": False,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 1.0,
            "eos_token_id": None,
            "use_cache": getattr(self, "use_cache", True),
        }
        if getattr(self, "use_cache", True):
            gen_kwargs["past_key_values"] = SimpleQuantizedKVCache(nbits=getattr(self, "nbits", 8))
        latency_metrics = {
            "ttft_ms": None,
            "generation_latency_ms": None,
        }
        timer = None
        if getattr(self, "report_latency", False):
            timer = FirstTokenTimer(input_ids.shape[-1])
            gen_kwargs["streamer"] = timer

        with torch.inference_mode():
            if timer is not None:
                timer.start()
                total_start = timer.start_time
            output_ids = self.model.generate(input_ids, **gen_kwargs)
            if timer is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_metrics["ttft_ms"] = timer.ttft_ms
                latency_metrics["generation_latency_ms"] = (time.perf_counter() - total_start) * 1000.0

        self.last_latency_metrics = latency_metrics
        if getattr(self, "report_latency", False):
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

        return self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Needle-in-a-Haystack with DHSA "
            "implementation from the local Block-Sparse-Attention module."
        )
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name or path")
    parser.add_argument(
        "--model_provider",
        type=str,
        default=None,
        choices=["LLaMA3", "Qwen", "Qwen2", "Qwen2.5"],
        help="Tokenizer/provider family. Defaults to inferring from --model_name.",
    )
    parser.add_argument(
        "--context-lengths",
        type=parse_int_list,
        default=DEFAULT_CONTEXT_LENGTHS,
        help="Comma-separated token lengths to test, for example 8192,16384,32768",
    )
    parser.add_argument(
        "--depth-percents",
        type=parse_int_list,
        default=DEFAULT_DEPTH_PERCENTS,
        help="Comma-separated insertion depths, for example 0,25,50,75,100",
    )
    parser.add_argument("-s", "--s_len", type=int, help="Optional minimum context length filter")
    parser.add_argument("-e", "--e_len", type=int, help="Optional maximum context length filter")
    parser.add_argument("--needle", type=str, default=None, help="Override the default needle string")
    parser.add_argument(
        "--retrieval_question",
        type=str,
        default=None,
        help="Override the default retrieval question",
    )
    parser.add_argument(
        "--haystack_dir",
        type=str,
        default="data/PaulGrahamEssays",
        help="Directory containing haystack text files",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="The DHSA patch requires flash_attention_2.",
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
        choices=SPARSITY_MASKS,
        help="Sparse block selection function from the DHSA patch.",
    )
    parser.add_argument("--q-block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="Query sparse block size.")
    parser.add_argument("--k-block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="Key sparse block size.")
    parser.add_argument(
        "--chunk_calculation",
        action="store_true",
        help="Chunk Llama layernorm, MLP, and attention output projection in the external block-sparse patch.",
    )
    parser.add_argument("--use_cache", type=parse_bool, default=True, help="Use KV cache during generation")
    parser.add_argument("--use_quant", action="store_true", help="Use quantization")
    parser.add_argument("--cache_nbits", type=int, default=8, help="Quantized KV cache bits")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Maximum generated answer tokens")
    parser.add_argument(
        "--report-latency",
        "--report_latency",
        dest="report_latency",
        action="store_true",
        help="Measure TTFT and total generation latency",
    )
    parser.add_argument("--save_results", action="store_true", help="Persist JSON results")
    parser.add_argument("--save_contexts", action="store_true", help="Persist generated contexts")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results_needle",
        help="Directory for saved results and contexts",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser


def build_tester(args: argparse.Namespace) -> "LLMNeedleHaystackTester":
    if args.attn_implementation != "flash_attention_2":
        raise ValueError(
            "run_needle_in_haystack.py "
            "requires --attn_implementation flash_attention_2."
        )

    density = 1.0 - args.sparsity_ratio if args.sparsity_ratio is not None else args.density
    q_block_size = args.q_block_size if args.q_block_size is not None else args.block_size
    k_block_size = args.k_block_size if args.k_block_size is not None else args.block_size
    validate_sparse_config(density, q_block_size, k_block_size)

    needle = args.needle
    retrieval_question = args.retrieval_question
    model_provider = args.model_provider or infer_model_provider(args.model_name)
    tester = LLMNeedleHaystackTester(
        needle=needle if needle is not None else "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        haystack_dir=args.haystack_dir,
        retrieval_question=retrieval_question if retrieval_question is not None else "The best thing to do in San Francisco is: ",
        model_name=args.model_name,
        model_provider=model_provider,
        save_contexts=args.save_contexts,
        save_results=args.save_results,
        save_dir=args.save_dir,
        print_ongoing_status=not args.quiet,
        context_lengths=normalize_context_lengths(args.context_lengths),
        document_depth_percents=normalize_depth_percents(args.depth_percents),
        attn_implementation=args.attn_implementation,
        use_quant=args.use_quant,
    )
    tester.__class__ = type(
        "DHSANeedleHaystackTester",
        (DHSATester, tester.__class__),
        {},
    )
    tester.max_new_tokens = args.max_new_tokens
    tester.use_cache = args.use_cache
    tester.nbits = args.cache_nbits
    tester.report_latency = args.report_latency
    tester.last_latency_metrics = {
        "ttft_ms": None,
        "generation_latency_ms": None,
    }

    configure_DHSA(
        tester,
        density=density,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        sparsity_mask=args.sparsity_mask,
        chunk_calculation=args.chunk_calculation,
    )
    override_tester_schedule(
        tester,
        normalize_context_lengths(args.context_lengths),
        normalize_depth_percents(args.depth_percents),
    )
    return tester


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    tester = build_tester(args)
    tester.start_test(args)


if __name__ == "__main__":
    main()
