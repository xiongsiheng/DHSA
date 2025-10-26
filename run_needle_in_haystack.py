"""
Pipeline for Needle in a Haystack Test (NIAH)
A standard evaluation framework for long context retrieval.
"""

import os
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

# Disable torch dynamo for stability
torch._dynamo.config.disable = True

from utils.config import *  # NEEDLE, DATA_DIR, RETRIEVAL_QUESTION, PERIOD_TOKENS, RESULTS_DIR, etc.
from utils.monkeypatch import replace_gemma, _configure_attention_layers
from boundary_predictor.models import BoundarySimilarityAttn

# argparse-based flags
from utils.flags_config import get_args


class LLMNeedleHaystackTester:
    """
    A comprehensive testing framework for evaluating LLM performance on
    needle-in-haystack tasks across depths and context lengths.
    """
    def __init__(
        self,
        needle: str = NEEDLE,
        haystack_dir: str = DATA_DIR,
        retrieval_question: str = RETRIEVAL_QUESTION,
        context_lengths_min: int | None = None,
        context_lengths_max: int | None = None,
        context_lengths: list[int] | None = None,
        document_depth_percent_min: int = 0,
        document_depth_percent_max: int = 100,
        document_depth_percent_intervals: int = 10,
        document_depth_percents: list[float] | None = None,
        document_depth_percent_interval_type: str = "linear",
        model_name: str = "gemma_2_2b",
        save_results: bool = True,
        save_contexts: bool = True,
        final_context_length_buffer: int = 200,
        print_ongoing_status: bool = True,
        step: int = 100,
        attn_implementation: str = "sdpa",
        method: str = "full",
        budget_prefill: int = -1,
        budget_decode: int = -1,
        block_size: int = 64,
        boundary_window_size: int = 4,
        boundary_ratio_theta: float = 1.1,
        chunk_beta: int = 8,
        use_nms: bool = True,
        nms_window_size: int = 8,
        loop_times: int = 4,
        chunk_repre_pooling: str = "avgpool_norm",
        share_boundaries: bool = False,
        share_sparsity_masks: bool = False,
        ckpt_path: str | None = None,
        predictor_channel_in: int = 1024,
        predictor_hidden_size: int = 256,
        predictor_num_heads: int = 8,
        predictor_use_window_pool: bool = False
    ):
        self._validate_inputs(needle, haystack_dir, retrieval_question)

        # Core configuration
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.save_results = save_results
        self.save_contexts = save_contexts
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results: list[dict[str, Any]] = []

        # Method configuration
        self.step = step
        self.attn_implementation = AttnImplementation[attn_implementation].value
        self.method = SparseAttnMethod[method].value
        self.budget_decode = budget_decode
        self.budget_prefill = budget_prefill
        self.block_size = block_size
        self.boundary_window_size = boundary_window_size
        self.boundary_ratio_theta = boundary_ratio_theta
        self.chunk_beta = chunk_beta
        self.use_nms = use_nms
        self.nms_window_size = nms_window_size
        self.loop_times = loop_times
        self.chunk_repre_pooling = chunk_repre_pooling
        self.share_boundaries = share_boundaries
        self.share_sparsity_masks = share_sparsity_masks

        # Boundary predictor configuration
        self.ckpt_path = ckpt_path
        self.predictor_channel_in = predictor_channel_in
        self.predictor_hidden_size = predictor_hidden_size
        self.predictor_num_heads = predictor_num_heads
        self.predictor_use_window_pool = predictor_use_window_pool

        # Model configuration
        self.model_name = model_name
        self.model_provider = MODEL_PROVIDERS[Model[model_name]]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base_name = os.path.basename(model_name)
        if self.method == SparseAttnMethod.full.value:
            self.model_version = f"{base_name}_full"
        elif self.method in [SparseAttnMethod.topk.value, SparseAttnMethod.blocksparse.value, SparseAttnMethod.dhsa.value]:
            self.model_version = f"{base_name}_{method}_budget_prefill_{budget_prefill}"
        else:
            self.model_version = f"{base_name}_{method}_budget_decode_{budget_decode}"

        # Context and depth configuration
        self.context_lengths = self._setup_context_lengths(
            context_lengths, context_lengths_min, context_lengths_max
        )
        self.document_depth_percents = self._setup_depth_percents(
            document_depth_percents,
            document_depth_percent_min,
            document_depth_percent_max,
            document_depth_percent_intervals,
            document_depth_percent_interval_type
        )

        # Initialize model and tokenizer
        self._initialize_model()

    def _validate_inputs(self, needle: str, haystack_dir: str, retrieval_question: str) -> None:
        if not all([needle, haystack_dir, retrieval_question]):
            raise ValueError("Needle, haystack_dir, and retrieval_question must be provided.")

    def _setup_context_lengths(
        self,
        context_lengths: list[int] | None,
        min_len: int | None,
        max_len: int | None
    ) -> np.ndarray:
        if context_lengths is None:
            if min_len is None or max_len is None:
                raise ValueError("Either context_lengths or min/max lengths must be provided.")
            return np.arange(min_len, max_len + 1, step=self.step)
        return np.array(context_lengths)

    def _setup_depth_percents(
        self,
        depth_percents: list[float] | None,
        min_percent: int,
        max_percent: int,
        intervals: int,
        interval_type: str
    ) -> np.ndarray:
        if depth_percents is None:
            if interval_type == "linear":
                return np.round(
                    np.linspace(min_percent, max_percent, num=intervals, endpoint=True)
                ).astype(int)
            elif interval_type == "sigmoid":
                return np.array([self._logistic_transform(x) for x in np.linspace(min_percent, max_percent, intervals)])
            else:
                raise ValueError("interval_type must be linear or sigmoid")
        return np.array(depth_percents)

    def _logistic_transform(self, x: float, L: float = 100, x0: float = 50, k: float = 0.1) -> float:
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def _initialize_model(self) -> None:
        model_path = MODEL_PATHS.get(Model[self.model_name], self.model_name)
        print(f"Loading model from: {model_path}")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        if self.method == SparseAttnMethod.full.value:
            self.model = self._load_full_model(model_path)
        else:
            self.model = self._load_compressed_model(model_path)

    def _load_full_model(self, model_path: str):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            use_cache=True
        ).to(self.device)
        model.eval()
        return model

    def _load_compressed_model(self, model_path: str):
        # Apply monkey patches for specific model providers
        if self.model_provider in ["Gemma2", "Gemma3"]:
            replace_gemma(self.method.lower())
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            use_cache=True
        ).to(self.device)
        model.eval()

        # Boundary predictor (optional)
        boundary_predictor = None
        if self.method == SparseAttnMethod.dhsa.value and self.ckpt_path is not None:
            boundary_predictor = BoundarySimilarityAttn(
                channel_in=self.predictor_channel_in,
                window_size=self.boundary_window_size,
                d_h=self.predictor_hidden_size,
                heads=self.predictor_num_heads,
                window_pool=self.predictor_use_window_pool
            ).to(self.device)
            state_dict = torch.load(self.ckpt_path, map_location="cpu")
            boundary_predictor.load_state_dict(state_dict)
            boundary_predictor.eval()

        # Configure attention layers
        _configure_attention_layers(self, model, predictor=boundary_predictor)
        return model

    def generate_prompt(self, context: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, retrieval_question=self.retrieval_question)

    def run_test(self, args) -> None:
        for context_length in self.context_lengths:
            if context_length < args.NIAH_s_len or context_length > args.NIAH_e_len:
                continue
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, float(depth_percent))

    def evaluate_and_log(self, context_length: int, depth_percent: float) -> None:
        context = self.generate_context(context_length, depth_percent)
        prompt = self.generate_prompt(context)

        test_start_time = time.time()
        response = self._run_inference(prompt)
        test_end_time = time.time()

        results = {
            "model": self.model_name,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "needle": self.needle,
            "model_response": response,
            "test_duration_seconds": test_end_time - test_start_time,
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            self._print_test_summary(results)

        if self.save_contexts or self.save_results:
            self._save_test_data(results, context, context_length, depth_percent)

    def _run_inference(self, prompt: str) -> str:
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded_prompt["input_ids"].to(self.model.device)

        instruction = INSTRUCTION_TEMPLATE.format(retrieval_question=self.retrieval_question)
        instruction_tokens = len(self.tokenizer(instruction, return_tensors="pt")["input_ids"][0])

        if self.method != SparseAttnMethod.full.value:
            for layer in self.model.model.layers:
                layer.self_attn.instruction_tokens = instruction_tokens
                layer.self_attn.act_kv_seq_len = 0

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                output_attentions=False,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=None
            )

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        return response

    def _print_test_summary(self, results: dict[str, Any]) -> None:
        print("-- Test Summary --")
        print(f"Duration: {results['test_duration_seconds']:.1f} seconds")
        print(f"Context: {results['context_length']} tokens")
        print(f"Depth: {results['depth_percent']}%")
        print(f"Response: {results['model_response']}\n")

    def _save_test_data(
        self,
        results: dict[str, Any],
        context: str,
        context_length: int,
        depth_percent: float
    ) -> None:
        context_file_location = f"len_{context_length}_depth_{int(depth_percent*100)}"

        # Save context
        if self.save_contexts:
            self._save_context(context, context_file_location)

        # Save results
        if self.save_results:
            self._save_results(results, context_file_location)

    def _save_context(self, context: str, file_location: str) -> None:
        out_dir = Path(RESULTS_DIR) / "contexts" / self.model_version
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{file_location}_context.txt").write_text(context, encoding="utf-8")

    def _save_results(self, results: dict[str, Any], file_location: str) -> None:
        out_dir = Path(RESULTS_DIR) / "results" / self.model_version
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"{file_location}_results.json"
        print(f"Writing results to: {results_path}")
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    def generate_context(self, context_length: int, depth_percent: float) -> str:
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needle(context, depth_percent, context_length)
        return context

    def read_context_files(self) -> str:
        context = ""
        max_context_length = int(max(self.context_lengths))
        folder = Path(self.haystack_dir)
        files = sorted(folder.glob("*.txt"))

        if not files:
            print(f"Warning: No .txt files found in {folder.resolve()}")

        # Reuse files cyclically until enough tokens are collected or no files
        while self.get_context_length_in_tokens(context) < max_context_length and files:
            for fp in files:
                try:
                    context += fp.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not read {fp}: {e}")
                if self.get_context_length_in_tokens(context) >= max_context_length:
                    break

        return context

    def encode_text_to_tokens(self, text: str) -> list[int]:
        if self.model_provider in ["Mistral", "LLaMA3", "Gemma2", "Gemma3"]:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: list[int], context_length: int | None = None) -> str:
        if self.model_provider in ["Mistral", "LLaMA3", "Gemma2", "Gemma3"]:
            return self.tokenizer.decode(tokens[:context_length], skip_special_tokens=True)
        else:
            return self.tokenizer.decode(tokens[:context_length])

    def get_context_length_in_tokens(self, context: str) -> int:
        if self.model_provider in ["Mistral", "LLaMA3", "Gemma2", "Gemma3"]:
            return len(self.tokenizer.encode(context, add_special_tokens=False))
        else:
            return len(self.tokenizer.encode(context))

    def encode_and_trim(self, context: str, context_length: int) -> str:
        tokens = self.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def insert_needle(
        self,
        context: str,
        depth_percent: float,
        context_length: int
    ) -> str:
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        effective_len = max(0, context_length - self.final_context_length_buffer)

        if len(tokens_context) + len(tokens_needle) > effective_len:
            keep = max(0, effective_len - len(tokens_needle))
            tokens_context = tokens_context[:keep]

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]

            period_tokens = PERIOD_TOKENS.get(self.model_provider, self.encode_text_to_tokens("."))

            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                if insertion_point <= 0:
                    insertion_point = 0
                    break
                tokens_new_context = tokens_context[:insertion_point]

            if self.print_ongoing_status:
                print(f"Insertion point: {insertion_point}")

            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        return self.decode_tokens(tokens_new_context)

    def get_results(self) -> list[dict[str, Any]]:
        return self.testing_results

    def print_start_test_summary(self) -> None:
        print("\n" + "=" * 60)
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(f"- Context Lengths: {len(self.context_lengths)}, Min: {int(min(self.context_lengths))}, Max: {int(max(self.context_lengths))}")
        print(f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needle: {self.needle.strip()}")
        print("=" * 60 + "\n")

    def start_test(self, args) -> None:
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args)


def main():
    args = get_args()

    tester = LLMNeedleHaystackTester(
        model_name=args.model_name,
        context_lengths_min=args.NIAH_s_len,
        context_lengths_max=args.NIAH_e_len,
        save_contexts=True,
        save_results=True,
        step=args.NIAH_step,
        attn_implementation=args.attn_implementation,
        method=args.method,
        budget_prefill=args.budget_prefill,
        budget_decode=args.budget_decode,
        block_size=args.block_size,
        boundary_window_size=args.dhsa_boundary_window_size,
        use_nms=args.dhsa_use_nms,
        nms_window_size=args.dhsa_nms_window_size,
        ckpt_path=args.dhsa_ckpt_path,
        predictor_channel_in=args.dhsa_predictor_channel_in,
        predictor_hidden_size=args.dhsa_predictor_hidden_size,
        predictor_num_heads=args.dhsa_predictor_num_heads,
        predictor_use_window_pool=args.dhsa_predictor_use_window_pool,
    )

    tester.start_test(args)


if __name__ == "__main__":
    main()