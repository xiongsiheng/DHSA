"""
Pipeline for Latency Test.
"""

from pathlib import Path
from typing import Union

import torch
import torch._dynamo
import transformers
import time

# Disable dynamo for stability
torch._dynamo.config.disable = True

from utils.config import *  # NEEDLE, DATA_DIR, RETRIEVAL_QUESTION, CONTEXT_LENGTHS_LATENCY_TEST, etc.
from utils.monkeypatch import replace_gemma, _configure_attention_layers
from boundary_predictor.models import BoundarySimilarityAttn

# argparse-based flags
from utils.flags_config import get_args


class LLMLatencyTester:
    """
    A comprehensive testing framework for evaluating LLM latency and performance
    across different context lengths and attention mechanisms.
    """
    def __init__(
        self,
        model_name: str = "gemma_2_2b",
        needle: str = NEEDLE,
        haystack_dir: str = DATA_DIR,
        retrieval_question: str = RETRIEVAL_QUESTION,
        context_lengths: list[int] | None = None,
        attn_implementation: str = "sdpa",
        method: str = "full",
        budget_decode: int = -1,
        budget_prefill: int = -1,
        block_size: int = 64,
        dhsa_boundary_window_size: int = 4,
        dhsa_boundary_ratio_theta: float = 1.1,
        dhsa_chunk_beta: int = 8,
        dhsa_use_nms: bool = True,
        dhsa_nms_window_size: int = 8,
        dhsa_chunk_repre_pooling: str = "avgpool_norm",
        dhsa_share_boundaries: bool = False,
        dhsa_share_sparsity_masks: bool = False,
        dhsa_ckpt_path: str | None = None,
        dhsa_predictor_channel_in: int = 1024,
        dhsa_predictor_hidden_size: int = 256,
        dhsa_predictor_num_heads: int = 8,
        dhsa_predictor_use_window_pool: bool = False,
        max_new_tokens: int = 100,
        final_context_length_buffer: int = 200,
        print_ongoing_status: bool = True,
        num_iterations: int = NUM_ITERATIONS_LATENCY_TEST
    ):
        # Testing parameters
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question

        self.context_lengths = context_lengths or CONTEXT_LENGTHS_LATENCY_TEST
        self.num_iterations = num_iterations
        self.final_context_length_buffer = final_context_length_buffer
        self.max_new_tokens = max_new_tokens

        # Model parameters
        self.model_name = model_name
        self.model_provider = MODEL_PROVIDERS[Model[model_name]]
        self.attn_implementation = AttnImplementation[attn_implementation].value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.method = SparseAttnMethod[method].value
        self.budget_decode = budget_decode
        self.budget_prefill = budget_prefill
        self.block_size = block_size

        # DHSA parameters
        self.dhsa_boundary_window_size = dhsa_boundary_window_size
        self.dhsa_boundary_ratio_theta = dhsa_boundary_ratio_theta
        self.dhsa_chunk_beta = dhsa_chunk_beta
        self.dhsa_use_nms = dhsa_use_nms
        self.dhsa_nms_window_size = dhsa_nms_window_size
        self.dhsa_chunk_repre_pooling = dhsa_chunk_repre_pooling
        self.dhsa_share_boundaries = dhsa_share_boundaries
        self.dhsa_share_sparsity_masks = dhsa_share_sparsity_masks

        # Boundary predictor configuration
        self.dhsa_ckpt_path = dhsa_ckpt_path
        self.dhsa_predictor_channel_in = dhsa_predictor_channel_in
        self.dhsa_predictor_hidden_size = dhsa_predictor_hidden_size
        self.dhsa_predictor_num_heads = dhsa_predictor_num_heads
        self.dhsa_predictor_use_window_pool = dhsa_predictor_use_window_pool

        # Output configuration
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        model_path = MODEL_PATHS.get(Model[self.model_name], self.model_name)

        if self.print_ongoing_status:
            print(f"Loading model from: {model_path}")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        if self.method == SparseAttnMethod.full.value:
            self.model = self._load_full_model(model_path)
        else:
            self.model = self._load_compressed_model(model_path)

    def _load_full_model(self, model_path: str):
        """Load the full model without optimizations."""
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            use_cache=True
        ).to(self.device)
        model.eval()
        return model

    def _load_compressed_model(self, model_path: str):
        """Load model with compressed attention."""
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

        # Boundary predictor (optional; only for DHSA)
        boundary_predictor = None
        if self.method == SparseAttnMethod.dhsa.value and self.dhsa_ckpt_path is not None:
            boundary_predictor = BoundarySimilarityAttn(
                channel_in=self.dhsa_predictor_channel_in,
                window_size=self.dhsa_boundary_window_size,
                d_h=self.dhsa_predictor_hidden_size,
                heads=self.dhsa_predictor_num_heads,
                window_pool=self.dhsa_predictor_use_window_pool
            ).to(self.device)
            state_dict = torch.load(self.dhsa_ckpt_path, map_location="cpu")
            boundary_predictor.load_state_dict(state_dict)
            boundary_predictor.eval()

        # Configure attention layers
        _configure_attention_layers(self, model, predictor=boundary_predictor)
        return model

    def run_test(self):
        """Run the complete test suite."""
        if self.print_ongoing_status:
            print(f"Starting test with {len(self.context_lengths)} context lengths")
            print(f"Context lengths: {self.context_lengths}")

        total_tests = len(self.context_lengths) * self.num_iterations
        current_test = 0

        for context_length in self.context_lengths:
            if self.print_ongoing_status:
                print(f"\nTesting context length: {context_length}")

            for iteration in range(self.num_iterations):
                current_test += 1

                if self.print_ongoing_status:
                    print(f"  Iteration {iteration + 1}/{self.num_iterations} "
                          f"(Test {current_test}/{total_tests})")

                # Test at 0% depth (needle at beginning)
                self._evaluate_and_log(context_length, 0)

    def _generate_prompt(self, context: str) -> Union[str, list[dict]]:
        """Generate the appropriate prompt format based on model provider."""
        return PROMPT_TEMPLATE.format(
            context=context, retrieval_question=self.retrieval_question
        )

    def _evaluate_and_log(self, context_length: int, depth_percent: float):
        """Evaluate the model and log results."""
        context = self._generate_context(context_length, depth_percent)

        prompt = self._generate_prompt(context)
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded_prompt["input_ids"].to(self.model.device)

        instruction = INSTRUCTION_TEMPLATE.format(retrieval_question=self.retrieval_question)
        instruction_tokens = len(self.tokenizer(instruction, return_tensors="pt")["input_ids"][0])

        if self.method != SparseAttnMethod.full.value:
            for layer in self.model.model.layers:
                layer.self_attn.instruction_tokens = instruction_tokens
                layer.self_attn.act_kv_seq_len = 0

        test_start_time = time.time()
        # Deterministic greedy decoding
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                output_attentions=False,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=None
            )
        test_end_time = time.time()

        print('Latency (s): ', test_end_time - test_start_time)

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=False
        ).strip()

        result = {
            "model": self.model_name,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "needle": self.needle,
            "model_response": response
        }
        self.testing_results.append(result)

    def _generate_context(
        self,
        context_length: int,
        depth_percent: float
    ) -> str:
        """Generate context with needle inserted at specified depth."""
        base_context = self._read_context_files()
        trimmed_context = self._encode_and_trim(base_context, context_length)
        final_context = self._insert_needle(trimmed_context, depth_percent, context_length)
        return final_context

    def _read_context_files(self) -> str:
        """Read and concatenate all context files."""
        context = ""
        max_context_length = max(self.context_lengths)
        haystack = Path(self.haystack_dir)
        files = sorted(haystack.glob("*.txt"))

        if not files:
            print(f"Warning: No .txt files found in {haystack.resolve()}")

        # Reuse files cyclically until enough tokens are collected
        while self._get_context_length_in_tokens(context) < max_context_length:
            for fp in files:
                try:
                    context += fp.read_text(encoding="utf-8") + "\n"
                except Exception as e:
                    print(f"Warning: Could not read {fp}: {e}")
                    continue

                if self._get_context_length_in_tokens(context) >= max_context_length:
                    break

            # If still no files, break to avoid infinite loop
            if not files:
                break

        return context

    def _encode_text_to_tokens(self, text: str) -> list[int]:
        """Encode text to tokens based on model provider."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _insert_needle(
        self,
        context: str,
        depth_percent: float,
        context_length: int
    ) -> str:
        """Insert needle into context at specified depth percentage."""
        needle_tokens = self._encode_text_to_tokens(self.needle)
        context_tokens = self._encode_text_to_tokens(context)

        # Account for buffer
        effective_context_length = max(0, context_length - self.final_context_length_buffer)

        # Trim context if necessary
        if len(context_tokens) + len(needle_tokens) > effective_context_length:
            keep = max(0, effective_context_length - len(needle_tokens))
            context_tokens = context_tokens[:keep]

        # Insert needle
        if depth_percent == 100:
            new_context_tokens = context_tokens + needle_tokens
        else:
            insertion_point = int(len(context_tokens) * (depth_percent / 100))
            insertion_point = self._find_sentence_boundary(context_tokens, insertion_point)

            # if self.print_ongoing_status:
            #     print(f"Inserting needle at position {insertion_point}")

            new_context_tokens = (
                context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]
            )

        return self._decode_tokens(new_context_tokens)

    def _find_sentence_boundary(self, tokens: list[int], insertion_point: int) -> int:
        """Find the nearest sentence boundary (period) before insertion point."""
        period_tokens = PERIOD_TOKENS.get(self.model_provider, self._encode_text_to_tokens("."))

        while insertion_point > 0 and tokens[insertion_point - 1] not in period_tokens:
            insertion_point -= 1

        return insertion_point

    def _get_context_length_in_tokens(self, context: str) -> int:
        """Get the length of context in tokens."""
        if self.model_provider in ["Gemma2", "Gemma3"]:
            return len(self.tokenizer.encode(context, add_special_tokens=False))
        else:
            return len(self.tokenizer.encode(context))

    def _decode_tokens(self, tokens: list[int], context_length: int | None = None) -> str:
        """Decode tokens back to text."""
        if context_length is not None:
            tokens = tokens[:context_length]

        if self.model_provider in ["Gemma2", "Gemma3"]:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            return self.tokenizer.decode(tokens)

    def _encode_and_trim(self, context: str, context_length: int) -> str:
        """Encode context and trim to specified length."""
        tokens = self._encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self._decode_tokens(tokens, context_length)
        return context

    def get_results(self) -> list[dict]:
        """Get all test results."""
        return self.testing_results

    def start_test(self):
        """Start the testing process."""
        try:
            self.run_test()
            if self.print_ongoing_status:
                print(f"\nTesting complete! Generated {len(self.testing_results)} results.")
        except Exception as e:
            print(f"Error during testing: {e}")
            raise


def main():
    args = get_args()

    tester = LLMLatencyTester(
        model_name=args.model_name,
        attn_implementation=args.attn_implementation,
        method=args.method,
        budget_prefill=args.budget_prefill,
        budget_decode=args.budget_decode,
        block_size=args.block_size,
        dhsa_boundary_window_size=args.dhsa_boundary_window_size,
        dhsa_use_nms=args.dhsa_use_nms,
        dhsa_nms_window_size=args.dhsa_nms_window_size,
        dhsa_ckpt_path=args.dhsa_ckpt_path,
        dhsa_chunk_repre_pooling=args.dhsa_chunk_repre_pooling,
        dhsa_share_boundaries=args.dhsa_share_boundaries,
        dhsa_share_sparsity_masks=args.dhsa_share_sparsity_masks,
        max_new_tokens=args.latency_test_max_new_tokens,
        num_iterations=args.latency_test_num_iterations,
        print_ongoing_status=not args.latency_test_quiet,
    )

    tester.start_test()


if __name__ == "__main__":
    main()