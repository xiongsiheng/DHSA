"""
LongBench Evaluation Script

This script evaluates language models on the LongBench dataset, which tests
long-context understanding across multiple tasks and domains.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import tqdm
import torch
import transformers

# Disable torch dynamo for compatibility
torch._dynamo.config.disable = True

from utils.config import *  # MODEL_MAX_PROMPT_LENGTHS, LongBench_* constants, etc.
from utils.monkeypatch import replace_gemma, _configure_attention_layers
from boundary_predictor.models import BoundarySimilarityAttn

# argparse-based flags
from utils.flags_config import get_args


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model_max_length(model_name: str) -> int:
    """Get the maximum sequence length for a given model."""
    model_path = model_name.lower()
    for key, max_len in MODEL_MAX_PROMPT_LENGTHS.items():
        if key.value.lower() in model_path:
            return max_len
    return 8192  # Default fallback


def load_test_data(
    data_file: str,
    dataset: str,
    max_examples: int | None = None,
    sample_method: str = "firstk"
) -> list[dict]:
    """Load and preprocess test data."""
    test_data = []
    input_max_len = 0
    data_path = Path(data_file)

    with data_path.open("rt", encoding="utf-8") as fp:
        for line in fp:
            example = json.loads(line)

            # Format instruction using template
            example["instruction"] = LongBench_INSTRUCTION_PROMPTS[dataset].format(**example)

            # Track maximum input length
            length = example["length"]
            if length > input_max_len:
                input_max_len = length

            # Format prompt using template
            template = LongBench_TASK_PROMPTS[dataset]
            prompt = template.format(**example)
            example["prompt"] = prompt

            test_data.append(example)

    print(f"Max input length: {input_max_len}")

    # Sample data if needed
    if max_examples and len(test_data) > max_examples:
        if sample_method == SampleMethod.random.name:
            test_data = random.sample(test_data, max_examples)
        elif sample_method == SampleMethod.firstk.name:
            test_data = test_data[:max_examples]
    return test_data


def generate_batch_outputs(
    model,
    tokenizer,
    batch_prompts: list[str],
    instructions: list[str],
    model_max_len: int,
    output_max_len: int,
    args
) -> list[str]:
    """Generate outputs for a batch of prompts."""
    # Tokenize prompts
    tokenized_prompts = tokenizer(
        batch_prompts,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_prompts = tokenized_prompts.to(device)
    batch_input_ids = tokenized_prompts.input_ids

    # Truncate if necessary (truncate to last model_max_len tokens)
    if batch_input_ids.shape[1] > model_max_len:
        prompt = tokenizer.decode(batch_input_ids[0][-model_max_len:], skip_special_tokens=True)
        tokenized_prompts = tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True
        ).to(device)
        batch_input_ids = tokenized_prompts.input_ids

    # Per-sample instruction lengths for DHSA/top-k/block-sparse to inform attention masks
    instruction_tokens = [len(tokenizer(instruction, return_tensors="pt")["input_ids"][0]) for instruction in instructions]
    if args.method != SparseAttnMethod.full.name:
        for layer in model.model.layers:
            layer.self_attn.instruction_tokens = instruction_tokens[0]

    context_length = batch_input_ids.shape[-1]

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": output_max_len,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 1.0,
        "min_length": context_length + 1,
        "eos_token_id": [tokenizer.eos_token_id],
        "output_attentions": False,
    }

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(**tokenized_prompts, **gen_kwargs)

    # Decode outputs
    batch_outputs = tokenizer.batch_decode(
        [output[context_length:] for output in outputs],
        skip_special_tokens=True
    )

    # print(batch_outputs)
    return batch_outputs


def reset_kv_cache(model, method: str) -> None:
    """Reset KV cache for streaming methods."""
    if method != SparseAttnMethod.full.name:
        layers = len(model.model.layers)
        for i in range(layers):
            model.model.layers[i].self_attn.act_kv_seq_len = 0


def evaluate_dataset(model, tokenizer, args, dataset: str) -> None:
    """Evaluate the model on a specific dataset."""
    print(f"Evaluating full attention on {dataset}")

    # Load test data
    data_file = str(Path(LongBench_DATA_DIR) / f"{dataset}.jsonl")
    test_data = load_test_data(
        data_file, dataset,
        args.longbench_max_num_examples,
        args.longbench_sample_method
    )

    # Extract data components
    data_components = {
        "prompts": [ex["prompt"] for ex in test_data],
        "inputs": [ex["input"] for ex in test_data],
        "contexts": [ex["context"] for ex in test_data],
        "instructions": [ex["instruction"] for ex in test_data],
        "answers": [ex["answers"] for ex in test_data],
        "lengths": [ex["length"] for ex in test_data],
        "datasets": [ex["dataset"] for ex in test_data],
        "languages": [ex["language"] for ex in test_data],
        "all_classes": [ex["all_classes"] for ex in test_data],
        "ids": [ex["_id"] for ex in test_data]
    }

    # Setup output file
    output_dir = Path(LongBench_RESULTS_DIR) / args.longbench_save_dir / args.model_name / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == SparseAttnMethod.full.name:
        output_file = "full.json"
    elif args.method in [SparseAttnMethod.topk.name, SparseAttnMethod.blocksparse.name, SparseAttnMethod.dhsa.name]:
        output_file = f"{args.method}_budget_prefill_{args.budget_prefill}.json"
    else:
        output_file = f"{args.method}_budget_decode_{args.budget_decode}.json"
    output_path = output_dir / output_file

    # Get model parameters
    model_max_len = get_model_max_length(args.model_name)
    output_max_len = LongBench_DATASET_MAX_LENGTHS[dataset]

    # Process in batches
    with output_path.open("wt", encoding="utf-8") as fout:
        for i in tqdm.tqdm(range(0, len(data_components["prompts"]), args.longbench_eval_batch_size)):
            # Get batch data
            batch_data = {key: values[i:i + args.longbench_eval_batch_size] for key, values in data_components.items()}

            # Reset KV cache if needed
            reset_kv_cache(model, args.method)

            # Generate outputs
            batch_outputs = generate_batch_outputs(
                model, tokenizer,
                batch_data["prompts"],
                batch_data["instructions"],
                model_max_len, output_max_len, args
            )

            # Save results
            for j in range(len(batch_outputs)):
                # Ensure we don't go out of bounds
                if j < len(batch_data["prompts"]):
                    result = {
                        "prompt": batch_data["prompts"][j],
                        "input": batch_data["inputs"][j],
                        "context": batch_data["contexts"][j],
                        "answers": batch_data["answers"][j],
                        "pred": batch_outputs[j],
                        "length": batch_data["lengths"][j],
                        "dataset": batch_data["datasets"][j],
                        "language": batch_data["languages"][j],
                        "all_classes": batch_data["all_classes"][j],
                        "_id": batch_data["ids"][j]
                    }
                    fout.write(json.dumps(result) + "\n")

            # Clear GPU memory
            torch.cuda.empty_cache()


def setup_model_and_tokenizer(args, use_cache=True, use_fast_tokenizer=True):
    """Setup model and tokenizer."""
    # Get model path
    model_path = MODEL_PATHS.get(Model[args.model_name], args.model_name)

    if MODEL_PROVIDERS[Model[args.model_name]] in ["Gemma2", "Gemma3"]:
        replace_gemma(args.method.lower())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=use_cache,
        attn_implementation=args.attn_implementation
    ).to(device)

    boundary_predictor = None
    # For DHSA, if set boundary predictor as None, the model will automatically
    # use attention to label boundaries (similar to training).
    if args.method == SparseAttnMethod.dhsa.name and args.dhsa_ckpt_path is not None:
        boundary_predictor = BoundarySimilarityAttn(
            channel_in=args.dhsa_predictor_channel_in,
            window_size=args.dhsa_boundary_window_size,
            d_h=args.dhsa_predictor_hidden_size,
            heads=args.dhsa_predictor_num_heads,
            window_pool=args.dhsa_predictor_use_window_pool
        ).to(device)
        state_dict = torch.load(args.dhsa_ckpt_path, map_location="cpu")
        boundary_predictor.load_state_dict(state_dict)
        boundary_predictor.eval()

    # Configure model layers
    if args.method != SparseAttnMethod.full.name:
        _configure_attention_layers(args, model, predictor=boundary_predictor)

    model.eval()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast_tokenizer,
        padding_side="left"
    )

    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def main():
    args = get_args()

    # Set seed
    set_seed(args.longbench_seed)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)

    # Evaluate on all datasets
    for idx, dataset in enumerate(LongBench_DATASETS):
        print(f"Processing dataset {dataset} ({idx + 1}/{len(LongBench_DATASETS)})")
        evaluate_dataset(model, tokenizer, args, dataset)

    print("Evaluation completed!")


if __name__ == "__main__":
    main()