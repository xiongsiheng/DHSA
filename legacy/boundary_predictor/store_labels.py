"""
Store boundary labels for each sample in the dataset.
"""
import json
import os
from pathlib import Path

from utils.config import *  # provides Boundary_TRAINING_DIR, enums, etc.

from base_model_utils import (
    reset_kv_cache, setup_lm_and_tokenizer, compute_ppl_with_full_response
)
from preprocess_utils import (
    is_foreign_language, prepare_prompt_and_response
)
from dataset_utils import prepare_datasets

# Use argparse-based flags from your updated flags_config
from flags_config import get_args


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def store_label(args):
    """
    Store boundary labels for each sample in the dataset.
    """
    train_dataset, val_dataset = prepare_datasets()

    lm, tokenizer = setup_lm_and_tokenizer(args)
    model_max_len = 8 * 1024
    model_min_len = 512

    label_folder = Path(f"{Boundary_TRAINING_DIR}/labels/{args.dataset}")
    _ensure_dir(label_folder)

    # ---------- Train ----------
    for sample in train_dataset:
        prompt, response = prepare_prompt_and_response(sample, args.dataset)

        if is_foreign_language(prompt) or is_foreign_language(response):
            continue

        success, _ = compute_ppl_with_full_response(
            lm,
            tokenizer,
            prompt,
            response,
            max_len=model_max_len,
            min_len=model_min_len
        )
        reset_kv_cache(lm, args.method)

        if not success:
            continue

        label_data = {}
        for layer_idx in range(args.num_layers):
            ratios = lm.model.layers[layer_idx].self_attn.ratios
            boundary = lm.model.layers[layer_idx].self_attn.boundaries
            label_data[layer_idx] = {
                "ratios": ratios.tolist(),
                "boundary": boundary.tolist()
            }

        out_path = label_folder / f'sample_{sample["uid"]}.json'
        _write_json(out_path, label_data)

    # ---------- Validation ----------
    for sample in val_dataset:
        prompt, response = prepare_prompt_and_response(sample, args.dataset)

        if is_foreign_language(prompt) or is_foreign_language(response):
            continue

        success, _ = compute_ppl_with_full_response(
            lm,
            tokenizer,
            prompt,
            response,
            max_len=model_max_len,
            min_len=model_min_len
        )
        reset_kv_cache(lm, args.method)

        if not success:
            continue

        label_data = {}
        for layer_idx in range(args.num_layers):
            ratios = lm.model.layers[layer_idx].self_attn.ratios
            boundary = lm.model.layers[layer_idx].self_attn.boundaries
            label_data[layer_idx] = {
                "ratios": ratios.tolist(),
                "boundary": boundary.tolist()
            }

        out_path = label_folder / f'val_sample_{sample["uid"]}.json'
        _write_json(out_path, label_data)


def main():
    args = get_args()
    store_label(args)


if __name__ == "__main__":
    main()