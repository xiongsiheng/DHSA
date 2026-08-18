"""
Store boundary labels for each sample in the dataset.
"""
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from legacy.utils.config import Boundary_TRAINING_DIR, SparseAttnMethod

from legacy.boundary_predictor.base_model_utils import (
    reset_kv_cache, setup_lm_and_tokenizer, compute_ppl_with_full_response
)
from legacy.boundary_predictor.preprocess_utils import is_foreign_language
from legacy.boundary_predictor.dataset_utils import prepare_datasets

# Use argparse-based flags from your updated flags_config
from legacy.boundary_predictor.flags_config import get_args


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _serialize_layer_labels(lm, num_layers: int) -> dict:
    if num_layers > len(lm.model.layers):
        raise ValueError(f"Requested {num_layers} layers, but the model has {len(lm.model.layers)}")

    label_data = {}
    for layer_idx in range(num_layers):
        self_attn = lm.model.layers[layer_idx].self_attn
        if self_attn.ratios is None or self_attn.boundaries is None:
            raise RuntimeError(f"Layer {layer_idx} did not produce DHSA boundary labels")
        label_data[layer_idx] = {
            "ratios": self_attn.ratios.detach().cpu().tolist(),
            "boundary": self_attn.boundaries.detach().cpu().tolist(),
        }
    return label_data


def _store_split_labels(dataset, split_prefix, lm, tokenizer, args, model_max_len, model_min_len):
    for sample in dataset:
        prompt, response = sample["prompt"], sample["response"]

        if is_foreign_language(prompt) or is_foreign_language(response):
            continue

        success, _ = compute_ppl_with_full_response(
            lm,
            tokenizer,
            prompt,
            response,
            max_len=model_max_len,
            min_len=model_min_len,
        )
        reset_kv_cache(lm, args.method)

        if not success:
            continue

        label_folder = Path(Boundary_TRAINING_DIR) / "labels" / sample["source"]
        _ensure_dir(label_folder)
        label_data = _serialize_layer_labels(lm, args.num_layers)
        out_path = label_folder / f'{split_prefix}sample_{sample["uid"]}.json'
        _write_json(out_path, label_data)


def store_label(args):
    """
    Store boundary labels for each sample in the dataset.
    """
    if args.method != SparseAttnMethod.dhsa.name:
        raise ValueError("Boundary label generation requires --method dhsa")
    if args.dhsa_share_boundaries:
        raise ValueError("Per-layer label generation requires --no-dhsa-share-boundaries")

    train_dataset, val_dataset = prepare_datasets(dataset_name=args.dataset)

    lm, tokenizer = setup_lm_and_tokenizer(args)
    model_max_len = 8 * 1024
    model_min_len = 512

    _store_split_labels(
        train_dataset, "", lm, tokenizer, args, model_max_len, model_min_len
    )
    _store_split_labels(
        val_dataset, "val_", lm, tokenizer, args, model_max_len, model_min_len
    )


def main():
    args = get_args()
    store_label(args)


if __name__ == "__main__":
    main()
