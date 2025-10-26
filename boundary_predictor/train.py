"""
Train a boundary predictor for each layer in a language model.
"""
import json
from pathlib import Path
from collections import defaultdict

import torch
import transformers
import tqdm
import numpy as np

from utils.config import *  # e.g., Boundary_TRAINING_DIR

from base_model_utils import (
    reset_kv_cache, setup_lm_and_tokenizer, compute_ppl_with_full_response, load_labels
)
from preprocess_utils import is_foreign_language
from dataset_utils import prepare_datasets
from loss import focal_bce_loss, soft_label_transform
from metrics import precision_recall_f1, topk_overlap
from models import BoundarySimilarityAttn

# argparse-based flags
from flags_config import get_args


def print_settings(args):
    """Print training settings."""
    print("---------Training setting summary---------:")
    print(f"Predictor hidden size: {args.dhsa_predictor_hidden_size}")
    print(f"Predictor number of heads: {args.dhsa_predictor_num_heads}")
    print(f"Predictor use window pooling: {args.dhsa_predictor_use_window_pool}")
    print("-" * 20)


def forward(
    feat: torch.Tensor,
    model: torch.nn.Module,
    ratios: torch.Tensor,
    alpha: float,
    beta: float,
    window_size: int,
    device
):
    """
    Forward pass of the boundary predictor to prepare training data.

    Args:
        feat: Boundary predictor features.
        model: Boundary predictor model.
        ratios: Ground-truth ratios.
        alpha: Alpha for soft label transformation.
        beta: Beta for soft label transformation (expects log-space in this script).
        window_size: Boundary predictor window size.
        device: Device to run the model.

    Returns:
        logits: Raw logits per token.
        probs: Probability per token of being a boundary.
        targets: Binary targets (0 or 1).
        mask: Mask to ignore the first and last 2w tokens.
    """
    # Get boundary predictor features
    batch, head, seq_len, dim = feat.shape
    feat = feat.permute(0, 2, 1, 3).reshape(batch, seq_len, head * dim).float()
    feat = feat.permute(0, 2, 1).contiguous()
    feat = feat.to(device)

    # Model forward
    logits = model(feat)  # (B, L)
    probs = torch.sigmoid(logits)

    # Targets from ratios
    ratios = ratios.clone()
    ratios[ratios < 1.0] = 1.0
    ratios = ratios.unsqueeze(0)  # (1, L)
    targets = soft_label_transform(ratios, alpha=alpha, beta=beta).to(device)
    targets = targets >= 0.5  # binarize

    # Mask (ignore the first and last 2w tokens)
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    if 2 * window_size - 1 < seq_len - 2 * window_size:
        mask[:, 2 * window_size - 1 : seq_len - 2 * window_size] = 1
    return logits, probs, targets, mask


def evaluate(
    val_set,
    lm: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    model: torch.nn.Module,
    model_max_len: int,
    model_min_len: int,
    args,
    alpha: float,
    beta: float,
    window_size: int,
    topk: int
):
    """
    Evaluate boundary predictor on validation set.
    """
    precision_hist = defaultdict(list)
    recall_hist = defaultdict(list)
    f1_hist = defaultdict(list)
    overlap_rate_hist = defaultdict(list)

    # iterate directly over samples (fixes a bug where enumerate() objects were treated as samples)
    for sample in tqdm.tqdm(val_set, total=len(val_set)):
        prompt, response = sample["prompt"], sample["response"]

        # Load labels from file
        label_file = Path(Boundary_TRAINING_DIR) / "labels" / sample["source"] / f'val_sample_{sample["uid"]}.json'
        if not label_file.exists():
            continue
        with label_file.open("r", encoding="utf-8") as f:
            label_data = json.load(f)
        load_labels(lm, label_data)

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

        # Evaluate boundary predictor on each layer
        for layer_idx in range(args.num_layers):
            feat = lm.model.layers[layer_idx].self_attn.key_states
            ratios = lm.model.layers[layer_idx].self_attn.ratio

            logits, probs, targets, mask = forward(
                feat, model, ratios,
                alpha, beta,
                window_size, lm.device
            )
            prec, rec, f1 = precision_recall_f1(probs, targets, mask)
            _, overlap_rate = topk_overlap(
                ratios=ratios,
                logits=logits,
                k=topk,
                mask=mask
            )

            precision_hist[layer_idx].append(prec.item())
            recall_hist[layer_idx].append(rec.item())
            f1_hist[layer_idx].append(f1.item())
            overlap_rate_hist[layer_idx].append(overlap_rate)

    return precision_hist, recall_hist, f1_hist, overlap_rate_hist


def train(args):
    """Train boundary predictor."""
    silent = args.silent
    if not silent:
        print_settings(args)

    # Hyperparameters
    global_seed = 42
    model_max_len = 8 * 1024
    model_min_len = 512

    window_size = 4         # boundary predictor window size
    channel_in = 1024       # input channel size

    alpha = 2.0             # soft label alpha
    beta = 2.0              # soft label beta
    beta = torch.log(torch.tensor(beta + 1e-6))  # as used in original code

    gamma = 2.0             # focal loss gamma
    pos_w = 1.3             # focal loss positive weight

    lr = 1e-4
    max_epochs = 2
    num_steps_save_ckpt = 2000
    topk = 500

    train_dataset, val_dataset = prepare_datasets(seed=global_seed)
    if not silent:
        print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Setup LM and tokenizer
    lm, tokenizer = setup_lm_and_tokenizer(args)

    model = BoundarySimilarityAttn(
        channel_in=channel_in,
        window_size=window_size,
        d_h=args.dhsa_predictor_hidden_size,
        heads=args.dhsa_predictor_num_heads,
        window_pool=args.dhsa_predictor_use_window_pool
    ).to(lm.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos_w = torch.tensor(pos_w, device=lm.device, dtype=torch.float32)

    label_root = Path(Boundary_TRAINING_DIR) / "labels"
    metric_hist_path = Path(Boundary_TRAINING_DIR) / "metric_history.json"
    metric_hist = {}
    for epoch in range(max_epochs):
        if not silent:
            print(f"Starting epoch {epoch} ...")

        metric_hist[f"epoch_{epoch}"] = {
            "loss": defaultdict(list),
            "train_precision": defaultdict(list),
            "train_recall": defaultdict(list),
            "train_F1": defaultdict(list),
            "train_topk_overlap": defaultdict(list),
            "val_precision": [],
            "val_recall": [],
            "val_F1": [],
            "val_topk_overlap": [],
            "average": {
                "loss": defaultdict(list),
                "train_precision": defaultdict(list),
                "train_recall": defaultdict(list),
                "train_F1": defaultdict(list),
                "train_topk_overlap": defaultdict(list),
                "val_precision": defaultdict(list),
                "val_recall": defaultdict(list),
                "val_F1": defaultdict(list),
                "val_topk_overlap": defaultdict(list)
            }
        }

        # Shuffle once per epoch if dataset supports it; otherwise fall back
        epoch_ds = train_dataset.shuffle(seed=global_seed + epoch) if hasattr(train_dataset, "shuffle") else train_dataset
        ckpt_path = Path(Boundary_TRAINING_DIR) / f"model_weights_epoch_{epoch}.pt"

        for (i, sample) in tqdm.tqdm(enumerate(epoch_ds), total=len(epoch_ds)):
            prompt, response = sample["prompt"], sample["response"]

            if not is_foreign_language(prompt) and not is_foreign_language(response):
                label_file = label_root / sample["source"] / f'sample_{sample["uid"]}.json'
                if label_file.exists():
                    # Read labels from file
                    with label_file.open("r", encoding="utf-8") as f:
                        label_data = json.load(f)

                    # Load labels into LM
                    load_labels(lm, label_data)

                    success, _ = compute_ppl_with_full_response(
                        lm,
                        tokenizer,
                        prompt,
                        response,
                        max_len=model_max_len,
                        min_len=model_min_len
                    )
                    reset_kv_cache(lm, args.method)

                    if success:
                        opt.zero_grad()

                        for layer_idx in range(args.num_layers):
                            feat = lm.model.layers[layer_idx].self_attn.key_states
                            ratios = lm.model.layers[layer_idx].self_attn.ratio

                            logits, probs, targets, mask = forward(
                                feat, model, ratios,
                                alpha, beta,
                                window_size, lm.device
                            )
                            prec, rec, f1 = precision_recall_f1(probs, targets, mask)
                            _, overlap_rate = topk_overlap(ratios, logits, topk, mask)

                            metric_hist[f"epoch_{epoch}"]["train_precision"][str(layer_idx)].append(prec.item())
                            metric_hist[f"epoch_{epoch}"]["train_recall"][str(layer_idx)].append(rec.item())
                            metric_hist[f"epoch_{epoch}"]["train_F1"][str(layer_idx)].append(f1.item())
                            metric_hist[f"epoch_{epoch}"]["train_topk_overlap"][str(layer_idx)].append(overlap_rate)

                            loss = focal_bce_loss(
                                logits, targets, mask,
                                pos_weight=pos_w,
                                gamma=gamma,
                                reduction="mean"
                            )
                            loss.backward()
                            metric_hist[f"epoch_{epoch}"]["loss"][str(layer_idx)].append(loss.item())

                        opt.step()
                        opt.zero_grad(set_to_none=True)  # frees grad buckets immediately
                        torch.cuda.empty_cache()         # optional

            # ----- checkpoint & validation -----
            if (i + 1) % num_steps_save_ckpt == 0 or (i + 1) == len(epoch_ds):
                torch.save(model.state_dict(), ckpt_path)
                print(f"Weights written to {ckpt_path}")

                val_precision, val_recall, val_f1, val_overlap = evaluate(
                    val_dataset, lm, tokenizer, model,
                    model_max_len, model_min_len, args,
                    alpha, beta, window_size, topk
                )
                metric_hist[f"epoch_{epoch}"]["val_precision"].append(val_precision)
                metric_hist[f"epoch_{epoch}"]["val_recall"].append(val_recall)
                metric_hist[f"epoch_{epoch}"]["val_F1"].append(val_f1)
                metric_hist[f"epoch_{epoch}"]["val_topk_overlap"].append(val_overlap)

                for layer_idx in range(args.num_layers):
                    mean_loss = np.mean(metric_hist[f"epoch_{epoch}"]["loss"][str(layer_idx)])
                    mean_train_precision = np.mean(metric_hist[f"epoch_{epoch}"]["train_precision"][str(layer_idx)])
                    mean_train_recall = np.mean(metric_hist[f"epoch_{epoch}"]["train_recall"][str(layer_idx)])
                    mean_train_f1 = np.mean(metric_hist[f"epoch_{epoch}"]["train_F1"][str(layer_idx)])
                    mean_train_topk_overlap = np.mean(metric_hist[f"epoch_{epoch}"]["train_topk_overlap"][str(layer_idx)])

                    metric_hist[f"epoch_{epoch}"]["average"]["loss"][str(layer_idx)].append(mean_loss)
                    metric_hist[f"epoch_{epoch}"]["average"]["train_precision"][str(layer_idx)].append(mean_train_precision)
                    metric_hist[f"epoch_{epoch}"]["average"]["train_recall"][str(layer_idx)].append(mean_train_recall)
                    metric_hist[f"epoch_{epoch}"]["average"]["train_F1"][str(layer_idx)].append(mean_train_f1)
                    metric_hist[f"epoch_{epoch}"]["average"]["train_topk_overlap"][str(layer_idx)].append(mean_train_topk_overlap)

                    mean_val_precision = np.mean(val_precision[layer_idx]) if len(val_precision[layer_idx]) else 0.0
                    mean_val_recall = np.mean(val_recall[layer_idx]) if len(val_recall[layer_idx]) else 0.0
                    mean_val_f1 = np.mean(val_f1[layer_idx]) if len(val_f1[layer_idx]) else 0.0
                    mean_val_topk_overlap = np.mean(val_overlap[layer_idx]) if len(val_overlap[layer_idx]) else 0.0

                    metric_hist[f"epoch_{epoch}"]["average"]["val_precision"][str(layer_idx)].append(mean_val_precision)
                    metric_hist[f"epoch_{epoch}"]["average"]["val_recall"][str(layer_idx)].append(mean_val_recall)
                    metric_hist[f"epoch_{epoch}"]["average"]["val_F1"][str(layer_idx)].append(mean_val_f1)
                    metric_hist[f"epoch_{epoch}"]["average"]["val_topk_overlap"][str(layer_idx)].append(mean_val_topk_overlap)

                # Save metric history
                with metric_hist_path.open("w", encoding="utf-8") as f:
                    json.dump(metric_hist, f)


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()