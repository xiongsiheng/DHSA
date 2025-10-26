"""
Visualize the results of needle in a haystack (NIAH) tests
"""

import io
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.config import *  # RESULTS_DIR, RETRIEVAL_ANSWER, PRETRAINED_LENS, Model, etc.
from utils.flags_config import get_args  # argparse-based flags


def main():
    # Parse CLI args (argparse)
    args = get_args()

    # Use the arguments
    FOLDER_PATH = args.NIAH_folder_path
    MODEL_NAME = args.model_name
    PRETRAINED_LEN = PRETRAINED_LENS[Model[MODEL_NAME]]
    BUDGET_PREFILL = args.budget_prefill
    BUDGET_DECODE = args.budget_decode
    METHOD_NAME = args.method

    print(f"model_name = {MODEL_NAME}")

    # Find all json files in the directory
    results_dir = Path(RESULTS_DIR) / FOLDER_PATH
    json_files = sorted(results_dir.glob("*.json"))

    # Collect rows
    data = []
    expected_answer_tokens = set(RETRIEVAL_ANSWER.lower().split())

    for file_path in json_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

        # Extract fields
        document_depth = json_data.get("depth_percent", None)
        context_length = json_data.get("context_length", None)
        model_response = (json_data.get("model_response", "") or "").lower()

        # Simple token-overlap score with expected answer
        resp_tokens = set(model_response.split())
        denom = len(expected_answer_tokens) or 1
        score = len(resp_tokens.intersection(expected_answer_tokens)) / denom

        data.append(
            {
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score,
            }
        )

    if not data:
        print(f"No JSON files found or parsed in {results_dir}. Nothing to plot.")
        return

    # DataFrame
    df = pd.DataFrame(data).dropna(subset=["Document Depth", "Context Length", "Score"])

    # Sort unique context lengths
    locations = sorted(df["Context Length"].unique())

    # Find column index for the pretrained length threshold
    pretrained_len_idx = 0
    for i, l in enumerate(locations):
        if l > PRETRAINED_LEN:
            pretrained_len_idx = i
            break
    else:
        # If none are greater, place line after the last column
        pretrained_len_idx = len(locations) - 1

    print(df.head())
    print(f"Overall score {df['Score'].mean():.3f}")

    # Pivot for heatmap
    pivot_table = (
        pd.pivot_table(
            df,
            values="Score",
            index=["Document Depth", "Context Length"],
            aggfunc="mean",
        )
        .reset_index()
        .pivot(index="Document Depth", columns="Context Length", values="Score")
    )

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )

    # Heatmap
    plt.figure(figsize=(38, 8))
    sns.heatmap(
        pivot_table,
        vmin=0,
        vmax=1,
        cmap=cmap,
        cbar_kws={"label": "Score"},
        linewidths=0.5,
        linecolor="grey",
        linestyle="--",
    )

    title = (
        f"Pressure Testing {MODEL_NAME} {METHOD_NAME} "
        f"Budget Prefill {BUDGET_PREFILL} Budget Decode {BUDGET_DECODE}\n"
        "Fact Retrieval Across Context Lengths (Needle In A Haystack)"
    )
    plt.title(title, fontsize=18)
    plt.xlabel("Token Limit", fontsize=18)
    plt.ylabel("Depth Percent", fontsize=18)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()

    # Vertical line at pretrained context length (approx by column index)
    plt.axvline(x=pretrained_len_idx + 0.8, color="white", linestyle="--", linewidth=4)

    # Save
    out_dir = Path(RESULTS_DIR) / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{MODEL_NAME}.png"
    print(f"saving at {save_path}")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150)
    save_path.write_bytes(buffer.getvalue())


if __name__ == "__main__":
    main()