"""visualize the results of needle in a haystack tests."""

import argparse
import glob
import json
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def find_json_files(folder_path):
    """Return result JSON files from a directory or glob-like prefix."""
    if os.path.isdir(folder_path):
        pattern = os.path.join(folder_path, "*.json")
    else:
        pattern = f"{folder_path}*.json"
    return sorted(glob.glob(pattern))


def score_from_response(json_data):
    saved_score = json_data.get("score", json_data.get("Score"))
    if saved_score is not None:
        score = float(saved_score)
        return score / 10 if score > 1 else score

    model_response = json_data.get("model_response", "").lower()
    expected_answer = (
        "eat a sandwich and sit in Dolores Park on a sunny day."
        .lower()
        .split()
    )

    return (
        len(set(model_response.split()).intersection(set(expected_answer)))
        / len(set(expected_answer))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate needle-in-haystack heatmap visualization"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the directory containing JSON results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being evaluated",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="full attention",
        help='Method name for the visualization (default: "full attention")',
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Density value to show in the figure title",
    )

    args = parser.parse_args()

    folder_path = args.folder_path
    model_name = args.model_name
    method_name = args.method_name
    density = args.density

    print("model_name = %s" % model_name)

    json_files = find_json_files(folder_path)

    if not json_files:
        raise FileNotFoundError(
            f"No JSON result files found under {folder_path!r}. "
            "Pass a directory containing *_results.json files."
        )

    data = []

    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)

        document_depth = json_data.get("depth_percent", None)
        context_length = json_data.get("context_length", None)
        score = score_from_response(json_data)

        data.append(
            {
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score,
            }
        )

    df = pd.DataFrame(data)

    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = (
        pd.pivot_table(
            df,
            values="Score",
            index=["Document Depth", "Context Length"],
            aggfunc="mean",
        )
        .reset_index()
        .pivot(
            index="Document Depth",
            columns="Context Length",
            values="Score",
        )
    )

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        ["#F0496E", "#EBB839", "#0CD79F"],
    )

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

    if density is None:
        density_text = ""
    else:
        density_text = f" Density {density}"

    title = (
        f'Pressure Testing {model_name} {method_name}{density_text}\n'
        'Fact Retrieval Across Context Lengths ("Needle In A HayStack")'
    )

    plt.title(title, fontsize=18)
    plt.xlabel("Token Limit", fontsize=18)
    plt.ylabel("Depth Percent", fontsize=18)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()

    # Save to sibling img/ directory of the top-level results folder.
    folder_path_norm = os.path.normpath(folder_path)

    if os.path.isdir(folder_path_norm):
        result_dir = folder_path_norm
    else:
        result_dir = os.path.dirname(folder_path_norm)

    parent_dir = os.path.dirname(os.path.dirname(result_dir))
    img_dir = os.path.join(parent_dir, "img")

    os.makedirs(img_dir, exist_ok=True)

    save_path = os.path.join(img_dir, f"{model_name}.png")
    print("saving at %s" % save_path)

    plt.savefig(save_path, dpi=150)


if __name__ == "__main__":
    main()
