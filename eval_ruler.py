#!/usr/bin/env python3
"""
RULER Evaluation Script

Loads JSON/JSONL files, computes string-match scores with `string_match_all`,
writes per-method `metrics.json` files, and saves a summary table to `results.csv`.
"""

import os
import json
import csv
import argparse
from pathlib import Path

from utils.metrics import string_match_all


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    return parser.parse_args(args)


def load_jsonl_or_json(path):
    predictions, answers, lengths = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return predictions, answers, lengths

    # Case 1: normal JSONL, one dict per line
    if "\n" in text:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                if "length" in data:
                    lengths.append(data["length"])
            except Exception as e:
                print(f"error parsing one line in {path}: {e}")
        return predictions, answers, lengths

    # Case 2: single JSON object or JSON list
    data = json.loads(text)

    if isinstance(data, dict):
        data = [data]

    for item in data:
        predictions.append(item["pred"])
        answers.append(item["answers"])
        if "length" in item:
            lengths.append(item["length"])

    return predictions, answers, lengths


def discover_result_files(results_dir, dataset_list):
    """
    Return:
        dataset_to_method_file = {
            dataset: {
                method_name: json_path
            }
        }

    Supports both layouts:

    Layout A:
        results_dir/dataset/method.json

    Layout B:
        results_dir/method/dataset.json
    """

    results_dir = Path(results_dir)
    dataset_set = set(dataset_list)

    dataset_to_method_file = {dataset: {} for dataset in dataset_list}

    for json_path in results_dir.rglob("*.json"):
        if json_path.name == "metrics.json":
            continue

        rel = json_path.relative_to(results_dir)
        parts = rel.parts

        # Layout A: results_dir / dataset / method.json
        # Example: niah_single_1 / blocksparse.json
        if len(parts) >= 2 and parts[0] in dataset_set:
            dataset = parts[0]
            method = json_path.stem
            dataset_to_method_file[dataset][method] = json_path
            continue

        # Layout B: results_dir / method / dataset.json
        # Example:
        # Llama-3.1-8B-Instruct_DHSA... / niah_single_1.json
        if json_path.stem in dataset_set:
            dataset = json_path.stem
            method = parts[0] if len(parts) >= 2 else "unknown"
            dataset_to_method_file[dataset][method] = json_path
            continue

    return dataset_to_method_file


if __name__ == '__main__':
    args = parse_args()

    dataset_list = [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt", "qa_1", "qa_2",
        "scat_arith_1", "scat_arith_2", "scat_arith_3"
    ]

    if args.datasets:
        dataset_list = args.datasets

    dataset_to_method_file = discover_result_files(args.results_dir, dataset_list)

    # Collect all discovered methods
    method_list = sorted({
        method
        for dataset in dataset_list
        for method in dataset_to_method_file[dataset].keys()
    })

    print("Discovered methods:")
    for method in method_list:
        print(" ", method)

    results_list = [["dataset"] + dataset_list]

    for method in method_list:
        row = [method]

        for dataset in dataset_list:
            eval_file = dataset_to_method_file[dataset].get(method)

            if eval_file is None:
                row.append(-1)
                continue

            try:
                predictions, answers, lengths = load_jsonl_or_json(eval_file)

                num_samples = len(predictions)
                print(f"dataset {dataset} method {method} file {eval_file} num_samples {num_samples}")

                if num_samples == 0:
                    row.append(-1)
                    print(f"dataset {dataset} method {method} skipped because num_samples is 0")
                    continue

                score = string_match_all(predictions, answers)
                row.append(score)

                output_dir = os.path.dirname(eval_file)
                metrics_path = os.path.join(output_dir, "metrics.json")

                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump({dataset: score}, f, ensure_ascii=False, indent=4)

                print(f"dataset {dataset} method {method} score {score}")

            except Exception as e:
                row.append(-1)
                print(f"dataset {dataset} method {method} failed on {eval_file}: {e}")

        results_list.append(row)

    output_csv = os.path.join(args.results_dir, "results.csv")

    with open(output_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)

    print(f"Saved results to {output_csv}")