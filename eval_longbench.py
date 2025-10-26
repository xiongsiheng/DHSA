"""
LongBench Evaluation Script

This script evaluates various methods on LongBench datasets and generates
performance metrics and comparison tables.
"""

import os
import json
import csv
from pathlib import Path
from typing import Any

import numpy as np

from utils.config import *  # LongBench_DATASET_METRICS, LongBench_DATASETS_REQUIRING_PREPROCESSING, etc.

# argparse-based flags
from utils.flags_config import get_args


def preprocess_prediction(prediction: str, dataset: str) -> str:
    """Preprocess prediction based on dataset requirements."""
    if dataset in LongBench_DATASETS_REQUIRING_PREPROCESSING:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def calculate_score_for_sample(
    prediction: str,
    ground_truths: list[str],
    dataset: str,
    all_classes: list[str] | None = None
) -> float:
    """Calculate score for a single sample."""
    if dataset not in LongBench_DATASET_METRICS:
        raise ValueError(f"Unknown dataset: {dataset}")

    metric_fn = LongBench_DATASET_METRICS[dataset]
    prediction = preprocess_prediction(prediction, dataset)

    # Take the maximum score across all ground truths
    max_score = 0.0
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, all_classes=all_classes)
        max_score = max(max_score, score)

    return max_score


def scorer_e(
    dataset: str,
    predictions: list[str],
    answers: list[list[str]],
    lengths: list[int],
    all_classes: list[str] | None = None
):
    """
    Length-stratified evaluation for LongBench-E.

    Returns:
        Dictionary with scores for different length ranges
    """
    scores = {"0-4k": [], "4-8k": [], "8k+": []}

    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = calculate_score_for_sample(prediction, ground_truths, dataset, all_classes)

        # Categorize by length
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)

    # Calculate mean scores and convert to percentages
    for key in scores:
        if scores[key]:  # Avoid division by zero
            scores[key] = round(100 * np.mean(scores[key]), 2)
        else:
            scores[key] = 0.0

    return scores


def scorer(
    dataset: str,
    predictions: list[str],
    answers: list[list[str]],
    all_classes: list[str] | None = None
) -> float:
    """
    Standard evaluation scorer.

    Returns:
        Average score as percentage
    """
    total_score = 0.0

    for prediction, ground_truths in zip(predictions, answers):
        score = calculate_score_for_sample(prediction, ground_truths, dataset, all_classes)
        total_score += score

    return round(100 * total_score / len(predictions), 2) if predictions else 0.0


def load_evaluation_data(
    eval_file: Path,
    num_samples: int | None = None
):
    """
    Load evaluation data from JSONL file.

    Returns:
        Tuple of (predictions, answers, lengths, all_classes)
    """
    predictions, answers, lengths = [], [], []
    all_classes = None

    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with eval_file.open("rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if num_samples is not None and len(predictions) >= num_samples:
                break

            try:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])

                if "all_classes" in data:
                    all_classes = data["all_classes"]

                if "length" in data:
                    lengths.append(data["length"])
                else:
                    lengths.append(0)  # Default length if not provided

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line {line_num} in {eval_file}: {e}")
                continue

    print(f"Loaded {len(predictions)} predictions from {eval_file}")
    return predictions, answers, lengths, all_classes


def evaluate_method_on_dataset(
    dataset: str,
    method: str,
    results_dir: Path,
    args
):
    """
    Evaluate a specific method on a dataset.

    Returns:
        Score or None if evaluation failed
    """
    eval_file = results_dir / dataset / f"{method}.json"

    try:
        predictions, answers, lengths, all_classes = load_evaluation_data(
            eval_file, args.eval_num_samples
        )

        if not predictions:
            return None

        if args.eval_longbench_e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)

        print(f"Dataset: {dataset}, Method: {method}, Score: {score}")
        return score

    except Exception as e:
        print(f"Evaluation failed for dataset={dataset}, method={method}: {e}")
        return None


def create_results_table(
    datasets: list[str],
    methods: list[str],
    results_dir: Path,
    args
):
    """
    Create results table with all method comparisons.

    Returns:
        Results table as list of lists
    """
    # Header row: "dataset" followed by dataset names
    results_table = [["method"] + datasets]

    for method in methods:
        method_results = [method]
        for dataset in datasets:
            score = evaluate_method_on_dataset(dataset, method, results_dir, args)
            method_results.append(score if score is not None else -1)
        results_table.append(method_results)

    return results_table


def save_results_table(results_table: list[list[Any]], output_file: Path) -> None:
    """Save results table to CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results_table)
    print(f"Results saved to {output_file}")


def main():
    args = get_args()

    results_dir = Path(LongBench_RESULTS_DIR) / args.eval_results_dir

    print(f"Starting evaluation on {len(args.eval_datasets)} datasets")
    print(f"Results directory: {results_dir}")
    print(f"LongBench-E mode: {args.eval_longbench_e}")

    # Collect methods from results directory
    methods = set()
    for dataset in args.eval_datasets:
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            continue
        for file in dataset_dir.iterdir():
            if file.suffix == ".json":
                methods.add(file.stem)

    methods = sorted(methods)

    # Create results table
    results_table = create_results_table(
        args.eval_datasets,
        methods,
        results_dir,
        args
    )

    # Save results
    output_file = results_dir / args.eval_output_file
    save_results_table(results_table, output_file)

    print("Evaluation completed successfully")


if __name__ == "__main__":
    main()