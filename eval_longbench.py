#!/usr/bin/env python3
"""
LongBench Evaluation Script

This script evaluates various methods on LongBench datasets and generates
performance metrics and comparison tables.
"""

import os
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from utils.metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)



# Dataset to metric mapping
DATASET_METRICS = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Datasets that require special preprocessing
DATASETS_REQUIRING_PREPROCESSING = {"trec", "triviaqa", "samsum", "lsht"}

# Default dataset list
DEFAULT_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]



def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate methods on LongBench datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        '--longbench_e', 
        action='store_true',
        help="Evaluate on LongBench-E (length-stratified evaluation)"
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=None,
        help="Limit number of samples to evaluate"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=DEFAULT_DATASETS,
        help="List of datasets to evaluate"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="results.csv",
        help="Output CSV filename"
    )
    
    return parser.parse_args(args)


def preprocess_prediction(prediction: str, dataset: str) -> str:
    """Preprocess prediction based on dataset requirements."""
    if dataset in DATASETS_REQUIRING_PREPROCESSING:
        return prediction.lstrip('\n').split('\n')[0]
    return prediction


def calculate_score_for_sample(
    prediction: str, 
    ground_truths: List[str], 
    dataset: str, 
    all_classes: Optional[List[str]] = None
) -> float:
    """Calculate score for a single sample."""
    if dataset not in DATASET_METRICS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    metric_fn = DATASET_METRICS[dataset]
    prediction = preprocess_prediction(prediction, dataset)
    
    # Take the maximum score across all ground truths
    max_score = 0.0
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, all_classes=all_classes)
        max_score = max(max_score, score)
    
    return max_score


def scorer_e(
    dataset: str, 
    predictions: List[str], 
    answers: List[List[str]], 
    lengths: List[int], 
    all_classes: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Length-stratified evaluation for LongBench-E.
    
    Args:
        dataset: Dataset name
        predictions: List of predictions
        answers: List of ground truth answers for each prediction
        lengths: List of input lengths
        all_classes: Optional list of all classes for classification tasks
    
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
    predictions: List[str], 
    answers: List[List[str]], 
    all_classes: Optional[List[str]] = None
) -> float:
    """
    Standard evaluation scorer.
    
    Args:
        dataset: Dataset name
        predictions: List of predictions
        answers: List of ground truth answers for each prediction
        all_classes: Optional list of all classes for classification tasks
    
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
    num_samples: Optional[int] = None
) -> Tuple[List[str], List[List[str]], List[int], Optional[List[str]]]:
    """
    Load evaluation data from JSON file.
    
    Args:
        eval_file: Path to evaluation file
        num_samples: Maximum number of samples to load
    
    Returns:
        Tuple of (predictions, answers, lengths, all_classes)
    """
    predictions, answers, lengths = [], [], []
    all_classes = None
    
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    with open(eval_file, "r", encoding="utf-8") as f:
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
    args: argparse.Namespace
) -> Optional[float]:
    """
    Evaluate a specific method on a dataset.
    
    Args:
        dataset: Dataset name
        method: Method name
        results_dir: Results directory
        args: Command line arguments
    
    Returns:
        Score or None if evaluation failed
    """
    eval_file = results_dir / dataset / f"{method}.json"
    
    try:
        predictions, answers, lengths, all_classes = load_evaluation_data(
            eval_file, args.num_samples
        )
        
        if not predictions:
            return None
        
        if args.longbench_e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        
        print(f"Dataset: {dataset}, Method: {method}, Score: {score}")
        return score
        
    except Exception as e:
        return None


def create_results_table(
    datasets: List[str], 
    methods: List[str], 
    results_dir: Path, 
    args: argparse.Namespace
) -> List[List[Any]]:
    """
    Create results table with all method comparisons.
    
    Args:
        datasets: List of dataset names
        methods: List of method names
        results_dir: Results directory
        args: Command line arguments
    
    Returns:
        Results table as list of lists
    """
    # Initialize results table
    results_table = [["dataset"] + datasets]
    
    for method in methods:
        method_results = [method]
        
        for dataset in datasets:
            score = evaluate_method_on_dataset(dataset, method, results_dir, args)
            method_results.append(score if score is not None else -1)
        
        results_table.append(method_results)
    
    return results_table


def save_results_table(results_table: List[List[Any]], output_file: Path) -> None:
    """Save results table to CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(results_table)

    print(f"Results saved to {output_file}")


def main():
    """Main evaluation function."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"Starting evaluation on {len(args.datasets)} datasets")
    print(f"Results directory: {results_dir}")
    print(f"LongBench-E mode: {args.longbench_e}")


    # Collect methods from results directory
    methods = set()
    for dataset in args.datasets:
        if dataset not in os.listdir(results_dir):
            continue
        for file in os.listdir(os.path.join(results_dir, dataset)):
            if file.endswith('.json'):
                method = Path(file).stem
                methods.add(method)
    
    # Create results table
    results_table = create_results_table(args.datasets, methods, results_dir, args)

    # Save results
    output_file = results_dir / args.output_file
    save_results_table(results_table, output_file)

    print("Evaluation completed successfully")


if __name__ == '__main__':
    main()