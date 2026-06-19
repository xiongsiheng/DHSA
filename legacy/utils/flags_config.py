import argparse
from .config import *


def get_args():
    parser = argparse.ArgumentParser(description="DHSA / evaluation configuration")

    # ---------------------------
    # Model configuration
    # ---------------------------
    parser.add_argument(
        "--model_name",
        type=str,
        default=Model.gemma_2_2b.name,
        choices=[e.name for e in Model],
        help="Model name to be tested.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=AttnImplementation.sdpa.name,
        choices=[e.name for e in AttnImplementation],
        help="The attention implementation to use.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=SparseAttnMethod.full.name,
        choices=[e.name for e in SparseAttnMethod],
        help="The attention method to use.",
    )
    parser.add_argument(
        "--budget_prefill",
        type=int,
        default=-1,
        help="Budget for prefill tokens (-1 means all tokens are used)",
    )
    parser.add_argument(
        "--budget_decode",
        type=int,
        default=-1,
        help="Budget for decode tokens (-1 means all tokens are used)",
    )

    # ---------------------------
    # DHSA configuration
    # ---------------------------
    parser.add_argument(
        "--block_size", type=int, default=64, help="Block size for block sparse attention"
    )
    parser.add_argument(
        "--dhsa_boundary_window_size", type=int, default=4, help="Window size for boundary prediction"
    )
    parser.add_argument(
        "--dhsa_use_nms",
        action="store_true",
        help="Whether to use NMS",
    )
    parser.add_argument(
        "--dhsa_nms_window_size", type=int, default=8, help="Window size for NMS"
    )
    parser.add_argument(
        "--dhsa_chunk_repre_pooling",
        type=str,
        default=PoolingMethod.avgpool_norm.name,
        choices=[e.name for e in PoolingMethod],
        help="Pooling method for chunk representations",
    )
    parser.add_argument(
        "--dhsa_share_boundaries",
        action="store_true",
        help="Whether to share the predicted boundaries",
    )
    parser.add_argument(
        "--dhsa_share_sparsity_masks",
        action="store_true",
        help="Whether to share the local/global sparsity masks",
    )

    # ---------------------------
    # DHSA boundary predictor configuration
    # ---------------------------
    parser.add_argument(
        "--dhsa_ckpt_path", type=str, default=None, help="Path to the checkpoint for boundary predictor"
    )
    parser.add_argument(
        "--dhsa_predictor_channel_in", type=int, default=1024, help="Channel input size for boundary predictor"
    )
    parser.add_argument(
        "--dhsa_predictor_hidden_size", type=int, default=256, help="Hidden size for boundary predictor"
    )
    parser.add_argument(
        "--dhsa_predictor_num_heads", type=int, default=8, help="Number of heads for boundary predictor"
    )
    parser.add_argument(
        "--dhsa_predictor_use_window_pool",
        action="store_true",
        help="Whether to use window pool for boundary predictor",
    )

    # ---------------------------
    # Needle-in-a-Haystack Test (NIAH)
    # ---------------------------
    parser.add_argument("--NIAH_s_len", type=int, default=1000, help="Starting context length")
    parser.add_argument("--NIAH_e_len", type=int, default=8000, help="Ending context length")
    parser.add_argument("--NIAH_step", type=int, default=100, help="Step size for context lengths")
    parser.add_argument(
        "--NIAH_folder_path",
        type=str,
        default="results/gemma_2_2b_full",
        help="Path to the directory containing JSON results",
    )

    # ---------------------------
    # Latency test
    # ---------------------------
    parser.add_argument(
        "--latency_test_max_new_tokens", type=int, default=100, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--latency_test_num_iterations",
        type=int,
        default=NUM_ITERATIONS_LATENCY_TEST,
        help="Number of iterations per context length",
    )
    parser.add_argument(
        "--latency_test_quiet",
        action="store_true",
        help="Whether to suppress ongoing status messages",
    )

    # ---------------------------
    # LongBench test
    # ---------------------------
    parser.add_argument("--longbench_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--longbench_save_dir", type=str, default="exp", help="Directory to save results"
    )
    parser.add_argument(
        "--longbench_max_num_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate per task",
    )
    parser.add_argument(
        "--longbench_sample_method",
        type=str,
        default=SampleMethod.firstk.name,
        choices=[e.name for e in SampleMethod],
        help="Method to sample test examples",
    )
    parser.add_argument(
        "--longbench_eval_batch_size", type=int, default=1, help="Batch size for evaluation"
    )

    # ---------------------------
    # LongBench evaluation
    # ---------------------------
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default="results_longbench/exp/gemma_2_2b",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--eval_longbench_e",
        action="store_true",
        help="Evaluate on LongBench-E (length-stratified evaluation)",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        default=LongBench_DATASETS,
        help="List of datasets to evaluate",
    )
    parser.add_argument(
        "--eval_output_file",
        type=str,
        default="results.csv",
        help="Output CSV filename",
    )

    return parser.parse_args()