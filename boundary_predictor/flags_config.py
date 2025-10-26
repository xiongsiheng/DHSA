"""
CLI arguments for boundary predictor training.
"""
import argparse
from utils.config import *


def get_args():
    parser = argparse.ArgumentParser(description="Boundary Predictor Training")

    # General arguments
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--silent", action="store_true", help="Whether to suppress ongoing status messages")
    parser.add_argument("--method", type=str, default=SparseAttnMethod.dhsa.name,
                        choices=[e.name for e in SparseAttnMethod],
                        help="The sparse attention method to use.")
    parser.add_argument("--num_layers", type=int, default=26, help="Number of layers")

    # Model settings
    parser.add_argument("--model_name", type=str, default=Model.gemma_2_2b.name,
                        choices=[e.name for e in Model],
                        help="Model name to be tested")
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True,
                        help="Whether to use fast tokenizer")
    parser.add_argument("--output_attentions", action="store_true", help="Whether to output attention weights")
    parser.add_argument("--use_cache", action="store_true", default=True,
                        help="Whether to use cache during evaluation")
    parser.add_argument("--attn_implementation", type=str, default=AttnImplementation.sdpa.name,
                        choices=[e.name for e in AttnImplementation],
                        help="The attention implementation to use.")

    # Attention budgets
    parser.add_argument("--budget_prefill", type=int, default=-1,
                        help="Budget for prefill tokens (-1 means all tokens are used)")
    parser.add_argument("--budget_decode", type=int, default=-1,
                        help="Budget for decode tokens (-1 means all tokens are used)")
    parser.add_argument("--block_size", type=int, default=32, help="Block size")
    parser.add_argument("--dhsa_boundary_window_size", type=int, default=None, help="Boundary window size")
    parser.add_argument("--dhsa_use_nms", action="store_true", default=True, help="Whether to use NMS")
    parser.add_argument("--dhsa_nms_window_size", type=int, default=8, help="Window size for NMS")

    # Predictor arguments
    parser.add_argument("--dhsa_predictor_hidden_size", type=int, default=256, help="Predictor hidden size")
    parser.add_argument("--dhsa_predictor_num_heads", type=int, default=4, help="Predictor number of heads")
    parser.add_argument("--dhsa_predictor_use_window_pool", action="store_true", default=False,
                        help="Whether to use window pooling in predictor")

    return parser.parse_args()