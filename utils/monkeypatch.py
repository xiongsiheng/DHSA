import os
import sys
from pathlib import Path


ATTENTION_METHODS = (
    "full",
    "DHSA_vs_optimized",
    "DHSA_vsb_memory_efficient",
    "DHSA_learned_topK_static",
    "DHSA_learned_topK_dynamic",
)
SPARSITY_MASKS = ATTENTION_METHODS[1:]
LEARNED_TOPK_METHODS = (
    "DHSA_learned_topK_static",
    "DHSA_learned_topK_dynamic",
)


def add_block_sparse_import_paths(block_sparse_root: Path) -> None:
    build_libs = sorted((block_sparse_root / "build").glob("lib.*"))
    for import_path in reversed([block_sparse_root] + build_libs):
        import_path_str = str(import_path)
        while import_path_str in sys.path:
            sys.path.remove(import_path_str)
        sys.path.insert(0, import_path_str)


def default_block_sparse_root() -> Path:
    return Path(__file__).resolve().parents[1] / "Block-Sparse-Attention"


def load_DHSA_patch_module(block_sparse_root: Path | None = None):
    block_sparse_root = block_sparse_root or default_block_sparse_root()
    patch_path = block_sparse_root / "DHSA_patch.py"
    if not patch_path.exists():
        raise FileNotFoundError(f"Unable to find DHSA module: {patch_path}")

    add_block_sparse_import_paths(block_sparse_root)
    try:
        import DHSA_patch as patch_module
        from DHSA_patch import _call_block_sparse_attn_func, patch_llama_with_block_sparse

        return patch_module
    except ModuleNotFoundError as exc:
        if exc.name == "block_sparse_attn_cuda":
            extension_files = sorted(block_sparse_root.glob("build/lib.*/block_sparse_attn_cuda*.so"))
            extension_list = ", ".join(str(path) for path in extension_files) or "none found"
            raise ModuleNotFoundError(
                "Unable to import block_sparse_attn_cuda. The DHSA "
                "CUDA extension must be built for the "
                f"active Python interpreter ({sys.version_info.major}.{sys.version_info.minor}). "
                f"Found extension files: {extension_list}"
            ) from exc
        raise


def load_DHSA_patch():
    return load_DHSA_patch_module().patch_llama_with_block_sparse


def infer_model_family(model) -> str:
    model_type = str(
        getattr(getattr(model, "config", None), "model_type", "")
    ).lower()
    if model_type == "qwen2" or "qwen2" in model_type:
        return "qwen2"
    if model_type == "llama" or "llama" in model_type:
        return "llama"
    raise ValueError(
        "DHSA_release supports Qwen2/Qwen2.5 and Llama models only; "
        f"got model_type={model_type!r}."
    )


def requires_predictor_checkpoint(attention_method: str) -> bool:
    if attention_method not in ATTENTION_METHODS:
        raise ValueError(
            f"attention method must be one of: {', '.join(ATTENTION_METHODS)}"
        )
    return attention_method in LEARNED_TOPK_METHODS


def validate_sparse_config(density: float, q_block_size: int, k_block_size: int) -> None:
    if not 0.0 < density <= 1.0:
        raise ValueError("--density must be in (0, 1].")
    if q_block_size % 128 != 0:
        raise ValueError("--q-block-size must be a multiple of 128.")
    if k_block_size not in (32, 64, 128):
        raise ValueError("--k-block-size must be one of 32, 64, or 128.")


def validate_attention_config(
    attention_method: str,
    density: float,
    q_block_size: int,
    k_block_size: int,
) -> None:
    if attention_method not in ATTENTION_METHODS:
        raise ValueError(
            f"attention method must be one of: {', '.join(ATTENTION_METHODS)}"
        )
    if attention_method == "full":
        if float(density) != 1.0:
            raise ValueError("full attention requires --density 1.0.")
        return
    validate_sparse_config(density, q_block_size, k_block_size)


def configure_DHSA(
    model,
    density: float,
    q_block_size: int,
    k_block_size: int,
    sparsity_mask: str,
    chunk_calculation: bool,
    predictor_checkpoint: str | Path | None = None,
) -> None:
    validate_attention_config(
        sparsity_mask,
        density,
        q_block_size,
        k_block_size,
    )
    if sparsity_mask == "full":
        return

    patch_module = load_DHSA_patch_module()
    model_family = infer_model_family(model)
    if sparsity_mask == "DHSA_vs_optimized":
        if model_family == "qwen2":
            mask_builder = (
                patch_module._generate_sparsity_mask_with_vertical_slash_sample32
            )
        else:
            mask_builder = (
                patch_module._generate_sparsity_mask_with_vertical_slash_sample64
            )
    elif sparsity_mask == "DHSA_vsb_memory_efficient":
        mask_builder = (
            patch_module._generate_sparsity_mask_with_vertical_slash_blockwise
        )
    else:
        checkpoint = predictor_checkpoint or os.getenv(
            "DHSA_PREDICTOR_CHECKPOINT"
        )
        if checkpoint is None:
            raise ValueError(
                f"{sparsity_mask} requires --predictor-checkpoint "
                "or DHSA_PREDICTOR_CHECKPOINT"
            )
        expected_variant = (
            "static"
            if sparsity_mask == "DHSA_learned_topK_static"
            else "dynamic"
        )
        patch_module.load_DHSA_topk_predictor(
            checkpoint,
            next(model.parameters()).device,
            expected_variant=expected_variant,
            density=density,
        )
        mask_builder = (
            patch_module._generate_sparsity_mask_with_learned_topk
        )

    patch_module._generate_sparsity_mask = mask_builder

    sparsity = 1.0 - float(density)
    patch_module.patch_model_with_block_sparse(
        model,
        sparsity=sparsity,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        only_prefill=True,
        chunk_calculation=chunk_calculation,
    )
    patched_modules = [
        module
        for module in model.modules()
        if hasattr(module, "_block_sparse_family")
    ]
    if not patched_modules:
        model_type = getattr(getattr(model, "config", None), "model_type", "unknown")
        raise RuntimeError(
            f"DHSA did not patch any attention modules for model type {model_type!r}"
        )
    for module in patched_modules:
        module._block_sparse_sparsity = sparsity
        module._block_sparse_q_block_size = q_block_size
        module._block_sparse_k_block_size = k_block_size
        module._block_sparse_chunk_calculation = chunk_calculation
