import sys
from pathlib import Path


SPARSITY_MASKS = ("DHSA_topk", "DHSA_a", "DHSA_vs", "DHSA_vsb")


def add_block_sparse_import_paths(block_sparse_root: Path) -> None:
    build_libs = sorted((block_sparse_root / "build").glob("lib.*"))
    for import_path in [block_sparse_root] + build_libs:
        import_path_str = str(import_path)
        if import_path_str not in sys.path:
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


def validate_sparse_config(density: float, q_block_size: int, k_block_size: int) -> None:
    if not 0.0 < density <= 1.0:
        raise ValueError("--density must be in (0, 1].")
    if q_block_size % 128 != 0:
        raise ValueError("--q-block-size must be a multiple of 128.")
    if k_block_size not in (32, 64, 128):
        raise ValueError("--k-block-size must be one of 32, 64, or 128.")


def configure_DHSA(
    model,
    density: float,
    q_block_size: int,
    k_block_size: int,
    sparsity_mask: str,
    chunk_calculation: bool,
) -> None:
    patch_module = load_DHSA_patch_module()
    if sparsity_mask == "DHSA_a":
        patch_module._generate_sparsity_mask = patch_module._generate_sparsity_mask_with_A_shape
    elif sparsity_mask == "DHSA_vs":
        patch_module._generate_sparsity_mask = patch_module._generate_sparsity_mask_with_vertical_slash
    elif sparsity_mask == "DHSA_vsb":
        patch_module._generate_sparsity_mask = patch_module._generate_sparsity_mask_with_vertical_slash_blockwise
    elif sparsity_mask != "DHSA_topk":
        raise ValueError(f"--sparsity-mask must be one of: {', '.join(SPARSITY_MASKS)}")

    sparsity = 1.0 - float(density)
    patch_module.patch_llama_with_block_sparse(
        model,
        sparsity=sparsity,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        only_prefill=True,
        chunk_calculation=chunk_calculation,
    )