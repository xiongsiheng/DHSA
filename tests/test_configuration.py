import os
import unittest
from types import SimpleNamespace
from unittest import mock

from utils import monkeypatch as dhsa


class FakeModel:
    def __init__(self, model_type):
        self.config = SimpleNamespace(model_type=model_type)
        self.attention = SimpleNamespace()
        self._parameter = SimpleNamespace(device="cpu")

    def parameters(self):
        yield self._parameter

    def modules(self):
        return [self, self.attention]


class FakePatchModule:
    def __init__(self):
        self._generate_sparsity_mask = None
        self._generate_sparsity_mask_with_vertical_slash_sample32 = object()
        self._generate_sparsity_mask_with_vertical_slash_sample64 = object()
        self._generate_sparsity_mask_with_vertical_slash_blockwise = object()
        self._generate_sparsity_mask_with_learned_topk = object()
        self.loaded_checkpoint = None

    def load_DHSA_topk_predictor(
        self,
        checkpoint,
        device,
        expected_variant=None,
        density=None,
    ):
        self.loaded_checkpoint = (
            str(checkpoint),
            device,
            expected_variant,
            density,
        )

    def patch_model_with_block_sparse(self, model, **kwargs):
        model.attention._block_sparse_family = dhsa.infer_model_family(model)
        self.patch_kwargs = kwargs


class ConfigurationTest(unittest.TestCase):
    def test_public_methods_are_exact(self):
        self.assertEqual(
            dhsa.ATTENTION_METHODS,
            (
                "full",
                "DHSA_vs_optimized",
                "DHSA_vsb_memory_efficient",
                "DHSA_learned_topK_static",
                "DHSA_learned_topK_dynamic",
            ),
        )

    def test_full_does_not_load_cuda_patch(self):
        model = FakeModel("qwen2")
        with mock.patch.object(
            dhsa,
            "load_DHSA_patch_module",
            side_effect=AssertionError("CUDA patch should not be loaded"),
        ):
            dhsa.configure_DHSA(
                model,
                density=1.0,
                q_block_size=128,
                k_block_size=32,
                sparsity_mask="full",
                chunk_calculation=False,
            )

    def test_optimized_vs_is_model_aware(self):
        for model_type, builder_name in (
            ("qwen2", "_generate_sparsity_mask_with_vertical_slash_sample32"),
            ("llama", "_generate_sparsity_mask_with_vertical_slash_sample64"),
        ):
            with self.subTest(model_type=model_type):
                patch_module = FakePatchModule()
                with mock.patch.object(
                    dhsa,
                    "load_DHSA_patch_module",
                    return_value=patch_module,
                ):
                    dhsa.configure_DHSA(
                        FakeModel(model_type),
                        density=0.125,
                        q_block_size=128,
                        k_block_size=32,
                        sparsity_mask="DHSA_vs_optimized",
                        chunk_calculation=False,
                    )
                self.assertIs(
                    patch_module._generate_sparsity_mask,
                    getattr(patch_module, builder_name),
                )

    def test_memory_efficient_vsb_uses_blockwise_builder(self):
        patch_module = FakePatchModule()
        with mock.patch.object(
            dhsa,
            "load_DHSA_patch_module",
            return_value=patch_module,
        ):
            dhsa.configure_DHSA(
                FakeModel("llama"),
                density=0.125,
                q_block_size=128,
                k_block_size=32,
                sparsity_mask="DHSA_vsb_memory_efficient",
                chunk_calculation=True,
            )
        self.assertIs(
            patch_module._generate_sparsity_mask,
            patch_module._generate_sparsity_mask_with_vertical_slash_blockwise,
        )

    def test_learned_method_passes_expected_variant(self):
        for method, expected_variant in (
            ("DHSA_learned_topK_static", "static"),
            ("DHSA_learned_topK_dynamic", "dynamic"),
        ):
            with self.subTest(method=method):
                patch_module = FakePatchModule()
                with mock.patch.object(
                    dhsa,
                    "load_DHSA_patch_module",
                    return_value=patch_module,
                ):
                    dhsa.configure_DHSA(
                        FakeModel("qwen2"),
                        density=0.0625,
                        q_block_size=128,
                        k_block_size=32,
                        sparsity_mask=method,
                        chunk_calculation=False,
                        predictor_checkpoint="predictor.pt",
                    )
                self.assertEqual(
                    patch_module.loaded_checkpoint,
                    ("predictor.pt", "cpu", expected_variant, 0.0625),
                )

    def test_checkpoint_environment_variable(self):
        patch_module = FakePatchModule()
        with (
            mock.patch.object(
                dhsa,
                "load_DHSA_patch_module",
                return_value=patch_module,
            ),
            mock.patch.dict(
                os.environ,
                {"DHSA_PREDICTOR_CHECKPOINT": "from-env.pt"},
            ),
        ):
            dhsa.configure_DHSA(
                FakeModel("llama"),
                density=0.125,
                q_block_size=128,
                k_block_size=32,
                sparsity_mask="DHSA_learned_topK_static",
                chunk_calculation=False,
            )
        self.assertEqual(
            patch_module.loaded_checkpoint,
            ("from-env.pt", "cpu", "static", 0.125),
        )

    def test_unsupported_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Qwen2/Qwen2.5 and Llama"):
            dhsa.infer_model_family(FakeModel("gemma2"))

    def test_sparse_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "q-block-size"):
            dhsa.validate_sparse_config(0.125, 64, 32)
        with self.assertRaisesRegex(ValueError, "k-block-size"):
            dhsa.validate_sparse_config(0.125, 128, 16)

    def test_full_requires_unit_density(self):
        with self.assertRaisesRegex(ValueError, "density 1.0"):
            dhsa.validate_attention_config("full", 0.125, 128, 32)


if __name__ == "__main__":
    unittest.main()
