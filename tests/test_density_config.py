import unittest

from topk_predictor import (
    resolve_density_config,
    validate_latest_checkpoint,
)


class DensityConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "dynamic_density_min_ratio": 0.5,
            "dynamic_density_max_ratio": 2.0,
        }
        self.overrides = {
            "0.0625": {
                "dynamic_density_min_ratio": 0.5,
                "dynamic_density_max_ratio": 2.0,
            },
            "0.125": {
                "dynamic_density_min_ratio": 0.875,
                "dynamic_density_max_ratio": 1.25,
            },
        }

    def test_selects_requested_density_without_mutating_base(self):
        resolved = resolve_density_config(
            self.config,
            self.overrides,
            0.125,
        )
        self.assertEqual(resolved["dynamic_density_min_ratio"], 0.875)
        self.assertEqual(resolved["dynamic_density_max_ratio"], 1.25)
        self.assertEqual(self.config["dynamic_density_min_ratio"], 0.5)

    def test_checkpoint_without_overrides_needs_no_density(self):
        self.assertEqual(
            resolve_density_config(self.config, None, None),
            self.config,
        )

    def test_requires_density_for_merged_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "density is required"):
            resolve_density_config(self.config, self.overrides, None)

    def test_rejects_unsupported_density(self):
        with self.assertRaisesRegex(ValueError, "supported densities"):
            resolve_density_config(self.config, self.overrides, 0.25)


class LatestCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = {
            "predictor_config": {
                "num_layers": 2,
                "num_heads": 4,
            },
            "state_dict": {},
            "sample_prototypes": None,
        }

    def test_latest_schema_rejects_removed_metadata(self):
        checkpoint = dict(self.checkpoint, predictor_version=3)
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_latest_checkpoint(checkpoint)

    def test_latest_schema_requires_predictor_config(self):
        checkpoint = dict(self.checkpoint)
        del checkpoint["predictor_config"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_latest_checkpoint(checkpoint)

    def test_latest_schema_accepts_optional_density_overrides(self):
        checkpoint = dict(
            self.checkpoint,
            density_config_overrides={
                "0.125": {
                    "dynamic_density_min_ratio": 0.875,
                    "dynamic_density_max_ratio": 1.25,
                }
            }
        )
        validate_latest_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
