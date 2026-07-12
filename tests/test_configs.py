"""Tests for Hydra benchmark config composition.

Related files:
- Validates `configs/config.yaml` and its config groups.
- Protects `main.py` from broken default composition.
"""

import unittest
from pathlib import Path
from dataclasses import fields
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.utils.config import ModelConfig, mv_prefix_len


def _filter_dataclass_kwargs(dataclass_type, values):
    allowed = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in values.items() if key in allowed}


class ConfigTest(unittest.TestCase):
    def test_default_config_composes(self):
        os.environ.setdefault("RUN_TS", "pytest")
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(
            config_dir=str(repo_root / "configs"),
            version_base=None,
        ):
            cfg = compose(config_name="config")
        OmegaConf.resolve(cfg)
        self.assertEqual(cfg.model.name, "edgerunner")
        self.assertEqual(cfg.environment.cuda, "12.4")
        self.assertTrue(cfg.eval.chamfer.squared)

    def test_mv_layout_loss_experiment_config_composes(self):
        os.environ.setdefault("RUN_TS", "pytest")
        OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(
            config_dir=str(repo_root / "configs"),
            version_base=None,
        ):
            cfg = compose(
                config_name="edgerunner_3d_front_trellis2_mv_stage1",
                overrides=["+experiment=mv_layout_loss_geometry"],
            )
        OmegaConf.resolve(cfg)
        model_values = OmegaConf.to_container(cfg.model, resolve=True)
        model_values.update(OmegaConf.to_container(cfg.dataset.model, resolve=True))
        model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_values))

        self.assertEqual(cfg.model.loss_layout_ordinal_sigma, 2.0)
        self.assertEqual(cfg.model.loss_layout_coord_weight, 1.0)
        self.assertEqual(cfg.model.loss_layout_center_weight, 0.5)
        self.assertEqual(cfg.model.loss_layout_size_weight, 0.5)
        self.assertEqual(mv_prefix_len(model_cfg), 2371)


if __name__ == "__main__":
    unittest.main()
