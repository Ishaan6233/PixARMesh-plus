"""Tests for Hydra benchmark config composition.

Related files:
- Validates `configs/config.yaml` and its config groups.
- Protects `main.py` from broken default composition.
"""

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class ConfigTest(unittest.TestCase):
    def test_default_config_composes(self):
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


if __name__ == "__main__":
    unittest.main()
