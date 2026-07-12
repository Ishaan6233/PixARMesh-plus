"""Regression tests for WSD scheduler composition on the real MV stage configs."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers.optimization import get_scheduler

from train import _build_model_config, _build_train_args

_REPO = Path(__file__).resolve().parents[1]
_STAGES = {
    "edgerunner_3d_front_trellis2_mv_stage1": {
        "batch": 8,
        "grad_accum": 2,
        "max_steps": 100000,
        "warmup_steps": 500,
        "save_steps": 5000,
        "eval_steps": 5000,
    },
    "edgerunner_3d_front_trellis2_mv_stage2": {
        "batch": 4,
        "grad_accum": 4,
        "max_steps": 30000,
        "warmup_steps": 500,
        "save_steps": 3000,
        "eval_steps": 3000,
    },
}


def _compose_stage(name: str):
    os.environ.setdefault("RUN_TS", "pytest")
    OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
    with initialize_config_dir(config_dir=str(_REPO / "configs"), version_base=None):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    return cfg


class WsdSchedulerConfigTest(unittest.TestCase):
    def test_mv_stage_configs_use_sv_equivalent_four_gpu_global_batch(self):
        for name, expected in _STAGES.items():
            with self.subTest(name=name):
                cfg = _compose_stage(name)
                args = cfg.train.train_args

                self.assertEqual(args.per_device_train_batch_size, expected["batch"])
                self.assertEqual(
                    args.gradient_accumulation_steps, expected["grad_accum"]
                )
                self.assertEqual(args.max_steps, expected["max_steps"])
                self.assertEqual(args.warmup_steps, expected["warmup_steps"])
                self.assertEqual(args.save_steps, expected["save_steps"])
                self.assertEqual(args.eval_steps, expected["eval_steps"])
                self.assertEqual(args.eval_strategy, "steps")
                self.assertEqual(args.logging_steps, 1)
                self.assertEqual(
                    args.per_device_train_batch_size
                    * args.gradient_accumulation_steps
                    * 4,
                    64,
                )
                self.assertEqual(
                    args.lr_scheduler_kwargs.num_decay_steps,
                    args.max_steps - args.warmup_steps,
                )

    def test_mv_stage_configs_build_working_wsd_scheduler(self):
        for name in _STAGES:
            with self.subTest(name=name):
                cfg = _compose_stage(name)
                cfg.train.train_args.bf16 = False
                model_cfg = _build_model_config(cfg)
                args, _ = _build_train_args(cfg, model_cfg)
                opt = torch.optim.AdamW(
                    [torch.nn.Parameter(torch.zeros(1))], lr=args.learning_rate
                )

                sched = get_scheduler(
                    args.lr_scheduler_type,
                    opt,
                    num_warmup_steps=args.get_warmup_steps(args.max_steps),
                    num_training_steps=args.max_steps,
                    scheduler_specific_kwargs=args.lr_scheduler_kwargs,
                )
                lr0 = sched.get_last_lr()[0]
                for _ in range(args.get_warmup_steps(args.max_steps)):
                    opt.step()
                    sched.step()
                lr_peak = sched.get_last_lr()[0]

                self.assertLess(lr0, lr_peak)
                self.assertAlmostEqual(lr_peak, args.learning_rate, places=8)


if __name__ == "__main__":
    unittest.main()
