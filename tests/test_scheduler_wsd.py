"""Regression test for the 2026-06-30 scheduler crash at create_scheduler.

The trellis2 stage-1 launch crashed with `get_cosine_schedule_with_warmup() got an
unexpected keyword argument 'num_stable_steps'`: the composed config had
lr_scheduler_type: cosine while lr_scheduler_kwargs carried warmup_stable_decay
arguments. Fixed by pinning lr_scheduler_type: warmup_stable_decay in
configs/edgerunner_3d_front_trellis2_mv.yaml (85b26db). This test composes the real
stage configs and builds the actual scheduler, so any future type/kwargs mismatch in
the config chain fails here instead of at step 0 of a training launch.

Related files:
- configs/edgerunner_3d_front_trellis2_mv.yaml (lr_scheduler_type)
- configs/base_config.yaml (lr_scheduler_kwargs incl. derived num_decay_steps)
"""

import os
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from transformers.optimization import get_scheduler

from train import _build_model_config, _build_train_args

_REPO = Path(__file__).resolve().parents[1]


class WsdSchedulerConfigTest(unittest.TestCase):
    def test_stage_configs_build_working_scheduler(self):
        os.environ.setdefault("RUN_TS", "test")
        with initialize_config_dir(config_dir=str(_REPO / "configs"), version_base=None):
            for name in (
                "edgerunner_3d_front_trellis2_mv_stage1",
                "edgerunner_3d_front_trellis2_mv_stage2",
            ):
                cfg = compose(config_name=name)
                # This unit test only validates scheduler type/kwargs compatibility.
                # Disable GPU-only bf16 validation so it can run on CPU workers.
                cfg.train.train_args.bf16 = False
                model_cfg = _build_model_config(cfg)
                args, _ = _build_train_args(cfg, model_cfg)
                opt = torch.optim.AdamW(
                    [torch.nn.Parameter(torch.zeros(1))], lr=args.learning_rate
                )
                # This exact call crashed on 2026-06-30 (type/kwargs mismatch).
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
