"""Training adapter that runs the existing src trainer under benchmark governance.

This module preserves the legacy TRL/Accelerate training path while forcing
the canonical benchmark configs, artifacts, adapter registry, and dataset
contract around it.

Related files:
- Loads models via `models/registry.py`.
- Loads standardized datasets via `utils/canonical_dataset.py`.
- Delegates actual training to `src.utils.trainer.CustomSFTTrainer`.
- Called by `main.py` when `task=train`.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def train_with_existing_stack(cfg: Any, output_dir: str | Path) -> None:
    """Train through the existing src trainer under benchmark governance."""
    import datasets
    import transformers
    from accelerate import Accelerator

    from models.registry import build_model_adapter
    from src.data.collator import get_mesh_data_collator
    from src.data.mesh import MeshProcessor
    from src.utils.ckpt import get_last_checkpoint
    from src.utils.config import DataConfig, ModelConfig
    from src.utils.logging import JsonlLoggerCallback
    from src.utils.sig import SaveAndStopOnSignalCallback, install_sigusr1_handler
    from src.utils.trainer import CustomSFTConfig, CustomSFTTrainer
    from utils.canonical_dataset import get_canonical_mesh_dataset

    install_sigusr1_handler()
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    data_cfg = DataConfig(**OmegaConf.to_container(cfg.dataset.src_data, resolve=True))
    allowed_model_keys = {field.name for field in fields(ModelConfig)}
    model_cfg = ModelConfig(
        **{
            key: value
            for key, value in OmegaConf.to_container(cfg.model, resolve=True).items()
            if key in allowed_model_keys
        }
    )
    train_args_dict = OmegaConf.to_container(cfg.train.train_args, resolve=True)
    resume_from_checkpoint = bool(train_args_dict.pop("resume_from_checkpoint", True))
    train_args_dict["output_dir"] = str(Path(output_dir) / "checkpoints")
    train_args_dict["logging_dir"] = str(Path(output_dir) / "logs")
    allowed_train_keys = {field.name for field in fields(CustomSFTConfig)}
    train_args_dict = {
        key: value for key, value in train_args_dict.items() if key in allowed_train_keys
    }
    train_args = CustomSFTConfig(
        **train_args_dict,
        max_length=model_cfg.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    with accelerator.local_main_process_first():
        adapter = build_model_adapter(cfg)
        train_set, val_set, _ = get_canonical_mesh_dataset(cfg)

    sig_cb = SaveAndStopOnSignalCallback()
    trainer = CustomSFTTrainer(
        model=adapter.model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=get_mesh_data_collator(data_cfg, model_cfg),
        processing_class=MeshProcessor(model_cfg),
        callbacks=[
            JsonlLoggerCallback(log_file_path=train_args.logging_dir),
            sig_cb,
        ],
    )
    resume_checkpoint = (
        get_last_checkpoint(train_args.output_dir) if resume_from_checkpoint else None
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    if not sig_cb.signal_received:
        trainer.save_model((Path(train_args.output_dir) / "final").as_posix())
    trainer.accelerator.wait_for_everyone()
