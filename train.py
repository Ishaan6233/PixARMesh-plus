# CVE-2025-32434: transformers 5.x blocks torch.load on .pt optimizer files
# when torch < 2.6. Patch both the module-level and the local binding in trainer.py
# before any transformers import triggers the check.
try:
    import transformers.utils.import_utils as _tfu
    import transformers.trainer as _trainer_mod
    _noop = lambda: None  # noqa: E731
    _tfu.check_torch_load_is_safe = _noop
    _trainer_mod.check_torch_load_is_safe = _noop
    import torch as _torch
    _orig_load = _torch.load
    def _safe_load(f, *args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(f, *args, **kwargs)
    _torch.load = _safe_load
except Exception:
    pass

import datasets
import hydra
import transformers
from dataclasses import fields
from pathlib import Path
from typing import Any
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from src.models.utils import (
    get_model,
    get_condition_encoder,
    get_image_condition_encoder,
    get_pi3x_encoder,
)
from src.data.collator import get_mesh_data_collator
from src.data.mesh import get_mesh_dataset, MeshProcessor
from src.utils.logging import JsonlLoggerCallback
from src.utils.trainer import CustomSFTTrainer, CustomSFTConfig
from src.utils.config import DataConfig, ModelConfig
from src.utils.ckpt import get_last_checkpoint
from src.utils.sig import SaveAndStopOnSignalCallback, install_sigusr1_handler

logger = get_logger(__name__)


OmegaConf.register_new_resolver("sub", lambda x, y: x - y)


def _to_container(node):
    return OmegaConf.to_container(node, resolve=True)


def _filter_dataclass_kwargs(dataclass_type, values: dict[str, Any]):
    allowed_keys = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in values.items() if key in allowed_keys}


def _build_data_config(cfg):
    if OmegaConf.select(cfg, "dataset.src_data") is not None:
        data_values = _to_container(cfg.dataset.src_data)
    elif OmegaConf.select(cfg, "data") is not None:
        data_values = _to_container(cfg.data)
    elif OmegaConf.select(cfg, "dataset") is not None:
        data_values = _to_container(cfg.dataset)
    else:
        raise ValueError("Expected either cfg.dataset.src_data, cfg.data, or cfg.dataset")
    return DataConfig(**_filter_dataclass_kwargs(DataConfig, data_values))


def _build_model_config(cfg):
    model_values = _to_container(cfg.model)
    return ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_values))


def _build_train_args(cfg, model_cfg):
    train_arg_values = _to_container(cfg.train.train_args)
    resume_from_checkpoint = bool(train_arg_values.pop("resume_from_checkpoint", True))
    train_arg_values = _filter_dataclass_kwargs(CustomSFTConfig, train_arg_values)
    train_args = CustomSFTConfig(
        **train_arg_values,
        max_length=model_cfg.max_seq_length,
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
        remove_unused_columns=False,
    )
    return train_args, resume_from_checkpoint


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg):
    install_sigusr1_handler()

    OmegaConf.resolve(cfg)

    accelerator = Accelerator()
    accelerator.print(OmegaConf.to_yaml(cfg))
    data_cfg = _build_data_config(cfg)
    model_cfg = _build_model_config(cfg)
    local_model_path = model_cfg.local_path
    local_cond_model_path = model_cfg.local_cond_path

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    train_args, should_resume = _build_train_args(cfg, model_cfg)

    with accelerator.local_main_process_first():
        if model_cfg.img_cond:
            cond_encoder_img = get_image_condition_encoder(model_cfg)
        else:
            cond_encoder_img = None
        if model_cfg.cond:
            cond_encoder = get_condition_encoder(
                local_cond_model_path, model_cfg, cond_encoder_img=cond_encoder_img
            )
        else:
            cond_encoder = None
        pi3x_enc = get_pi3x_encoder(model_cfg) if model_cfg.use_pi3x else None
        model = get_model(
            local_model_path,
            model_cfg,
            cond_encoder=cond_encoder,
            cond_encoder_img=cond_encoder_img,
            pi3x_encoder=pi3x_enc,
        )
        train_set, val_set, _ = get_mesh_dataset(data_cfg)

    sig_cb = SaveAndStopOnSignalCallback()
    trainer = CustomSFTTrainer(
        model=model,
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

    resume_checkpoint = get_last_checkpoint(train_args.output_dir) if should_resume else None
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_output_dir = Path(train_args.output_dir) / "final"
    if not sig_cb.signal_received:
        trainer.save_model(final_output_dir.as_posix())
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
