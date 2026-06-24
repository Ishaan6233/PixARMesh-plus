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
from src.utils.config import DataConfig, ModelConfig, mv_prefix_len
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
    # The multi-view dataset config (dataset/canonical_3d_front_multiview.yaml) carries
    # the MV model overrides (prefix_len=322, mv_voxel_encoder=True, mv_num_*_queries...)
    # under cfg.dataset.model. Merge them over cfg.model, otherwise training silently
    # builds a single-view model (the MV encoder is never created).
    ds_model = OmegaConf.select(cfg, "dataset.model")
    if ds_model is not None:
        model_values.update(_to_container(ds_model))
    model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_values))
    # Derive prefix_len from the active conditioning channels (single source of truth
    # in mv_prefix_len) so the collator matches what the model produces — avoids a
    # hardcoded prefix_len drifting from pc_latent_len / query counts.
    if getattr(model_cfg, "mv_voxel_encoder", False) or model_cfg.mv_obj_pc_cond:
        model_cfg.prefix_len = mv_prefix_len(model_cfg)
    return model_cfg


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
        if getattr(model_cfg, "freeze_decoder", False):
            # Test-2 / frozen-decoder regime: train ONLY the mv_voxel_encoder.
            n_train = 0
            for name, p in model.named_parameters():
                p.requires_grad_("mv_voxel_encoder" in name)
                n_train += p.requires_grad
            logger.info(
                f"freeze_decoder=True: {n_train} mv_voxel_encoder tensors trainable, "
                "rest frozen."
            )
        train_set, val_set, _ = get_mesh_dataset(data_cfg)
        if getattr(data_cfg, "overfit_n", 0):
            # Test-2: overfit a tiny fixed subset; eval on the same objects.
            train_set = train_set.select(range(data_cfg.overfit_n))
            val_set = train_set
            logger.info(f"overfit_n={data_cfg.overfit_n}: training on {len(train_set)} objects")

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
