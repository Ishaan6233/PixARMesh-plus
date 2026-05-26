"""Adapters that wrap existing src model loaders without changing benchmark policy.

These classes keep existing PixARMesh/BPT/EdgeRunner/MeshXL implementations
intact while making them swappable through the benchmark registry.

Related files:
- Calls `src.models.utils.get_model` and condition-encoder helpers.
- Uses `src.utils.config.ModelConfig` for legacy model construction.
- Registered in `models/registry.py`.
- Configured by `configs/model/*.yaml`.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from omegaconf import OmegaConf

from models.base import BaseModelAdapter


class SrcModelAdapter(BaseModelAdapter):
    """Thin adapter around the existing src.models loader."""

    model_key = "src"
    ar_model_type = "meshxl"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.model = None
        self._build_model()

    def _build_model(self) -> None:
        from src.models.utils import (
            get_condition_encoder,
            get_image_condition_encoder,
            get_model,
        )
        from src.utils.config import ModelConfig

        allowed = {field.name for field in fields(ModelConfig)}
        model_cfg_data = {
            key: value
            for key, value in OmegaConf.to_container(self.cfg.model, resolve=True).items()
            if key in allowed
        }
        model_cfg_data["ar_model_type"] = self.ar_model_type
        model_cfg = ModelConfig(**model_cfg_data)

        cond_encoder_img = (
            get_image_condition_encoder(model_cfg) if model_cfg.img_cond else None
        )
        cond_encoder = (
            get_condition_encoder(
                model_cfg.local_cond_path,
                model_cfg,
                cond_encoder_img=cond_encoder_img,
            )
            if model_cfg.cond
            else None
        )
        self.model = get_model(
            model_cfg.local_path,
            model_cfg,
            cond_encoder=cond_encoder,
            cond_encoder_img=cond_encoder_img,
        )

    def forward(self, batch: dict[str, Any]) -> Any:
        return self.model(**batch)

    def infer(self, batch: dict[str, Any]) -> Any:
        if hasattr(self.model, "generate"):
            return self.model.generate(**self.preprocess(batch))
        return super().infer(batch)


class PixARMeshAdapter(SrcModelAdapter):
    model_key = "pixarmesh"
    ar_model_type = "meshxl"


class MeshXLAdapter(SrcModelAdapter):
    model_key = "meshxl"
    ar_model_type = "meshxl"


class BPTAdapter(SrcModelAdapter):
    model_key = "bpt"
    ar_model_type = "bpt"


class EdgeRunnerAdapter(SrcModelAdapter):
    model_key = "edgerunner"
    ar_model_type = "edgerunner"
