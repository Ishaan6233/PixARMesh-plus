"""Registry for config-driven model adapter loading.

This file is the only place benchmark model keys should map to adapter classes.
Changing a model should be a config change, not a training/eval code change.

Related files:
- Adapter classes come from `models/adapters/src_adapters.py`.
- `main.py` and `trainers/canonical.py` call `build_model_adapter`.
- Model keys are selected by `configs/model/*.yaml`.
"""

from __future__ import annotations

from typing import Any, Type

from models.adapters import BPTAdapter, EdgeRunnerAdapter, MeshXLAdapter, PixARMeshAdapter
from models.base import BaseModelAdapter


MODEL_REGISTRY: dict[str, Type[BaseModelAdapter]] = {
    "pixarmesh": PixARMeshAdapter,
    "bpt": BPTAdapter,
    "edgerunner": EdgeRunnerAdapter,
    "meshxl": MeshXLAdapter,
}


def get_model_adapter_class(name: str) -> Type[BaseModelAdapter]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model adapter {name!r}. Available: {available}") from exc


def build_model_adapter(cfg: Any) -> BaseModelAdapter:
    name = str(cfg.model.name)
    return get_model_adapter_class(name)(cfg)
