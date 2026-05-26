"""Canonical dataset wrapper around existing src data loading.

This file bridges the FAIR benchmark layer to the legacy PixARMesh dataset
implementation while enforcing the shared dataset contract.

Related files:
- Calls `src.data.mesh.get_mesh_dataset` for actual loading/transforms.
- Validates `utils/specs.py::DatasetSpec` against `src.utils.config.DataConfig`.
- Uses `utils/splits.py` to enforce immutable split manifests.
- Used by `trainers/canonical.py` for benchmark-governed training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from utils.specs import DatasetSpec
from utils.splits import read_split


def _filter_by_uid(dataset: Any, allowed_uids: set[str]) -> Any:
    if not allowed_uids:
        raise ValueError("Split manifest is empty.")
    return dataset.filter(lambda example: str(example["uid"]) in allowed_uids)


def get_canonical_mesh_dataset(cfg: Any):
    """Return standardized train/val/test datasets via existing src transforms."""
    from src.data.mesh import get_mesh_dataset
    from src.utils.config import DataConfig

    dataset_spec = DatasetSpec.from_mapping(
        OmegaConf.to_container(cfg.dataset.spec, resolve=True)
    )
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.dataset.src_data, resolve=True))
    dataset_spec.assert_data_config(data_cfg)

    train_set, val_set, test_set = get_mesh_dataset(data_cfg)
    split_dir = Path(dataset_spec.split_dir)
    train_set = _filter_by_uid(train_set, read_split(split_dir, "train"))
    val_set = _filter_by_uid(val_set, read_split(split_dir, "val"))
    test_set = _filter_by_uid(test_set, read_split(split_dir, "test"))
    return train_set, val_set, test_set
