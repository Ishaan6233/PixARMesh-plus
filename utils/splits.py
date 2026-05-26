"""Immutable split-manifest readers for canonical train/val/test governance.

This module prevents train/eval code from dynamically generating splits.

Related files:
- `scripts/generate_splits.py` creates `splits/train.txt`, `val.txt`,
  and `test.txt` once.
- `utils/canonical_dataset.py` reads these manifests to filter loaded datasets.
- `configs/dataset/canonical_3d_front.yaml` points `DatasetSpec.split_dir` here.
"""

from __future__ import annotations

from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


def read_split(split_dir: str | Path, split_name: str) -> set[str]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split_name!r}; expected one of {SPLIT_NAMES}.")
    path = Path(split_dir) / f"{split_name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing immutable split manifest: {path}. "
            "Generate it once with scripts/generate_splits.py."
        )
    with path.open() as fp:
        return {line.strip() for line in fp if line.strip()}


def assert_splits_exist(split_dir: str | Path) -> None:
    for split_name in SPLIT_NAMES:
        read_split(split_dir, split_name)
