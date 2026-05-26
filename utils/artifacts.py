"""Artifact layout, config saving, JSON writing, and run-id helpers.

This module owns the benchmark output directory contract.

Related files:
- `main.py` uses these helpers to create `outputs/{date}/{experiment}/...`.
- `utils/environment.py` supplies the environment hash used in run IDs.
- Hydra configs under `configs/` are saved here as resolved artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


ARTIFACT_SUBDIRS = (
    "config",
    "checkpoints",
    "logs",
    "metrics",
    "meshes",
    "renders",
    "runtime",
    "environment",
)


def make_output_dir(base_dir: str, experiment: str, date: str | None = None) -> Path:
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(base_dir) / date / experiment
    for subdir in ARTIFACT_SUBDIRS:
        (out / subdir).mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True, default=str)
        fp.write("\n")


def save_resolved_config(cfg: DictConfig, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config" / "resolved_config.yaml", resolve=True)
    overrides = cfg.get("hydra_overrides", [])
    OmegaConf.save(
        OmegaConf.create({"overrides": list(overrides)}),
        output_dir / "config" / "overrides.yaml",
        resolve=True,
    )


def compute_run_id(resolved_config: str, git_commit: str | None, environment_hash: str) -> str:
    payload = f"{resolved_config}\n{git_commit or 'unknown'}\n{environment_hash}"
    return sha256(payload.encode("utf-8")).hexdigest()
