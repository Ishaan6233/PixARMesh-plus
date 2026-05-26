"""Environment capture and hashing for reproducible benchmark artifacts.

This module records the software/hardware state that makes a benchmark run
auditable and comparable.

Related files:
- `main.py` captures this metadata at run start.
- `utils/artifacts.py` writes the captured metadata to `outputs/.../environment`.
- `configs/environment/h200_cu124.yaml` declares the intended canonical stack.
- `environment/requirements.lock.txt` and Docker files define that stack.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": None,
        "cuda_runtime": None,
        "cudnn_version": None,
        "gpu_name": None,
        "driver_version": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }
    try:
        import torch
    except Exception as exc:
        info["torch_import_error"] = repr(exc)
        return info

    info["torch_version"] = getattr(torch, "__version__", None)
    info["cuda_runtime"] = getattr(torch.version, "cuda", None)
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn_version"] = None
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def capture_environment(repo_root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(repo_root)
    dirty = _run(["git", "-C", str(repo_root), "status", "--short"])
    env = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "flash_attn_version": _version("flash_attn"),
        "torchvision_version": _version("torchvision"),
        "triton_version": _version("triton"),
        "pytorch3d_version": _version("pytorch3d"),
        "pytorch3d_commit": "75ebeeaea0908c5527e7b1e305fbc7681382db47",
        "git_commit": _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "git_dirty": bool(dirty),
    }
    env.update(_torch_info())
    env["environment_hash"] = environment_hash(env)
    return env


def environment_hash(environment: dict[str, Any]) -> str:
    payload = json.dumps(environment, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()
