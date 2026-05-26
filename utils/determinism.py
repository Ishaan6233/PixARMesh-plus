"""Determinism setup for FAIR benchmark runs.

This module centralizes seeds and deterministic backend flags so training,
evaluation, and runtime benchmarking share the same policy.

Related files:
- `main.py` calls `setup_determinism` once per run.
- `configs/config.yaml` provides the seed and strictness settings.
- `utils/artifacts.py` saves the resulting determinism report.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeterminismReport:
    seed: int
    deterministic_algorithms: bool
    tf32_allowed: bool
    cublas_workspace_config: str


def setup_determinism(seed: int, strict: bool = True) -> DeterminismReport:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except Exception:
        return DeterminismReport(
            seed=seed,
            deterministic_algorithms=False,
            tf32_allowed=False,
            cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(strict, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    return DeterminismReport(
        seed=seed,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        tf32_allowed=bool(torch.backends.cuda.matmul.allow_tf32),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
    )
