"""Canonical Chamfer distance implementation for FAIR evaluation.

All benchmarked models must use this implementation through shared evaluators;
model-specific Chamfer code is not allowed.

Related files:
- `configs/eval/canonical.yaml` defines squared and reduction policy.
- `evaluators/canonical.py` calls this metric.
- `tests/test_metrics.py` covers basic expected behavior.
"""

from __future__ import annotations

from typing import Literal


def chamfer_distance(
    points_a,
    points_b,
    *,
    squared: bool = True,
    reduction: Literal["mean", "none"] = "mean",
    single_directional: bool = False,
):
    """Canonical Chamfer distance.

    Inputs are tensors shaped [B, N, 3] or [N, 3]. Distances are squared by
    default to match the benchmark config.
    """
    import torch

    if points_a.ndim == 2:
        points_a = points_a.unsqueeze(0)
    if points_b.ndim == 2:
        points_b = points_b.unsqueeze(0)
    if points_a.shape[-1] != points_b.shape[-1]:
        raise ValueError("Point clouds must have the same feature dimension.")

    distances = torch.cdist(points_a, points_b, p=2)
    if squared:
        distances = distances.square()
    a_to_b = distances.min(dim=-1).values
    if single_directional:
        per_batch = a_to_b.mean(dim=-1)
    else:
        b_to_a = distances.min(dim=-2).values
        per_batch = a_to_b.mean(dim=-1) + b_to_a.mean(dim=-1)
    if reduction == "none":
        return per_batch
    if reduction != "mean":
        raise ValueError("Canonical Chamfer supports reduction='mean' or 'none'.")
    return per_batch.mean()
