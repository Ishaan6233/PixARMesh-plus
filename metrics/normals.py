"""Canonical normal-consistency metric for FAIR evaluation.

This file owns the shared normal metric used when normal evaluation is enabled.

Related files:
- `configs/eval/canonical.yaml` toggles normal evaluation.
- `evaluators/canonical.py` calls this metric when normals are provided.
"""

from __future__ import annotations


def normal_consistency(normals_a, normals_b, *, eps: float = 1e-12):
    """Mean absolute cosine similarity between paired normals."""
    import torch

    if normals_a.shape != normals_b.shape:
        raise ValueError("Normal tensors must have identical shapes.")
    normals_a = torch.nn.functional.normalize(normals_a, dim=-1, eps=eps)
    normals_b = torch.nn.functional.normalize(normals_b, dim=-1, eps=eps)
    return torch.abs((normals_a * normals_b).sum(dim=-1)).mean()
