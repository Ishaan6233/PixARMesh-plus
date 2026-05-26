"""Canonical F-score implementation for FAIR evaluation.

All benchmarked models must use this implementation through shared evaluators;
model-specific F-score code is not allowed.

Related files:
- `configs/eval/canonical.yaml` defines the F-score threshold.
- `evaluators/canonical.py` calls this metric.
- `tests/test_metrics.py` covers basic expected behavior.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _as_numpy(points) -> np.ndarray:
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float64)


def fscore(points_a, points_b, *, threshold: float = 0.01, squared: bool = True) -> float:
    """Canonical F-score for two point clouds."""
    if threshold <= 0:
        raise ValueError("F-score threshold must be positive.")
    a = _as_numpy(points_a)
    b = _as_numpy(points_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("F-score expects [N, D] point arrays.")
    dist_ab, _ = cKDTree(b).query(a)
    dist_ba, _ = cKDTree(a).query(b)
    if squared:
        dist_ab = dist_ab**2
        dist_ba = dist_ba**2
    precision = np.mean(dist_ab <= threshold)
    recall = np.mean(dist_ba <= threshold)
    denom = precision + recall
    if denom <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / denom)
