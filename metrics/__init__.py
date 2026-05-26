"""Canonical metric exports used by all benchmark evaluators.

This package is the only allowed benchmark metric surface.

Related files:
- `evaluators/canonical.py` imports metrics from here.
- Metric policy is configured by `configs/eval/canonical.yaml`.
"""

from .chamfer import chamfer_distance
from .fscore import fscore
from .normals import normal_consistency

__all__ = ["chamfer_distance", "fscore", "normal_consistency"]
