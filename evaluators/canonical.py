"""Shared evaluator that calls only centralized benchmark metrics.

This evaluator is the fairness gate for metric computation. Benchmark scripts
and model adapters should call this instead of implementing their own metrics.

Related files:
- Metric implementations live in `metrics/`.
- Metric policy comes from `utils/specs.py::EvaluationSpec` and
  `configs/eval/canonical.yaml`.
- `main.py` validates and instantiates this evaluator for eval tasks.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from metrics.chamfer import chamfer_distance
from metrics.fscore import fscore
from metrics.normals import normal_consistency
from utils.specs import EvaluationSpec


class CanonicalEvaluator:
    def __init__(self, spec: EvaluationSpec):
        spec.validate()
        self.spec = spec

    def evaluate_point_clouds(
        self,
        predicted_points: Any,
        target_points: Any,
        predicted_normals: Any | None = None,
        target_normals: Any | None = None,
    ) -> dict[str, float | int | str | bool]:
        results: dict[str, float | int | str | bool] = {
            "sample_points": self.spec.sample_points,
            "chamfer_squared": self.spec.chamfer_squared,
            "fscore_threshold": self.spec.fscore_threshold,
        }
        cd = chamfer_distance(
            predicted_points,
            target_points,
            squared=self.spec.chamfer_squared,
            reduction=self.spec.chamfer_reduction,  # type: ignore[arg-type]
        )
        results["chamfer"] = float(cd.detach().cpu().item() if hasattr(cd, "detach") else cd)
        results["fscore"] = fscore(
            predicted_points,
            target_points,
            threshold=self.spec.fscore_threshold,
            squared=self.spec.chamfer_squared,
        )
        if self.spec.normals_enabled and predicted_normals is not None and target_normals is not None:
            nc = normal_consistency(predicted_normals, target_normals)
            results["normal_consistency"] = float(
                nc.detach().cpu().item() if hasattr(nc, "detach") else nc
            )
        return results

    def metadata(self) -> dict[str, Any]:
        return asdict(self.spec)
