"""Fairness specs for datasets, normalization, coordinates, and evaluation.

This file is the benchmark contract. Other modules should validate against
these dataclasses instead of inventing per-model assumptions.

Related files:
- `configs/dataset/canonical_3d_front.yaml` maps into `DatasetSpec`.
- `configs/eval/canonical.yaml` maps into `EvaluationSpec`.
- `utils/canonical_dataset.py` enforces `DatasetSpec` before loading data.
- `evaluators/canonical.py` consumes `EvaluationSpec` before computing metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CoordinateFrameSpec:
    handedness: str = "right"
    up_axis: str = "y"
    forward_axis: str = "z"
    units: str = "meters"

    def validate(self) -> None:
        if self.handedness != "right":
            raise ValueError("Benchmark coordinate frame must be right-handed.")
        if self.up_axis != "y":
            raise ValueError("Benchmark coordinate frame must be Y-up.")
        if self.forward_axis != "z":
            raise ValueError("Benchmark coordinate frame must be Z-forward.")
        if self.units != "meters":
            raise ValueError("Benchmark unit scale must be meters.")


@dataclass(frozen=True)
class NormalizationSpec:
    mode: str = "bbox"
    bound: float = 0.95
    center: bool = True
    preserve_aspect_ratio: bool = True

    def validate(self) -> None:
        if self.mode != "bbox":
            raise ValueError("Benchmark normalization mode must be bbox.")
        if abs(float(self.bound) - 0.95) > 1e-12:
            raise ValueError("Benchmark normalization bound must be exactly 0.95.")
        if not self.center:
            raise ValueError("Benchmark normalization must center geometry.")
        if not self.preserve_aspect_ratio:
            raise ValueError("Benchmark normalization must preserve aspect ratio.")


@dataclass(frozen=True)
class DatasetSpec:
    resolution: int = 518
    num_points: int = 4096
    local_obj_num_points: int = 2048
    num_ctx_points: int = 0
    normalization: NormalizationSpec = field(default_factory=NormalizationSpec)
    coordinate_frame: CoordinateFrameSpec = field(default_factory=CoordinateFrameSpec)
    axis_convention: str = "y_up_z_forward"
    unit_scale: float = 1.0
    augmentation_policy: str = "canonical_train_only"
    split_dir: str = "splits"

    def validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("Image resolution must be positive.")
        if self.num_points <= 0:
            raise ValueError("num_points must be positive.")
        if self.local_obj_num_points < -1:
            raise ValueError("local_obj_num_points must be -1 or non-negative.")
        if self.num_ctx_points < 0:
            raise ValueError("num_ctx_points must be non-negative.")
        if self.axis_convention != "y_up_z_forward":
            raise ValueError("axis_convention must be y_up_z_forward.")
        if abs(float(self.unit_scale) - 1.0) > 1e-12:
            raise ValueError("unit_scale must be 1.0 meters.")
        self.normalization.validate()
        self.coordinate_frame.validate()

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "DatasetSpec":
        norm = cfg.get("normalization", {})
        coord = cfg.get("coordinate_frame", {})
        spec = cls(
            resolution=int(cfg.get("resolution", 518)),
            num_points=int(cfg.get("num_points", 4096)),
            local_obj_num_points=int(cfg.get("local_obj_num_points", 2048)),
            num_ctx_points=int(cfg.get("num_ctx_points", 0)),
            normalization=NormalizationSpec(**norm),
            coordinate_frame=CoordinateFrameSpec(**coord),
            axis_convention=str(cfg.get("axis_convention", "y_up_z_forward")),
            unit_scale=float(cfg.get("unit_scale", 1.0)),
            augmentation_policy=str(
                cfg.get("augmentation_policy", "canonical_train_only")
            ),
            split_dir=str(cfg.get("split_dir", "splits")),
        )
        spec.validate()
        return spec

    def assert_data_config(self, data_cfg: Any) -> None:
        self.validate()
        checks = {
            "num_points": self.num_points,
            "norm_bound": self.normalization.bound,
            "local_obj_num_points": self.local_obj_num_points,
            "num_ctx_points": self.num_ctx_points,
        }
        for name, expected in checks.items():
            if hasattr(data_cfg, name):
                observed = getattr(data_cfg, name)
                if isinstance(expected, float):
                    ok = abs(float(observed) - expected) <= 1e-12
                else:
                    ok = int(observed) == int(expected)
                if not ok:
                    raise ValueError(
                        f"Dataset config mismatch for {name}: "
                        f"expected {expected}, observed {observed}."
                    )


@dataclass(frozen=True)
class EvaluationSpec:
    sample_points: int = 100000
    chamfer_squared: bool = True
    chamfer_reduction: str = "mean"
    fscore_threshold: float = 0.01
    normals_enabled: bool = True
    mesh_sampling_method: str = "area_weighted"
    mesh_sampling_seed: int = 0
    remove_degenerate_faces: bool = True
    remove_duplicate_vertices: bool = True
    watertight_required: bool = False

    def validate(self) -> None:
        if self.sample_points <= 0:
            raise ValueError("Evaluation sample_points must be positive.")
        if self.chamfer_reduction != "mean":
            raise ValueError("Canonical Chamfer reduction must be mean.")
        if self.fscore_threshold <= 0:
            raise ValueError("F-score threshold must be positive.")
        if self.mesh_sampling_method != "area_weighted":
            raise ValueError("Canonical mesh sampling must be area_weighted.")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "EvaluationSpec":
        chamfer = cfg.get("chamfer", {})
        fscore = cfg.get("fscore", {})
        normals = cfg.get("normals", {})
        sampling = cfg.get("mesh_sampling", {})
        cleanup = cfg.get("mesh_cleanup", {})
        spec = cls(
            sample_points=int(cfg.get("sample_points", 100000)),
            chamfer_squared=bool(chamfer.get("squared", True)),
            chamfer_reduction=str(chamfer.get("reduction", "mean")),
            fscore_threshold=float(fscore.get("threshold", 0.01)),
            normals_enabled=bool(normals.get("enabled", True)),
            mesh_sampling_method=str(sampling.get("method", "area_weighted")),
            mesh_sampling_seed=int(sampling.get("seed", 0)),
            remove_degenerate_faces=bool(
                cleanup.get("remove_degenerate_faces", True)
            ),
            remove_duplicate_vertices=bool(
                cleanup.get("remove_duplicate_vertices", True)
            ),
            watertight_required=bool(cleanup.get("watertight_required", False)),
        )
        spec.validate()
        return spec
