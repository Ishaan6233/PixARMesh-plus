"""Base model-adapter interface that defines the benchmark fairness boundary.

Adapters translate between canonical benchmark batches and model-specific
implementations. They may reshape/tokenize, but must not alter shared benchmark
policy such as normalization, metrics, splits, or runtime protocol.

Related files:
- Concrete adapters live in `models/adapters/`.
- `models/registry.py` exposes adapters by config key.
- `main.py`, `trainers/canonical.py`, and `runtime/benchmark.py` interact with
  models through this boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - supports config-only environments.
    torch = None

    class _Module:
        pass

    class nn:  # type: ignore[no-redef]
        Module = _Module


class BaseModelAdapter(nn.Module, ABC):
    """Fairness boundary between benchmark code and model implementations."""

    model_key: str = "base"

    def __init__(self, cfg: Any | None = None):
        super().__init__()
        self.cfg = cfg

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError

    def postprocess(self, outputs: Any) -> Any:
        return outputs

    def compute_loss(self, outputs: Any, batch: dict[str, Any]) -> Any:
        if hasattr(outputs, "loss"):
            return outputs.loss
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        raise ValueError("Adapter output does not expose a loss.")

    def infer(self, batch: dict[str, Any]) -> Any:
        processed = self.preprocess(batch)
        outputs = self.forward(processed)
        return self.postprocess(outputs)

    def validate_fairness(self, dataset_spec: Any, evaluation_spec: Any) -> None:
        dataset_spec.validate()
        evaluation_spec.validate()
