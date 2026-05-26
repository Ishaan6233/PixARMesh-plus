"""Model adapter package exports.

This package exposes registry helpers so benchmark code can load models by
config key without depending on concrete implementation files.

Related files:
- Registry implementation: `models/registry.py`.
- Adapter implementations: `models/adapters/`.
"""

from .registry import MODEL_REGISTRY, build_model_adapter, get_model_adapter_class

__all__ = ["MODEL_REGISTRY", "build_model_adapter", "get_model_adapter_class"]
