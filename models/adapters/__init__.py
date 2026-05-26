"""Adapter exports for existing PixARMesh-family implementations.

This package currently exposes wrappers around legacy `src.models` loaders.

Related files:
- Concrete classes are in `models/adapters/src_adapters.py`.
- Exported classes are registered by `models/registry.py`.
"""

from .src_adapters import BPTAdapter, EdgeRunnerAdapter, MeshXLAdapter, PixARMeshAdapter

__all__ = ["PixARMeshAdapter", "BPTAdapter", "EdgeRunnerAdapter", "MeshXLAdapter"]
