"""Instance-discovery method registry for the MV segmentation tournament.

This is the single plug-in boundary the segmentation-tournament plan hangs off
of. Every registered method shares the EXACT call/return contract of the
baseline ``discover_instance_points_mv``
(``src/models/frozen_geo_encoder.py``): it consumes per-view frozen-geometry points +
panoptic masks + a scene-space seed cloud and returns scene-space ``obj_voxels``
/ ``ctx_voxels``. Return arity is controlled by the ``return_diagnostics`` /
``return_target_ids`` flags, identically to the baseline::

    return_target_ids=True                  -> (obj, ctx, target_ids, obj_geom)
    return_diagnostics=True                 -> (obj, ctx, diag)
    return_diagnostics & return_target_ids  -> (obj, ctx, diag, target_ids)
    default                                 -> (obj, ctx)

Selecting a method (training/inference and the eval_voxels harness both do this)::

    from src.models.discovery import get_discovery_fn
    fn = get_discovery_fn(cfg.mv_discovery_method)        # default "consensus"
    obj, ctx, tids, geom = fn(..., return_target_ids=True)

Adding a Family-A candidate (Stage 1 of docs/ plan): create a module in this
package, implement a function with the contract above (it may import the shared
helpers from ``frozen_geo_encoder`` — ``_get_per_view_target_ids``,
``_batched_obj_fps``, etc.), decorate it ``@register_discovery("name")``, and
import the module below so registration runs at import time.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from src.models.frozen_geo_encoder import discover_instance_points_mv

_REGISTRY: Dict[str, Callable] = {}


def register_discovery(name: str) -> Callable[[Callable], Callable]:
    """Decorator registering an instance-discovery function under ``name``."""

    def _wrap(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"discovery method '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return _wrap


def get_discovery_fn(name: Optional[str]) -> Callable:
    """Resolve a discovery method by name (``None`` -> the ``consensus`` baseline)."""
    key = name or "consensus"
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown mv_discovery_method '{key}'; available: {available_methods()}"
        )
    return _REGISTRY[key]


def available_methods() -> List[str]:
    return sorted(_REGISTRY)


# Baseline: cross-view Grounded-SAM mask-consensus voting — the current MV
# operating point. Registering it (rather than special-casing) keeps the
# default path byte-identical while making the harness method-agnostic.
_REGISTRY["consensus"] = discover_instance_points_mv

# --- Register Family-A candidate modules below (each module self-registers) ---
# from . import maskclustering  # noqa: F401  (added in Stage 1)
