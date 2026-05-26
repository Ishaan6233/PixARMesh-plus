"""Canonical evaluator package exports.

This package exposes the shared evaluator that benchmark scripts should use for
all metric computation.

Related files:
- Implementation: `evaluators/canonical.py`.
- Metrics: `metrics/`.
"""

from .canonical import CanonicalEvaluator

__all__ = ["CanonicalEvaluator"]
