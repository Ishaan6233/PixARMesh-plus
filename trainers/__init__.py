"""Reusable trainer exports for benchmark-governed training.

This package exposes the adapter that runs existing PixARMesh training under
the canonical benchmark wrapper.

Related files:
- Implementation: `trainers/canonical.py`.
- Legacy trainer: `src/utils/trainer.py`.
"""

from .canonical import train_with_existing_stack

__all__ = ["train_with_existing_stack"]
