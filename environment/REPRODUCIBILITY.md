# Reproducibility And Fairness Contract

## Tier 1: Research Reproducibility

Same trends and same conclusions are expected. This tier requires fixed seeds,
fixed split manifests, fixed configs, pinned Python dependencies, and
deterministic evaluation.

## Tier 2: Numerical Reproducibility

This is the benchmark standard. Runs must use the canonical CUDA 12.4
environment with pinned Torch, TorchVision, PyTorch3D, Triton, flash-attn,
deterministic kernels, disabled TF32, fixed matmul precision, and fixed AMP
policy.

## Tier 3: Bitwise Reproducibility

Identical checkpoints, logs, and weights are documented as a goal but are not
required across H100, H200, and A100 clusters.

## Fairness Rule

MODEL = interchangeable

BENCHMARK = immutable

Adapters may reshape, tokenize, or convert formats, but may not change splits,
normalization, coordinate systems, point counts, metrics, evaluation sampling,
precision policy, or runtime timing protocol.

