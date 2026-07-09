# File Structure Guide

Use this guide when deciding where new files belong. Keep placement based on
semantic purpose and runtime ownership.

## Agent And Navigation Files

- `AGENTS.md`: shared instructions for coding agents.
- `CLAUDE.md`: Claude-specific working notes and command reminders.
- `docs/FILE_STRUCTURE.md`: this placement guide.
- `graphify/`, `graphify-out/`: generated local code-graph artifacts. They are
  ignored by git and should be used only as navigation hints.
- `scripts/graphify_refresh.sh`: refreshes `graphify-out/graph.json` after
  material file-structure changes and stores its local fingerprint in
  `graphify-out/`.

## Runtime Code

- `train.py`, `launch.py`, `main.py`: top-level executable entry points.
- `src/data/`: dataset loaders, transforms, collators, tokenization-facing data
  contracts, and dataset-side validation.
- `src/models/`: model classes, encoders, loss/generation behavior, and
  model-owned conditioning logic.
- `src/models/discovery/`: object discovery, view selection, voxel sampling,
  and other geometry-discovery modules used by MV conditioning.
- `src/utils/`: reusable runtime utilities shared across training, inference,
  configs, checkpoint handling, or evaluation paths.
- `models/adapters/`: adapters around external model components or legacy
  model interfaces.

## Configuration

- `configs/`: Hydra configs for training/eval/runtime composition.
- `configs/model/`: model-family config fragments.
- `configs/dataset/`: dataset contracts and data-path config fragments.
- `configs/train/`: trainer arguments and optimizer/scheduler defaults.
- `configs/runtime/`, `configs/environment/`: machine/runtime-specific config.
- `configs/eval/`, `configs/experiment/`, `configs/robustness/`: evaluation,
  experiment, and robustness-specific config groups.

## Scripts

- `scripts/`: maintained command-line workflows.
- `scripts/data/`: dataset preparation, conversion, filtering, or inspection.
- `scripts/eval/`: evaluation probes and metric-specific entry points.
- `scripts/experiments/`: bounded experiment runners that are useful to repeat.
- `scripts/figures/`: figure generation from existing results.
- `scripts/vis/`: visualization and diagnostic rendering scripts.

If a script becomes imported by runtime code, move the reusable logic into
`src/` and keep `scripts/` as a thin CLI wrapper.

## Tests

- `tests/`: focused pytest coverage for runtime contracts and regressions.
- New behavior in `src/data/` should usually get data-contract or collator
  tests.
- New model/config behavior should usually get config, registry, or runtime
  tests.
- New metrics/eval behavior should usually get deterministic small-input tests.

## Local Data And Generated Outputs

- `metadata/`, `splits/`: small repo-level metadata and split definitions.
- `datasets/`: local datasets; ignored by git.
- `checkpoints/`: local checkpoint roots. Track only policy docs and directory
  markers; do not track checkpoint payloads.
- `outputs/`: local experiment outputs. Active work is grouped under `da3/` and
  `sv/`; legacy material lives under `archive/`.
- `results/`, `figures/`, `logs/`, `debug/`, `artifacts/`: generated local
  outputs unless a specific file is intentionally promoted to docs or tests.

## Placement Rules

- Put importable production behavior in `src/`, not `scripts/`.
- Put one-off or operator-facing commands in `scripts/`.
- Put Hydra changes in the narrowest matching `configs/` group.
- Put durable explanations in `docs/`; keep generated evidence in `outputs/`.
- Put tests next to the behavior they protect conceptually, using `tests/`.
- Do not add generated checkpoint, dataset, graphify, result, or figure payloads
  to git.
- After material file-structure changes, run
  `bash scripts/graphify_refresh.sh` before finishing the change.
