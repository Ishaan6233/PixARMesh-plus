# Outputs Layout

`outputs/` is local experiment storage. Checkpoint payloads, generated meshes,
eval dumps, diagnostics, and visualizations are ignored by git.

- `da3/` contains active da3 work: training runs under `train/`, diagnostics
  under `diagnostics/`, and current visual evidence under `vis/`.
- `sv/` contains retained single-view baselines: training runs under `train/`,
  inference under `infer/`, and evaluation outputs under `eval/`.
- `archive/` contains moved legacy material. Checkpoint-heavy runs live under
  `archive/checkpoints/`; old inference/eval/diagnostic artifacts live under
  `archive/artifacts/`.

See `archive/MANIFEST.tsv` for the old path, new path, size, mtime, class, and
reason for each path moved during the archive pass.
