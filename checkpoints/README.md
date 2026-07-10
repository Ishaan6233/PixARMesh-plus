# Local checkpoints

Checkpoint payloads are local artifacts and should not be committed.

- `da3/` is the checkpoint root for new runs produced from the `mv-pixarmesh-da3` branch.
- Keep only lightweight marker or documentation files in git.
- Store model weights, optimizer states, and resumed training checkpoints under ignored subdirectories.
- Active Trellis2-MV runs launched by `scripts/train_mv.sh` write final training checkpoints under
  `outputs/da3/train/stage1/*/checkpoints/final` and `outputs/da3/train/stage2/*/checkpoints/final`;
  this directory remains for local external model roots such as DA3 or manually staged weights.
