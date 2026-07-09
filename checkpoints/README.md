# Local checkpoints

Checkpoint payloads are local artifacts and should not be committed.

- `da3/` is the checkpoint root for new runs produced from the `mv-pixarmesh-da3` branch.
- Keep only lightweight marker or documentation files in git.
- Store model weights, optimizer states, and resumed training checkpoints under ignored subdirectories.

