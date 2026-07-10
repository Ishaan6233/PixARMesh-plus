# AGENTS.md

This file provides guidance to Codex when working with code in this repository.
Shared file-placement rules live in `docs/FILE_STRUCTURE.md`; read that before creating new files.
Generated `graphify/` and `graphify-out/` artifacts are optional navigation aids, not source of truth.

PixARMesh (CVPR 2026) is a **mesh-native autoregressive** 3D scene reconstructor: object poses and
meshes are emitted as one unified token sequence, no intermediate volumetric/implicit stage. This
fork (`PixARMesh+`) extends the single-view (SV) method to **multi-view (MV)**; active work lives on
the `mv-pixarmesh-da3` branch.

## Environment

- Conda/micromamba env: **`pixarmesh124`** (`micromamba run -n pixarmesh124 python ...`, or the
  interpreter at `/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python`).
- Requires the external **EdgeRunner tokenizer** (`meto`, from NVlabs/EdgeRunner) — the mesh
  token codec; nothing decodes without it. Deps: `requirements.txt` + `requirements-no-iso.txt`.
- When invoking a script directly (not via `accelerate launch --module`), set `PYTHONPATH=.` or the
  `from src...` imports fail.

## Active MV Workflow

- Canonical operator wrapper: `bash scripts/train_mv.sh`; see `README.md` for flags and examples.
- Cache reproducibility contract: `environment/REPRODUCIBILITY.md`.
- File-placement rules for MV scripts, caches, checkpoints, and outputs: `docs/FILE_STRUCTURE.md`.

# Agent Guidance

Before creating new files, read `docs/FILE_STRUCTURE.md` and place files by
semantic purpose, not by whichever directory is currently open.

`graphify/` and `graphify-out/` are local generated navigation artifacts. They
can help an agent find relevant files with fewer tokens, but they are not a
source of truth and should not be committed. Verify claims against live code,
configs, tests, and command output.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After material file-structure changes, run `bash scripts/graphify_refresh.sh` before the final response. Material structure changes include creating, deleting, moving, or renaming files/directories; adding new modules, configs, scripts, docs, tests, checkpoint roots, or output roots; or changing this file-placement policy.
- After code-only changes, run `graphify update .` when graphify-out/graph.json exists to keep the graph current (AST-only, no API cost).
