# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
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

## Common commands

```bash
make test            # python -m pytest tests/ -v
make lint            # ruff check src/ scripts/ tests/
make format          # ruff format ...
python -m pytest tests/test_edgerunner_generation.py -q          # single test file
python -m pytest tests/test_x.py::ClassName::test_method         # single test
```

Training (2-stage; see "Architecture"):
```bash
bash scripts/train_full.sh              # SV: stage1 layout-only -> stage2 full, auto-resumes
bash scripts/train_mv.sh                # MV: stage1 layout-only -> stage2 full, auto-skips existing stage1
bash scripts/train_mv.sh --precompute-cache  # MV: precompute DA3+DINO cache for train+val, then train
python launch.py [--num_processes N] train.py --config-name <cfg> [hydra.overrides=...]
```

Inference + eval (distributed via Accelerate):
```bash
accelerate launch --module scripts.infer --model-type edgerunner --run-type obj \
  --checkpoint <hf-or-local-ckpt> --output-dir outputs/sv/infer        # SV
# MV model construction must use the same MV config family as training; current
# stage1/stage2 configs derive prefix_len=2371 via mv_prefix_len because
# mv_obj_pc_cond=true and mv_obj_aabb_token=true.
# EdgeRunner decode flags: --num-beams --min-faces --max-faces --gt-layout --gt-mask
accelerate launch --module scripts.eval_obj --pred-dir <preds> --save-dir <out>   # or: make eval-obj ARGS=...
```

**Accelerate gotcha:** `launch.py` wraps `accelerate launch` and auto-loads repo `accelerate.yaml`
(`num_processes: 8`, `gpu_ids: all`). If you restrict GPUs with `CUDA_VISIBLE_DEVICES` to fewer than
8, the proc-count/GPU mismatch **crashes a rank**. To run on a GPU subset, call `accelerate launch
--num_processes N --gpu_ids 0,1,2,3 ...` directly (bypasses `launch.py` and `accelerate.yaml`).

## Architecture (the big picture)

**Unified AR sequence.** Each object is encoded as a **layout** segment (bbox pose tokens) followed
by an **object** segment (mesh tokens). [src/models/loss.py](src/models/loss.py) splits the loss into
`loss_layout` / `loss_object` by token type. Two training stages are selected purely by data flags:
`ignore_obj_seq: true` ⇒ **stage 1 = layout-only** (object tokens stripped in
[src/data/collator.py](src/data/collator.py); `loss_object≡0`); stage 2 sets it false and
**warm-starts from stage-1's checkpoint** (`model.local_path=<stage1 final>`).

**Two decoder families**, chosen via the `configs/model/` group: **EdgeRunner** (`ShapeOPT` decoder,
[src/models/edgerunner.py](src/models/edgerunner.py)) and **BPT** ([src/models/bpt.py](src/models/bpt.py)).
EdgeRunner uses a constrained token grammar — `0=PAD 1=BOS 2=EOS 3=L 4=R 5=BOM 6+=coords`; BOM starts
a patch (9 coords), L/R extend it (3 coords). Constrained decoding lives in
`get_prefix_allowed_tokens_fn_edgerunner` ([src/utils/inference.py](src/utils/inference.py)) and **must
stay stateless** (beam search calls it per-beam; mutable per-batch state corrupts beams).

**Conditioning — SV vs MV (the core of this fork):**
- **SV:** Depth Pro (metric, RANSAC-aligned to GT via `align_depth` in
  [src/data/utils.py](src/data/utils.py)) + Grounded-SAM masks + DINOv2 image features → the
  EdgeRunner `cond_encoder` turns the object point cloud into 2048 latents.
- **MV:** `Trellis2MVDataset` ([src/data/trellis2_mv.py](src/data/trellis2_mv.py))
  selects covisible object-support views, emits per-view cameras, masks, `view_mask`, and `ref_view`,
  and can load precomputed frozen features from `dataset.src_data.mv_feature_cache`.
  `discover_instance_points_mv` then uses the selected views for object/context geometry, while
  `MultiViewVoxelAlignedEncoder` ([src/models/mv_voxel_encoder.py](src/models/mv_voxel_encoder.py))
  projects voxels into every valid view, samples DINOv2, and does IBRNet **confidence-weighted** fusion →
  `z_i` (object) + `z_scene`. The decoder prefix is `[obj-PC latents] + [z_i, z_scene] + [obj-AABB] + num_face`
  scattered into `pc_token` slots when those channels are enabled. Entry point:
  `EdgeRunner.get_mv_inputs_with_cond`. `prefix_len` is **derived** from the active conditioning
  channels (`mv_prefix_len`); the collator must emit exactly that many `pc_token` slots.

**MV training setup.** Use [scripts/train_mv.sh](scripts/train_mv.sh) as the canonical wrapper.
It defaults to `edgerunner_3d_front_trellis2_mv_stage1` then
`edgerunner_3d_front_trellis2_mv_stage2`, writes under `outputs/da3/train/stage1` and
`outputs/da3/train/stage2`, skips Stage 1 when a final checkpoint already exists, and supports
`--force-stage1`, `--stage1-only`, and `--stage2-only`. Set `MV_FEATURE_CACHE=<dir>` to pass
`dataset.src_data.mv_feature_cache=<dir>` into both stages. Add `--precompute-cache` to first run
[scripts/data/precompute_mv_features.py](scripts/data/precompute_mv_features.py) for train and val;
each cache file stores `cache_version`, `local_points`, `conf`, `dino_feats`,
`view_indices`, `view_mask`, and `ref_view`, letting `train.py` skip constructing
the live DA3 geo encoder.

**Reference-view policy.** The runtime policy is intentionally simple: score HF views by target-object
projected support, select up to `mv_covis_k_max` diverse covisible views, and use local slot `0`
(the highest-support selected view) as `ref_view`. If panoptic masks exist and the target id can be
resolved, the chosen support views must also pass cheap sanity checks: visible target area and a few
projected seed hits. GT silhouette agreement is diagnostic, not the training selection objective.

**Frame handling (a recurring source of bugs).** The decoder emits per-object **canonical** vertices
(`normalize_vertices(bound=0.95)`); `obj_voxels` live in the scene-normalized frame.
`obj_canon_transform` ([src/data/mesh.py](src/data/mesh.py), `transform_3d_front_multiview`) maps
scene→object, while the geometry stream re-centers/re-scales obj voxels by their
**own observed extent** ([edgerunner.py](src/models/edgerunner.py)) instead of trusting a global
scene scale. See [docs/mv_degradation_audit.md](docs/mv_degradation_audit.md).

**Evaluation is scale-free.** `eval_obj` / `eval_scene` score 7-DOF **free-scale ICP** CD/F with
`gt_layout`+`gt_mask` ([src/utils/evaluation.py](src/utils/evaluation.py)); global pose and scale are
erased — **only intrinsic shape is scored.** Coverage (fraction of objects that produced a mesh) is
reported alongside CD/F and must be ≥~95% for a CD number to be trustworthy.

**Data.** Hydra configs in `configs/` (`configs/dataset/canonical_3d_front*.yaml`,
`configs/model/`). The active MV training config uses Trellis2 per-object mesh dumps plus
`datasets/3d-front-multiview-full` for images/cameras/masks. The optional MV feature cache is
configured via `dataset.src_data.mv_feature_cache`; rebuild it whenever the selected views,
image preprocessing, DA3 checkpoint, DINO encoder, or reference-view policy changes.
`launch.py` also injects `RUN_TS` and forwards SIGUSR1 to workers.

**Debug/oracle flags** (MV): `mv_obj_pc_oracle` feeds GT-canonical points as conditioning — a
**leak, ceiling-probe only**, never report as a real number. `mv_obj_pc_cond`,
`mv_use_voxel_encoder`, `mv_obj_pc_appearance`, `mv_min_views`, `mv_depth_rtol` toggle the MV
conditioning channels.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After material file-structure changes, run `bash scripts/graphify_refresh.sh` before the final response. Material structure changes include creating, deleting, moving, or renaming files/directories; adding new modules, configs, scripts, docs, tests, checkpoint roots, or output roots; or changing this file-placement policy.
- After code-only changes, run `graphify update .` when graphify-out/graph.json exists to keep the graph current (AST-only, no API cost).
