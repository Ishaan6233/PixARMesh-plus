# MV Layout Loss Evidence Protocol

This branch keeps the MV layout-loss work experimental. Do not promote these
losses into default training until the evidence gates below pass.

## Implemented Losses

- Baseline layout CE remains the default and is still logged as `loss_layout`.
- `loss_layout_ordinal`: ordinal-smoothed CE over position-token logits only.
- `loss_layout_coord`: SmoothL1 between expected dequantized coordinates and GT coordinates.
- `loss_layout_center`: SmoothL1 between predicted and GT bbox centers.
- `loss_layout_size`: SmoothL1 between predicted and GT log extents.

All added losses are off by default. Enable them with the configs in
`configs/experiment/mv_layout_loss_*.yaml`.

## Required Ablations

Use one fixed validation split, cache state, checkpoint source, training-step
budget, and seed list:

- A: `+experiment=mv_layout_loss_ce`
- B: `+experiment=mv_layout_loss_ordinal`
- C: `+experiment=mv_layout_loss_coord`
- D: `+experiment=mv_layout_loss_geometry`
- E: best verified stage-1 loss carried into stage 2

Run at least three seeds for A-D before selecting E. Generate the command
manifest with:

```bash
PYTHONPATH=. /home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/experiments/mv_layout_loss_ablation.py \
  --gpus 0,1,2,3 \
  --num-processes 4 \
  --stage1-steps 100000 \
  --stage2-steps 30000
```

The generator writes `outputs/da3/experiments/mv_layout_loss_ablation/commands.sh`,
`precompute_cache.sh`, `verifier_prompts/`, and a JSON manifest. Review the
commands before running them on a GPU machine. The generated `commands.sh`
starts with
`scripts/experiments/check_mv_layout_loss_readiness.py`, which writes
`outputs/da3/experiments/mv_layout_loss_ablation/readiness.json` and exits
nonzero if GPU access, the Trellis2-MV dataset, feature cache, SV downstream
baseline, or stale-output guards are not ready. The readiness JSON contains
strict `issues` plus actionable `remediations`; the remediations tell you what
to fix next, but they do not relax any evidence gate. The generated training
commands use direct Accelerate launch with `--num_processes 4 --gpu_ids
0,1,2,3`, set
`train.train_args.resume_from_checkpoint=false` and guard against existing
checkpoint directories so stale partial runs do not silently contaminate
ablations. The manifest also emits object-level stage-2 inference and
`scripts/eval_obj.py` commands for each E-stage seed, producing candidate
downstream files under `outputs/da3/eval/stage2_best/seed*/`.

The expected frozen-feature cache root is
`datasets/mv-feature-cache/da3/trellis2-mv`. If readiness reports that it is
missing or has the wrong cache contract, rebuild it with
the generated `precompute_cache.sh` before launching A-D training; it invokes
`scripts/data/precompute_mv_features.py` for both train and val splits. The
older flat `datasets/mv-feature-cache/*.npz` layout is not accepted by the
current Trellis2-MV loader. The readiness report records the exact accepted
cache keys and `cache_version` from `src.data.trellis2_mv` so stale cache files
are caught before training starts. It also enforces 1:1 cache-file coverage
against the `conditioning_filter.csv` keep set after the loader's deterministic
train/val split. Runtime-rejected instances must have explicit marker `.npz`
files with an all-false `view_mask`; missing files are still fatal because a
silent fallback would change the data distribution.

## Metrics

Evaluate every stage-1 checkpoint with `scripts/eval/eval_layout_mv.py`. The
required paired metrics are:

- token accuracy
- valid-token fraction
- bin MAE
- corner L1/L2
- center error
- size relative error
- 3D bbox IoU

Stage-2 runs must additionally report downstream CD/F against the exact same
object IDs used for the SV baseline. Use `eval_obj_results.jsonl` files that
include object-level rows followed by the aggregate row; aggregate-only JSON is
not accepted by the evidence gates. Do not claim "beats SV PixARMesh" from bbox
metrics alone.

The generated stage-2 `scripts/eval_obj.py` commands use the frozen SV
baseline's object protocol: `--num-sample-points 10000` for scoring and
`--align-sample-points 5000` for fitting the 7-DOF ICP transform on a separate
sample. Same-sample alignment (`--align-sample-points 0`) is a different
protocol and must not be compared against the frozen SV baseline headline CD/F.
`eval_obj_results.jsonl` records these protocol fields on object rows and the
summary row.

The Step-2 manifest intentionally omits `--uid-metadata`, so object-category
stability is not evaluated and must not be claimed from these runs. To add a
category-stability gate later, regenerate with an explicit UID metadata file.
The file may be JSON, JSONL, or CSV and should contain a `uid` or `image_id`
column plus one of `category`, `object_category`, `model_category`,
`semantic_category`, `class`, `label`, `synset`, or `category_id`. Do not use
raw 3D-FUTURE `model_id` values as a semantic category substitute unless the
evidence claim is explicitly scoped to per-model stability rather than
object-category stability.

After layout eval and downstream `eval_obj.py` runs exist, summarize the evidence
with:

```bash
PYTHONPATH=. /home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/eval/summarize_mv_layout_evidence.py \
  --layout-root outputs/da3/eval/layout_mv \
  --run A_ce --run B_ordinal --run C_coord --run D_geometry \
  --seeds 11 23 37 \
  --ce-run A_ce \
  --sv-downstream outputs/sv/eval/baseline/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/seed11/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/seed23/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/seed37/eval_obj_results.jsonl
```

This writes `summary.json` and `summary.md` with per-seed confidence intervals,
UID-paired deltas against CE, downstream CD/F deltas against SV, explicit
missing-evidence entries, and no category-stability result unless
`--uid-metadata` is supplied in a later manifest. Repeating `--downstream` with
the same run name groups stage-2 seed evals and reports per-seed downstream CD/F
plus confidence intervals over those seeds. For a multi-seed evidence run, the
council and bundle gates require one downstream file per requested seed and
reject a grouped downstream win if any seed fails to beat SV on CD or F-score.
They also reject duplicate, missing, or unexpected downstream seed labels, so
passing `seed11` twice cannot stand in for the requested
`seed11/seed23/seed37` set.

Every main A-D `eval_layout_mv.py` report must come from the same
`config_name`, split, requested sample count, eval batch size, and
`mv_feature_cache`, with `view_limit=0`, `reference_only=false`, and
`shuffle_views=false`. It must also include `view_usage` with the per-object
enabled-view counts summarized from `view_mask`; the bundle checker rejects a
main eval that is effectively single-view. These fields make the "are extra
views actually used?" claim auditable before interpreting the 1/2/4/8-view,
reference-only, and shuffled-view controls. The bundle checker validates these
report identity fields so view-mask/control outputs cannot be substituted for
the main ablations.

After choosing the best stage-1 run, build the evidence plots and fixed
comparison galleries:

```bash
PYTHONPATH=. /home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/eval/build_mv_layout_evidence_figures.py \
  --summary-json outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json \
  --layout-root outputs/da3/eval/layout_mv \
  --run A_ce --run B_ordinal --run C_coord --run D_geometry \
  --seeds 11 23 37 \
  --ce-run A_ce \
  --baseline-run A_ce \
  --candidate-run D_geometry \
  --out outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures
```

This writes per-seed metric plots, paired-delta plots, ranked UID lists, and
fixed/improved/regressed side-by-side galleries from saved visual cases. It also
writes a worst-case `failure_uids.txt` plus a `failures/` gallery ranked by low
valid-token fraction, low IoU, and high bin error. It additionally writes
`gallery_uids.txt`, the union of fixed/improved/regressed/failure UIDs. The
generated `commands.sh` uses this file for a second visual-only
`eval_layout_mv.py --visual-uids ...` pass on both CE and the selected best run,
then reruns `build_mv_layout_evidence_figures.py` so the final galleries are
not limited to whichever objects happened to be saved during the first eval.
Use `--baseline-run` to compare against CE, SV-layout artifacts, or another run
with the same `eval_layout_mv.py` output format.

Before accepting the evidence bundle, run the hard gate:

```bash
PYTHONPATH=. /home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/eval/write_mv_layout_council_review.py \
  --summary-json outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json \
  --best-layout-run D_geometry \
  --best-layout-report outputs/da3/eval/layout_mv/D_geometry/seed11/report.json \
  --best-layout-report outputs/da3/eval/layout_mv/D_geometry/seed23/report.json \
  --best-layout-report outputs/da3/eval/layout_mv/D_geometry/seed37/report.json \
  --downstream-run E_stage2_best \
  --negative-control one_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed11/one_view_eval/report.json \
  --negative-control one_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed23/one_view_eval/report.json \
  --negative-control one_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed37/one_view_eval/report.json \
  --negative-control two_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed11/two_view_eval/report.json \
  --negative-control two_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed23/two_view_eval/report.json \
  --negative-control two_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed37/two_view_eval/report.json \
  --negative-control four_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed11/four_view_eval/report.json \
  --negative-control four_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed23/four_view_eval/report.json \
  --negative-control four_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed37/four_view_eval/report.json \
  --negative-control eight_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed11/eight_view_eval/report.json \
  --negative-control eight_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed23/eight_view_eval/report.json \
  --negative-control eight_view_eval=outputs/da3/eval/layout_mv/D_geometry/seed37/eight_view_eval/report.json \
  --negative-control reference_only_eval=outputs/da3/eval/layout_mv/D_geometry/seed11/reference_only_eval/report.json \
  --negative-control reference_only_eval=outputs/da3/eval/layout_mv/D_geometry/seed23/reference_only_eval/report.json \
  --negative-control reference_only_eval=outputs/da3/eval/layout_mv/D_geometry/seed37/reference_only_eval/report.json \
  --negative-control shuffled_views_eval=outputs/da3/eval/layout_mv/D_geometry/seed11/shuffled_views_eval/report.json \
  --negative-control shuffled_views_eval=outputs/da3/eval/layout_mv/D_geometry/seed23/shuffled_views_eval/report.json \
  --negative-control shuffled_views_eval=outputs/da3/eval/layout_mv/D_geometry/seed37/shuffled_views_eval/report.json \
  --negative-control no_aabb=outputs/da3/eval/layout_mv/D_geometry_no_aabb/seed11/report.json \
  --negative-control no_aabb=outputs/da3/eval/layout_mv/D_geometry_no_aabb/seed23/report.json \
  --negative-control no_aabb=outputs/da3/eval/layout_mv/D_geometry_no_aabb/seed37/report.json \
  --negative-control no_voxel_encoder=outputs/da3/eval/layout_mv/D_geometry_no_voxel_encoder/seed11/report.json \
  --negative-control no_voxel_encoder=outputs/da3/eval/layout_mv/D_geometry_no_voxel_encoder/seed23/report.json \
  --negative-control no_voxel_encoder=outputs/da3/eval/layout_mv/D_geometry_no_voxel_encoder/seed37/report.json \
  --negative-control no_obj_pc_cond=outputs/da3/eval/layout_mv/D_geometry_no_obj_pc_cond/seed11/report.json \
  --negative-control no_obj_pc_cond=outputs/da3/eval/layout_mv/D_geometry_no_obj_pc_cond/seed23/report.json \
  --negative-control no_obj_pc_cond=outputs/da3/eval/layout_mv/D_geometry_no_obj_pc_cond/seed37/report.json \
  --negative-control no_obj_pc_appearance=outputs/da3/eval/layout_mv/D_geometry_no_obj_pc_appearance/seed11/report.json \
  --negative-control no_obj_pc_appearance=outputs/da3/eval/layout_mv/D_geometry_no_obj_pc_appearance/seed23/report.json \
  --negative-control no_obj_pc_appearance=outputs/da3/eval/layout_mv/D_geometry_no_obj_pc_appearance/seed37/report.json \
  --out outputs/da3/experiments/mv_layout_loss_ablation/verifiers/council_review.md

PYTHONPATH=. /home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/eval/check_mv_layout_evidence_bundle.py \
  --layout-root outputs/da3/eval/layout_mv \
  --run A_ce --run B_ordinal --run C_coord --run D_geometry \
  --seeds 11 23 37 \
  --ce-run A_ce \
  --sv-downstream outputs/sv/eval/baseline/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/seed11/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/seed23/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/seed37/eval_obj_results.jsonl \
  --verifier-dir outputs/da3/experiments/mv_layout_loss_ablation/verifiers \
  --figure-dir outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures \
  --summary-json outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json \
  --require-visuals \
  --require-figures
```

The council writer exits nonzero unless the selected layout run improves all
required bbox metrics over CE across exactly paired seeds, the selected stage-2
run beats SV on object-paired downstream CD/F for every grouped seed, and every
required negative-control report degrades relative to the matching best-run
report on the same validation UIDs. Control `report.json` paths must keep their sibling
`per_sample.jsonl`; the council checks those UID sets before accepting control
degradation. It checks each negative-control report identity as well:
view-limit/reference/shuffle controls must record the matching eval flag, and
training controls must record the expected Hydra override in `overrides`. It
also rejects a nominated best layout run when another summarized run clearly
dominates it across all required bbox metrics; metric tradeoffs are left for
verifier review instead of forced through an arbitrary scalar score. It writes an explicit
`recommendation: merge|keep-experimental|reject` line. The bundle checker then
exits nonzero if required layout reports, exact paired UID records, controlled
layout-eval identity, visual
projection/conditioning artifacts, enabled-view projection images, multi-view
`view_usage` summaries, downstream object-level CD/F rows, per-seed downstream
CD/F wins over SV, verifier findings, the council review recommendation,
per-seed plots, UID lists, or comparison galleries are missing. The figure gate
also requires `gallery_uids.txt` to match the union of fixed, improved,
regressed, and failure UID lists, and it rejects any nonempty UID list whose
corresponding gallery has no created comparison images.

## Negative Controls

For the selected best stage-1 loss, evaluate:

- `--view-limit 1`
- `--view-limit 2`
- `--view-limit 4`
- `--view-limit 8`
- `--reference-only`
- `--shuffle-views` (deliberately shuffles per-view observations/features/masks
  while leaving cameras fixed, breaking camera/view alignment)

Train or compose separate runs for:

- `dataset.model.mv_obj_aabb_token=false`
- `dataset.model.mv_use_voxel_encoder=false` (disables the z_i/z_scene voxel
  encoder channel while leaving checkpoint/config construction compatible)
- `dataset.model.mv_obj_pc_cond=false`
- `dataset.model.mv_obj_pc_appearance=false`

If shuffled views do not hurt, or reference-only matches all views, the current
model is not providing strong evidence that additional views are being used.
Generated training-control commands use `BEST_EXPERIMENT` and
`BEST_STAGE1_RUN`, so controls follow whichever loss stack wins A-D rather than
assuming `D_geometry` is best.

## Independent Verification

Each verifier must write a short finding with exact files, commands, and
artifacts checked. Disagreements block promotion.

After summaries and figures exist, the generated `commands.sh` also runs
`scripts/experiments/write_mv_layout_verifier_prompts.py`, which writes prompt
files under `outputs/da3/experiments/mv_layout_loss_ablation/verifier_prompts/`.
Those prompt files are only reviewer instructions; they are not verifier
findings and cannot satisfy the evidence-bundle gate.

Store verifier findings under
`outputs/da3/experiments/mv_layout_loss_ablation/verifiers/` with these exact
names so the evidence checker can validate them:

- `loss_verifier.md`
- `data_verifier.md`
- `experiment_verifier.md`
- `visual_verifier.md`
- `council_review.md`

Each file must include an explicit `status: pass` line plus the checked
commands/artifacts. The evidence checker intentionally fails on unresolved
markers such as `blocker`, `todo`, `not run`, `missing evidence`, or
`status: fail`; do not use placeholder verifier files to pass the gate. It also
checks for role-specific evidence terms so a generic "checked commands" note is
not enough.

- Loss verifier: audit `src/models/loss.py`, `src/models/edgerunner.py`,
  gradient flow, token offsets, 24-token reshape, masking, and default-off config.
- Data verifier: audit `src/data/trellis2_mv.py`, `src/data/collator.py`, cache
  metadata, view masks, and absence of unintended GT leakage.
- Experiment verifier: rerun metric extraction from saved checkpoints/logs and
  verify `summary.json`, `per_sample.jsonl`, `eval_obj_results.jsonl`, paired
  object IDs, and seed statistics.
- Visual verifier: inspect gallery projection correctness, fixed-object
  coverage, `gallery_manifest.json`, `conditioning.json`, improved cases,
  regressed cases, and failure cases. Use
  `scripts/eval/eval_layout_mv.py --visual-selection best|worst|failures|all`
  or `--visual-uids path/to/uids.txt`; each saved case records its selection
  policy, view shuffle permutation, projection validity summary, conditioning
  metadata, enabled-view projection images, and conditioning point arrays under
  `visuals/<uid>/`. The bundle checker requires `view_mask`, `ref_view`,
  `obj_aabb`, projection metadata, `viewXX_projection.png` for every enabled
  view, `topdown_bbox.png`, and `conditioning_points.npz`.

## Council Pass

After the verifiers agree, run one adversarial review against the conclusion:

- Did the gain come from loss quality or oracle-like AABB/conditioning?
- Are extra views actually used?
- Are SV and MV metrics paired on identical validation objects?
- Are visual examples fixed and non-cherry-picked?
- Are improvements stable across seeds? Object-category stability is out of
  scope unless a later manifest supplies UID metadata and enables that gate.

The final evidence bundle should contain metric tables with confidence
intervals, per-seed plots, ablation summaries, fixed-object galleries,
failure-case galleries, and a concise merge/keep-experimental/reject
recommendation.
