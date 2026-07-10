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
/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/experiments/mv_layout_loss_ablation.py
```

The generator writes `outputs/da3/experiments/mv_layout_loss_ablation/commands.sh`
and a JSON manifest. Review the commands before running them on a GPU machine.
The generated training commands set `train.train_args.resume_from_checkpoint=false`
and guard against existing checkpoint directories so stale partial runs do not
silently contaminate ablations. The manifest also emits object-level stage-2
inference and `scripts/eval_obj.py` commands for each E-stage seed, producing
candidate downstream files under `outputs/da3/eval/stage2_best/seed*/`.

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

After layout eval and downstream `eval_obj.py` runs exist, summarize the evidence
with:

```bash
/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/eval/summarize_mv_layout_evidence.py \
  --layout-root outputs/da3/eval/layout_mv \
  --run A_ce --run B_ordinal --run C_coord --run D_geometry \
  --seeds 11 23 37 \
  --ce-run A_ce \
  --sv-downstream outputs/sv/eval/baseline/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/eval_obj_results.jsonl
```

This writes `summary.json` and `summary.md` with per-seed confidence intervals,
UID-paired deltas against CE, downstream CD/F deltas against SV, and explicit
missing-evidence entries.

After choosing the best stage-1 run, build the evidence plots and fixed
comparison galleries:

```bash
/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
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
valid-token fraction, low IoU, and high bin error. Use `--baseline-run` to
compare against CE, SV-layout artifacts, or another run with the same
`eval_layout_mv.py` output format.

Before accepting the evidence bundle, run the hard gate:

```bash
/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
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

/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python \
  scripts/eval/check_mv_layout_evidence_bundle.py \
  --layout-root outputs/da3/eval/layout_mv \
  --run A_ce --run B_ordinal --run C_coord --run D_geometry \
  --seeds 11 23 37 \
  --ce-run A_ce \
  --sv-downstream outputs/sv/eval/baseline/eval_obj_results.jsonl \
  --downstream E_stage2_best=outputs/da3/eval/stage2_best/eval_obj_results.jsonl \
  --verifier-dir outputs/da3/experiments/mv_layout_loss_ablation/verifiers \
  --figure-dir outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures \
  --require-visuals \
  --require-figures
```

The council writer exits nonzero unless the selected layout run improves all
required bbox metrics over CE across exactly paired seeds, the selected stage-2
run beats SV on object-paired downstream CD/F, and every required negative
control report degrades relative to the matching best-run report. It also writes
an explicit `recommendation: merge|keep-experimental|reject` line. The bundle
checker then exits nonzero if required layout reports, exact paired UID records,
visual projection/conditioning artifacts, downstream object-level CD/F rows,
verifier findings, the council review recommendation, per-seed plots, UID lists,
or comparison galleries are missing.

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
`status: fail`; do not use placeholder verifier files to pass the gate.

- Loss verifier: audit `src/models/loss.py`, `src/models/edgerunner.py`,
  gradient flow, token offsets, 24-token reshape, masking, and default-off config.
- Data verifier: audit `src/data/trellis2_mv.py`, `src/data/collator.py`, cache
  metadata, view masks, and absence of unintended GT leakage.
- Experiment verifier: rerun metric extraction from saved checkpoints/logs and
  verify paired object IDs and seed statistics.
- Visual verifier: inspect gallery projection correctness, fixed-object
  coverage, improved cases, regressed cases, and failure cases. Use
  `scripts/eval/eval_layout_mv.py --visual-selection best|worst|failures|all`
  or `--visual-uids path/to/uids.txt`; each saved case records its selection
  policy, view shuffle permutation, projection validity summary, conditioning
  metadata, and conditioning point arrays under `visuals/<uid>/`.

## Council Pass

After the verifiers agree, run one adversarial review against the conclusion:

- Did the gain come from loss quality or oracle-like AABB/conditioning?
- Are extra views actually used?
- Are SV and MV metrics paired on identical validation objects?
- Are visual examples fixed and non-cherry-picked?
- Are improvements stable across seeds and object categories?

The final evidence bundle should contain metric tables with confidence
intervals, per-seed plots, ablation summaries, fixed-object galleries,
failure-case galleries, and a concise merge/keep-experimental/reject
recommendation.
