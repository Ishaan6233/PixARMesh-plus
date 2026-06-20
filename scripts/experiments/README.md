# MV>SV experiment ladder — commands & interpretation

A gated sequence of cheap tests that each de-risk the next, before spending the
2–4-day full train. **The bet:** single-view object CD is capped because the model
sees only the *front* of each object; multi-view conditioning sees ~4 sides, so the
latent encodes the real back surface. The lever is **conditioning completeness**
(what the 0.725 hit-rate measures). Run the tests **in order** — each result gates
the next. All figures + JSON summaries land in `figures/`.

**Baseline bar (validated single-view EdgeRunner, same `eval_obj.py` metric):**
pred-depth **5.52 / 77.4%** (fair bar) · gt-depth 4.55 / 80.3% · paper 4.04 / 3.64.

Common vars:
```bash
ENV=/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin
CKPT=outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx/20260610-155657-small-oldstage1-20260527-cd5p806-f76p349/checkpoints/final
SMALL="--image-encoder facebook/dinov2-with-registers-small --image-preprocessor facebook/dpt-dinov2-small-nyu"
```

---

## Test 0 — object-size / 512-budget sanity (read-only, seconds)

```bash
$ENV/python -m scripts.experiments.test0_size_vs_cd \
    --results outputs/evaluations-obj/<run>/eval_obj_results.jsonl --tag <run>
```
**Interpret:** Spearman(size, CD). |ρ|≳0.3 ⇒ CD is size-biased → scale `mv_num_obj_voxels`
with object size and report size-stratified CD (aggregate alone is misleading). On
single-view gt-depth ρ=−0.09 (no strong bias). Re-run on every MV output below.

---

## Test 1 — zero-training headroom (DO FIRST). Can the frozen decoder exploit completeness?

Geometry-only, single-view model. Run two arms on the same objects:
```bash
# (a) control: depth object PC, image dropped
RUN_TS=inference CUDA_VISIBLE_DEVICES=0 $ENV/accelerate launch --num_processes 1 \
  --main_process_port 29551 --module scripts.infer \
  --model-type edgerunner --run-type obj --checkpoint $CKPT $SMALL \
  --drop-image -o outputs/test1-depthgeom

# (b) treatment: DENSE GT-mesh PC (placed into the cond frame via the object transform)
RUN_TS=inference CUDA_VISIBLE_DEVICES=0 $ENV/accelerate launch --num_processes 1 \
  --main_process_port 29551 --module scripts.infer \
  --model-type edgerunner --run-type obj --checkpoint $CKPT $SMALL \
  --gt-cond --drop-image -o outputs/test1-gtgeom
# (use 8 GPUs: --num_processes 8; add --limit N for a quick subset)

# eval each
for r in depthgeom gtgeom; do
  CUDA_VISIBLE_DEVICES=0 $ENV/accelerate launch --num_processes 1 --module scripts.eval.eval_obj \
    --pred-dir outputs/test1-$r/obj/edgerunner/gt_layout_gt_mask_pred_depth \
    --save-dir outputs/evaluations-obj/test1-$r ; done

# figure
$ENV/python -m scripts.experiments.compare_runs --tag test1_headroom --mode bars \
    depth_geom=outputs/evaluations-obj/test1-depthgeom/eval_obj_results.jsonl \
    gt_geom=outputs/evaluations-obj/test1-gtgeom/eval_obj_results.jsonl \
    --hline 4.55:SV-headline
```
**Interpret / GATE:** `gt_geom` CD **≪** `depth_geom` CD ⇒ the frozen decoder *can*
exploit complete geometry → completeness is the lever, the frozen-decoder ladder is
viable → continue to Test 2. CD **≈** unchanged ⇒ the frozen decoder is the bottleneck →
switch to **Regime B** (fine-tune the decoder) before anything else — learned on day one.
(Frame correctness is auto-checked: the run prints `[gt-cond frame check]` — depth and
GT bboxes should roughly coincide; verified for `3078_0`.)

---

## Test 2 — overfit sanity (minutes–hours). Does the encoder/loss/grad path work?

Train **only** the MV encoder (decoder frozen) on a tiny subset:
```bash
RUN_TS=$(date +%s) $ENV/accelerate launch --config_file accelerate.yaml train.py \
  --config-name edgerunner_3d_front_multiview \
  model.freeze_decoder=true dataset.src_data.overfit_n=16 \
  train.train_args.max_steps=2000 train.train_args.save_steps=2000 \
  train.train_args.eval_steps=2000
# then infer (--mv --limit 16) + eval those 16 objects
```
**Interpret / GATE:** train CD on the 16 objects drives toward the per-object oracle
ceiling ⇒ encoder + loss + gradient-through-frozen-decoder all work → continue.
Can't overfit 16 ⇒ there is a bug; full training is doomed — debug first.
(Enabled by: `train.py` config-merge fix so the MV encoder is actually built;
`freeze_decoder` trains only `mv_voxel_encoder`; `overfit_n` selects the subset.)

---

## Test 3 — N-views ablation (the MV>SV proof)

Frozen decoder; vary only the view count on a fixed eval set:
```bash
for K in 1 2 3 4; do
  RUN_TS=inference CUDA_VISIBLE_DEVICES=0 $ENV/accelerate launch --num_processes 1 \
    --module scripts.infer --model-type edgerunner --run-type obj --mv \
    --checkpoint <trained-mv-ckpt-or-$CKPT> --num-views $K --limit 200 \
    -o outputs/test3-n$K
  CUDA_VISIBLE_DEVICES=0 $ENV/accelerate launch --num_processes 1 --module scripts.eval.eval_obj \
    --mv-dataset --pred-dir outputs/test3-n$K/obj/edgerunner/mv \
    --save-dir outputs/evaluations-obj/test3-n$K ; done

$ENV/python -m scripts.experiments.compare_runs --tag test3_nviews --mode curve \
  --x-values 1,2,3,4 \
  v1=outputs/evaluations-obj/test3-n1/eval_obj_results.jsonl \
  v2=outputs/evaluations-obj/test3-n2/eval_obj_results.jsonl \
  v3=outputs/evaluations-obj/test3-n3/eval_obj_results.jsonl \
  v4=outputs/evaluations-obj/test3-n4/eval_obj_results.jsonl
```
**Interpret:** **monotonic CD decrease 1→2→3→4 views** = the smoking gun. Nothing but
view count changed (same decoder, same objects), so views causally improve
reconstruction. This single curve is the headline figure. Requires a trained MV
encoder (from Test 2 / a short train); with the random encoder the curve is flat noise.

---

## Test 4 — coverage → CD

Vary conditioning completeness (hit-rate), same harness as Test 3:
```bash
# min_views=2 (~0.90 hit) vs 3 (~0.725) via config override on the MV run:
#   add  dataset.model.mv_min_views=2   to the --mv infer (Hydra: --mv-config edits)
# or subsample obj_voxels:  dataset.model.mv_num_obj_voxels=256
$ENV/python -m scripts.experiments.compare_runs --tag test4_coverage --mode bars \
    mv2_hit0.90=outputs/evaluations-obj/test4-mv2/eval_obj_results.jsonl \
    mv3_hit0.725=outputs/evaluations-obj/test4-mv3/eval_obj_results.jsonl
```
**Interpret:** CD decreases as coverage (hit-rate) rises ⇒ purity/hit-rate are
**validated predictors** of reconstruction CD — this is what makes Experiments 1–5
count toward the final claim instead of sitting beside it.

---

## Only after 0–4 pass — full train
```bash
RUN_TS=$(date +%s) $ENV/accelerate launch --config_file accelerate.yaml train.py \
  --config-name edgerunner_3d_front_multiview      # 30k steps; ~2–4 days
```
Success = MV CD < **5.52 / 77.4%** (pred bar); stretch 4.55 / 4.04.

## Decision flow
```
T0 size-bias?  ── adjust 512 budget if biased
T1 headroom ≪ 4.55? ──no──> Regime B (fine-tune decoder); redo ladder
   │yes
T2 overfit 16 → ceiling? ──no──> bug; fix before full train
   │yes
T3 CD ↓ as views ↑? ──no──> views don't help here; stop / rethink conditioning
   │yes  (← the proof)
T4 CD ↓ as coverage ↑? ── validates purity/hit-rate as CD predictors
   │
FULL TRAIN → headline vs 5.52
```
