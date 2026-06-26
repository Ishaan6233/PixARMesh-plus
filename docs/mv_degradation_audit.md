# MV Extension Degradation Audit — why MV underperforms SV (and where the error leaks in)

**Scope.** SV PixARMesh ≈ CD 4.55 (HF 3.79, paper 4.04). The MV extension was *expected* to
beat SV because it observes **+64% more object surface**, yet it regressed. This document
enumerates every empirically-confirmed place the MV pipeline **loses or corrupts information**
relative to SV, with the measurement that proves it. "Leak" = error injected into the
conditioning/training signal that SV never had.

Eval reminder: scoring is 7-DOF **free-scale ICP** CD/F with `gt_layout`+`gt_mask`
([evaluation.py:12-57](../src/utils/evaluation.py#L12-L57)). Global pose/scale is **erased**;
only intrinsic *shape* is scored. This is central to which leaks matter.

---

## A. Conditioning-geometry leaks (the decoder learns from a messy point cloud)

Measured this session: the **canonicalized obj-PC the stage-1 decoder actually conditions on**
(`obj_geom_voxels`, [edgerunner.py:433-441](../src/models/edgerunner.py#L433-L441)) vs the **GT
canonical surface** (`gt_obj_vertices`) — both in the *same* canonical frame, 60 objects,
checkpoint-15000:

| metric | value | meaning |
|---|---|---|
| CD_raw (obj-PC vs GT) | **0.232** mean / 0.211 median | total mess the decoder must align away (canonical box ≈ 2.0 wide) |
| CD after best similarity-ICP | **0.190** | irreducible |
| **recoverable by a global scale/rot/trans** | **only 18%** | **82% of the error is structural, not a fixable transform** |
| GT surface coverage @5% | **20.7%** | ~79% of GT surface unobserved/misplaced; **27% of objects <10% covered** |
| ICP scale needed | **0.66–0.73** | the obj-PC is **~1.4× too large** — systematic inflation |
| per-axis extent ratio | x=0.98, **y=0.71, z=1.58** | aspect-ratio distortion vs GT |
| per-object anisotropy (max/min axis) | **4.24×** | severe non-uniform stretch |

**A1. Pi3X metric inconsistency (root leak).** Pi3 is scale-invariant *by design*; the point map
and the GT object pose are in *"inconsistent metric worlds"* — composing the GT obj pose with
Pi3X geometry gave a documented **3–8× scale failure**
([mesh.py:772-778](../src/data/mesh.py#L761-L783)). SV avoided this: Depth Pro was RANSAC-anchored
to GT metric ([utils.py:34-47](../src/data/utils.py#L34-L47)). MV has no metric anchor.
*Why it matters even under free-scale eval:* global scale is erased, but the **per-view
registration residue (8–13%)** and the resulting cross-view inconsistency are not.

**A2. Partial-extent self-normalization (scale + aspect leak).** Because Pi3X scale is unusable,
the pipeline re-centers/re-scales the obj voxels by their **own observed extent**
([edgerunner.py:437-441](../src/models/edgerunner.py#L437-L441)). With only ~21% of the surface
observed, that observed bbox ≠ GT bbox, so normalizing it to the canonical box **inflates the
object ~1.4× (ICP scale 0.66) and distorts its aspect ratio (anisotropy 4.24, z-extent 1.58×)**.
The decoder is trained to map an inflated, squashed partial cloud → the true shape.

**A3. 82% irreducible mess.** The headline: a single better alignment transform cannot rescue the
conditioning — 82% of the obj-PC↔GT gap survives best-fit similarity. The leak is **structural**
(partial coverage + cross-view registration noise), so the fix must be *better fusion/coverage*
(plan 1C CMVSA + confidence), not a smarter normalization.

## B. DINOv2 appearance-fusion leak (caused by A)

The voxel encoder projects the **scene-frame obj_voxels** into every view and samples DINOv2 at
the projected pixels, gating by a depth-visibility check, then IBR-fuses
([mv_voxel_encoder.py:368-403](../src/models/mv_voxel_encoder.py#L368-L403)).

**B1. Wrong-pixel sampling.** The voxels are the A-section partial/scale-inconsistent cloud. A
voxel that is correct in its *source* view projects to an **offset pixel in other views** because
Pi3X depth disagrees across views → DINOv2 sampled at the wrong location → the IBR fusion averages
**inconsistent features**. The appearance attached to each obj point is a noisy mix,
**off-distribution** from the clean Depth-Pro-aligned features the SV cond_encoder was trained on.

**B2. Visibility-gate misfire.** `_compute_visibility_mask` compares the voxel's projected depth
against the Pi3X depth map. Under Pi3X cross-view scale error this misfires — the **scale_err≈55%**
finding ([diagnose_pool_misses.py](../scripts/eval/diagnose_pool_misses.py)) means a majority of
"missed" object points are within the object but rejected by the depth check → valid views dropped
from fusion (fewer, noisier views) or mask-bleed admitted.

## C. Training-target / objective leaks

**C1. Layout-only oracle that "tests nothing" (config leak).** The stage-1 oracle ships
`ignore_obj_seq: true`, which strips the object sequence entirely
([collator.py:307](../src/data/collator.py#L307)) → `loss_object ≡ 0`, no mesh ever generated.
Observed: `loss_layout` plateaued ~1.75 (step 1405/30000) while the run could not answer the
shape-ceiling question it was launched for. The shape-ceiling oracle **must** set
`ignore_obj_seq: false`.

**C2. Gating on `loss_layout` (eval-blind signal).** Layout = bbox pose, which free-scale ICP
**erases**. Optimizing/gating on it spends capacity on a quantity the benchmark does not reward
(and which obj-PC centering strips). Gate on stage-2 `gt_layout` **CD/F + coverage** instead.

**C3. Oracle GT-point leak (intentional, but must stay quarantined).** `mv_obj_pc_oracle: true`
feeds GT-canonical points as conditioning ([edgerunner.py:448-451](../src/models/edgerunner.py#L448)).
Legitimate as a ceiling probe only; never let it into a reported "real" number.

## D. Decoding leak (FIXED, pending coverage confirmation)

**D1. Runaway + early-EOS collapse.** Prior MV eval ran ~4% coverage (bimodal: early-EOS blobs +
unbounded runaway). Root grammar bug: the EdgeRunner constraint fn kept **one mutable counter per
batch item**, which beam search corrupts across divergent beams. Fix (working tree): stateless
beam-safe grammar + `min_new_tokens` floor + `10*max_faces` cap + honest decode-failure placeholder
([infer.py](../scripts/infer.py), [inference.py:176](../src/utils/inference.py#L176)). Unit test
4/4 ([tests/test_edgerunner_generation.py](../tests/test_edgerunner_generation.py)). Coverage
number: **<PENDING the running eval>** (smoke: 3/3 real meshes, 798–799 faces, no placeholders).

## E. Architectural / hygiene leaks

**E1. Two heterogeneous conditioning latent spaces.** The from-scratch decoder must reconcile the
SV obj-PC latents (2048 tok) and the from-scratch `mv_voxel_encoder` z_i/z_scene (321 tok)
([edgerunner.py:527-535](../src/models/edgerunner.py#L527)). Unify (plan 1B) for cleaner
convergence.

**E2. Confidence underused.** Pi3X confidence gates point *selection* and weights fusion, but the
surviving z_i/z_scene queries carry **no confidence signal forward** — the decoder can't tell a
well-observed region from a guessed one. Expose it as an explicit per-voxel feature (plan 2B).

**E3. Fallback path leak.** When panoptic masks are absent, ctx voxels are built with plain
`fps_centroid_seeded` (no confidence) ([edgerunner.py:419-420](../src/models/edgerunner.py#L419));
if ever hit in training, z_scene is contaminated by low-confidence Pi3X points.

---

## Priority (what actually moves CD under the scored metric)

1. **A2/A3 — kill the partial-extent self-norm; do object-centric multi-view registration + robust
   norm (CMVSA, plan 1C).** Biggest lever: 82% of the conditioning mess is structural and rides
   straight into B (DINO fusion). Confidence-filter + register all views → fuller, undistorted obj-PC.
2. **C1/C2 — relaunch the oracle with `ignore_obj_seq: false`; gate on CD/F not `loss_layout`.**
   Cheap, unblocks every downstream decision.
3. **D1 — confirm coverage ≥95% with the fixed decoder** so any CD is trustworthy.
4. **E1/E2 — unify conditioning + add the confidence channel** (also the stage-2 novelty).

*Not worth chasing:* metric grounding of Pi3X (global scale is eval-erased); a smarter single
normalization transform (only 18% of the error is transform-recoverable).
