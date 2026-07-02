#!/usr/bin/env bash
# Full two-stage MV PixARMesh training (mirrors scripts/train_full.sh for SV).
# Uses the Trellis2-MV mesh_dataset (datasets/mesh_datasets/..., Trellis2MVDataset) —
# higher-quality GT meshes + gravity-aligned T_norm_from_output + covisibility view
# selection — replacing the old 3d-front-multiview-full HF-Arrow path.
# NOTE: Trellis2MVDataset has no mv_feature_cache support yet, so both stages run live
# Pi3X/DINOv2 forwards every step (no ~6x cache speedup the old path had).
#
# Stage 1: layout-only, FROM SCRATCH off the EdgeRunner base (NOT the 3D-FRONT SV final),
#          100k steps — the full SV-length recipe. Warm-starting from the SV final + a short
#          re-ground was found to UNDERPERFORM this.
# Stage 2: full mesh generation, 30k steps, warm-started from Stage-1's final checkpoint.
# Both stages: decoder UNFROZEN, LR 1e-4 — on 4 GPUs (effective batch 8, memory-bound).
#
# Usage (inside tmux):  bash scripts/train_mv.sh
# Stage 1 is skipped automatically if a Stage-1 final/ checkpoint already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0,1,2,3}
NP=${NP:-4}
S1_PREFIX="outputs/edgerunner-3d-front-trellis2-mv-stage1"

echo "[train_mv] ══════════════ Stage 1: layout-only ══════════════"
S1_CKPT=$(ls -td "${S1_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)
if [[ -n "$S1_CKPT" ]]; then
  echo "[train_mv] Stage-1 final exists: $S1_CKPT — skipping Stage 1."
else
  RUN_TS=$(date +%Y%m%d-%H%M%S) CUDA_VISIBLE_DEVICES=$GPUS \
    python launch.py --num_processes "$NP" train.py \
    --config-name edgerunner_3d_front_trellis2_mv_stage1
  S1_CKPT=$(ls -td "${S1_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)
  [[ -z "$S1_CKPT" ]] && { echo "[train_mv] ERROR: Stage-1 final/ not found"; exit 1; }
fi
echo "[train_mv] Stage-1 checkpoint: $S1_CKPT"

echo "[train_mv] ══════════════ Stage 2: full (warm from Stage 1) ══════════════"
RUN_TS=$(date +%Y%m%d-%H%M%S) CUDA_VISIBLE_DEVICES=$GPUS \
  python launch.py --num_processes "$NP" train.py \
  --config-name edgerunner_3d_front_trellis2_mv_stage2 \
  "model.local_path=${S1_CKPT}"

echo "[train_mv] ══════════════ Done. ══════════════"
