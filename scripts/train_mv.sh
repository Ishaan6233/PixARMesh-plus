#!/usr/bin/env bash
# Full two-stage MV PixARMesh training (mirrors scripts/train_full.sh for SV).
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
# -full train split + frozen Pi3X/DINOv2 feature cache (≈6x speedup). The cache covers
# the -full TRAIN objects; eval runs live on the val split separately.
DATA_OVERRIDES=(
  "dataset.src_data.path=datasets/3d-front-multiview-full"
  "+dataset.src_data.mv_feature_cache=datasets/mv-feature-cache"
)
S1_PREFIX="outputs/edgerunner-3d-front-multiview-stage1"

echo "[train_mv] ══════════════ Stage 1: layout-only ══════════════"
S1_CKPT=$(ls -td "${S1_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)
if [[ -n "$S1_CKPT" ]]; then
  echo "[train_mv] Stage-1 final exists: $S1_CKPT — skipping Stage 1."
else
  RUN_TS=$(date +%Y%m%d-%H%M%S) CUDA_VISIBLE_DEVICES=$GPUS \
    python launch.py --num_processes "$NP" train.py \
    --config-name edgerunner_3d_front_multiview_stage1 "${DATA_OVERRIDES[@]}"
  S1_CKPT=$(ls -td "${S1_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)
  [[ -z "$S1_CKPT" ]] && { echo "[train_mv] ERROR: Stage-1 final/ not found"; exit 1; }
fi
echo "[train_mv] Stage-1 checkpoint: $S1_CKPT"

echo "[train_mv] ══════════════ Stage 2: full (warm from Stage 1) ══════════════"
RUN_TS=$(date +%Y%m%d-%H%M%S) CUDA_VISIBLE_DEVICES=$GPUS \
  python launch.py --num_processes "$NP" train.py \
  --config-name edgerunner_3d_front_multiview_stage2 \
  "model.local_path=${S1_CKPT}" "${DATA_OVERRIDES[@]}"

echo "[train_mv] ══════════════ Done. ══════════════"
