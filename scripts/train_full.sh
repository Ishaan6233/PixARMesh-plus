#!/usr/bin/env bash
# Full 2-stage PixARMesh+ training (EdgeRunner).
#
# Usage (inside a tmux session):
#   bash scripts/train_full.sh           # baseline (Depth Pro depth)
#   bash scripts/train_full.sh --pi3x   # Pi3X frozen encoder variant
#
# Stage 1 is skipped automatically if a 'final/' checkpoint already exists
# (safe to re-run after an interruption).

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Argument parsing ─────────────────────────────────────────────────────────
USE_PI3X=false
for arg in "$@"; do
    [[ "$arg" == "--pi3x" ]] && USE_PI3X=true
done

if $USE_PI3X; then
    STAGE1_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx_pi3x_layout_only"
    STAGE2_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx_pi3x"
    S1_OUT_PREFIX="outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-pi3x-layout-only"
else
    STAGE1_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx_layout_only"
    STAGE2_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx"
    S1_OUT_PREFIX="outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-layout-only"
fi

echo "[train_full] ══════════════════════════════════════════════"
echo "[train_full] Pi3X: $USE_PI3X"
echo "[train_full] Stage 1 config: $STAGE1_CFG"
echo "[train_full] Stage 2 config: $STAGE2_CFG"
echo "[train_full] ══════════════════════════════════════════════"

# ── Stage 1 ──────────────────────────────────────────────────────────────────
STAGE1_CKPT=$(ls -td "${S1_OUT_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)

if [[ -n "$STAGE1_CKPT" ]]; then
    echo "[train_full] Stage 1 final checkpoint already exists:"
    echo "[train_full]   $STAGE1_CKPT"
    echo "[train_full] Skipping Stage 1."
else
    echo "[train_full] ── Stage 1: layout-only training ──"
    python launch.py train.py --config-name="${STAGE1_CFG}"

    STAGE1_CKPT=$(ls -td "${S1_OUT_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)
    if [[ -z "$STAGE1_CKPT" ]]; then
        echo "[train_full] ERROR: Stage 1 finished but final checkpoint not found under:"
        echo "[train_full]   ${S1_OUT_PREFIX}/"
        echo "[train_full] Training may have been interrupted before saving 'final/'."
        exit 1
    fi
    echo "[train_full] Stage 1 complete → $STAGE1_CKPT"
fi

# ── Stage 2 ──────────────────────────────────────────────────────────────────
echo "[train_full] ── Stage 2: full training (init from Stage 1) ──"
echo "[train_full] Stage 1 checkpoint: $STAGE1_CKPT"
python launch.py train.py \
    --config-name="${STAGE2_CFG}" \
    "model.local_path=${STAGE1_CKPT}"

echo "[train_full] ══════════════════════════════════════════════"
echo "[train_full] Done. Full 2-stage training complete."
echo "[train_full] ══════════════════════════════════════════════"
