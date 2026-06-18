#!/usr/bin/env bash
# Full 2-stage PixARMesh+ training.
#
# Usage (inside a tmux session):
#   bash scripts/train_full.sh                    # EdgeRunner baseline (Depth Pro depth)
#   bash scripts/train_full.sh --pi3x             # EdgeRunner + Pi3X frozen encoder variant
#   bash scripts/train_full.sh --bpt              # BPT variant (reproduces paper BPT numbers)
#   bash scripts/train_full.sh --force-stage1     # force Stage 1 even if checkpoint exists
#   bash scripts/train_full.sh --stage1-only      # run Stage 1 and exit (do not proceed to Stage 2)
#   bash scripts/train_full.sh --stage2-only      # skip Stage 1, run Stage 2 from latest Stage 1 ckpt
#
# Stage 1 is skipped automatically if a 'final/' checkpoint already exists
# (safe to re-run after an interruption). Use --force-stage1 when the old
# Stage 1 checkpoint is invalid (e.g. trained with a bug) and must be replaced.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-micromamba run -n pixarmesh124 python}"

# ── Argument parsing ─────────────────────────────────────────────────────────
USE_PI3X=false
USE_BPT=false
FORCE_STAGE1=false
STAGE1_ONLY=false
STAGE2_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--pi3x" ]] && USE_PI3X=true
    [[ "$arg" == "--bpt" ]] && USE_BPT=true
    [[ "$arg" == "--force-stage1" ]] && FORCE_STAGE1=true
    [[ "$arg" == "--stage1-only" ]] && STAGE1_ONLY=true
    [[ "$arg" == "--stage2-only" ]] && STAGE2_ONLY=true
done

if $STAGE1_ONLY && $STAGE2_ONLY; then
    echo "[train_full] ERROR: --stage1-only and --stage2-only are mutually exclusive."
    exit 1
fi

if $USE_BPT && $USE_PI3X; then
    echo "[train_full] ERROR: --bpt and --pi3x are mutually exclusive (no BPT-Pi3X config exists)."
    exit 1
fi

if $USE_BPT; then
    STAGE1_CFG="bpt_3d_front_global_obj_pose_w_img_ctx_layout_only"
    STAGE2_CFG="bpt_3d_front_global_obj_pose_w_img_ctx"
    S1_OUT_PREFIX="outputs/bpt-3d-front-global-obj-pose-w-img-ctx-layout-only"
elif $USE_PI3X; then
    STAGE1_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx_pi3x_layout_only"
    STAGE2_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx_pi3x"
    S1_OUT_PREFIX="outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-pi3x-layout-only"
else
    STAGE1_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx_layout_only"
    STAGE2_CFG="edgerunner_3d_front_global_obj_pose_w_img_ctx"
    S1_OUT_PREFIX="outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-layout-only"
fi

echo "[train_full] ══════════════════════════════════════════════"
echo "[train_full] BPT: $USE_BPT  Pi3X: $USE_PI3X"
echo "[train_full] Stage 1 config: $STAGE1_CFG"
echo "[train_full] Stage 2 config: $STAGE2_CFG"
echo "[train_full] ══════════════════════════════════════════════"

# ── Stage 1 ──────────────────────────────────────────────────────────────────
STAGE1_CKPT=$(ls -td "${S1_OUT_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)

if $STAGE2_ONLY; then
    if [[ -z "$STAGE1_CKPT" ]]; then
        echo "[train_full] ERROR: --stage2-only requires an existing Stage 1 checkpoint under:"
        echo "[train_full]   ${S1_OUT_PREFIX}/"
        echo "[train_full] Run Stage 1 first: bash scripts/train_full.sh --bpt --stage1-only"
        exit 1
    fi
    echo "[train_full] --stage2-only: skipping Stage 1, using: $STAGE1_CKPT"
elif [[ -n "$STAGE1_CKPT" ]] && [[ "$FORCE_STAGE1" == "false" ]]; then
    echo "[train_full] Stage 1 final checkpoint already exists:"
    echo "[train_full]   $STAGE1_CKPT"
    echo "[train_full] Skipping Stage 1. (pass --force-stage1 to override)"
else
    echo "[train_full] ── Stage 1: layout-only training ──"
    $PYTHON launch.py train.py --config-name="${STAGE1_CFG}"

    STAGE1_CKPT=$(ls -td "${S1_OUT_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)
    if [[ -z "$STAGE1_CKPT" ]]; then
        echo "[train_full] ERROR: Stage 1 finished but final checkpoint not found under:"
        echo "[train_full]   ${S1_OUT_PREFIX}/"
        echo "[train_full] Training may have been interrupted before saving 'final/'."
        exit 1
    fi
    echo "[train_full] Stage 1 complete → $STAGE1_CKPT"
fi

if $STAGE1_ONLY; then
    echo "[train_full] --stage1-only: done. Stage 2 not started."
    echo "[train_full] To run Stage 2: bash scripts/train_full.sh --bpt --stage2-only"
    exit 0
fi

# ── Stage 2 ──────────────────────────────────────────────────────────────────
echo "[train_full] ── Stage 2: full training (init from Stage 1) ──"
echo "[train_full] Stage 1 checkpoint: $STAGE1_CKPT"
$PYTHON launch.py train.py \
    --config-name="${STAGE2_CFG}" \
    "model.local_path=${STAGE1_CKPT}"

echo "[train_full] ══════════════════════════════════════════════"
echo "[train_full] Done. Full 2-stage training complete."
echo "[train_full] ══════════════════════════════════════════════"
