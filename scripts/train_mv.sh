#!/usr/bin/env bash
# Full two-stage DA3 Trellis2-MV PixARMesh training.
#
# Usage:
#   bash scripts/train_mv.sh
#   bash scripts/train_mv.sh --force-stage1
#   bash scripts/train_mv.sh --stage1-only
#   bash scripts/train_mv.sh --stage2-only
#   bash scripts/train_mv.sh --precompute-cache
#
# Optional:
#   MV_FEATURE_CACHE=datasets/mv-feature-cache/da3/train bash scripts/train_mv.sh
#   PYTHON="/path/to/python" GPUS=0,1,2,3 NP=4 bash scripts/train_mv.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-micromamba run -n pixarmesh124 python}"
GPUS="${GPUS:-0,1,2,3}"
NP="${NP:-4}"
STAGE1_CFG="edgerunner_3d_front_trellis2_mv_stage1"
STAGE2_CFG="edgerunner_3d_front_trellis2_mv_stage2"
S1_PREFIX="${S1_PREFIX:-outputs/da3/train/stage1}"

FORCE_STAGE1=false
STAGE1_ONLY=false
STAGE2_ONLY=false
PRECOMPUTE_CACHE=false
for arg in "$@"; do
    [[ "$arg" == "--force-stage1" ]] && FORCE_STAGE1=true
    [[ "$arg" == "--stage1-only" ]] && STAGE1_ONLY=true
    [[ "$arg" == "--stage2-only" ]] && STAGE2_ONLY=true
    [[ "$arg" == "--precompute-cache" ]] && PRECOMPUTE_CACHE=true
done

if $STAGE1_ONLY && $STAGE2_ONLY; then
    echo "[train_mv] ERROR: --stage1-only and --stage2-only are mutually exclusive."
    exit 1
fi

if $PRECOMPUTE_CACHE && [[ -z "${MV_FEATURE_CACHE:-}" ]]; then
    MV_FEATURE_CACHE="datasets/mv-feature-cache/da3/trellis2-mv"
fi

COMMON_OVERRIDES=()
if [[ -n "${MV_FEATURE_CACHE:-}" ]]; then
    COMMON_OVERRIDES+=("dataset.src_data.mv_feature_cache=${MV_FEATURE_CACHE}")
    echo "[train_mv] Using MV feature cache: ${MV_FEATURE_CACHE}"
else
    echo "[train_mv] MV_FEATURE_CACHE is unset; training will run live DA3/DINO forwards."
fi

echo "[train_mv] ══════════════════════════════════════════════"
echo "[train_mv] Stage 1 config: ${STAGE1_CFG}"
echo "[train_mv] Stage 2 config: ${STAGE2_CFG}"
echo "[train_mv] GPUs: ${GPUS}  processes: ${NP}"
echo "[train_mv] ══════════════════════════════════════════════"

accelerate_launch() {
    CUDA_VISIBLE_DEVICES="${GPUS}" $PYTHON -m accelerate.commands.launch \
        --num_processes "${NP}" \
        --gpu_ids "${GPUS}" \
        "$@"
}

if $PRECOMPUTE_CACHE; then
    echo "[train_mv] ── Precomputing DA3+DINO MV feature cache ──"
    echo "[train_mv] Cache dir: ${MV_FEATURE_CACHE}"
    for SPLIT in train val; do
        export RUN_TS="precompute-${SPLIT}-$(date +%Y%m%d-%H%M%S)"
        accelerate_launch \
            --module scripts.data.precompute_mv_features \
            --config-name="${STAGE1_CFG}" \
            --split="${SPLIT}" \
            --out="${MV_FEATURE_CACHE}"
    done
fi

S1_CKPT=$(ls -td "${S1_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)

if $STAGE2_ONLY; then
    if [[ -z "$S1_CKPT" ]]; then
        echo "[train_mv] ERROR: --stage2-only requires an existing Stage 1 final under ${S1_PREFIX}/"
        exit 1
    fi
    echo "[train_mv] --stage2-only: using existing Stage 1 checkpoint: $S1_CKPT"
elif [[ -n "$S1_CKPT" ]] && [[ "$FORCE_STAGE1" == "false" ]]; then
    echo "[train_mv] Stage 1 final exists: $S1_CKPT"
    echo "[train_mv] Skipping Stage 1. Use --force-stage1 to retrain it."
else
    echo "[train_mv] ── Stage 1: layout-only DA3 MV training ──"
    export RUN_TS=$(date +%Y%m%d-%H%M%S)
    accelerate_launch train.py \
        --config-name="${STAGE1_CFG}" \
        "${COMMON_OVERRIDES[@]}"
    S1_CKPT=$(ls -td "${S1_PREFIX}"/*/checkpoints/final 2>/dev/null | head -1 || true)
    if [[ -z "$S1_CKPT" ]]; then
        echo "[train_mv] ERROR: Stage 1 finished but final checkpoint was not found under ${S1_PREFIX}/"
        exit 1
    fi
fi

if $STAGE1_ONLY; then
    echo "[train_mv] --stage1-only: done. Stage 1 checkpoint: $S1_CKPT"
    exit 0
fi

echo "[train_mv] ── Stage 2: full mesh DA3 MV training ──"
echo "[train_mv] Stage 1 checkpoint: $S1_CKPT"
export RUN_TS=$(date +%Y%m%d-%H%M%S)
accelerate_launch train.py \
    --config-name="${STAGE2_CFG}" \
    "model.local_path=${S1_CKPT}" \
    "model.local_path_load_mode=strict_warm_start" \
    "${COMMON_OVERRIDES[@]}"

echo "[train_mv] ══════════════════════════════════════════════"
echo "[train_mv] Done."
echo "[train_mv] ══════════════════════════════════════════════"
