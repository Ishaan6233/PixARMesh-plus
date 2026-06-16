#!/usr/bin/env bash
# Wait for training PID to exit, then run BPT inference + object eval.
# Usage: bash scripts/run_eval_bpt.sh <training_pid> <checkpoint_dir> <output_dir>
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_PID="${1:-52274}"
CKPT="${2:-outputs/bpt-3d-front-global-obj-pose-w-img-ctx/20260615-024721/checkpoints/final}"
OUT="${3:-outputs/eval_bpt_run1}"
PRED_DIR="${OUT}/obj/bpt/gt_layout_gt_mask_gt_depth"

echo "[eval_bpt] Waiting for training PID ${TRAIN_PID} to exit..."
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 30
done
echo "[eval_bpt] Training done. Starting inference at $(date)."

micromamba run -n pixarmesh124 python launch.py infer.py \
    --run-type obj \
    --model-type bpt \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT}" \
    --batch-size 4 \
    --gt-layout --gt-mask --gt-depth

echo "[eval_bpt] Inference done at $(date). Running eval_obj..."

micromamba run -n pixarmesh124 python scripts/eval_obj.py \
    --pred-dir "${PRED_DIR}" \
    --align-sample-points 5000 \
    --save-dir outputs/evaluations-obj

echo "[eval_bpt] Eval complete at $(date)."
