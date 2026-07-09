"""
Terminal health check for BPT stage-1 (layout-only) or stage-2 (full mesh) training.
Reads tensorboard events — no GPU required.

Usage:
    python scripts/check_training_health.py --run-dir <output_dir> --stage {1,2}

Example:
    python scripts/check_training_health.py \
        --run-dir outputs/sv/train/bpt-3d-front-global-obj-pose-w-img-ctx-layout-only/20260618-120000 \
        --stage 1
"""
import sys
import os
import argparse
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True, help="Training output directory (contains checkpoints/)")
parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
args = parser.parse_args()

run_dir = Path(args.run_dir)

# Find tensorboard events
event_patterns = [
    run_dir / "checkpoints" / "runs" / "**" / "events.out.*",
    run_dir / "runs" / "**" / "events.out.*",
    run_dir / "**" / "events.out.*",
]
event_files = []
for pat in event_patterns:
    event_files = glob.glob(str(pat), recursive=True)
    if event_files:
        break

if not event_files:
    print(f"ERROR: No tensorboard event files found under {run_dir}")
    sys.exit(1)

# Use most recently modified events file
event_file = max(event_files, key=os.path.getmtime)
log_dir = str(Path(event_file).parent)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)

ea = EventAccumulator(log_dir)
ea.Reload()
available = set(ea.Tags().get("scalars", []))

def get_scalar(tag):
    if tag not in available:
        return {}
    return {int(e.step): e.value for e in ea.Scalars(tag)}

# ── Collect metrics ───────────────────────────────────────────────────────────
if args.stage == 1:
    eval_loss     = get_scalar("eval/loss_layout")
    eval_acc      = get_scalar("eval/mean_token_accuracy")
    train_loss    = get_scalar("train/loss_layout")
    max_steps     = 30000
    stage_label   = "Stage-1 (layout-only)"

    # Thresholds: (step, max_eval_loss, min_accuracy)
    thresholds = [
        (5000,  1.70, 0.35),
        (10000, 1.55, 0.50),
        (20000, 1.40, 0.65),
        (30000, 1.30, 0.75),
    ]
    bf16_fail_check = ("eval/mean_token_accuracy", 10000, 0.30,
                       "layout accuracy < 30% at step 10k — possible bf16 regression")

else:
    eval_loss     = get_scalar("eval/loss")
    eval_acc      = get_scalar("eval/mean_token_accuracy")
    train_loss    = get_scalar("train/loss")
    max_steps     = 25000
    stage_label   = "Stage-2 (full mesh)"

    thresholds = [
        (5000,  0.30, 0.90),
        (8000,  0.22, 0.95),
        (15000, 0.20, 0.97),
        (25000, 0.18, 0.975),
    ]
    bf16_fail_check = ("eval/loss", 5000, 0.40,
                       "eval_loss > 0.40 at step 5k — model not learning, check bf16 regression")

# ── Determine current step ────────────────────────────────────────────────────
current_step = max(eval_loss.keys()) if eval_loss else (max(train_loss.keys()) if train_loss else 0)

print()
print(f"=== {stage_label} Health Check ===")
print(f"Run dir : {run_dir}")
print(f"Events  : {log_dir}")
print(f"Step    : {current_step:,} / {max_steps:,}  ({100*current_step/max_steps:.0f}%)")
print()

if not eval_loss:
    print("WARNING: No eval metrics found yet (training may not have reached first eval step).")
    if train_loss:
        latest_train = train_loss[max(train_loss)]
        print(f"  Latest train loss: {latest_train:.4f} at step {max(train_loss):,}")
    print()
    sys.exit(0)

# ── Status table ──────────────────────────────────────────────────────────────
col_w = [28, 12, 16, 10]
header = f"{'Metric':<{col_w[0]}} {'Current':>{col_w[1]}} {'Target':>{col_w[2]}} {'Status':>{col_w[3]}}"
sep    = "─" * sum(col_w + [3 * 2])
print(header)
print(sep)

current_eval_loss = eval_loss.get(current_step) or eval_loss[max(eval_loss)]
current_eval_acc  = eval_acc.get(current_step)  or (eval_acc[max(eval_acc)] if eval_acc else None)

loss_label = "eval_loss_layout" if args.stage == 1 else "eval_loss"

# Find the applicable threshold for current step
applicable = [(s, ml, ma) for s, ml, ma in thresholds if current_step >= s]
tgt_loss = applicable[-1][1] if applicable else thresholds[0][1]
tgt_acc  = applicable[-1][2] if applicable else thresholds[0][2]
next_thr = next(((s, ml, ma) for s, ml, ma in thresholds if s > current_step), None)

loss_ok = current_eval_loss < tgt_loss
acc_ok  = (current_eval_acc is not None) and (current_eval_acc > tgt_acc)

def fmt_status(ok): return "✓ GO" if ok else "✗ FAIL"
def fmt_pct(v): return f"{100*v:.1f}%" if v is not None else "N/A"

print(f"{loss_label:<{col_w[0]}} {current_eval_loss:>{col_w[1]}.4f} {'< '+str(tgt_loss):>{col_w[2]}} {fmt_status(loss_ok):>{col_w[3]}}")
if current_eval_acc is not None:
    print(f"{'eval_mean_token_accuracy':<{col_w[0]}} {fmt_pct(current_eval_acc):>{col_w[1]}} {'> '+fmt_pct(tgt_acc):>{col_w[2]}} {fmt_status(acc_ok):>{col_w[3]}}")

# Latest train loss
if train_loss:
    latest_train = train_loss[max(train_loss)]
    print(f"{'train_loss (latest)':<{col_w[0]}} {latest_train:>{col_w[1]}.4f} {'(no target)':>{col_w[2]}}")

# Convergence trend: compare last two eval checkpoints
sorted_steps = sorted(eval_loss)
trend_str = "(only 1 eval so far)"
trend_ok  = True
if len(sorted_steps) >= 2:
    prev_step = sorted_steps[-2]
    delta = eval_loss[sorted_steps[-1]] - eval_loss[prev_step]
    arrow = "↓" if delta < 0 else "↑"
    trend_str = f"{arrow} {delta:+.4f} (steps {prev_step:,}→{sorted_steps[-1]:,})"
    trend_ok = delta <= 0

# Overfitting check for stage-2: loss rising over last 5k steps
overfit_warn = False
if args.stage == 2 and len(sorted_steps) >= 6:
    window_start = sorted(s for s in sorted_steps if s <= current_step - 5000)
    if window_start:
        ref_loss = eval_loss[window_start[-1]]
        if current_eval_loss > ref_loss + 0.02:
            overfit_warn = True

trend_status = "✓ CONVERGING" if trend_ok else "⚠ RISING"
print(f"{'Trend (eval loss)':<{col_w[0]}} {trend_str:>{col_w[1]+col_w[2]+2}} {trend_status:>{col_w[3]}}")

print(sep)

# ── Overall verdict ───────────────────────────────────────────────────────────
issues = []

# bf16 failure check
tag_name, fail_step, fail_thresh, fail_msg = bf16_fail_check
if current_step >= fail_step:
    check_dict = eval_loss if "loss" in tag_name else eval_acc
    if check_dict:
        val_at_step = check_dict.get(fail_step) or check_dict.get(
            min(check_dict, key=lambda s: abs(s - fail_step))
        )
        if val_at_step is not None:
            condition = (val_at_step > fail_thresh) if "loss" in tag_name else (val_at_step < fail_thresh)
            if condition:
                issues.append(f"FAIL: {fail_msg}")

if not loss_ok:
    issues.append(f"eval loss {current_eval_loss:.4f} exceeds threshold {tgt_loss} at step {current_step:,}")
if not acc_ok and current_eval_acc is not None:
    issues.append(f"accuracy {fmt_pct(current_eval_acc)} below threshold {fmt_pct(tgt_acc)} at step {current_step:,}")
if not trend_ok and len(sorted_steps) >= 3:
    # Only flag if rising for 2+ consecutive checkpoints
    last3 = [eval_loss[s] for s in sorted_steps[-3:]]
    if last3[2] > last3[1] > last3[0]:
        issues.append("eval loss rising for 3+ consecutive checkpoints — possible overfitting")
if overfit_warn:
    issues.append("eval loss increased >0.02 over last 5k steps (overfitting onset)")

print()
if issues:
    print("Overall: ⚠ WARN")
    for issue in issues:
        print(f"  • {issue}")
else:
    if not applicable:
        print(f"Overall: — (not yet at first threshold step {thresholds[0][0]:,})")
    else:
        print(f"Overall: ✓ GO — training appears healthy")

if next_thr:
    steps_to_next = next_thr[0] - current_step
    print(f"Next check: step {next_thr[0]:,} (~{steps_to_next:,} steps away)")
    print(f"  Targets: eval_loss < {next_thr[1]}, accuracy > {fmt_pct(next_thr[2])}")

# ── Previous degraded run comparison ─────────────────────────────────────────
print()
if args.stage == 1:
    print("Reference — previous degraded run (bf16 bug, Jun 14):")
    print("  30k: eval_loss_layout=1.376, accuracy=68.1%")
    print("  Target for clean retrain: eval_loss < 1.30, accuracy > 75%")
else:
    print("Reference — previous degraded stage-2 run (from bf16 stage-1):")
    print("  Best eval_loss=0.189 at step 8k; final CD=6.98e-3, F=74.6%")
    print("  Target for clean retrain: eval_loss < 0.15, CD < 5.0e-3, F > 78%")

# ── Suggested plot command ────────────────────────────────────────────────────
print()
label = "Stage1-clean" if args.stage == 1 else "Stage2-clean"
metrics = "loss_layout eval_loss mean_token_accuracy" if args.stage == 1 else "loss eval_loss mean_token_accuracy"
smooth = "0.0 --downsample 1" if args.stage == 1 else "0.95 --downsample 5"
out = "figures/stage1_health.png" if args.stage == 1 else "figures/stage2_health.png"
print("Plot command:")
print(f"  python scripts/plot_training.py --current-dir \"{label}\" {run_dir} \\")
print(f"    --metrics {metrics} --smooth {smooth} --out {out}")
print()
