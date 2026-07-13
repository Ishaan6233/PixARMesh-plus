"""Terminal summary + plots for an MV layout-loss training run.

Reads the run's log.jsonl directly (no tensorboard needed), prints a compact
progress table, and saves a multi-panel PNG covering the metrics relevant to
this loss stack: overall loss, grad_norm (log scale), token accuracy, learning
rate, and enabled layout-loss components (loss_layout_token/ordinal/coord).

Usage:
    python scripts/figures/plot_geoloss_progress.py
    python scripts/figures/plot_geoloss_progress.py --log-file <path> --out <path>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plot_training import extract_series, load_log, smooth

DEFAULT_LOG = "outputs/da3/train/mv_layout_loss/C_coord/seed11/logs/log.jsonl"
DEFAULT_OUT = "outputs/da3/experiments/geoloss/figures/C_coord_seed11_progress.png"
DEFAULT_MAX_STEPS = 100000

TRAIN_KEYS = [
    "loss", "grad_norm", "mean_token_accuracy", "learning_rate",
    "loss_layout_token", "loss_layout_ordinal", "loss_layout_coord",
]


def print_summary(records, max_steps, n_recent):
    train = [r for r in records if "loss" in r]
    evals = [r for r in records if any(k.startswith("eval_") for k in r)]
    if not train:
        print("No training steps logged yet.")
        return

    last = train[-1]
    step = last["step"]
    print(f"=== layout-loss progress: step {step:,} / {max_steps:,} "
          f"({100 * step / max_steps:.2f}%), epoch {last.get('epoch', 0):.2f} ===\n")

    cols = ["step", "loss", "grad_norm", "mean_token_accuracy", "loss_layout_token",
            "loss_layout_ordinal", "loss_layout_coord"]
    widths = [7, 9, 11, 10, 11, 11, 10]
    print(" ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + len(widths) - 1))
    for r in train[-n_recent:]:
        vals = []
        for c, w in zip(cols, widths):
            v = r.get(c)
            if v is None:
                vals.append(f"{'-':>{w}}")
            elif c == "step":
                vals.append(f"{v:>{w}}")
            elif c == "grad_norm":
                vals.append(f"{v:>{w}.3g}")
            else:
                vals.append(f"{v:>{w}.4f}")
        print(" ".join(vals))

    if len(train) >= n_recent + 1:
        prev = train[-n_recent - 1]
        d_loss = last["loss"] - prev["loss"]
        d_acc = last.get("mean_token_accuracy", 0) - prev.get("mean_token_accuracy", 0)
        print(f"\nOver last {n_recent} logged steps: loss {'↓' if d_loss < 0 else '↑'} "
              f"{d_loss:+.3f}, accuracy {'↑' if d_acc >= 0 else '↓'} {d_acc:+.4f}")

    if evals:
        e = evals[-1]
        print(f"\nLatest eval @ step {e.get('step', '?')}: "
              + ", ".join(f"{k}={v:.4f}" for k, v in e.items()
                          if k.startswith("eval_") and isinstance(v, (int, float))))
    else:
        print("\nNo eval checkpoint reached yet.")


def plot_progress(records, out_path, smooth_weight=0.9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("loss", False),
        ("grad_norm", True),
        ("mean_token_accuracy", False),
        ("learning_rate", False),
        ("loss_layout_token", False),
        ("loss_layout_ordinal", False),
        ("loss_layout_coord", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (metric, log_scale) in zip(axes.flat, panels):
        steps, vals = extract_series(records, metric)
        if not steps:
            ax.set_title(f"{metric} (no data)")
            continue
        ax.plot(steps, vals, color="tab:blue", alpha=0.15, linewidth=0.6)
        ax.plot(steps, smooth(vals, smooth_weight), color="tab:blue", linewidth=1.8)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.suptitle("MV layout-loss stage-1 training progress")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plots to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", default=DEFAULT_LOG)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--recent", type=int, default=10, help="rows in the recent-steps table")
    parser.add_argument("--smooth", type=float, default=0.9)
    args = parser.parse_args()

    records = load_log(args.log_file)
    if not records:
        print(f"No log records found at {args.log_file}")
        sys.exit(1)

    print_summary(records, args.max_steps, args.recent)
    plot_progress(records, args.out, args.smooth)


if __name__ == "__main__":
    main()
