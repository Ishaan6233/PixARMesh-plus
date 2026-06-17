import argparse
import json
import os
import sys


def load_log(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"[warn] log file not found: {jsonl_path}", file=sys.stderr)
        return []
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def smooth(values, weight=0.95):
    smoothed, last = [], values[0] if values else 0.0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def extract_series(records, key):
    steps, vals = [], []
    for r in records:
        if key in r and r[key] is not None:
            steps.append(r["step"])
            vals.append(r[key])
    return steps, vals


def plot_runs(runs, metrics, out_path, smooth_weight, downsample):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric in zip(axes, metrics):
        for i, (name, records, is_current) in enumerate(runs):
            steps, vals = extract_series(records, metric)
            if not steps:
                continue
            if downsample > 1:
                steps = steps[::downsample]
                vals = vals[::downsample]
            color = colors[i % len(colors)]
            lw = 2.5 if is_current else 1.2
            zorder = 3 if is_current else 2
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.6, zorder=zorder - 1)
            label = f"{name} (step {steps[-1]:,})" if is_current else name
            ax.plot(steps, smooth(vals, smooth_weight), color=color, linewidth=lw,
                    label=label, zorder=zorder)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


def find_log(run_dir):
    for p in [os.path.join(run_dir, "logs", "log.jsonl"), os.path.join(run_dir, "log.jsonl")]:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss curves across runs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--run", nargs=2, metavar=("NAME", "LOG_JSONL"),
                        action="append", dest="runs")
    parser.add_argument("--current", nargs=2, metavar=("NAME", "LOG_JSONL"))
    parser.add_argument("--run-dir", nargs=2, metavar=("NAME", "RUN_DIR"),
                        action="append", dest="run_dirs")
    parser.add_argument("--current-dir", nargs=2, metavar=("NAME", "RUN_DIR"),
                        dest="current_dir")
    parser.add_argument("--metrics", nargs="+",
                        default=["loss", "loss_object", "loss_layout"])
    parser.add_argument("--smooth", type=float, default=0.97)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()

    runs_raw = []
    for name, run_dir in (args.run_dirs or []):
        p = find_log(run_dir)
        if p:
            runs_raw.append((name, p, False))
        else:
            print(f"[warn] no log.jsonl under {run_dir}", file=sys.stderr)
    for name, path in (args.runs or []):
        runs_raw.append((name, path, False))
    if args.current_dir:
        name, run_dir = args.current_dir
        p = find_log(run_dir)
        if p:
            runs_raw.append((name, p, True))
        else:
            print(f"[error] no log.jsonl under {run_dir}", file=sys.stderr)
            sys.exit(1)
    elif args.current:
        name, path = args.current
        runs_raw.append((name, path, True))

    if not runs_raw:
        parser.error("Specify at least one --run-dir or --current-dir.")

    runs = []
    for name, path, is_current in runs_raw:
        records = load_log(path)
        if records:
            print(f"  {name}: {len(records)} steps")
            runs.append((name, records, is_current))

    plot_runs(runs, args.metrics, args.out, args.smooth, args.downsample)


if __name__ == "__main__":
    main()
