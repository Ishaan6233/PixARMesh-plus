"""Generate training and eval loss plots from TensorBoard event files."""

import os
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

RUNS = {
    "layout-only": [
        {
            "label": "Run 1 (May 27)",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-layout-only/20260527-150204/checkpoints/runs/May27_15-02-38_gpu-h200-204",
        },
        {
            "label": "Run 2 (Jun 02)",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-layout-only/20260602-012724/checkpoints/runs/Jun02_01-27-58_gpu-h200-204",
        },
        {
            "label": "Run 3 (Jun 04)",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-layout-only/20260604-185418/checkpoints/runs/Jun04_18-54-53_gpu-h200-204",
        },
        {
            "label": "Run 4 (Jun 07)",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx-layout-only/20260607-173300/checkpoints/runs/Jun07_17-33-50_gpu-h200-204",
        },
    ],
    "img-ctx": [
        {
            "label": "Run 1 (May 28, baseline)",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx/20260528-125338/checkpoints/runs/May28_12-54-11_gpu-h200-204",
        },
        {
            "label": "Run 2 (Jun 05, pre-fix)",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx/20260605-055027/checkpoints/runs/Jun05_05-51-06_gpu-h200-204",
        },
        {
            "label": "Run 3 (Jun 07, extra_feat fix)",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx/20260607-184311/checkpoints/runs/Jun07_18-43-48_gpu-h200-204",
        },
        {
            "label": "Run 4 (Jun 10, dinov2-base) ◀ current",
            "tb_dir": ROOT / "outputs/edgerunner-3d-front-global-obj-pose-w-img-ctx/20260610-155657/checkpoints/runs/Jun10_15-57-33_gpu-h200-204",
        },
    ],
}


def load_scalar(tb_dir: Path, tag: str):
    """Load a scalar tag from a TensorBoard run directory. Returns (steps, values) or ([], [])."""
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


PALETTE = {
    "layout-only": ["#4878D0", "#6ACC65", "#D65F5F", "#FF9F0A"],
    "img-ctx":     ["#EE854A", "#956CB4", "#8C613C", "#3DB4C8"],
}


def plot_variant(variant: str, runs: list, axes: list, tags_train: list, tags_eval: list):
    colors = PALETTE[variant]
    label_prefix = "Layout-only" if variant == "layout-only" else "Img-ctx"

    for ax, tag in zip(axes[:len(tags_train)], tags_train):
        for run, color in zip(runs, colors):
            steps, vals = load_scalar(run["tb_dir"], tag)
            if not steps:
                continue
            ax.plot(steps, vals, alpha=0.2, color=color, linewidth=0.8)
            ax.plot(steps, smooth(vals), color=color, linewidth=1.6, label=f"{label_prefix} — {run['label']}")
        tag_short = tag.split("/")[-1].replace("_", " ").title()
        ax.set_title(f"Train / {tag_short}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(tag_short)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for ax, tag in zip(axes[len(tags_train):], tags_eval):
        for run, color in zip(runs, colors):
            steps, vals = load_scalar(run["tb_dir"], tag)
            if not steps:
                continue
            ax.plot(steps, vals, "o-", color=color, linewidth=1.6, markersize=5, label=f"{label_prefix} — {run['label']}")
        tag_short = tag.split("/")[-1].replace("_", " ").title()
        ax.set_title(f"Eval / {tag_short}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(tag_short)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)


def main():
    # ---- Figure 1: layout-only ----
    fig1, axes1 = plt.subplots(2, 4, figsize=(18, 8))
    fig1.suptitle("Layout-only Training Curves (edgerunner-3d-front — layout-only)", fontsize=13, fontweight="bold", y=1.01)
    axes1_flat = axes1.flatten()

    train_tags_lo = ["train/loss", "train/loss_layout", "train/loss_object", "train/grad_norm",
                     "train/learning_rate", "train/mean_token_accuracy"]
    eval_tags_lo  = ["eval/loss", "eval/loss_layout"]  # these runs don't have eval, handled gracefully

    for i, (tag, ax) in enumerate(zip(train_tags_lo, axes1_flat[:6])):
        runs_lo = RUNS["layout-only"]
        colors = PALETTE["layout-only"]
        for run, color in zip(runs_lo, colors):
            steps, vals = load_scalar(run["tb_dir"], tag)
            if not steps:
                continue
            ax.plot(steps, vals, alpha=0.18, color=color, linewidth=0.8)
            ax.plot(steps, smooth(vals, 0.95), color=color, linewidth=1.8, label=run["label"])
        tag_short = tag.split("/")[-1].replace("_", " ").title()
        ax.set_title(f"Train / {tag_short}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel(tag_short, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for i, (tag, ax) in enumerate(zip(eval_tags_lo, axes1_flat[6:8])):
        runs_lo = RUNS["layout-only"]
        colors = PALETTE["layout-only"]
        for run, color in zip(runs_lo, colors):
            steps, vals = load_scalar(run["tb_dir"], tag)
            if not steps:
                continue
            ax.plot(steps, vals, "o-", color=color, linewidth=1.8, markersize=5, label=run["label"])
        tag_short = tag.split("/")[-1].replace("_", " ").title()
        ax.set_title(f"Eval / {tag_short}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel(tag_short, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig1.tight_layout()
    out1 = OUT_DIR / "training_curves_layout_only.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")

    # ---- Figure 2: Stage-2 (img-ctx) training + eval curves ----
    # 7 train panels + 5 eval panels = 12 → 3x4 grid, no unused cells
    train_tags_ic = ["train/loss", "train/loss_layout", "train/loss_object", "train/grad_norm",
                     "train/learning_rate", "train/mean_token_accuracy", "train/entropy"]
    eval_tags_ic  = ["eval/loss", "eval/loss_layout", "eval/loss_object",
                     "eval/mean_token_accuracy", "eval/entropy"]

    fig2, axes2 = plt.subplots(3, 4, figsize=(18, 12))
    fig2.suptitle("Stage-2 Training Curves (img-ctx, joint pose+mesh)\n"
                  "Run 1=May28 (baseline) · Run 2=Jun05 (pre-fix) · Run 3=Jun07 (extra_feat fix) · Run 4=Jun10 (dinov2-base) ◀ current",
                  fontsize=12, fontweight="bold", y=1.01)
    axes2_flat = axes2.flatten()

    for ax, tag in zip(axes2_flat[:7], train_tags_ic):
        colors = PALETTE["img-ctx"]
        for run, color in zip(RUNS["img-ctx"], colors):
            steps, vals = load_scalar(run["tb_dir"], tag)
            if not steps:
                continue
            ax.plot(steps, vals, alpha=0.18, color=color, linewidth=0.8)
            ax.plot(steps, smooth(vals, 0.95), color=color, linewidth=1.8, label=run["label"])
        tag_short = tag.split("/")[-1].replace("_", " ").title()
        ax.set_title(f"Train / {tag_short}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel(tag_short, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for ax, tag in zip(axes2_flat[7:12], eval_tags_ic):
        colors = PALETTE["img-ctx"]
        for run, color in zip(RUNS["img-ctx"], colors):
            steps, vals = load_scalar(run["tb_dir"], tag)
            if not steps:
                continue
            ax.plot(steps, vals, "o-", color=color, linewidth=1.8, markersize=5, label=run["label"])
        tag_short = tag.split("/")[-1].replace("_", " ").title()
        ax.set_title(f"Eval / {tag_short}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel(tag_short, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig2.tight_layout()
    out2 = OUT_DIR / "training_curves_img_ctx.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")

    # ---- Figure 3: Stage-1 vs Stage-2 loss comparison (best complete run each) ----
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
    fig3.suptitle("Stage-1 vs. Stage-2 — Run Comparison (incl. current Jun10 dinov2-base)", fontsize=13, fontweight="bold", y=1.01)

    best = {
        "Stage-1 layout-only (Jun 04)":           (RUNS["layout-only"][2], PALETTE["layout-only"][2]),
        "Stage-2 Jun05 (pre-fix)":                (RUNS["img-ctx"][1],     PALETTE["img-ctx"][1]),
        "Stage-2 Jun07 (extra_feat fix)":         (RUNS["img-ctx"][2],     PALETTE["img-ctx"][2]),
        "Stage-2 Jun10 (dinov2-base) ◀ current": (RUNS["img-ctx"][3],     PALETTE["img-ctx"][3]),
    }

    compare_pairs = [
        ("train/loss",        "Train Loss"),
        ("train/loss_layout", "Train Layout Loss"),
        ("train/loss_object", "Train Object Loss"),
        ("eval/loss",         "Eval Loss"),
        ("eval/loss_layout",  "Eval Layout Loss"),
        ("eval/loss_object",  "Eval Object Loss"),
    ]

    for ax, (tag, title) in zip(axes3.flatten(), compare_pairs):
        for label, (run, color) in best.items():
            steps, vals = load_scalar(run["tb_dir"], tag)
            if not steps:
                continue
            if tag.startswith("eval/"):
                ax.plot(steps, vals, "o-", color=color, linewidth=1.8, markersize=5, label=label)
            else:
                ax.plot(steps, vals, alpha=0.18, color=color, linewidth=0.8)
                ax.plot(steps, smooth(vals, 0.95), color=color, linewidth=1.8, label=label)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig3.tight_layout()
    out3 = OUT_DIR / "training_curves_stage1_vs_stage2.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")

    # ---- Figure 4: Paper comparison — CD and F-score vs. reported numbers ----
    # Paper numbers: PixARMesh-EdgeRunner (Table 1, object-level, 3D-FRONT)
    PAPER = {
        "GT layout\n(paper main)":   {"cd": 4.04,  "f": 82.27},
        "GT depth\n(paper)":         {"cd": 3.04,  "f": 86.66},
        "Predicted inputs\n(paper)": {"cd": 4.13,  "f": 81.64},
    }
    OURS = {
        "baseline\n(stage1 only)":        {"cd": 8.452, "f": 71.66},
        "stage2-30k\n(broken align)":     {"cd": 11.688, "f": 59.12},
        "stage2-30k-fixed\n(align fixed)":{"cd": 8.275,  "f": 69.41},
        "stage2-0605\n(pre extra_feat fix)":{"cd": 8.316, "f": 67.10},
        "stage2-new-pred-depth\n(Pi3X pred)": {"cd": 5.519, "f": 77.37},
        "stage2-new-gt-depth\n(Pi3X GT)": {"cd": 4.550,  "f": 80.30},
    }

    fig4, (ax_cd, ax_f) = plt.subplots(1, 2, figsize=(16, 7))
    fig4.suptitle("PixARMesh Paper Numbers vs. Our Runs (Object-level, 3D-FRONT)",
                  fontsize=13, fontweight="bold")

    paper_labels = list(PAPER.keys())
    our_labels   = list(OURS.keys())

    paper_cd = [PAPER[k]["cd"] for k in paper_labels]
    paper_f  = [PAPER[k]["f"]  for k in paper_labels]
    our_cd   = [OURS[k]["cd"]  for k in our_labels]
    our_f    = [OURS[k]["f"]   for k in our_labels]

    paper_color = "#2166AC"
    ours_color  = "#D6604D"

    all_labels = paper_labels + our_labels
    n_paper, n_ours = len(paper_labels), len(our_labels)
    x = np.arange(len(all_labels))

    # CD plot (lower is better)
    bars = ax_cd.bar(x[:n_paper], paper_cd, color=paper_color, alpha=0.85, label="Paper (PixARMesh-EdgeRunner)", zorder=3)
    bars2 = ax_cd.bar(x[n_paper:], our_cd,   color=ours_color,  alpha=0.85, label="Our runs", zorder=3)
    for bar, val in zip(list(bars) + list(bars2), paper_cd + our_cd):
        ax_cd.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_cd.axvline(n_paper - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_cd.set_xticks(x)
    ax_cd.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
    ax_cd.set_ylabel("Chamfer Distance (×10⁻³)  ↓ lower is better", fontsize=9)
    ax_cd.set_title("Chamfer Distance", fontsize=11, fontweight="bold")
    ax_cd.legend(fontsize=9)
    ax_cd.grid(True, axis="y", alpha=0.3, zorder=0)
    ax_cd.spines[["top", "right"]].set_visible(False)
    ax_cd.set_ylim(0, max(paper_cd + our_cd) * 1.2)

    # F-score plot (higher is better)
    bars3 = ax_f.bar(x[:n_paper], paper_f, color=paper_color, alpha=0.85, label="Paper (PixARMesh-EdgeRunner)", zorder=3)
    bars4 = ax_f.bar(x[n_paper:], our_f,   color=ours_color,  alpha=0.85, label="Our runs", zorder=3)
    for bar, val in zip(list(bars3) + list(bars4), paper_f + our_f):
        ax_f.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_f.axvline(n_paper - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
    ax_f.set_ylabel("F-Score (%)  ↑ higher is better", fontsize=9)
    ax_f.set_title("F-Score", fontsize=11, fontweight="bold")
    ax_f.legend(fontsize=9)
    ax_f.grid(True, axis="y", alpha=0.3, zorder=0)
    ax_f.spines[["top", "right"]].set_visible(False)
    ax_f.set_ylim(0, 100)

    fig4.tight_layout()
    out4 = OUT_DIR / "paper_comparison.png"
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Saved: {out4}")

    plt.close("all")


if __name__ == "__main__":
    main()
