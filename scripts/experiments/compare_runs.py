"""Generic experiment figure generator.

Reads one or more eval_obj_results.jsonl files (written by scripts/eval/eval_obj.py),
computes mean CD / F-Score, and renders a comparison figure into figures/.

Two modes:
  * bars  (default): one bar group per run — for Test 1 (depth-geom vs gt-geom),
           Test 4 (min_views=2 vs 3), or any A/B/C comparison.
  * curve (--x-values): a line over an ordered axis — for Test 3 (1/2/3/4 views).

A `--paired` jsonl A and B with matching uid keys also reports the per-object
delta (how many objects improve), the most defensible MV>SV statistic.

Usage:
  # Test 1 headroom
  python -m scripts.experiments.compare_runs --tag test1_headroom --mode bars \
      depth_geom=outputs/evaluations-obj/test1-depthgeom/eval_obj_results.jsonl \
      gt_geom=outputs/evaluations-obj/test1-gtgeom/eval_obj_results.jsonl \
      --hline 4.55:paper-comparable-SV

  # Test 3 N-views curve
  python -m scripts.experiments.compare_runs --tag test3_nviews --mode curve \
      --x-values 1,2,3,4 \
      v1=.../n1/eval_obj_results.jsonl v2=.../n2/... v3=.../n3/... v4=.../n4/...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    cds, fs, per_obj = [], [], {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("cd") is not None:
                cds.append(r["cd"] * 1000.0)
                fs.append(r["f_score"])
                key = f"{r.get('uid')}_{r.get('obj_id')}"
                per_obj[key] = r["cd"] * 1000.0
    return np.array(cds), np.array(fs), per_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path/to/eval_obj_results.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["bars", "curve"], default="bars")
    ap.add_argument("--x-values", default=None, help="comma list for curve mode")
    ap.add_argument("--hline", default=None, help="value:label reference line on CD")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    labels, paths = [], []
    for r in args.runs:
        lab, path = r.split("=", 1)
        labels.append(lab)
        paths.append(path)

    data = [load(p) for p in paths]
    mean_cd = [d[0].mean() for d in data]
    med_cd = [np.median(d[0]) for d in data]
    mean_f = [d[1].mean() for d in data]
    ns = [len(d[0]) for d in data]

    print(f"{'run':<22}{'N':>6}{'mean CD':>10}{'med CD':>10}{'mean F':>9}")
    print("-" * 57)
    for lab, mc, mdc, mf, n in zip(labels, mean_cd, med_cd, mean_f, ns):
        print(f"{lab:<22}{n:>6}{mc:>10.3f}{mdc:>10.3f}{mf:>9.2f}")

    # Paired delta vs the first run (the control)
    if len(data) >= 2:
        base = data[0][2]
        for i in range(1, len(data)):
            cur = data[i][2]
            common = set(base) & set(cur)
            if common:
                d = np.array([cur[k] - base[k] for k in common])
                print(
                    f"\n[paired {labels[i]} vs {labels[0]}] N={len(common)} "
                    f"mean ΔCD={d.mean():+.3f}e-3  improved={int((d<0).sum())}/{len(common)} "
                    f"({100*(d<0).mean():.0f}%)"
                )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    if args.mode == "curve":
        xs = [float(x) for x in args.x_values.split(",")]
        ax[0].plot(xs, mean_cd, "o-", label="mean")
        ax[0].plot(xs, med_cd, "s--", alpha=0.6, label="median")
        ax[0].set_xlabel("number of views")
        ax[0].set_xticks(xs)
        ax[0].legend()
        ax[1].plot(xs, mean_f, "o-", color="tab:green")
        ax[1].set_xlabel("number of views")
    else:
        x = range(len(labels))
        ax[0].bar(x, mean_cd, color="tab:blue")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        for i, (c, n) in enumerate(zip(mean_cd, ns)):
            ax[0].text(i, c, f"{c:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
        ax[1].bar(x, mean_f, color="tab:green")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    if args.hline:
        val, lab = args.hline.split(":")
        ax[0].axhline(float(val), color="red", ls=":", label=lab)
        ax[0].legend(fontsize=8)
    ax[0].set_ylabel("mean CD (×10⁻³)  — lower better")
    ax[0].set_title(f"CD [{args.tag}]")
    ax[1].set_ylabel("mean F-Score (%)  — higher better")
    ax[1].set_title(f"F-Score [{args.tag}]")
    fig.tight_layout()
    out = Path(args.fig_dir) / f"{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main()
