"""Test 0 — object-size / point-budget sanity.

Does per-object Chamfer Distance correlate with object size? A fixed obj-voxel
budget (mv_num_obj_voxels=512) undersamples large objects (a wardrobe gets a
sparser *relative* sample than a lamp), so aggregate CD may hide a size bias and
the budget may need to scale with size.

Reads an eval_obj_results.jsonl (per-object records written by scripts/eval/eval_obj.py)
and the GT meshes; plots CD vs object size + size-binned mean CD to figures/.

Usage:
  python -m scripts.experiments.test0_size_vs_cd \
      --results outputs/evaluations-obj/stage2-new-gt-depth/eval_obj_results.jsonl \
      --gt-dir datasets/3D-FUTURE-model-ply \
      --tag sv_gt_depth
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def gt_extent(gt_dir: Path, model_id: str):
    """Return the GT mesh bounding-box diagonal (object size proxy), or None."""
    p = gt_dir / f"{model_id}.ply"
    if not p.exists():
        return None
    m = o3d.io.read_triangle_mesh(str(p))
    v = np.asarray(m.vertices)
    if len(v) == 0:
        return None
    return float(np.linalg.norm(v.max(0) - v.min(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="eval_obj_results.jsonl")
    ap.add_argument("--gt-dir", default="datasets/3D-FUTURE-model-ply")
    ap.add_argument("--tag", default="run", help="label for output filenames/title")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    recs = []
    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            if r.get("cd") is not None and r.get("model_id"):
                recs.append(r)
    print(f"Loaded {len(recs)} per-object records from {args.results}")

    sizes, cds, fs = [], [], []
    for r in recs:
        d = gt_extent(gt_dir, r["model_id"])
        if d is None:
            continue
        sizes.append(d)
        cds.append(r["cd"] * 1000.0)  # ×10^-3
        fs.append(r["f_score"])
    sizes, cds, fs = np.array(sizes), np.array(cds), np.array(fs)
    print(f"Matched {len(sizes)} objects to GT meshes")

    rho_cd, p_cd = spearmanr(sizes, cds)
    rho_f, p_f = spearmanr(sizes, fs)
    print(f"Spearman(size, CD)     = {rho_cd:+.3f}  (p={p_cd:.2e})")
    print(f"Spearman(size, F-score)= {rho_f:+.3f}  (p={p_f:.2e})")

    # Size-binned means (quartiles)
    qs = np.quantile(sizes, [0, 0.25, 0.5, 0.75, 1.0])
    labels, bin_cd, bin_f, bin_n = [], [], [], []
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = (sizes >= lo) & (sizes <= hi if i == 3 else sizes < hi)
        if m.sum() == 0:
            continue
        labels.append(f"Q{i+1}\n[{lo:.2f},{hi:.2f}]")
        bin_cd.append(cds[m].mean())
        bin_f.append(fs[m].mean())
        bin_n.append(int(m.sum()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(sizes, cds, s=8, alpha=0.4)
    axes[0].set_xlabel("GT object size (bbox diagonal)")
    axes[0].set_ylabel("CD (×10⁻³)")
    axes[0].set_title(f"CD vs size  (Spearman ρ={rho_cd:+.2f}, p={p_cd:.1e})")

    axes[1].bar(range(len(labels)), bin_cd, color="tab:blue")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("mean CD (×10⁻³)")
    axes[1].set_title("Mean CD by size quartile")
    for i, (c, n) in enumerate(zip(bin_cd, bin_n)):
        axes[1].text(i, c, f"{c:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)

    axes[2].bar(range(len(labels)), bin_f, color="tab:green")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("mean F-Score (%)")
    axes[2].set_title("Mean F-Score by size quartile")

    fig.suptitle(f"Test 0 — object size vs reconstruction quality [{args.tag}]")
    fig.tight_layout()
    out = Path(args.fig_dir) / f"test0_size_vs_cd_{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out}")

    # Also dump the numbers for the interpretation write-up
    summary = {
        "tag": args.tag, "n": len(sizes),
        "spearman_size_cd": rho_cd, "p_size_cd": p_cd,
        "spearman_size_fscore": rho_f, "p_size_fscore": p_f,
        "bins": [
            {"label": l.replace("\n", " "), "mean_cd_e3": c, "mean_fscore": ff, "n": n}
            for l, c, ff, n in zip(labels, bin_cd, bin_f, bin_n)
        ],
    }
    out_json = Path(args.fig_dir) / f"test0_size_vs_cd_{args.tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {out_json}")


if __name__ == "__main__":
    main()
