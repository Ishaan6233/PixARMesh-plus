"""Pi3X Fusion Fragmentation Diagnostics.

Loads the PLY and companion metadata saved by vis_pi3x_rgb.py, then runs
four diagnostic suites to identify the root cause of disconnected geometry.

Usage:
  python scripts/diagnose_pi3x_fusion.py \
      --pointcloud outputs/pi3x_fusion/50efaf87_pi3x_multiview.ply \
      --meta       outputs/pi3x_fusion/50efaf87_metadata.npz \
      --output_dir results/50efaf87_diag
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pointcloud", required=True)
    p.add_argument("--meta",       required=True)
    p.add_argument("--output_dir", default="results/diag")
    p.add_argument("--image-hw",   type=int, nargs=2, default=[224, 224],
                   help="Image H W fed to Pi3X (default 224 224)")
    return p.parse_args()


def _savefig(path, **kw):
    plt.tight_layout()
    plt.savefig(path, dpi=120, **kw)
    plt.close()
    print(f"  saved {Path(path).name}")


def _build_connectivity(xyz, median_nn, factor=5.0):
    """Radius-graph connected components. Returns (n_comp, labels, sizes, r)."""
    r = factor * median_nn
    scene_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    r = min(r, 0.05 * scene_diag)
    tree = scipy.spatial.cKDTree(xyz)
    pairs = tree.query_pairs(r, output_type="ndarray")
    n = len(xyz)
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adj  = scipy.sparse.csr_matrix(
            (np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    else:
        adj = scipy.sparse.csr_matrix((n, n))
    n_comp, labels = scipy.sparse.csgraph.connected_components(
        adj, directed=False, return_labels=True)
    sizes = np.bincount(labels)
    return n_comp, labels, sizes, r


def _rotation_error_deg(R):
    trace = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(trace)))


def _per_view_median_nn(xyz_world, view_labels, n_views):
    """Median NN distance computed per-view to reflect local density."""
    medians = []
    for v in range(n_views):
        pts = xyz_world[view_labels == v]
        if len(pts) < 2:
            continue
        sample = pts[:min(3000, len(pts))]
        tree = scipy.spatial.cKDTree(sample)
        nn_dists, _ = tree.query(sample, k=2)
        medians.append(float(np.median(nn_dists[:, 1])))
    return float(np.median(medians))


# ─── Suite 1: Pose Consistency ────────────────────────────────────────────────

def suite1_poses(camera_poses, K_per_view, xyz_world, view_labels, img_hw, out_dir, report):
    print("\n[Suite 1] Pose consistency")
    N = len(camera_poses)
    H, W = img_hw

    # 1B — Cycle consistency
    print("  1B: cycle consistency")
    rot_drifts, trans_drifts = [], []
    triplets_checked = 0
    for a, b, c in combinations(range(N), 3):
        T_A, T_B, T_C = camera_poses[a], camera_poses[b], camera_poses[c]
        T_AB = np.linalg.inv(T_A) @ T_B
        T_BC = np.linalg.inv(T_B) @ T_C
        T_CA = np.linalg.inv(T_C) @ T_A
        cycle = T_AB @ T_BC @ T_CA
        rot_drifts.append(_rotation_error_deg(cycle[:3, :3]))
        trans_drifts.append(float(np.linalg.norm(cycle[:3, 3])))
        triplets_checked += 1
        if triplets_checked >= 200:
            break

    cycle_rot   = float(np.mean(rot_drifts))
    cycle_trans = float(np.mean(trans_drifts))
    print(f"    triplets checked: {triplets_checked}")
    print(f"    mean rotation drift:    {cycle_rot:.2f}°")
    print(f"    mean translation drift: {cycle_trans:.4f} m")
    report.update(cycle_rot_drift_deg=round(cycle_rot, 3),
                  cycle_trans_drift_m=round(cycle_trans, 5))

    # 1C — Overlap alignment error
    print("  1C: overlap alignment error")
    align_errors, pair_count = [], 0
    for a, b in combinations(range(N), 2):
        xyz_a = xyz_world[view_labels == a]
        xyz_b = xyz_world[view_labels == b]
        if len(xyz_a) < 100 or len(xyz_b) < 100:
            continue
        K_b      = K_per_view[b]
        T_b_inv  = np.linalg.inv(camera_poses[b])
        xyz_a_hom = np.concatenate([xyz_a, np.ones((len(xyz_a), 1))], axis=-1)
        xyz_b_cam = (T_b_inv @ xyz_a_hom.T).T[:, :3]
        valid_depth = xyz_b_cam[:, 2] > 0
        xyz_b_cam_v = xyz_b_cam[valid_depth]
        if len(xyz_b_cam_v) < 50:
            continue
        uv = (K_b @ xyz_b_cam_v.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        in_bounds = ((uv[:, 0] >= 0) & (uv[:, 0] < W) &
                     (uv[:, 1] >= 0) & (uv[:, 1] < H))
        if in_bounds.sum() < 50:
            continue
        overlap_xyz = xyz_a[valid_depth][in_bounds]
        tree_b = scipy.spatial.cKDTree(xyz_b)
        dists, _ = tree_b.query(overlap_xyz)
        align_errors.extend(dists.tolist())
        pair_count += 1

    if align_errors:
        ov_mean = float(np.mean(align_errors))
        ov_rms  = float(np.sqrt(np.mean(np.array(align_errors) ** 2)))
        print(f"    view pairs with overlap: {pair_count}")
        print(f"    mean alignment error: {ov_mean:.4f} m")
        print(f"    RMS  alignment error: {ov_rms:.4f} m")
        report.update(overlap_alignment_mean_m=round(ov_mean, 5),
                      overlap_alignment_rms_m=round(ov_rms, 5),
                      overlap_view_pairs=pair_count)
    else:
        print("    no overlapping view pairs found")
        report.update(overlap_alignment_mean_m=None,
                      overlap_alignment_rms_m=None,
                      overlap_view_pairs=0)


# ─── Suite 2: Cross-View Geometry ─────────────────────────────────────────────

def suite2_geometry(xyz_world, view_labels, out_dir, report):
    print("\n[Suite 2] Cross-view geometry")
    N = int(view_labels.max()) + 1
    all_dists = []

    for a, b in combinations(range(N), 2):
        xa = xyz_world[view_labels == a]
        xb = xyz_world[view_labels == b]
        if len(xa) < 50 or len(xb) < 50:
            continue
        tree_a = scipy.spatial.cKDTree(xa)
        d, _   = tree_a.query(xb)
        all_dists.extend(d.tolist())

    if not all_dists:
        print("  no cross-view pairs found")
        return

    all_dists = np.array(all_dists, dtype=np.float32)
    cv_mean   = float(np.mean(all_dists))
    cv_median = float(np.median(all_dists))
    print(f"  cross-view mean NN dist:   {cv_mean:.4f} m")
    print(f"  cross-view median NN dist: {cv_median:.4f} m")
    report.update(cross_view_mean_dist_m=round(cv_mean, 5),
                  cross_view_median_dist_m=round(cv_median, 5))

    fig, ax = plt.subplots(figsize=(7, 4))
    clip = float(np.percentile(all_dists, 95))
    ax.hist(all_dists[all_dists <= clip], bins=60, color="steelblue", edgecolor="none")
    ax.set_xlabel("Cross-view NN distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Cross-View Surface Distance (95th pct clip)")
    _savefig(out_dir / "cross_view_distance_histogram.png")


# ─── Suite 3C / 4A: Fragment Identity ─────────────────────────────────────────

def suite4a_fragment_identity(xyz_world, view_labels, median_nn, out_dir, report):
    print("\n[Suite 3C/4A] Fragment identity — component view diversity")
    n_comp, comp_labels, sizes, r = _build_connectivity(xyz_world, median_nn)
    print(f"  radius: {r:.4f} m,  components: {n_comp}")

    views_per_comp = {}
    for comp_id in range(n_comp):
        mask = comp_labels == comp_id
        views_per_comp[comp_id] = len(set(int(v) for v in view_labels[mask]))

    n_views_arr = np.array([views_per_comp[c] for c in range(n_comp)], dtype=np.int32)
    single_view_comps = int((n_views_arr == 1).sum())
    pct_sv_comps = 100.0 * single_view_comps / n_comp
    sv_pts_mask  = np.array([views_per_comp[comp_labels[i]] == 1 for i in range(len(xyz_world))])
    pct_sv_pts   = 100.0 * sv_pts_mask.sum() / len(xyz_world)
    mean_views_weighted = float(np.sum(n_views_arr * sizes) / np.sum(sizes))

    print(f"  single-view components:           {single_view_comps} / {n_comp}  ({pct_sv_comps:.1f}%)")
    print(f"  points in single-view components: {pct_sv_pts:.1f}%")
    print(f"  mean views/component (size-wtd):  {mean_views_weighted:.2f}")
    report.update(
        num_components_diag=int(n_comp),
        largest_component_ratio_diag=round(float(sizes.max() / len(xyz_world)), 4),
        pct_single_view_components=round(pct_sv_comps, 1),
        pct_points_in_single_view_components=round(pct_sv_pts, 1),
        mean_views_per_component=round(mean_views_weighted, 3),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sizes, n_views_arr, s=6, alpha=0.4, color="royalblue")
    ax.set_xlabel("Component size (points)")
    ax.set_ylabel("Number of views contributing")
    ax.set_xscale("log")
    ax.set_title("Fragment Identity: Component Size vs View Diversity")
    ax.axhline(1, color="red", linestyle="--", linewidth=1, label="single-view")
    ax.legend()
    _savefig(out_dir / "component_view_diversity.png")

    return comp_labels, sizes, n_comp, r


# ─── Suite 4B: Inter-Component Gaps ───────────────────────────────────────────

def suite4b_gaps(xyz_world, comp_labels, sizes, r, out_dir, report):
    print("\n[Suite 4B] Inter-component gap analysis")
    n_comp = int(comp_labels.max()) + 1
    if n_comp <= 1:
        print("  single component — skipping")
        report["pct_gaps_under_5cm"] = 100.0
        return

    rng = np.random.default_rng(0)
    comp_xyz = {}
    for c in range(n_comp):
        idx = np.where(comp_labels == c)[0]
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        comp_xyz[c] = xyz_world[idx]

    large = [c for c in range(n_comp) if sizes[c] > 50]
    gap_dists = []
    checked = 0
    for a, b in combinations(large[:50], 2):
        tree_a = scipy.spatial.cKDTree(comp_xyz[a])
        d, _   = tree_a.query(comp_xyz[b])
        gap_dists.append(float(d.min()))
        checked += 1
        if checked >= 500:
            break

    if not gap_dists:
        print("  not enough components for gap analysis")
        return

    gap_dists = np.array(gap_dists)
    pct_under_5cm = 100.0 * (gap_dists < 0.05).mean()
    print(f"  component pairs checked: {checked}")
    print(f"  median gap:              {float(np.median(gap_dists)):.4f} m")
    print(f"  gaps < 5 cm:             {pct_under_5cm:.1f}%")
    report.update(median_inter_component_gap_m=round(float(np.median(gap_dists)), 5),
                  pct_gaps_under_5cm=round(pct_under_5cm, 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    clip = min(float(np.percentile(gap_dists, 95)), 1.0)
    ax.hist(gap_dists[gap_dists <= clip], bins=50, color="tomato", edgecolor="none")
    ax.axvline(0.05, color="black", linestyle="--", label="5 cm")
    ax.set_xlabel("Gap distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-Component Gap Distances")
    ax.legend()
    _savefig(out_dir / "inter_component_gaps.png")


# ─── Suite 4C: ICP Refinement ─────────────────────────────────────────────────

def suite4c_icp(xyz_world, view_labels, median_nn, n_comp_before, largest_before, out_dir, report):
    print("\n[Suite 4C] ICP refinement test")
    N = int(view_labels.max()) + 1
    corr_dist = 3.0 * median_nn

    pcds = {}
    for v in range(N):
        mask = view_labels == v
        if mask.sum() < 200:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_world[mask])
        pcds[v] = (pcd, np.where(mask)[0])

    views = sorted(pcds.keys())
    refined_xyz = xyz_world.copy()
    pairs_refined = 0

    for a, b in combinations(views, 2):
        src_pcd, src_idx = pcds[b]
        tgt_pcd, _       = pcds[a]
        result = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, corr_dist,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        if result.fitness > 0.1:
            T = np.asarray(result.transformation)
            src_pts = refined_xyz[src_idx]
            refined_xyz[src_idx] = (T[:3, :3] @ src_pts.T + T[:3, 3:4]).T
            pairs_refined += 1

    print(f"  ICP pairs refined: {pairs_refined}")
    n_comp_after, _, sizes_after, _ = _build_connectivity(refined_xyz, median_nn)
    largest_after = float(sizes_after.max() / len(refined_xyz))
    improved = (n_comp_after < n_comp_before) or (largest_after > largest_before + 0.02)

    print(f"  components:         {n_comp_before} → {n_comp_after}")
    print(f"  largest comp ratio: {largest_before:.3f} → {largest_after:.3f}")
    print(f"  ICP improved:       {improved}")
    report.update(
        components_before_icp=int(n_comp_before),
        components_after_icp=int(n_comp_after),
        largest_ratio_before_icp=round(largest_before, 4),
        largest_ratio_after_icp=round(largest_after, 4),
        icp_improved=bool(improved),
    )


# ─── Final Diagnosis ──────────────────────────────────────────────────────────

def diagnose(report):
    evidence = defaultdict(float)
    crd = report.get("cycle_rot_drift_deg", 0)
    if crd > 10:  evidence["pose_estimation"]  += 0.9
    elif crd > 5: evidence["pose_estimation"]  += 0.5
    ov_rms = report.get("overlap_alignment_rms_m")
    if ov_rms is not None:
        if ov_rms > 0.1:   evidence["pose_estimation"]   += 0.8
        elif ov_rms > 0.05: evidence["pose_estimation"]  += 0.4
    sv = report.get("pct_single_view_components", 0)
    if sv > 70:   evidence["cross_view_fusion"] += 0.95
    elif sv > 40: evidence["cross_view_fusion"] += 0.6
    cv = report.get("cross_view_mean_dist_m", 0)
    if cv > 0.2:   evidence["cross_view_fusion"] += 0.8
    elif cv > 0.1: evidence["cross_view_fusion"] += 0.4
    if report.get("icp_improved"):
        evidence["pose_estimation"]   += 0.5
    else:
        evidence["cross_view_fusion"] += 0.5
    pg = report.get("pct_gaps_under_5cm", 0)
    if pg > 70: evidence["connectivity_radius"] += 0.6
    if not evidence:
        return "undetermined", {}
    total  = sum(evidence.values())
    scaled = {k: round(v / total * 100, 1) for k, v in evidence.items()}
    primary = max(scaled, key=scaled.get)
    return primary, scaled


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diagnose] Loading {args.pointcloud}")
    pcd = o3d.io.read_point_cloud(str(args.pointcloud))

    print(f"[diagnose] Loading metadata {args.meta}")
    meta = np.load(args.meta, allow_pickle=False)
    xyz_world    = meta["xyz_merged"].astype(np.float32)
    view_labels  = meta["view_labels"]
    camera_poses = meta["camera_poses"]
    K_per_view   = meta["K_per_view"]
    N = len(camera_poses)
    print(f"[diagnose] {len(xyz_world):,} world-frame points, {N} views")

    median_nn = _per_view_median_nn(xyz_world, view_labels, N)
    print(f"[diagnose] Median NN distance (per-view): {median_nn:.4f} m")

    report   = {}
    img_hw   = tuple(args.image_hw)

    suite1_poses(camera_poses, K_per_view, xyz_world, view_labels, img_hw, out_dir, report)
    suite2_geometry(xyz_world, view_labels, out_dir, report)
    comp_labels, sizes, n_comp, r = suite4a_fragment_identity(
        xyz_world, view_labels, median_nn, out_dir, report)
    suite4b_gaps(xyz_world, comp_labels, sizes, r, out_dir, report)
    suite4c_icp(xyz_world, view_labels, median_nn,
                n_comp, report.get("largest_component_ratio_diag", 0),
                out_dir, report)

    primary, breakdown = diagnose(report)
    report["primary_failure_source"] = primary
    report["failure_evidence"]       = breakdown

    json_path = out_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[diagnose] Report → {json_path}")

    secondary = {k: v for k, v in sorted(breakdown.items(), key=lambda x: -x[1]) if k != primary}
    print("\n" + "═" * 55)
    print(f"  FRAGMENTATION DIAGNOSIS")
    print("═" * 55)
    print(f"  Primary failure source : {primary:<30} ({breakdown.get(primary, 0):.0f}%)")
    for k, v in secondary.items():
        print(f"  Secondary              : {k:<30} ({v:.0f}%)")
    print("─" * 55)
    print(f"  cycle_rot_drift_deg          {report.get('cycle_rot_drift_deg', 'N/A')}")
    print(f"  overlap_alignment_rms_m      {report.get('overlap_alignment_rms_m', 'N/A')}")
    print(f"  cross_view_mean_dist_m       {report.get('cross_view_mean_dist_m', 'N/A')}")
    print(f"  pct_single_view_components   {report.get('pct_single_view_components', 'N/A')}%")
    print(f"  pct_gaps_under_5cm           {report.get('pct_gaps_under_5cm', 'N/A')}%")
    print(f"  components before/after ICP  "
          f"{report.get('components_before_icp', '?')} → {report.get('components_after_icp', '?')}")
    print("═" * 55)


if __name__ == "__main__":
    main()
