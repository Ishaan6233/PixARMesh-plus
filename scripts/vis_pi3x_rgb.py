"""Visualize Pi3X multi-view fused point cloud with image-derived RGB colors.

Collects all views for a scene, runs Pi3X once on all N views so the decoder's
cross-view attention produces a self-consistent point cloud, then saves a merged
PLY.  Use --voxel-size to downsample after fusion.

Usage:
  python scripts/vis_pi3x_rgb.py --uid 50efaf87
  python scripts/vis_pi3x_rgb.py --uid 50efaf87 --max-views 8 --voxel-size 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

import datasets
from src.models.pi3x_cond import Pi3XFrozenEncoder
from src.pi3x.utils.basic import write_ply
from src.pi3x.utils.geometry import depth_edge


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uid", required=True)
    p.add_argument("--dataset", default="datasets/3d-front-ar-packed")
    p.add_argument("--pi3x-ckpt", default="checkpoints/pi3x")
    p.add_argument("--out", default="outputs/pi3x_fusion")
    p.add_argument("--conf-thresh", type=float, default=0.3,
                   help="Sigmoid confidence threshold (0–1)")
    p.add_argument("--edge-rtol", type=float, default=0.03,
                   help="Relative depth-edge tolerance; 0 disables")
    p.add_argument("--voxel-size", type=float, default=0.0,
                   help="Voxel grid cell size in world units; 0 disables")
    p.add_argument("--max-views", type=int, default=0,
                   help="Cap number of views (0 = all); reduces memory for large scenes")
    p.add_argument("--image-encoder", default="facebook/dinov2-with-registers-base")
    p.add_argument("--size-divisor", type=int, default=28)
    p.add_argument("--no-save-meta", action="store_true",
                   help="Skip saving the companion metadata .npz file")
    return p.parse_args()


def find_all_views(data, uid):
    """Return list of dataset examples matching uid (prefix or full UUID)."""
    is_prefix = "-" not in uid
    results = []
    for split in data.keys():
        for idx, sid in enumerate(data[split]["scene_id"]):
            if (is_prefix and sid.startswith(uid)) or sid == uid:
                results.append(data[split][idx])
    return results


def preprocess(image_np, preprocessor):
    """Run the DiNOv2 preprocessor; return pixel_values (1, 3, H, W)."""
    return preprocessor(images=image_np, return_tensors="pt")["pixel_values"]


def denorm_colors(pixel_values_1chw):
    """Invert ImageNet normalisation → uint8 RGB (H, W, 3)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = (pixel_values_1chw[0].cpu() * std + mean).clamp(0, 1)
    return (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset_path = Path(args.dataset).absolute().as_posix()
    print(f"[vis_pi3x_rgb] Loading dataset from {dataset_path}")
    data = datasets.load_dataset(dataset_path)

    examples = find_all_views(data, args.uid)
    if not examples:
        raise ValueError(f"scene_id '{args.uid}' not found in any split")

    if args.max_views > 0:
        examples = examples[:args.max_views]

    scene_id = examples[0]["scene_id"]
    N = len(examples)
    print(f"[vis_pi3x_rgb] scene_id: {scene_id}  ({N} views)")

    # ── Load preprocessor and encoder ────────────────────────────────────────
    preprocessor = AutoImageProcessor.from_pretrained(
        args.image_encoder, size_divisor=args.size_divisor
    )

    print(f"[vis_pi3x_rgb] Loading Pi3X from {args.pi3x_ckpt}")
    encoder = Pi3XFrozenEncoder(
        ckpt_path=args.pi3x_ckpt,
        greedy_anchor=True,
        disable_multimodal=True,
    )
    encoder.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)

    # ── Preprocess all views ──────────────────────────────────────────────────
    images_np      = []
    all_pv         = []
    K_per_view     = []

    for ex in examples:
        image = np.array(ex["image"])
        pv    = preprocess(image, preprocessor)   # (1, 3, H, W)
        images_np.append(image)
        all_pv.append(pv)
        K_per_view.append(np.array(ex["K"], dtype=np.float32))   # (3, 3)

        # Save each input image
        view_idx = len(images_np) - 1
        Image.fromarray(image).save(out_dir / f"{args.uid}_view{view_idx}_input.png")

    # Stack: (1, N, 3, H, W) — all views in one batch
    imgs_stack = torch.cat(all_pv, dim=0).unsqueeze(0).to(device)   # (1, N, 3, H', W')
    _, _, _, H, W = imgs_stack.shape
    patch_h, patch_w = H // 14, W // 14
    print(f"[vis_pi3x_rgb] Input: {N} views at {W}x{H}")

    # ── Single multi-view Pi3X forward ────────────────────────────────────────
    # Runs encode + N-view decode (cross-view attention) + head once.
    # out['points'] (1, N, H, W, 3) are all in Pi3X's self-consistent world frame.
    print(f"[vis_pi3x_rgb] Running Pi3X multi-view forward ({N} views)...")
    with torch.no_grad():
        hidden, _, _, _, _ = encoder.pi3x.encode(imgs_stack, with_prior=False)
        hidden = hidden.reshape(1, N, -1, encoder.pi3x.dec_embed_dim)
        hidden, pos = encoder.pi3x.decode(hidden, N, H, W, None, None)
        out = encoder.pi3x.forward_head(hidden, pos, 1, N, H, W, patch_h, patch_w)

    # (N, H, W, 3/1) — drop batch dim
    points_world = out["points"][0]       # (N, H, W, 3) in Pi3X world frame
    local_points  = out["local_points"][0]  # (N, H, W, 3) per-view camera space
    conf_all      = out["conf"][0]          # (N, H, W, 1)

    # ── Per-view filtering and color assignment ───────────────────────────────
    all_xyz, all_rgb = [], []
    # metadata accumulators (world-frame, no Y-flip, for diagnostics)
    all_xyz_world  = []
    all_conf_vals  = []
    all_view_labels = []

    for view_idx in range(N):
        pts      = points_world[view_idx]         # (H, W, 3)
        lp       = local_points[view_idx]         # (H, W, 3)
        conf     = conf_all[view_idx]             # (H, W, 1)
        pv       = all_pv[view_idx]               # (1, 3, H, W)

        conf_map = torch.sigmoid(conf).squeeze(-1)    # (H, W)

        # Depth-edge masking: zero confidence at depth discontinuities
        if args.edge_rtol > 0:
            edge_mask = depth_edge(lp[..., 2].unsqueeze(0), rtol=args.edge_rtol)
            conf_map  = conf_map.clone()
            conf_map[edge_mask.squeeze(0)] = 0.0

        conf_np = conf_map.cpu().numpy()
        depth   = lp[..., 2].cpu().numpy()
        mask    = (conf_np > args.conf_thresh) & (depth > 0)

        colors  = denorm_colors(pv)                   # (H, W, 3) uint8
        xyz_w   = pts.cpu().numpy()[mask]             # (Npts, 3) Pi3X world, no flip
        rgb     = colors[mask]                        # (Npts, 3)

        # Pi3X world frame = first view's OpenCV camera (Y-down); flip for Y-up viewers
        xyz_viz = xyz_w.copy()
        xyz_viz[:, 1] *= -1

        all_xyz.append(xyz_viz)
        all_rgb.append(rgb)
        all_xyz_world.append(xyz_w)
        all_conf_vals.append(conf_np[mask].astype(np.float32))
        all_view_labels.append(np.full(len(xyz_w), view_idx, dtype=np.int32))
        print(f"[vis_pi3x_rgb] view {view_idx}: {mask.sum():,} points  "
              f"({100 * mask.mean():.1f}%)")

    # ── Merge ────────────────────────────────────────────────────────────────
    xyz_all = np.concatenate(all_xyz, axis=0)
    rgb_all = np.concatenate(all_rgb, axis=0)
    n_raw   = len(xyz_all)

    # ── Optional voxel-grid fusion ────────────────────────────────────────────
    if args.voxel_size > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_all)
        pcd.colors = o3d.utility.Vector3dVector(rgb_all / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        xyz_all = np.asarray(pcd.points)
        rgb_all = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        print(f"[vis_pi3x_rgb] Voxel {args.voxel_size}m: {n_raw:,} → {len(xyz_all):,} points")

    # ── Write PLY ─────────────────────────────────────────────────────────────
    ply_path = out_dir / f"{args.uid}_pi3x_multiview.ply"
    write_ply(xyz_all, rgb_all, path=str(ply_path))
    print(f"[vis_pi3x_rgb] Merged {N} views → {len(xyz_all):,} points")
    print(f"[vis_pi3x_rgb] Saved PLY → {ply_path}")

    # ── Save companion metadata for diagnostics ───────────────────────────────
    if not args.no_save_meta:
        meta_path = out_dir / f"{args.uid}_metadata.npz"
        xyz_merged  = np.concatenate(all_xyz_world, axis=0)   # Pi3X world, no Y-flip
        conf_merged = np.concatenate(all_conf_vals, axis=0)
        labels      = np.concatenate(all_view_labels, axis=0)
        n_per_view  = np.array([len(a) for a in all_xyz_world], dtype=np.int32)
        np.savez_compressed(
            meta_path,
            xyz_merged      = xyz_merged.astype(np.float32),
            conf_values     = conf_merged,
            view_labels     = labels,
            n_points_per_view = n_per_view,
            camera_poses    = out["camera_poses"][0].cpu().numpy().astype(np.float32),
            K_per_view      = np.stack(K_per_view, axis=0),   # (N, 3, 3)
        )
        print(f"[vis_pi3x_rgb] Saved metadata → {meta_path}")


if __name__ == "__main__":
    main()
