"""Pi3X multi-view fusion using ground-truth camera poses from the dataset.

Pi3X runs in single-view mode per view to estimate depth (local_points) and
confidence.  Ground-truth wrd2cam matrices from the dataset replace Pi3X's
predicted poses for the world-space transform, bypassing Pi3X pose estimation.

Usage:
  python scripts/vis_pi3x_gtpose.py --uid 50efaf87
  python scripts/vis_pi3x_gtpose.py --uid 50efaf87 --max-views 6 --voxel-size 0.02
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import datasets
from src.models.pi3x_cond import Pi3XFrozenEncoder
from src.pi3x.utils.basic import write_ply
from src.pi3x.utils.geometry import depth_edge


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uid",           required=True)
    p.add_argument("--dataset",       default="datasets/3d-front-ar-packed")
    p.add_argument("--pi3x-ckpt",     default="checkpoints/pi3x")
    p.add_argument("--out",           default="outputs/pi3x_gtpose")
    p.add_argument("--conf-thresh",   type=float, default=0.3)
    p.add_argument("--edge-rtol",     type=float, default=0.03)
    p.add_argument("--voxel-size",    type=float, default=0.0)
    p.add_argument("--max-views",     type=int,   default=0)
    p.add_argument("--image-encoder", default="facebook/dinov2-with-registers-base")
    p.add_argument("--size-divisor",  type=int,   default=28)
    p.add_argument("--no-yflip",      action="store_true",
                   help="Skip Y-axis flip (use if world frame is already Y-up)")
    return p.parse_args()


def find_all_views(data, uid):
    is_prefix = "-" not in uid
    results = []
    for split in data.keys():
        for idx, sid in enumerate(data[split]["scene_id"]):
            if (is_prefix and sid.startswith(uid)) or sid == uid:
                results.append(data[split][idx])
    return results


def denorm_colors(pixel_values_1chw):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = (pixel_values_1chw[0].cpu() * std + mean).clamp(0, 1)
    return (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset).absolute().as_posix()
    print(f"[vis_gtpose] Loading dataset from {dataset_path}")
    data = datasets.load_dataset(dataset_path)

    examples = find_all_views(data, args.uid)
    if not examples:
        raise ValueError(f"scene_id '{args.uid}' not found in any split")
    if args.max_views > 0:
        examples = examples[:args.max_views]

    scene_id = examples[0]["scene_id"]
    N = len(examples)
    print(f"[vis_gtpose] scene_id: {scene_id}  ({N} views)")

    preprocessor = AutoImageProcessor.from_pretrained(
        args.image_encoder, size_divisor=args.size_divisor
    )
    print(f"[vis_gtpose] Loading Pi3X from {args.pi3x_ckpt}")
    encoder = Pi3XFrozenEncoder(
        ckpt_path=args.pi3x_ckpt,
        greedy_anchor=True,
        disable_multimodal=True,
    )
    encoder.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)

    all_xyz, all_rgb = [], []

    for view_idx, ex in enumerate(examples):
        image  = np.array(ex["image"])
        pv     = preprocessor(images=image, return_tensors="pt")["pixel_values"]
        pv_dev = pv.to(device)

        with torch.no_grad():
            out = encoder.forward(pv_dev)

        local_pts = out["local_points"][0]    # (H, W, 3) camera-space
        conf      = out["conf"][0]            # (H, W, 1)
        conf_map  = torch.sigmoid(conf).squeeze(-1)

        if args.edge_rtol > 0:
            edge_mask = depth_edge(local_pts[..., 2].unsqueeze(0), rtol=args.edge_rtol)
            conf_map  = conf_map.clone()
            conf_map[edge_mask.squeeze(0)] = 0.0

        conf_np = conf_map.cpu().numpy()
        depth   = local_pts[..., 2].cpu().numpy()
        mask    = (conf_np > args.conf_thresh) & (depth > 0)

        xyz_cam  = local_pts.cpu().numpy()[mask]
        wrd2cam  = np.array(ex["wrd2cam"], dtype=np.float64)
        cam2wrd  = np.linalg.inv(wrd2cam)
        xyz_hom  = np.concatenate(
            [xyz_cam, np.ones((len(xyz_cam), 1), dtype=np.float64)], axis=-1
        )
        xyz_world = (cam2wrd @ xyz_hom.T).T[:, :3].astype(np.float32)

        colors = denorm_colors(pv)
        rgb    = colors[mask]

        xyz_viz = xyz_world.copy()
        if not args.no_yflip:
            xyz_viz[:, 1] *= -1

        all_xyz.append(xyz_viz)
        all_rgb.append(rgb)

        Image.fromarray(image).save(out_dir / f"{args.uid}_view{view_idx}_input.png")
        print(f"[vis_gtpose] view {view_idx}: {mask.sum():,} pts  "
              f"({100 * mask.mean():.1f}%)  "
              f"depth [{depth[mask].min():.2f}, {depth[mask].max():.2f}]")

    xyz_all = np.concatenate(all_xyz, axis=0)
    rgb_all = np.concatenate(all_rgb, axis=0)
    n_raw   = len(xyz_all)

    if args.voxel_size > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_all)
        pcd.colors = o3d.utility.Vector3dVector(rgb_all / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        xyz_all = np.asarray(pcd.points)
        rgb_all = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        print(f"[vis_gtpose] Voxel {args.voxel_size}m: {n_raw:,} → {len(xyz_all):,} pts")

    ply_path = out_dir / f"{args.uid}_gtpose.ply"
    write_ply(xyz_all, rgb_all, path=str(ply_path))
    print(f"[vis_gtpose] Merged {N} views → {len(xyz_all):,} pts")
    print(f"[vis_gtpose] Saved → {ply_path}")


if __name__ == "__main__":
    main()
