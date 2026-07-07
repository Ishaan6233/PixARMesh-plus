import argparse
import csv
import jsonlines
import datasets
import open3d as o3d
import numpy as np
import json
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from src.utils import evaluation
from src.data.utils import get_masks_by_ids


def flatten_3d_front_for_eval(examples, erode_size=0):
    result_scene_ids = []
    result_obj_ids = []
    result_model_ids = []
    result_mask_areas = []
    all_uids = examples["uid"]
    all_objects = examples["objects"]
    all_masks = examples["panoptic_mask"]
    for scene_uid, objs, pano_mask in zip(all_uids, all_objects, all_masks):
        model_ids = objs["model_ids"]
        inst_ids = objs["inst_ids"]
        n_insts = len(model_ids)
        obj_masks = get_masks_by_ids(
            pano_mask,
            inst_ids,
            erode_size=erode_size,
        )
        mask_area = obj_masks.sum(axis=(1, 2))
        result_scene_ids.extend([scene_uid] * n_insts)
        result_obj_ids.extend(list(range(n_insts)))
        result_model_ids.extend(model_ids)
        result_mask_areas.extend(mask_area.tolist())
    return {
        "uid": result_scene_ids,
        "obj_id": result_obj_ids,
        "model_id": result_model_ids,
        "mask_area": result_mask_areas,
    }


def get_mesh(mesh_path: Path):
    if mesh_path.exists():
        return o3d.io.read_triangle_mesh(str(mesh_path))
    return None


def _load_hf_split(path: str, preferred=("validation", "val", "test", "train")):
    local_path = Path(path).absolute().as_posix()
    try:
        data = datasets.load_from_disk(local_path)
    except Exception:
        data = datasets.load_dataset(local_path)
    if isinstance(data, datasets.Dataset):
        return data
    split = next(s for s in preferred if s in data)
    return data[split]


def build_trellis2_mv_eval_subset(
    trellis2_path: str,
    hf_path: str,
    compute_mask_area: bool = True,
):
    """Evaluation subset matching Trellis2MVDataset(is_train=False).

    Trellis2-MV inference emits only the deterministic 5% validation split from
    the per-object mesh dataset. The older --mv-dataset path scores an entire HF
    split, which gives the wrong denominator for Trellis2 checkpoints.
    """
    import torch

    root = Path(trellis2_path)
    instances = []
    with (root / "metadata.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            instances.append(row["sha256"])

    rng = np.random.RandomState(42)
    all_idx = list(range(len(instances)))
    rng.shuffle(all_idx)
    n_val = max(1, int(0.05 * len(all_idx)))
    val_instances = [instances[i] for i in sorted(all_idx[:n_val])]

    filter_path = root / "conditioning_filter.csv"
    if filter_path.exists():
        keep = set()
        with filter_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if str(row["keep"]).strip().lower() in ("true", "1"):
                    keep.add(row["sha256"])
        val_instances = [sha for sha in val_instances if sha in keep]

    hf = _load_hf_split(hf_path, preferred=("train", "validation", "val", "test"))
    keep_cols = ("uid", "objects")
    if compute_mask_area:
        keep_cols = keep_cols + ("panoptic_masks", "panoptic_mask")
    hf = hf.remove_columns([c for c in hf.column_names if c not in keep_cols])
    uid_to_row = {u if isinstance(u, str) else u[0]: i for i, u in enumerate(hf["uid"])}

    subset = []
    missing = 0
    for sha in val_instances:
        cond_path = root / "mv_cond" / f"{sha}.pt"
        cond = torch.load(cond_path, map_location="cpu", weights_only=False)["cond"]
        uid = cond["uid"]
        row_idx = uid_to_row.get(uid)
        if row_idx is None:
            missing += 1
            continue
        row = hf[row_idx]
        objs = row.get("objects")
        mids = objs.get("model_ids") if objs else None
        if not mids:
            missing += 1
            continue
        model_id = mids[0]

        area = 10 ** 9
        if compute_mask_area:
            pan = row.get("panoptic_masks")
            if pan is None:
                pan = row.get("panoptic_mask")
            inst_ids = objs.get("inst_ids") if objs else None
            if pan is not None and inst_ids:
                ref_mask = pan[0] if isinstance(pan, list) else pan
                try:
                    masks = get_masks_by_ids(ref_mask, [inst_ids[0]])
                    area = int(np.asarray(masks[0]).sum())
                except Exception:
                    area = 10 ** 9

        subset.append(
            {"uid": uid, "obj_id": None, "model_id": model_id, "mask_area": area}
        )

    if missing:
        print(f"[eval_obj] WARNING: skipped {missing} Trellis2 val items missing HF metadata")
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/3d-front-ar-packed")
    parser.add_argument(
        "--metadata", type=str, default="metadata/test_obj_sub_100.jsonl"
    )
    parser.add_argument("--gt-dir", type=str, default="datasets/3D-FUTURE-model-ply")
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument(
        "--num-sample-points",
        type=int,
        default=10000,
        help="Number of points to sample from each mesh for evaluation",
    )
    parser.add_argument(
        "--mask-area-thresh",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip pose alignment; compare normalized clouds directly (for eval protocol testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing eval results",
    )
    parser.add_argument("--save-dir", type=str, default="outputs/evaluations-obj")
    parser.add_argument(
        "--mv-dataset",
        action="store_true",
        help="Evaluate multi-view PLYs named <uid>.ply against GT via the "
             "3d-front-multiview uid->model_id map (one item per object).",
    )
    parser.add_argument("--mv-path", type=str, default="datasets/3d-front-multiview")
    parser.add_argument(
        "--trellis2-mv-dataset",
        action="store_true",
        help="Evaluate Trellis2-MV PLYs on the same deterministic filtered 5% val split "
             "used by configs/edgerunner_3d_front_trellis2_mv*.yaml.",
    )
    parser.add_argument(
        "--trellis2-path",
        type=str,
        default="datasets/mesh_datasets/datasets/3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629",
    )
    parser.add_argument(
        "--trellis2-hf-path",
        type=str,
        default="datasets/3d-front-multiview-full",
    )
    parser.add_argument(
        "--trellis2-no-mask-area",
        action="store_true",
        help="Do not decode HF panoptic masks for Trellis2 mask-area filtering; all "
             "Trellis2 val objects pass --mask-area-thresh. Use only for diagnostics.",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.trellis2_mv_dataset:
        subset = build_trellis2_mv_eval_subset(
            args.trellis2_path,
            args.trellis2_hf_path,
            compute_mask_area=not args.trellis2_no_mask_area,
        )
        sharded_subset = subset[accelerator.process_index :: accelerator.num_processes]
    elif args.mv_dataset:
        # Multi-view: one item per object PLY (<uid>.ply); GT via uid->model_id map.
        from src.utils.inference import build_mv_uid_to_model_id
        from src.utils.config import DataConfig

        uid2mid = build_mv_uid_to_model_id(
            DataConfig(type="3d-front-multiview", path=args.mv_path)
        )
        # Real ref-view mask area applies the same small-object filter as the SV protocol
        # (matching mask_area_thresh), so MV is scored on a comparable object population.
        subset = [
            {"uid": uid, "obj_id": None, "model_id": mid, "mask_area": area}
            for uid, (mid, area) in uid2mid.items()
        ]
        sharded_subset = subset[accelerator.process_index :: accelerator.num_processes]
    else:
        with jsonlines.open(args.metadata, "r") as reader:
            valid_uids = {line["image_id"] for line in reader}
        with accelerator.local_main_process_first():
            dataset = datasets.load_dataset(args.dataset, split="test", num_proc=16)
        valid_indices = [
            i for i, uid in enumerate(dataset["uid"]) if str(uid) in valid_uids
        ]
        subset = dataset.select(valid_indices)
        with accelerator.local_main_process_first():
            subset = subset.map(
                flatten_3d_front_for_eval,
                batched=True,
                batch_size=4,
                remove_columns=subset.column_names,
            )
        sharded_subset = subset.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index,
        )

    for item in tqdm(sharded_subset):
        uid = item["uid"]
        obj_id = item["obj_id"]
        model_id = item["model_id"]
        mask_area = item["mask_area"]
        if mask_area < args.mask_area_thresh:
            continue

        stem = f"{uid}" if obj_id is None else f"{uid}_{obj_id}"
        gt_mesh_path = gt_dir / f"{model_id}.ply"
        pred_mesh_path = pred_dir / f"{stem}.ply"
        out_json_path = save_dir / f"{stem}.json"

        if out_json_path.exists() and not args.overwrite:
            continue

        gt_mesh = get_mesh(gt_mesh_path)
        pred_mesh = get_mesh(pred_mesh_path)
        has_gt = gt_mesh is not None
        has_pred = pred_mesh is not None

        record = {
            "uid": uid,
            "obj_id": obj_id,
            "model_id": model_id,
            "has_gt": has_gt,
            "has_pred": has_pred,
            "cd": None,
            "f_score": None,
        }

        # A degenerate decode (0 faces / 0 surface area) still writes a valid PLY, so
        # has_pred=True, but open3d's sample_points_uniformly raises on it. Treat it as
        # an unscorable prediction (still counts toward coverage; cd/f_score left None)
        # rather than letting the exception kill the whole eval shard.
        if has_gt and has_pred and len(pred_mesh.triangles) == 0:
            record["degenerate_pred"] = True
        elif has_gt and has_pred:
            try:
                gt_pcds = evaluation.sample_points_from_o3d_mesh(
                    gt_mesh, args.num_sample_points
                )
                pred_pcds = evaluation.sample_points_from_o3d_mesh(
                    pred_mesh, args.num_sample_points
                )
                gt_pcds = evaluation.get_normalized_pcd(gt_pcds)
                pred_pcds = evaluation.get_normalized_pcd(pred_pcds)
                if args.no_align:
                    eval_pred_pcds = pred_pcds
                else:
                    transform_matrices = evaluation.get_object_transformations(
                        [pred_pcds], [gt_pcds]
                    )
                    eval_pred_pcds = evaluation.apply_transformation_matrix(
                        pred_pcds,
                        transform_matrices[0],
                    )
                cd_loss = chamfer_distance(
                    gt_pcds.unsqueeze(0).cuda(),
                    eval_pred_pcds.unsqueeze(0).cuda(),
                )[0].item()
                f_score = evaluation.f_score(gt_pcds.numpy(), eval_pred_pcds.numpy())
                record["cd"] = float(cd_loss)
                record["f_score"] = float(f_score)
            except (RuntimeError, ValueError) as e:
                # e.g. open3d "Invalid surface area 0" on a triangulated-but-zero-area mesh
                record["degenerate_pred"] = True
                record["error"] = str(e)

        with out_json_path.open("w") as f:
            json.dump(record, f)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        results = []
        all_cds = []
        all_f_scores = []
        n_total = 0          # objects above the mask threshold (the eval denominator)
        n_has_pred = 0       # objects that produced a USABLE (non-degenerate) mesh
        n_degenerate = 0     # objects whose PLY decoded to 0 faces / 0 surface area
        for item in subset:
            uid = item["uid"]
            obj_id = item["obj_id"]
            stem = f"{uid}" if obj_id is None else f"{uid}_{obj_id}"
            out_json_path = save_dir / f"{stem}.json"
            mask_area = item["mask_area"]
            if mask_area < args.mask_area_thresh:
                continue
            n_total += 1
            with out_json_path.open("r") as f:
                record = json.load(f)
            # A degenerate (0-face/0-area) decode wrote a PLY but produced no usable
            # geometry — it is a miss, not coverage. Exclude it from has_pred so the
            # coverage metric stays honest, and surface the count separately.
            if record.get("degenerate_pred"):
                n_degenerate += 1
            elif record.get("has_pred"):
                n_has_pred += 1
            if record["cd"] is not None:
                all_cds.append(record["cd"])
                all_f_scores.append(record["f_score"])
            results.append(record)

        # Coverage makes silent misses (no PLY / decode failures / degenerate meshes)
        # visible: a low CD over a fraction of objects is not a real win. Report it
        # alongside CD/F.
        coverage = n_has_pred / max(n_total, 1)
        avg_cd = float(np.mean(all_cds)) if all_cds else float("nan")
        avg_f_scores = float(np.mean(all_f_scores)) if all_f_scores else float("nan")
        results.append(
            {
                "avg_cd": avg_cd,
                "avg_f_score": avg_f_scores,
                "num_evaluated": len(all_cds),
                "num_total": n_total,
                "num_degenerate": n_degenerate,
                "coverage": coverage,
            }
        )
        results_path = save_dir / "eval_obj_results.jsonl"
        with jsonlines.open(results_path, "w") as writer:
            writer.write_all(results)
        print(
            f"""
Evaluation results saved to {results_path}.
Num valid objects (scored): {len(all_cds)} / {n_total} total
Coverage (usable mesh): {coverage * 100:.1f}%   (degenerate decodes: {n_degenerate})
Average Chamfer Distance (x10^{-3}): {avg_cd * 1000:.3f}
Average F-Score (%): {avg_f_scores:.3f}
"""
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
