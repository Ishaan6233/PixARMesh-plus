"""Prepare a multi-view 3D-FRONT dataset for PixARMesh+ training.

Takes the per-scene "3d-front-ar-packed" dataset (one row per scene, one camera
view per row) and groups by scene_id to produce one row per object with N views:

Output schema per row:
  uid          str           unique object identifier
  scene_id     str           scene identifier
  images       List[Image]   N RGB images
  depths       List[Image]   N depth images
  Ks           List[array]   N (3,3) intrinsic matrices
  wrd2cam_rects List[array]  N (4,4) world-to-camera-rect matrices
  rect_invs    List[array]   N (3,3) rectification inverses
  objects      dict          single-object dict with vertices, faces, bounds, etc.

Usage:
  python scripts/data/prepare_multiview_dataset.py \\
      --input  datasets/3d-front-ar-packed/data \\
      --output datasets/3d-front-multiview \\
      [--max-views 8]  \\
      [--min-views 2]  \\
      [--splits val test]
"""

import argparse
import os
from collections import defaultdict
import datasets
from datasets import Dataset, DatasetDict


def group_by_scene(dataset):
    """Return {scene_id: [row_idx, ...]} mapping."""
    groups = defaultdict(list)
    for i, scene_id in enumerate(dataset["scene_id"]):
        groups[scene_id].append(i)
    return groups


def build_multiview_generator(dataset, scene_groups, max_views=None, min_views=1):
    """Yield one multi-view row per object.

    Uses batch index access (dataset[indices]) instead of per-row access to
    avoid repeated Arrow scans, and yields one row at a time so the caller
    can stream to disk via Dataset.from_generator without accumulating all rows
    in memory first.
    """
    has_pan = "panoptic_mask" in dataset.column_names

    for scene_id, indices in scene_groups.items():
        if max_views is not None:
            indices = indices[:max_views]
        if len(indices) < min_views:
            continue

        # Single batch access — one Arrow scan for all N views of this scene.
        batch = dataset[indices]               # dict of lists, length N
        objects = batch["objects"][0]          # all views share the same object list

        num_objects = len(objects["model_ids"])
        for obj_idx in range(num_objects):
            uid = f"{scene_id}__obj{obj_idx:04d}"

            obj_keys = ["model_ids", "bounds", "transforms", "inst_ids",
                        "vertices", "faces"]
            single_obj = {}
            for k in obj_keys:
                if k in objects:
                    val = objects[k]
                    single_obj[k] = [val[obj_idx]] if isinstance(val, list) else val

            row = {
                "uid":           uid,
                "scene_id":      scene_id,
                "images":        batch["image"],
                "depths":        batch["depth"],
                "Ks":            batch["K"],
                "wrd2cam_rects": batch["wrd2cam_rect"],
                "rect_invs":     batch["rect_inv"],
                "objects":       single_obj,
            }
            if has_pan:
                row["panoptic_masks"] = batch["panoptic_mask"]
            yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="datasets/3d-front-ar-packed/data",
                        help="HuggingFace dataset path (parquet dir or save_to_disk dir)")
    parser.add_argument("--output",    default="datasets/3d-front-multiview",
                        help="Output dataset path")
    parser.add_argument("--max-views", type=int, default=None,
                        help="Cap number of views per scene (None = unlimited)")
    parser.add_argument("--min-views", type=int, default=1,
                        help="Skip scenes with fewer than this many views")
    parser.add_argument("--splits",    nargs="+", default=None,
                        help="Only process these splits, e.g. --splits val test")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    if (os.path.isdir(args.input) and
            os.path.exists(os.path.join(args.input, "dataset_dict.json"))):
        raw = datasets.load_from_disk(args.input)
    else:
        raw = datasets.load_dataset(args.input)

    splits_to_process = args.splits if args.splits else list(raw.keys())
    print(f"Splits to process: {splits_to_process}")

    result = {}
    for split in splits_to_process:
        if split not in raw:
            print(f"  Warning: split '{split}' not found, skipping.")
            continue
        data = raw[split]
        print(f"  Processing split '{split}' ({len(data)} rows) ...")
        groups = group_by_scene(data)
        print(f"    Found {len(groups)} unique scenes")

        result[split] = Dataset.from_generator(
            build_multiview_generator,
            gen_kwargs={
                "dataset":      data,
                "scene_groups": groups,
                "max_views":    args.max_views,
                "min_views":    args.min_views,
            },
            writer_batch_size=50,  # avoid PyArrow 32-bit offset overflow on large image batches
        )
        print(f"    Generated {len(result[split])} multi-view rows")

    if not result:
        print("No splits processed — nothing to save.")
        return

    out = DatasetDict(result)
    print(f"Saving to {args.output} ...")
    os.makedirs(args.output, exist_ok=True)
    out.save_to_disk(args.output, max_shard_size="1GB")
    print("Done.")


if __name__ == "__main__":
    main()
