"""Evaluate MV stage-1 layout prediction quality vs GT bboxes.

Loads a checkpoint, runs the MV conditioning prefix, generates layout tokens (no GT),
decodes to 8-corner bbox, and compares against GT bboxes.

Usage (single GPU, small sample):
    PYTHONPATH=. python scripts/eval_layout_mv.py \
        --checkpoint outputs/edgerunner-3d-front-multiview-stage1/20260628-233241/checkpoints/final \
        [--limit 200] [--batch-size 4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collator import get_mesh_data_collator
from src.data.utils import dequantize_points
from src.utils.inference import prepare_mv_model_for_inference, prepare_mv_test_set


def decode_layout_tokens(tokens, pos_token_offset, num_pos_tokens):
    """tokens: np.ndarray (24,) of raw generated ids → (8, 3) float corners in [-1,1]."""
    q = (tokens - pos_token_offset).reshape(8, 3)
    return dequantize_points(q, num_pos_tokens)


def bbox_center(corners):
    return corners.mean(axis=0)


def bbox_size(corners):
    return corners.max(axis=0) - corners.min(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    state = PartialState()
    device = state.device

    model, model_cfg, data_cfg = prepare_mv_model_for_inference(
        checkpoint=args.checkpoint,
        config_name="edgerunner_3d_front_multiview_stage1",
    )
    model.to(device)
    model.eval()

    collator = get_mesh_data_collator(data_cfg, model_cfg)
    prefix_len = collator.prefix_len
    pc_token_id = collator.pc_token_id
    bos_token_id = collator.bos_token_id
    pos_token_offset = model_cfg.pos_token_offset
    num_pos_tokens = data_cfg.num_pos_tokens

    test_set = prepare_mv_test_set(data_cfg)
    n_total = len(test_set) if args.limit is None else min(len(test_set), args.limit)
    indices = list(range(n_total))
    shard = indices[state.process_index :: state.num_processes]

    bs = args.batch_size
    n_iters = (len(shard) + bs - 1) // bs

    center_errs, size_errs, corner_errs = [], [], []

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for it in tqdm(range(n_iters), position=state.process_index, desc="eval_layout"):
                batch_idx = shard[it * bs : (it + 1) * bs]
                examples = [test_set[i] for i in batch_idx]
                batch = collator(examples)

                # Build prefix WITHOUT GT layout: [pc]*prefix_len + [bos]
                prefix_ids = torch.full(
                    (len(examples), prefix_len + 1),
                    fill_value=pc_token_id,
                    dtype=torch.long,
                    device=device,
                )
                prefix_ids[:, -1] = bos_token_id

                def _to(x):
                    return x.to(device) if torch.is_tensor(x) else x

                try:
                    inputs_embeds = model.get_mv_inputs_with_cond(
                        input_ids=prefix_ids,
                        pixel_values=_to(batch["pixel_values"]),
                        scene_transforms=_to(batch["scene_transforms"]),
                        K_per_view=_to(batch["K_per_view"]),
                        view_mask=_to(batch["view_mask"]),
                        panoptic_masks=_to(batch.get("panoptic_masks")),
                        cond_pcs=_to(batch["cond_pcs"]),
                        cond_pcs_2d=_to(batch["cond_pcs_2d"]),
                        cond_num_faces=None,
                        obj_canon_transform=_to(batch.get("obj_canon_transform")),
                        gt_obj_vertices=_to(batch.get("gt_obj_vertices")),
                        ref_view=_to(batch.get("ref_view")),
                    )
                except Exception as e:
                    if state.is_main_process:
                        print(f"[WARN] batch {it} conditioning failed: {e}")
                    continue

                # Generate 25 tokens: 24 layout coords + 1 indicator/EOS
                try:
                    results = model.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=25,
                        use_cache=True,
                        do_sample=False,
                    )
                    # With inputs_embeds, generate() returns only new tokens: (B, 25)
                    results_cpu = results.cpu().numpy()
                    layout_tokens = results_cpu[:, :24]  # drop last (indicator/EOS) → (B, 24)
                except Exception as e:
                    if state.is_main_process:
                        print(f"[WARN] batch {it} generation failed: {e}")
                    continue

                for i, ex in enumerate(examples):
                    bboxes = np.array(ex["bboxes"], dtype=np.float32)
                    obj_idx = int(ex["obj_indices"])
                    gt_corners = bboxes[obj_idx]  # (8, 3) in normalized space

                    try:
                        pred_corners = decode_layout_tokens(
                            layout_tokens[i], pos_token_offset, num_pos_tokens
                        )
                    except Exception:
                        continue

                    gt_center = bbox_center(gt_corners)
                    pred_center = bbox_center(pred_corners)
                    center_errs.append(float(np.linalg.norm(pred_center - gt_center)))

                    gt_size = bbox_size(gt_corners)
                    pred_size = bbox_size(pred_corners)
                    size_errs.append(float(np.linalg.norm(pred_size - gt_size)))

                    # Align predicted corners to GT center for per-corner RMSE
                    # (removes global translation so we measure shape, not position)
                    pred_shifted = pred_corners - pred_center + gt_center
                    corner_errs.append(
                        float(np.sqrt(np.mean(np.sum((pred_shifted - gt_corners) ** 2, axis=-1))))
                    )

    # Gather results from all ranks onto rank 0
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * state.num_processes
        dist.all_gather_object(gathered, (center_errs, size_errs, corner_errs))
        if state.is_main_process:
            all_center = [v for g in gathered for v in g[0]]
            all_size   = [v for g in gathered for v in g[1]]
            all_corner = [v for g in gathered for v in g[2]]
    else:
        all_center, all_size, all_corner = center_errs, size_errs, corner_errs

    if state.is_main_process:
        print("\n========== Layout Eval Results ==========")
        if all_center:
            n = len(all_center)
            print(f"  N objects evaluated : {n}")
            print(f"  Center L2 (norm)    : mean={np.mean(all_center):.4f}  "
                  f"med={np.median(all_center):.4f}  p90={np.percentile(all_center,90):.4f}")
            print(f"  Size   L2 (norm)    : mean={np.mean(all_size):.4f}  "
                  f"med={np.median(all_size):.4f}  p90={np.percentile(all_size,90):.4f}")
            print(f"  Corner RMSE (norm)  : mean={np.mean(all_corner):.4f}  "
                  f"med={np.median(all_corner):.4f}  p90={np.percentile(all_corner,90):.4f}")
            for thr in [0.05, 0.10, 0.20]:
                hit = np.mean(np.array(all_center) < thr) * 100
                print(f"  Center err < {thr:.2f}    : {hit:.1f}%")
        else:
            print("  No objects successfully evaluated.")
        print("=========================================")


if __name__ == "__main__":
    main()
