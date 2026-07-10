#!/usr/bin/env python3
"""Write repeatable command manifests for MV layout-loss evidence runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ABLATIONS = {
    "A_ce": "mv_layout_loss_ce",
    "B_ordinal": "mv_layout_loss_ordinal",
    "C_coord": "mv_layout_loss_coord",
    "D_geometry": "mv_layout_loss_geometry",
}

NEGATIVE_CONTROLS = {
    "no_aabb": ["dataset.model.mv_obj_aabb_token=false"],
    "no_voxel_encoder": ["dataset.model.mv_use_voxel_encoder=false"],
    "no_obj_pc_cond": ["dataset.model.mv_obj_pc_cond=false"],
    "no_obj_pc_appearance": ["dataset.model.mv_obj_pc_appearance=false"],
    "one_view_eval": ["--view-limit", "1"],
    "two_view_eval": ["--view-limit", "2"],
    "four_view_eval": ["--view-limit", "4"],
    "eight_view_eval": ["--view-limit", "8"],
    "reference_only_eval": ["--reference-only"],
    "shuffled_views_eval": ["--shuffle-views"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/da3/experiments/mv_layout_loss_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--stage1-config", default="edgerunner_3d_front_trellis2_mv_stage1")
    parser.add_argument("--stage2-config", default="edgerunner_3d_front_trellis2_mv_stage2")
    parser.add_argument("--sv-downstream", default="outputs/sv/eval/baseline/eval_obj_results.jsonl")
    parser.add_argument("--mv-feature-cache", default="${MV_FEATURE_CACHE:-datasets/mv-feature-cache/da3/trellis2-mv}")
    parser.add_argument("--python", default="${PYTHON:-micromamba run -n pixarmesh124 python}")
    parser.add_argument("--gpus", default="${GPUS:-0,1,2,3}")
    parser.add_argument("--num-processes", default="${NP:-4}")
    parser.add_argument("--stage1-steps", type=int, default=5000)
    parser.add_argument("--stage2-steps", type=int, default=30000)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--stage2-infer-batch-size", type=int, default=8)
    parser.add_argument("--eval-obj-dataset", default="datasets/3d-front-ar-packed")
    parser.add_argument("--eval-obj-metadata", default="metadata/test_obj_sub_100.jsonl")
    parser.add_argument("--eval-obj-gt-dir", default="datasets/3D-FUTURE-model-ply")
    parser.add_argument("--eval-obj-points", type=int, default=10000)
    parser.add_argument("--eval-obj-mask-area-thresh", type=int, default=1600)
    return parser.parse_args()


def _quote_parts(parts: list[str]) -> str:
    return " ".join(parts)


def _as_eval_overrides(overrides: list[str] | None) -> list[str]:
    return [f"--override {override}" for override in overrides or []]


def ensure_clean_command(run_name: str, seed: int | str, *, root: str = "outputs/da3/train/mv_layout_loss") -> str:
    run_dir = f"{root}/{run_name}/seed{seed}/checkpoints"
    return f"test ! -e {run_dir} || {{ echo '[mv_layout_loss_ablation] existing output: {run_dir}'; exit 2; }}"


def train_stage1_command(args: argparse.Namespace, run_name: str, exp_cfg: str, seed: int, extra: list[str] | None = None) -> str:
    output_dir = f"outputs/da3/train/mv_layout_loss/{run_name}"
    overrides = [
        f"--config-name={args.stage1_config}",
        f"+experiment={exp_cfg}",
        f"all.name={run_name}",
        f"all.output_dir={output_dir}",
        f"train.train_args.seed={seed}",
        f"train.train_args.data_seed={seed}",
        f"train.train_args.max_steps={args.stage1_steps}",
        "train.train_args.resume_from_checkpoint=false",
        f"dataset.src_data.mv_feature_cache={args.mv_feature_cache}",
    ]
    if extra:
        overrides.extend(extra)
    run_ts = f"seed{seed}"
    return (
        f"RUN_TS={run_ts} CUDA_VISIBLE_DEVICES={args.gpus} {args.python} "
        f"launch.py --num_processes {args.num_processes} train.py {_quote_parts(overrides)}"
    )


def eval_command(
    args: argparse.Namespace,
    run_name: str,
    seed: int | str,
    extra: list[str] | None = None,
    out_override: str | None = None,
) -> str:
    ckpt = f"outputs/da3/train/mv_layout_loss/{run_name}/seed{seed}/checkpoints/final"
    out = out_override or f"outputs/da3/eval/layout_mv/{run_name}/seed{seed}"
    parts = [
        f"{args.python}",
        "scripts/eval/eval_layout_mv.py",
        f"--checkpoint {ckpt}",
        f"--config-name {args.stage1_config}",
        f"--num-samples {args.eval_samples}",
        f"--out {out}",
        f"--mv-feature-cache {args.mv_feature_cache}",
    ]
    if extra:
        parts.extend(extra)
    return " ".join(parts)


def train_stage2_command(args: argparse.Namespace, run_name: str, seed: int, stage1_run_name: str) -> str:
    stage1_ckpt = f"outputs/da3/train/mv_layout_loss/{stage1_run_name}/seed{seed}/checkpoints/final"
    output_dir = f"outputs/da3/train/mv_layout_loss_stage2/{run_name}"
    overrides = [
        f"--config-name={args.stage2_config}",
        f"model.local_path={stage1_ckpt}",
        f"all.name={run_name}",
        f"all.output_dir={output_dir}",
        f"train.train_args.seed={seed}",
        f"train.train_args.data_seed={seed}",
        f"train.train_args.max_steps={args.stage2_steps}",
        "train.train_args.resume_from_checkpoint=false",
        f"dataset.src_data.mv_feature_cache={args.mv_feature_cache}",
    ]
    return (
        f"RUN_TS=seed{seed} CUDA_VISIBLE_DEVICES={args.gpus} {args.python} "
        f"launch.py --num_processes {args.num_processes} train.py {_quote_parts(overrides)}"
    )


def stage2_checkpoint(run_name: str, seed: int) -> str:
    return f"outputs/da3/train/mv_layout_loss_stage2/{run_name}/seed{seed}/checkpoints/final"


def stage2_infer_root(run_name: str, seed: int) -> str:
    return f"outputs/da3/infer/mv_layout_loss_stage2/{run_name}/seed{seed}"


def stage2_pred_dir(run_name: str, seed: int) -> str:
    return f"{stage2_infer_root(run_name, seed)}/obj/edgerunner/gt_layout_gt_mask_pred_depth"


def stage2_eval_dir(seed: int) -> str:
    return f"outputs/da3/eval/stage2_best/seed{seed}"


def infer_stage2_command(args: argparse.Namespace, run_name: str, seed: int) -> str:
    return (
        f"RUN_TS=eval_seed{seed} CUDA_VISIBLE_DEVICES={args.gpus} {args.python} "
        f"launch.py --num_processes {args.num_processes} --module scripts.infer "
        f"--model-type edgerunner --run-type obj "
        f"--checkpoint {stage2_checkpoint(run_name, seed)} "
        f"--output-dir {stage2_infer_root(run_name, seed)} "
        f"--batch-size {args.stage2_infer_batch_size} "
        f"--seed {seed}"
    )


def eval_stage2_command(args: argparse.Namespace, run_name: str, seed: int) -> str:
    return (
        f"RUN_TS=eval_seed{seed} CUDA_VISIBLE_DEVICES={args.gpus} {args.python} "
        f"launch.py --num_processes {args.num_processes} --module scripts.eval_obj "
        f"--dataset {args.eval_obj_dataset} "
        f"--metadata {args.eval_obj_metadata} "
        f"--gt-dir {args.eval_obj_gt_dir} "
        f"--pred-dir {stage2_pred_dir(run_name, seed)} "
        f"--save-dir {stage2_eval_dir(seed)} "
        f"--num-sample-points {args.eval_obj_points} "
        f"--mask-area-thresh {args.eval_obj_mask_area_thresh} "
        f"--overwrite"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    manifest: dict[str, object] = {
        "seeds": args.seeds,
        "stage1_config": args.stage1_config,
        "stage2_config": args.stage2_config,
        "sv_downstream": args.sv_downstream,
        "ablations": [],
        "negative_controls": NEGATIVE_CONTROLS,
    }

    for label, exp_cfg in ABLATIONS.items():
        for seed in args.seeds:
            run_name = label
            commands.append(ensure_clean_command(run_name, seed))
            commands.append(train_stage1_command(args, run_name, exp_cfg, seed))
            commands.append(eval_command(args, run_name, seed))
            commands.append("")
            manifest["ablations"].append(
                {"label": label, "experiment": exp_cfg, "seed": seed, "run_name": run_name}
            )

    commands.append("# Negative eval controls for the selected best run across the seed list.")
    commands.append(": \"${BEST_STAGE1_RUN:?set BEST_STAGE1_RUN to the verified stage-1 run name}\"")
    for name, extra in NEGATIVE_CONTROLS.items():
        if name.endswith("_eval"):
            for seed in args.seeds:
                commands.append(
                    eval_command(
                        args,
                        "${BEST_STAGE1_RUN}",
                        seed,
                        extra,
                        out_override=f"outputs/da3/eval/layout_mv/${{BEST_STAGE1_RUN}}/seed{seed}/{name}",
                    )
                )
    commands.append("")
    commands.append("# Negative training controls for the selected best loss stack.")
    commands.append(": \"${BEST_EXPERIMENT:?set BEST_EXPERIMENT to mv_layout_loss_ordinal/coord/geometry/etc.}\"")
    commands.append(": \"${BEST_STAGE1_RUN:?set BEST_STAGE1_RUN to the verified stage-1 run name}\"")
    for name, extra in NEGATIVE_CONTROLS.items():
        if name.endswith("_eval"):
            continue
        for seed in args.seeds:
            run_name = f"${{BEST_STAGE1_RUN}}_{name}"
            commands.append(ensure_clean_command(run_name, seed))
            commands.append(train_stage1_command(args, run_name, "${BEST_EXPERIMENT}", seed, extra))
            commands.append(eval_command(args, run_name, seed, _as_eval_overrides(extra)))
            commands.append("")

    commands.append("# Carry the best verified stage-1 loss into stage 2 after metrics pick BEST_STAGE1_RUN.")
    for seed in args.seeds:
        run_name = f"E_stage2_best_seed{seed}"
        commands.append(
            ensure_clean_command(
                run_name,
                seed,
                root="outputs/da3/train/mv_layout_loss_stage2",
            )
        )
        commands.append(train_stage2_command(args, run_name, seed, "${BEST_STAGE1_RUN}"))
        commands.append(infer_stage2_command(args, run_name, seed))
        commands.append(eval_stage2_command(args, run_name, seed))
        commands.append("")

    commands.append("")
    commands.append("# Summarize layout evidence and attach downstream CD/F after selecting the E-stage seed or merged downstream file.")
    commands.append(f": \"${{SV_DOWNSTREAM:={args.sv_downstream}}}\"")
    commands.append(": \"${BEST_SEED:?set BEST_SEED to the E-stage seed used for downstream CD/F if using the default BEST_STAGE2_DOWNSTREAM}\"")
    commands.append(f": \"${{BEST_STAGE2_DOWNSTREAM:=outputs/da3/eval/stage2_best/seed${{BEST_SEED}}/eval_obj_results.jsonl}}\"")
    seed_args = " ".join(str(seed) for seed in args.seeds)
    run_args = " ".join(f"--run {name}" for name in ABLATIONS)
    commands.append(
        f"{args.python} scripts/eval/summarize_mv_layout_evidence.py "
        f"--layout-root outputs/da3/eval/layout_mv "
        f"{run_args} "
        f"--seeds {seed_args} "
        f"--ce-run A_ce "
        f"--sv-downstream ${{SV_DOWNSTREAM}} "
        f"--downstream E_stage2_best=${{BEST_STAGE2_DOWNSTREAM}} "
        f"--out {args.out}/evidence_summary"
    )
    commands.append("")
    commands.append("# Write the deterministic council review after negative controls exist.")
    best_reports = " ".join(
        f"--best-layout-report outputs/da3/eval/layout_mv/${{BEST_STAGE1_RUN}}/seed{seed}/report.json"
        for seed in args.seeds
    )
    council_controls = []
    for name in NEGATIVE_CONTROLS:
        for seed in args.seeds:
            if name.endswith("_eval"):
                path = f"outputs/da3/eval/layout_mv/${{BEST_STAGE1_RUN}}/seed{seed}/{name}/report.json"
            else:
                path = f"outputs/da3/eval/layout_mv/${{BEST_STAGE1_RUN}}_{name}/seed{seed}/report.json"
            council_controls.append(f"--negative-control {name}={path}")
    commands.append(
        f"{args.python} scripts/eval/write_mv_layout_council_review.py "
        f"--summary-json {args.out}/evidence_summary/summary.json "
        f"--best-layout-run ${{BEST_STAGE1_RUN}} "
        f"{best_reports} "
        f"--downstream-run E_stage2_best "
        f"{' '.join(council_controls)} "
        f"--out {args.out}/verifiers/council_review.md"
    )
    commands.append(
        f"{args.python} scripts/eval/check_mv_layout_evidence_bundle.py "
        f"--layout-root outputs/da3/eval/layout_mv "
        f"{run_args} "
        f"--seeds {seed_args} "
        f"--ce-run A_ce "
        f"--sv-downstream ${{SV_DOWNSTREAM}} "
        f"--downstream E_stage2_best=${{BEST_STAGE2_DOWNSTREAM}} "
        f"--verifier-dir {args.out}/verifiers "
        f"--require-visuals "
        f"--out {args.out}/evidence_bundle_check.json"
    )

    (out_dir / "commands.sh").write_text("\n".join(commands) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_dir / 'commands.sh'}")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
