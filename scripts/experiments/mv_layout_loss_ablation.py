#!/usr/bin/env python3
"""Write repeatable command manifests for MV layout-loss evidence runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.mv_layout_evidence_common import (
    ABLATIONS,
    checkpoint_dir,
    DEFAULT_HF_DATASET,
    DEFAULT_MESH_DATASET,
    DEFAULT_MV_FEATURE_CACHE,
    NEGATIVE_CONTROLS,
    stage1_checkpoint,
    stage1_output_dir,
    STAGE1_TRAIN_ROOT,
    stage2_checkpoint,
    stage2_eval_dir,
    stage2_eval_results,
    stage2_infer_root,
    stage2_output_dir,
    stage2_pred_dir,
    stage2_run_name,
    STAGE2_TRAIN_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="outputs/da3/experiments/mv_layout_loss_ablation"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument(
        "--stage1-config", default="edgerunner_3d_front_trellis2_mv_stage1"
    )
    parser.add_argument(
        "--stage2-config", default="edgerunner_3d_front_trellis2_mv_stage2"
    )
    parser.add_argument(
        "--sv-downstream", default="outputs/sv/eval/baseline/eval_obj_results.jsonl"
    )
    parser.add_argument("--mv-feature-cache", default=DEFAULT_MV_FEATURE_CACHE)
    parser.add_argument("--mesh-dataset", default=DEFAULT_MESH_DATASET)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--da3-ckpt", default="")
    parser.add_argument("--cache-precompute-limit", type=int, default=0)
    parser.add_argument(
        "--python",
        default="PYTHONPATH=. ${PYTHON:-micromamba run -n pixarmesh124 python}",
    )
    parser.add_argument("--gpus", default="${GPUS:-0,1,2,3}")
    parser.add_argument("--num-processes", default="${NP:-4}")
    parser.add_argument("--stage1-steps", type=int, default=100000)
    parser.add_argument("--stage2-steps", type=int, default=30000)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--stage2-infer-batch-size", type=int, default=8)
    parser.add_argument("--eval-obj-dataset", default="datasets/3d-front-ar-packed")
    parser.add_argument(
        "--eval-obj-metadata", default="metadata/test_obj_sub_100.jsonl"
    )
    parser.add_argument("--eval-obj-gt-dir", default="datasets/3D-FUTURE-model-ply")
    parser.add_argument("--eval-obj-points", type=int, default=10000)
    parser.add_argument(
        "--eval-obj-align-sample-points",
        type=int,
        default=5000,
        help=(
            "Number of points used to fit eval_obj ICP alignment. "
            "Default 5000 matches the frozen SV baseline paper protocol."
        ),
    )
    parser.add_argument("--eval-obj-mask-area-thresh", type=int, default=1600)
    parser.add_argument(
        "--uid-metadata",
        default="",
        help=(
            "Optional JSON/JSONL/CSV uid metadata. When omitted, generated "
            "readiness, council, and bundle commands do not require or claim "
            "object-category stability."
        ),
    )
    return parser.parse_args()


def _quote_parts(parts: list[str]) -> str:
    return " ".join(parts)


def _as_eval_overrides(overrides: list[str] | None) -> list[str]:
    return [f"--override {override}" for override in overrides or []]


def accelerate_launch_prefix(args: argparse.Namespace) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={args.gpus} {args.python} "
        f"-m accelerate.commands.launch --num_processes {args.num_processes} "
        f"--gpu_ids {args.gpus}"
    )


def ensure_clean_command(
    run_name: str, seed: int | str, *, root: str = STAGE1_TRAIN_ROOT
) -> str:
    run_dir = checkpoint_dir(root, run_name, seed).as_posix()
    return f"test ! -e {run_dir} || {{ echo '[mv_layout_loss_ablation] existing output: {run_dir}'; exit 2; }}"


def train_stage1_command(
    args: argparse.Namespace,
    run_name: str,
    exp_cfg: str,
    seed: int,
    extra: list[str] | None = None,
) -> str:
    output_dir = stage1_output_dir(run_name)
    overrides = [
        f"--config-name={args.stage1_config}",
        f"+experiment={exp_cfg}",
        f"all.name={run_name}",
        f"all.output_dir={output_dir}",
        f"++train.train_args.seed={seed}",
        f"++train.train_args.data_seed={seed}",
        f"train.train_args.max_steps={args.stage1_steps}",
        "++train.train_args.resume_from_checkpoint=false",
        f"dataset.src_data.mv_feature_cache={args.mv_feature_cache}",
    ]
    if extra:
        overrides.extend(extra)
    run_ts = f"seed{seed}"
    return (
        f"RUN_TS={run_ts} {accelerate_launch_prefix(args)} "
        f"train.py {_quote_parts(overrides)}"
    )


def eval_command(
    args: argparse.Namespace,
    run_name: str,
    seed: int | str,
    extra: list[str] | None = None,
    out_override: str | None = None,
) -> str:
    ckpt = stage1_checkpoint(run_name, seed)
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


def build_figures_command(
    args: argparse.Namespace, run_args: str, seed_args: str
) -> str:
    return (
        f"{args.python} scripts/eval/build_mv_layout_evidence_figures.py "
        f"--summary-json {args.out}/evidence_summary/summary.json "
        f"--layout-root outputs/da3/eval/layout_mv "
        f"{run_args} "
        f"--seeds {seed_args} "
        f"--ce-run A_ce "
        f"--baseline-run A_ce "
        f"--candidate-run ${{BEST_STAGE1_RUN}} "
        f"--out {args.out}/evidence_figures"
    )


def train_stage2_command(
    args: argparse.Namespace, run_name: str, seed: int, stage1_run_name: str
) -> str:
    stage1_ckpt = stage1_checkpoint(stage1_run_name, seed)
    output_dir = stage2_output_dir(run_name)
    overrides = [
        f"--config-name={args.stage2_config}",
        f"model.local_path={stage1_ckpt}",
        f"all.name={run_name}",
        f"all.output_dir={output_dir}",
        f"++train.train_args.seed={seed}",
        f"++train.train_args.data_seed={seed}",
        f"train.train_args.max_steps={args.stage2_steps}",
        "++train.train_args.resume_from_checkpoint=false",
        f"dataset.src_data.mv_feature_cache={args.mv_feature_cache}",
    ]
    return (
        f"RUN_TS=seed{seed} {accelerate_launch_prefix(args)} "
        f"train.py {_quote_parts(overrides)}"
    )


def infer_stage2_command(args: argparse.Namespace, run_name: str, seed: int) -> str:
    return (
        f"RUN_TS=eval_seed{seed} {accelerate_launch_prefix(args)} "
        f"--module scripts.infer "
        f"--model-type edgerunner --run-type obj "
        f"--checkpoint {stage2_checkpoint(run_name, seed)} "
        f"--output-dir {stage2_infer_root(run_name, seed)} "
        f"--batch-size {args.stage2_infer_batch_size} "
        f"--seed {seed}"
    )


def eval_stage2_command(args: argparse.Namespace, run_name: str, seed: int) -> str:
    return (
        f"RUN_TS=eval_seed{seed} {accelerate_launch_prefix(args)} "
        f"--module scripts.eval_obj "
        f"--dataset {args.eval_obj_dataset} "
        f"--metadata {args.eval_obj_metadata} "
        f"--gt-dir {args.eval_obj_gt_dir} "
        f"--pred-dir {stage2_pred_dir(run_name, seed)} "
        f"--save-dir {stage2_eval_dir(seed)} "
        f"--num-sample-points {args.eval_obj_points} "
        f"--align-sample-points {args.eval_obj_align_sample_points} "
        f"--mask-area-thresh {args.eval_obj_mask_area_thresh} "
        f"--overwrite"
    )


def precompute_cache_command(args: argparse.Namespace, split: str) -> str:
    parts = [
        f"RUN_TS=precompute_{split}",
        accelerate_launch_prefix(args),
        "--module scripts.data.precompute_mv_features",
        f"--config-name={args.stage1_config}",
        f"--split={split}",
        f"--out={args.mv_feature_cache}",
        f"--mesh-dataset={args.mesh_dataset}",
        f"--hf-dataset={args.hf_dataset}",
    ]
    if args.da3_ckpt:
        parts.append(f"--da3-ckpt={args.da3_ckpt}")
    if args.cache_precompute_limit > 0:
        parts.append(f"--limit={args.cache_precompute_limit}")
    return " ".join(parts)


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
        "uid_metadata": args.uid_metadata,
        "mesh_dataset": args.mesh_dataset,
        "hf_dataset": args.hf_dataset,
        "mv_feature_cache": args.mv_feature_cache,
        "eval_obj_protocol": {
            "num_sample_points": args.eval_obj_points,
            "align_sample_points": args.eval_obj_align_sample_points,
            "mask_area_thresh": args.eval_obj_mask_area_thresh,
        },
        "cache_precompute_script": str(out_dir / "precompute_cache.sh"),
        "verifier_prompt_dir": str(out_dir / "verifier_prompts"),
        "ablations": [],
        "negative_controls": NEGATIVE_CONTROLS,
    }

    seed_args = " ".join(str(seed) for seed in args.seeds)
    uid_metadata_arg = (
        f"--uid-metadata {args.uid_metadata} " if args.uid_metadata else ""
    )
    uid_requirement_arg = "" if args.uid_metadata else "--no-require-uid-metadata "
    commands.append("# Preflight prerequisites before launching expensive ablations.")
    commands.append(
        f"{args.python} scripts/experiments/check_mv_layout_loss_readiness.py "
        f"--mesh-dataset {args.mesh_dataset} "
        f"--hf-dataset {args.hf_dataset} "
        f"--mv-feature-cache {args.mv_feature_cache} "
        f"--cache-precompute-script {out_dir / 'precompute_cache.sh'} "
        f"--sv-downstream {args.sv_downstream} "
        f"{uid_metadata_arg}"
        f"{uid_requirement_arg}"
        f"--seeds {seed_args} "
        f"--out {args.out}/readiness.json"
    )
    commands.append("")

    precompute_commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Build the frozen DA3+DINO cache required by commands.sh readiness checks.",
        "# Rebuild if Trellis2 split/filtering, view selection, frame correction, DA3/DINO, or datasets change.",
        precompute_cache_command(args, "train"),
        precompute_cache_command(args, "val"),
        "",
    ]

    for label, exp_cfg in ABLATIONS.items():
        for seed in args.seeds:
            run_name = label
            commands.append(ensure_clean_command(run_name, seed))
            commands.append(train_stage1_command(args, run_name, exp_cfg, seed))
            commands.append(eval_command(args, run_name, seed))
            commands.append("")
            manifest["ablations"].append(
                {
                    "label": label,
                    "experiment": exp_cfg,
                    "seed": seed,
                    "run_name": run_name,
                }
            )

    commands.append(
        "# Negative eval controls for the selected best run across the seed list."
    )
    commands.append(
        ': "${BEST_STAGE1_RUN:?set BEST_STAGE1_RUN to the verified stage-1 run name}"'
    )
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
    commands.append(
        ': "${BEST_EXPERIMENT:?set BEST_EXPERIMENT to mv_layout_loss_ordinal/coord/geometry/etc.}"'
    )
    commands.append(
        ': "${BEST_STAGE1_RUN:?set BEST_STAGE1_RUN to the verified stage-1 run name}"'
    )
    for name, extra in NEGATIVE_CONTROLS.items():
        if name.endswith("_eval"):
            continue
        for seed in args.seeds:
            run_name = f"${{BEST_STAGE1_RUN}}_{name}"
            commands.append(ensure_clean_command(run_name, seed))
            commands.append(
                train_stage1_command(args, run_name, "${BEST_EXPERIMENT}", seed, extra)
            )
            commands.append(
                eval_command(args, run_name, seed, _as_eval_overrides(extra))
            )
            commands.append("")

    commands.append(
        "# Carry the best verified stage-1 loss into stage 2 after metrics pick BEST_STAGE1_RUN."
    )
    for seed in args.seeds:
        run_name = stage2_run_name(seed)
        commands.append(
            ensure_clean_command(
                run_name,
                seed,
                root=STAGE2_TRAIN_ROOT,
            )
        )
        commands.append(
            train_stage2_command(args, run_name, seed, "${BEST_STAGE1_RUN}")
        )
        commands.append(infer_stage2_command(args, run_name, seed))
        commands.append(eval_stage2_command(args, run_name, seed))
        commands.append("")

    commands.append("")
    commands.append(
        "# Summarize layout evidence and attach downstream CD/F across the E-stage seed list."
    )
    commands.append(f': "${{SV_DOWNSTREAM:={args.sv_downstream}}}"')
    run_args = " ".join(f"--run {name}" for name in ABLATIONS)
    downstream_args = " ".join(
        f"--downstream E_stage2_best={stage2_eval_results(seed)}" for seed in args.seeds
    )
    category_stability_arg = (
        "--require-category-stability " if args.uid_metadata else ""
    )
    best_layout_arg = (
        "--best-layout-run ${BEST_STAGE1_RUN} " if args.uid_metadata else ""
    )
    commands.append(
        f"{args.python} scripts/eval/summarize_mv_layout_evidence.py "
        f"--layout-root outputs/da3/eval/layout_mv "
        f"{run_args} "
        f"--seeds {seed_args} "
        f"--ce-run A_ce "
        f"{uid_metadata_arg}"
        f"--sv-downstream ${{SV_DOWNSTREAM}} "
        f"{downstream_args} "
        f"--out {args.out}/evidence_summary"
    )
    commands.append(build_figures_command(args, run_args, seed_args))
    commands.append(
        "# Re-render fixed visual evidence for ranked gallery UIDs, then rebuild galleries."
    )
    commands.append(
        f': "${{GALLERY_UIDS:={args.out}/evidence_figures/gallery_uids.txt}}"'
    )
    for seed in args.seeds:
        visual_args = [
            "--visual-uids",
            "${GALLERY_UIDS}",
            "--visual-selection",
            "all",
            "--max-visuals",
            "0",
        ]
        commands.append(eval_command(args, "A_ce", seed, visual_args))
        commands.append(eval_command(args, "${BEST_STAGE1_RUN}", seed, visual_args))
    commands.append(build_figures_command(args, run_args, seed_args))
    commands.append(
        f"{args.python} scripts/experiments/write_mv_layout_verifier_prompts.py "
        f"--out {args.out}/verifier_prompts "
        f"--verifier-dir {args.out}/verifiers "
        f"--evidence-root {args.out} "
        f"--layout-root outputs/da3/eval/layout_mv "
        f"{run_args} "
        f"--seeds {seed_args} "
        f"--best-layout-run ${{BEST_STAGE1_RUN}} "
        f"--sv-downstream ${{SV_DOWNSTREAM}} "
        f"{downstream_args} "
        f"{uid_metadata_arg}"
    )
    commands.append("")
    commands.append(
        "# Write the deterministic council review after negative controls exist."
    )
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
        f"{category_stability_arg}"
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
        f"{downstream_args} "
        f"--verifier-dir {args.out}/verifiers "
        f"--figure-dir {args.out}/evidence_figures "
        f"--summary-json {args.out}/evidence_summary/summary.json "
        f"{best_layout_arg}"
        f"{category_stability_arg}"
        f"--require-visuals "
        f"--require-figures "
        f"--out {args.out}/evidence_bundle_check.json"
    )

    (out_dir / "commands.sh").write_text("\n".join(commands) + "\n")
    (out_dir / "precompute_cache.sh").write_text("\n".join(precompute_commands) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_dir / 'commands.sh'}")
    print(f"Wrote {out_dir / 'precompute_cache.sh'}")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
