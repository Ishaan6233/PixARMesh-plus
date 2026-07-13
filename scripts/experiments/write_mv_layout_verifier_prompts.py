#!/usr/bin/env python3
"""Write independent-verifier prompt files for MV layout-loss evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.mv_layout_evidence_common import (
    DEFAULT_LAYOUT_RUNS,
    REQUIRED_VERIFIER_TERMS,
    stage2_eval_results,
)


PROMPTS = {
    "loss_verifier": {
        "finding": "loss_verifier.md",
        "title": "Loss Verifier",
        "question": (
            "Are the added stage-1 layout losses mathematically correct, masked to "
            "layout position tokens, differentiable where intended, logged separately, "
            "and default-off unless an experiment config enables them?"
        ),
        "required_checks": [
            "Audit src/models/loss.py and src/models/edgerunner.py.",
            "Verify 24-token layout reshape, position-token offsets, unsupported-layout skip behavior, and gradient flow.",
            "Check configs/experiment/mv_layout_loss_*.yaml and config defaults.",
            "Run or cite the focused layout-loss/config tests.",
        ],
        "suggested_commands": [
            "git diff -- src/models/loss.py src/models/edgerunner.py configs/experiment tests/test_layout_losses.py tests/test_configs.py",
            "/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python -m pytest tests/test_layout_losses.py tests/test_configs.py -q",
        ],
    },
    "data_verifier": {
        "finding": "data_verifier.md",
        "title": "Data Verifier",
        "question": (
            "Does the MV loader/collator/cache path provide the intended multi-view "
            "conditioning without unintended GT leakage beyond training labels?"
        ),
        "required_checks": [
            "Audit src/data/trellis2_mv.py and src/data/collator.py.",
            "Inspect readiness.json, manifest.json, and MV feature-cache key/version evidence.",
            "Verify view_mask/ref_view/view_indices handling and stale-cache rejection.",
            "Run or cite loader/readiness tests.",
        ],
        "suggested_commands": [
            "git diff -- src/data/trellis2_mv.py src/data/collator.py scripts/experiments/check_mv_layout_loss_readiness.py tests/test_trellis2_mv_loader.py tests/test_mv_layout_loss_readiness.py",
            "/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python -m pytest tests/test_trellis2_mv_loader.py tests/test_mv_layout_loss_readiness.py -q",
        ],
    },
    "experiment_verifier": {
        "finding": "experiment_verifier.md",
        "title": "Experiment Verifier",
        "question": (
            "Are A-D/E, negative controls, seeds, paired UIDs, downstream CD/F, and "
            "summary statistics extracted from the saved artifacts fairly and reproducibly?"
        ),
        "required_checks": [
            "Inspect summary.json/summary.md, every run report.json, per_sample.jsonl, and eval_obj_results.jsonl.",
            "Confirm exact seed coverage and no duplicate/missing downstream seed labels.",
            "Confirm CE pairing, object-level SV/MV downstream pairing, and category metadata coverage only when uid_metadata is supplied.",
            "Re-run summary and bundle-check commands where possible.",
        ],
        "suggested_commands": [
            "/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python -m pytest tests/test_mv_layout_evidence_summary.py tests/test_mv_layout_council_review.py tests/test_mv_layout_evidence_bundle.py -q",
            "PYTHONPATH=. /home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python scripts/eval/check_mv_layout_evidence_bundle.py --help",
        ],
    },
    "visual_verifier": {
        "finding": "visual_verifier.md",
        "title": "Visual Verifier",
        "question": (
            "Do the fixed, improved, regressed, and failure galleries correctly show "
            "GT/pred bbox projections, top-down boxes, conditioning points, valid views, "
            "reference view, and AABB evidence without cherry-picking?"
        ),
        "required_checks": [
            "Inspect gallery_manifest.json, ranked_uids.json, ranked_failures.json, and per-case conditioning.json.",
            "Open representative fixed/improved/regressed/failure composites and enabled-view projection images.",
            "Confirm projection metadata agrees with generated viewXX_projection.png files.",
            "Run or cite visual/evidence-figure tests.",
        ],
        "suggested_commands": [
            "/home/vision-ishaan/.local/share/mamba/envs/pixarmesh124/bin/python -m pytest tests/test_mv_layout_evidence_figures.py tests/test_eval_layout_mv.py -q",
            "find outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures -maxdepth 3 -type f | sort | sed -n '1,120p'",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/da3/experiments/mv_layout_loss_ablation/verifier_prompts",
    )
    parser.add_argument(
        "--verifier-dir",
        default="outputs/da3/experiments/mv_layout_loss_ablation/verifiers",
    )
    parser.add_argument(
        "--evidence-root", default="outputs/da3/experiments/mv_layout_loss_ablation"
    )
    parser.add_argument("--layout-root", default="outputs/da3/eval/layout_mv")
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--best-layout-run", default="${BEST_STAGE1_RUN}")
    parser.add_argument(
        "--sv-downstream", default="outputs/sv/eval/baseline/eval_obj_results.jsonl"
    )
    parser.add_argument("--downstream", action="append", default=[])
    parser.add_argument("--uid-metadata", default="")
    return parser.parse_args()


def bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def artifact_lines(args: argparse.Namespace) -> list[str]:
    runs = args.run or list(DEFAULT_LAYOUT_RUNS)
    downstream = args.downstream or [
        f"E_stage2_best={stage2_eval_results(seed)}" for seed in args.seeds
    ]
    return [
        f"layout_root: {args.layout_root}",
        f"evidence_root: {args.evidence_root}",
        f"summary_json: {args.evidence_root}/evidence_summary/summary.json",
        f"figure_dir: {args.evidence_root}/evidence_figures",
        f"verifier_dir: {args.verifier_dir}",
        f"runs: {', '.join(runs)}",
        f"seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"best_layout_run: {args.best_layout_run}",
        f"sv_downstream: {args.sv_downstream}",
        f"downstream: {', '.join(downstream)}",
        f"uid_metadata: {args.uid_metadata}",
    ]


def render_prompt(name: str, spec: dict[str, Any], args: argparse.Namespace) -> str:
    finding_path = Path(args.verifier_dir) / spec["finding"]
    required_terms = REQUIRED_VERIFIER_TERMS.get(spec["finding"], ())
    return dedent(
        f"""\
        # {spec["title"]}

        Write the final finding to `{finding_path}`. Do not edit source code or
        generated metrics while verifying. Disagreements block promotion until
        resolved with code, tests, or stronger evidence.

        ## Question

        {spec["question"]}

        ## Required Artifact Context

        {bullet(artifact_lines(args))}

        ## Required Checks

        {bullet(spec["required_checks"])}

        ## Suggested Commands

        {bullet(spec["suggested_commands"])}

        ## Required Finding Format

        Start the output file with exactly one of:

        ```text
        status: pass
        ```

        or:

        ```text
        status: fail
        ```

        Include sections named `Checked Commands`, `Artifacts Checked`,
        `Findings`, and `Open Issues`. Only use `status: pass` when every
        required check above was actually performed and no open issue remains.

        The evidence-bundle checker expects role-specific evidence terms in this
        finding, including: {", ".join(required_terms)}.
        """
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = {}
    for name, spec in PROMPTS.items():
        path = out_dir / f"{name}_prompt.md"
        path.write_text(render_prompt(name, spec, args))
        prompt_files[name] = str(path)

    manifest = {
        "prompt_dir": str(out_dir),
        "verifier_dir": args.verifier_dir,
        "required_findings": [spec["finding"] for spec in PROMPTS.values()]
        + ["council_review.md"],
        "prompt_files": prompt_files,
        "runs": args.run or list(DEFAULT_LAYOUT_RUNS),
        "seeds": args.seeds,
        "best_layout_run": args.best_layout_run,
        "sv_downstream": args.sv_downstream,
        "downstream": args.downstream,
        "uid_metadata": args.uid_metadata,
        "note": "Prompt files are not verifier findings and will not satisfy the evidence-bundle gate.",
    }
    (out_dir / "verifier_prompt_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (out_dir / "README.md").write_text(
        dedent(
            f"""\
            # MV Layout Verifier Prompts

            These files are prompts for independent reviewers. They are not
            verifier findings and they intentionally do not satisfy the evidence
            bundle gate.

            Real verifier findings must be written under `{args.verifier_dir}`
            with the exact filenames listed in `verifier_prompt_manifest.json`.
            """
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
