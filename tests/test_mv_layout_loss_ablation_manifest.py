import json
import subprocess
import sys


def test_mv_layout_loss_ablation_manifest_emits_runnable_seed_group_commands(tmp_path):
    out = tmp_path / "mv_layout_loss_ablation"

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/mv_layout_loss_ablation.py",
            "--out",
            str(out),
            "--seeds",
            "11",
            "23",
            "--stage1-steps",
            "1",
            "--stage2-steps",
            "1",
            "--eval-samples",
            "1",
        ],
        check=True,
    )

    commands = (out / "commands.sh").read_text()
    precompute = (out / "precompute_cache.sh").read_text()
    manifest = json.loads((out / "manifest.json").read_text())

    assert "PYTHONPATH=. ${PYTHON:-micromamba run -n pixarmesh124 python}" in commands
    assert "scripts/experiments/check_mv_layout_loss_readiness.py" in commands
    assert "scripts/experiments/write_mv_layout_verifier_prompts.py" in commands
    assert f"--cache-precompute-script {out / 'precompute_cache.sh'}" in commands
    assert "--uid-metadata" not in commands
    assert "--require-category-stability" not in commands
    assert "--no-require-uid-metadata" in commands
    assert f"--out {out}/verifier_prompts" in commands
    assert f"--verifier-dir {out}/verifiers" in commands
    assert f"--out {out}/readiness.json" in commands
    assert f": \"${{GALLERY_UIDS:={out}/evidence_figures/gallery_uids.txt}}\"" in commands
    assert "--visual-uids ${GALLERY_UIDS} --visual-selection all --max-visuals 0" in commands
    assert "scripts/eval/eval_layout_mv.py --checkpoint outputs/da3/train/mv_layout_loss/A_ce/seed11/checkpoints/final" in commands
    assert commands.count("--align-sample-points 5000") == 2
    assert (
        "scripts/eval/eval_layout_mv.py --checkpoint "
        "outputs/da3/train/mv_layout_loss/${BEST_STAGE1_RUN}/seed11/checkpoints/final"
        in commands
    )
    assert commands.count("scripts/eval/build_mv_layout_evidence_figures.py") == 2
    assert "scripts.data.precompute_mv_features" in precompute
    assert "--split=train" in precompute
    assert "--split=val" in precompute
    assert "--out=${MV_FEATURE_CACHE:-datasets/mv-feature-cache/da3/trellis2-mv}" in precompute
    assert "-m accelerate.commands.launch --num_processes ${NP:-4} --gpu_ids ${GPUS:-0,1,2,3}" in commands
    assert "-m accelerate.commands.launch --num_processes ${NP:-4} --gpu_ids ${GPUS:-0,1,2,3}" in precompute
    assert "BEST_SEED" not in commands
    assert "BEST_STAGE2_DOWNSTREAM" not in commands
    assert (
        "--downstream E_stage2_best=outputs/da3/eval/stage2_best/seed11/eval_obj_results.jsonl"
        in commands
    )
    assert (
        "--downstream E_stage2_best=outputs/da3/eval/stage2_best/seed23/eval_obj_results.jsonl"
        in commands
    )
    assert commands.count("--downstream E_stage2_best=") == 6
    assert manifest["seeds"] == [11, 23]
    assert manifest["cache_precompute_script"] == str(out / "precompute_cache.sh")
    assert manifest["verifier_prompt_dir"] == str(out / "verifier_prompts")
    assert manifest["uid_metadata"] == ""
    assert manifest["mv_feature_cache"] == "${MV_FEATURE_CACHE:-datasets/mv-feature-cache/da3/trellis2-mv}"
    assert manifest["eval_obj_protocol"] == {
        "num_sample_points": 10000,
        "align_sample_points": 5000,
        "mask_area_thresh": 1600,
    }
    assert len(manifest["ablations"]) == 8


def test_mv_layout_loss_ablation_manifest_can_enable_category_stability(tmp_path):
    out = tmp_path / "mv_layout_loss_ablation"

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/mv_layout_loss_ablation.py",
            "--out",
            str(out),
            "--seeds",
            "11",
            "--uid-metadata",
            "metadata/layout_uid_categories.jsonl",
            "--stage1-steps",
            "1",
            "--stage2-steps",
            "1",
        ],
        check=True,
    )

    commands = (out / "commands.sh").read_text()
    manifest = json.loads((out / "manifest.json").read_text())

    assert "--uid-metadata metadata/layout_uid_categories.jsonl" in commands
    assert "--no-require-uid-metadata" not in commands
    assert "--require-category-stability" in commands
    assert "--best-layout-run ${BEST_STAGE1_RUN}" in commands
    assert manifest["uid_metadata"] == "metadata/layout_uid_categories.jsonl"
