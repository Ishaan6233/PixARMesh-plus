import json
import subprocess
import sys


def test_mv_layout_verifier_prompt_writer_emits_role_prompts(tmp_path):
    out = tmp_path / "prompts"
    verifier_dir = tmp_path / "verifiers"

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/write_mv_layout_verifier_prompts.py",
            "--out",
            str(out),
            "--verifier-dir",
            str(verifier_dir),
            "--run",
            "A_ce",
            "--run",
            "B_ordinal",
            "--seeds",
            "11",
            "23",
            "--best-layout-run",
            "B_ordinal",
            "--downstream",
            "E_stage2_best=outputs/da3/eval/stage2_best/seed11/eval_obj_results.jsonl",
            "--downstream",
            "E_stage2_best=outputs/da3/eval/stage2_best/seed23/eval_obj_results.jsonl",
        ],
        check=True,
    )

    manifest = json.loads((out / "verifier_prompt_manifest.json").read_text())
    loss_prompt = (out / "loss_verifier_prompt.md").read_text()
    visual_prompt = (out / "visual_verifier_prompt.md").read_text()

    assert manifest["required_findings"] == [
        "loss_verifier.md",
        "data_verifier.md",
        "experiment_verifier.md",
        "visual_verifier.md",
        "council_review.md",
    ]
    assert manifest["runs"] == ["A_ce", "B_ordinal"]
    assert manifest["seeds"] == [11, 23]
    assert manifest["uid_metadata"] == ""
    assert "not verifier findings" in manifest["note"]
    assert f"`{verifier_dir / 'loss_verifier.md'}`" in loss_prompt
    assert "src/models/loss.py" in loss_prompt
    assert "gradient" in loss_prompt
    assert "status: pass" in loss_prompt
    assert "Only use `status: pass`" in loss_prompt
    assert "gallery_manifest.json" in visual_prompt
    assert "conditioning.json" in visual_prompt
    assert (out / "README.md").exists()
