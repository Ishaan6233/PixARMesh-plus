import os
import subprocess


def test_train_mv_wrapper_honors_gpus_and_np_for_stage1(tmp_path):
    log_path = tmp_path / "fake_python.log"
    stage1_root = tmp_path / "stage1"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\n' \"${CUDA_VISIBLE_DEVICES:-}\" >> \"$FAKE_PYTHON_LOG\"\n"
        "printf 'ARGS=%s\\n' \"$*\" >> \"$FAKE_PYTHON_LOG\"\n"
        "mkdir -p \"$S1_PREFIX/fake-run/checkpoints/final\"\n"
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "FAKE_PYTHON_LOG": str(log_path),
            "GPUS": "2,3",
            "NP": "2",
            "PYTHON": str(fake_python),
            "S1_PREFIX": str(stage1_root),
        }
    )
    result = subprocess.run(
        ["bash", "scripts/train_mv.sh", "--force-stage1", "--stage1-only"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    log_text = log_path.read_text()
    assert "CUDA_VISIBLE_DEVICES=2,3" in log_text
    assert "-m accelerate.commands.launch --num_processes 2 --gpu_ids 2,3 train.py" in log_text
    assert "--config-name=edgerunner_3d_front_trellis2_mv_stage1" in log_text
