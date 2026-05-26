"""Canonical FAIR benchmark entrypoint.

Composes Hydra configs, creates governed run artifacts, captures environment
metadata, applies determinism controls, validates fairness specs, and dispatches
benchmark tasks.

Related files:
- Reads Hydra groups from `configs/`.
- Uses `utils/artifacts.py`, `utils/environment.py`, and `utils/determinism.py`
  for reproducible run setup.
- Validates contracts from `utils/specs.py`.
- Dispatches training through `trainers/canonical.py`, evaluation through
  `evaluators/canonical.py`, runtime timing through `runtime/benchmark.py`, and
  model loading through `models/registry.py`.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from evaluators import CanonicalEvaluator
from runtime import RuntimeBenchmarkConfig, benchmark_inference
from trainers import train_with_existing_stack
from utils.artifacts import compute_run_id, make_output_dir, save_json, save_resolved_config
from utils.determinism import setup_determinism
from utils.environment import capture_environment
from utils.specs import DatasetSpec, EvaluationSpec


def _with_overrides(cfg: DictConfig) -> DictConfig:
    try:
        overrides = HydraConfig.get().overrides.task
    except Exception:
        overrides = []
    with open_dict(cfg):
        cfg.hydra_overrides = list(overrides)
    return cfg


def _prepare_run(cfg: DictConfig) -> Path:
    OmegaConf.resolve(cfg)
    cfg = _with_overrides(cfg)
    repo_root = Path(get_original_cwd())
    base_dir = Path(str(cfg.output.base_dir))
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    output_dir = make_output_dir(
        str(base_dir),
        cfg.experiment.name,
        date=cfg.output.date,
    )
    env = capture_environment(repo_root)
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    run_id = compute_run_id(
        resolved_yaml,
        env.get("git_commit"),
        env.get("environment_hash", "unknown"),
    )
    with open_dict(cfg):
        cfg.run_id = run_id
    save_resolved_config(cfg, output_dir)
    save_json(output_dir / "environment" / "environment.json", env)
    save_json(
        output_dir / "environment" / "reproducibility.json",
        {
            "tier_1": "research reproducibility: fixed seeds, splits, configs, deps, deterministic evaluation",
            "tier_2": "benchmark standard: pinned CUDA/Torch/TorchVision/PyTorch3D/Triton/flash-attn and deterministic kernels",
            "tier_3": "bitwise reproducibility documented but not required across GPU classes",
            "run_id": run_id,
        },
    )
    return output_dir


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = _prepare_run(cfg)
    determinism = setup_determinism(int(cfg.seed), strict=bool(cfg.deterministic.strict))
    save_json(output_dir / "environment" / "determinism.json", determinism.__dict__)

    dataset_spec = DatasetSpec.from_mapping(
        OmegaConf.to_container(cfg.dataset.spec, resolve=True)
    )
    evaluation_spec = EvaluationSpec.from_mapping(
        OmegaConf.to_container(cfg.eval, resolve=True)
    )
    dataset_spec.validate()
    evaluation_spec.validate()

    task = str(cfg.task)
    if task == "validate":
        save_json(
            output_dir / "metrics" / "validation.json",
            {
                "status": "ok",
                "dataset_contract": "valid",
                "evaluation_contract": "valid",
                "model": cfg.model.name,
            },
        )
        return
    if task == "train":
        train_with_existing_stack(cfg, output_dir)
        return
    if task == "runtime":
        from models.registry import build_model_adapter

        adapter = build_model_adapter(cfg)
        runtime_cfg = RuntimeBenchmarkConfig(
            precision=str(cfg.runtime.precision),
            batch_size=int(cfg.runtime.batch_size),
            warmup_steps=int(cfg.runtime.warmup_steps),
            benchmark_steps=int(cfg.runtime.benchmark_steps),
            execution_modes=tuple(cfg.runtime.execution_mode),
        )
        result = benchmark_inference(adapter.infer, {}, runtime_cfg)
        save_json(output_dir / "runtime" / "runtime.json", result)
        return
    if task == "eval":
        evaluator = CanonicalEvaluator(evaluation_spec)
        save_json(output_dir / "metrics" / "evaluation_spec.json", evaluator.metadata())
        return
    raise ValueError(f"Unknown task {task!r}. Expected validate, train, eval, or runtime.")


if __name__ == "__main__":
    main()
