import argparse
import glob
import json
import os
import sys


def load_obj_summary(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"[warn] obj-eval file not found: {jsonl_path}", file=sys.stderr)
        return None
    with open(jsonl_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        print(f"[warn] obj-eval file is empty: {jsonl_path}", file=sys.stderr)
        return None
    return json.loads(lines[-1])


def load_scene_summary(scene_dir):
    if not scene_dir or not os.path.exists(scene_dir):
        return None
    files = glob.glob(os.path.join(scene_dir, "*.json"))
    if not files:
        return None
    records = []
    for f in files:
        try:
            records.append(json.load(open(f)))
        except Exception:
            pass
    if not records:
        return None
    keys = ["cd", "cd_s", "f_score", "f_score_2"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in records if k in r]
        summary[f"avg_{k}"] = sum(vals) / len(vals) if vals else None
    summary["num_evaluated"] = len(records)
    return summary


def fmt_cd(v):
    if v is None:
        return "—"
    return f"{v * 1000:.4f}"


def fmt_fs(v):
    if v is None:
        return "—"
    return f"{v:.2f}%"


def build_tables(models):
    """models: list of (name, obj_summary, scene_summary)"""
    sections = []

    obj_rows = [(name, obj) for name, obj, _ in models if obj is not None]
    if obj_rows:
        header = "| Model | CD ↓ (×10⁻³) | F-Score τ=0.002 ↑ | N |"
        sep    = "|-------|--------------|-------------------|---|"
        rows = [
            f"| {name} "
            f"| {fmt_cd(obj.get('avg_cd'))} "
            f"| {fmt_fs(obj.get('avg_f_score'))} "
            f"| {obj.get('num_evaluated', '—')} |"
            for name, obj in obj_rows
        ]
        sections.append("## Obj-level results\n\n" + "\n".join([header, sep] + rows))

    scene_rows = [(name, sc) for name, _, sc in models if sc is not None]
    if scene_rows:
        header = "| Model | CD ↓ (×10⁻³) | CD-S ↓ (×10⁻³) | F-Score τ=0.002 ↑ | F-Score τ=0.1 ↑ | N |"
        sep    = "|-------|--------------|-----------------|-------------------|-----------------|---|"
        rows = [
            f"| {name} "
            f"| {fmt_cd(sc.get('avg_cd'))} "
            f"| {fmt_cd(sc.get('avg_cd_s'))} "
            f"| {fmt_fs(sc.get('avg_f_score'))} "
            f"| {fmt_fs(sc.get('avg_f_score_2'))} "
            f"| {sc.get('num_evaluated', '—')} |"
            for name, sc in scene_rows
        ]
        sections.append("## Scene-level results\n\n" + "\n".join([header, sep] + rows))

    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Build a results table from eval outputs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        nargs=3,
        metavar=("NAME", "OBJ_JSONL", "SCENE_DIR"),
        action="append",
        dest="models",
        help=(
            "One row in the table. Repeat for multiple models.\n"
            "  NAME       display name\n"
            "  OBJ_JSONL  path to eval_obj_results.jsonl  (use '' to skip)\n"
            "  SCENE_DIR  dir of per-scene uid.json files (use '' to skip)"
        ),
    )
    parser.add_argument("--out", default="results.md", help="Markdown output file")
    args = parser.parse_args()

    if not args.models:
        parser.error(
            "Specify at least one --model NAME OBJ_JSONL SCENE_DIR.\n"
            "Example:\n"
            "  --model 'baseline' outputs/evaluations-obj/baseline/eval_obj_results.jsonl ''"
        )

    model_data = []
    for name, obj_path, scene_dir in args.models:
        obj_summary = load_obj_summary(obj_path) if obj_path else None
        scene_summary = load_scene_summary(scene_dir) if scene_dir else None
        model_data.append((name, obj_summary, scene_summary))

    tables = build_tables(model_data)
    if not tables:
        print("No eval results found for any model.")
        return

    print(tables)
    with open(args.out, "w") as f:
        f.write(f"# PixARMesh+ Results\n\n{tables}\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
