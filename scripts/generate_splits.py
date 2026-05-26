"""One-time generator for immutable canonical split manifests.

Run this script once in the canonical environment to create `splits/train.txt`,
`splits/val.txt`, and `splits/test.txt` from dataset UIDs. It refuses to
overwrite existing manifests.

Related files:
- `utils/splits.py` reads the generated manifests.
- `utils/canonical_dataset.py` uses those manifests during benchmark loading.
- `configs/dataset/canonical_3d_front.yaml` points to the split directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _write_split(dataset, split_name: str, out_dir: Path) -> None:
    out_path = out_dir / f"{split_name}.txt"
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Split manifests are immutable; "
            "delete intentionally before regenerating."
        )
    with out_path.open("w") as fp:
        for uid in dataset[split_name]["uid"]:
            fp.write(f"{uid}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate immutable benchmark splits.")
    parser.add_argument("--dataset", default="datasets/3d-front-ar-packed-flattened")
    parser.add_argument("--out-dir", default="splits")
    args = parser.parse_args()

    import datasets

    data = datasets.load_dataset(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        source_name = split_name
        if split_name == "val" and "val" not in data:
            source_name = "test"
        if source_name not in data:
            raise KeyError(f"Dataset has no {source_name!r} split.")
        _write_split({split_name: data[source_name]}, split_name, out_dir)


if __name__ == "__main__":
    main()
