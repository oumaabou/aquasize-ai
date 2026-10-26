"""
data_prep.py

Dataset preparation for Phase 1 (species detection). Run this on Colab,
where you have full internet access to Roboflow Universe / Kaggle -- this
sandbox's network is restricted to PyPI and github.com, so dataset download
is intentionally NOT attempted here (see README's "Sandbox limitations"
section for why).

Usage (on Colab):
    from data_prep import download_roboflow_dataset, make_splits

    ds_path = download_roboflow_dataset(
        api_key="YOUR_ROBOFLOW_KEY",
        workspace="some-workspace",
        project="reef-fish-species",
        version=1,
    )
    make_splits(ds_path, train=0.7, val=0.2, test=0.1)

Suggested starting datasets (search these names on Roboflow Universe):
  - "Fish Species detection" / "Aquarium Combined" (COCO-format, multi-species)
  - "Reef Fish Identification" datasets tagged for the Indo-Pacific/Red Sea
  - If nothing Red-Sea-specific exists (likely), start with a general reef
    fish species set for the Phase 1 proof of concept, and note in your
    write-up that species-list adaptation to Red Sea taxa is future work --
    this is an honest, defensible framing for the application essay.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path


def download_roboflow_dataset(api_key: str, workspace: str, project: str,
                               version: int, out_dir: str = "data/raw") -> Path:
    """Requires `pip install roboflow` (Colab only -- not installed in this sandbox)."""
    from roboflow import Roboflow  # noqa: PLC0415

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        "yolov8", location=out_dir)
    return Path(ds.location)


def make_splits(dataset_dir: str | Path, train: float = 0.7, val: float = 0.2,
                 test: float = 0.1, seed: int = 42) -> None:
    """
    Re-splits a flat folder of class subfolders (data/raw/<species>/*.jpg)
    into data/processed/{train,val,test}/<species>/*.jpg for YOLO
    classification, or adapt the same split indices for detection labels.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "splits must sum to 1.0"
    dataset_dir = Path(dataset_dir)
    out_root = dataset_dir.parent / "processed"
    rng = random.Random(seed)

    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        images = sorted(class_dir.glob("*.*"))
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * train)
        n_val = int(n * val)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split_name, files in splits.items():
            split_dir = out_root / split_name / class_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, split_dir / f.name)

    print(f"Wrote train/val/test splits to {out_root}")


if __name__ == "__main__":
    print(__doc__)
