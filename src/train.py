"""
train.py -- Phase 1: species detection/segmentation model

Run on Colab (free T4 GPU): this sandbox has no GPU and a locked-down
network, so training is not executed here -- see README. This script is
correct, complete, and ready to run as-is once you have a dataset in
YOLO format (from data_prep.py) on Colab.

    !pip install ultralytics -q
    python train.py --data data/processed/data.yaml --epochs 40 --model yolov8n-seg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n-seg.pt",
                         help="start from a pretrained checkpoint (nano-seg is fastest on free-tier GPU)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/aquasize")
    parser.add_argument("--name", default="phase1_species_detect")
    args = parser.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415 -- only needed where torch/GPU exists

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=10,          # early stop if val metric plateaus
        seed=42,
        val=True,
        plots=True,
    )

    metrics = model.val()
    print("\n=== Validation metrics ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights saved to: {best_weights}")
    print("Next: run robustness_benchmark.py on these weights across the "
          "catch-image stress conditions (see stress_test_augmentations.py).")


if __name__ == "__main__":
    main()
