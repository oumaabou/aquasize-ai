"""
robustness_benchmark.py -- Run on Colab after train.py produces best.pt

Ties together stress_test_augmentations.py + a trained model +
evaluate_metrics.py to produce the "Model Robustness Benchmark Report":
species-ID accuracy under each realistic catch-image condition, and (once
Phase 2 sizing is wired in) length-measurement error under each condition.

    python robustness_benchmark.py --weights runs/aquasize/phase1_species_detect/weights/best.pt \
        --test-dir data/processed/test --out docs/robustness_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from evaluate_metrics import robustness_benchmark
from stress_test_augmentations import apply_condition, list_conditions


def run(weights: str, test_dir: str, out_csv: str) -> None:
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(weights)
    test_dir = Path(test_dir)

    results_by_condition: dict[str, tuple[list[str], list[str]]] = {}

    for condition in list_conditions():
        y_true, y_pred = [], []
        for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            for img_path in class_dir.glob("*.*"):
                image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                stressed = apply_condition(image, condition, seed=42)
                pred = model.predict(stressed, verbose=False)[0]
                predicted_class = (pred.names[int(pred.probs.top1)]
                                   if pred.probs is not None
                                   else (pred.names[int(pred.boxes.cls[0])]
                                         if len(pred.boxes) else "no_detection"))
                y_true.append(class_dir.name)
                y_pred.append(predicted_class)
        results_by_condition[condition] = (y_true, y_pred)
        print(f"[{condition}] evaluated on {len(y_true)} images")

    report = robustness_benchmark(results_by_condition)
    report.to_csv(out_csv, index=False)
    print(f"\nSaved robustness report to {out_csv}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out", default="docs/robustness_report.csv")
    args = parser.parse_args()
    run(args.weights, args.test_dir, args.out)
