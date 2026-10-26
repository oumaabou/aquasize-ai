"""
evaluate_metrics.py

Phase 1 evaluation: species classification/detection accuracy, PLUS
robustness degradation across the catch-image stress conditions defined in
stress_test_augmentations.py. This is the module that produces the
"Model Robustness Benchmark Report" deliverable and the numbers that go in
the application essay / GitHub README -- not just a demo screenshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)


def classification_report_dict(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    per_class = pd.DataFrame({
        "class": labels, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    })
    return {"overall_accuracy": round(float(acc), 4), "per_class": per_class}


def confusion_matrix_df(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def robustness_benchmark(results_by_condition: dict[str, tuple[list[str], list[str]]]) -> pd.DataFrame:
    """
    results_by_condition: {condition_name: (y_true, y_pred)} -- run your
    model once per stress condition (see stress_test_augmentations.py) and
    collect predictions, then pass them here.

    Returns a table of accuracy per condition + the drop relative to
    'baseline', which is the number that actually answers the brief's
    "test how different image settings influence the performance of the
    model" deliverable.
    """
    rows = []
    baseline_acc = None
    for condition, (y_true, y_pred) in results_by_condition.items():
        acc = accuracy_score(y_true, y_pred)
        if condition == "baseline":
            baseline_acc = acc
        rows.append({"condition": condition, "accuracy": round(float(acc), 4)})

    df = pd.DataFrame(rows)
    if baseline_acc is not None:
        df["accuracy_drop_vs_baseline"] = round(baseline_acc, 4) - df["accuracy"]
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Smoke test with synthetic labels
    rng = np.random.default_rng(0)
    species = ["grouper", "snapper", "parrotfish", "emperor"]
    y_true = list(rng.choice(species, size=100))
    y_pred = [s if rng.random() > 0.15 else rng.choice(species) for s in y_true]
    report = classification_report_dict(y_true, y_pred)
    print("Overall accuracy:", report["overall_accuracy"])
    print(report["per_class"])
