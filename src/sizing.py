"""
sizing.py -- Phase 2 (built on top of Phase 1's detection/segmentation output)

Converts a fish mask/contour + a reference object of known size into an
estimated real-world length, then VALIDATES that estimate statistically
against ground-truth measurements. The validation step is what turns this
from "an app that prints a number" into evidence of research-grade rigor
(directly answers the "background in statistics" line in the project
brief).

Known limitation, stated up front rather than glossed over: single-image
monocular length estimation is scale-ambiguous and biased by fish curvature
and non-parallel camera angle. Published fisheries work (e.g. ToroCam-style
pipelines, ICES J. Mar. Sci. 77(4):1330) handles this by requiring a
reference object in-frame and reporting error against ground truth rather
than claiming raw accuracy -- this module follows the same pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SizingResult:
    pixel_length: float
    px_per_cm: float
    estimated_length_cm: float


def compute_px_per_cm(reference_contour: np.ndarray, known_length_cm: float) -> float:
    """
    reference_contour: contour (Nx1x2 int array, from cv2.findContours) of a
        reference object of KNOWN physical length (e.g. a ruler, a measuring
        board tick mark, or a checkerboard square placed in-frame).
    known_length_cm: the real-world length of that reference object.
    """
    (_, _), (w, h), _ = cv2.minAreaRect(reference_contour)
    reference_px_length = max(w, h)
    if reference_px_length <= 0:
        raise ValueError("Reference object contour has zero length in pixels.")
    return reference_px_length / known_length_cm


def measure_fish_length_px(fish_contour: np.ndarray) -> float:
    """Longest axis of the fish's rotated bounding box (snout-to-tail proxy)."""
    (_, _), (w, h), _ = cv2.minAreaRect(fish_contour)
    return float(max(w, h))


def estimate_length(fish_contour: np.ndarray, px_per_cm: float) -> SizingResult:
    pixel_length = measure_fish_length_px(fish_contour)
    estimated_cm = pixel_length / px_per_cm
    return SizingResult(pixel_length=pixel_length, px_per_cm=px_per_cm,
                         estimated_length_cm=estimated_cm)


# ---------------------------------------------------------------------------
# Statistical validation against ground truth
# ---------------------------------------------------------------------------

def validate_against_ground_truth(predicted_cm: list[float] | np.ndarray,
                                   ground_truth_cm: list[float] | np.ndarray) -> dict:
    """
    Standard measurement-agreement statistics used in fisheries/biometrics
    papers when comparing an automated method to hand measurement:
      - MAE, RMSE, MAPE
      - Pearson r and R^2 (linear association)
      - Bland-Altman bias + limits of agreement (95%) -- the standard way to
        report whether a measurement method is interchangeable with a
        reference method, NOT just "correlated" with it.
    """
    y_pred = np.asarray(predicted_cm, dtype=float)
    y_true = np.asarray(ground_truth_cm, dtype=float)
    if y_pred.shape != y_true.shape:
        raise ValueError("predicted_cm and ground_truth_cm must be the same length")

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(abs_errors / y_true) * 100)

    r, p_value = stats.pearsonr(y_true, y_pred)
    r_squared = r ** 2

    bias = float(np.mean(errors))
    sd_diff = float(np.std(errors, ddof=1))
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff

    return {
        "n": len(y_true),
        "mae_cm": round(mae, 3),
        "rmse_cm": round(rmse, 3),
        "mape_pct": round(mape, 2),
        "pearson_r": round(float(r), 4),
        "r_squared": round(float(r_squared), 4),
        "pearson_p_value": p_value,
        "bland_altman_bias_cm": round(bias, 3),
        "bland_altman_loa_lower_cm": round(loa_lower, 3),
        "bland_altman_loa_upper_cm": round(loa_upper, 3),
    }


def validation_report_df(predicted_cm, ground_truth_cm) -> pd.DataFrame:
    """Per-fish error table, useful for spotting systematic bias (e.g. curled fish)."""
    y_pred = np.asarray(predicted_cm, dtype=float)
    y_true = np.asarray(ground_truth_cm, dtype=float)
    return pd.DataFrame({
        "ground_truth_cm": y_true,
        "predicted_cm": y_pred,
        "error_cm": y_pred - y_true,
        "abs_error_cm": np.abs(y_pred - y_true),
        "pct_error": np.abs(y_pred - y_true) / y_true * 100,
    })


if __name__ == "__main__":
    # Sanity check with synthetic data: predictions with a small known bias + noise
    rng = np.random.default_rng(42)
    truth = rng.uniform(15, 45, size=30)
    preds = truth + rng.normal(0.5, 1.2, size=30)  # +0.5cm bias, 1.2cm noise
    report = validate_against_ground_truth(preds, truth)
    for k, v in report.items():
        print(f"{k}: {v}")
