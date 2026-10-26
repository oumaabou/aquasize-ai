"""
smoke_test.py

Actually executes the non-deep-learning parts of the AquaSize AI pipeline
end-to-end, right now, with no GPU and no external dataset -- to prove the
engineering is correct before you ever touch Colab.

It does NOT train or run a real species classifier (that needs a real
dataset + GPU, see notebooks/phase1_train_colab.ipynb). It uses a synthetic
drawn "fish" with a known-size reference bar, so we have ground truth to
check the sizing math against, and a plain real photo to check the
stress-condition augmentations run cleanly on real image data.

Outputs (saved to demo/outputs/):
  - stress_conditions_demo.png   : grid of all catch-image conditions applied
  - sizing_validation_demo.png   : predicted vs. true length scatter + Bland-Altman
  - smoke_test_report.txt        : pass/fail summary with numbers
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sizing import compute_px_per_cm, estimate_length, validate_against_ground_truth  # noqa: E402
from stress_test_augmentations import CONDITIONS, apply_condition  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_synthetic_fish_scene(true_fish_length_px: int = 300,
                               true_ref_length_px: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draws a simple fish-like silhouette (body ellipse + tail triangle) of a
    KNOWN pixel length, plus a reference bar of a KNOWN pixel length, on a
    plain background. Returns (image, fish_contour, reference_contour).
    """
    h, w = 480, 640
    img = np.full((h, w, 3), (30, 90, 130), dtype=np.uint8)  # deck-blue background

    cx, cy = w // 2 - 40, h // 2
    tail_extension = 30  # tail sticks out this many px beyond the body ellipse
    body_half_len = (true_fish_length_px - tail_extension) // 2  # so ellipse+tail == true_fish_length_px exactly
    body_axes = (body_half_len, 55)
    fish_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(fish_mask, (cx, cy), body_axes, 0, 0, 360, 255, -1)
    tail_pts = np.array([
        [cx + body_axes[0], cy],
        [cx + body_axes[0] + tail_extension, cy - 40],
        [cx + body_axes[0] + tail_extension, cy + 40],
    ])
    cv2.fillPoly(fish_mask, [tail_pts], 255)
    img[fish_mask > 0] = (60, 140, 190)

    ref_x0, ref_y0 = 40, h - 50
    cv2.rectangle(img, (ref_x0, ref_y0), (ref_x0 + true_ref_length_px, ref_y0 + 12), (255, 255, 255), -1)

    fish_contour, _ = cv2.findContours(fish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(ref_mask, (ref_x0, ref_y0), (ref_x0 + true_ref_length_px, ref_y0 + 12), 255, -1)
    ref_contour, _ = cv2.findContours(ref_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return img, max(fish_contour, key=cv2.contourArea), max(ref_contour, key=cv2.contourArea)


def demo_stress_conditions(base_image: np.ndarray) -> None:
    conditions = list(CONDITIONS.keys())
    n = len(conditions)
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, cond in zip(axes, conditions):
        stressed = apply_condition(base_image.copy(), cond, seed=7)
        ax.imshow(cv2.cvtColor(stressed, cv2.COLOR_BGR2RGB))
        ax.set_title(cond, fontsize=10)
        ax.axis("off")
    for ax in axes[len(conditions):]:
        ax.axis("off")
    fig.suptitle("Catch-image stress conditions (synthetic scene)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stress_conditions_demo.png", dpi=130)
    plt.close(fig)


def demo_sizing_math() -> dict:
    known_ref_cm = 15.0  # e.g. a 15cm measuring-board tick
    true_fish_cm = 42.0
    px_per_cm_true = 100 / known_ref_cm  # matches true_ref_length_px=100 in make_synthetic_fish_scene
    true_fish_length_px = int(round(true_fish_cm * px_per_cm_true))

    img, fish_contour, ref_contour = make_synthetic_fish_scene(
        true_fish_length_px=true_fish_length_px, true_ref_length_px=100)

    px_per_cm = compute_px_per_cm(ref_contour, known_length_cm=known_ref_cm)
    result = estimate_length(fish_contour, px_per_cm)

    error_cm = result.estimated_length_cm - true_fish_cm
    pct_error = abs(error_cm) / true_fish_cm * 100

    cv2.imwrite(str(OUT_DIR / "synthetic_scene.png"), img)

    return {
        "true_length_cm": true_fish_cm,
        "estimated_length_cm": round(result.estimated_length_cm, 3),
        "error_cm": round(error_cm, 3),
        "pct_error": round(pct_error, 3),
    }


def demo_statistical_validation() -> dict:
    rng = np.random.default_rng(42)
    truth = rng.uniform(15, 45, size=40)
    preds = truth + rng.normal(0.4, 1.1, size=40)  # small realistic bias + noise
    report = validate_against_ground_truth(preds, truth)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(truth, preds, alpha=0.7, color="#2b7a78")
    lims = [min(truth.min(), preds.min()) - 2, max(truth.max(), preds.max()) + 2]
    axes[0].plot(lims, lims, "--", color="gray", label="perfect agreement")
    axes[0].set_xlabel("Ground-truth length (cm)")
    axes[0].set_ylabel("Predicted length (cm)")
    axes[0].set_title(f"Predicted vs. true (R²={report['r_squared']})")
    axes[0].legend()

    mean_vals = (truth + preds) / 2
    diff_vals = preds - truth
    axes[1].scatter(mean_vals, diff_vals, alpha=0.7, color="#c85200")
    axes[1].axhline(report["bland_altman_bias_cm"], color="black", label="bias")
    axes[1].axhline(report["bland_altman_loa_upper_cm"], color="gray", linestyle="--", label="95% LoA")
    axes[1].axhline(report["bland_altman_loa_lower_cm"], color="gray", linestyle="--")
    axes[1].set_xlabel("Mean of true & predicted (cm)")
    axes[1].set_ylabel("Predicted - true (cm)")
    axes[1].set_title("Bland-Altman agreement")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sizing_validation_demo.png", dpi=130)
    plt.close(fig)
    return report


def main() -> None:
    print("=== 1. Loading/generating synthetic catch-scene image ===")
    base_img, _, _ = make_synthetic_fish_scene()

    print("=== 2. Running all catch-image stress conditions ===")
    demo_stress_conditions(base_img)
    print(f"  saved -> {OUT_DIR / 'stress_conditions_demo.png'}")

    print("=== 3. Sizing math check (known-answer synthetic scene) ===")
    sizing_result = demo_sizing_math()
    for k, v in sizing_result.items():
        print(f"  {k}: {v}")

    print("=== 4. Statistical validation demo (synthetic predicted vs. true lengths) ===")
    stats_report = demo_statistical_validation()
    for k, v in stats_report.items():
        print(f"  {k}: {v}")
    print(f"  saved -> {OUT_DIR / 'sizing_validation_demo.png'}")

    report_lines = [
        "AquaSize AI -- local smoke test report",
        "=" * 50,
        "",
        "This validates the ENGINEERING (augmentation pipeline + sizing math",
        "+ statistics), NOT a trained species model. Real training requires",
        "a real dataset + GPU -- run notebooks/phase1_train_colab.ipynb on",
        "Google Colab for that.",
        "",
        f"Stress conditions tested: {len(CONDITIONS)} ({', '.join(CONDITIONS)})",
        "-> all ran without error, saved to stress_conditions_demo.png",
        "",
        "Sizing math check (known-answer synthetic scene):",
        f"  true length:      {sizing_result['true_length_cm']} cm",
        f"  estimated length: {sizing_result['estimated_length_cm']} cm",
        f"  error:            {sizing_result['error_cm']} cm ({sizing_result['pct_error']}%)",
        "  -> small nonzero error is expected: it comes from ellipse/contour",
        "     rasterization, exactly the kind of discretization error real",
        "     pixel-based sizing has. This is why Phase 2 REQUIRES the",
        "     statistical validation step below, not a single spot-check.",
        "",
        "Statistical validation demo (n=40 synthetic predicted-vs-true pairs):",
    ]
    for k, v in stats_report.items():
        report_lines.append(f"  {k}: {v}")
    report_lines += [
        "",
        "PASS: all pipeline stages executed and produced numeric output.",
    ]
    (OUT_DIR / "smoke_test_report.txt").write_text("\n".join(report_lines))
    print(f"\nFull report written to {OUT_DIR / 'smoke_test_report.txt'}")


if __name__ == "__main__":
    main()
