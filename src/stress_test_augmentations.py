"""
stress_test_augmentations.py

Simulates the realistic variability found in CATCH images (fish landed on a
boat deck, dock, or measuring board) rather than underwater dive photos.

Why this matters: the KAUST project brief asks the model to be evaluated
under "different image settings" for CATCH images specifically. Underwater
turbidity/blue-green tint (a common default choice in generic marine-CV
demos) is the wrong domain here -- a catch photo is not taken underwater.
The realistic sources of variation for landed-fish photos are:

  1. Lighting     -- harsh sun, overcast, shade, deck lighting at night
  2. Camera angle  -- phone tilt, non-perpendicular shots, oblique angles
  3. Distance/perspective -- fish not flat/parallel to the camera plane
  4. Background clutter -- nets, crates, other fish, hands, deck texture
  5. Motion blur   -- handheld phone photos on a moving boat
  6. Compression   -- photos sent over WhatsApp/low-bandwidth mobile data

Each condition is its own named Albumentations pipeline so you can
benchmark model/measurement accuracy degradation condition-by-condition
(see evaluate_metrics.py) instead of one blended "hard" transform.
"""

from __future__ import annotations

import albumentations as A
import numpy as np

# ---------------------------------------------------------------------------
# Named, realistic catch-image stress conditions
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, A.Compose] = {
    "baseline": A.Compose([]),  # no augmentation -- control group

    "harsh_sunlight": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.25, 0.45),
                                    contrast_limit=(0.15, 0.35), p=1.0),
        A.RandomShadow(shadow_roi=(0, 0.4, 1, 1), num_shadows_limit=(1, 2), p=0.6),
    ]),

    "low_light_deck": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3),
                                    contrast_limit=(-0.2, 0.0), p=1.0),
        A.GaussNoise(std_range=(0.04, 0.1), p=0.5),
    ]),

    "oblique_angle": A.Compose([
        A.Perspective(scale=(0.08, 0.18), p=1.0),
        A.Rotate(limit=25, p=0.8),
    ]),

    "handheld_motion_blur": A.Compose([
        A.MotionBlur(blur_limit=(7, 15), p=1.0),
    ]),

    "background_clutter": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), p=0.5),
        A.CoarseDropout(num_holes_range=(2, 5), hole_height_range=(0.05, 0.12),
                         hole_width_range=(0.05, 0.12), fill=0, p=0.7),
    ]),

    "mobile_compression": A.Compose([
        A.ImageCompression(quality_range=(25, 45), p=1.0),
        A.Downscale(scale_range=(0.4, 0.6), p=0.6),
    ]),

    "combined_worst_case": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.35),
                                    contrast_limit=(-0.2, 0.2), p=0.9),
        A.Perspective(scale=(0.05, 0.12), p=0.7),
        A.MotionBlur(blur_limit=(5, 11), p=0.5),
        A.ImageCompression(quality_range=(30, 55), p=0.7),
    ]),
}


def apply_condition(image: np.ndarray, condition: str, seed: int | None = None) -> np.ndarray:
    """Apply one named stress condition to an RGB image (H, W, 3) uint8 array."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Options: {list(CONDITIONS)}")
    pipeline = CONDITIONS[condition]
    if seed is not None:
        result = A.Compose(pipeline.transforms, seed=seed)(image=image)
    else:
        result = pipeline(image=image)
    return result["image"]


def list_conditions() -> list[str]:
    return list(CONDITIONS.keys())


if __name__ == "__main__":
    print("Registered catch-image stress conditions:")
    for name in list_conditions():
        print(f"  - {name}")
