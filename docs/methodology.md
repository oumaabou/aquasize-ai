# Methodology & Alignment to Project Brief

## Mapping to the stated deliverables

The KAUST visiting-student brief lists these deliverables:

| Brief deliverable | This project |
|---|---|
| "Develop a model that can identify and size species" | Phase 1 (species ID) + Phase 2 (sizing), built as one pipeline where Phase 2 reuses Phase 1's detection/segmentation output |
| "Pilot a field experiment that tests how different image settings influence performance" | `stress_test_augmentations.py` + `robustness_benchmark.py` — 8 named conditions reflecting realistic **catch-photo** variability (not underwater conditions), each benchmarked independently |
| "Review relevant scientific literature" | `docs/literature_review.md` |
| "Collaborate to develop data visualizations, publications, or analytical tools" | Confusion matrix, robustness-by-condition bar chart, predicted-vs-true scatter, Bland-Altman plot — all reproducible from `evaluate_metrics.py` / `sizing.py` |
| "Interpret results within a sustainability and food systems context" | See literature review Part 1 — length/size-structure data feeds directly into the stock-assessment and yield models used in the PI's own published work |
| "Reproducible research practices" | Pinned `requirements.txt`, deterministic seeds throughout, a local smoke test that runs with no external dependencies, clear Colab handoff for the GPU-dependent steps |

## Why "catch images," not underwater imagery

The brief specifically says catch images. A generic marine-CV demo project
defaults to underwater conditions (turbidity, blue/green color cast) — the
wrong domain for a photo of a fish already landed on a boat deck or
measuring board. This project's stress-test conditions were chosen instead
to reflect actual catch-photo variability: harsh sun/shade, low light on a
deck or at night, non-perpendicular phone-camera angles, motion blur from
a moving boat, background clutter (nets, crates, hands), and
compression/downscaling from mobile data transfer. This isn't a stylistic
choice — it's the difference between a project that superficially uses
the right buzzwords and one that actually engaged with what the photos in
this study will look like.

## The statistical validation, explicitly

The brief calls out a preference for students with a background in
**statistics**. Beyond training a model and reporting accuracy, this
project's contribution is treating "does the sizing estimate agree with
reality" as a measurement-agreement problem, not a single accuracy number:

- **MAE / RMSE / MAPE** — standard error magnitude metrics.
- **Pearson r / R²** — linear association between predicted and true length.
- **Bland-Altman bias + 95% limits of agreement** — the standard method in
  biometrics/clinical measurement literature for reporting whether a new
  measurement method is interchangeable with a reference method, rather
  than just "correlated" with it (correlation can be high even when a
  method has a consistent, unacceptable bias).

This mirrors how the reference papers in `docs/literature_review.md`
report their own accuracy (e.g., RMSD against reference length
histograms), so results from this pipeline would be directly comparable to
published numbers.

## Reproducibility practices followed

- All randomness is seeded (`numpy.random.default_rng(seed)`,
  Albumentations `seed=` argument) so results are exactly repeatable.
- Dependencies are version-pinned in `requirements.txt`.
- The pipeline is split so the parts that don't need a GPU or a dataset
  (augmentation logic, sizing math, statistics) can be verified completely
  independently of the parts that do (model training) — this is what
  `demo/smoke_test.py` demonstrates.
- Every module has a `if __name__ == "__main__":` self-test so correctness
  can be checked in isolation before wiring modules together.

## Honest scope statement for the application essay

This repository is presented as a working, validated pipeline design and
proof-of-concept — not a finished, peer-reviewed model. The honest claim
to make in an application essay is: *"I designed and implemented an
end-to-end pipeline for reef fish species identification and length
estimation from catch photos, including a statistically rigorous
validation methodology, and tested the full non-training pipeline
end-to-end. Training on a real Red Sea dataset and full field validation
is the next step I'd want to take under your supervision."* That is a
stronger and more credible statement than implying a fully trained,
field-tested system already exists.
