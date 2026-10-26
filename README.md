# AquaSize AI — Reef Fish Species Detection & Automated Length Estimation

A two-phase computer vision pipeline for monitoring reef fisheries and
aquatic food systems from catch photos: **Phase 1** identifies fish
species and benchmarks how identification accuracy holds up under
realistic catch-image conditions (lighting, camera angle, blur,
compression, background clutter); **Phase 2** (built on top of Phase 1)
estimates real-world fish length from the same image and validates that
estimate statistically against known measurements.

Built to align with KAUST's Red Sea Research Center visiting-student
project on monitoring coral reef fisheries and aquatic food systems.

**Status: Phase 1 has been trained and evaluated on a real forked Roboflow
dataset (not just scaffolded), on the full held-out test split.** After
fixing an undertraining issue (100→300 epochs), the model reaches
**76% top-1 accuracy across all 68 test-split images** — ranging from
100% recall on the best-represented class (Acanthuridae) down to 33% on
the weakest (Lutjanidae). Full per-class breakdown, confusion matrix, and
what still needs work are in `docs/results.md`.

## Why two phases

Species identification is well-established and straightforward to
validate rigorously (accuracy, precision/recall, confusion matrices).
Length estimation from a single photo is a harder, more error-prone
problem (scale ambiguity, fish curvature, perspective distortion — see
`docs/literature_review.md`). Building and validating Phase 1 completely
first means there's always a solid, defensible deliverable, and Phase 2 is
an extension rather than a single point of failure. See `docs/methodology.md`
for the full reasoning.

## Repository structure

```
AquaSize_AI/
├── README.md
├── requirements.txt
├── notebooks/
│   └── phase1_train_colab.ipynb   # run on Google Colab (free GPU) — real training
├── src/
│   ├── data_prep.py               # dataset download + train/val/test split (Colab)
│   ├── stress_test_augmentations.py   # catch-image condition simulation (Albumentations)
│   ├── train.py                   # YOLOv8-seg training (Colab)
│   ├── evaluate_metrics.py        # accuracy, confusion matrix, robustness benchmark
│   ├── robustness_benchmark.py    # ties model + stress conditions + metrics together (Colab)
│   └── sizing.py                  # Phase 2: pixel→cm sizing + statistical validation
├── demo/
│   ├── smoke_test.py              # runs locally, no GPU/dataset needed — proves the
│   │                               # non-training parts of the pipeline are correct
│   └── outputs/                   # smoke test results (plots + report)
└── docs/
    ├── literature_review.md
    ├── methodology.md
    ├── results.md              # real training run: dataset, accuracy, robustness demo
    └── real_robustness_demo.png
```

## Quick start

**1. Local smoke test (no GPU, no dataset, ~10 seconds):**

```bash
pip install -r requirements.txt
python demo/smoke_test.py
```

This validates the augmentation pipeline and sizing math end-to-end on a
synthetic scene with a known-answer check, and produces:
- `demo/outputs/stress_conditions_demo.png` — all 8 catch-image conditions applied to one scene
- `demo/outputs/sizing_validation_demo.png` — predicted-vs-true scatter + Bland-Altman plot
- `demo/outputs/smoke_test_report.txt` — pass/fail summary with numbers

**2. Real training (Google Colab, free T4 GPU):**

Open `notebooks/phase1_train_colab.ipynb` in Colab, upload the `src/`
folder to the session, add a free Roboflow API key, and run top to bottom.
See the notebook for dataset suggestions.

## Current status / limitations (stated plainly)

- Phase 1 has been trained and evaluated for real: a public Roboflow
  dataset was forked into a project-owned copy and trained twice
  (`yolov11n`, 100 then 300 epochs, Roboflow's hosted training — no local
  GPU needed) to isolate whether undertraining or class imbalance was the
  limiting factor. It was mostly undertraining — but not entirely: the
  300-epoch model scores **76% top-1 accuracy across the full 68-image
  test split**, with strong performance on well-represented classes
  (100% on Acanthuridae, 90% on Scombridae) and weak performance on
  under-learned ones (33% on Lutjanidae, 60% on Serranidae). Six classes
  with ≤5 training examples had zero test-split representation and remain
  unvalidated. Full breakdown in `docs/results.md`.
- No Red-Sea-specific labeled dataset currently exists publicly; this run
  used a general reef-fish dataset as a proof of concept, with Red Sea taxa
  adaptation as a named next step, not something to gloss over.
- Single-image length estimation has a known, published bias from fish
  curvature and non-parallel camera angle (~3–7% error in the literature,
  see `docs/literature_review.md`). `sizing.py`'s statistical validation
  step exists specifically to measure and report this rather than assume
  it away.
- Phase 2 (sizing) is still a scaffold, not yet run against real
  measurements — that's the natural next step once Phase 1's class
  imbalance is fixed.

## Requirements

See `requirements.txt`. Local smoke test only needs `opencv-python-headless`,
`albumentations`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`.
`ultralytics` (and its `torch` dependency) is only needed for the Colab
training/inference steps.
