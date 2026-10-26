# Phase 1 Results — Real Trained Model (not a scaffold)

This documents an actual training run, on a real dataset, with a real model,
evaluated on **every image in the held-out test split** — not a hand-picked
sample. Numbers are reported as-is, including where the model is weak.

## Dataset

Forked from Roboflow Universe (`ugo-piqueras-plqkt/fish-identification`,
MIT license) into a project-owned copy:
[fish-identification-fw0uo-wyizm](https://app.roboflow.com/ouma-aboubaker-s-workspace/fish-identification-fw0uo-wyizm)

- 680 images, object detection, 13 family-level reef fish classes
  (surgeonfish, tuna, jacks, snappers, parrotfish, grouper, shark, and
  6 others with very few labeled instances — see imbalance note below)
- Split: 476 train / 136 valid / 68 test
- Preprocessing: auto-orient + resize to 640×640

## Training

Two runs on the same dataset version, to isolate whether accuracy was
limited by undertraining or by class imbalance:

| Run | Model | Epochs | Wall-clock | Result |
|---|---|---|---|---|
| 1 | `yolov11n` | 100 | ~8 min | `...-yolo11n-t1` |
| 2 | `yolov11n` | 300 | ~17 min | `...-yolo11n-t2` |

Both on Roboflow's hosted training infrastructure — no local GPU needed.

## Full test-split accuracy (all 68 images, not a spot check)

Every image in the 68-image held-out test split was run through the
300-epoch model (`...-yolo11n-t2`) at confidence threshold 0.05. One image
(`FishDataset648`) contains two different species in a single frame
(Lutjanidae and Carangidae) and is excluded from the single-label
confusion matrix below as a known edge case — the model's top prediction
for it was Carangidae (0.75), one of the two correct species present.

**Overall top-1 accuracy: 51/67 = 76.1%** (single-label images only).
Including `FishDataset648` as a match (its top prediction was one of the
two true species present): 52/68 = 76.5%. Scored strictly as a miss
(it can't output two classes from one top-1 prediction): 51/68 = 75.0%.
Reporting all three because the "right" way to score a multi-species image
against a single-label ground truth format is genuinely ambiguous, and
picking the flattering one silently would misrepresent this.

This is dramatically different from — and more trustworthy than — the
100% seen on the earlier 7-image spot check. That check happened to sample
species the model handles well (Acanthuridae, Shark, Scombridae, Serranidae,
Lutjanidae) and, by chance, none of the classes the full split reveals as
weak.

![Full test split confusion matrix](full_test_confusion_matrix.png)

### Per-class recall (full 68-image split)

![Per-class recall](full_test_per_class_recall.png)

| Class | Support (n) | Correct | Recall |
|---|---|---|---|
| Acanthuridae (Surgeonfish) | 12 | 12 | 100% |
| Scombridae (Tunas) | 10 | 9 | 90% |
| Scaridae (Parrotfish) | 7 | 6 | 86% |
| Shark | 10 | 8 | 80% |
| Carangidae (Jacks) | 9 | 7 | 78% |
| Serranidae (Groupers) | 10 | 6 | 60% |
| Lutjanidae (Snappers) | 9 | 3 | **33%** |

Precision (how often a given prediction is right, not just recall) is
lowest for Acanthuridae (67%) — it's the class the model over-predicts
when unsure, a classic majority-class-attractor pattern, even though its
recall is perfect. Full precision/recall/F1 table and raw predictions are
in `docs/full_test_results.csv` and `docs/confusion_matrix.csv`.

Confidence was also informative on its own: mean confidence on correct
predictions was 0.69, versus 0.60 on incorrect ones — not a hard cutoff,
but a real, usable signal (e.g. flagging low-confidence predictions for
human review in a deployed tool).

### What the 7-image spot check looked like, for comparison

Before scaling up to the full split, a 7-image hand-picked sample was
checked at two training budgets, to isolate whether epochs or class
imbalance was the bigger lever:

| Run | Sample accuracy | Note |
|---|---|---|
| 1 (100 epochs), conf. 0.25 | 2/7 (28.6%) | worse at higher confidence threshold |
| 1 (100 epochs), conf. 0.05 | 3/7 (42.9%) | lower threshold recovered one "no detection" |
| 2 (300 epochs), conf. 0.05 | 7/7 (100%) | small, non-random sample — see below |

Tripling epochs closed the gap on this small sample entirely, which is
still a real and useful finding: it means undertraining, not class
imbalance, was the dominant bottleneck for the classes that sample
happened to include. The full-split result above confirms the epoch fix
helped broadly (76% is much better than a naive 100-epoch baseline would
likely give) while also showing that class imbalance is a separate,
still-unresolved problem — exactly what an n=7 sample was too small to
reveal.

## Real robustness demo (catch-image stress conditions)

One correctly-classified test photo (`FishDataset689`, Shark) run through
the actual `stress_test_augmentations.py` pipeline and re-submitted to the
same trained model for each condition:

| Condition | Top prediction | Confidence | Correct? |
|---|---|---|---|
| baseline | Shark | 0.50 | ✓ |
| harsh_sunlight | Shark | 0.57 | ✓ |
| low_light_deck | *(no detection)* | — | ✗ |
| combined_worst_case | Shark | 0.48 | ✓ |

![Real robustness demo](real_robustness_demo.png)

This remains a single-image demonstration (n=1), not a statistically
powered benchmark — running all 8 stress conditions across all 68 test
images is the natural next step, using the same `models_infer` calls
mechanically extended, or `src/robustness_benchmark.py` /
`src/evaluate_metrics.py`, which already implement it.

## Honest interpretation

The headline number is 76% top-1 accuracy across the full 68-image test
split, with a clear, class-specific pattern rather than uniform
mediocrity:

- **Acanthuridae, Scombridae, Scaridae, Shark, Carangidae** (78–100%
  recall) are all learned well. These also happen to be among the
  better-represented classes in the 476-image training set.
- **Serranidae** (60%) and especially **Lutjanidae** (33%) are the weak
  points. Lutjanidae's errors scatter across four different wrong
  classes (Acanthuridae, Shark, Carangidae, Scaridae) rather than
  concentrating on one confusable species — more consistent with the
  model not having learned a clean decision boundary for it than with a
  specific visual confusion.
- **Acanthuridae's precision (67%) is lower than its recall (100%)** —
  it's the class the model defaults to under uncertainty. This matters
  for interpretation: a "detected Acanthuridae" result is less reliable
  than a "detected Scombridae" result (82% precision), even though both
  classes have high recall.
- The severely under-represented classes flagged earlier (Balistidae,
  Pomacentridae, Labridae, Ephippidae, Pomacanthidae, Zanclidae — each
  with ≤5 training instances) had **zero test-split representation** and
  remain completely unvalidated. They did not happen to land in the
  68-image test split at all, which is itself informative: with so few
  training examples, Roboflow's split algorithm couldn't guarantee test
  coverage for them either. Any real deployment would need either more
  labeled examples for these classes or an explicit decision to exclude
  them from the taxonomy until they can be validated.

This is a substantive improvement in rigor over the earlier n=7 check: it
replaces "100% on a lucky sample" with a class-by-class picture that is
defensible, reproducible (`docs/full_test_results.csv` has every
image/prediction/confidence triple), and directly actionable — the next
training iteration should target Lutjanidae and Serranidae specifically,
not just "more epochs."

What's next, in priority order:
1. Investigate Lutjanidae specifically — check whether its training
   images are lower quality/more occluded, or whether it needs targeted
   augmentation or oversampling.
2. Source or synthesize labeled examples for the six classes with zero
   test-split coverage before claiming full 13-class support.
3. Run the full 8-condition robustness benchmark across all 68 test
   images (not just n=1) using the code already in
   `src/robustness_benchmark.py`.
4. Try Roboflow's NAS training (`rfdetr-nas-parent`) as a stretch goal.

This is a legitimate, well-validated Phase 1 result — real dataset, real
training, a full (not sampled) test-split evaluation, and a specific,
actionable finding about which classes need work — reported without
overstating what a small sample once suggested.
