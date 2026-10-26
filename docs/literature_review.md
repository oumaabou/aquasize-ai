# Literature Review — AquaSize AI

Short annotated bibliography supporting the project. Organized in two parts:
(1) why size/length data matters for reef fishery sustainability — grounding
in the PI's own research angle — and (2) how automated species ID + length
estimation from photos is actually done in the fisheries CV literature, so
the technical design choices in this project aren't arbitrary.

## Part 1 — Why this matters for reef fishery sustainability (the PI's angle)

**Zamborain-Mason, J. et al. (2025). "Potential yield and food provisioning
gains from rebuilding the world's coral reef fish stocks." *PNAS* 122(51),
e2508805122.**
Analyzes reef fish stock status worldwide and estimates that recovering
depleted stocks could raise sustainable yield by up to ~50%, with the
largest food-security gains concentrated in regions with the least
existing stock data. This is the direct motivation for building
lightweight, low-cost monitoring tools: yield/stock models are only as
good as the catch composition and size-structure data feeding them, and
that data is often missing for exactly the reefs that need it most.

**Zamborain-Mason, J. et al. (2025). "Downscaling global reference points
to assess the sustainability of local fisheries." *Conservation Biology*.**
Addresses the mismatch between global stock-assessment reference points
and local/small-scale fishery data — relevant because a Red Sea
catch-monitoring pipeline needs to produce size-structure data at a local
resolution that's usable in exactly this kind of downscaled assessment,
not just aggregate counts.

**KAUST Integrated Reef Fisheries Laboratory — Red Sea fisheries and
aquatic food systems research program.**
The lab's stated focus is compiling, harmonizing, and analyzing coral reef
fisheries and food-system data across the Red Sea — i.e., the
infrastructure problem (how do you get consistent species + size data from
many small, non-standardized catch photos) is as central as the modeling
problem. This shaped the choice to prioritize a reproducible, well-
documented pipeline over a flashy but one-off demo.

## Part 2 — Automated species ID and length estimation from catch photos

**Álvarez-Ellacuría, A. et al. (2020). "Image-based, unsupervised estimation
of fish size from commercial landings using deep learning." *ICES Journal
of Marine Science* 77(4), 1330–1339.**
Landing-site cameras + a reference object (fish box) in frame; deep
learning detects non-occluded fish and a cm/pixel ratio derived from the
known box size converts pixel length to real length. This is the direct
precedent for this project's "reference object + pixel ratio" sizing
approach (`src/sizing.py`) — and confirms the domain should be landing/
catch photos, not underwater imagery.

**García-d'Urso, N. et al. (2024). "Length estimation of fish detected as
non-occluded using a smartphone application and deep learning techniques."
*Fisheries Research* / bioRxiv 2023.03.12.532319.**
A smartphone-camera pipeline (the realistic capture device for a visiting
student's field pilot) for detecting non-occluded fish and estimating
length, explicitly restricting to non-occluded individuals because
overlapping fish bias the length estimate — informs why the sizing module
here should flag/exclude occluded detections rather than force a length
estimate on every box.

**Palconit, M. et al. (2021). "Absolute 3D Pose Estimation and Length
Measurement of Severely Deformed Fish from Monocular Videos in Longline
Fishing." *arXiv:2102.04639*.**
Fits a deformable 3D template to a 2D segmentation mask to correct for
fish curvature/bending, achieving ~7.3% RMSD against reference length
histograms even under deformation. Directly relevant to the known
limitation flagged in `sizing.py`: a straight-line pixel measurement
overestimates a curled fish's true length, and this is a documented,
quantifiable bias in the literature rather than a project-specific flaw.

**Wageeh, Y. et al. and related "ToroCam"-style landing-camera systems
(referenced via the ICES 2020 paper above).**
Established pattern of adding known-size reference coordinates directly
into the frame (rather than assuming camera calibration) specifically
because landing/dock conditions make consistent camera calibration
impractical — the basis for this project's reference-bar approach instead
of requiring stereo cameras or fixed rigs.

## What this means for project scope

The technical design in this repo (YOLOv8 segmentation → reference-object
calibration → statistical validation against ground truth) is not a novel
method — it deliberately follows the established pattern from the
landings/catch-photo literature above, rather than the underwater
stereo-video approaches used for live reef surveys (a different sub-field
with different hardware assumptions). The contribution of a Phase 1/Phase 2
project like this is demonstrating the pipeline works end-to-end on a
lightweight, low-cost setup (a phone camera + a reference bar) — which is
exactly the constraint real small-scale Red Sea fisheries monitoring is
likely to face.
