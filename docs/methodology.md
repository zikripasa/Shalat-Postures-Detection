# Methodology

## 1. Problem framing

The dataset annotates **4 visual postures** (single images, object-detection format). The product needs **6 prayer
phases in video**. Two pairs of phases are visually identical in a single frame (standing vs. iktidal; sitting between
prostrations vs. tasyahud), so the system is decomposed into:

1. a per-person, per-frame **posture classifier** (4 classes), and
2. a **temporal decoder** that uses the order of prayer movements to segment the sequence and name the phases.

Nothing claims to *see* iktidal or tasyahud; their names are inferred from context and labelled as such in the API
(`evidence: "visual + context"`).

## 2. Data handling

* Safe extraction (no absolute paths, no `..`, no symlinks, size/ratio limits, data file types only).
* Pascal VOC parsing incl. one `<polygon>` annotation (bounding rectangle). Unknown labels would be rejected; spelling
  variants are normalised (`sitting`/`julus`, `sujud`/`Sujud`).
* Labelled unit = **person instance** (box). Crops use the box + 15% padding on each side.
* **Grouping for leakage control:** union-find over (a) perceptual-hash near-duplicates (Hamming ≤ 18; threshold chosen
  from the observed gap between same-video frames, 0–18, and unrelated images, ≥ 20) and (b) capture sessions found by
  manual review of every image (`configs/session_groups.yaml`). 40 instances → 29 groups.

## 3. Features

* **Pose:** MediaPipe Pose Landmarker *full* (selected over *lite*/*heavy*: 85% vs 80%/82.5% detection on annotated
  crops at 34 ms vs 30/69 ms). From 33 landmarks: 3-D torso inclination, hip and knee angles, thigh and shin
  inclination (world coordinates, y down), 2-D torso inclination and knee angle, head/shoulder/wrist/knee/ankle heights
  relative to the hips (normalised by the landmark-box diagonal), horizontal spreads, landmark-box aspect, upper/lower
  visibility, and the person-box aspect. Landmarks with visibility < 0.35 are missing; imputation adds indicators.
  The model files are downloaded from version-pinned URLs and verified against the bundle's SHA-256.
* **Why not YOLO Pose** (the brief asked for it only if appropriate): it was considered and not used. This is a design
  decision, not a measured comparison. (1) The widely used Ultralytics YOLO-Pose code and weights are AGPL-3.0. Serving
  them in this web app would put the whole service under AGPL or require a commercial licence, while the project is MIT
  and its models are Apache-2.0/BSD. (2) MediaPipe already runs at about 34 ms per crop on CPU, and person detection is
  handled by EfficientDet-Lite0. (3) The real weakness is pose detection in prostration (67% found). With 12 annotated
  prostrations, a comparison between pose estimators could not separate them reliably. Comparing pose estimators on
  the full dataset is future work.
* **Image:** ImageNet MobileNetV3-Large (960-d) and ResNet18 (512-d) pooled features of the **letterboxed** crop
  (aspect ratio preserved — box shape is a posture cue). Training embeddings are doubled with horizontal flips.

## 4. Models compared (one protocol)

| Family | Models |
|---|---|
| Reference | majority class |
| Training-free | geometric rules with thresholds fixed a priori (knee 140°, torso 45°/55°) |
| Pose features | logistic regression, RBF-SVM, random forest, MLP (inner-CV grid search) |
| Frozen CNN ("linear probe") | MobileNetV3-Large / ResNet18 embedding + logistic regression |
| Fine-tuned CNN | MobileNetV3-Large / ResNet18, AdamW (backbone 1e-4, head 1e-3), cosine schedule, label smoothing 0.05, class weights, frozen BatchNorm, augmentation (flip, zoom, ±8° rotation, colour), early stopping on an inner group-wise validation split (patience 8), checkpoints |
| Fusion | mean of pose logistic regression + MobileNetV3 probe — **pre-registered** before results |

## 5. Evaluation protocol

* 3 × repeated **stratified group 5-fold** CV over instances (`sklearn.model_selection.StratifiedGroupKFold`), so no
  group is in train and test simultaneously; hyper-parameters selected by an inner stratified group 3-fold on the
  training fold only.
* Two test views: **annotated crops** (classification in isolation) and **whole images** through the real
  localisation chain (end-to-end).
* Metrics: macro-F1 (primary), accuracy, per-class precision/recall/F1, confusion matrices, abstention rate (rules),
  top-label ECE (5 bins) and Brier score.
* Uncertainty: 95% **cluster bootstrap** (2,000 resamples of capture groups) on pooled out-of-fold predictions.

## 6. Temporal decoding

* Frames are decoded and resampled by FFmpeg at 5 fps (configurable 2/5/10 in the UI).
* **HMM:** 7 hidden states (berdiri, rukuk, iktidal, sujud₁, duduk antara sujud, sujud₂, tasyahud) with grammar
  transitions; per-frame switch probability 1 − exp(−Δt / 2 s); 0.02/s probability mass for off-grammar jumps; emissions
  = fused posture scores of each state's visual class (uniform when no person is found). Viterbi path + forward–backward
  marginals.
* **Hybrid naming (deployed):** keep the HMM segmentation, name phases with local context rules. Chosen on a validation
  seed among {global HMM, hybrid, smoothing baseline} × {2, 3, 5 s} × {0.02, 0.1, 0.3}.
* Uncertain segment: mean score of its visual class < 0.55, or no person in > 50% of its frames.
* **Benchmark:** no labelled prayer videos exist, so 300 sequences per regime (2–4 rakaat, plausible phase durations,
  0.4 s motion blends, dropped frames) use **real out-of-fold scores** of held-out people as emissions; *segment*
  regime (one held-out prediction per segment, conservative) and *iid* regime (fresh prediction per frame, optimistic).
  Metrics: frame phase accuracy, segmental edit score, F1@{10,25,50}, segments/true ratio.

## 7. Model selection rule

Highest mean macro-F1 on both views among candidates, preferring pre-registered and faster pipelines when confidence
intervals overlap. Result: the fusion pipeline (see `docs/experiment_report.md`).
