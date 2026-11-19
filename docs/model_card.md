# Model card — Salat Vision posture model v1.0.0

Bundle: `models/salat_vision_v1/` (manifest with classes, preprocessing, metrics, SHA-256 of every file).

## Model

| | |
|---|---|
| Task | classify the visible posture of one person: standing (qiyam), bowing (ruku), prostration (sujud), sitting (julus) |
| Video output | 6 prayer phases via a temporal decoder (Iktidal, Duduk di antara dua sujud and Tasyahud are inferred from order) |
| Localisation | EfficientDet-Lite0 (COCO person) → MediaPipe pose box → full frame |
| Pose branch | MediaPipe Pose Landmarker (full) → 19 geometric features (+ missing indicators) → logistic regression (C = 10) |
| Image branch | MobileNetV3-Large, ImageNet weights, frozen, ONNX → 960-d embedding → logistic regression (C = 0.01) |
| Fusion | mean of the two probability vectors; image branch alone when no pose is found |
| Decoder | 7-state HMM (prayer grammar, 2 s mean-phase prior, 0.02/s off-grammar mass) + local context naming |
| Training data | 40 annotated people (37 images, 29 independent groups) from the public sample of the Salat Postures Dataset |
| Size / speed | 11.9 MB bundle; 84 ms per frame end-to-end on 4 vCPU (Intel Xeon 2.1 GHz), no GPU |

## Evaluation (3 × repeated stratified group 5-fold CV; 95% cluster-bootstrap CIs)

| View | Macro-F1 | Accuracy |
|---|---|---|
| Annotated person crops | **0.80** [0.68, 0.90] | 0.83 [0.72, 0.96] |
| Whole images, automatic localisation | **0.68** [0.53, 0.86] | 0.72 [0.56, 0.93] |

Per class (crops, out-of-fold): standing F1 0.94, bowing 0.67 (recall 0.56), prostration 0.91, sitting 0.67 (recall 0.60).
Main confusions: bowing → sitting, sitting → standing/prostration. End-to-end, sitting is confused with prostration in
43% of cases (crops from the person detector differ from annotated boxes).

Calibration: ECE 0.18 — scores are **under-confident** (e.g. bin with mean score 0.50 is 83% accurate). The
"uncertain" flag (score < 0.55) is therefore conservative.

Temporal decoding (semi-synthetic, segment-consistent errors, 300 sequences): frame phase accuracy 0.79, segmental edit
75.7, F1@50 71.0, versus 0.70 / 23.8 / 25.6 for raw per-frame labels.

## Intended use

Research and education demo; portfolio project; prototyping feedback tools for learning the order of prayer movements.

## Out of scope

Judging whether a prayer is valid or accepted; assessing devotion; identifying people; surveillance or monitoring of
individuals without consent; any high-stakes decision.

## Limitations and risks

* Trained on 40 people; the full dataset and real prayer videos are needed before any claim of robustness.
* Capture source is confounded with class in the training data (shortcut risk); pose features mitigate but do not remove it.
* Pose estimation fails more often in prostration (pose found on 67% of prostration crops); the image branch covers those frames.
* Untested across camera heights, frontal views of bowing, children, women's prayer clothing, groups (jamaah), low light.
* Scores are not calibrated probabilities.

## Ethical considerations

Prayer is a private, religious act. The system processes video locally, deletes uploads after analysis, never needs an
external API, and frames its output as posture recognition — not religious assessment.
