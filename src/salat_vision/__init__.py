"""Salat Vision — prayer-posture recognition from images and video.

The package is organised as:

* ``salat_vision.config``      – central, overridable configuration and paths
* ``salat_vision.labels``      – visual posture classes, prayer phases and label normalisation
* ``salat_vision.data``        – safe dataset extraction, Pascal VOC parsing, audit, group-aware splits
* ``salat_vision.features``    – MediaPipe pose landmarks, engineered pose features, CNN embeddings
* ``salat_vision.models``      – CNN transfer learning, landmark classifiers, rule baseline, fusion
* ``salat_vision.evaluation``  – metrics with bootstrap confidence intervals, calibration, plots
* ``salat_vision.inference``   – image / video inference, temporal decoding, annotation, bundle loading
"""

__version__ = "0.1.0"
