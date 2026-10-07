"""Reference detectors (Table V). Every operating threshold is fitted for F1 on the
detector training split, the same criterion the optimisers use."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import best_f1_threshold


class ThresholdDetector:
    """Threshold a single raw descriptor; `score_fn` maps descriptors (n, 4) to a
    score where larger means more likely adversarial."""

    def __init__(self, score_fn):
        self.score_fn = score_fn

    def fit(self, D, y):
        self.threshold = best_f1_threshold(self.score_fn(D), y)
        return self

    def score(self, D):
        return self.score_fn(D)

    def predict(self, D):
        return (self.score(D) >= self.threshold).astype(np.int64)


def msp_detector():
    """Maximum softmax probability (Hendrycks and Gimpel): low confidence is suspicious."""
    return ThresholdDetector(lambda D: 1.0 - D[:, 0])


def entropy_detector():
    return ThresholdDetector(lambda D: D[:, 1])


def knn_distance_detector():
    return ThresholdDetector(lambda D: D[:, 3])


class LogisticDetector:
    """Logistic regression on the four normalised descriptors, the opaque reference
    point that quantifies what interpretability costs."""

    def fit(self, Z, y):
        self.model = LogisticRegression(max_iter=1000).fit(Z, y)
        self.threshold = best_f1_threshold(self.score(Z), y)
        return self

    def score(self, Z):
        return self.model.predict_proba(Z)[:, 1]

    def predict(self, Z):
        return (self.score(Z) >= self.threshold).astype(np.int64)
