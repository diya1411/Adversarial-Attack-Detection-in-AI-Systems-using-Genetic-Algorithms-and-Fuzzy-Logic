"""UCI Optical Recognition of Handwritten Digits (Section V.A)."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Pixels take the integer values 0..16, i.e. seventeen grey levels.
GREY_LEVELS = 16
IMAGE_SHAPE = (8, 8)
N_CLASSES = 10


def load_split(holdout_fraction=0.4, seed=0):
    """Return the stratified classifier-training / held-out split, pixels in [0, 1].

    With the default 60/40 split this gives 1078 training and 719 held-out images.
    """
    digits = load_digits()
    X = digits.data.astype(np.float64) / GREY_LEVELS
    y = digits.target.astype(np.int64)
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y, random_state=seed
    )
    return X_train, y_train, X_hold, y_hold
