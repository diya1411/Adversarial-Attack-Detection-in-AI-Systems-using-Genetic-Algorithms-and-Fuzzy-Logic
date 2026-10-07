"""Fast Gradient Sign Method with valid-image quantisation (Section IV.C.1, Eq. 1)."""

import numpy as np

from .data import GREY_LEVELS


def quantise(X, levels=GREY_LEVELS):
    """Snap pixels back onto the valid grey-level grid, x'' = round(16 x') / 16.

    Without this step a detector could flag adversarial images simply by
    noticing off-grid pixel values.
    """
    return np.round(X * levels) / levels


def fgsm(model, X, y, eps, levels=GREY_LEVELS, return_unquantised=False):
    """White-box L-infinity FGSM step, projected onto [0, 1] and quantised."""
    grad = model.input_gradient(X, y)
    x_adv = np.clip(X + eps * np.sign(grad), 0.0, 1.0)
    x_q = quantise(x_adv, levels)
    return (x_q, x_adv) if return_unquantised else x_q
