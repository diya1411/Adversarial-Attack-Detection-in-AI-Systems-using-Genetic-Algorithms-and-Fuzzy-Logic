"""Inference-time detection descriptors C, U, P, F (Section IV.C.2, Eqs. 2-5, Table I).

None of them needs the clean counterpart of the input, so all four are
available in deployment.
"""

import numpy as np

NAMES = ("C", "U", "P", "F")


def confidence(probs):
    """C(x) = max_k p_k(x)  (Eq. 2)."""
    return probs.max(axis=1)


def _plogp(p, log):
    safe = np.where(p > 0, p, 1.0)
    return np.where(p > 0, p * log(safe), 0.0)


def uncertainty(probs):
    """Shannon entropy normalised by its maximum log K  (Eq. 3)."""
    return -_plogp(probs, np.log).sum(axis=1) / np.log(probs.shape[1])


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits, row-wise."""
    m = 0.5 * (p + q)
    kl_pm = (_plogp(p, np.log2) - p * np.log2(np.where(m > 0, m, 1.0))).sum(axis=1)
    kl_qm = (_plogp(q, np.log2) - q * np.log2(np.where(m > 0, m, 1.0))).sum(axis=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def perturbation_response(model, X, draws=16, sigma=0.12, rng=None):
    """Mean JS divergence between p(x) and p(x + eta_r), eta_r ~ N(0, sigma^2 I)  (Eq. 4).

    The R noise vectors are drawn once and shared by every input in the batch,
    so P(x) is a fixed function of x that does not depend on batch composition.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    noise = rng.normal(0.0, sigma, (draws, X.shape[1]))
    p0 = model.predict_proba(X)
    total = np.zeros(len(X))
    for eta in noise:
        total += js_divergence(p0, model.predict_proba(X + eta))
    return total / draws


class FeatureInconsistency:
    """Mean L2 distance in penultimate space to the k nearest training samples of
    the predicted class (Eq. 5), closely related to Deep k-NN credibility."""

    def __init__(self, model, X_train, y_train, k=10):
        self.k = k
        self.reps = {int(c): model.penultimate(X_train[y_train == c]) for c in np.unique(y_train)}

    @classmethod
    def from_representations(cls, reps, k=10):
        obj = cls.__new__(cls)
        obj.k, obj.reps = k, dict(reps)
        return obj

    def __call__(self, H, y_pred):
        out = np.empty(len(H))
        for c in np.unique(y_pred):
            mask = y_pred == c
            T = self.reps[c]
            d2 = (H[mask] ** 2).sum(1)[:, None] + (T ** 2).sum(1)[None, :] - 2.0 * H[mask] @ T.T
            d = np.sqrt(np.maximum(d2, 0.0))
            k = min(self.k, T.shape[0])
            out[mask] = np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)
        return out


def extract(model, X, feature_inconsistency, draws=16, sigma=0.12, rng=None):
    """Raw descriptor matrix (n, 4) with columns C, U, P, F."""
    probs = model.predict_proba(X)
    C = confidence(probs)
    U = uncertainty(probs)
    P = perturbation_response(model, X, draws, sigma, rng)
    F = feature_inconsistency(model.penultimate(X), probs.argmax(axis=1))
    return np.column_stack([C, U, P, F])


class QuantileNormaliser:
    """Map each descriptor through its empirical distribution function (Eq. 6).

    Fitted on the detector training split only. The map is monotone, so it
    cannot change the ranking induced by any single descriptor.
    """

    def fit(self, V):
        self.sorted_ = np.sort(np.asarray(V, dtype=np.float64), axis=0)
        return self

    def transform(self, V):
        V = np.atleast_2d(V)
        n = self.sorted_.shape[0]
        return np.column_stack([
            np.searchsorted(self.sorted_[:, j], V[:, j], side="right") / n for j in range(V.shape[1])
        ])
