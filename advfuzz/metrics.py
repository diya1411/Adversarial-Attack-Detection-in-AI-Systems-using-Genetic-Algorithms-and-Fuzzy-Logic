"""Detection metrics and significance tests (Sections V.B and V.D)."""

from math import comb

import numpy as np


def f1_score(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def batch_f1(y, preds):
    """F1 of every row of a boolean prediction matrix (m, n) against labels y (n,)."""
    y = y.astype(bool)
    tp = (preds & y).sum(axis=1)
    fp = (preds & ~y).sum(axis=1)
    fn = (~preds & y).sum(axis=1)
    den = 2 * tp + fp + fn
    return np.where(tp > 0, 2 * tp / np.maximum(den, 1), 0.0)


def confusion(y, pred):
    """[[TN, FP], [FN, TP]] with rows = true class (clean, adversarial)."""
    return np.array([
        [np.sum((y == 0) & (pred == 0)), np.sum((y == 0) & (pred == 1))],
        [np.sum((y == 1) & (pred == 0)), np.sum((y == 1) & (pred == 1))],
    ])


def roc_auc(y, scores):
    """Area under the ROC curve via the Mann-Whitney statistic (ties count one half)."""
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    n1 = int(np.sum(y == 1))
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_curve(y, scores):
    """False- and true-positive rates for every distinct threshold, descending."""
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    last = np.r_[s[1:] != s[:-1], True]
    tps = np.cumsum(yy)[last]
    fps = np.cumsum(1 - yy)[last]
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]]


def detection_metrics(y, pred, scores=None):
    (tn, fp), (fn, tp) = confusion(y, pred)
    out = {
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": f1_score(y, pred),
    }
    if scores is not None:
        out["auc"] = roc_auc(y, scores)
    return {k: float(v) for k, v in out.items()}


def best_f1_threshold(scores, y):
    """Threshold t maximising training F1 for the rule `score >= t`.

    Cuts are only placed between distinct score values; the threshold sits at
    the midpoint of the gap so it generalises to unseen scores.
    """
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    f1 = 2 * tp / (tp + fp + yy.sum())
    last = np.r_[s[1:] != s[:-1], True]
    i = int(np.argmax(np.where(last, f1, -1.0)))
    return float(s[i]) if i == len(s) - 1 else float(0.5 * (s[i] + s[i + 1]))


def mcnemar(y, pred_a, pred_b):
    """Exact two-sided McNemar test on paired decisions.

    Returns (b, c, p) where b counts samples A gets right and B gets wrong,
    and c the reverse.
    """
    ok_a, ok_b = pred_a == y, pred_b == y
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return b, c, float(min(1.0, 2 * tail))


def bootstrap_f1_difference(y, pred_a, pred_b, n_boot=5000, rng=None):
    """Bootstrap distribution of F1(A) - F1(B) over test-set resamples.

    Returns (mean, (lo, hi)) with a 95% percentile interval.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(len(y), size=(n_boot, len(y)))
    yb = y[idx].astype(bool)
    diff = batch_f1(yb, pred_a[idx].astype(bool)) - batch_f1(yb, pred_b[idx].astype(bool))
    return float(diff.mean()), (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
