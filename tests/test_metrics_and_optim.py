import numpy as np
import pytest
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import roc_auc_score

from advfuzz.fuzzy import N_RULES, FuzzyInferenceSystem, decode_partition
from advfuzz.ga import GAConfig, run_ga
from advfuzz.metrics import batch_f1, best_f1_threshold, bootstrap_f1_difference, f1_score, mcnemar, roc_auc
from advfuzz.pso import PSOConfig, run_pso
from advfuzz.tuning import partition_fitness, tune_ga, tune_hybrid, tune_pso

rng = np.random.default_rng(0)


def test_auc_matches_sklearn_with_ties():
    y = rng.integers(2, size=300)
    s = np.round(rng.random(300) + 0.3 * y, 1)  # heavy ties
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_f1_matches_sklearn():
    y, p = rng.integers(2, size=200), rng.integers(2, size=200)
    assert f1_score(y, p) == pytest.approx(sk_f1(y, p))
    assert batch_f1(y, np.stack([p, 1 - p]).astype(bool))[1] == pytest.approx(sk_f1(y, 1 - p))


def test_best_threshold_attains_max_f1():
    y = rng.integers(2, size=300)
    s = rng.random(300) + 0.5 * y
    t = best_f1_threshold(s, y)
    brute = max(f1_score(y, (s >= c).astype(int)) for c in np.unique(s))
    assert f1_score(y, (s >= t).astype(int)) == pytest.approx(brute)


@pytest.mark.parametrize("b,c,p", [(77, 23, 5.5e-8), (42, 22, 0.017)])
def test_mcnemar_reproduces_paper_values(b, c, p):
    y = np.ones(b + c, dtype=int)
    pred_a = np.r_[np.ones(b), np.zeros(c)].astype(int)
    _, _, pval = mcnemar(y, pred_a, 1 - pred_a)
    assert pval == pytest.approx(p, rel=0.03)


def test_bootstrap_of_identical_predictions_is_zero():
    y = rng.integers(2, size=100)
    p = rng.integers(2, size=100)
    mean, (lo, hi) = bootstrap_f1_difference(y, p, p, n_boot=200)
    assert mean == lo == hi == 0.0


def _sphere(target):
    return lambda X: -((X - target) ** 2).sum(axis=1)


def test_ga_converges_and_history_is_monotone():
    res = run_ga(_sphere(0.3), 5, GAConfig(generations=60), np.random.default_rng(1))
    assert np.all(np.diff(res.history) >= 0)
    np.testing.assert_allclose(res.best, 0.3, atol=0.05)


def test_pso_converges_and_history_is_monotone():
    res = run_pso(_sphere(0.7), 5, PSOConfig(iterations=80), np.random.default_rng(1))
    assert np.all(np.diff(res.history) >= 0)
    np.testing.assert_allclose(res.best, 0.7, atol=0.05)


def _toy_detection(n=400):
    r = np.random.default_rng(3)
    y = r.integers(2, size=n)
    Z = np.clip(r.random((n, 4)) * 0.7 + 0.3 * y[:, None] * r.random((n, 4)), 0, 1)
    return Z, y


def test_tuners_return_valid_systems():
    Z, y = _toy_detection()
    ga = tune_ga(Z, y, GAConfig(pop_size=20, generations=5), np.random.default_rng(0))
    pso = tune_pso(Z, y, PSOConfig(swarm_size=10, iterations=5), np.random.default_rng(0))
    hyb = tune_hybrid(Z, y, GAConfig(pop_size=20, generations=5), PSOConfig(swarm_size=10, iterations=5),
                      np.random.default_rng(0))
    for run in (ga, pso, hyb):
        assert run.fis.weights.shape == (N_RULES,)
        assert f1_score(y, run.fis.predict(Z)) == pytest.approx(run.train_f1)
    assert len(hyb.history) == 5 + 1 + 5


def test_partition_fitness_matches_direct_evaluation():
    Z, y = _toy_detection()
    genes = np.random.default_rng(2).random((3, 13))
    fit = partition_fitness(Z, y)(genes)
    for g, f in zip(genes, fit):
        fis = FuzzyInferenceSystem(decode_partition(g[:12]), np.ones(N_RULES), g[12])
        assert f1_score(y, fis.predict(Z)) == pytest.approx(f)
