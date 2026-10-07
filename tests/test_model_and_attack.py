import numpy as np
import pytest

from advfuzz.attack import fgsm, quantise
from advfuzz.data import GREY_LEVELS, load_split
from advfuzz.descriptors import (
    FeatureInconsistency,
    QuantileNormaliser,
    js_divergence,
    perturbation_response,
    uncertainty,
)
from advfuzz.mlp import MLP


@pytest.fixture(scope="module")
def trained():
    X_tr, y_tr, X_ho, y_ho = load_split()
    model = MLP(seed=0).fit(X_tr, y_tr, epochs=15)
    return model, X_tr, y_tr, X_ho, y_ho


def test_split_sizes():
    X_tr, _, X_ho, _ = load_split()
    assert (len(X_tr), len(X_ho)) == (1078, 719)
    assert X_tr.min() >= 0 and X_tr.max() <= 1


def test_input_gradient_matches_finite_differences():
    model = MLP(seed=1)
    rng = np.random.default_rng(0)
    X = rng.random((4, 64))
    y = rng.integers(10, size=4)
    g = model.input_gradient(X, y)

    def loss(X):
        return -np.log(model.predict_proba(X)[np.arange(len(y)), y]).sum()

    h = 1e-6
    for i, j in [(0, 3), (1, 20), (2, 40), (3, 63)]:
        e = np.zeros_like(X)
        e[i, j] = h
        assert g[i, j] == pytest.approx((loss(X + e) - loss(X - e)) / (2 * h), rel=1e-4, abs=1e-8)


def test_training_reduces_loss(trained):
    model, X_tr, y_tr, X_ho, y_ho = trained
    assert model.loss(X_tr, y_tr) < MLP(seed=0).loss(X_tr, y_tr)
    assert np.mean(model.predict(X_ho) == y_ho) > 0.9


def test_state_dict_round_trip(trained):
    model, X_tr, *_ = trained
    back = MLP.from_state_dict(model.state_dict())
    np.testing.assert_allclose(back.logits(X_tr[:10]), model.logits(X_tr[:10]))


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_fgsm_is_valid_image_within_budget(trained, level):
    model, _, _, X_ho, y_ho = trained
    eps = level / GREY_LEVELS
    adv = fgsm(model, X_ho, y_ho, eps)
    np.testing.assert_array_equal(adv, quantise(adv))          # on the grey-level grid
    assert adv.min() >= 0 and adv.max() <= 1
    assert np.abs(adv - X_ho).max() <= eps + 1e-12              # L-infinity budget


def test_fgsm_reduces_accuracy(trained):
    model, _, _, X_ho, y_ho = trained
    adv = fgsm(model, X_ho, y_ho, 4 / GREY_LEVELS)
    assert np.mean(model.predict(adv) == y_ho) < 0.2


def test_quantise_snaps_off_grid_values():
    np.testing.assert_allclose(quantise(np.array([0.03, 0.5, 0.97])), [0.0, 0.5, 1.0])


def test_uncertainty_extremes():
    np.testing.assert_allclose(uncertainty(np.eye(10)[:2]), 0.0, atol=1e-12)
    np.testing.assert_allclose(uncertainty(np.full((1, 10), 0.1)), 1.0)


def test_js_divergence_bits():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    np.testing.assert_allclose(js_divergence(p, q), [1.0, 0.0], atol=1e-12)


def test_perturbation_response_is_independent_of_batch(trained):
    model, _, _, X_ho, _ = trained
    batch = perturbation_response(model, X_ho[:20], rng=np.random.default_rng(5))
    single = perturbation_response(model, X_ho[7:8], rng=np.random.default_rng(5))
    assert single[0] == pytest.approx(batch[7])


def test_feature_inconsistency_is_zero_on_duplicated_training_points(trained):
    model, X_tr, y_tr, *_ = trained
    fi = FeatureInconsistency(model, np.repeat(X_tr[:5], 3, axis=0), np.repeat(y_tr[:5], 3), k=3)
    np.testing.assert_allclose(fi(model.penultimate(X_tr[:5]), y_tr[:5]), 0.0, atol=1e-6)


def test_quantile_normaliser_is_monotone_and_uniform():
    rng = np.random.default_rng(0)
    V = rng.lognormal(size=(1000, 2))
    q = QuantileNormaliser().fit(V)
    Z = q.transform(V)
    assert np.all(np.diff(Z[np.argsort(V[:, 0]), 0]) >= 0)
    assert abs(Z[:, 0].mean() - 0.5) < 0.01
