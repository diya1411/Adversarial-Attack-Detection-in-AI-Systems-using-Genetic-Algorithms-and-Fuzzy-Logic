import numpy as np

from advfuzz.descriptors import FeatureInconsistency, QuantileNormaliser
from advfuzz.detector import AdversarialDetector
from advfuzz.fuzzy import N_RULES, FuzzyInferenceSystem
from advfuzz.mlp import MLP


def test_save_load_reproduces_scores(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.random((60, 64)), rng.integers(10, size=60)
    model = MLP(seed=0).fit(X, y, epochs=2)
    fi = FeatureInconsistency(model, X, y, k=3)
    det = AdversarialDetector(model, fi, None, FuzzyInferenceSystem(weights=rng.random(N_RULES), theta=0.4))
    det.normaliser = QuantileNormaliser().fit(det.descriptors(X))

    path = tmp_path / "detector.npz"
    det.save(path)
    back = AdversarialDetector.load(path)

    np.testing.assert_allclose(back.score(X), det.score(X))
    assert list(back.triage(X)) == list(det.triage(X))
    ex = back.explain(X[0])
    assert set(ex["descriptors"]) == set("CUPF")
