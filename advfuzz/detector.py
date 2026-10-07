"""Deployable detector: a fixed function wrapped around an unmodified classifier (Fig. 1).

input -> classifier -> descriptors C, U, P, F -> quantile normalisation
      -> fuzzy inference -> risk score -> Safe / Suspicious / Adversarial
"""

import json

import numpy as np

from .descriptors import FeatureInconsistency, QuantileNormaliser, extract
from .fuzzy import FuzzyInferenceSystem
from .mlp import MLP


class AdversarialDetector:
    def __init__(self, model, feature_inconsistency, normaliser, fis, noise_draws=16, noise_sigma=0.12, noise_seed=0):
        self.model = model
        self.feature_inconsistency = feature_inconsistency
        self.normaliser = normaliser
        self.fis = fis
        self.noise_draws = noise_draws
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed

    def descriptors(self, X):
        # A fixed noise seed makes the perturbation-response descriptor, and
        # therefore the whole detector, deterministic at inference time.
        rng = np.random.default_rng(self.noise_seed)
        return extract(self.model, np.atleast_2d(X), self.feature_inconsistency, self.noise_draws, self.noise_sigma, rng)

    def normalised(self, X):
        return self.normaliser.transform(self.descriptors(X))

    def score(self, X):
        return self.fis.score(self.normalised(X))

    def predict(self, X):
        return self.fis.predict(self.normalised(X))

    def triage(self, X):
        return self.fis.triage(self.normalised(X))

    def explain(self, x, top=5):
        d = self.descriptors(x)
        out = self.fis.explain(self.normaliser.transform(d)[0], top)
        out["descriptors"] = dict(zip("CUPF", map(float, d[0])))
        out["predicted_class"] = int(self.model.predict(np.atleast_2d(x))[0])
        return out

    # ------------------------------------------------------------- persistence
    def save(self, path):
        reps = self.feature_inconsistency.reps
        classes = sorted(reps)
        arrays = {f"model_{k}": v for k, v in self.model.state_dict().items()}
        arrays.update({f"reps_{c}": reps[c] for c in classes})
        arrays["normaliser_sorted"] = self.normaliser.sorted_
        meta = {
            "fis": self.fis.to_dict(),
            "classes": [int(c) for c in classes],
            "knn_k": self.feature_inconsistency.k,
            "noise_draws": self.noise_draws,
            "noise_sigma": self.noise_sigma,
            "noise_seed": self.noise_seed,
        }
        np.savez(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            model = MLP.from_state_dict({k[len("model_"):]: f[k] for k in f.files if k.startswith("model_")})
            reps = {c: f[f"reps_{c}"] for c in meta["classes"]}
            normaliser = QuantileNormaliser()
            normaliser.sorted_ = f["normaliser_sorted"]
        fi = FeatureInconsistency.from_representations(reps, meta["knn_k"])
        fis = FuzzyInferenceSystem.from_dict(meta["fis"])
        return cls(model, fi, normaliser, fis, meta["noise_draws"], meta["noise_sigma"], meta["noise_seed"])
