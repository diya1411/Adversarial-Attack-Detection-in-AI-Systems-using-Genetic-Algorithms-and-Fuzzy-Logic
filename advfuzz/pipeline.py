"""Construction of the detection dataset (Section IV.B, stages i-vii)."""

from dataclasses import dataclass, field
from math import ceil

import numpy as np

from .attack import fgsm
from .data import GREY_LEVELS, N_CLASSES, load_split
from .descriptors import FeatureInconsistency, QuantileNormaliser, extract
from .ga import GAConfig
from .mlp import MLP
from .pso import PSOConfig


@dataclass
class ExperimentConfig:
    data_seed: int = 0
    holdout_fraction: float = 0.4
    hidden: tuple = (64, 32)
    epochs: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 32
    eps_levels: tuple = (1, 2, 3, 4)  # budgets eps = level / 16
    knn_k: int = 10
    noise_draws: int = 16
    noise_sigma: float = 0.12
    noise_seed: int = 0
    test_fraction: float = 0.4
    n_seeds: int = 10
    ga: GAConfig = field(default_factory=GAConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    hybrid_generations: int = 40
    hybrid_iterations: int = 40
    n_bootstrap: int = 5000


@dataclass
class DetectionData:
    model: MLP
    feature_inconsistency: FeatureInconsistency
    normaliser: QuantileNormaliser
    classifier_stats: dict
    attack_table: list      # Table III rows
    X: np.ndarray           # record images (n, 64)
    X_clean: np.ndarray     # clean source image of each record
    label: np.ndarray       # 1 = adversarial
    eps_level: np.ndarray   # 0 for clean records
    source: np.ndarray      # attack-pool index of the source image
    D: np.ndarray           # raw descriptors (n, 4)
    Z: np.ndarray           # quantile-normalised descriptors (n, 4)
    train: np.ndarray       # boolean mask of the detector training split
    pool_labels: np.ndarray = None

    @property
    def test(self):
        return ~self.train


def build_detection_data(cfg=None, log=print):
    cfg = cfg or ExperimentConfig()
    rng = np.random.default_rng(cfg.data_seed)

    # (i) train the target classifier
    X_tr, y_tr, X_hold, y_hold = load_split(cfg.holdout_fraction, cfg.data_seed)
    model = MLP((X_tr.shape[1], *cfg.hidden, N_CLASSES), seed=cfg.data_seed)
    model.fit(X_tr, y_tr, cfg.epochs, cfg.learning_rate, cfg.batch_size, seed=cfg.data_seed)
    stats = {
        "n_train": len(X_tr),
        "n_holdout": len(X_hold),
        "train_accuracy": float(np.mean(model.predict(X_tr) == y_tr)),
        "holdout_accuracy": float(np.mean(model.predict(X_hold) == y_hold)),
    }

    # (ii) attack pool: held-out images the classifier already gets right
    correct = model.predict(X_hold) == y_hold
    X_pool, y_pool = X_hold[correct], y_hold[correct]
    stats["pool_size"] = int(correct.sum())
    log(f"classifier: train acc {stats['train_accuracy']:.4f}, held-out acc "
        f"{stats['holdout_accuracy']:.4f}, attack pool {stats['pool_size']}")

    # (iii)+(iv) quantised FGSM at every budget, keep successful attacks only
    adversarial, success, off_grid = {}, {}, {}
    for k in cfg.eps_levels:
        adversarial[k], raw = fgsm(model, X_pool, y_pool, k / GREY_LEVELS, return_unquantised=True)
        success[k] = model.predict(adversarial[k]) != y_pool
        off_grid[k] = float(np.mean(adversarial[k] != raw))
    n_per_budget = min(int(s.sum()) for s in success.values())

    attack_table = []
    for k in cfg.eps_levels:
        attack_table.append({
            "eps_level": k,
            "success_rate": float(success[k].mean()),
            "classifier_accuracy": float(1.0 - success[k].mean()),
            "mean_l2": float(np.linalg.norm(adversarial[k] - X_pool, axis=1).mean()),
            "used": n_per_budget,
            "off_grid_pixel_fraction": off_grid[k],  # pixels moved by the quantisation step
        })

    # Equal number of adversarial examples per budget plus every clean image.
    n_pool = len(X_pool)
    X_parts, src_parts, lvl_parts = [X_pool], [np.arange(n_pool)], [np.zeros(n_pool, dtype=int)]
    for k in cfg.eps_levels:
        chosen = np.sort(rng.choice(np.flatnonzero(success[k]), n_per_budget, replace=False))
        X_parts.append(adversarial[k][chosen])
        src_parts.append(chosen)
        lvl_parts.append(np.full(n_per_budget, k))
    X = np.concatenate(X_parts)
    source = np.concatenate(src_parts)
    eps_level = np.concatenate(lvl_parts)
    label = (eps_level > 0).astype(np.int64)

    # (v) descriptors, none of which uses the clean counterpart
    fi = FeatureInconsistency(model, X_tr, y_tr, k=cfg.knn_k)
    D = extract(model, X, fi, cfg.noise_draws, cfg.noise_sigma, rng=np.random.default_rng(cfg.noise_seed))

    # (vi) split grouped by source image so no image straddles train and test
    test_sources = rng.permutation(n_pool)[:ceil(cfg.test_fraction * n_pool)]
    train = ~np.isin(source, test_sources)

    # (vii) the normaliser sees the training split only
    normaliser = QuantileNormaliser().fit(D[train])
    Z = normaliser.transform(D)

    log(f"detection dataset: {label.sum()} adversarial ({n_per_budget} per budget) + "
        f"{n_pool} clean = {len(label)} records; {train.sum()} train / {(~train).sum()} test")
    return DetectionData(model, fi, normaliser, stats, attack_table, X, X_pool[source], label,
                         eps_level, source, D, Z, train, pool_labels=y_pool[source])
