"""Real-coded genetic algorithm (Section IV.C.5, Algorithm 1).

Binary tournament selection, BLX-alpha crossover, per-gene Gaussian mutation
and elitism, over chromosomes in the unit box [0, 1]^d. The fitness function
receives a whole population (N, d) and returns N fitness values.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GAConfig:
    pop_size: int = 60
    generations: int = 60
    p_crossover: float = 0.85
    p_mutation: float = 0.12
    blx_alpha: float = 0.40
    elites: int = 2
    mutation_sigma: float = 0.10  # not given in the paper


@dataclass
class OptimResult:
    best: np.ndarray
    best_fitness: float
    history: list = field(default_factory=list)  # best fitness after each generation, incl. generation 0


def _tournament(fit, n, rng):
    """Indices of n winners of binary tournaments."""
    a = rng.integers(len(fit), size=n)
    b = rng.integers(len(fit), size=n)
    return np.where(fit[a] >= fit[b], a, b)


def _blx(pa, pb, alpha, rng):
    """BLX-alpha: sample each gene uniformly from the parental interval extended by alpha."""
    lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
    span = hi - lo
    lo, hi = lo - alpha * span, hi + alpha * span
    return lo + rng.random(pa.shape) * (hi - lo), lo + rng.random(pa.shape) * (hi - lo)


def run_ga(fitness, n_genes, config=None, rng=None):
    cfg = config or GAConfig()
    rng = np.random.default_rng() if rng is None else rng
    N = cfg.pop_size
    pop = rng.random((N, n_genes))
    fit = fitness(pop)
    history = [float(fit.max())]

    n_children = N - cfg.elites
    n_pairs = (n_children + 1) // 2
    for _ in range(cfg.generations):
        elite_idx = np.argsort(-fit, kind="stable")[:cfg.elites]

        pa = pop[_tournament(fit, n_pairs, rng)]
        pb = pop[_tournament(fit, n_pairs, rng)]
        ca, cb = _blx(pa, pb, cfg.blx_alpha, rng)
        cross = rng.random(n_pairs) < cfg.p_crossover
        ca = np.where(cross[:, None], ca, pa)
        cb = np.where(cross[:, None], cb, pb)
        children = np.concatenate([ca, cb])[:n_children]

        mutate = rng.random(children.shape) < cfg.p_mutation
        children = children + mutate * rng.normal(0.0, cfg.mutation_sigma, children.shape)
        children = np.clip(children, 0.0, 1.0)

        pop = np.concatenate([pop[elite_idx], children])
        fit = np.concatenate([fit[elite_idx], fitness(children)])
        history.append(float(fit.max()))

    best = int(np.argmax(fit))
    return OptimResult(pop[best].copy(), float(fit[best]), history)
