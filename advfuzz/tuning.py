"""Evolutionary tuning of the fuzzy knowledge base (Sections IV.C.5-7).

* GA-FIS:   GA over 13 genes (12 breakpoints + threshold), unit rule weights.
* PSO-FIS:  PSO over 16 dimensions (15 rule weights + threshold), uniform partition.
* Hybrid:   GA for the partition, then PSO for the weights and threshold with
            that partition frozen.

Fitness is F1 on the detector training split in every case.
"""

from dataclasses import dataclass, replace

import numpy as np

from .fuzzy import (
    N_BREAKPOINTS,
    N_RULES,
    FuzzyInferenceSystem,
    decode_partition,
    defuzzify,
    firing_strengths,
    memberships,
    uniform_partition,
)
from .ga import GAConfig, run_ga
from .metrics import batch_f1
from .pso import PSOConfig, run_pso


@dataclass
class TuningRun:
    fis: FuzzyInferenceSystem
    train_f1: float
    history: list


def partition_fitness(Z, y):
    """Batched GA fitness for chromosomes (N, 13)."""
    y = y.astype(bool)
    unit = np.ones(N_RULES)

    def fitness(pop):
        preds = np.empty((len(pop), len(Z)), dtype=bool)
        for i, genes in enumerate(pop):
            alpha = firing_strengths(memberships(Z, decode_partition(genes[:N_BREAKPOINTS])))
            preds[i] = defuzzify(alpha, unit) >= genes[N_BREAKPOINTS]
        return batch_f1(y, preds)

    return fitness


def weight_fitness(Z, y, breakpoints):
    """Batched PSO fitness for particles (M, 16). Firing strengths are fixed by the
    partition, so they are computed once."""
    y = y.astype(bool)
    alpha = firing_strengths(memberships(Z, breakpoints))

    def fitness(swarm):
        scores = defuzzify(alpha[None, :, :], swarm[:, None, :N_RULES])
        return batch_f1(y, scores >= swarm[:, N_RULES, None])

    return fitness


def tune_ga(Z, y, config=None, rng=None):
    res = run_ga(partition_fitness(Z, y), N_BREAKPOINTS + 1, config or GAConfig(), rng)
    fis = FuzzyInferenceSystem(decode_partition(res.best[:N_BREAKPOINTS]), np.ones(N_RULES), float(res.best[-1]))
    return TuningRun(fis, res.best_fitness, res.history)


def tune_pso(Z, y, config=None, rng=None, breakpoints=None, init=None):
    bp = uniform_partition() if breakpoints is None else breakpoints
    res = run_pso(weight_fitness(Z, y, bp), N_RULES + 1, config or PSOConfig(), rng, init)
    fis = FuzzyInferenceSystem(bp.copy(), res.best[:N_RULES].copy(), float(res.best[-1]))
    return TuningRun(fis, res.best_fitness, res.history)


def tune_hybrid(Z, y, ga_config=None, pso_config=None, rng=None):
    """Sequential GA -> PSO.

    The GA fixes the partition; a fresh PSO swarm then searches the rule
    weights and threshold with that partition frozen. The returned history has
    one entry per GA generation followed by one per PSO iteration, with the PSO
    initial swarm placed at the GA's final step.
    """
    ga_config = ga_config or replace(GAConfig(), generations=40)
    pso_config = pso_config or replace(PSOConfig(), iterations=40)
    ga = tune_ga(Z, y, ga_config, rng)
    pso = tune_pso(Z, y, pso_config, rng, ga.fis.breakpoints)
    return TuningRun(pso.fis, pso.train_f1, ga.history[:-1] + pso.history)
