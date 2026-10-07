"""Particle swarm optimisation with linearly decaying inertia (Section IV.C.6, Eq. 10).

Positions live in the unit box [0, 1]^d and are projected back onto it after
every move; velocities are clamped to +/- v_max.
"""

from dataclasses import dataclass

import numpy as np

from .ga import OptimResult


@dataclass
class PSOConfig:
    swarm_size: int = 40
    iterations: int = 60
    w_start: float = 0.9
    w_end: float = 0.4
    c1: float = 1.6
    c2: float = 1.6
    v_max: float = 0.25


def run_pso(fitness, dim, config=None, rng=None, init=None):
    """Maximise `fitness` (batched, (M, dim) -> (M,)).

    `init` optionally provides positions for the first particles; the rest of
    the swarm is initialised uniformly at random.
    """
    cfg = config or PSOConfig()
    rng = np.random.default_rng() if rng is None else rng
    M = cfg.swarm_size
    S = rng.random((M, dim))
    if init is not None:
        init = np.atleast_2d(init)
        S[:len(init)] = init
    V = rng.uniform(-cfg.v_max, cfg.v_max, (M, dim))

    f = fitness(S)
    B, fb = S.copy(), f.copy()
    g_idx = int(np.argmax(fb))
    g, fg = B[g_idx].copy(), float(fb[g_idx])
    history = [fg]

    T = cfg.iterations
    for t in range(T):
        w = cfg.w_start - (cfg.w_start - cfg.w_end) * t / max(T - 1, 1)
        r1, r2 = rng.random((M, dim)), rng.random((M, dim))
        V = w * V + cfg.c1 * r1 * (B - S) + cfg.c2 * r2 * (g - S)
        V = np.clip(V, -cfg.v_max, cfg.v_max)
        S = np.clip(S + V, 0.0, 1.0)

        f = fitness(S)
        improved = f > fb
        B[improved], fb[improved] = S[improved], f[improved]
        g_idx = int(np.argmax(fb))
        if fb[g_idx] > fg:
            g, fg = B[g_idx].copy(), float(fb[g_idx])
        history.append(fg)

    return OptimResult(g, fg, history)
