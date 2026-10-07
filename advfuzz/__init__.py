"""Adversarial attack detection with a GA/PSO-tuned fuzzy inference system.

Implementation of "Adversarial Attack Detection in AI Systems using Genetic
Algorithms and Fuzzy Logic" (Bhat, Kote, Thaware). Every learning component -
the target classifier and its input gradients, the FGSM attack, the fuzzy
inference system, the genetic algorithm and particle swarm optimisation - is
written directly in NumPy so that it can be audited end to end.
"""

from .fuzzy import FuzzyInferenceSystem, RULES, uniform_partition
from .mlp import MLP

__all__ = ["FuzzyInferenceSystem", "MLP", "RULES", "uniform_partition"]
