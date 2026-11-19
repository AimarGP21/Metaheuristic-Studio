"""Catálogo de algoritmos metaheurísticos implementados en Metaheuristic Studio."""

from algorithms.aco import AntColonyOptimization
from algorithms.ais import ArtificialImmuneSystem
from algorithms.de import DifferentialEvolution
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import ParticleSwarmOptimization

ALGORITHM_REGISTRY = {
    "GA": GeneticAlgorithm,
    "PSO": ParticleSwarmOptimization,
    "DE": DifferentialEvolution,
    "ACO": AntColonyOptimization,
    "AIS": ArtificialImmuneSystem,
}

__all__ = [
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
    "DifferentialEvolution",
    "AntColonyOptimization",
    "ArtificialImmuneSystem",
    "ALGORITHM_REGISTRY",
]
