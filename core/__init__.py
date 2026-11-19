"""Paquete core de Metaheuristic Studio."""

from core.base import (
    ALGORITHM_ALIASES,
    COMPATIBILITY_MATRIX,
    PROBLEM_ALIASES,
    BaseMetaheuristic,
    BaseProblem,
    EvaluationResult,
    MetaheuristicAlgorithm,
    ObjectiveWrapper,
    OptimizationSense,
    Problem,
    RepresentationType,
    check_compatibility,
    evaluate_with_objective,
    get_compatible_algorithms,
)

__all__ = [
    "ALGORITHM_ALIASES",
    "COMPATIBILITY_MATRIX",
    "PROBLEM_ALIASES",
    "BaseMetaheuristic",
    "BaseProblem",
    "EvaluationResult",
    "MetaheuristicAlgorithm",
    "ObjectiveWrapper",
    "OptimizationSense",
    "Problem",
    "RepresentationType",
    "check_compatibility",
    "evaluate_with_objective",
    "get_compatible_algorithms",
]
