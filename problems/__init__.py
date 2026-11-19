"""Catálogo de problemas de optimización para Metaheuristic Studio."""

from problems.ackley import AckleyProblem
from problems.delivery_routing import DeliveryRoutingProblem, haversine_km
from problems.onemax import OneMaxProblem
from problems.task_assignment import TaskAssignmentProblem

PROBLEM_REGISTRY = {
    "ackley": AckleyProblem,
    "onemax": OneMaxProblem,
    "task_assignment": TaskAssignmentProblem,
    "delivery_routing": DeliveryRoutingProblem,
}

__all__ = [
    "AckleyProblem",
    "OneMaxProblem",
    "TaskAssignmentProblem",
    "DeliveryRoutingProblem",
    "PROBLEM_REGISTRY",
    "haversine_km",
]

