"""
Problema Discreto: Asignación de T Tareas a R Recursos con Restricción de Capacidad.

Representación:
    Vector entero x de longitud T donde x[t] in {0, 1, ..., R - 1} indica el recurso
    asignado a la tarea t.

Función Objetivo (Minimización natural):
    f(x) = sum_{t=0}^{T-1} Costo[t, x[t]] + lambda_pen * sum_{r=0}^{R-1} max(0, Carga(r) - CapMax(r))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from core.base import BaseProblem, OptimizationSense, RepresentationType


class TaskAssignmentProblem(BaseProblem):
    """
    Asignación discreta de T tareas a R recursos utilizando una matriz de costos T x R
    (generable aleatoriamente con semilla o editable por el usuario) y penalización por
    exceder la capacidad máxima por recurso.
    """

    problem_id: str = "task_assignment"
    name: str = "Asignación de Tareas a Recursos (Discreto)"
    description: str = (
        "Asigna T tareas a R recursos minimizando el costo total de la matriz T x R "
        "con penalización por sobrecarga cuando un recurso supera su capacidad máxima."
    )
    representation: RepresentationType = RepresentationType.DISCRETE
    default_sense: OptimizationSense = OptimizationSense.MINIMIZE

    def __init__(
        self,
        n_tasks: int = 12,
        n_resources: int = 4,
        max_capacity: Union[int, float, Sequence[float]] = 3.0,
        penalty_factor: float = 120.0,
        seed: Optional[int] = 42,
        cost_matrix: Optional[np.ndarray] = None,
        task_demands: Optional[Sequence[float]] = None,
        cost_range: Tuple[float, float] = (10.0, 100.0),
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        if n_tasks <= 0 or n_resources <= 0:
            raise ValueError("El número de tareas (T) y recursos (R) debe ser mayor a 0.")
        super().__init__(dimension=n_tasks, sense=sense)

        self.n_tasks: int = int(n_tasks)
        self.n_resources: int = int(n_resources)
        self.penalty_factor: float = float(penalty_factor)
        self.seed: Optional[int] = seed
        self.cost_range: Tuple[float, float] = (float(cost_range[0]), float(cost_range[1]))

        # Demanda por tarea (por defecto cada tarea consume 1 unidad de capacidad del recurso)
        if task_demands is None:
            self.task_demands: np.ndarray = np.ones(self.n_tasks, dtype=float)
        else:
            demands_arr = np.asarray(task_demands, dtype=float).ravel()
            if demands_arr.size != self.n_tasks:
                raise ValueError(
                    f"task_demands debe tener longitud T={self.n_tasks}, recibido {demands_arr.size}."
                )
            self.task_demands = demands_arr

        # Capacidad máxima por recurso (escalar o vector de longitud R)
        if np.ndim(max_capacity) == 0:
            self.max_capacity_per_resource: np.ndarray = np.full(
                self.n_resources, float(max_capacity), dtype=float
            )
            self.max_capacity: float = float(max_capacity)
        else:
            cap_arr = np.asarray(max_capacity, dtype=float).ravel()
            if cap_arr.size != self.n_resources:
                raise ValueError(
                    f"max_capacity vectorial debe tener longitud R={self.n_resources}, "
                    f"recibido {cap_arr.size}."
                )
            self.max_capacity_per_resource = cap_arr
            self.max_capacity = float(np.mean(cap_arr))

        # Matriz de costos T x R
        if cost_matrix is not None:
            self.set_cost_matrix(cost_matrix)
        else:
            self.cost_matrix: np.ndarray = self.generate_random_cost_matrix(seed=self.seed)

    def generate_random_cost_matrix(self, seed: Optional[int] = None) -> np.ndarray:
        """Genera una matriz de costos T x R aleatoria reproducible usando la semilla indicada."""
        rng = np.random.default_rng(seed)
        low, high = self.cost_range
        matrix = rng.integers(int(low), int(high) + 1, size=(self.n_tasks, self.n_resources))
        self.cost_matrix = matrix.astype(float)
        return self.cost_matrix

    def set_cost_matrix(self, matrix: Union[np.ndarray, "pd.DataFrame", Sequence[Sequence[float]]]) -> None:
        """Actualiza o reemplaza la matriz de costos T x R (soporta edición desde la UI)."""
        if hasattr(matrix, "to_numpy"):
            arr = matrix.to_numpy(dtype=float)
        else:
            arr = np.asarray(matrix, dtype=float)

        if arr.shape != (self.n_tasks, self.n_resources):
            raise ValueError(
                f"La matriz de costos debe tener dimensión ({self.n_tasks}, {self.n_resources}), "
                f"pero se recibió {arr.shape}."
            )
        self.cost_matrix = arr.copy()

    def get_cost_dataframe(self) -> "pd.DataFrame":
        """Devuelve la matriz de costos como DataFrame etiquetado para `st.data_editor`."""
        import pandas as pd

        index_labels = [f"Tarea {t + 1}" for t in range(self.n_tasks)]
        col_labels = [f"Recurso {r + 1}" for r in range(self.n_resources)]
        return pd.DataFrame(self.cost_matrix, index=index_labels, columns=col_labels)

    def compute_resource_loads(self, x: np.ndarray) -> np.ndarray:
        """Calcula la carga total acumulada en cada uno de los R recursos para la asignación `x`."""
        assignment = self.repair(x)
        loads = np.zeros(self.n_resources, dtype=float)
        np.add.at(loads, assignment, self.task_demands)
        return loads

    def evaluate_raw(self, x: np.ndarray) -> float:
        """
        Evalúa una asignación discreta `x` de longitud T con valores en {0, ..., R-1}.
        Suma el costo de asignación más la penalización por exceso de capacidad.
        """
        assignment = self.repair(x)
        task_indices = np.arange(self.n_tasks)
        base_cost = float(np.sum(self.cost_matrix[task_indices, assignment]))

        loads = np.zeros(self.n_resources, dtype=float)
        np.add.at(loads, assignment, self.task_demands)

        excess_per_resource = np.maximum(0.0, loads - self.max_capacity_per_resource)
        penalty = float(self.penalty_factor * np.sum(excess_per_resource))
        return base_cost + penalty

    def evaluate_population_raw(self, population: np.ndarray) -> np.ndarray:
        """Evaluación vectorizada de una población de asignaciones de tamaño (M, T)."""
        pop = np.asarray(population)
        if pop.ndim != 2 or pop.shape[1] != self.n_tasks:
            raise ValueError(
                f"Se esperaba matriz de población (M, {self.n_tasks}), se recibió {pop.shape}."
            )
        repaired_pop = np.clip(np.rint(pop), 0, self.n_resources - 1).astype(int)
        task_idx = np.arange(self.n_tasks)[None, :]
        base_costs = np.sum(self.cost_matrix[task_idx, repaired_pop], axis=1)

        # Cargas por recurso para cada individuo
        penalties = np.zeros(repaired_pop.shape[0], dtype=float)
        for r in range(self.n_resources):
            mask = (repaired_pop == r).astype(float)
            load_r = np.sum(mask * self.task_demands[None, :], axis=1)
            excess_r = np.maximum(0.0, load_r - self.max_capacity_per_resource[r])
            penalties += self.penalty_factor * excess_r

        return base_costs + penalties

    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Genera una asignación aleatoria de T tareas en {0, ..., R-1}."""
        generator = rng if rng is not None else np.random.default_rng()
        return generator.integers(0, self.n_resources, size=self.n_tasks, dtype=int)

    def repair(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Redondea y recorta los genes al conjunto discreto válido {0, 1, ..., R - 1}."""
        arr = np.asarray(x).ravel()
        if arr.size != self.n_tasks:
            raise ValueError(
                f"El vector de asignación debe tener longitud T={self.n_tasks}, recibido {arr.size}."
            )
        return np.clip(np.rint(arr), 0, self.n_resources - 1).astype(int)

    def solution_details(self, x: np.ndarray) -> Dict[str, Any]:
        """Devuelve desglose completo de costos por tarea, carga por recurso y penalizaciones."""
        assignment = self.repair(x)
        task_indices = np.arange(self.n_tasks)
        individual_costs = self.cost_matrix[task_indices, assignment]
        base_cost = float(np.sum(individual_costs))

        loads = self.compute_resource_loads(assignment)
        excess = np.maximum(0.0, loads - self.max_capacity_per_resource)
        total_excess = float(np.sum(excess))
        penalty = float(self.penalty_factor * total_excess)
        total_fitness = base_cost + penalty

        overloaded_resources: List[int] = [
            int(r) for r in range(self.n_resources) if excess[r] > 1e-9
        ]

        return {
            "total_fitness": total_fitness,
            "base_cost": base_cost,
            "penalty": penalty,
            "is_feasible": total_excess <= 1e-9,
            "sense": self.sense.value,
            "n_tasks": self.n_tasks,
            "n_resources": self.n_resources,
            "assignment": assignment.tolist(),
            "task_costs": individual_costs.tolist(),
            "resource_loads": loads.tolist(),
            "max_capacity_per_resource": self.max_capacity_per_resource.tolist(),
            "resource_excess": excess.tolist(),
            "total_excess": total_excess,
            "overloaded_resources": overloaded_resources,
        }

    def get_bounds_info(self) -> Dict[str, Any]:
        info = super().get_bounds_info()
        info.update(
            {
                "n_tasks": self.n_tasks,
                "n_resources": self.n_resources,
                "domain": [0, self.n_resources - 1],
                "max_capacity": self.max_capacity,
                "penalty_factor": self.penalty_factor,
                "seed": self.seed,
            }
        )
        return info
