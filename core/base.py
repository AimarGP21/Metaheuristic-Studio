"""
Módulo base de arquitectura para Metaheuristic Studio.

Define:
- Clases abstractas para problemas de optimización (`BaseProblem` / `Problem`) y
  algoritmos metaheurísticos (`BaseMetaheuristic` / `MetaheuristicAlgorithm`).
- Función y clase wrapper de evaluación (`evaluate_with_objective`, `ObjectiveWrapper`)
  para minimizar o maximizar cualquier problema.
- Matriz de compatibilidad (`COMPATIBILITY_MATRIX`) entre los 4 problemas y los 5 algoritmos
  metaheurísticos (GA, PSO, DE, ACO, AIS) con mensajes explicativos detallados.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class RepresentationType(str, Enum):
    """Tipos de representación de soluciones soportados en Metaheuristic Studio."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    DISCRETE = "discrete"
    PERMUTATION = "permutation"


class OptimizationSense(str, Enum):
    """Sentido de optimización."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @classmethod
    def normalize(cls, sense: Union[str, "OptimizationSense"]) -> "OptimizationSense":
        """Normaliza cadenas en español o inglés al enum OptimizationSense."""
        if isinstance(sense, cls):
            return sense
        val = str(sense).strip().lower()
        if val in ("min", "minimize", "minimizar", "minimizacion", "minimización"):
            return cls.MINIMIZE
        if val in ("max", "maximize", "maximizar", "maximizacion", "maximización"):
            return cls.MAXIMIZE
        raise ValueError(
            f"Sentido de optimización no reconocido: '{sense}'. "
            "Usa 'minimize'/'minimizar' o 'maximize'/'maximizar'."
        )


# ============================================================================
# MATRIZ DE COMPATIBILIDAD PROBLEMA <-> ALGORITMO
# ============================================================================
# Cubre los 4 problemas del estudio (ackley, onemax, task_assignment, delivery_routing)
# y los 5 algoritmos metaheurísticos implementados (GA, PSO, DE, ACO, AIS).

COMPATIBILITY_MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {
    "ackley": {
        "GA": {
            "compatible": True,
            "algorithm_name": "Algoritmo Genético (GA)",
            "message": "Compatible mediante codificación real con cruza aritmética/BLX-alpha y mutación gaussiana acotada.",
        },
        "PSO": {
            "compatible": True,
            "algorithm_name": "Enjambre de Partículas (PSO)",
            "message": "Totalmente compatible: PSO continuo con peso de inercia y límites de velocidad opera nativamente en R^D.",
        },
        "DE": {
            "compatible": True,
            "algorithm_name": "Evolución Diferencial (DE)",
            "message": "Totalmente compatible: DE/rand/1/bin es uno de los optimizadores más eficaces en funciones multimodales continuas como Ackley.",
        },
        "ACO": {
            "compatible": False,
            "algorithm_name": "Colonia de Hormigas (ACO)",
            "message": (
                "No compatible: ACO construye soluciones discretas sobre grafos de decisión mediante tablas de feromonas "
                "y no opera directamente sobre variables continuas en [min, max]^D."
            ),
        },
        "AIS": {
            "compatible": True,
            "algorithm_name": "Sistema Inmune Artificial (AIS - CLONALG)",
            "message": "Compatible: CLONALG aplica selección clonal e hipermutación gaussiana inversamente proporcional a la afinidad en espacios continuos.",
        },
    },
    "onemax": {
        "GA": {
            "compatible": True,
            "algorithm_name": "Algoritmo Genético (GA)",
            "message": "Totalmente compatible: OneMax es el problema benchmark clásico para GA con cruza de dos puntos/uniforme y mutación bit-flip.",
        },
        "PSO": {
            "compatible": True,
            "algorithm_name": "Enjambre de Partículas (Binary PSO)",
            "message": "Compatible mediante Binary PSO (BPSO): transforma la velocidad continua con una función sigmoide S(v) para muestrear bits en {0, 1}.",
        },
        "DE": {
            "compatible": False,
            "algorithm_name": "Evolución Diferencial (DE)",
            "message": (
                "No compatible: El esquema diferencial v = x_r1 + F*(x_r2 - x_r3) opera en el continuo R^D y destruye "
                "la estructura booleana discreta {0, 1}^N de OneMax."
            ),
        },
        "ACO": {
            "compatible": True,
            "algorithm_name": "Colonia de Hormigas (ACO Binario)",
            "message": "Compatible: modela un grafo de decisión binaria donde cada bit i in {1..N} tiene feromonas (tau_{i,0}, tau_{i,1}).",
        },
        "AIS": {
            "compatible": True,
            "algorithm_name": "Sistema Inmune Artificial (AIS - CLONALG)",
            "message": "Compatible: CLONALG realiza hipermutación somática bit-flip con tasa inversamente proporcional a la afinidad del anticuerpo.",
        },
    },
    "task_assignment": {
        "GA": {
            "compatible": True,
            "algorithm_name": "Algoritmo Genético (GA)",
            "message": "Compatible usando cromosomas enteros en {0, ..., R-1}^T con cruza uniforme/multipunto y mutación de reasignación aleatoria de recurso.",
        },
        "PSO": {
            "compatible": False,
            "algorithm_name": "Enjambre de Partículas (PSO)",
            "message": (
                "No compatible: Los índices de recursos {0, ..., R-1} son identificadores nominales sin orden métrico; "
                "calcular velocidades continuas entre categorías de recursos carece de sentido físico."
            ),
        },
        "DE": {
            "compatible": False,
            "algorithm_name": "Evolución Diferencial (DE)",
            "message": (
                "No compatible: La diferencia vectorial continua entre índices nominales de recursos genera valores reales "
                "inválidos y asume una linealidad inexistente entre recursos."
            ),
        },
        "ACO": {
            "compatible": True,
            "algorithm_name": "Colonia de Hormigas (ACO Discreto)",
            "message": "Totalmente compatible: deposita feromona tau[t, r] en el grafo bipartito Tarea-Recurso guiado por la heurística eta[t, r] = 1 / Costo[t, r].",
        },
        "AIS": {
            "compatible": True,
            "algorithm_name": "Sistema Inmune Artificial (AIS - CLONALG)",
            "message": "Compatible: los anticuerpos discretos sufren hipermutación proporcional reasignando un mayor número de tareas en anticuerpos de baja afinidad.",
        },
    },
    "delivery_routing": {
        "GA": {
            "compatible": True,
            "algorithm_name": "Algoritmo Genético (GA)",
            "message": "Compatible utilizando codificación de permutación con cruza de orden (OX - Order Crossover) y mutación por intercambio (swap) e inversión (2-opt).",
        },
        "PSO": {
            "compatible": False,
            "algorithm_name": "Enjambre de Partículas (PSO)",
            "message": (
                "No compatible: Las ecuaciones cinemáticas de posición y velocidad de PSO producen vectores continuos con "
                "valores repetidos, violando la restricción biyectiva de una permutación de clientes."
            ),
        },
        "DE": {
            "compatible": False,
            "algorithm_name": "Evolución Diferencial (DE)",
            "message": (
                "No compatible: DE/rand/1/bin está diseñado para espacios vectoriales continuos y rompe la validez "
                "combinatoria de las rutas de visita sin repetición."
            ),
        },
        "ACO": {
            "compatible": True,
            "algorithm_name": "Colonia de Hormigas (ACO Ruteo)",
            "message": "Totalmente compatible: ACO construye rutas paso a paso sobre los arcos de Guadalajara combinando feromona tau[i, j] y visibilidad eta[i, j] = 1 / d(i, j).",
        },
        "AIS": {
            "compatible": False,
            "algorithm_name": "Sistema Inmune Artificial (AIS - CLONALG)",
            "message": (
                "No compatible: La implementación CLONALG está definida para dominios continuo, binario y discreto nominal, "
                "no para permutaciones combinatorias de ruteo vehicular."
            ),
        },
    },
}

# Alias de nombres de problemas y algoritmos para facilitar consultas flexibles
PROBLEM_ALIASES: Dict[str, str] = {
    "ackley": "ackley",
    "continuous": "ackley",
    "continuo": "ackley",
    "onemax": "onemax",
    "one_max": "onemax",
    "binary": "onemax",
    "binario": "onemax",
    "task_assignment": "task_assignment",
    "tasks": "task_assignment",
    "discrete": "task_assignment",
    "discreto": "task_assignment",
    "delivery_routing": "delivery_routing",
    "routing": "delivery_routing",
    "tsp": "delivery_routing",
    "vrp": "delivery_routing",
    "real": "delivery_routing",
    "permutation": "delivery_routing",
}

ALGORITHM_ALIASES: Dict[str, str] = {
    "ga": "GA",
    "genetic_algorithm": "GA",
    "algoritmo_genetico": "GA",
    "pso": "PSO",
    "particle_swarm": "PSO",
    "enjambre_particulas": "PSO",
    "de": "DE",
    "differential_evolution": "DE",
    "evolucion_diferencial": "DE",
    "aco": "ACO",
    "ant_colony": "ACO",
    "colonia_hormigas": "ACO",
    "ais": "AIS",
    "clonalg": "AIS",
    "artificial_immune_system": "AIS",
    "sistema_inmune": "AIS",
}


def check_compatibility(problem_key: str, algorithm_key: str) -> Tuple[bool, str]:
    """
    Verifica si un algoritmo metaheurístico es compatible con un problema dado.

    Returns
    -------
    Tuple[bool, str]
        Tupla `(es_compatible, mensaje_explicativo)`.
    """
    norm_prob = PROBLEM_ALIASES.get(str(problem_key).strip().lower(), str(problem_key).strip().lower())
    norm_algo = ALGORITHM_ALIASES.get(str(algorithm_key).strip().lower(), str(algorithm_key).strip().upper())

    if norm_prob not in COMPATIBILITY_MATRIX:
        return False, f"Problema '{problem_key}' no registrado en COMPATIBILITY_MATRIX."

    prob_matrix = COMPATIBILITY_MATRIX[norm_prob]
    if norm_algo not in prob_matrix:
        return (
            False,
            f"El algoritmo '{algorithm_key}' no está registrado en la matriz de compatibilidad para el problema '{norm_prob}'.",
        )

    entry = prob_matrix[norm_algo]
    return bool(entry["compatible"]), str(entry["message"])


def get_compatible_algorithms(problem_key: str) -> List[str]:
    """Devuelve la lista de claves de algoritmos compatibles con el problema especificado."""
    norm_prob = PROBLEM_ALIASES.get(str(problem_key).strip().lower(), str(problem_key).strip().lower())
    if norm_prob not in COMPATIBILITY_MATRIX:
        return []
    return [
        algo_key
        for algo_key, info in COMPATIBILITY_MATRIX[norm_prob].items()
        if info.get("compatible", False)
    ]


# ============================================================================
# WRAPPER DE EVALUACIÓN (MINIMIZAR / MAXIMIZAR)
# ============================================================================

@dataclass
class EvaluationResult:
    """Resultado enriquecido de evaluar una solución mediante el wrapper de objetivo."""

    raw_value: float
    objective_value: float
    sense: OptimizationSense
    is_feasible: bool = True
    penalty: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def evaluate_with_objective(
    problem_or_fn: Union["BaseProblem", Callable[[np.ndarray], float]],
    solution: Union[Sequence[Union[int, float]], np.ndarray],
    sense: Union[str, OptimizationSense, None] = None,
    canonical_minimization: bool = False,
) -> Union[float, np.ndarray]:
    """
    Función wrapper de evaluación que permite elegir entre minimizar o maximizar cualquier problema.

    - Si `canonical_minimization=False` (por defecto), devuelve el valor de fitness real `f(x)` del
      problema.
    - Si `canonical_minimization=True`, transforma el valor para que cualquier algoritmo formulado
      internamente como minimizador pueda optimizar el problema:
      * En modo `minimize`: retorna `+f(x)` (menor es mejor).
      * En modo `maximize`: retorna `-f(x)` (menor valor transformado equivale a mayor `f(x)`).
    """
    arr = np.asarray(solution)

    if isinstance(problem_or_fn, BaseProblem):
        resolved_sense = (
            OptimizationSense.normalize(sense)
            if sense is not None
            else problem_or_fn.default_sense
        )
        if arr.ndim == 2:
            raw_val = problem_or_fn.evaluate_population_raw(arr)
        else:
            raw_val = float(problem_or_fn.evaluate_raw(arr))
    else:
        resolved_sense = (
            OptimizationSense.normalize(sense)
            if sense is not None
            else OptimizationSense.MINIMIZE
        )
        if arr.ndim == 2:
            raw_val = np.array([float(problem_or_fn(ind)) for ind in arr], dtype=float)
        else:
            raw_val = float(problem_or_fn(arr))

    if canonical_minimization and resolved_sense == OptimizationSense.MAXIMIZE:
        return -raw_val
    return raw_val


class ObjectiveWrapper:
    """
    Wrapper con estado alrededor de un `BaseProblem` que gestiona el sentido de optimización
    (minimizar o maximizar), lleva el conteo de evaluaciones de función objetivo (FE) y provee
    comparadores agnósticos al sentido (`is_better`, `best_index`, `worst_index`, `argsort`).
    """

    def __init__(
        self,
        problem: "BaseProblem",
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        self.problem = problem
        self.sense: OptimizationSense = (
            OptimizationSense.normalize(sense)
            if sense is not None
            else problem.default_sense
        )
        self.n_evaluations: int = 0

    def reset_counter(self) -> None:
        """Reinicia el contador de evaluaciones."""
        self.n_evaluations = 0

    def __call__(
        self,
        x: Union[Sequence[Union[int, float]], np.ndarray],
        canonical_minimization: bool = False,
    ) -> Union[float, np.ndarray]:
        """Evalúa una solución o población e incrementa el contador de evaluaciones."""
        arr = np.asarray(x)
        if arr.ndim == 2:
            self.n_evaluations += int(arr.shape[0])
        else:
            self.n_evaluations += 1
        return evaluate_with_objective(
            self.problem,
            arr,
            sense=self.sense,
            canonical_minimization=canonical_minimization,
        )

    def evaluate_detailed(self, x: Union[Sequence[Union[int, float]], np.ndarray]) -> EvaluationResult:
        """Evalúa una solución individual devolviendo métricas detalladas y penalizaciones."""
        arr = np.asarray(x)
        self.n_evaluations += 1
        details = self.problem.solution_details(arr)
        raw_val = float(details.get("total_fitness", self.problem.evaluate_raw(arr)))
        obj_val = -raw_val if self.sense == OptimizationSense.MAXIMIZE else raw_val
        return EvaluationResult(
            raw_value=raw_val,
            objective_value=obj_val,
            sense=self.sense,
            is_feasible=bool(details.get("is_feasible", True)),
            penalty=float(details.get("penalty", 0.0)),
            details=details,
        )

    def is_better(self, candidate_val: float, current_val: float, strict: bool = True) -> bool:
        """Compara dos valores crudos `f(x)` según el sentido configurado (`minimize` o `maximize`)."""
        if self.sense == OptimizationSense.MINIMIZE:
            return candidate_val < current_val if strict else candidate_val <= current_val
        return candidate_val > current_val if strict else candidate_val >= current_val

    def best_index(self, fitness_values: Union[Sequence[float], np.ndarray]) -> int:
        """Devuelve el índice del mejor valor en un arreglo de fitness crudo."""
        vals = np.asarray(fitness_values, dtype=float)
        if self.sense == OptimizationSense.MINIMIZE:
            return int(np.argmin(vals))
        return int(np.argmax(vals))

    def worst_index(self, fitness_values: Union[Sequence[float], np.ndarray]) -> int:
        """Devuelve el índice del peor valor en un arreglo de fitness crudo."""
        vals = np.asarray(fitness_values, dtype=float)
        if self.sense == OptimizationSense.MINIMIZE:
            return int(np.argmax(vals))
        return int(np.argmin(vals))

    def argsort(self, fitness_values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Devuelve los índices que ordenan los valores del mejor al peor según el sentido."""
        vals = np.asarray(fitness_values, dtype=float)
        order = np.argsort(vals)
        if self.sense == OptimizationSense.MAXIMIZE:
            return order[::-1]
        return order

    def compute_normalized_affinity(self, fitness_values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Convierte un vector de fitness crudo en un vector de afinidades normalizadas en [0, 1]
        donde 1.0 corresponde siempre al mejor individuo de la población y 0.0 al peor.
        Útil para selección proporcional, CLONALG (AIS) y probabilidades guiadas.
        """
        vals = np.asarray(fitness_values, dtype=float)
        f_min = float(np.min(vals))
        f_max = float(np.max(vals))
        span = f_max - f_min
        if span < 1e-12:
            return np.ones_like(vals, dtype=float)
        if self.sense == OptimizationSense.MINIMIZE:
            return (f_max - vals) / span
        return (vals - f_min) / span


# ============================================================================
# CLASE BASE ABSTRACTA PARA PROBLEMAS DE OPTIMIZACIÓN
# ============================================================================

class BaseProblem(ABC):
    """Clase base abstracta para todos los problemas de optimización en Metaheuristic Studio."""

    problem_id: str = "base"
    name: str = "Base Problem"
    description: str = ""
    representation: RepresentationType = RepresentationType.CONTINUOUS
    default_sense: OptimizationSense = OptimizationSense.MINIMIZE

    def __init__(
        self,
        dimension: int,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"La dimensión del problema debe ser mayor a 0 (recibido: {dimension}).")
        self.dimension: int = int(dimension)
        self.sense: OptimizationSense = (
            OptimizationSense.normalize(sense)
            if sense is not None
            else self.default_sense
        )

    @abstractmethod
    def evaluate_raw(self, x: np.ndarray) -> float:
        """Evalúa una única solución `x` y devuelve su valor escalar crudo `f(x)`."""

    @abstractmethod
    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Genera una solución aleatoria válida dentro del dominio del problema."""

    @abstractmethod
    def repair(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Repara o proyecta una solución candidata para garantizar su validez."""

    def evaluate_population_raw(self, population: np.ndarray) -> np.ndarray:
        """Evalúa un conjunto de soluciones de forma `(pop_size, dimension)`."""
        pop = np.asarray(population)
        if pop.ndim != 2 or pop.shape[1] != self.dimension:
            raise ValueError(
                f"Se esperaba una población de forma (N, {self.dimension}), "
                f"pero se recibió {pop.shape}."
            )
        return np.array([float(self.evaluate_raw(ind)) for ind in pop], dtype=float)

    def evaluate(
        self,
        x: Union[Sequence[Union[int, float]], np.ndarray],
        sense: Union[str, OptimizationSense, None] = None,
        canonical_minimization: bool = False,
    ) -> Union[float, np.ndarray]:
        """Evalúa una solución o población utilizando el wrapper `evaluate_with_objective`."""
        active_sense = sense if sense is not None else self.sense
        return evaluate_with_objective(
            self,
            x,
            sense=active_sense,
            canonical_minimization=canonical_minimization,
        )

    def get_objective_wrapper(
        self,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> ObjectiveWrapper:
        """Devuelve un `ObjectiveWrapper` configurado para este problema."""
        return ObjectiveWrapper(self, sense=sense if sense is not None else self.sense)

    def random_population(
        self,
        pop_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Genera una población inicial de `pop_size` individuos válidos."""
        if pop_size <= 0:
            raise ValueError("El tamaño de la población debe ser mayor a 0.")
        generator = rng if rng is not None else np.random.default_rng()
        return np.array([self.random_solution(generator) for _ in range(pop_size)])

    def is_compatible_with(self, algorithm_key: str) -> Tuple[bool, str]:
        """Consulta en `COMPATIBILITY_MATRIX` si este problema soporta el algoritmo indicado."""
        return check_compatibility(self.problem_id, algorithm_key)

    def solution_details(self, x: np.ndarray) -> Dict[str, Any]:
        """Devuelve un diccionario con el desglose detallado de la solución para visualización en UI."""
        val = float(self.evaluate_raw(np.asarray(x)))
        return {
            "total_fitness": val,
            "base_cost": val,
            "penalty": 0.0,
            "is_feasible": True,
            "sense": self.sense.value,
        }

    def get_bounds_info(self) -> Dict[str, Any]:
        """Devuelve metadatos del espacio de búsqueda y sus límites."""
        return {
            "problem_id": self.problem_id,
            "name": self.name,
            "representation": self.representation.value,
            "dimension": self.dimension,
            "default_sense": self.default_sense.value,
            "active_sense": self.sense.value,
        }


Problem = BaseProblem


# ============================================================================
# CLASE BASE ABSTRACTA PARA ALGORITMOS METAHEURÍSTICOS
# ============================================================================

class BaseMetaheuristic(ABC):
    """
    Clase base común para todos los algoritmos metaheurísticos en Metaheuristic Studio.

    Garantiza:
    - Verificación automática de compatibilidad problema-algoritmo mediante `COMPATIBILITY_MATRIX`.
    - Reproducibilidad exacta mediante `seed` (`np.random.Generator`).
    - Criterio de parada dual: máximo de iteraciones (`max_iterations`) o convergencia
      anticipada (`early_stopping_patience` iteraciones sin mejora superior a `tol`).
    - Formato de salida estandarizado con `best_solution`, `best_fitness`, `stop_reason`,
      `total_iterations` y `history` (lista con `iteration`, `best_fitness`, `best_solution`,
      `mean_fitness`).
    """

    algorithm_id: str = "BASE"
    name: str = "Base Metaheuristic"
    description: str = ""

    def __init__(
        self,
        max_iterations: int = 100,
        population_size: int = 40,
        seed: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        tol: float = 1e-8,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations debe ser mayor a 0.")
        if population_size <= 0:
            raise ValueError("population_size debe ser mayor a 0.")

        self.max_iterations: int = int(max_iterations)
        self.population_size: int = int(population_size)
        self.seed: Optional[int] = int(seed) if seed is not None else None
        self.early_stopping_patience: Optional[int] = (
            int(early_stopping_patience)
            if early_stopping_patience is not None and int(early_stopping_patience) > 0
            else None
        )
        self.tol: float = float(tol)
        self.sense: Optional[OptimizationSense] = (
            OptimizationSense.normalize(sense) if sense is not None else None
        )
        self._step_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._stop_check: Optional[Callable[[], bool]] = None
        self._user_stopped_flag: bool = False

    def _create_rng(self, override_seed: Optional[int] = None) -> np.random.Generator:
        """Crea un generador pseudoaleatorio reproducible."""
        active_seed = override_seed if override_seed is not None else self.seed
        return np.random.default_rng(active_seed)

    def validate_compatibility(self, problem: BaseProblem) -> None:
        """Valida en `COMPATIBILITY_MATRIX` que el algoritmo soporte el problema dado."""
        is_ok, msg = check_compatibility(problem.problem_id, self.algorithm_id)
        if not is_ok:
            raise ValueError(
                f"Incompatibilidad detectada [{self.algorithm_id} -> {problem.problem_id}]: {msg}"
            )

    def _record_history_step(
        self,
        history: List[Dict[str, Any]],
        iteration: int,
        best_fitness: float,
        best_solution: np.ndarray,
        population_fitness: np.ndarray,
    ) -> None:
        """Registra una entrada estándar en el historial de convergencia por iteración."""
        entry = {
            "iteration": int(iteration),
            "best_fitness": float(best_fitness),
            "best_solution": np.asarray(best_solution).copy().tolist(),
            "mean_fitness": float(np.mean(population_fitness)),
        }
        history.append(entry)
        if self._step_callback is not None:
            self._step_callback(entry)

    def _check_early_stopping(
        self,
        improved: bool,
        improvement_amount: float,
        stagnation_counter: int,
    ) -> Tuple[int, bool]:
        """
        Actualiza el contador de estancamiento y determina si se cumple la parada por
        convergencia anticipada o interrupción manual del usuario.
        """
        if self._stop_check is not None and self._stop_check():
            self._user_stopped_flag = True
            return stagnation_counter, True

        if improved and improvement_amount > self.tol:
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if (
            self.early_stopping_patience is not None
            and stagnation_counter >= self.early_stopping_patience
        ):
            return stagnation_counter, True
        return stagnation_counter, False

    @abstractmethod
    def _optimize(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        """
        Método interno que ejecuta el bucle principal de la metaheurística.

        Returns
        -------
        Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]
            `(best_solution, best_fitness, stop_reason, total_iterations, history)`
        """

    def run(
        self,
        problem: BaseProblem,
        sense: Union[str, OptimizationSense, None] = None,
        seed: Optional[int] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el algoritmo metaheurístico sobre el `problem` indicado y devuelve el
        diccionario estándar de resultados.
        """
        self.validate_compatibility(problem)
        active_sense = (
            OptimizationSense.normalize(sense)
            if sense is not None
            else (self.sense if self.sense is not None else problem.sense)
        )
        wrapper = problem.get_objective_wrapper(sense=active_sense)
        rng = self._create_rng(override_seed=seed)

        self._step_callback = step_callback
        self._stop_check = stop_check
        self._user_stopped_flag = False

        start_time = time.perf_counter()
        try:
            best_sol, best_fit, stop_reason, total_iters, history = self._optimize(
                problem=problem,
                wrapper=wrapper,
                rng=rng,
            )
        finally:
            self._step_callback = None
            self._stop_check = None

        elapsed = time.perf_counter() - start_time
        if self._user_stopped_flag:
            stop_reason = "user_stopped (detenido manualmente por el usuario)"

        best_sol_arr = problem.repair(np.asarray(best_sol))
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.name,
            "problem_id": problem.problem_id,
            "sense": wrapper.sense.value,
            "best_solution": best_sol_arr.copy(),
            "best_fitness": float(best_fit),
            "stop_reason": str(stop_reason),
            "total_iterations": int(total_iters),
            "n_evaluations": int(wrapper.n_evaluations),
            "execution_time_sec": float(elapsed),
            "history": history,
            "solution_details": problem.solution_details(best_sol_arr),
        }

    # Alias conveniente
    solve = run

    def get_params(self) -> Dict[str, Any]:
        """Devuelve un diccionario con los hiperparámetros configurados del algoritmo."""
        return {
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "max_iterations": self.max_iterations,
            "population_size": self.population_size,
            "seed": self.seed,
            "early_stopping_patience": self.early_stopping_patience,
            "tol": self.tol,
        }


MetaheuristicAlgorithm = BaseMetaheuristic
