"""
Problema Continuo: Función Multimodal de Ackley.

Fórmula matemática en dimensión D:
    f(x) = -a * exp(-b * sqrt((1/D) * sum(x_i^2)))
           - exp((1/D) * sum(cos(c * x_i)))
           + a + exp(1)

Óptimo global:
    x* = (0, 0, ..., 0) con f(x*) = 0.0 (en minimización).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from core.base import BaseProblem, OptimizationSense, RepresentationType


class AckleyProblem(BaseProblem):
    """
    Implementación de la función de prueba continua de Ackley con dimensión D
    y límites [lower_bound, upper_bound] configurables.
    """

    problem_id: str = "ackley"
    name: str = "Función de Ackley (Continuo)"
    description: str = (
        "Función multimodal no convexa continua con múltiples óptimos locales "
        "y un único mínimo global en el origen f(0,...,0) = 0."
    )
    representation: RepresentationType = RepresentationType.CONTINUOUS
    default_sense: OptimizationSense = OptimizationSense.MINIMIZE

    def __init__(
        self,
        dimension: int = 10,
        lower_bound: float = -32.768,
        upper_bound: float = 32.768,
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2.0 * np.pi,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        super().__init__(dimension=dimension, sense=sense)
        if lower_bound >= upper_bound:
            raise ValueError(
                f"El límite inferior ({lower_bound}) debe ser estrictamente menor "
                f"que el límite superior ({upper_bound})."
            )
        self.lower_bound: float = float(lower_bound)
        self.upper_bound: float = float(upper_bound)
        self.a: float = float(a)
        self.b: float = float(b)
        self.c: float = float(c)

    @property
    def bounds(self) -> Tuple[float, float]:
        """Devuelve la tupla (min, max) del espacio de búsqueda."""
        return (self.lower_bound, self.upper_bound)

    @property
    def bounds_array(self) -> np.ndarray:
        """Devuelve una matriz de forma (D, 2) con los límites por dimensión."""
        return np.tile([self.lower_bound, self.upper_bound], (self.dimension, 1))

    def evaluate_raw(self, x: np.ndarray) -> float:
        """Evalúa un vector continuo x de longitud D en la función de Ackley."""
        vec = np.asarray(x, dtype=float).ravel()
        if vec.size != self.dimension:
            raise ValueError(
                f"Dimensión incorrecta para Ackley: se esperaba D={self.dimension}, "
                f"se recibió {vec.size}."
            )
        d = float(self.dimension)
        sum_sq = np.sum(vec ** 2)
        sum_cos = np.sum(np.cos(self.c * vec))

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        result = term1 + term2 + self.a + np.e
        # Evitar errores de redondeo cercanos a 0
        return float(max(0.0, result)) if result < 1e-15 and result > -1e-15 else float(result)

    def evaluate_population_raw(self, population: np.ndarray) -> np.ndarray:
        """Evaluación vectorizada de una población de tamaño (N, D)."""
        pop = np.asarray(population, dtype=float)
        if pop.ndim != 2 or pop.shape[1] != self.dimension:
            raise ValueError(
                f"Se esperaba matriz de población (N, {self.dimension}), se recibió {pop.shape}."
            )
        d = float(self.dimension)
        sum_sq = np.sum(pop ** 2, axis=1)
        sum_cos = np.sum(np.cos(self.c * pop), axis=1)

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        res = term1 + term2 + self.a + np.e
        return np.where(np.abs(res) < 1e-15, 0.0, res)

    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Genera un vector continuo uniforme dentro de [lower_bound, upper_bound]^D."""
        generator = rng if rng is not None else np.random.default_rng()
        return generator.uniform(self.lower_bound, self.upper_bound, size=self.dimension)

    def repair(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Recorta (clip) las coordenadas de `x` al hiper-rectángulo [lower_bound, upper_bound]^D."""
        return np.clip(np.asarray(x, dtype=float), self.lower_bound, self.upper_bound)

    def solution_details(self, x: np.ndarray) -> Dict[str, Any]:
        """Devuelve métricas detalladas de la solución continua respecto al origen."""
        vec = np.asarray(x, dtype=float).ravel()
        val = self.evaluate_raw(vec)
        in_bounds = bool(
            np.all(vec >= self.lower_bound - 1e-9) and np.all(vec <= self.upper_bound + 1e-9)
        )
        l2_dist = float(np.linalg.norm(vec))
        return {
            "total_fitness": val,
            "base_cost": val,
            "penalty": 0.0,
            "is_feasible": in_bounds,
            "sense": self.sense.value,
            "dimension": self.dimension,
            "bounds": [self.lower_bound, self.upper_bound],
            "l2_distance_to_optimum": l2_dist,
            "max_abs_coordinate": float(np.max(np.abs(vec))),
            "mean_coordinate": float(np.mean(vec)),
        }

    def generate_surface_mesh(
        self,
        resolution: int = 80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera una malla 2D (X, Y, Z) de la función de Ackley para visualización 3D / contorno en Plotly.
        """
        axis = np.linspace(self.lower_bound, self.upper_bound, resolution)
        xx, yy = np.meshgrid(axis, axis)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        sum_sq = np.sum(grid_points ** 2, axis=1)
        sum_cos = np.sum(np.cos(self.c * grid_points), axis=1)
        zz = (
            -self.a * np.exp(-self.b * np.sqrt(sum_sq / 2.0))
            - np.exp(sum_cos / 2.0)
            + self.a
            + np.e
        ).reshape(xx.shape)
        return xx, yy, zz

    def get_bounds_info(self) -> Dict[str, Any]:
        info = super().get_bounds_info()
        info.update(
            {
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "global_optimum_value": 0.0,
                "global_optimum_position": np.zeros(self.dimension).tolist(),
            }
        )
        return info

