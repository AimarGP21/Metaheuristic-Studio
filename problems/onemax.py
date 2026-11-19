"""
Problema Binario: OneMax.

Fórmula matemática para un vector binario x in {0, 1}^N:
    f(x) = sum_{i=1}^{N} x_i

Óptimo global natural (Maximización):
    x* = (1, 1, ..., 1) con f(x*) = N.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from core.base import BaseProblem, OptimizationSense, RepresentationType


class OneMaxProblem(BaseProblem):
    """
    Implementación del problema binario OneMax con longitud de vector N configurable.
    """

    problem_id: str = "onemax"
    name: str = "Problema OneMax (Binario)"
    description: str = (
        "Maximiza (o minimiza) la suma de bits encendidos (1s) en una cadena binaria "
        "de longitud N configurable."
    )
    representation: RepresentationType = RepresentationType.BINARY
    default_sense: OptimizationSense = OptimizationSense.MAXIMIZE

    def __init__(
        self,
        n_bits: int = 50,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        super().__init__(dimension=n_bits, sense=sense)
        self.n_bits: int = int(n_bits)

    def evaluate_raw(self, x: np.ndarray) -> float:
        """Calcula el número total de bits con valor 1 en el vector binario."""
        vec = self.repair(x)
        if vec.size != self.n_bits:
            raise ValueError(
                f"Longitud incorrecta para OneMax: se esperaba N={self.n_bits}, "
                f"se recibió {vec.size}."
            )
        return float(np.sum(vec))

    def evaluate_population_raw(self, population: np.ndarray) -> np.ndarray:
        """Evaluación vectorizada de una población binaria de forma (pop_size, N)."""
        pop = np.asarray(population)
        if pop.ndim != 2 or pop.shape[1] != self.n_bits:
            raise ValueError(
                f"Se esperaba matriz de población (M, {self.n_bits}), se recibió {pop.shape}."
            )
        bin_pop = (np.rint(np.clip(pop, 0, 1))).astype(int)
        return np.sum(bin_pop, axis=1).astype(float)

    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Genera un vector aleatorio de 0s y 1s de longitud N."""
        generator = rng if rng is not None else np.random.default_rng()
        return generator.integers(0, 2, size=self.n_bits, dtype=int)

    def repair(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Convierte cualquier vector numérico al dominio binario estricto {0, 1}^N."""
        arr = np.asarray(x).ravel()
        return np.rint(np.clip(arr, 0, 1)).astype(int)

    def solution_details(self, x: np.ndarray) -> Dict[str, Any]:
        """Devuelve estadísticas de proporción de unos y ceros para la UI."""
        vec = self.repair(x)
        ones_count = int(np.sum(vec))
        zeros_count = int(self.n_bits - ones_count)
        ratio = float(ones_count / self.n_bits) if self.n_bits > 0 else 0.0
        return {
            "total_fitness": float(ones_count),
            "base_cost": float(ones_count),
            "penalty": 0.0,
            "is_feasible": True,
            "sense": self.sense.value,
            "n_bits": self.n_bits,
            "ones_count": ones_count,
            "zeros_count": zeros_count,
            "ones_ratio": ratio,
            "hamming_distance_to_all_ones": zeros_count,
        }

    def get_bounds_info(self) -> Dict[str, Any]:
        info = super().get_bounds_info()
        info.update(
            {
                "n_bits": self.n_bits,
                "domain": [0, 1],
                "global_max_value": float(self.n_bits),
                "global_min_value": 0.0,
            }
        )
        return info

