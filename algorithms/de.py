"""
Evolución Diferencial (DE/rand/1/bin) para Metaheuristic Studio.

Esquema clásico de Storn & Price (1997) para optimización continua:
1. Mutación diferencial (rand/1):
   v_i = x_{r1} + F * (x_{r2} - x_{r3}), con r1 != r2 != r3 != i.
2. Cruza binomial (bin):
   u_{i,j} = v_{i,j} si rand(0,1) < CR o j == j_rand, sino x_{i,j}.
3. Selección codiciosa (1-a-1):
   x_i^{(t+1)} = u_i si u_i es igual o mejor que x_i^{(t)}, sino x_i^{(t)}.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.base import (
    BaseMetaheuristic,
    BaseProblem,
    ObjectiveWrapper,
    OptimizationSense,
    RepresentationType,
)


class DifferentialEvolution(BaseMetaheuristic):
    """
    Evolución Diferencial con estrategia DE/rand/1/bin para problemas continuos.
    """

    algorithm_id: str = "DE"
    name: str = "Evolución Diferencial (DE/rand/1/bin)"
    description: str = (
        "Algoritmo evolutivo para espacios continuos basado en diferencias vectoriales "
        "escaladas (factor F) y recombinación binomial (tasa CR)."
    )

    def __init__(
        self,
        max_iterations: int = 100,
        population_size: int = 40,
        mutation_factor: float = 0.7,
        crossover_rate: float = 0.85,
        seed: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        tol: float = 1e-8,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        if population_size < 4:
            raise ValueError("DE/rand/1/bin requiere al menos population_size >= 4.")
        super().__init__(
            max_iterations=max_iterations,
            population_size=population_size,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            tol=tol,
            sense=sense,
        )
        self.mutation_factor: float = float(np.clip(mutation_factor, 0.05, 2.0))
        self.crossover_rate: float = float(np.clip(crossover_rate, 0.0, 1.0))

    def _optimize(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        if problem.representation != RepresentationType.CONTINUOUS:
            raise ValueError(
                f"DE/rand/1/bin solo soporta problemas continuos (recibido: {problem.representation})."
            )

        np_size = self.population_size
        dim = problem.dimension
        lower = float(getattr(problem, "lower_bound", -32.768))
        upper = float(getattr(problem, "upper_bound", 32.768))

        population = problem.random_population(np_size, rng=rng).astype(float)
        fitness = np.asarray(wrapper(population), dtype=float)

        best_idx = wrapper.best_index(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        history: List[Dict[str, Any]] = []
        # Registrar Hito Inicial real (Iteración 0: población inicial aleatoria evaluada)
        self._record_history_step(
            history=history,
            iteration=0,
            best_fitness=best_fitness,
            best_solution=best_solution,
            population_fitness=fitness,
        )

        stagnation_counter = 0
        stop_reason = "max_iterations"
        total_iterations = 0

        all_indices = np.arange(np_size)

        for it in range(1, self.max_iterations + 1):
            total_iterations = it
            trials = np.empty_like(population)

            for i in range(np_size):
                # Elegir 3 índices distintos r1, r2, r3 != i
                candidates = all_indices[all_indices != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Vector mutante (donor): v_i = x_r1 + F * (x_r2 - x_r3)
                mutant = population[r1] + self.mutation_factor * (population[r2] - population[r3])
                mutant = np.clip(mutant, lower, upper)

                # Cruza binomial con garantía de al menos una coordenada mutada (j_rand)
                cross_mask = rng.random(dim) < self.crossover_rate
                j_rand = int(rng.integers(0, dim))
                cross_mask[j_rand] = True

                trials[i] = np.where(cross_mask, mutant, population[i])

            # Evaluar vectores de prueba en lote
            trial_fitness = np.asarray(wrapper(trials), dtype=float)

            # Selección codiciosa 1 a 1
            for i in range(np_size):
                if wrapper.is_better(float(trial_fitness[i]), float(fitness[i]), strict=False):
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]

            # Actualizar mejor global
            iter_best_idx = wrapper.best_index(fitness)
            iter_best_fit = float(fitness[iter_best_idx])
            improved = wrapper.is_better(iter_best_fit, best_fitness, strict=True)
            improvement_amount = abs(iter_best_fit - best_fitness) if improved else 0.0

            if improved:
                best_fitness = iter_best_fit
                best_solution = population[iter_best_idx].copy()

            self._record_history_step(
                history=history,
                iteration=it,
                best_fitness=best_fitness,
                best_solution=best_solution,
                population_fitness=fitness,
            )

            stagnation_counter, should_stop = self._check_early_stopping(
                improved=improved,
                improvement_amount=improvement_amount,
                stagnation_counter=stagnation_counter,
            )
            if should_stop:
                stop_reason = f"early_convergence (sin mejora por {self.early_stopping_patience} iteraciones)"
                break

        return best_solution, best_fitness, stop_reason, total_iterations, history

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "mutation_factor": self.mutation_factor,
                "crossover_rate": self.crossover_rate,
                "strategy": "DE/rand/1/bin",
            }
        )
        return params

