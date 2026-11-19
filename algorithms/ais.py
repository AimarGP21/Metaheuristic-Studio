"""
Sistema Inmune Artificial (AIS - Algoritmo de Selección Clonal CLONALG) para Metaheuristic Studio.

Basado en el algoritmo CLONALG (De Castro & Von Zuben, 2002), adaptado para:
- Continuo (Ackley): Hipermutación somática gaussiana con varianza inversamente proporcional
  a la afinidad antigénica normalizada A in [0, 1].
- Binario (OneMax): Hipermutación bit-flip con tasa p = exp(-rho * A).
- Discreto (Asignación de Tareas): Reasignación proporcional de recursos según exp(-rho * A).

Fases del ciclo inmune por generación:
1. Evaluación de afinidad de la población de anticuerpos.
2. Selección clonal de los `n_select` anticuerpos con mayor afinidad.
3. Expansión clonal proporcional al rango de afinidad (`clone_factor` beta).
4. Maduración de afinidad mediante hipermutación somática inversamente proporcional (`exp(-rho * A)`).
5. Selección de memoria clonal y edición de receptores (`replace_rate` d) con nuevos anticuerpos.
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


class ArtificialImmuneSystem(BaseMetaheuristic):
    """
    Sistema Inmune Artificial basado en Selección Clonal (CLONALG) para problemas
    continuos, binarios y discretos.
    """

    algorithm_id: str = "AIS"
    name: str = "Sistema Inmune Artificial (AIS - CLONALG)"
    description: str = (
        "Algoritmo bio-inspirado en la teoría de selección clonal del sistema inmune adquirido: "
        "clona los mejores anticuerpos, aplica hipermutación inversamente proporcional a la "
        "afinidad y renueva los peores receptores."
    )

    def __init__(
        self,
        max_iterations: int = 100,
        population_size: int = 40,
        selection_ratio: float = 0.5,
        clone_factor: float = 0.6,
        mutation_decay: float = 2.5,
        replace_rate: float = 0.15,
        seed: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        tol: float = 1e-8,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations,
            population_size=population_size,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            tol=tol,
            sense=sense,
        )
        self.selection_ratio: float = float(np.clip(selection_ratio, 0.1, 1.0))
        self.clone_factor: float = float(np.clip(clone_factor, 0.1, 3.0))
        self.mutation_decay: float = float(max(0.2, mutation_decay))
        self.replace_rate: float = float(np.clip(replace_rate, 0.0, 0.5))

    def _hypermutate(
        self,
        clone: np.ndarray,
        affinity: float,
        problem: BaseProblem,
        progress: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Aplica hipermutación somática inversamente proporcional a la afinidad normalizada `affinity` in [0, 1].
        Anticuerpos con alta afinidad (cercanos a 1.0) sufren mutaciones finas locales;
        anticuerpos con baja afinidad (cercanos a 0.0) sufren mutaciones intensas de exploración.
        """
        rep = problem.representation
        dim = problem.dimension
        mutated = clone.copy()

        # Tasa de mutación decae exponencialmente con la afinidad: alpha = exp(-rho * A)
        alpha_rate = float(np.exp(-self.mutation_decay * np.clip(affinity, 0.0, 1.0)))

        if rep == RepresentationType.CONTINUOUS:
            lower = float(getattr(problem, "lower_bound", -32.768))
            upper = float(getattr(problem, "upper_bound", 32.768))
            span = upper - lower

            # Enfriamiento suave a lo largo de las iteraciones + intensidad por afinidad
            cooling = 1.0 - 0.80 * progress
            sigma = span * 0.18 * alpha_rate * cooling + 1e-4
            p_coord = np.clip(0.25 + 0.65 * alpha_rate, 1.0 / dim, 1.0)
            mask = rng.random(dim) < p_coord
            if not np.any(mask):
                mask[rng.integers(0, dim)] = True

            mutated = mutated.astype(float)
            mutated[mask] += rng.normal(0.0, sigma, size=int(np.sum(mask)))
            return problem.repair(mutated)

        if rep == RepresentationType.BINARY:
            p_flip = float(np.clip(alpha_rate * 0.35, 1.0 / dim, 0.45))
            flips = rng.random(dim) < p_flip
            if not np.any(flips):
                flips[rng.integers(0, dim)] = True
            mutated[flips] = 1 - mutated[flips]
            return mutated.astype(int)

        if rep == RepresentationType.DISCRETE:
            n_res = int(getattr(problem, "n_resources", 2))
            p_mut = float(np.clip(alpha_rate * 0.40, 1.0 / dim, 0.50))
            mask = rng.random(dim) < p_mut
            if not np.any(mask):
                mask[rng.integers(0, dim)] = True
            mutated[mask] = rng.integers(0, n_res, size=int(np.sum(mask)))
            return mutated.astype(int)

        raise ValueError(f"Representación '{rep}' no soportada por AIS (CLONALG).")

    def _optimize(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        n_pop = self.population_size
        n_select = max(2, int(round(self.selection_ratio * n_pop)))
        n_replace = int(round(self.replace_rate * n_pop))

        # Repertorio inicial de anticuerpos
        repertoire = problem.random_population(n_pop, rng=rng)
        fitness = np.asarray(wrapper(repertoire), dtype=float)

        best_idx = wrapper.best_index(fitness)
        best_solution = repertoire[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        history: List[Dict[str, Any]] = []
        # Registrar Hito Inicial real (Iteración 0: repertorio inmune inicial evaluado)
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

        for it in range(1, self.max_iterations + 1):
            total_iterations = it
            progress = float(it - 1) / max(1.0, float(self.max_iterations - 1))

            # 1. Ordenar anticuerpos por afinidad y calcular afinidad normalizada A in [0, 1]
            sorted_idx = wrapper.argsort(fitness)
            affinities = wrapper.compute_normalized_affinity(fitness)
            selected_idx = sorted_idx[:n_select]

            # 2. Expansión Clonal e Hipermutación Somática proporcional al rango de afinidad
            matured_clones: List[np.ndarray] = []
            for rank_zero, ab_idx in enumerate(selected_idx):
                rank = rank_zero + 1  # rank = 1 es el mejor anticuerpo
                n_clones = max(1, int(round((self.clone_factor * n_pop) / rank)))
                parent_ab = repertoire[ab_idx]
                ab_affinity = float(affinities[ab_idx])

                for _ in range(n_clones):
                    mut_clone = self._hypermutate(
                        clone=parent_ab,
                        affinity=ab_affinity,
                        problem=problem,
                        progress=progress,
                        rng=rng,
                    )
                    matured_clones.append(mut_clone)

            clones_arr = np.array(matured_clones)
            clones_fit = np.asarray(wrapper(clones_arr), dtype=float)

            # 3. Selección de Memoria Clonal: combinar repertorio actual + clones madurados
            combined_pop = np.vstack([repertoire, clones_arr])
            combined_fit = np.concatenate([fitness, clones_fit])
            best_combined_order = wrapper.argsort(combined_fit)

            n_keep = n_pop - n_replace
            survivors_idx = best_combined_order[:n_keep]
            new_repertoire = combined_pop[survivors_idx].copy()
            new_fitness = combined_fit[survivors_idx].copy()

            # 4. Edición de Receptores: introducir `n_replace` nuevos anticuerpos aleatorios
            if n_replace > 0:
                fresh_antibodies = problem.random_population(n_replace, rng=rng)
                fresh_fitness = np.asarray(wrapper(fresh_antibodies), dtype=float)
                repertoire = np.vstack([new_repertoire, fresh_antibodies])
                fitness = np.concatenate([new_fitness, fresh_fitness])
            else:
                repertoire = new_repertoire
                fitness = new_fitness

            # 5. Actualizar mejor anticuerpo global
            iter_best_idx = wrapper.best_index(fitness)
            iter_best_fit = float(fitness[iter_best_idx])
            improved = wrapper.is_better(iter_best_fit, best_fitness, strict=True)
            improvement_amount = abs(iter_best_fit - best_fitness) if improved else 0.0

            if improved:
                best_fitness = iter_best_fit
                best_solution = repertoire[iter_best_idx].copy()

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
                "selection_ratio": self.selection_ratio,
                "clone_factor": self.clone_factor,
                "mutation_decay": self.mutation_decay,
                "replace_rate": self.replace_rate,
            }
        )
        return params

