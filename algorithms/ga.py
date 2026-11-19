"""
Algoritmo Genético (GA) multi-representación para Metaheuristic Studio.

Soporta los 4 dominios de problemas:
- Continuo (Ackley): Cruza BLX-alpha (Blended Crossover) + mutación gaussiana adaptativa.
- Binario (OneMax): Cruza de dos puntos / uniforme + mutación bit-flip.
- Discreto (Asignación de Tareas): Cruza uniforme/dos puntos + mutación de reasignación aleatoria.
- Permutación (Ruteo Guadalajara): Cruza de Orden (OX1 - Order Crossover) + mutación swap y 2-opt.
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


class GeneticAlgorithm(BaseMetaheuristic):
    """
    Algoritmo Genético (GA) con selección por torneo, elitismo y operadores de
    cruza y mutación especializados según la representación del problema.
    """

    algorithm_id: str = "GA"
    name: str = "Algoritmo Genético (GA)"
    description: str = (
        "Metaheurística evolutiva basada en selección natural, cruza recombinativa "
        "y mutación adaptada al dominio (continuo, binario, discreto o permutación)."
    )

    def __init__(
        self,
        max_iterations: int = 100,
        population_size: int = 50,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elitism_count: int = 2,
        blx_alpha: float = 0.35,
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
        self.crossover_rate: float = float(np.clip(crossover_rate, 0.0, 1.0))
        self.mutation_rate: float = float(np.clip(mutation_rate, 0.0, 1.0))
        self.tournament_size: int = max(2, int(tournament_size))
        self.elitism_count: int = max(1, min(int(elitism_count), self.population_size - 1))
        self.blx_alpha: float = float(max(0.0, blx_alpha))

    # -------------------------------------------------------------------------
    # Operadores de Selección
    # -------------------------------------------------------------------------
    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Selecciona un individuo padre mediante torneo de tamaño `tournament_size`."""
        k = min(self.tournament_size, len(population))
        contenders = rng.choice(len(population), size=k, replace=False)
        best_local = wrapper.best_index(fitness[contenders])
        return population[contenders[best_local]].copy()

    # -------------------------------------------------------------------------
    # Operadores de Cruza (Crossover) por Representación
    # -------------------------------------------------------------------------
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        problem: BaseProblem,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica el operador de cruza apropiado según `problem.representation`."""
        if rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        rep = problem.representation
        dim = problem.dimension

        if rep == RepresentationType.CONTINUOUS:
            # Cruza BLX-alpha para vectores reales
            c_min = np.minimum(parent1, parent2)
            c_max = np.maximum(parent1, parent2)
            diff = c_max - c_min
            low = c_min - self.blx_alpha * diff
            high = c_max + self.blx_alpha * diff
            child1 = problem.repair(rng.uniform(low, high))
            child2 = problem.repair(rng.uniform(low, high))
            return child1, child2

        if rep in (RepresentationType.BINARY, RepresentationType.DISCRETE):
            if dim <= 2 or rng.random() < 0.5:
                # Cruza uniforme
                mask = rng.random(dim) < 0.5
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
            else:
                # Cruza de dos puntos
                pt1, pt2 = np.sort(rng.choice(dim, size=2, replace=False))
                child1 = parent1.copy()
                child2 = parent2.copy()
                child1[pt1:pt2] = parent2[pt1:pt2]
                child2[pt1:pt2] = parent1[pt1:pt2]
            return child1.astype(int), child2.astype(int)

        if rep == RepresentationType.PERMUTATION:
            # Order Crossover (OX1) para permutaciones de clientes
            return (
                self._order_crossover(parent1, parent2, rng),
                self._order_crossover(parent2, parent1, rng),
            )

        return parent1.copy(), parent2.copy()

    @staticmethod
    def _order_crossover(
        p1: np.ndarray,
        p2: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Operador Order Crossover (OX1) que preserva la validez de permutaciones."""
        size = p1.size
        start, end = np.sort(rng.choice(size, size=2, replace=False))
        child = np.full(size, -1, dtype=int)
        child[start : end + 1] = p1[start : end + 1]

        in_segment = set(child[start : end + 1].tolist())
        p2_filtered = [int(gene) for gene in p2 if int(gene) not in in_segment]

        fill_idx = 0
        for pos in range(size):
            if child[pos] == -1:
                child[pos] = p2_filtered[fill_idx]
                fill_idx += 1
        return child

    # -------------------------------------------------------------------------
    # Operadores de Mutación por Representación
    # -------------------------------------------------------------------------
    def _mutate(
        self,
        individual: np.ndarray,
        problem: BaseProblem,
        progress: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Aplica mutación específica según el tipo de representación del problema."""
        rep = problem.representation
        dim = problem.dimension
        mutated = individual.copy()

        if rep == RepresentationType.CONTINUOUS:
            # Mutación gaussiana con desviación estándar decreciente conforme avanza la búsqueda
            gene_mask = rng.random(dim) < max(self.mutation_rate, 1.0 / dim)
            if np.any(gene_mask) or rng.random() < self.mutation_rate:
                if not np.any(gene_mask):
                    gene_mask[rng.integers(0, dim)] = True
                lower = getattr(problem, "lower_bound", -32.768)
                upper = getattr(problem, "upper_bound", 32.768)
                span = float(upper - lower)
                sigma = span * (0.12 * (1.0 - 0.85 * progress) + 0.005)
                noise = rng.normal(0.0, sigma, size=int(np.sum(gene_mask)))
                mutated = mutated.astype(float)
                mutated[gene_mask] += noise
                mutated = problem.repair(mutated)
            return mutated

        if rep == RepresentationType.BINARY:
            # Mutación bit-flip por gen con tasa `mutation_rate / dim` o activación por individuo
            p_bit = max(1.0 / dim, self.mutation_rate * 0.25)
            if rng.random() < self.mutation_rate:
                flips = rng.random(dim) < p_bit
                if not np.any(flips):
                    flips[rng.integers(0, dim)] = True
                mutated[flips] = 1 - mutated[flips]
            return mutated.astype(int)

        if rep == RepresentationType.DISCRETE:
            n_res = getattr(problem, "n_resources", 2)
            if rng.random() < self.mutation_rate:
                p_gene = max(1.0 / dim, self.mutation_rate * 0.3)
                mask = rng.random(dim) < p_gene
                if not np.any(mask):
                    mask[rng.integers(0, dim)] = True
                mutated[mask] = rng.integers(0, n_res, size=int(np.sum(mask)))
            return mutated.astype(int)

        if rep == RepresentationType.PERMUTATION:
            if rng.random() < self.mutation_rate:
                i, j = np.sort(rng.choice(dim, size=2, replace=False))
                if rng.random() < 0.5:
                    # Mutación Swap (intercambio de dos clientes)
                    mutated[i], mutated[j] = mutated[j], mutated[i]
                else:
                    # Mutación 2-opt (inversión de sub-ruta)
                    mutated[i : j + 1] = mutated[i : j + 1][::-1]
            return mutated.astype(int)

        return problem.repair(mutated)

    # -------------------------------------------------------------------------
    # Bucle Evolutivo Principal
    # -------------------------------------------------------------------------
    def _optimize(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        population = problem.random_population(self.population_size, rng=rng)
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

        for it in range(1, self.max_iterations + 1):
            total_iterations = it
            progress = float(it - 1) / max(1.0, float(self.max_iterations - 1))

            # 1. Elitismo: conservar los mejores individuos de la generación actual
            sorted_indices = wrapper.argsort(fitness)
            elites = population[sorted_indices[: self.elitism_count]].copy()

            # 2. Reproducción: selección por torneo, cruza y mutación
            offspring_list: List[np.ndarray] = [ind for ind in elites]
            while len(offspring_list) < self.population_size:
                p1 = self._tournament_selection(population, fitness, wrapper, rng)
                p2 = self._tournament_selection(population, fitness, wrapper, rng)
                c1, c2 = self._crossover(p1, p2, problem, rng)
                c1 = self._mutate(c1, problem, progress, rng)
                offspring_list.append(c1)
                if len(offspring_list) < self.population_size:
                    c2 = self._mutate(c2, problem, progress, rng)
                    offspring_list.append(c2)

            population = np.array(offspring_list[: self.population_size])
            fitness = np.asarray(wrapper(population), dtype=float)

            # 3. Actualizar mejor global
            gen_best_idx = wrapper.best_index(fitness)
            gen_best_fit = float(fitness[gen_best_idx])
            improved = wrapper.is_better(gen_best_fit, best_fitness, strict=True)
            improvement_amount = abs(gen_best_fit - best_fitness) if improved else 0.0

            if improved:
                best_fitness = gen_best_fit
                best_solution = population[gen_best_idx].copy()

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
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "tournament_size": self.tournament_size,
                "elitism_count": self.elitism_count,
                "blx_alpha": self.blx_alpha,
            }
        )
        return params

