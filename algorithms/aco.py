"""
Optimización por Colonia de Hormigas (ACO) para Metaheuristic Studio.

Adaptado a tres representaciones combinatorias:
1. Binario (OneMax):
   Grafo de cadena binaria con matriz de feromonas tau[i, b] para i in {0..N-1}, b in {0, 1}.
2. Discreto (Asignación de Tareas a Recursos):
   Grafo bipartito Tarea-Recurso con matriz de feromonas tau[t, r] y visibilidad heurística
   eta[t, r] inversamente proporcional al costo C[t, r] y consciente de la capacidad restante.
3. Permutaciones de Rutas (Ruteo Guadalajara):
   Grafo completo de N+1 nodos (CEDIS + N clientes) con feromona de arco tau[u, v],
   visibilidad eta[u, v] = 1 / distancia_km(u, v) y refinamiento local 2-opt ligero en la
   hormiga élite de cada iteración.
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


class AntColonyOptimization(BaseMetaheuristic):
    """
    Colonia de Hormigas (ACO) con evaporación y refuerzo elitista de feromonas para
    problemas binarios, discretos de asignación y permutaciones de ruteo.
    """

    algorithm_id: str = "ACO"
    name: str = "Colonia de Hormigas (ACO)"
    description: str = (
        "Metaheurística constructiva inspirada en el rastro de feromonas de hormigas reales; "
        "combina memoria colectiva (tau^alpha) y visibilidad heurística (eta^beta)."
    )

    def __init__(
        self,
        max_iterations: int = 100,
        population_size: int = 35,
        alpha: float = 1.0,
        beta: float = 2.2,
        evaporation_rate: float = 0.18,
        pheromone_deposit: float = 1.5,
        elitist_weight: float = 2.0,
        tau_min: float = 0.01,
        tau_max: float = 25.0,
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
        self.alpha: float = float(max(0.0, alpha))
        self.beta: float = float(max(0.0, beta))
        self.evaporation_rate: float = float(np.clip(evaporation_rate, 0.01, 0.95))
        self.pheromone_deposit: float = float(max(0.1, pheromone_deposit))
        self.elitist_weight: float = float(max(0.0, elitist_weight))
        self.tau_min: float = float(max(1e-5, tau_min))
        self.tau_max: float = float(max(self.tau_min + 1.0, tau_max))

    # -------------------------------------------------------------------------
    # 1. Construcción y actualización para PROBLEMA BINARIO (OneMax)
    # -------------------------------------------------------------------------
    def _run_binary_aco(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        dim = problem.dimension
        n_ants = self.population_size

        # tau[i, 0] y tau[i, 1] para cada bit i
        tau = np.ones((dim, 2), dtype=float)
        # Visibilidad heurística ligeramente favorable según el sentido de optimización
        if wrapper.sense == OptimizationSense.MAXIMIZE:
            eta = np.tile([0.4, 0.6], (dim, 1))
        else:
            eta = np.tile([0.6, 0.4], (dim, 1))

        init_ants = problem.random_population(n_ants, rng=rng)
        init_fitness = np.asarray(wrapper(init_ants), dtype=float)
        best_idx = wrapper.best_index(init_fitness)
        best_solution = init_ants[best_idx].copy()
        best_fitness = float(init_fitness[best_idx])

        history: List[Dict[str, Any]] = []
        self._record_history_step(history, 0, best_fitness, best_solution, init_fitness)
        stagnation_counter = 0
        stop_reason = "max_iterations"
        total_iterations = 0

        for it in range(1, self.max_iterations + 1):
            total_iterations = it
            desirability = (tau ** self.alpha) * (eta ** self.beta)
            prob_one = desirability[:, 1] / np.sum(desirability, axis=1)

            # Muestrear hormigas de forma vectorizada
            rand_vals = rng.random(size=(n_ants, dim))
            ants = (rand_vals < prob_one[None, :]).astype(int)
            fitness = np.asarray(wrapper(ants), dtype=float)

            iter_best_idx = wrapper.best_index(fitness)
            iter_best_fit = float(fitness[iter_best_idx])
            iter_best_ant = ants[iter_best_idx].copy()

            improved = wrapper.is_better(iter_best_fit, best_fitness, strict=True)
            improvement_amount = abs(iter_best_fit - best_fitness) if improved else 0.0
            if improved:
                best_fitness = iter_best_fit
                best_solution = iter_best_ant.copy()

            # Evaporación y depósito de feromona (mejores hormigas de la iteración + élite global)
            tau *= 1.0 - self.evaporation_rate
            affinities = wrapper.compute_normalized_affinity(fitness)
            top_k = max(1, n_ants // 4)
            sorted_idx = wrapper.argsort(fitness)[:top_k]

            for idx in sorted_idx:
                deposit = self.pheromone_deposit * (0.3 + 0.7 * float(affinities[idx])) / top_k
                ant = ants[idx]
                tau[np.arange(dim), ant] += deposit

            # Refuerzo elitista con la mejor solución global
            tau[np.arange(dim), best_solution] += self.elitist_weight * self.pheromone_deposit * 0.5
            tau = np.clip(tau, self.tau_min, self.tau_max)

            self._record_history_step(history, it, best_fitness, best_solution, fitness)
            stagnation_counter, should_stop = self._check_early_stopping(
                improved, improvement_amount, stagnation_counter
            )
            if should_stop:
                stop_reason = f"early_convergence (sin mejora por {self.early_stopping_patience} iteraciones)"
                break

        return best_solution, best_fitness, stop_reason, total_iterations, history

    # -------------------------------------------------------------------------
    # 2. Construcción y actualización para ASIGNACIÓN DISCRETA (Task Assignment)
    # -------------------------------------------------------------------------
    def _run_discrete_aco(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        n_tasks = problem.dimension
        n_res = int(getattr(problem, "n_resources", 4))
        cost_matrix = np.asarray(getattr(problem, "cost_matrix", np.ones((n_tasks, n_res))), dtype=float)
        demands = np.asarray(getattr(problem, "task_demands", np.ones(n_tasks)), dtype=float)
        max_cap = np.asarray(
            getattr(problem, "max_capacity_per_resource", np.full(n_res, max(1.0, n_tasks / n_res))),
            dtype=float,
        )

        # Matriz de feromona tau[t, r]
        tau = np.ones((n_tasks, n_res), dtype=float)

        # Visibilidad heurística eta[t, r]: mayor para menor costo (en minimización) o mayor costo (en maximización)
        c_min = np.min(cost_matrix)
        c_max = np.max(cost_matrix)
        if wrapper.sense == OptimizationSense.MINIMIZE:
            eta_base = 1.0 / (cost_matrix - c_min + 1.0)
        else:
            eta_base = (cost_matrix - c_min + 1.0) / max(1.0, c_max - c_min + 1.0)

        n_ants = self.population_size
        init_ants = problem.random_population(n_ants, rng=rng)
        init_fitness = np.asarray(wrapper(init_ants), dtype=float)
        best_idx = wrapper.best_index(init_fitness)
        best_solution = init_ants[best_idx].copy()
        best_fitness = float(init_fitness[best_idx])

        history: List[Dict[str, Any]] = []
        self._record_history_step(history, 0, best_fitness, best_solution, init_fitness)
        stagnation_counter = 0
        stop_reason = "max_iterations"
        total_iterations = 0

        for it in range(1, self.max_iterations + 1):
            total_iterations = it
            ants = np.zeros((n_ants, n_tasks), dtype=int)
            tau_alpha = tau ** self.alpha
            eta_beta = eta_base ** self.beta

            for k in range(n_ants):
                loads = np.zeros(n_res, dtype=float)
                # Recorrer tareas en orden aleatorio para no sesgar los últimos recursos
                task_order = rng.permutation(n_tasks)
                for t in task_order:
                    # Penalizar heurísticamente recursos que excederían su capacidad máxima
                    projected_loads = loads + demands[t]
                    cap_factor = np.where(projected_loads <= max_cap, 1.0, 0.08)
                    weights = tau_alpha[t] * eta_beta[t] * cap_factor
                    w_sum = float(np.sum(weights))
                    if w_sum <= 1e-15:
                        probs = np.full(n_res, 1.0 / n_res)
                    else:
                        probs = weights / w_sum
                    chosen_r = int(rng.choice(n_res, p=probs))
                    ants[k, t] = chosen_r
                    loads[chosen_r] += demands[t]

            fitness = np.asarray(wrapper(ants), dtype=float)
            iter_best_idx = wrapper.best_index(fitness)
            iter_best_fit = float(fitness[iter_best_idx])
            iter_best_ant = ants[iter_best_idx].copy()

            improved = wrapper.is_better(iter_best_fit, best_fitness, strict=True)
            improvement_amount = abs(iter_best_fit - best_fitness) if improved else 0.0
            if improved:
                best_fitness = iter_best_fit
                best_solution = iter_best_ant.copy()

            # Evaporación y depósito de feromona
            tau *= 1.0 - self.evaporation_rate
            affinities = wrapper.compute_normalized_affinity(fitness)
            top_k = max(1, n_ants // 3)
            sorted_idx = wrapper.argsort(fitness)[:top_k]
            task_range = np.arange(n_tasks)

            for idx in sorted_idx:
                deposit = self.pheromone_deposit * (0.25 + 0.75 * float(affinities[idx])) / top_k
                tau[task_range, ants[idx]] += deposit

            tau[task_range, best_solution] += self.elitist_weight * self.pheromone_deposit * 0.5
            tau = np.clip(tau, self.tau_min, self.tau_max)

            self._record_history_step(history, it, best_fitness, best_solution, fitness)
            stagnation_counter, should_stop = self._check_early_stopping(
                improved, improvement_amount, stagnation_counter
            )
            if should_stop:
                stop_reason = f"early_convergence (sin mejora por {self.early_stopping_patience} iteraciones)"
                break

        return best_solution, best_fitness, stop_reason, total_iterations, history

    # -------------------------------------------------------------------------
    # 3. Construcción y actualización para PERMUTACIONES DE RUTEO (Delivery Routing)
    # -------------------------------------------------------------------------
    def _run_permutation_aco(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        n_clients = problem.dimension
        n_nodes = n_clients + 1  # Nodo 0 = CEDIS, Nodos 1..N = Clientes 0..N-1
        dist_mat = np.asarray(
            getattr(problem, "distance_matrix", np.ones((n_nodes, n_nodes))),
            dtype=float,
        )

        # Feromona entre todos los pares de nodos (incluyendo arcos desde/hacia el CEDIS)
        tau = np.ones((n_nodes, n_nodes), dtype=float)

        # Visibilidad eta[u, v]: en minimización prefiere arcos cortos (1 / d_uv)
        with np.errstate(divide="ignore"):
            if wrapper.sense == OptimizationSense.MINIMIZE:
                eta = 1.0 / np.maximum(dist_mat, 0.05)
            else:
                eta = np.maximum(dist_mat, 0.05)
        np.fill_diagonal(eta, 0.0)

        n_ants = self.population_size
        init_ants = problem.random_population(n_ants, rng=rng)
        init_fitness = np.asarray(wrapper(init_ants), dtype=float)
        best_idx = wrapper.best_index(init_fitness)
        best_solution = init_ants[best_idx].copy()
        best_fitness = float(init_fitness[best_idx])

        history: List[Dict[str, Any]] = []
        self._record_history_step(history, 0, best_fitness, best_solution, init_fitness)
        stagnation_counter = 0
        stop_reason = "max_iterations"
        total_iterations = 0

        for it in range(1, self.max_iterations + 1):
            total_iterations = it
            ants = np.zeros((n_ants, n_clients), dtype=int)
            tau_alpha = tau ** self.alpha
            eta_beta = eta ** self.beta

            for k in range(n_ants):
                current_node = 0  # Sale del almacén central (CEDIS)
                unvisited = list(range(1, n_nodes))
                tour_clients: List[int] = []

                for _ in range(n_clients):
                    candidates = np.array(unvisited, dtype=int)
                    weights = tau_alpha[current_node, candidates] * eta_beta[current_node, candidates]
                    w_sum = float(np.sum(weights))
                    if w_sum <= 1e-15:
                        next_node = int(rng.choice(candidates))
                    else:
                        probs = weights / w_sum
                        next_node = int(rng.choice(candidates, p=probs))

                    tour_clients.append(next_node - 1)
                    unvisited.remove(next_node)
                    current_node = next_node

                ants[k] = np.array(tour_clients, dtype=int)

            fitness = np.asarray(wrapper(ants), dtype=float)
            iter_best_idx = wrapper.best_index(fitness)
            iter_best_fit = float(fitness[iter_best_idx])
            iter_best_ant = ants[iter_best_idx].copy()

            # Búsqueda local 2-opt rápida sobre la mejor hormiga de la iteración
            candidate_2opt = iter_best_ant.copy()
            i_swap, j_swap = np.sort(rng.choice(n_clients, size=2, replace=False))
            candidate_2opt[i_swap : j_swap + 1] = candidate_2opt[i_swap : j_swap + 1][::-1]
            cand_fit = float(wrapper(candidate_2opt))
            if wrapper.is_better(cand_fit, iter_best_fit, strict=True):
                iter_best_ant = candidate_2opt
                iter_best_fit = cand_fit
                ants[iter_best_idx] = candidate_2opt
                fitness[iter_best_idx] = cand_fit

            improved = wrapper.is_better(iter_best_fit, best_fitness, strict=True)
            improvement_amount = abs(iter_best_fit - best_fitness) if improved else 0.0
            if improved:
                best_fitness = iter_best_fit
                best_solution = iter_best_ant.copy()

            # Evaporación y actualización de feromona en los arcos recorridos
            tau *= 1.0 - self.evaporation_rate
            affinities = wrapper.compute_normalized_affinity(fitness)
            top_k = max(1, n_ants // 3)
            sorted_idx = wrapper.argsort(fitness)[:top_k]

            for idx in sorted_idx:
                deposit = self.pheromone_deposit * (0.3 + 0.7 * float(affinities[idx])) / top_k
                self._deposit_tour_pheromone(tau, ants[idx], deposit)

            self._deposit_tour_pheromone(
                tau, best_solution, self.elitist_weight * self.pheromone_deposit * 0.6
            )
            tau = np.clip(tau, self.tau_min, self.tau_max)

            self._record_history_step(history, it, best_fitness, best_solution, fitness)
            stagnation_counter, should_stop = self._check_early_stopping(
                improved, improvement_amount, stagnation_counter
            )
            if should_stop:
                stop_reason = f"early_convergence (sin mejora por {self.early_stopping_patience} iteraciones)"
                break

        return best_solution, best_fitness, stop_reason, total_iterations, history

    @staticmethod
    def _deposit_tour_pheromone(tau: np.ndarray, perm: np.ndarray, amount: float) -> None:
        """Deposita feromona simétrica a lo largo de los arcos del circuito CEDIS -> perm -> CEDIS."""
        seq = [0] + [int(c) + 1 for c in perm] + [0]
        for s in range(len(seq) - 1):
            u, v = seq[s], seq[s + 1]
            tau[u, v] += amount
            tau[v, u] += amount

    # -------------------------------------------------------------------------
    # Punto de entrada de optimización
    # -------------------------------------------------------------------------
    def _optimize(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        rep = problem.representation
        if rep == RepresentationType.BINARY:
            return self._run_binary_aco(problem, wrapper, rng)
        if rep == RepresentationType.DISCRETE:
            return self._run_discrete_aco(problem, wrapper, rng)
        if rep == RepresentationType.PERMUTATION:
            return self._run_permutation_aco(problem, wrapper, rng)
        raise ValueError(f"Representación '{rep}' no soportada por ACO.")

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "evaporation_rate": self.evaporation_rate,
                "pheromone_deposit": self.pheromone_deposit,
                "elitist_weight": self.elitist_weight,
            }
        )
        return params

