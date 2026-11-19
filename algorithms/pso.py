"""
Optimización por Enjambre de Partículas (PSO y Binary PSO) para Metaheuristic Studio.

Soporta:
- Continuo (Ackley): PSO estándar con decaimiento lineal de inercia (w_max -> w_min),
  coeficientes cognitivo (c1) y social (c2), y límites de velocidad [-v_max, v_max].
- Binario (OneMax): Binary PSO (Kennedy & Eberhart, 1997) mediante función de transferencia
  sigmoide S(v) = 1 / (1 + exp(-v)) para muestrear probabilidades de encendido de bits.
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


class ParticleSwarmOptimization(BaseMetaheuristic):
    """
    Enjambre de Partículas (PSO) con soporte dual para dominios continuos (estándar)
    y dominios binarios (Binary PSO con función sigmoide).
    """

    algorithm_id: str = "PSO"
    name: str = "Enjambre de Partículas (PSO / BPSO)"
    description: str = (
        "Metaheurística bio-inspirada en el movimiento colectivo de bandadas; "
        "usa inercia y límites de velocidad en continuo, y función sigmoide en binario."
    )

    def __init__(
        self,
        max_iterations: int = 100,
        population_size: int = 40,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.8,
        c2: float = 1.8,
        vel_clamp_factor: float = 0.2,
        bpso_v_max: float = 6.0,
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
        self.w_max: float = float(w_max)
        self.w_min: float = float(w_min)
        self.c1: float = float(c1)
        self.c2: float = float(c2)
        self.vel_clamp_factor: float = float(np.clip(vel_clamp_factor, 0.01, 1.0))
        self.bpso_v_max: float = float(max(1.0, bpso_v_max))

    @staticmethod
    def _sigmoid(v: np.ndarray) -> np.ndarray:
        """Función de transferencia sigmoide numéricamente estable para Binary PSO."""
        v_clipped = np.clip(v, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-v_clipped))

    def _optimize(
        self,
        problem: BaseProblem,
        wrapper: ObjectiveWrapper,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, str, int, List[Dict[str, Any]]]:
        dim = problem.dimension
        rep = problem.representation
        n_particles = self.population_size

        positions = problem.random_population(n_particles, rng=rng)
        fitness = np.asarray(wrapper(positions), dtype=float)

        # Inicializar velocidades según la representación
        if rep == RepresentationType.CONTINUOUS:
            lower = float(getattr(problem, "lower_bound", -32.768))
            upper = float(getattr(problem, "upper_bound", 32.768))
            v_max = self.vel_clamp_factor * (upper - lower)
            velocities = rng.uniform(-v_max, v_max, size=(n_particles, dim))
        elif rep == RepresentationType.BINARY:
            v_max = self.bpso_v_max
            velocities = rng.uniform(-1.5, 1.5, size=(n_particles, dim))
        else:
            raise ValueError(f"Representación '{rep}' no soportada por PSO.")

        # Mejor personal (pbest) y mejor global (gbest)
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()

        gbest_idx = wrapper.best_index(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = float(pbest_fit[gbest_idx])

        history: List[Dict[str, Any]] = []
        # Registrar Hito Inicial real (Iteración 0: enjambre inicial aleatorio evaluado)
        self._record_history_step(
            history=history,
            iteration=0,
            best_fitness=gbest_fit,
            best_solution=gbest_pos,
            population_fitness=fitness,
        )

        stagnation_counter = 0
        stop_reason = "max_iterations"
        total_iterations = 0

        for it in range(1, self.max_iterations + 1):
            total_iterations = it
            # Peso de inercia con decaimiento lineal de w_max a w_min
            progress = float(it - 1) / max(1.0, float(self.max_iterations - 1))
            w = self.w_max - (self.w_max - self.w_min) * progress

            r1 = rng.random(size=(n_particles, dim))
            r2 = rng.random(size=(n_particles, dim))

            cognitive = self.c1 * r1 * (pbest_pos - positions)
            social = self.c2 * r2 * (gbest_pos[None, :] - positions)
            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -v_max, v_max)

            if rep == RepresentationType.CONTINUOUS:
                new_positions = positions + velocities
                # Amortiguar velocidad en componentes que rebasen los límites del dominio
                out_of_bounds = (new_positions < lower) | (new_positions > upper)
                velocities[out_of_bounds] *= -0.5
                positions = np.clip(new_positions, lower, upper)
            else:
                # Binary PSO: probabilidades de bit con función sigmoide S(v)
                probs = self._sigmoid(velocities)
                rand_matrix = rng.random(size=(n_particles, dim))
                positions = (rand_matrix < probs).astype(int)

            fitness = np.asarray(wrapper(positions), dtype=float)

            # Actualizar mejores personales (pbest)
            for idx in range(n_particles):
                if wrapper.is_better(float(fitness[idx]), float(pbest_fit[idx]), strict=True):
                    pbest_fit[idx] = fitness[idx]
                    pbest_pos[idx] = positions[idx].copy()

            # Actualizar mejor global (gbest)
            iter_best_idx = wrapper.best_index(pbest_fit)
            iter_best_fit = float(pbest_fit[iter_best_idx])
            improved = wrapper.is_better(iter_best_fit, gbest_fit, strict=True)
            improvement_amount = abs(iter_best_fit - gbest_fit) if improved else 0.0

            if improved:
                gbest_fit = iter_best_fit
                gbest_pos = pbest_pos[iter_best_idx].copy()

            self._record_history_step(
                history=history,
                iteration=it,
                best_fitness=gbest_fit,
                best_solution=gbest_pos,
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

        return gbest_pos, gbest_fit, stop_reason, total_iterations, history

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "w_max": self.w_max,
                "w_min": self.w_min,
                "c1": self.c1,
                "c2": self.c2,
                "vel_clamp_factor": self.vel_clamp_factor,
                "bpso_v_max": self.bpso_v_max,
            }
        )
        return params

