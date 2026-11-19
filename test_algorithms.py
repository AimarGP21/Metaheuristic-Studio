"""
Script de verificación integral para los 5 algoritmos metaheurísticos de Metaheuristic Studio:
GA, PSO, DE, ACO y AIS.

Prueba:
1. Convergencia efectiva en todos los pares (Problema, Algoritmo) compatibles.
2. Estructura exacta del diccionario de salida (`best_solution`, `best_fitness`, `stop_reason`,
   `total_iterations`, `history` con `iteration`, `best_fitness`, `best_solution`, `mean_fitness`).
3. Reproducibilidad exacta con semilla (`seed`).
4. Criterio de parada por convergencia anticipada (`early_stopping_patience`).
5. Bloqueo con mensaje explicativo en combinaciones no compatibles de `COMPATIBILITY_MATRIX`.
"""

from __future__ import annotations

import numpy as np

from algorithms import (
    ALGORITHM_REGISTRY,
    AntColonyOptimization,
    ArtificialImmuneSystem,
    DifferentialEvolution,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
)
from core.base import COMPATIBILITY_MATRIX, check_compatibility
from problems import (
    AckleyProblem,
    DeliveryRoutingProblem,
    OneMaxProblem,
    TaskAssignmentProblem,
)


def validate_result_structure(res: dict, sense: str) -> None:
    """Verifica que el diccionario devuelto cumpla todos los requisitos obligatorios."""
    required_keys = {
        "best_solution",
        "best_fitness",
        "stop_reason",
        "total_iterations",
        "history",
    }
    missing = required_keys - set(res.keys())
    assert not missing, f"Faltan claves obligatorias en el resultado: {missing}"
    assert isinstance(res["history"], list) and len(res["history"]) == res["total_iterations"] + 1
    assert res["history"][0]["iteration"] == 0
    assert res["history"][-1]["iteration"] == res["total_iterations"]

    first_step = res["history"][0]
    last_step = res["history"][-1]
    for step_key in ("iteration", "best_fitness", "best_solution", "mean_fitness"):
        assert step_key in first_step, f"Falta clave '{step_key}' en history."

    # Verificar que el mejor fitness al final sea igual o mejor que en la iteración 1
    if sense == "minimize":
        assert last_step["best_fitness"] <= first_step["best_fitness"] + 1e-12, (
            f"El algoritmo empeoró en minimización: {first_step['best_fitness']} -> {last_step['best_fitness']}"
        )
    else:
        assert last_step["best_fitness"] >= first_step["best_fitness"] - 1e-12, (
            f"El algoritmo empeoró en maximización: {first_step['best_fitness']} -> {last_step['best_fitness']}"
        )


def main() -> None:
    print("=" * 80)
    print("VERIFICACIÓN DE CONVERGENCIA DE METAHEURISTIC STUDIO (GA, PSO, DE, ACO, AIS)")
    print("=" * 80)

    problems = {
        "ackley": AckleyProblem(dimension=6, lower_bound=-15.0, upper_bound=15.0),
        "onemax": OneMaxProblem(n_bits=30),
        "task_assignment": TaskAssignmentProblem(
            n_tasks=10, n_resources=4, max_capacity=3, penalty_factor=100.0, seed=42
        ),
        "delivery_routing": DeliveryRoutingProblem(
            n_clients=12, vehicle_capacity=200.0, penalty_per_kg=15.0
        ),
    }

    for prob_key, problem in problems.items():
        print(f"\n>>> Problema: {problem.name} [{prob_key}] | Sentido: {problem.sense.value}")
        print("-" * 80)

        for algo_key, algo_cls in ALGORITHM_REGISTRY.items():
            is_compat, msg = check_compatibility(prob_key, algo_key)
            algo = algo_cls(max_iterations=45, population_size=30, seed=12345)

            if not is_compat:
                # Verificar que lance ValueError cuando no es compatible
                try:
                    algo.run(problem)
                    raise AssertionError(
                        f"Se esperaba ValueError para par incompatible ({prob_key}, {algo_key})"
                    )
                except ValueError:
                    print(f"  [SKIP Compatible=False] {algo_key:<4} -> Bloqueado correctamente ({msg[:55]}...)")
                continue

            res = algo.run(problem)
            validate_result_structure(res, problem.sense.value)

            init_best = res["history"][0]["best_fitness"]
            final_best = res["best_fitness"]
            final_mean = res["history"][-1]["mean_fitness"]
            iters = res["total_iterations"]

            print(
                f"  [OK] {algo_key:<4} | Iter 1 Best: {init_best:10.4f} -> "
                f"Final Best: {final_best:10.4f} (Mean: {final_mean:10.4f}) | "
                f"Iters: {iters} | Stop: {res['stop_reason']}"
            )

    # -------------------------------------------------------------------------
    # Prueba de Reproducibilidad Exacta con Semilla (seed)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA DE REPRODUCIBILIDAD EXACTA (SEED) Y PARADA ANTICIPADA")
    print("=" * 80)

    ack_prob = problems["ackley"]
    for algo_key, algo_cls in ALGORITHM_REGISTRY.items():
        target_prob = ack_prob if algo_key != "ACO" else problems["delivery_routing"]
        a1 = algo_cls(max_iterations=25, population_size=20, seed=999)
        a2 = algo_cls(max_iterations=25, population_size=20, seed=999)
        r1 = a1.run(target_prob)
        r2 = a2.run(target_prob)
        assert np.allclose(r1["best_solution"], r2["best_solution"])
        assert abs(r1["best_fitness"] - r2["best_fitness"]) < 1e-12
        print(f"  [SEED OK] {algo_key:<4} produce resultados idénticos con seed=999 (fitness={r1['best_fitness']:.4f})")

    # Prueba de convergencia anticipada (early stopping) en OneMax pequeño
    small_onemax = OneMaxProblem(n_bits=12)
    ga_early = GeneticAlgorithm(
        max_iterations=100,
        population_size=25,
        early_stopping_patience=8,
        seed=42,
    )
    res_early = ga_early.run(small_onemax)
    assert res_early["total_iterations"] < 100
    assert "early_convergence" in res_early["stop_reason"]
    print(
        f"  [EARLY STOP OK] GA en OneMax(12) se detuvo en iteración {res_early['total_iterations']}/100 "
        f"con motivo: '{res_early['stop_reason']}' y fitness={res_early['best_fitness']}"
    )

    print("\n¡TODAS LAS PRUEBAS DE LOS 5 ALGORITMOS PASARON EXITOSAMENTE!")


if __name__ == "__main__":
    main()

