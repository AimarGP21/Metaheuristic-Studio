"""
Metaheuristic Studio — Aplicación Web Interactiva en Streamlit.

Permite configurar, validar, ejecutar y comparar 5 algoritmos metaheurísticos
(GA, PSO, DE, ACO, AIS) sobre 4 problemas de optimización (Continuo Ackley,
Binario OneMax, Discreto Asignación de Tareas y Problema Real de Ruteo de
Última Milla en Guadalajara, Jalisco).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from algorithms import (
    ALGORITHM_REGISTRY,
    AntColonyOptimization,
    ArtificialImmuneSystem,
    DifferentialEvolution,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
)
from core.base import (
    COMPATIBILITY_MATRIX,
    BaseMetaheuristic,
    BaseProblem,
    OptimizationSense,
    check_compatibility,
)
from problems import (
    AckleyProblem,
    DeliveryRoutingProblem,
    OneMaxProblem,
    TaskAssignmentProblem,
)
from ui.visualizations import (
    create_ackley_figures,
    create_convergence_figure,
    create_delivery_routing_figures,
    create_onemax_figure,
    create_task_assignment_visuals,
    export_history_to_csv,
    export_results_to_json,
    extract_four_milestones,
)


# ============================================================================
# CATÁLOGOS DESCRIPTIVOS PARA LA UI
# ============================================================================

PROBLEM_CATALOG: Dict[str, Dict[str, str]] = {
    "ackley": {
        "label": "1. Continuo — Función Multimodal de Ackley",
        "short": "Continuo (Ackley)",
        "badge": "Dominio Continuo ℝ^D",
        "default_sense": "Minimizar",
    },
    "onemax": {
        "label": "2. Binario — Problema OneMax",
        "short": "Binario (OneMax)",
        "badge": "Dominio Binario {0, 1}^N",
        "default_sense": "Maximizar",
    },
    "task_assignment": {
        "label": "3. Discreto — Asignación de Tareas a Recursos",
        "short": "Discreto (Asignación T×R)",
        "badge": "Dominio Discreto {0..R-1}^T",
        "default_sense": "Minimizar",
    },
    "delivery_routing": {
        "label": "4. Problema Real — Ruteo Última Milla (Guadalajara, Jal.)",
        "short": "Real (Rutas Guadalajara)",
        "badge": "Permutación S_N (12 Puntos Reales GDL)",
        "default_sense": "Minimizar",
    },
}

ALGORITHM_CATALOG: Dict[str, Dict[str, str]] = {
    "GA": {
        "label": "GA — Algoritmo Genético",
        "short": "GA (Algoritmo Genético)",
        "summary": "Selección por torneo, elitismo, cruza (BLX-α / Uniforme / 2-Puntos / OX1) y mutación adaptada.",
    },
    "PSO": {
        "label": "PSO — Enjambre de Partículas (Continuo y Binary PSO)",
        "short": "PSO (Enjambre de Partículas)",
        "summary": "Velocidad con peso de inercia decreciente y límites de velocidad; función sigmoide en binario.",
    },
    "DE": {
        "label": "DE — Evolución Diferencial (DE/rand/1/bin)",
        "short": "DE (Evolución Diferencial)",
        "summary": "Mutación diferencial v = x_r1 + F·(x_r2 - x_r3) y recombinación binomial para espacios continuos.",
    },
    "ACO": {
        "label": "ACO — Colonia de Hormigas",
        "short": "ACO (Colonia de Hormigas)",
        "summary": "Construcción probabilística guiada por feromona (τ^α) y visibilidad heurística (η^β) con evaporación.",
    },
    "AIS": {
        "label": "AIS — Sistema Inmune Artificial (CLONALG)",
        "short": "AIS (Sistema Inmune CLONALG)",
        "summary": "Selección clonal, expansión e hipermutación somática inversamente proporcional a la afinidad.",
    },
}


# ============================================================================
# INICIALIZACIÓN Y GESTIÓN DE ESTADO (st.session_state)
# ============================================================================

def init_session_state() -> None:
    """Inicializa todas las variables de control de flujo y estado en `st.session_state`."""
    defaults: Dict[str, Any] = {
        "active_step": 1,
        "problem_id": "ackley",
        "algorithm_id": "GA",
        "sense_choice": "Minimizar",
        "is_running": False,
        "stop_requested": False,
        "execution_result": None,
        "last_problem_instance": None,
        "last_problem_config": None,
        "last_algorithm_params": None,
        "reset_counter": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def trigger_reset() -> None:
    """Reinicia el estado de ejecución y parámetros de la sesión."""
    st.session_state["active_step"] = 1
    st.session_state["is_running"] = False
    st.session_state["stop_requested"] = False
    st.session_state["execution_result"] = None
    st.session_state["last_problem_instance"] = None
    st.session_state["last_problem_config"] = None
    st.session_state["last_algorithm_params"] = None
    st.session_state["reset_counter"] = int(st.session_state.get("reset_counter", 0)) + 1


def request_stop() -> None:
    """Activa la bandera de interrupción para detener la ejecución en curso."""
    st.session_state["stop_requested"] = True


def go_previous_step() -> None:
    """Retrocede al paso anterior del flujo o limpia la vista de resultados para ajustar parámetros."""
    current = int(st.session_state.get("active_step", 1))
    if current > 1:
        st.session_state["active_step"] = current - 1


def _is_missing_or_nan(val: Any) -> bool:
    """Verifica si un valor de entrada está vacío (None) o es NaN/Infinito."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    try:
        f_val = float(val)
        return bool(np.isnan(f_val) or np.isinf(f_val))
    except (TypeError, ValueError):
        return True


# ============================================================================
# VALIDACIÓN ESTRICTA DE PARÁMETROS ANTES DE EJECUTAR
# ============================================================================

def validate_configuration(
    problem_id: str,
    problem_cfg: Dict[str, Any],
    algorithm_id: str,
    algo_cfg: Dict[str, Any],
) -> List[str]:
    """
    Realiza una validación estricta de todos los parámetros del problema y del algoritmo
    seleccionado, bloqueando valores vacíos (None/NaN), negativos o fuera de rango antes
    de permitir la ejecución.
    """
    errors: List[str] = []

    # 1. Compatibilidad en COMPATIBILITY_MATRIX
    is_compat, compat_msg = check_compatibility(problem_id, algorithm_id)
    if not is_compat:
        errors.append(f"Combinación incompatible: {compat_msg}")

    # 2. Validación de parámetros del problema
    if problem_id == "ackley":
        dim_raw = problem_cfg.get("dimension")
        low_raw = problem_cfg.get("lower_bound")
        high_raw = problem_cfg.get("upper_bound")
        if _is_missing_or_nan(dim_raw):
            errors.append("Ackley: La dimensión D no puede estar vacía.")
        elif int(dim_raw) < 1 or int(dim_raw) > 200:
            errors.append("Ackley: La dimensión D debe estar en el rango [1, 200].")

        if _is_missing_or_nan(low_raw) or _is_missing_or_nan(high_raw):
            errors.append("Ackley: Los límites inferior (min) y superior (max) no pueden estar vacíos.")
        else:
            low, high = float(low_raw), float(high_raw)
            if low >= high:
                errors.append(
                    f"Ackley: El límite inferior ({low}) debe ser estrictamente menor que el límite superior ({high})."
                )
            if low < -1000.0 or high > 1000.0:
                errors.append("Ackley: Los límites deben estar dentro del rango [-1000, 1000].")

    elif problem_id == "onemax":
        nb_raw = problem_cfg.get("n_bits")
        if _is_missing_or_nan(nb_raw):
            errors.append("OneMax: La longitud del vector N no puede estar vacía.")
        elif int(nb_raw) < 2 or int(nb_raw) > 1000:
            errors.append("OneMax: La longitud del vector N debe estar entre 2 y 1000 bits.")

    elif problem_id == "task_assignment":
        t_raw = problem_cfg.get("n_tasks")
        r_raw = problem_cfg.get("n_resources")
        cap_raw = problem_cfg.get("max_capacity")
        pen_raw = problem_cfg.get("penalty_factor")
        mseed_raw = problem_cfg.get("matrix_seed")

        if _is_missing_or_nan(t_raw):
            errors.append("Asignación de Tareas: El número de tareas T no puede estar vacío.")
        elif int(t_raw) < 2 or int(t_raw) > 100:
            errors.append("Asignación de Tareas: El número de tareas T debe estar entre 2 y 100.")

        if _is_missing_or_nan(r_raw):
            errors.append("Asignación de Tareas: El número de recursos R no puede estar vacío.")
        elif int(r_raw) < 2 or int(r_raw) > 50:
            errors.append("Asignación de Tareas: El número de recursos R debe estar entre 2 y 50.")

        if _is_missing_or_nan(cap_raw):
            errors.append("Asignación de Tareas: La capacidad máxima por recurso no puede estar vacía.")
        elif float(cap_raw) <= 0:
            errors.append("Asignación de Tareas: La capacidad máxima por recurso debe ser mayor a 0.")

        if _is_missing_or_nan(pen_raw):
            errors.append("Asignación de Tareas: El factor de penalización no puede estar vacío.")
        elif float(pen_raw) < 0:
            errors.append("Asignación de Tareas: El factor de penalización no puede ser negativo.")

        if _is_missing_or_nan(mseed_raw) or int(mseed_raw) < 0:
            errors.append("Asignación de Tareas: La semilla de la matriz de costos debe ser un entero >= 0.")

        c_mat = problem_cfg.get("cost_matrix")
        if c_mat is None:
            errors.append("Asignación de Tareas: La matriz de costos T×R no es válida.")
        else:
            c_arr = np.asarray(c_mat, dtype=float)
            if np.any(np.isnan(c_arr)) or np.any(np.isinf(c_arr)):
                errors.append("Asignación de Tareas: La matriz de costos contiene celdas vacías o NaN.")
            elif np.any(c_arr < 0):
                errors.append("Asignación de Tareas: La matriz de costos no puede contener costos negativos.")

    elif problem_id == "delivery_routing":
        n_cli_raw = problem_cfg.get("n_clients")
        v_cap_raw = problem_cfg.get("vehicle_capacity")
        pen_kg_raw = problem_cfg.get("penalty_per_kg")

        if _is_missing_or_nan(n_cli_raw):
            errors.append("Ruteo Guadalajara: El número de clientes N no puede estar vacío.")
        elif int(n_cli_raw) < 2 or int(n_cli_raw) > 12:
            errors.append("Ruteo Guadalajara: El número de clientes N debe estar entre 2 y 12.")

        if _is_missing_or_nan(v_cap_raw):
            errors.append("Ruteo Guadalajara: La capacidad del vehículo no puede estar vacía.")
        elif float(v_cap_raw) <= 0:
            errors.append("Ruteo Guadalajara: La capacidad del vehículo debe ser estrictamente mayor a 0 kg.")

        if _is_missing_or_nan(pen_kg_raw):
            errors.append("Ruteo Guadalajara: La penalización por kg excedido no puede estar vacía.")
        elif float(pen_kg_raw) < 0:
            errors.append("Ruteo Guadalajara: La penalización por kg excedido no puede ser negativa.")

        c_weights = problem_cfg.get("custom_weights")
        if c_weights is not None:
            w_arr = np.asarray(c_weights, dtype=float)
            if np.any(np.isnan(w_arr)) or np.any(np.isinf(w_arr)):
                errors.append("Ruteo Guadalajara: La tabla de paquetes contiene pesos vacíos o inválidos.")
            elif np.any(w_arr < 0):
                errors.append("Ruteo Guadalajara: Los pesos de los paquetes no pueden ser negativos.")

    # 3. Validación de parámetros comunes de ejecución
    max_iters_raw = algo_cfg.get("max_iterations")
    pop_size_raw = algo_cfg.get("population_size")

    if _is_missing_or_nan(max_iters_raw):
        errors.append("Ejecución: El número máximo de iteraciones no puede estar vacío.")
        max_iters = 100
    else:
        max_iters = int(max_iters_raw)
        if max_iters < 1 or max_iters > 5000:
            errors.append("Ejecución: El número máximo de iteraciones debe estar entre 1 y 5000.")

    if _is_missing_or_nan(pop_size_raw):
        errors.append("Ejecución: El tamaño de población no puede estar vacío.")
        pop_size = 40
    else:
        pop_size = int(pop_size_raw)
        if pop_size < 4 or pop_size > 1000:
            errors.append("Ejecución: El tamaño de población debe estar entre 4 y 1000 individuos.")

    if algo_cfg.get("use_early_stopping", False):
        pat_raw = algo_cfg.get("early_stopping_patience")
        if _is_missing_or_nan(pat_raw):
            errors.append("Parada Anticipada: La paciencia de iteraciones no puede estar vacía.")
        elif int(pat_raw) < 1 or int(pat_raw) > max_iters:
            errors.append(
                f"Parada Anticipada: La paciencia debe estar entre 1 y max_iterations ({max_iters})."
            )

    if algo_cfg.get("use_seed", False):
        seed_raw = algo_cfg.get("seed")
        if _is_missing_or_nan(seed_raw):
            errors.append("Semilla Aleatoria: El campo de semilla no puede estar vacío cuando está activado.")
        elif int(seed_raw) < 0:
            errors.append("Semilla Aleatoria: La semilla debe ser un entero no negativo (>= 0).")

    # 4. Validación específica por algoritmo
    if algorithm_id == "GA":
        cx_raw = algo_cfg.get("crossover_rate")
        mut_raw = algo_cfg.get("mutation_rate")
        tourn_raw = algo_cfg.get("tournament_size")
        elit_raw = algo_cfg.get("elitism_count")
        blx_raw = algo_cfg.get("blx_alpha")

        if _is_missing_or_nan(cx_raw) or not (0.0 <= float(cx_raw) <= 1.0):
            errors.append("GA: La probabilidad de cruza (Pc) debe estar en el rango [0.0, 1.0].")
        if _is_missing_or_nan(mut_raw) or not (0.0 <= float(mut_raw) <= 1.0):
            errors.append("GA: La probabilidad de mutación (Pm) debe estar en el rango [0.0, 1.0].")
        if _is_missing_or_nan(tourn_raw) or int(tourn_raw) < 2 or int(tourn_raw) > pop_size:
            errors.append(f"GA: El tamaño de torneo (k) debe estar entre 2 y el tamaño de población ({pop_size}).")
        if _is_missing_or_nan(elit_raw) or int(elit_raw) < 1 or int(elit_raw) >= pop_size:
            errors.append(f"GA: El número de individuos élite debe estar entre 1 y {max(1, pop_size - 1)}.")
        if _is_missing_or_nan(blx_raw) or not (0.0 <= float(blx_raw) <= 2.0):
            errors.append("GA: El factor BLX-alpha debe estar entre 0.0 y 2.0.")

    elif algorithm_id == "PSO":
        w_min_raw = algo_cfg.get("w_min")
        w_max_raw = algo_cfg.get("w_max")
        c1_raw = algo_cfg.get("c1")
        c2_raw = algo_cfg.get("c2")
        vc_raw = algo_cfg.get("vel_clamp_factor")

        if _is_missing_or_nan(w_min_raw) or _is_missing_or_nan(w_max_raw):
            errors.append("PSO: Los pesos de inercia w_min y w_max no pueden estar vacíos.")
        else:
            w_min, w_max = float(w_min_raw), float(w_max_raw)
            if w_min <= 0 or w_max <= 0 or w_min > w_max or w_max > 2.0:
                errors.append("PSO: Los pesos de inercia deben cumplir 0 < w_min <= w_max <= 2.0.")

        if _is_missing_or_nan(c1_raw) or _is_missing_or_nan(c2_raw):
            errors.append("PSO: Los coeficientes cognitivo (c1) y social (c2) no pueden estar vacíos.")
        else:
            c1, c2 = float(c1_raw), float(c2_raw)
            if c1 < 0 or c2 < 0 or (c1 + c2) <= 0:
                errors.append("PSO: c1 y c2 deben ser >= 0 y al menos uno debe ser mayor a 0.")

        if _is_missing_or_nan(vc_raw) or not (0.01 <= float(vc_raw) <= 1.0):
            errors.append("PSO: El factor de límite de velocidad debe estar en [0.01, 1.0].")

    elif algorithm_id == "DE":
        f_raw = algo_cfg.get("mutation_factor")
        cr_raw = algo_cfg.get("crossover_rate")
        if _is_missing_or_nan(f_raw) or not (0.05 <= float(f_raw) <= 2.0):
            errors.append("DE: El factor de escala diferencial F debe estar en [0.05, 2.0].")
        if _is_missing_or_nan(cr_raw) or not (0.0 <= float(cr_raw) <= 1.0):
            errors.append("DE: La tasa de cruza binomial CR debe estar en [0.0, 1.0].")

    elif algorithm_id == "ACO":
        evap_raw = algo_cfg.get("evaporation_rate")
        alpha_raw = algo_cfg.get("alpha")
        beta_raw = algo_cfg.get("beta")
        q_raw = algo_cfg.get("pheromone_deposit")
        el_raw = algo_cfg.get("elitist_weight")

        if _is_missing_or_nan(evap_raw) or not (0.01 <= float(evap_raw) <= 0.95):
            errors.append("ACO: La tasa de evaporación ρ debe estar en [0.01, 0.95].")
        if _is_missing_or_nan(alpha_raw) or _is_missing_or_nan(beta_raw):
            errors.append("ACO: Los exponentes α (feromona) y β (heurística) no pueden estar vacíos.")
        elif float(alpha_raw) < 0 or float(beta_raw) < 0 or (float(alpha_raw) + float(beta_raw)) <= 0:
            errors.append("ACO: α y β deben ser >= 0 y su suma debe ser > 0.")
        if _is_missing_or_nan(q_raw) or float(q_raw) <= 0:
            errors.append("ACO: El factor de depósito de feromona Q debe ser mayor a 0.")
        if _is_missing_or_nan(el_raw) or float(el_raw) < 0:
            errors.append("ACO: El peso de la hormiga élite no puede ser negativo.")

    elif algorithm_id == "AIS":
        sel_raw = algo_cfg.get("selection_ratio")
        beta_raw = algo_cfg.get("clone_factor")
        rep_raw = algo_cfg.get("replace_rate")
        dec_raw = algo_cfg.get("mutation_decay")

        if _is_missing_or_nan(sel_raw) or not (0.1 <= float(sel_raw) <= 1.0):
            errors.append("AIS: La proporción de selección clonal debe estar en [0.1, 1.0].")
        if _is_missing_or_nan(beta_raw) or not (0.1 <= float(beta_raw) <= 3.0):
            errors.append("AIS: El factor de expansión clonal β debe estar en [0.1, 3.0].")
        if _is_missing_or_nan(rep_raw) or not (0.0 <= float(rep_raw) <= 0.5):
            errors.append("AIS: La tasa de reemplazo (edición de receptores) debe estar en [0.0, 0.5].")
        if _is_missing_or_nan(dec_raw) or float(dec_raw) <= 0.1:
            errors.append("AIS: El decaimiento de hipermutación ρ debe ser > 0.1.")

    return errors


# ============================================================================
# CONSTRUCTORES DE INSTANCIAS DE PROBLEMA Y ALGORITMO
# ============================================================================

def build_problem_instance(
    problem_id: str,
    problem_cfg: Dict[str, Any],
    sense_str: str,
) -> BaseProblem:
    """Instancia el objeto de problema configurado con el sentido de optimización elegido."""
    sense_norm = OptimizationSense.normalize(sense_str)

    if problem_id == "ackley":
        return AckleyProblem(
            dimension=int(problem_cfg["dimension"]),
            lower_bound=float(problem_cfg["lower_bound"]),
            upper_bound=float(problem_cfg["upper_bound"]),
            a=float(problem_cfg.get("a", 20.0)),
            b=float(problem_cfg.get("b", 0.2)),
            c=float(problem_cfg.get("c", 2.0 * np.pi)),
            sense=sense_norm,
        )

    if problem_id == "onemax":
        return OneMaxProblem(
            n_bits=int(problem_cfg["n_bits"]),
            sense=sense_norm,
        )

    if problem_id == "task_assignment":
        return TaskAssignmentProblem(
            n_tasks=int(problem_cfg["n_tasks"]),
            n_resources=int(problem_cfg["n_resources"]),
            max_capacity=float(problem_cfg["max_capacity"]),
            penalty_factor=float(problem_cfg["penalty_factor"]),
            seed=int(problem_cfg["matrix_seed"]),
            cost_matrix=problem_cfg.get("cost_matrix"),
            sense=sense_norm,
        )

    if problem_id == "delivery_routing":
        return DeliveryRoutingProblem(
            n_clients=int(problem_cfg["n_clients"]),
            vehicle_capacity=float(problem_cfg["vehicle_capacity"]),
            penalty_per_kg=float(problem_cfg["penalty_per_kg"]),
            custom_weights=problem_cfg.get("custom_weights"),
            sense=sense_norm,
        )

    raise ValueError(f"Problema desconocido: {problem_id}")


def build_algorithm_instance(
    algorithm_id: str,
    algo_cfg: Dict[str, Any],
    sense_str: str,
) -> BaseMetaheuristic:
    """Instancia el algoritmo metaheurístico seleccionado con sus hiperparámetros."""
    sense_norm = OptimizationSense.normalize(sense_str)
    patience_val = (
        int(algo_cfg["early_stopping_patience"])
        if algo_cfg.get("use_early_stopping") and not _is_missing_or_nan(algo_cfg.get("early_stopping_patience"))
        else None
    )
    seed_val = (
        int(algo_cfg["seed"])
        if algo_cfg.get("use_seed") and not _is_missing_or_nan(algo_cfg.get("seed"))
        else None
    )

    common_kwargs = dict(
        max_iterations=int(algo_cfg["max_iterations"]),
        population_size=int(algo_cfg["population_size"]),
        seed=seed_val,
        early_stopping_patience=patience_val,
        tol=float(algo_cfg.get("tol", 1e-8)),
        sense=sense_norm,
    )

    if algorithm_id == "GA":
        return GeneticAlgorithm(
            crossover_rate=float(algo_cfg["crossover_rate"]),
            mutation_rate=float(algo_cfg["mutation_rate"]),
            tournament_size=int(algo_cfg["tournament_size"]),
            elitism_count=int(algo_cfg["elitism_count"]),
            blx_alpha=float(algo_cfg["blx_alpha"]),
            **common_kwargs,
        )

    if algorithm_id == "PSO":
        return ParticleSwarmOptimization(
            w_max=float(algo_cfg["w_max"]),
            w_min=float(algo_cfg["w_min"]),
            c1=float(algo_cfg["c1"]),
            c2=float(algo_cfg["c2"]),
            vel_clamp_factor=float(algo_cfg["vel_clamp_factor"]),
            bpso_v_max=float(algo_cfg["bpso_v_max"]),
            **common_kwargs,
        )

    if algorithm_id == "DE":
        return DifferentialEvolution(
            mutation_factor=float(algo_cfg["mutation_factor"]),
            crossover_rate=float(algo_cfg["crossover_rate"]),
            **common_kwargs,
        )

    if algorithm_id == "ACO":
        return AntColonyOptimization(
            alpha=float(algo_cfg["alpha"]),
            beta=float(algo_cfg["beta"]),
            evaporation_rate=float(algo_cfg["evaporation_rate"]),
            pheromone_deposit=float(algo_cfg["pheromone_deposit"]),
            elitist_weight=float(algo_cfg["elitist_weight"]),
            **common_kwargs,
        )

    if algorithm_id == "AIS":
        return ArtificialImmuneSystem(
            selection_ratio=float(algo_cfg["selection_ratio"]),
            clone_factor=float(algo_cfg["clone_factor"]),
            mutation_decay=float(algo_cfg["mutation_decay"]),
            replace_rate=float(algo_cfg["replace_rate"]),
            **common_kwargs,
        )

    raise ValueError(f"Algoritmo desconocido: {algorithm_id}")


# ============================================================================
# RENDERIZADO VISUAL DE UNA SOLUCIÓN SEGÚN EL PROBLEMA
# ============================================================================

def render_problem_solution_visuals(
    problem: BaseProblem,
    solution: np.ndarray,
    title_suffix: str = "",
    key_prefix: str = "main",
) -> None:
    """
    Renderiza la visualización gráfica del dominio para cualquier solución dada
    (tanto para la mejor solución global como para cada uno de los 4 hitos del historial).
    """
    prob_id = problem.problem_id

    if prob_id == "ackley" and isinstance(problem, AckleyProblem):
        fig_contour, fig_coords = create_ackley_figures(problem, solution, title_suffix=title_suffix)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_contour, width="stretch", key=f"{key_prefix}_ack_contour")
        with col2:
            st.plotly_chart(fig_coords, width="stretch", key=f"{key_prefix}_ack_coords")

    elif prob_id == "onemax" and isinstance(problem, OneMaxProblem):
        fig_bits = create_onemax_figure(problem, solution, title_suffix=title_suffix)
        st.plotly_chart(fig_bits, width="stretch", key=f"{key_prefix}_onemax_bits")

    elif prob_id == "task_assignment" and isinstance(problem, TaskAssignmentProblem):
        table_rows, fig_loads, fig_heatmap = create_task_assignment_visuals(
            problem, solution, title_suffix=title_suffix
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_loads, width="stretch", key=f"{key_prefix}_ta_loads")
        with col2:
            st.plotly_chart(fig_heatmap, width="stretch", key=f"{key_prefix}_ta_heat")

        st.markdown(f"**Tabla Detallada de Asignación de Tareas {title_suffix}:**")
        df_assign = pd.DataFrame(table_rows)
        st.dataframe(df_assign, width="stretch", hide_index=True)

    elif prob_id == "delivery_routing" and isinstance(problem, DeliveryRoutingProblem):
        fig_map, fig_graph, fig_weight = create_delivery_routing_figures(
            problem, solution, title_suffix=title_suffix
        )
        col_map, col_graph = st.columns(2)
        with col_map:
            st.plotly_chart(fig_map, width="stretch", key=f"{key_prefix}_gdl_map")
        with col_graph:
            st.plotly_chart(fig_graph, width="stretch", key=f"{key_prefix}_gdl_graph")

        st.plotly_chart(fig_weight, width="stretch", key=f"{key_prefix}_gdl_weight")

        details = problem.solution_details(solution)
        stops_df = pd.DataFrame(details["ordered_stops"])[
            [
                "stop_order",
                "name",
                "zone",
                "lat",
                "lon",
                "weight_kg",
                "cumulative_weight_kg",
            ]
        ].rename(
            columns={
                "stop_order": "Orden Parada",
                "name": "Punto en Guadalajara",
                "zone": "Zona",
                "lat": "Latitud",
                "lon": "Longitud",
                "weight_kg": "Peso Paquete (kg)",
                "cumulative_weight_kg": "Peso Acumulado (kg)",
            }
        )
        with st.expander(f"📋 Ver itinerario detallado de paradas en Guadalajara {title_suffix}", expanded=False):
            st.dataframe(stops_df, width="stretch", hide_index=True)


# ============================================================================
# SECCIONES DE CONFIGURACIÓN POR PASOS (PASO 1, PASO 2, PASO 3)
# ============================================================================

def render_step_1_problem(reset_token: int) -> Tuple[str, Dict[str, Any]]:
    """
    Paso 1: Elegir tipo de problema y configurar sus parámetros específicos.
    """
    st.subheader("Paso 1: Selección y Configuración del Problema")

    prob_keys = list(PROBLEM_CATALOG.keys())
    current_idx = prob_keys.index(st.session_state["problem_id"])

    selected_prob = st.selectbox(
        "Selecciona el problema de optimización:",
        options=prob_keys,
        index=current_idx,
        format_func=lambda k: PROBLEM_CATALOG[k]["label"],
        help="Elige uno de los 4 dominios disponibles: Continuo, Binario, Discreto o Problema Real en Guadalajara.",
        key=f"prob_select_{reset_token}",
    )

    if selected_prob != st.session_state["problem_id"]:
        st.session_state["problem_id"] = selected_prob
        st.session_state["sense_choice"] = PROBLEM_CATALOG[selected_prob]["default_sense"]

    st.caption(f"🏷️ **Representación:** `{PROBLEM_CATALOG[selected_prob]['badge']}`")

    problem_cfg: Dict[str, Any] = {"problem_id": selected_prob}

    # --- 1A. Configuración Continuo (Ackley) ---
    if selected_prob == "ackley":
        c1, c2, c3 = st.columns(3)
        with c1:
            dim = st.number_input(
                "Dimensión (D)",
                value=10,
                step=1,
                help="Número de variables continuas x_1, ..., x_D de la función de Ackley (rango válido: 1 a 200).",
                key=f"ack_dim_{reset_token}",
            )
        with c2:
            lower_b = st.number_input(
                "Límite Inferior (min)",
                value=-32.768,
                step=1.0,
                format="%.3f",
                help="Valor mínimo permitido para cada coordenada x_i (estándar Ackley: -32.768).",
                key=f"ack_min_{reset_token}",
            )
        with c3:
            upper_b = st.number_input(
                "Límite Superior (max)",
                value=32.768,
                step=1.0,
                format="%.3f",
                help="Valor máximo permitido para cada coordenada x_i (estándar Ackley: +32.768).",
                key=f"ack_max_{reset_token}",
            )

        problem_cfg.update(
            {
                "dimension": dim,
                "lower_bound": lower_b,
                "upper_bound": upper_b,
            }
        )

    # --- 1B. Configuración Binario (OneMax) ---
    elif selected_prob == "onemax":
        n_bits = st.number_input(
            "Longitud del Vector Binario (N)",
            value=50,
            step=2,
            help="Cantidad de bits {0, 1} del cromosoma (rango válido: 2 a 1000).",
            key=f"onemax_n_{reset_token}",
        )
        problem_cfg["n_bits"] = n_bits

    # --- 1C. Configuración Discreto (Asignación de Tareas) ---
    elif selected_prob == "task_assignment":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_tasks = st.number_input(
                "Número de Tareas (T)",
                value=12,
                step=1,
                help="Cantidad total de tareas a asignar (rango válido: 2 a 100).",
                key=f"ta_tasks_{reset_token}",
            )
        with c2:
            n_res = st.number_input(
                "Número de Recursos (R)",
                value=4,
                step=1,
                help="Cantidad de recursos disponibles {0, ..., R-1} (rango válido: 2 a 50).",
                key=f"ta_res_{reset_token}",
            )
        with c3:
            max_cap = st.number_input(
                "Capacidad Máx. por Recurso",
                value=3.0,
                step=1.0,
                help="Número máximo de tareas que puede atender cada recurso sin penalización (debe ser > 0).",
                key=f"ta_cap_{reset_token}",
            )
        with c4:
            pen_factor = st.number_input(
                "Penalización por Exceso",
                value=120.0,
                step=10.0,
                help="Costo adicional sumado por cada unidad de carga que exceda la capacidad máxima (>= 0).",
                key=f"ta_pen_{reset_token}",
            )

        col_seed, col_mode = st.columns([1, 2])
        with col_seed:
            mat_seed = st.number_input(
                "Semilla de Matriz de Costos",
                value=42,
                step=1,
                help="Semilla entera (>= 0) para generar aleatoriamente la matriz de costos T x R.",
                key=f"ta_mseed_{reset_token}",
            )
        with col_mode:
            edit_matrix = st.checkbox(
                "✏️ Mostrar y editar manualmente la matriz de costos T × R",
                value=False,
                help="Activa esta casilla para inspeccionar o modificar celda por celda los costos C[t, r].",
                key=f"ta_edit_chk_{reset_token}",
            )

        safe_t = int(n_tasks) if not _is_missing_or_nan(n_tasks) and 2 <= int(n_tasks) <= 100 else 12
        safe_r = int(n_res) if not _is_missing_or_nan(n_res) and 2 <= int(n_res) <= 50 else 4
        safe_seed = int(mat_seed) if not _is_missing_or_nan(mat_seed) and int(mat_seed) >= 0 else 42

        temp_prob = TaskAssignmentProblem(
            n_tasks=safe_t,
            n_resources=safe_r,
            max_capacity=3.0,
            penalty_factor=120.0,
            seed=safe_seed,
        )
        active_matrix = temp_prob.cost_matrix

        if edit_matrix:
            st.caption("Edita cualquier celda de la matriz $T \\times R$ (no se permiten valores negativos ni vacíos):")
            df_cost = temp_prob.get_cost_dataframe()
            edited_df = st.data_editor(
                df_cost,
                width="stretch",
                key=f"ta_editor_{safe_t}_{safe_r}_{safe_seed}_{reset_token}",
            )
            active_matrix = edited_df.to_numpy(dtype=float)

        problem_cfg.update(
            {
                "n_tasks": n_tasks,
                "n_resources": n_res,
                "max_capacity": max_cap,
                "penalty_factor": pen_factor,
                "matrix_seed": mat_seed,
                "cost_matrix": active_matrix,
            }
        )

    # --- 1D. Configuración Problema Real (Ruteo Guadalajara) ---
    elif selected_prob == "delivery_routing":
        c1, c2, c3 = st.columns(3)
        with c1:
            n_clients = st.number_input(
                "Número de Clientes en GDL (N)",
                value=12,
                step=1,
                help="Cantidad de puntos de entrega activos en Guadalajara (rango válido: 2 a 12).",
                key=f"gdl_n_{reset_token}",
            )
        with c2:
            veh_cap = st.number_input(
                "Capacidad Máx. Vehículo (kg)",
                value=235.0,
                step=5.0,
                help="Capacidad máxima de carga del vehículo en kg (debe ser > 0).",
                key=f"gdl_cap_{reset_token}",
            )
        with c3:
            pen_kg = st.number_input(
                "Penalización por kg Excedido",
                value=15.0,
                step=1.0,
                help="Factor de penalización por cada kg de sobrepeso (>= 0).",
                key=f"gdl_pen_{reset_token}",
            )

        safe_n = int(n_clients) if not _is_missing_or_nan(n_clients) and 2 <= int(n_clients) <= 12 else 12
        safe_cap = float(veh_cap) if not _is_missing_or_nan(veh_cap) and float(veh_cap) > 0 else 235.0

        temp_gdl = DeliveryRoutingProblem(
            n_clients=safe_n,
            vehicle_capacity=safe_cap,
            penalty_per_kg=15.0,
        )

        custom_weights: Optional[List[float]] = None
        with st.expander("📍 Ver / Editar los puntos reales de entrega en Guadalajara (`data/rutas_gdl.json`)"):
            gdl_df = pd.DataFrame(temp_gdl.clients)[
                ["id", "name", "zone", "lat", "lon", "weight_kg"]
            ].rename(
                columns={
                    "id": "ID",
                    "name": "Punto de Entrega",
                    "zone": "Zona GDL",
                    "lat": "Latitud",
                    "lon": "Longitud",
                    "weight_kg": "Peso Paquete (kg)",
                }
            )
            edited_gdl = st.data_editor(
                gdl_df,
                disabled=["ID", "Punto de Entrega", "Zona GDL", "Latitud", "Longitud"],
                width="stretch",
                hide_index=True,
                key=f"gdl_editor_{safe_n}_{reset_token}",
            )
            custom_weights = [float(w) for w in edited_gdl["Peso Paquete (kg)"].tolist()]

        total_w = float(np.nansum(custom_weights)) if custom_weights is not None else temp_gdl.total_package_weight
        if not _is_missing_or_nan(veh_cap) and float(veh_cap) > 0:
            if total_w > float(veh_cap):
                st.warning(
                    f"⚖️ **Aviso de Capacidad:** El peso total de los {safe_n} paquetes es "
                    f"**{total_w:.1f} kg**, lo cual excede la capacidad del vehículo (**{float(veh_cap):.1f} kg**) "
                    f"por **{total_w - float(veh_cap):.1f} kg**. Se aplicará penalización por sobrecarga."
                )
            else:
                st.info(
                    f"📦 Peso total de los {safe_n} paquetes en Guadalajara: **{total_w:.1f} kg** "
                    f"(Capacidad del vehículo: **{float(veh_cap):.1f} kg** — Holgura: **{float(veh_cap) - total_w:.1f} kg**)."
                )

        problem_cfg.update(
            {
                "n_clients": n_clients,
                "vehicle_capacity": veh_cap,
                "penalty_per_kg": pen_kg,
                "custom_weights": custom_weights,
            }
        )

    return selected_prob, problem_cfg


def render_step_2_algorithm_and_sense(
    problem_id: str,
    reset_token: int,
) -> Tuple[str, str, bool, str]:
    """
    Paso 2: Elegir algoritmo metaheurístico y objetivo (Minimizar / Maximizar).
    Verifica `COMPATIBILITY_MATRIX` y muestra alerta explicativa si no son compatibles.
    """
    st.subheader("Paso 2: Algoritmo Metaheurístico y Sentido de Optimización")

    algo_keys = list(ALGORITHM_CATALOG.keys())
    curr_algo_idx = (
        algo_keys.index(st.session_state["algorithm_id"])
        if st.session_state["algorithm_id"] in algo_keys
        else 0
    )

    col_algo, col_sense = st.columns([2, 1])
    with col_algo:
        selected_algo = st.selectbox(
            "Selecciona el Algoritmo Metaheurístico:",
            options=algo_keys,
            index=curr_algo_idx,
            format_func=lambda k: ALGORITHM_CATALOG[k]["label"],
            help="Elige uno de los 5 algoritmos reales implementados en algorithms/.",
            key=f"algo_select_{reset_token}",
        )
        st.session_state["algorithm_id"] = selected_algo

    with col_sense:
        sense_options = ["Minimizar", "Maximizar"]
        curr_sense_idx = (
            sense_options.index(st.session_state["sense_choice"])
            if st.session_state["sense_choice"] in sense_options
            else 0
        )
        selected_sense = st.radio(
            "Objetivo de Optimización:",
            options=sense_options,
            index=curr_sense_idx,
            horizontal=True,
            help="Define si el wrapper de evaluación buscará minimizar o maximizar la función objetivo.",
            key=f"sense_radio_{problem_id}_{reset_token}",
        )
        st.session_state["sense_choice"] = selected_sense

    is_compat, compat_msg = check_compatibility(problem_id, selected_algo)

    if is_compat:
        st.success(f"✅ **Combinación Compatible ({selected_algo} + {PROBLEM_CATALOG[problem_id]['short']}):** {compat_msg}")
    else:
        st.error(
            f"🚫 **Combinación No Compatible ({selected_algo} + {PROBLEM_CATALOG[problem_id]['short']}):**\n\n"
            f"{compat_msg}\n\n"
            "*El botón **Ejecutar** ha sido deshabilitado automáticamente. Selecciona un algoritmo o problema compatible.*"
        )

    with st.expander("🔎 Consultar Matriz Completa de Compatibilidad (`COMPATIBILITY_MATRIX`)"):
        matrix_rows = []
        for p_key, p_info in PROBLEM_CATALOG.items():
            row: Dict[str, Any] = {"Problema": p_info["short"]}
            for a_key in algo_keys:
                ok, _ = check_compatibility(p_key, a_key)
                row[a_key] = "✅ Sí" if ok else "❌ No"
            matrix_rows.append(row)
        st.dataframe(pd.DataFrame(matrix_rows), width="stretch", hide_index=True)

    return selected_algo, selected_sense, is_compat, compat_msg


def render_step_3_hyperparameters(
    algorithm_id: str,
    reset_token: int,
) -> Dict[str, Any]:
    """
    Paso 3: Formulario dinámico que muestra únicamente los parámetros del algoritmo elegido,
    con valores iniciales razonables, tooltips explicativos y semilla aleatoria opcional.
    """
    st.subheader(f"Paso 3: Hiperparámetros de {ALGORITHM_CATALOG[algorithm_id]['short']}")
    st.caption(ALGORITHM_CATALOG[algorithm_id]["summary"])

    algo_cfg: Dict[str, Any] = {"algorithm_id": algorithm_id}

    # --- Parámetros generales de presupuesto computacional y parada ---
    st.markdown("##### ⚙️ Criterios Generales de Iteración, Población y Semilla")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        max_iters = st.number_input(
            "Máximo de Iteraciones",
            value=80,
            step=10,
            help="Número máximo de iteraciones o generaciones (rango válido: 1 a 5000).",
            key=f"p_iters_{algorithm_id}_{reset_token}",
        )
    with g2:
        pop_size = st.number_input(
            "Tamaño de Población / Enjambre",
            value=40,
            step=5,
            help="Cantidad de soluciones candidatas por iteración (rango válido: 4 a 1000).",
            key=f"p_pop_{algorithm_id}_{reset_token}",
        )
    with g3:
        use_early = st.checkbox(
            "Activar Parada Anticipada",
            value=False,
            help="Detiene la ejecución antes de max_iterations si el mejor fitness no mejora durante N iteraciones seguidas.",
            key=f"p_early_chk_{algorithm_id}_{reset_token}",
        )
        if use_early:
            patience = st.number_input(
                "Paciencia (Iteraciones sin mejora)",
                value=20,
                step=1,
                help="Iteraciones consecutivas sin mejora antes de detener por convergencia anticipada (1 .. max_iterations).",
                key=f"p_patience_{algorithm_id}_{reset_token}",
            )
        else:
            patience = None
    with g4:
        use_seed = st.checkbox(
            "Usar Semilla Aleatoria Fija",
            value=True,
            help="Garantiza reproducibilidad exacta de principio a fin usando la semilla indicada.",
            key=f"p_seed_chk_{algorithm_id}_{reset_token}",
        )
        if use_seed:
            seed_val = st.number_input(
                "Valor de Semilla (Seed)",
                value=42,
                step=1,
                help="Semilla entera no negativa (>= 0) para el generador pseudoaleatorio de NumPy.",
                key=f"p_seed_val_{algorithm_id}_{reset_token}",
            )
        else:
            seed_val = None

    algo_cfg.update(
        {
            "max_iterations": max_iters,
            "population_size": pop_size,
            "use_early_stopping": bool(use_early),
            "early_stopping_patience": patience,
            "tol": 1e-8,
            "use_seed": bool(use_seed),
            "seed": seed_val,
        }
    )

    st.markdown(f"##### 🎛️ Parámetros Específicos de `{algorithm_id}`")

    # --- 3A. Hiperparámetros exclusivos de GA ---
    if algorithm_id == "GA":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            cx_rate = st.slider(
                "Probabilidad de Cruza (Pc)",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                help="Fracción de parejas de padres que se recombinan en cada generación (0.0 a 1.0).",
                key=f"ga_cx_{reset_token}",
            )
        with c2:
            mut_rate = st.slider(
                "Probabilidad de Mutación (Pm)",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                help="Probabilidad de aplicar perturbación genética a un descendiente (0.0 a 1.0).",
                key=f"ga_mut_{reset_token}",
            )
        with c3:
            tourn_size = st.number_input(
                "Tamaño de Torneo (k)",
                value=3,
                step=1,
                help="Número de individuos que compiten en cada torneo de selección (2 <= k <= población).",
                key=f"ga_tourn_{reset_token}",
            )
        with c4:
            elitism = st.number_input(
                "Individuos Élite",
                value=2,
                step=1,
                help="Cantidad de mejores soluciones que pasan intactas a la siguiente generación (1 <= élite < población).",
                key=f"ga_elit_{reset_token}",
            )
        with c5:
            blx_a = st.number_input(
                "Factor BLX-alpha (Continuo)",
                value=0.35,
                step=0.05,
                help="Expansión del intervalo de cruza real BLX-alpha (0.0 a 2.0).",
                key=f"ga_blx_{reset_token}",
            )
        algo_cfg.update(
            {
                "crossover_rate": cx_rate,
                "mutation_rate": mut_rate,
                "tournament_size": tourn_size,
                "elitism_count": elitism,
                "blx_alpha": blx_a,
            }
        )

    # --- 3B. Hiperparámetros exclusivos de PSO ---
    elif algorithm_id == "PSO":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            w_max = st.number_input(
                "Inercia Inicial (w_max)",
                value=0.90,
                step=0.05,
                help="Peso de inercia al inicio de la búsqueda para favorecer exploración global (0 < w_max <= 2.0).",
                key=f"pso_wmax_{reset_token}",
            )
        with c2:
            w_min = st.number_input(
                "Inercia Final (w_min)",
                value=0.40,
                step=0.05,
                help="Peso de inercia al final de la búsqueda (debe cumplir 0 < w_min <= w_max).",
                key=f"pso_wmin_{reset_token}",
            )
        with c3:
            coeff_c1 = st.number_input(
                "Coeficiente Cognitivo (c1)",
                value=1.80,
                step=0.1,
                help="Atracción de cada partícula hacia su mejor posición histórica personal pbest (>= 0).",
                key=f"pso_c1_{reset_token}",
            )
        with c4:
            coeff_c2 = st.number_input(
                "Coeficiente Social (c2)",
                value=1.80,
                step=0.1,
                help="Atracción de cada partícula hacia la mejor posición global del enjambre gbest (>= 0).",
                key=f"pso_c2_{reset_token}",
            )
        with c5:
            v_clamp = st.slider(
                "Factor Límite Velocidad",
                min_value=0.05,
                max_value=1.0,
                value=0.20,
                step=0.05,
                help="Fracción del rango [min, max] usada como velocidad máxima |v_max| en continuo.",
                key=f"pso_vclamp_{reset_token}",
            )
        algo_cfg.update(
            {
                "w_max": w_max,
                "w_min": w_min,
                "c1": coeff_c1,
                "c2": coeff_c2,
                "vel_clamp_factor": v_clamp,
                "bpso_v_max": 6.0,
            }
        )

    # --- 3C. Hiperparámetros exclusivos de DE ---
    elif algorithm_id == "DE":
        c1, c2 = st.columns(2)
        with c1:
            f_factor = st.number_input(
                "Factor de Escala Diferencial (F)",
                value=0.70,
                step=0.05,
                help="Amplificación de la diferencia vectorial x_r2 - x_r3 en DE/rand/1/bin (rango válido: 0.05 a 2.0).",
                key=f"de_f_{reset_token}",
            )
        with c2:
            cr_rate = st.slider(
                "Tasa de Cruza Binomial (CR)",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                help="Probabilidad por coordenada de heredar componentes del vector mutante donante (0.0 a 1.0).",
                key=f"de_cr_{reset_token}",
            )
        algo_cfg.update(
            {
                "mutation_factor": f_factor,
                "crossover_rate": cr_rate,
            }
        )

    # --- 3D. Hiperparámetros exclusivos de ACO ---
    elif algorithm_id == "ACO":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            alpha = st.number_input(
                "Importancia de Feromona (α)",
                value=1.0,
                step=0.2,
                help="Exponente α que pondera el rastro histórico de feromona τ^α (>= 0).",
                key=f"aco_alpha_{reset_token}",
            )
        with c2:
            beta = st.number_input(
                "Importancia Heurística (β)",
                value=2.2,
                step=0.2,
                help="Exponente β que pondera la visibilidad heurística η^β (>= 0).",
                key=f"aco_beta_{reset_token}",
            )
        with c3:
            evap = st.slider(
                "Tasa de Evaporación (ρ)",
                min_value=0.01,
                max_value=0.95,
                value=0.18,
                step=0.02,
                help="Fracción de feromona que se evapora al final de cada iteración (0.01 a 0.95).",
                key=f"aco_evap_{reset_token}",
            )
        with c4:
            q_dep = st.number_input(
                "Factor de Depósito (Q)",
                value=1.5,
                step=0.5,
                help="Cantidad base de feromona depositada por las hormigas destacadas (> 0).",
                key=f"aco_q_{reset_token}",
            )
        with c5:
            elit_w = st.number_input(
                "Peso Hormiga Élite",
                value=2.0,
                step=0.5,
                help="Refuerzo adicional de feromona sobre la mejor solución global (>= 0).",
                key=f"aco_elit_{reset_token}",
            )
        algo_cfg.update(
            {
                "alpha": alpha,
                "beta": beta,
                "evaporation_rate": evap,
                "pheromone_deposit": q_dep,
                "elitist_weight": elit_w,
            }
        )

    # --- 3E. Hiperparámetros exclusivos de AIS (CLONALG) ---
    elif algorithm_id == "AIS":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sel_ratio = st.slider(
                "Proporción Selección Clonal",
                min_value=0.10,
                max_value=1.0,
                value=0.50,
                step=0.05,
                help="Fracción del repertorio de anticuerpos seleccionada para clonación (0.10 a 1.0).",
                key=f"ais_sel_{reset_token}",
            )
        with c2:
            clone_beta = st.slider(
                "Factor de Expansión Clonal (β)",
                min_value=0.10,
                max_value=2.5,
                value=0.60,
                step=0.10,
                help="Controla cuántos clones produce cada anticuerpo seleccionado proporcionalmente a su afinidad.",
                key=f"ais_beta_{reset_token}",
            )
        with c3:
            mut_decay = st.number_input(
                "Decaimiento Hipermutación (ρ)",
                value=2.5,
                step=0.2,
                help="Controla la tasa exp(-ρ·A): debe ser > 0.1.",
                key=f"ais_decay_{reset_token}",
            )
        with c4:
            rep_rate = st.slider(
                "Tasa Edición Receptores (d)",
                min_value=0.0,
                max_value=0.50,
                value=0.15,
                step=0.05,
                help="Fracción de los peores anticuerpos reemplazada por nuevos anticuerpos aleatorios (0.0 a 0.50).",
                key=f"ais_rep_{reset_token}",
            )
        algo_cfg.update(
            {
                "selection_ratio": sel_ratio,
                "clone_factor": clone_beta,
                "mutation_decay": mut_decay,
                "replace_rate": rep_rate,
            }
        )

    return algo_cfg


# ============================================================================
# SECCIÓN DE VISUALIZACIÓN DE RESULTADOS, 4 HITOS Y EXPORTACIÓN
# ============================================================================

def render_results_section(
    result: Dict[str, Any],
    problem: BaseProblem,
    problem_cfg: Dict[str, Any],
    algo_cfg: Dict[str, Any],
) -> None:
    """
    Muestra el tablero completo de resultados:
    - Resumen de métricas y configuración.
    - Curva de convergencia interactiva con Plotly.
    - Visualización gráfica de la mejor solución encontrada según el problema.
    - Pestañas para inspeccionar los 4 hitos obligatorios: Iteración 0, 1/3, 2/3 e Iteración Final.
    - Botones de descarga en JSON y CSV.
    """
    st.markdown("---")
    st.header("📊 Progreso, Convergencia y Resultados de la Ejecución")

    history = result.get("history", [])
    best_fit = float(result["best_fitness"])
    init_fit = float(history[0]["best_fitness"]) if history else best_fit
    delta_fit = best_fit - init_fit

    # 1. Tarjetas de métricas principales
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="🏆 Mejor Fitness",
            value=f"{best_fit:.4f}",
            delta=f"{delta_fit:+.4f} vs Iter 0",
            delta_color="inverse" if result["sense"] == "minimize" else "normal",
        )
    with m2:
        st.metric(
            label="🔄 Iteraciones Ejecutadas",
            value=f"{result['total_iterations']} / {int(algo_cfg['max_iterations'])}",
        )
    with m3:
        st.metric(
            label="🧮 Evaluaciones (FE)",
            value=f"{result['n_evaluations']:,}",
        )
    with m4:
        st.metric(
            label="⏱️ Tiempo de Cómputo",
            value=f"{result['execution_time_sec']:.3f} s",
        )
    with m5:
        details = result.get("solution_details", {})
        feasible_txt = "✅ Factible" if details.get("is_feasible", True) else f"⚠️ Penalizado (+{details.get('penalty', 0):.1f})"
        st.metric(
            label="📌 Estado de Restricciones",
            value=feasible_txt,
        )

    seed_display = algo_cfg.get("seed") if algo_cfg.get("use_seed") else "Aleatoria (Sin fijar)"
    st.info(
        f"**Motivo de Término (`stop_reason`):** `{result['stop_reason']}`  |  "
        f"**Algoritmo:** `{result['algorithm_name']}`  |  "
        f"**Problema:** `{problem.name}`  |  "
        f"**Sentido:** `{result['sense'].upper()}`  |  "
        f"**Semilla:** `{seed_display}`"
    )

    # Resumen expandible de la Mejor Solución y Configuración Utilizada
    col_sol, col_cfg = st.columns(2)
    with col_sol:
        with st.expander("🧬 Ver Vector de la Mejor Solución Encontrada (`best_solution`)", expanded=True):
            sol_list = np.asarray(result["best_solution"]).tolist()
            if problem.problem_id == "ackley":
                st.code(np.array2string(np.asarray(sol_list), precision=5, separator=", "))
            else:
                st.code(str(sol_list))
            if details.get("penalty", 0.0) > 0:
                st.caption(
                    f"Costo Base: `{details.get('base_cost', 0.0):.4f}` + "
                    f"Penalización: `{details.get('penalty', 0.0):.4f}` = "
                    f"**Fitness Total: `{best_fit:.4f}`**"
                )

    with col_cfg:
        with st.expander("🛠️ Ver Configuración Utilizada (Problema y Algoritmo)", expanded=True):
            cfg_display = {
                "problema": {
                    k: v for k, v in problem_cfg.items() if k not in ("cost_matrix", "custom_weights")
                },
                "algoritmo": algo_cfg,
            }
            st.json(cfg_display)

    # 2. Gráfica Interactiva de Convergencia con Plotly
    st.subheader("📈 Curva Interactiva de Convergencia (Mejor Fitness vs. Iteración)")
    fig_conv = create_convergence_figure(
        history=history,
        algorithm_name=result["algorithm_name"],
        problem_name=problem.name,
        sense=result["sense"],
    )
    st.plotly_chart(fig_conv, width="stretch", key="main_convergence_chart")

    # 3. Visualización Específica de la Mejor Solución Global
    st.subheader(f"🗺️ Visualización de la Mejor Solución — {PROBLEM_CATALOG[problem.problem_id]['short']}")
    render_problem_solution_visuals(
        problem=problem,
        solution=np.asarray(result["best_solution"]),
        title_suffix="(Mejor Solución Final)",
        key_prefix="best_global",
    )

    # 4. Inspector de los 4 Hitos Obligatorios del Historial (Iteración 0, 1/3, 2/3 y Final)
    st.markdown("---")
    st.subheader("⏱️ Inspección de los 4 Hitos Obligatorios del Historial (Iteración 0, 1/3, 2/3 y Final)")
    st.caption(
        "Cada pestaña refleja los datos reales almacenados en `history` para la Iteración 0 (población inicial), "
        "1/3 del recorrido, 2/3 del recorrido y la Iteración Final."
    )

    milestones = extract_four_milestones(history)
    if milestones:
        ms_summary_rows = []
        for m in milestones:
            ms_summary_rows.append(
                {
                    "Hito": m["label"],
                    "Iteración Exacta en Historial": m["recorded_iteration"],
                    "Mejor Fitness": round(m["best_fitness"], 6),
                    "Fitness Promedio Población": round(m["mean_fitness"], 6),
                    "Muestra de Solución (Primeros 8 genes)": str(
                        np.round(m["best_solution"][:8], 4).tolist()
                    ),
                }
            )
        st.dataframe(pd.DataFrame(ms_summary_rows), width="stretch", hide_index=True)

        tab_labels = [
            f"🏁 {m['label']} (Iter {m['recorded_iteration']})" for m in milestones
        ]
        tabs = st.tabs(tab_labels)
        for idx, (tab, ms) in enumerate(zip(tabs, milestones)):
            with tab:
                k1, k2, k3 = st.columns(3)
                k1.metric("Hito del Historial", ms["label"])
                k2.metric("Iteración Real", f"{ms['recorded_iteration']} / {result['total_iterations']}")
                k3.metric("Mejor Fitness en este Hito", f"{ms['best_fitness']:.6f}")

                st.markdown("**Mejor Solución Registrada en este Hito (`best_solution`):**")
                st.code(
                    np.array2string(ms["best_solution"], precision=4, separator=", ")
                    if problem.problem_id == "ackley"
                    else str(ms["best_solution"].tolist())
                )

                render_problem_solution_visuals(
                    problem=problem,
                    solution=ms["best_solution"],
                    title_suffix=f"[{ms['label']} — Iter {ms['recorded_iteration']}]",
                    key_prefix=f"milestone_{idx}",
                )

    # 5. Botones de Descarga en JSON y CSV
    st.markdown("---")
    st.subheader("💾 Exportar Resultados e Historial Completo")
    dl_col1, dl_col2 = st.columns(2)

    json_payload = export_results_to_json(
        result=result,
        problem_config={k: v for k, v in problem_cfg.items() if k != "cost_matrix"},
        algorithm_params=algo_cfg,
    )
    csv_payload = export_history_to_csv(result=result)

    file_stem = f"metaheuristic_{problem.problem_id}_{result['algorithm_id']}_{result['sense']}"

    with dl_col1:
        st.download_button(
            label="📥 Descargar Resultados e Historial Completo (.JSON)",
            data=json_payload.encode("utf-8"),
            file_name=f"{file_stem}.json",
            mime="application/json",
            width="stretch",
        )
    with dl_col2:
        st.download_button(
            label="📥 Descargar Historial de Convergencia (.CSV)",
            data=csv_payload.encode("utf-8"),
            file_name=f"{file_stem}_history.csv",
            mime="text/csv",
            width="stretch",
        )


# ============================================================================
# FUNCIÓN PRINCIPAL DE LA APLICACIÓN STREAMLIT
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Metaheuristic Studio",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    reset_token = int(st.session_state["reset_counter"])

    # Barra lateral con guía de pasos, estado actual y accesos rápidos
    with st.sidebar:
        st.title("🧬 Metaheuristic Studio")
        st.markdown(
            "Plataforma interactiva para configurar, validar, ejecutar y comparar "
            "**algoritmos metaheurísticos** sobre problemas continuos, binarios, discretos "
            "y ruteo real en **Guadalajara, Jalisco**."
        )
        st.markdown("---")

        st.markdown("### 🧭 Flujo de Trabajo")
        step_options = [
            "1️⃣ Paso 1: Problema",
            "2️⃣ Paso 2: Algoritmo y Objetivo",
            "3️⃣ Paso 3: Hiperparámetros",
            "4️⃣ Paso 4: Ejecución y Resultados",
        ]
        active_step = int(st.session_state.get("active_step", 1))
        st.progress(active_step / 4.0, text=f"Etapa activa: Paso {active_step} de 4")

        view_mode = st.radio(
            "Modo de visualización del formulario:",
            options=["Vista Integrada (Pasos 1-3 visibles)", "Vista Guiada Paso a Paso"],
            index=0,
            help="Puedes ver los 3 pasos de configuración simultáneamente o avanzar uno por uno.",
        )

        st.markdown("---")
        st.markdown("### ⚡ Acciones Rápidas")
        if st.button("🔄 Reiniciar Todo", width="stretch"):
            trigger_reset()
            st.rerun()

    # Encabezado principal
    st.title("🧬 Metaheuristic Studio — Laboratorio de Optimización Heurística")

    if view_mode == "Vista Guiada Paso a Paso":
        nav_cols = st.columns(4)
        for idx, label in enumerate(step_options, start=1):
            if nav_cols[idx - 1].button(
                label,
                type="primary" if st.session_state["active_step"] == idx else "secondary",
                width="stretch",
            ):
                st.session_state["active_step"] = idx
                st.rerun()
        st.markdown("---")

    # Ejecución de los formularios de Paso 1, Paso 2 y Paso 3
    with st.container(border=True):
        selected_prob, problem_cfg = render_step_1_problem(reset_token=reset_token)

    with st.container(border=True):
        selected_algo, selected_sense, is_compat, compat_msg = render_step_2_algorithm_and_sense(
            problem_id=selected_prob,
            reset_token=reset_token,
        )

    with st.container(border=True):
        algo_cfg = render_step_3_hyperparameters(
            algorithm_id=selected_algo,
            reset_token=reset_token,
        )

    # Validación estricta antes de habilitar ejecución
    validation_errors = validate_configuration(
        problem_id=selected_prob,
        problem_cfg=problem_cfg,
        algorithm_id=selected_algo,
        algo_cfg=algo_cfg,
    )

    if validation_errors and is_compat:
        st.error(
            "🚫 **Se detectaron parámetros inválidos (vacíos, negativos o fuera de rango). "
            "Corrige los siguientes puntos para habilitar el botón Ejecutar:**\n\n"
            + "\n".join(f"- {err}" for err in validation_errors)
        )

    can_execute = is_compat and (len(validation_errors) == 0)

    # =========================================================================
    # CONTROLES DE EJECUCIÓN (EJECUTAR, DETENER, REINICIAR, REGRESAR)
    # =========================================================================
    st.markdown("### 🎮 Controles de Ejecución")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        run_clicked = st.button(
            "▶️ Ejecutar Algoritmo",
            type="primary",
            disabled=not can_execute,
            width="stretch",
            help=(
                "Inicia la ejecución de la metaheurística seleccionada."
                if can_execute
                else "Ejecución bloqueada: revisa las alertas de compatibilidad o validación de rangos."
            ),
        )
    with btn_col2:
        stop_clicked = st.button(
            "⏹️ Detener",
            on_click=request_stop,
            width="stretch",
            help="Solicita detener la ejecución en la iteración actual y conserva la mejor solución encontrada hasta el momento.",
        )
    with btn_col3:
        reset_clicked = st.button(
            "🔄 Reiniciar",
            on_click=trigger_reset,
            width="stretch",
            help="Limpia los resultados actuales y restaura todos los parámetros a sus valores iniciales.",
        )
    with btn_col4:
        back_clicked = st.button(
            "⬅️ Regresar",
            on_click=go_previous_step,
            width="stretch",
            help="Regresa al paso anterior de configuración o cierra la vista de resultados.",
        )

    if reset_clicked:
        st.rerun()

    if back_clicked and st.session_state.get("execution_result") is not None:
        st.session_state["execution_result"] = None
        st.info("Has regresado al panel de configuración. Puedes modificar los parámetros y volver a ejecutar.")

    if stop_clicked and not st.session_state.get("is_running", False):
        st.info("⏹️ Bandera de parada activada. Cualquier ejecución en curso se ha detenido.")

    # =========================================================================
    # LÓGICA DE EJECUCIÓN CON PROGRESO INTERACTIVO
    # =========================================================================
    if run_clicked and can_execute:
        st.session_state["stop_requested"] = False
        st.session_state["is_running"] = True
        st.session_state["active_step"] = 4

        problem_instance = build_problem_instance(selected_prob, problem_cfg, selected_sense)
        algorithm_instance = build_algorithm_instance(selected_algo, algo_cfg, selected_sense)

        progress_bar = st.progress(0.0, text="Evaluando población inicial (Iteración 0)...")
        max_it = int(algo_cfg["max_iterations"])

        def on_step(step_entry: Dict[str, Any]) -> None:
            it_num = int(step_entry["iteration"])
            bf = float(step_entry["best_fitness"])
            mf = float(step_entry["mean_fitness"])
            pct = min(1.0, float(it_num) / max(1.0, float(max_it)))
            progress_bar.progress(
                pct,
                text=(
                    f"Ejecutando {selected_algo} — Iteración {it_num}/{max_it} | "
                    f"Mejor Fitness: {bf:.4f} | Promedio: {mf:.4f}"
                ),
            )

        def check_stop_flag() -> bool:
            return bool(st.session_state.get("stop_requested", False))

        result = algorithm_instance.run(
            problem=problem_instance,
            sense=selected_sense,
            seed=algorithm_instance.seed,
            step_callback=on_step,
            stop_check=check_stop_flag,
        )

        st.session_state["is_running"] = False
        progress_bar.progress(
            1.0,
            text=f"✅ Ejecución completada en {result['total_iterations']} iteraciones ({result['execution_time_sec']:.3f} s).",
        )

        st.session_state["execution_result"] = result
        st.session_state["last_problem_instance"] = problem_instance
        st.session_state["last_problem_config"] = problem_cfg
        st.session_state["last_algorithm_params"] = algo_cfg

    # =========================================================================
    # RENDERIZADO DE RESULTADOS SI EXISTEN EN SESIÓN
    # =========================================================================
    if (
        st.session_state.get("execution_result") is not None
        and st.session_state.get("last_problem_instance") is not None
    ):
        render_results_section(
            result=st.session_state["execution_result"],
            problem=st.session_state["last_problem_instance"],
            problem_cfg=st.session_state["last_problem_config"],
            algo_cfg=st.session_state["last_algorithm_params"],
        )


if __name__ == "__main__":
    main()
