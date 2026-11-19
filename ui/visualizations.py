"""
Módulo de visualizaciones interactivas con Plotly y utilidades de exportación
para Metaheuristic Studio.

Incluye:
- Gráfica interactiva de convergencia (Mejor Fitness y Fitness Promedio vs. Iteración)
  con marcadores para los 4 hitos obligatorios (Iteración 0, 1/3, 2/3 y Final).
- Visualizaciones específicas por dominio:
  1. Continuo (Ackley): Mapa de calor / contorno 2D de la superficie + perfil de las D coordenadas.
  2. Binario (OneMax): Gráfica de barras de bits (0 vs 1) por posición.
  3. Discreto (Asignación de Tareas): Tabla detallada de asignación, mapa de calor T x R y
     gráfica de carga acumulada vs. capacidad máxima por recurso.
  4. Problema Real (Ruteo Guadalajara): Mapa geográfico / grafo 2D de la ruta en Guadalajara
     + gráfica de peso acumulado de paquetes vs. capacidad del vehículo.
- Extracción de los 4 hitos del historial y exportación a JSON y CSV.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from problems.ackley import AckleyProblem
from problems.delivery_routing import DeliveryRoutingProblem
from problems.onemax import OneMaxProblem
from problems.task_assignment import TaskAssignmentProblem


# ============================================================================
# EXTRACCIÓN DE LOS 4 HITOS OBLIGATORIOS DEL HISTORIAL
# ============================================================================

def extract_four_milestones(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extrae los 4 hitos obligatorios del historial real de ejecución:
    - Hito 1: Iteración 0 (Inicial) -> población inicial en `history[0]` (`iteration == 0`).
    - Hito 2: Iteración 1/3 -> tercio exacto del recorrido ejecutado.
    - Hito 3: Iteración 2/3 -> dos tercios exactos del recorrido ejecutado.
    - Hito 4: Iteración Final -> último registro del historial (`history[-1]`).
    """
    if not history:
        return []

    n = len(history)
    idx_0 = 0
    idx_1_3 = min(n - 1, max(0, int(round((n - 1) / 3.0))))
    idx_2_3 = min(n - 1, max(0, int(round(2.0 * (n - 1) / 3.0))))
    idx_final = n - 1

    raw_milestones = [
        ("Iteración 0 (Inicial)", idx_0),
        ("Iteración 1/3", idx_1_3),
        ("Iteración 2/3", idx_2_3),
        ("Iteración Final", idx_final),
    ]

    milestones: List[Dict[str, Any]] = []
    for label, list_idx in raw_milestones:
        entry = history[list_idx]
        rec_iter = int(entry["iteration"])
        milestones.append(
            {
                "label": label,
                "history_index": int(list_idx),
                "display_iteration": rec_iter,
                "recorded_iteration": rec_iter,
                "best_fitness": float(entry["best_fitness"]),
                "mean_fitness": float(entry["mean_fitness"]),
                "best_solution": np.asarray(entry["best_solution"]).copy(),
            }
        )
    return milestones


# ============================================================================
# GRÁFICA DE CONVERGENCIA (MEJOR FITNESS VS ITERACIÓN)
# ============================================================================

def create_convergence_figure(
    history: List[Dict[str, Any]],
    algorithm_name: str,
    problem_name: str,
    sense: str,
) -> go.Figure:
    """
    Construye la gráfica interactiva de convergencia (Mejor Fitness y Fitness Promedio vs. Iteración)
    destacando los 4 hitos del historial.
    """
    iterations = [int(h["iteration"]) for h in history]
    best_vals = [float(h["best_fitness"]) for h in history]
    mean_vals = [float(h["mean_fitness"]) for h in history]

    milestones = extract_four_milestones(history)
    ms_iters = [m["recorded_iteration"] for m in milestones]
    ms_best = [m["best_fitness"] for m in milestones]
    ms_labels = [f"{m['label']} (Iter {m['recorded_iteration']})" for m in milestones]

    sense_label = "Minimización" if "min" in sense.lower() else "Maximización"

    fig = go.Figure()

    # Traza de Fitness Promedio de la población
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=mean_vals,
            mode="lines",
            name="Fitness Promedio Población",
            line=dict(color="#94A3B8", width=2, dash="dot"),
            hovertemplate="Iteración %{x}<br>Promedio: %{y:.6f}<extra></extra>",
        )
    )

    # Traza principal de Mejor Fitness
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=best_vals,
            mode="lines",
            name="Mejor Fitness Acumulado",
            line=dict(color="#2563EB", width=3.2),
            hovertemplate="Iteración %{x}<br>Mejor Fitness: %{y:.6f}<extra></extra>",
        )
    )

    # Marcadores de los 4 hitos obligatorios
    fig.add_trace(
        go.Scatter(
            x=ms_iters,
            y=ms_best,
            mode="markers+text",
            name="Hitos (0, 1/3, 2/3, Final)",
            text=[m["label"].replace(" (Inicial)", "") for m in milestones],
            textposition="top center",
            marker=dict(
                size=11,
                color="#F59E0B",
                symbol="diamond",
                line=dict(width=2, color="#1E293B"),
            ),
            customdata=ms_labels,
            hovertemplate="<b>%{customdata}</b><br>Mejor Fitness: %{y:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Curva de Convergencia — {algorithm_name} en {problem_name} ({sense_label})",
            font=dict(size=16),
        ),
        xaxis_title="Iteración",
        yaxis_title="Valor de Fitness f(x)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=70, b=45),
        height=430,
    )
    return fig


# ============================================================================
# 1. VISUALIZACIÓN PARA PROBLEMA CONTINUO (ACKLEY)
# ============================================================================

def create_ackley_figures(
    problem: AckleyProblem,
    solution: np.ndarray,
    title_suffix: str = "",
) -> Tuple[go.Figure, go.Figure]:
    """
    Devuelve dos figuras de Plotly para el problema continuo de Ackley:
    1. Mapa de calor / contorno 2D de la superficie Ackley proyectado en (x_1, x_2)
       mostrando el óptimo global (0, 0) y la solución actual.
    2. Gráfica de barras / perfil de coordenadas para las D dimensiones del vector solución.
    """
    vec = problem.repair(np.asarray(solution, dtype=float).ravel())
    fitness_val = problem.evaluate_raw(vec)

    # Figura 1: Mapa de calor 2D de Ackley sobre (x_1, x_2)
    xx, yy, zz = problem.generate_surface_mesh(resolution=75)
    x1_val = float(vec[0])
    x2_val = float(vec[1]) if vec.size >= 2 else 0.0

    fig_contour = go.Figure()
    fig_contour.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            colorscale="Viridis",
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title="f(x₁, x₂)"),
            hovertemplate="x₁: %{x:.3f}<br>x₂: %{y:.3f}<br>Ackley 2D: %{z:.4f}<extra></extra>",
        )
    )

    # Óptimo global en (0, 0)
    fig_contour.add_trace(
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers",
            name="Óptimo Global (0, 0)",
            marker=dict(size=14, color="#10B981", symbol="star", line=dict(width=2, color="white")),
            hovertemplate="Óptimo Global (0, 0)<br>f(x*) = 0.0<extra></extra>",
        )
    )

    # Solución proyectada
    fig_contour.add_trace(
        go.Scatter(
            x=[x1_val],
            y=[x2_val],
            mode="markers",
            name="Solución Actual (x₁, x₂)",
            marker=dict(size=13, color="#EF4444", symbol="circle", line=dict(width=2.5, color="white")),
            hovertemplate=(
                f"Solución {title_suffix}<br>"
                "x₁: %{x:.4f}<br>x₂: %{y:.4f}<br>"
                f"f(x) D-dim: {fitness_val:.6f}<extra></extra>"
            ),
        )
    )

    fig_contour.update_layout(
        title=f"Mapa de Calor / Contorno 2D de Ackley (Proyección x₁ vs x₂) {title_suffix}",
        xaxis_title="Coordenada x₁",
        yaxis_title="Coordenada x₂",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=45, r=30, t=65, b=45),
        height=420,
    )

    # Figura 2: Perfil de coordenadas en las D dimensiones
    dims = [f"x_{i + 1}" for i in range(problem.dimension)]
    colors = ["#2563EB" if abs(v) < 1.0 else "#F59E0B" for v in vec]

    fig_coords = go.Figure()
    fig_coords.add_trace(
        go.Bar(
            x=dims,
            y=vec,
            marker_color=colors,
            name="Valor de Coordenada",
            hovertemplate="Dimensión %{x}<br>Valor: %{y:.6f}<extra></extra>",
        )
    )
    fig_coords.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#10B981",
        annotation_text="Óptimo (x_i = 0)",
    )
    fig_coords.update_layout(
        title=f"Valores por Coordenada en D={problem.dimension} Dimensiones {title_suffix}",
        xaxis_title="Dimensión",
        yaxis_title="Valor x_i",
        yaxis=dict(range=[problem.lower_bound, problem.upper_bound]),
        template="plotly_white",
        margin=dict(l=45, r=30, t=65, b=45),
        height=420,
    )

    return fig_contour, fig_coords


# ============================================================================
# 2. VISUALIZACIÓN PARA PROBLEMA BINARIO (ONEMAX)
# ============================================================================

def create_onemax_figure(
    problem: OneMaxProblem,
    solution: np.ndarray,
    title_suffix: str = "",
) -> go.Figure:
    """
    Construye una gráfica interactiva de barras de bits para el problema OneMax,
    mostrando claramente qué posiciones tienen bit 1 (verde) y cuáles tienen bit 0 (gris).
    """
    bits = problem.repair(solution)
    n_bits = problem.n_bits
    ones_count = int(np.sum(bits))
    zeros_count = int(n_bits - ones_count)
    pct = 100.0 * ones_count / max(1, n_bits)

    bit_positions = np.arange(1, n_bits + 1)
    # Para que los bits en 0 también sean visibles como barras bajas en la gráfica:
    visual_heights = np.where(bits == 1, 1.0, 0.12)
    bar_colors = ["#10B981" if b == 1 else "#CBD5E1" for b in bits]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bit_positions,
            y=visual_heights,
            marker_color=bar_colors,
            customdata=bits,
            text=bits,
            textposition="outside",
            name="Estado del Bit",
            hovertemplate="Posición Bit #%{x}<br>Valor Binario: <b>%{customdata}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            f"Vector de Bits OneMax (N={n_bits}) {title_suffix} — "
            f"Unos: {ones_count}/{n_bits} ({pct:.1f}%) | Ceros: {zeros_count}"
        ),
        xaxis_title="Índice del Bit (1 .. N)",
        yaxis=dict(
            title="Estado Binario (0 / 1)",
            tickvals=[0.12, 1.0],
            ticktext=["0 (Apagado)", "1 (Encendido)"],
            range=[0, 1.22],
        ),
        template="plotly_white",
        margin=dict(l=50, r=30, t=65, b=45),
        height=390,
    )
    return fig


# ============================================================================
# 3. VISUALIZACIÓN PARA PROBLEMA DISCRETO (ASIGNACIÓN DE TAREAS)
# ============================================================================

def create_task_assignment_visuals(
    problem: TaskAssignmentProblem,
    solution: np.ndarray,
    title_suffix: str = "",
) -> Tuple[List[Dict[str, Any]], go.Figure, go.Figure]:
    """
    Genera los elementos visuales para el problema de Asignación de Tareas:
    1. Lista de diccionarios (tabla de asignación tarea -> recurso con costos y estado).
    2. Gráfica de barras de carga por recurso vs. capacidad máxima permitida.
    3. Mapa de calor de la matriz de costos T x R con las celdas seleccionadas resaltadas.
    """
    details = problem.solution_details(solution)
    assignment = details["assignment"]
    loads = details["resource_loads"]
    max_caps = details["max_capacity_per_resource"]
    excess = details["resource_excess"]

    # 1. Filas para tabla de asignación
    table_rows: List[Dict[str, Any]] = []
    for t_idx in range(problem.n_tasks):
        r_idx = int(assignment[t_idx])
        cost_val = float(problem.cost_matrix[t_idx, r_idx])
        demand_val = float(problem.task_demands[t_idx])
        is_overloaded = excess[r_idx] > 1e-9
        table_rows.append(
            {
                "Tarea": f"Tarea {t_idx + 1}",
                "Recurso Asignado": f"Recurso {r_idx + 1}",
                "ID Recurso (0..R-1)": r_idx,
                "Costo C[t, r]": round(cost_val, 2),
                "Demanda Tarea": round(demand_val, 2),
                "Carga Total del Recurso": round(float(loads[r_idx]), 2),
                "Capacidad Máx. Recurso": round(float(max_caps[r_idx]), 2),
                "Estado Recurso": "⚠️ Sobrecargado" if is_overloaded else "✅ Dentro de Capacidad",
            }
        )

    # 2. Gráfica de barras: Carga por Recurso vs Capacidad Máxima
    res_labels = [f"Recurso {r + 1}" for r in range(problem.n_resources)]
    bar_colors = [
        "#EF4444" if excess[r] > 1e-9 else "#10B981" for r in range(problem.n_resources)
    ]

    fig_loads = go.Figure()
    fig_loads.add_trace(
        go.Bar(
            x=res_labels,
            y=loads,
            name="Carga Asignada",
            marker_color=bar_colors,
            text=[f"{ld:.1f} / {cp:.1f}" for ld, cp in zip(loads, max_caps)],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Carga asignada: %{y:.2f}<extra></extra>",
        )
    )
    fig_loads.add_trace(
        go.Scatter(
            x=res_labels,
            y=max_caps,
            mode="lines+markers",
            name="Capacidad Máxima",
            line=dict(color="#1E293B", width=2.5, dash="dash"),
            marker=dict(size=8, symbol="line-ew-open"),
            hovertemplate="<b>%{x}</b><br>Capacidad Máxima: %{y:.2f}<extra></extra>",
        )
    )
    fig_loads.update_layout(
        title=f"Carga Acumulada por Recurso vs. Capacidad Máxima {title_suffix}",
        xaxis_title="Recurso",
        yaxis_title="Unidades de Carga / Tareas",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=45, r=30, t=65, b=45),
        height=390,
    )

    # 3. Mapa de calor de la matriz de costos T x R con marcadores en la asignación elegida
    task_labels = [f"Tarea {t + 1}" for t in range(problem.n_tasks)]
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(
        go.Heatmap(
            z=problem.cost_matrix,
            x=res_labels,
            y=task_labels,
            colorscale="Blues",
            colorbar=dict(title="Costo"),
            hovertemplate="%{y} → %{x}<br>Costo: %{z:.1f}<extra></extra>",
        )
    )

    chosen_x = [f"Recurso {int(r) + 1}" for r in assignment]
    chosen_costs = [float(problem.cost_matrix[t, int(assignment[t])]) for t in range(problem.n_tasks)]
    fig_heatmap.add_trace(
        go.Scatter(
            x=chosen_x,
            y=task_labels,
            mode="markers+text",
            name="Asignación Seleccionada",
            text=[f"{c:.0f}" for c in chosen_costs],
            textfont=dict(color="white", size=11),
            marker=dict(
                size=22,
                color="#F59E0B",
                symbol="square",
                line=dict(width=2, color="#1E293B"),
            ),
            hovertemplate="<b>Asignado:</b> %{y} → %{x}<br>Costo: %{text}<extra></extra>",
        )
    )
    fig_heatmap.update_layout(
        title=f"Matriz de Costos ({problem.n_tasks}x{problem.n_resources}) y Asignación Activa {title_suffix}",
        xaxis_title="Recurso",
        yaxis=dict(title="Tarea", autorange="reversed"),
        template="plotly_white",
        margin=dict(l=50, r=30, t=65, b=45),
        height=420,
    )

    return table_rows, fig_loads, fig_heatmap


# ============================================================================
# 4. VISUALIZACIÓN PARA PROBLEMA REAL (RUTEO ÚLTIMA MILLA EN GUADALAJARA)
# ============================================================================

def create_delivery_routing_figures(
    problem: DeliveryRoutingProblem,
    solution: np.ndarray,
    title_suffix: str = "",
) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Devuelve tres figuras interactivas para el problema de Ruteo de Última Milla en Guadalajara:
    1. Mapa Geográfico Interactivo (OpenStreetMap Scattermapbox) con la ruta sobre Guadalajara.
    2. Grafo 2D Cartesiano (Longitud vs. Latitud) con orden de visita numerado y distancias por tramo.
    3. Gráfica de evolución del peso acumulado de paquetes (kg) vs. capacidad máxima del vehículo.
    """
    details = problem.solution_details(solution)
    stops = details["ordered_stops"]
    leg_dists = details["leg_distances_km"]

    lats = [s["lat"] for s in stops]
    lons = [s["lon"] for s in stops]
    names = [s["name"] for s in stops]
    zones = [s["zone"] for s in stops]
    weights = [s["weight_kg"] for s in stops]
    cum_weights = [s["cumulative_weight_kg"] for s in stops]
    orders = [s["stop_order"] for s in stops]

    # -------------------------------------------------------------------------
    # Figura 1: Mapa Geográfico Interactivo sobre Guadalajara (OpenStreetMap)
    # Compatible tanto con Plotly >= 6/7 (Scattermap / layout.map) como Plotly 5 (Scattermapbox)
    # -------------------------------------------------------------------------
    fig_map = go.Figure()
    has_scattermap = hasattr(go, "Scattermap")
    ScatterGeoCls = getattr(go, "Scattermap") if has_scattermap else getattr(go, "Scattermapbox")
    map_layout_key = "map" if has_scattermap else "mapbox"

    # Línea de la ruta completa
    fig_map.add_trace(
        ScatterGeoCls(
            lat=lats,
            lon=lons,
            mode="lines",
            name="Ruta de Entrega",
            line=dict(width=3.5, color="#2563EB"),
            hoverinfo="skip",
        )
    )

    # Puntos de clientes (paradas 1..N)
    client_stops = stops[1:-1]
    fig_map.add_trace(
        ScatterGeoCls(
            lat=[s["lat"] for s in client_stops],
            lon=[s["lon"] for s in client_stops],
            mode="markers+text",
            name="Clientes Visitados",
            text=[f"#{s['stop_order']}" for s in client_stops],
            textposition="top right",
            marker=dict(size=12, color="#EF4444"),
            customdata=np.column_stack(
                [
                    [s["stop_order"] for s in client_stops],
                    [s["name"] for s in client_stops],
                    [s["zone"] for s in client_stops],
                    [s["weight_kg"] for s in client_stops],
                    [s["cumulative_weight_kg"] for s in client_stops],
                ]
            ),
            hovertemplate=(
                "<b>Parada #%{customdata[0]}: %{customdata[1]}</b><br>"
                "Zona: %{customdata[2]}<br>"
                "Peso paquete: %{customdata[3]} kg<br>"
                "Peso acumulado: %{customdata[4]} kg<extra></extra>"
            ),
        )
    )

    # Punto del Almacén Central (CEDIS)
    depot_stop = stops[0]
    fig_map.add_trace(
        ScatterGeoCls(
            lat=[depot_stop["lat"]],
            lon=[depot_stop["lon"]],
            mode="markers+text",
            name="Almacén Central (CEDIS)",
            text=["CEDIS (Inicio/Fin)"],
            textposition="bottom right",
            marker=dict(size=16, color="#10B981"),
            hovertemplate=(
                f"<b>{depot_stop['name']}</b><br>"
                f"Zona: {depot_stop['zone']}<br>"
                f"Lat: {depot_stop['lat']:.4f}, Lon: {depot_stop['lon']:.4f}<extra></extra>"
            ),
        )
    )

    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    layout_kwargs: Dict[str, Any] = {
        "title": (
            f"Mapa de Ruta en Guadalajara {title_suffix} — "
            f"Distancia: {details['base_distance_km']:.2f} km | "
            f"Penalización: {details['penalty']:.2f}"
        ),
        map_layout_key: dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11.0,
        ),
        "legend": dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        "margin": dict(l=15, r=15, t=60, b=15),
        "height": 460,
    }
    fig_map.update_layout(**layout_kwargs)

    # -------------------------------------------------------------------------
    # Figura 2: Grafo 2D de Coordenadas (Longitud vs Latitud) con tramos y flechas
    # -------------------------------------------------------------------------
    fig_graph = go.Figure()
    fig_graph.add_trace(
        go.Scatter(
            x=lons,
            y=lats,
            mode="lines+markers+text",
            name="Secuencia de Visita",
            text=[
                "CEDIS" if s["stop_order"] in (0, len(stops) - 1) else f"#{s['stop_order']}"
                for s in stops[:-1]
            ]
            + [""],
            textposition="top center",
            line=dict(color="#2563EB", width=2.8),
            marker=dict(
                size=[15 if i in (0, len(stops) - 1) else 10 for i in range(len(stops))],
                color=["#10B981" if i in (0, len(stops) - 1) else "#EF4444" for i in range(len(stops))],
                line=dict(width=1.5, color="white"),
            ),
            customdata=np.column_stack([orders, names, zones, weights, cum_weights]),
            hovertemplate=(
                "<b>Parada #%{customdata[0]}: %{customdata[1]}</b><br>"
                "Zona: %{customdata[2]}<br>"
                "Paquete: %{customdata[3]} kg | Acumulado: %{customdata[4]} kg<br>"
                "Lon: %{x:.4f}, Lat: %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig_graph.update_layout(
        title=f"Grafo 2D de Secuencia de Entrega (Coordenadas GPS Guadalajara) {title_suffix}",
        xaxis_title="Longitud (Oeste)",
        yaxis_title="Latitud (Norte)",
        template="plotly_white",
        margin=dict(l=50, r=30, t=65, b=45),
        height=420,
    )

    # -------------------------------------------------------------------------
    # Figura 3: Peso Acumulado por Parada vs. Capacidad del Vehículo
    # -------------------------------------------------------------------------
    stop_labels = [
        "0: CEDIS"
        if s["stop_order"] == 0
        else (
            f"{s['stop_order']}: Retorno"
            if s["stop_order"] == len(stops) - 1
            else f"#{s['stop_order']} (ID {s['id']})"
        )
        for s in stops
    ]
    cap = float(problem.vehicle_capacity)
    bar_colors = ["#EF4444" if cw > cap else "#3B82F6" for cw in cum_weights]

    fig_weight = go.Figure()
    fig_weight.add_trace(
        go.Bar(
            x=stop_labels,
            y=cum_weights,
            name="Peso Acumulado (kg)",
            marker_color=bar_colors,
            customdata=np.column_stack([names, weights, [0.0] + leg_dists]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Peso paquete parada: %{customdata[1]} kg<br>"
                "Peso acumulado: %{y:.1f} kg<br>"
                "Distancia tramo previo: %{customdata[2]:.2f} km<extra></extra>"
            ),
        )
    )
    fig_weight.add_hline(
        y=cap,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Capacidad Máxima Vehículo ({cap:.1f} kg)",
    )
    fig_weight.update_layout(
        title=f"Evolución de Carga Acumulada vs. Capacidad del Vehículo {title_suffix}",
        xaxis_title="Orden de Parada en la Ruta",
        yaxis_title="Peso Acumulado (kg)",
        template="plotly_white",
        margin=dict(l=50, r=30, t=65, b=45),
        height=370,
    )

    return fig_map, fig_graph, fig_weight


# ============================================================================
# UTILIDADES DE EXPORTACIÓN A JSON Y CSV
# ============================================================================

def _json_default(obj: Any) -> Any:
    """Serializador auxiliar para tipos de NumPy en exportación JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def export_results_to_json(
    result: Dict[str, Any],
    problem_config: Dict[str, Any],
    algorithm_params: Dict[str, Any],
) -> str:
    """Genera una cadena JSON formateada con la configuración, resultados e historial completo."""
    payload = {
        "studio": "Metaheuristic Studio",
        "problem_configuration": problem_config,
        "algorithm_configuration": algorithm_params,
        "summary": {
            "algorithm_id": result.get("algorithm_id"),
            "algorithm_name": result.get("algorithm_name"),
            "problem_id": result.get("problem_id"),
            "sense": result.get("sense"),
            "best_fitness": result.get("best_fitness"),
            "best_solution": result.get("best_solution"),
            "stop_reason": result.get("stop_reason"),
            "total_iterations": result.get("total_iterations"),
            "n_evaluations": result.get("n_evaluations"),
            "execution_time_sec": result.get("execution_time_sec"),
        },
        "solution_details": result.get("solution_details", {}),
        "milestones": extract_four_milestones(result.get("history", [])),
        "history": result.get("history", []),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)


def export_history_to_csv(result: Dict[str, Any]) -> str:
    """Genera un archivo CSV en memoria con todas las iteraciones del historial."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "iteration",
            "best_fitness",
            "mean_fitness",
            "best_solution",
            "algorithm_id",
            "problem_id",
            "sense",
        ]
    )

    algo_id = result.get("algorithm_id", "")
    prob_id = result.get("problem_id", "")
    sense = result.get("sense", "")

    for row in result.get("history", []):
        sol_str = json.dumps(row.get("best_solution", []), default=_json_default)
        writer.writerow(
            [
                row.get("iteration"),
                f"{float(row.get('best_fitness', 0.0)):.8f}",
                f"{float(row.get('mean_fitness', 0.0)):.8f}",
                sol_str,
                algo_id,
                prob_id,
                sense,
            ]
        )
    return output.getvalue()

