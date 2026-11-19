"""
Problema Real: Optimización de Rutas de Entrega de Última Milla en Guadalajara, Jalisco.

Representación:
    Permutación sin repetición x de longitud N (con valores en {0, 1, ..., N - 1})
    que define el orden de visita a los N clientes partiendo desde el almacén central
    (CEDIS El Álamo) y regresando al mismo al finalizar el recorrido.

Función Objetivo (Minimización natural):
    f(x) = DistanciaTotal_Haversine(Depot -> x[0] -> ... -> x[N-1] -> Depot)
           + PenalizaciónCapacidad(exceso_kg, distancia_en_sobrecarga)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.base import BaseProblem, OptimizationSense, RepresentationType

DEFAULT_GDL_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rutas_gdl.json"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula la distancia ortodrómica (Haversine) en kilómetros entre dos coordenadas GPS.
    """
    earth_radius_km = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2)
    return float(2.0 * earth_radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


class DeliveryRoutingProblem(BaseProblem):
    """
    Optimización de rutas de entrega de última milla en Guadalajara, Jalisco.
    Carga coordenadas reales de `data/rutas_gdl.json` (1 almacén central + hasta 12 clientes)
    y calcula la distancia total del recorrido más una penalización si el peso acumulado
    de los paquetes excede la capacidad máxima del vehículo.
    """

    problem_id: str = "delivery_routing"
    name: str = "Rutas de Entrega Última Milla - Guadalajara (Permutación)"
    description: str = (
        "Optimización de secuencia de visita a N clientes en la Zona Metropolitana de Guadalajara "
        "partiendo de un CEDIS central, minimizando kilómetros recorridos y penalizando excesos "
        "sobre la capacidad de carga del vehículo."
    )
    representation: RepresentationType = RepresentationType.PERMUTATION
    default_sense: OptimizationSense = OptimizationSense.MINIMIZE

    def __init__(
        self,
        n_clients: Optional[int] = None,
        vehicle_capacity: Optional[float] = None,
        penalty_per_kg: float = 15.0,
        overload_distance_factor: float = 0.5,
        data_path: Optional[Union[str, Path]] = None,
        custom_weights: Optional[Sequence[float]] = None,
        sense: Union[str, OptimizationSense, None] = None,
    ) -> None:
        self.data_path: Path = Path(data_path) if data_path is not None else DEFAULT_GDL_DATA_PATH
        raw_data = self._load_json_data(self.data_path)

        self.metadata: Dict[str, Any] = raw_data.get("metadata", {})
        self.depot: Dict[str, Any] = raw_data["depot"]
        all_clients: List[Dict[str, Any]] = raw_data["clients"]

        max_available = len(all_clients)
        selected_n = max_available if n_clients is None else int(n_clients)
        if selected_n < 2 or selected_n > max_available:
            raise ValueError(
                f"El número de clientes N debe estar entre 2 y {max_available} "
                f"(recibido: {selected_n})."
            )

        super().__init__(dimension=selected_n, sense=sense)
        self.n_clients: int = selected_n
        self.clients: List[Dict[str, Any]] = [dict(c) for c in all_clients[:selected_n]]

        if custom_weights is not None:
            weights_arr = np.asarray(custom_weights, dtype=float).ravel()
            if weights_arr.size != self.n_clients:
                raise ValueError(
                    f"custom_weights debe tener longitud N={self.n_clients}, "
                    f"recibido {weights_arr.size}."
                )
            for idx, w in enumerate(weights_arr):
                self.clients[idx]["weight_kg"] = float(w)

        self.client_weights: np.ndarray = np.array(
            [float(c["weight_kg"]) for c in self.clients], dtype=float
        )
        self.total_package_weight: float = float(np.sum(self.client_weights))

        default_cap = float(self.metadata.get("default_vehicle_capacity_kg", 180.0))
        self.vehicle_capacity: float = (
            float(vehicle_capacity) if vehicle_capacity is not None else default_cap
        )
        self.penalty_per_kg: float = float(penalty_per_kg)
        self.overload_distance_factor: float = float(overload_distance_factor)

        # Precalcular matriz de distancias Haversine (N+1) x (N+1)
        # Índice 0 = Almacén Central (depot), Índices 1..N = Clientes 0..N-1
        self.distance_matrix: np.ndarray = self._build_distance_matrix()

    @staticmethod
    def _load_json_data(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de coordenadas de Guadalajara en: {path}"
            )
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_distance_matrix(self) -> np.ndarray:
        """Construye la matriz simétrica de distancias en km entre CEDIS (0) y los N clientes (1..N)."""
        nodes = [self.depot] + self.clients
        total_nodes = len(nodes)
        mat = np.zeros((total_nodes, total_nodes), dtype=float)
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                dist = haversine_km(
                    float(nodes[i]["lat"]),
                    float(nodes[i]["lon"]),
                    float(nodes[j]["lat"]),
                    float(nodes[j]["lon"]),
                )
                mat[i, j] = dist
                mat[j, i] = dist
        return mat

    def compute_route_metrics(self, permutation: np.ndarray) -> Dict[str, Any]:
        """
        Calcula la distancia total del recorrido y la penalización por capacidad.

        El vehículo sale del CEDIS (nodo 0), visita a los clientes según la permutación
        `p[0], p[1], ..., p[N-1]` (nodos `p[i] + 1` en la matriz de distancias) y regresa al CEDIS.
        A medida que avanza la ruta se acumula el peso de los paquetes recolectados/entregados;
        si el peso total supera `vehicle_capacity`, se aplica:
          1. Penalización fija por exceso de peso: `penalty_per_kg * exceso_total_kg`.
          2. Penalización operativa por kilómetros recorridos en estado de sobrecarga
             (desde el cliente donde la carga acumulada rebasa `vehicle_capacity` hasta el retorno al CEDIS),
             lo que incentiva postergar cargas pesadas o minimizar los tramos en sobrecarga.
        """
        perm = self.repair(permutation)
        # Mapear índices de cliente 0..N-1 a índices de matriz 1..N
        node_seq = [0] + [int(idx) + 1 for idx in perm] + [0]

        leg_distances: List[float] = []
        cumulative_weights: List[float] = []
        overloaded_distance_km: float = 0.0
        running_weight: float = 0.0

        for step in range(len(node_seq) - 1):
            u = node_seq[step]
            v = node_seq[step + 1]
            d_uv = float(self.distance_matrix[u, v])
            leg_distances.append(d_uv)

            if step < self.n_clients:
                client_idx = int(perm[step])
                running_weight += float(self.client_weights[client_idx])
                cumulative_weights.append(running_weight)

            # Si el vehículo ya superó su capacidad máxima en este tramo, acumular km en sobrecarga
            if running_weight > self.vehicle_capacity:
                overloaded_distance_km += d_uv

        base_distance_km = float(np.sum(leg_distances))
        excess_kg = max(0.0, running_weight - self.vehicle_capacity)

        if excess_kg > 0.0:
            static_penalty = self.penalty_per_kg * excess_kg
            dynamic_overload_penalty = (
                self.overload_distance_factor
                * overloaded_distance_km
                * (excess_kg / max(1.0, self.vehicle_capacity))
            )
            penalty = float(static_penalty + dynamic_overload_penalty)
        else:
            penalty = 0.0

        return {
            "permutation": perm,
            "node_sequence": node_seq,
            "leg_distances_km": leg_distances,
            "cumulative_weights_kg": cumulative_weights,
            "base_distance_km": base_distance_km,
            "total_weight_kg": running_weight,
            "vehicle_capacity_kg": self.vehicle_capacity,
            "excess_weight_kg": excess_kg,
            "overloaded_distance_km": overloaded_distance_km,
            "penalty": penalty,
            "total_fitness": base_distance_km + penalty,
            "is_feasible": excess_kg <= 1e-9,
        }

    def evaluate_raw(self, x: np.ndarray) -> float:
        """Evalúa una permutación de N clientes devolviendo distancia (km) + penalización."""
        metrics = self.compute_route_metrics(x)
        return float(metrics["total_fitness"])

    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Genera una permutación aleatoria válida de {0, 1, ..., N - 1}."""
        generator = rng if rng is not None else np.random.default_rng()
        return generator.permutation(self.n_clients).astype(int)

    def repair(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Garantiza que `x` sea una permutación válida de {0, 1, ..., N - 1}.
        - Si `x` ya es una permutación válida, la devuelve como `int`.
        - Si contiene valores reales o duplicados, conserva la primera aparición de cada
          entero válido en [0, N-1] y rellena los faltantes según el orden relativo (random keys).
        """
        arr = np.asarray(x).ravel()
        if arr.size != self.n_clients:
            raise ValueError(
                f"La ruta debe contener exactamente N={self.n_clients} clientes, "
                f"pero se recibió longitud {arr.size}."
            )

        # Caso rápido: ya es permutación exacta
        int_arr = np.rint(arr).astype(int)
        if np.array_equal(np.sort(int_arr), np.arange(self.n_clients)):
            return int_arr

        # Reparación determinista conservando posiciones válidas no repetidas
        seen = set()
        repaired = np.full(self.n_clients, -1, dtype=int)
        empty_positions: List[int] = []

        for pos, val in enumerate(int_arr):
            if 0 <= val < self.n_clients and val not in seen:
                repaired[pos] = val
                seen.add(val)
            else:
                empty_positions.append(pos)

        missing = [c for c in range(self.n_clients) if c not in seen]
        if empty_positions:
            # Ordenar los faltantes según los valores originales en esas posiciones
            sub_vals = arr[empty_positions]
            order = np.argsort(sub_vals, kind="mergesort")
            for rank_idx, pos_idx in enumerate(order):
                repaired[empty_positions[pos_idx]] = missing[rank_idx]

        return repaired

    def get_ordered_stops(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """
        Devuelve la lista ordenada de paradas (incluyendo salida y regreso al CEDIS)
        con coordenadas GPS y pesos para graficar en mapa (Plotly Mapbox / Scattermapbox).
        """
        perm = self.repair(x)
        stops: List[Dict[str, Any]] = [
            {
                "stop_order": 0,
                "node_type": "depot",
                "id": self.depot["id"],
                "name": self.depot["name"],
                "zone": self.depot.get("zone", "CEDIS"),
                "lat": float(self.depot["lat"]),
                "lon": float(self.depot["lon"]),
                "weight_kg": 0.0,
                "cumulative_weight_kg": 0.0,
            }
        ]
        cum_weight = 0.0
        for order_idx, client_idx in enumerate(perm, start=1):
            client = self.clients[int(client_idx)]
            w = float(client["weight_kg"])
            cum_weight += w
            stops.append(
                {
                    "stop_order": order_idx,
                    "node_type": "client",
                    "client_index": int(client_idx),
                    "id": client["id"],
                    "name": client["name"],
                    "zone": client.get("zone", ""),
                    "lat": float(client["lat"]),
                    "lon": float(client["lon"]),
                    "weight_kg": w,
                    "cumulative_weight_kg": cum_weight,
                }
            )
        stops.append(
            {
                "stop_order": self.n_clients + 1,
                "node_type": "depot_return",
                "id": self.depot["id"],
                "name": f"{self.depot['name']} (Retorno)",
                "zone": self.depot.get("zone", "CEDIS"),
                "lat": float(self.depot["lat"]),
                "lon": float(self.depot["lon"]),
                "weight_kg": 0.0,
                "cumulative_weight_kg": cum_weight,
            }
        )
        return stops

    def solution_details(self, x: np.ndarray) -> Dict[str, Any]:
        """Devuelve todos los detalles geográficos, de kilometraje y de capacidad para la UI."""
        metrics = self.compute_route_metrics(x)
        stops = self.get_ordered_stops(metrics["permutation"])
        return {
            "total_fitness": float(metrics["total_fitness"]),
            "base_cost": float(metrics["base_distance_km"]),
            "base_distance_km": float(metrics["base_distance_km"]),
            "penalty": float(metrics["penalty"]),
            "is_feasible": bool(metrics["is_feasible"]),
            "sense": self.sense.value,
            "n_clients": self.n_clients,
            "permutation": metrics["permutation"].tolist(),
            "leg_distances_km": metrics["leg_distances_km"],
            "cumulative_weights_kg": metrics["cumulative_weights_kg"],
            "total_weight_kg": float(metrics["total_weight_kg"]),
            "vehicle_capacity_kg": float(metrics["vehicle_capacity_kg"]),
            "excess_weight_kg": float(metrics["excess_weight_kg"]),
            "overloaded_distance_km": float(metrics["overloaded_distance_km"]),
            "ordered_stops": stops,
        }

    def get_bounds_info(self) -> Dict[str, Any]:
        info = super().get_bounds_info()
        info.update(
            {
                "n_clients": self.n_clients,
                "city": self.metadata.get("city", "Guadalajara, Jalisco"),
                "depot_name": self.depot["name"],
                "total_package_weight_kg": self.total_package_weight,
                "vehicle_capacity_kg": self.vehicle_capacity,
                "penalty_per_kg": self.penalty_per_kg,
            }
        )
        return info

