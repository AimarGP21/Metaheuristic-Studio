# 🧬 Metaheuristic Studio

Aplicación web interactiva construida en **Python + Streamlit + Plotly** para configurar, validar, ejecutar y comparar **5 algoritmos metaheurísticos** sobre **4 problemas de optimización** (continuo, binario, discreto y ruteo real de última milla en Guadalajara, Jalisco).

---

## 📁 Estructura del Proyecto

```text
ProyectoFinal_AM/
├── app.py                      # Interfaz web interactiva principal en Streamlit
├── test_algorithms.py          # Suite de verificación de convergencia, semilla y parada anticipada
├── requirements.txt            # Dependencias del proyecto (streamlit, numpy, pandas, plotly)
├── README.md                   # Documentación e instrucciones de uso
├── core/
│   ├── __init__.py
│   └── base.py                 # Clases base (BaseProblem, BaseMetaheuristic), ObjectiveWrapper y COMPATIBILITY_MATRIX
├── problems/
│   ├── __init__.py
│   ├── ackley.py               # 1. Continuo: Función multimodal de Ackley en dimensión D
│   ├── onemax.py               # 2. Binario: Problema OneMax de longitud N
│   ├── task_assignment.py      # 3. Discreto: Asignación de T tareas a R recursos con capacidad máxima
│   └── delivery_routing.py     # 4. Real: Ruteo de última milla en Guadalajara (Haversine + capacidad vehicular)
├── algorithms/
│   ├── __init__.py
│   ├── ga.py                   # Algoritmo Genético (Continuo, Binario, Discreto y Permutación)
│   ├── pso.py                  # Enjambre de Partículas (Continuo estándar y Binary PSO con sigmoide)
│   ├── de.py                   # Evolución Diferencial (DE/rand/1/bin para problemas continuos)
│   ├── aco.py                  # Colonia de Hormigas (Binario, Asignación Discreta y Ruteo en GDL)
│   └── ais.py                  # Sistema Inmune Artificial (CLONALG con selección clonal e hipermutación)
├── data/
│   └── rutas_gdl.json          # Coordenadas GPS reales de CEDIS + 12 puntos en Guadalajara, Jalisco
└── ui/
    ├── __init__.py
    └── visualizations.py       # Gráficas interactivas Plotly, extracción de los 4 hitos y exportación JSON/CSV
```

---

## 🚀 Instalación y Ejecución

### 1. Requisitos previos
- **Python 3.10+** instalado en el sistema.

### 2. Instalar dependencias
Desde la raíz del proyecto (`ProyectoFinal_AM`), ejecuta:

```powershell
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación web en Streamlit
Para levantar **Metaheuristic Studio** en el navegador:

```powershell
python -m streamlit run app.py
```
*(O bien `streamlit run app.py` si la carpeta `Scripts` de Python está en el `PATH`).*

### 4. Ejecutar las pruebas de convergencia en consola
Para verificar por terminal que los 5 algoritmos convergen en sus problemas compatibles, respetan la semilla aleatoria (`seed`) y aplican convergencia anticipada:

```powershell
python test_algorithms.py
```

---

## 🧩 Problemas Implementados (`problems/`)

| Clave | Problema | Representación | Descripción y Restricciones |
| :--- | :--- | :---: | :--- |
| `ackley` | **Función de Ackley** | Continuo $\mathbb{R}^D$ | Dimensión $D$ y límites $[\text{min}, \text{max}]$ configurables. Óptimo global en $f(0,\dots,0)=0$. |
| `onemax` | **Problema OneMax** | Binario $\{0,1\}^N$ | Longitud $N$ configurable. Maximiza (o minimiza) la suma de bits encendidos. |
| `task_assignment` | **Asignación de Tareas** | Discreto $\{0,\dots,R-1\}^T$ | Matriz de costos $T \times R$ generable con semilla o editable en vivo + penalización por exceder la capacidad máxima por recurso. |
| `delivery_routing` | **Ruteo Última Milla GDL** | Permutación $S_N$ | 12 ubicaciones reales en Guadalajara (`data/rutas_gdl.json`) desde el CEDIS El Álamo, con distancias Haversine (km) y penalización por sobrepeso vehicular. |

---

## ⚙️ Matriz de Compatibilidad (`COMPATIBILITY_MATRIX`)

| Problema \ Algoritmo | **GA** (Genético) | **PSO** (Partículas) | **DE** (Diferencial) | **ACO** (Hormigas) | **AIS** (CLONALG) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Continuo (`ackley`)** | ✅ Compatible | ✅ Compatible | ✅ Compatible | ❌ No compatible | ✅ Compatible |
| **2. Binario (`onemax`)** | ✅ Compatible | ✅ Compatible (BPSO) | ❌ No compatible | ✅ Compatible | ✅ Compatible |
| **3. Discreto (`task_assignment`)** | ✅ Compatible | ❌ No compatible | ❌ No compatible | ✅ Compatible | ✅ Compatible |
| **4. Real (`delivery_routing`)** | ✅ Compatible | ❌ No compatible | ❌ No compatible | ✅ Compatible | ❌ No compatible |

---

## 📊 Características Principales de la Interfaz (`app.py`)

1. **Flujo Guiado y Validación Estricta**:
   - **Paso 1**: Selección del problema y configuración de sus parámetros (incluyendo edición interactiva de matriz de costos $T \times R$ o pesos de paquetes en Guadalajara).
   - **Paso 2**: Selección de algoritmo y objetivo (**Minimizar / Maximizar**). Si el par no es compatible según `COMPATIBILITY_MATRIX`, muestra una alerta técnica y deshabilita el botón **Ejecutar**.
   - **Paso 3**: Formulario dinámico que muestra únicamente los hiperparámetros del algoritmo seleccionado, con validación estricta que bloquea valores vacíos, negativos o fuera de rango.
2. **Controles de Ejecución**:
   - Botones de **Ejecutar**, **Detener** (interrupción limpia por bandera de estado), **Reiniciar** y **Regresar**.
3. **Visualización Interactiva y 4 Hitos del Historial**:
   - Curva de convergencia interactiva con Plotly (**Mejor Fitness** y **Fitness Promedio** vs. **Iteración**).
   - Gráficas específicas por dominio: Mapa geográfico OpenStreetMap y grafo 2D para Guadalajara, contorno 2D para Ackley, barras de bits para OneMax y carga vs. capacidad para Asignación de Tareas.
   - Pestañas para inspeccionar el estado real guardado en los **4 hitos obligatorios**: **Iteración 0 (Inicial)**, **Iteración 1/3**, **Iteración 2/3** e **Iteración Final**.
   - Descarga directa de resultados e historial completo en **JSON** y **CSV**.

