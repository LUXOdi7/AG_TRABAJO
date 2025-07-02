# 🧬 ALGORITMO GENÉTICO PARA EL PROBLEMA DEL VENDEDOR VIAJERO (TSP)

## 📖 Índice
1. [Introducción al Algoritmo Genético](#introducción-al-algoritmo-genético)
2. [El Problema del Vendedor Viajero (TSP)](#el-problema-del-vendedor-viajero-tsp)
3. [Representación Genética](#representación-genética)
4. [Componentes del Algoritmo Genético](#componentes-del-algoritmo-genético)
5. [Operadores Genéticos](#operadores-genéticos)
6. [Proceso Evolutivo](#proceso-evolutivo)
7. [Parámetros del Algoritmo](#parámetros-del-algoritmo)
8. [Análisis de Diversidad Genética](#análisis-de-diversidad-genética)
9. [Criterios de Parada](#criterios-de-parada)
10. [Ventajas y Desventajas](#ventajas-y-desventajas)
11. [Implementación Específica](#implementación-específica)

---

## 🎯 Introducción al Algoritmo Genético

### ¿Qué es un Algoritmo Genético?

Un **Algoritmo Genético (AG)** es una técnica de optimización metaheurística inspirada en el proceso de **evolución natural**. Simula la evolución biológica para encontrar soluciones óptimas o near-óptimas a problemas complejos.

### Principios Fundamentales:

🧬 **Selección Natural**: Los individuos más aptos tienen mayor probabilidad de sobrevivir y reproducirse
🔄 **Reproducción**: Combinación de características de los padres para crear descendencia
🎲 **Mutación**: Cambios aleatorios que introducen diversidad genética
⏳ **Evolución**: Mejora gradual de la población a través de generaciones

---

## 🗺️ El Problema del Vendedor Viajero (TSP)

### Definición del Problema:

> **Objetivo**: Encontrar la ruta más corta que visite todas las ciudades exactamente una vez y regrese al punto de partida.

### Características del TSP:
- **NP-Hard**: No existe algoritmo eficiente conocido para encontrar la solución óptima
- **Factorial**: Para n ciudades existen (n-1)!/2 rutas posibles
- **Aplicaciones**: Logística, manufactura, routing, DNA sequencing

### Ejemplo Práctico - Ciudades de Lambayeque:
```
Ciudades: Chiclayo, Lambayeque, Ferreñafe, Monsefú, Túcume, Olmos, Motupe, Chongoyape
Objetivo: Minimizar la distancia total del recorrido
```

---

## 🧬 Representación Genética

### Cromosoma (Individuo):
Un cromosoma representa una **solución candidata** al TSP. En nuestro caso:

```python
# Ejemplo de cromosoma (ruta):
cromosoma = [0, 3, 1, 5, 2, 7, 4, 6]
# Significa: Chiclayo → Monsefú → Lambayeque → Olmos → Ferreñafe → Chongoyape → Túcume → Motupe → Chiclayo
```

### Gen:
Cada **posición** en el cromosoma que representa una ciudad específica.

### Alelo:
El **valor específico** (índice de ciudad) en cada posición del cromosoma.

### Población:
Un **conjunto de cromosomas** que representan diferentes soluciones al problema.

```python
poblacion = [
    [0, 3, 1, 5, 2, 7, 4, 6],  # Individuo 1
    [0, 1, 2, 3, 4, 5, 6, 7],  # Individuo 2
    [0, 7, 6, 5, 4, 3, 2, 1],  # Individuo 3
    # ... más individuos
]
```

---

## ⚙️ Componentes del Algoritmo Genético

### 1. 🎯 Función de Fitness (Evaluación)

**Propósito**: Medir qué tan "buena" es una solución.

```python
def evaluate_fitness(individual):
    """
    Calcula el fitness de un individuo (ruta)
    Fitness = 1 / distancia_total (mayor fitness = mejor solución)
    """
    total_distance = get_total_distance(individual)
    return 1.0 / total_distance if total_distance > 0 else 0
```

**Características**:
- ✅ **Mayor fitness** = **Menor distancia** = **Mejor solución**
- ✅ **Normalización**: Valores entre 0 y 1 para facilitar comparaciones
- ✅ **Robustez**: Maneja casos extremos (distancia = 0)

### 2. 👥 Selección de Padres

**Propósito**: Elegir individuos para reproducirse basándose en su fitness.

#### Método Implementado: **Selección por Torneo**

```python
def select_parents(population, num_parents):
    """
    Selecciona padres mediante torneo binario
    """
    parents = []
    for _ in range(num_parents):
        # Torneo entre 2 individuos aleatorios
        candidate1 = random.choice(population)
        candidate2 = random.choice(population)
        
        # El de mayor fitness gana
        if evaluate_fitness(candidate1) > evaluate_fitness(candidate2):
            parents.append(candidate1)
        else:
            parents.append(candidate2)
    
    return parents
```

**Ventajas del Torneo**:
- ✅ Simple de implementar
- ✅ Mantiene diversidad
- ✅ No requiere ordenamiento completo de la población
- ✅ Presión selectiva controlable

### 3. 🔄 Elitismo

**Propósito**: Preservar las mejores soluciones entre generaciones.

```python
ELITISM_COUNT = 20  # Los 20 mejores individuos pasan automáticamente

# En cada generación:
nueva_poblacion = poblacion_ordenada[:ELITISM_COUNT]  # Élite
# ... completar con descendencia
```

**Beneficios**:
- ✅ **Garantiza que no se pierdan buenas soluciones**
- ✅ **Acelera la convergencia**
- ✅ **Proporciona estabilidad al algoritmo**

---

## 🧬 Operadores Genéticos

### 1. 🤝 Crossover (Cruce/Reproducción)

**Propósito**: Combinar características de dos padres para crear descendencia.

#### Método Implementado: **Order Crossover (OX)**

```python
def crossover(parent1, parent2):
    """
    Order Crossover - preserva orden relativo y evita duplicados
    """
    size = len(parent1)
    
    # Seleccionar segmento aleatorio
    start = random.randint(0, size - 1)
    end = random.randint(start + 1, size)
    
    # Crear hijo con segmento del padre1
    child = [-1] * size
    child[start:end] = parent1[start:end]
    
    # Completar con genes del padre2 en orden
    parent2_filtered = [gene for gene in parent2 if gene not in child]
    
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2_filtered[j]
            j += 1
    
    return child
```

**Ejemplo Visual**:
```
Padre1: [A, B, C, D, E, F, G, H]
Padre2: [C, F, A, D, B, H, E, G]

Segmento seleccionado (posiciones 2-5): [C, D, E]

Hijo:   [F, A, C, D, E, H, B, G]
        └─────────┘
        Segmento del Padre1
```

**Ventajas del Order Crossover**:
- ✅ **Preserva orden relativo** de ciudades
- ✅ **Evita duplicados** automáticamente
- ✅ **Combina características** de ambos padres
- ✅ **Específico para permutaciones** (ideal para TSP)

### 2. 🎲 Mutación

**Propósito**: Introducir diversidad genética y evitar convergencia prematura.

#### Método Implementado: **Swap Mutation**

```python
def mutate(individual):
    """
    Intercambia dos ciudades aleatorias
    """
    mutated = individual[:]  # Copia
    
    if random.random() < MUTATION_PROBABILITY:
        # Seleccionar dos posiciones aleatorias
        i = random.randint(0, len(individual) - 1)
        j = random.randint(0, len(individual) - 1)
        
        # Intercambiar
        mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated
```

**Ejemplo**:
```
Antes:    [A, B, C, D, E, F, G, H]
          └─────────┘   └─────────┘
          Posición 1    Posición 5

Después:  [A, F, C, D, E, B, G, H]
```

**Tipos de Mutación Considerados**:

1. **Swap Mutation** ✅ (Implementada):
   - Intercambia dos ciudades aleatorias
   - Simple y efectiva

2. **Inversion Mutation**:
   - Invierte un segmento de la ruta
   - Preserva adyacencias locales

3. **Insertion Mutation**:
   - Mueve una ciudad a otra posición
   - Cambios más suaves

**Parámetros de Mutación**:
```python
MUTATION_PROBABILITY = 0.1  # 10% de probabilidad
```

---

## 🔄 Proceso Evolutivo

### Algoritmo Principal:

```python
def genetic_algorithm():
    # 1. INICIALIZACIÓN
    poblacion = [create_individual() for _ in range(POPULATION_SIZE)]
    
    for generacion in range(GENERATIONS):
        # 2. EVALUACIÓN
        poblacion.sort(key=evaluate_fitness, reverse=True)
        
        # 3. SELECCIÓN Y ELITISMO
        nueva_poblacion = poblacion[:ELITISM_COUNT]
        
        # 4. REPRODUCCIÓN Y MUTACIÓN
        while len(nueva_poblacion) < POPULATION_SIZE:
            padres = select_parents(poblacion[:POPULATION_SIZE//2], 2)
            hijo = crossover(padres[0], padres[1])
            hijo = mutate(hijo)
            nueva_poblacion.append(hijo)
        
        # 5. REEMPLAZO
        poblacion = nueva_poblacion
        
        # 6. ANÁLISIS Y REGISTRO
        analizar_generacion(poblacion, generacion)
    
    return poblacion[0]  # Mejor solución
```

### Ciclo de Vida de una Generación:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EVALUACIÓN    │───▶│    SELECCIÓN    │───▶│   REPRODUCCIÓN  │
│                 │    │                 │    │                 │
│ • Calcular      │    │ • Torneo        │    │ • Crossover     │
│   fitness       │    │ • Elitismo      │    │ • Mutación      │
│ • Ordenar       │    │ • Padres        │    │ • Nueva         │
│   población     │    │   seleccionados │    │   descendencia  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │              ┌─────────────────┐              │
         └──────────────│    REEMPLAZO    │◀─────────────┘
                        │                 │
                        │ • Nueva         │
                        │   generación    │
                        │ • Análisis      │
                        │ • Criterios     │
                        │   de parada     │
                        └─────────────────┘
```

---

## ⚙️ Parámetros del Algoritmo

### Parámetros Principales:

```python
# POBLACIÓN
POPULATION_SIZE = 100        # Tamaño de la población
ELITISM_COUNT = 20          # Número de individuos élite

# EVOLUCIÓN
GENERATIONS = 2000          # Número máximo de generaciones
MUTATION_PROBABILITY = 0.1  # Probabilidad de mutación (10%)

# PROBLEMA
NUM_CITIES = 8             # Número de ciudades
```

### Impacto de los Parámetros:

#### 🏘️ **Tamaño de Población**:
- **Pequeña** (20-50): 
  - ✅ Convergencia rápida
  - ❌ Poca diversidad, puede caer en óptimos locales
- **Media** (100-200): 
  - ✅ Balance entre exploración y explotación
  - ✅ Estabilidad
- **Grande** (500+): 
  - ✅ Alta diversidad
  - ❌ Convergencia lenta, computacionalmente costosa

#### 🧬 **Tasa de Mutación**:
- **Baja** (0.01-0.05):
  - ✅ Convergencia estable
  - ❌ Poca exploración
- **Media** (0.1-0.2):
  - ✅ Balance exploración/explotación
- **Alta** (0.5+):
  - ✅ Alta exploración
  - ❌ Puede impedir convergencia

#### 👑 **Elitismo**:
- **Sin elitismo** (0%):
  - ❌ Puede perder buenas soluciones
- **Elitismo moderado** (10-20%):
  - ✅ Preserva buenas soluciones
  - ✅ Mantiene diversidad
- **Elitismo alto** (50%+):
  - ❌ Reduce diversidad
  - ❌ Convergencia prematura

---

## 📊 Análisis de Diversidad Genética

### ¿Qué es la Diversidad Genética?

La **diversidad genética** mide qué tan diferentes son los individuos en la población.

### Cálculo de Diversidad:

```python
def calculate_genetic_diversity(population):
    """
    Calcula diversidad como promedio de distancias entre individuos
    """
    if len(population) < 2:
        return 0.0
    
    total_distance = 0
    comparisons = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Distancia de Hamming normalizada
            differences = sum(1 for a, b in zip(population[i], population[j]) if a != b)
            distance = differences / len(population[i])
            total_distance += distance
            comparisons += 1
    
    return total_distance / comparisons if comparisons > 0 else 0.0
```

### Evolución de la Diversidad:

```
Diversidad
    ▲
1.0 ├─●─────●────●
    │  \     \    \
0.8 │   ●─────●───●─────●
    │          \   \     \
0.6 │           ●───●─────●────●
    │                \     \    \
0.4 │                 ●─────●────●───●
    │                           \    \
0.2 │                            ●────●───●
    │                                  \   \
0.0 ├─────────────────────────────────────●───●──▶
    0    200   400   600   800  1000  1200  1400  Generación

Fases:
[Alta Exploración][  Transición  ][    Explotación    ]
```

### Interpretación:

- **Alta diversidad inicial**: Buena exploración del espacio de soluciones
- **Decaimiento gradual**: Convergencia natural hacia buenas soluciones
- **Decaimiento abrupto**: ⚠️ Posible convergencia prematura
- **Diversidad constante**: Posible estancamiento

---

## ⏹️ Criterios de Parada

### Criterios Implementados:

#### 1. **Número Máximo de Generaciones**:
```python
if generacion >= GENERATIONS:
    break
```

#### 2. **Convergencia de Fitness** (Opcional):
```python
def check_convergence(fitness_history, tolerance=1e-6, generations=100):
    if len(fitness_history) < generations:
        return False
    
    recent_fitness = fitness_history[-generations:]
    return max(recent_fitness) - min(recent_fitness) < tolerance
```

#### 3. **Diversidad Mínima** (Opcional):
```python
def check_diversity_threshold(population, min_diversity=0.01):
    current_diversity = calculate_genetic_diversity(population)
    return current_diversity < min_diversity
```

#### 4. **Tiempo Máximo** (Opcional):
```python
def check_time_limit(start_time, max_time_seconds=300):
    return (time.time() - start_time) > max_time_seconds
```

---

## ⚖️ Ventajas y Desventajas

### ✅ Ventajas del Algoritmo Genético:

1. **Robustez**:
   - No requiere derivadas o información específica del problema
   - Maneja espacios de búsqueda discontinuos

2. **Paralelismo Implícito**:
   - Explora múltiples regiones simultáneamente
   - Menos propenso a óptimos locales

3. **Flexibilidad**:
   - Adaptable a diferentes tipos de problemas
   - Fácil incorporación de restricciones

4. **Escalabilidad**:
   - Funciona bien con problemas de alta dimensionalidad
   - Rendimiento predecible

### ❌ Desventajas:

1. **No Garantiza Óptimo Global**:
   - Heurístico, no algoritmo exacto
   - Soluciones aproximadas

2. **Parámetros Sensibles**:
   - Requiere ajuste fino de parámetros
   - Rendimiento variable según configuración

3. **Computacionalmente Intensivo**:
   - Múltiples evaluaciones de fitness
   - Puede ser lento para problemas grandes

4. **Convergencia Lenta**:
   - Especialmente en fase de explotación
   - Puede requerir muchas generaciones

---

## 🛠️ Implementación Específica

### Estructura del Código:

```python
# CONFIGURACIÓN DEL PROBLEMA
CITIES = {
    'Chiclayo': (6, 5),
    'Lambayeque': (5, 6),
    # ... más ciudades
}

DISTANCES_MATRIX = {
    ('Chiclayo', 'Lambayeque'): 12,
    # ... más distancias
}

# PARÁMETROS DEL ALGORITMO
POPULATION_SIZE = 100
GENERATIONS = 2000
MUTATION_PROBABILITY = 0.1
ELITISM_COUNT = 20

# FUNCIONES PRINCIPALES
def create_individual():          # Crear individuo aleatorio
def evaluate_fitness(individual): # Evaluar fitness
def select_parents(population, num_parents): # Selección
def crossover(parent1, parent2):  # Reproducción
def mutate(individual):          # Mutación
def genetic_algorithm():         # Algoritmo principal
```

### Análisis y Visualización:

```python
# ANÁLISIS DE EVOLUCIÓN
def plot_evolution(generations, avg_fitness, best_fitness)
def plot_genetic_diversity_evolution(diversity_history)
def analyze_fitness_evolution(population_history)

# COMPARACIÓN DE ALGORITMOS
def hill_climbing(start_solution)
def simulated_annealing(start_solution)
def compare_algorithm_performance()

# ANÁLISIS DE PARÁMETROS
def analyze_parameter_sensitivity()
def create_parameter_heatmap()

# ANIMACIONES
def create_route_evolution_animation()
def create_fitness_distribution_animation()
```

### Métricas de Rendimiento:

```python
# MÉTRICAS CALCULADAS
- Mejor distancia por generación
- Distancia promedio por generación
- Diversidad genética por generación
- Tiempo de convergencia
- Número de evaluaciones de fitness
- Porcentaje de mejora vs solución inicial
```

---

## 📈 Flujo Completo del Algoritmo

### Diagrama de Flujo:

```
        ┌─────────────────┐
        │   INICIALIZAR   │
        │   POBLACIÓN     │
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
        │    EVALUAR      │
        │    FITNESS      │
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
        │   ORDENAR POR   │
        │    FITNESS      │
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
   ┌────│  CRITERIO DE    │
   │    │     PARADA?     │
   │    └─────────┬───────┘
   │              │ NO
   │    ┌─────────▼───────┐
   │    │   SELECCIONAR   │
   │    │     ÉLITE       │
   │    └─────────┬───────┘
   │              │
   │    ┌─────────▼───────┐
   │    │   SELECCIONAR   │
   │    │     PADRES      │
   │    └─────────┬───────┘
   │              │
   │    ┌─────────▼───────┐
   │    │   CROSSOVER     │
   │    │   (REPRODUCIR)  │
   │    └─────────┬───────┘
   │              │
   │    ┌─────────▼───────┐
   │    │    MUTAR        │
   │    │   DESCENDENCIA  │
   │    └─────────┬───────┘
   │              │
   │    ┌─────────▼───────┐
   │    │   REEMPLAZAR    │
   │    │   POBLACIÓN     │
   │    └─────────┬───────┘
   │              │
   │    ┌─────────▼───────┐
   │    │  INCREMENTAR    │
   │    │  GENERACIÓN     │
   │    └─────────┬───────┘
   │              │
   └──────────────┘
                  │ SÍ
        ┌─────────▼───────┐
        │   RETORNAR      │
        │ MEJOR SOLUCIÓN  │
        └─────────────────┘
```

---

## 🎯 Conclusión

El **Algoritmo Genético para TSP** implementado combina los principios fundamentales de la evolución natural con técnicas computacionales para resolver eficientemente el Problema del Vendedor Viajero.

### Características Clave:

- ✅ **Representación por permutaciones** específica para TSP
- ✅ **Order Crossover** que preserva rutas válidas
- ✅ **Selección por torneo** balanceada
- ✅ **Elitismo** para preservar buenas soluciones
- ✅ **Análisis completo** de diversidad y convergencia
- ✅ **Visualizaciones avanzadas** para interpretación

### Resultados Esperados:

- **Convergencia**: Típicamente en 500-1000 generaciones
- **Calidad**: Soluciones dentro del 5-15% del óptimo
- **Robustez**: Resultados consistentes entre ejecuciones
- **Escalabilidad**: Maneja eficientemente hasta 20+ ciudades

Este algoritmo proporciona una **base sólida** para entender y aplicar técnicas evolutivas a problemas de optimización combinatorial, con extensiones posibles hacia otros dominios y mejoras algorítmicas.

---

*Documentación del Algoritmo Genético para TSP - Desarrollado para el análisis completo de optimización heurística*
