# 📊 ANÁLISIS COMPLETO DEL TSP CON ALGORITMO GENÉTICO

## 🎯 Descripción General

Este proyecto implementa un sistema completo de análisis para el Problema del Vendedor Viajero (TSP) utilizando Algoritmos Genéticos, con funcionalidades avanzadas de visualización, comparación de algoritmos y análisis estadístico.

## 🚀 Nuevas Funcionalidades Implementadas

### 1. 📈 Gráficos Comparativos entre Algoritmos

#### Comparación de Rendimiento:
- **Convergencia**: Curvas de convergencia superpuestas de diferentes algoritmos:
  - Algoritmo Genético (GA)
  - Hill Climbing
  - Simulated Annealing (implementación adaptada)
  - Particle Swarm Optimization (PSO adaptado)

- **Tiempo de Ejecución**: Comparación del tiempo que tarda cada algoritmo
- **Boxplot de Calidad**: Distribución de resultados en múltiples ejecuciones

### 2. 🧬 Análisis del Algoritmo Genético

#### Evolución del Fitness:
- Gráficos de Mejor/Peor/Media Fitness por Generación
- Visualización de la mejora de la población en el tiempo

#### Diversidad Genética:
- Análisis de la diversidad poblacional vs. generación
- Detección de convergencia prematura

#### Análisis de Parámetros:
- **Heatmap**: Tasa de Cruce vs. Mutación
- **Impacto del Tamaño de Población**: Gráfico de línea con fitness final

### 3. 🗺️ Visualización de Soluciones

#### Rutas Óptimas:
- Mapa 2D con ciudades y rutas
- Comparación visual entre algoritmos

#### Animaciones:
- Evolución de la mejor ruta a través de las generaciones
- Distribución del fitness mediante animaciones

### 4. 📊 Análisis de Escalabilidad

#### Curvas de Rendimiento:
- Fitness Final vs. Número de Ciudades
- Tiempo de CPU vs. Número de Ciudades (escala logarítmica)

### 5. 📉 Gráficos Estadísticos

#### Distribuciones:
- Histograma de Fitness Final en múltiples ejecuciones
- Gráfico de Dispersión (Exploración vs. Explotación)

## 📁 Estructura de Archivos Generados

```
Resultados_TSP/
├── Análisis_Completo_[timestamp]/
│   ├── 📊 Gráficos Básicos:
│   │   ├── solucion_tsp.png
│   │   ├── evolucion_algoritmo.png
│   │   └── distribucion_distancias.png
│   │
│   ├── 🔄 Comparaciones:
│   │   ├── algorithm_convergence_comparison.png
│   │   └── algorithm_results_boxplot.png
│   │
│   ├── 🧬 Análisis GA:
│   │   ├── parameter_sensitivity_heatmap.png
│   │   ├── fitness_evolution_analysis.png
│   │   ├── genetic_diversity_evolution.png
│   │   └── exploration_exploitation_analysis.png
│   │
│   ├── 🎬 Animaciones:
│   │   └── fitness_distribution_evolution.gif
│   │
│   ├── 📄 Reportes:
│   │   ├── evolucion_tsp_log.txt
│   │   └── reporte_final_analisis.txt
│   │
│   └── 📊 Análisis Separado (si se ejecuta analysis_tsp.py):
│       ├── convergence_comparison.png
│       ├── boxplot_comparison.png
│       ├── population_size_analysis.png
│       ├── parameters_heatmap.png
│       ├── scalability_analysis.png
│       ├── final_distances_histogram.png
│       ├── route_evolution.gif
│       └── resumen_analisis.txt
```

## 🛠️ Cómo Ejecutar

### Opción 1: Análisis Completo Integrado
```bash
python EJECUCION_TSP_GA.py
```

### Opción 2: Análisis Avanzado Separado
```bash
python analysis_tsp.py
```

### Opción 3: Ejecutar Ambos
```bash
python ejecutar_analisis_completo.py
```

## 📋 Dependencias

Las siguientes librerías han sido añadidas al `requirements.txt`:

```
matplotlib==3.8.0
numpy==1.26.4
seaborn==0.13.2
pandas==2.1.4
scipy==1.11.4
```

### Instalación:
```bash
pip install -r requirements.txt
```

## 🔧 Configuración

### Parámetros del Algoritmo Genético:
- `POPULATION_SIZE`: Tamaño de la población (default: 100)
- `GENERATIONS`: Número de generaciones (default: 2000)
- `MUTATION_PROBABILITY`: Probabilidad de mutación (default: 0.1)
- `ELITISM_COUNT`: Número de individuos élite (default: 20)

### Parámetros de Análisis:
- Número de ejecuciones para comparaciones: configurable en cada función
- Intervalos de muestreo para análisis de diversidad
- Generaciones para captura de histogramas

## 📈 Tipos de Análisis Implementados

### 1. Análisis Básico (Original)
- ✅ Evolución del algoritmo genético
- ✅ Visualización de la mejor ruta
- ✅ Distribución de distancias por generación

### 2. Análisis Comparativo (Nuevo)
- ✅ Comparación con Hill Climbing
- ✅ Comparación con Simulated Annealing
- ✅ Comparación con PSO adaptado
- ✅ Análisis de tiempo de ejecución

### 3. Análisis de Parámetros (Nuevo)
- ✅ Sensibilidad del tamaño de población
- ✅ Heatmap de cruce vs mutación
- ✅ Análisis de convergencia

### 4. Análisis de Diversidad (Nuevo)
- ✅ Evolución de la diversidad genética
- ✅ Detección de convergencia prematura
- ✅ Análisis exploración vs explotación

### 5. Análisis Estadístico (Nuevo)
- ✅ Distribuciones de resultados finales
- ✅ Análisis de robustez
- ✅ Métricas de rendimiento

### 6. Visualizaciones Avanzadas (Nuevo)
- ✅ Animaciones de evolución
- ✅ Heatmaps interactivos
- ✅ Gráficos de dispersión multidimensionales

## 🎯 Casos de Uso

### Para Investigadores:
- Comparar eficacia del GA con otros métodos
- Analizar convergencia y diversidad genética
- Optimizar parámetros del algoritmo

### Para Estudiantes:
- Entender el comportamiento del algoritmo genético
- Visualizar conceptos de exploración vs explotación
- Aprender sobre optimización heurística

### Para Desarrolladores:
- Evaluar rendimiento en diferentes escenarios
- Identificar mejores configuraciones de parámetros
- Validar implementaciones del algoritmo

## 📊 Métricas y Estadísticas

### Métricas de Calidad:
- Distancia final de la mejor solución
- Porcentaje de mejora respecto a la solución inicial
- Desviación estándar entre múltiples ejecuciones

### Métricas de Convergencia:
- Generación en la que se alcanza el mejor resultado
- Tasa de mejora por generación
- Estabilidad de la solución

### Métricas de Diversidad:
- Diversidad genética promedio
- Variabilidad de la población
- Tasa de convergencia prematura

## 🔍 Interpretación de Resultados

### Gráficos de Convergencia:
- **Pendiente pronunciada**: Convergencia rápida
- **Plateau extendido**: Posible convergencia prematura
- **Oscilaciones**: Alta diversidad genética

### Análisis de Diversidad:
- **Diversidad alta inicial**: Buena exploración
- **Caída gradual**: Convergencia normal
- **Caída abrupta**: Convergencia prematura

### Comparación de Algoritmos:
- **GA vs Hill Climbing**: GA debería mostrar mejor exploración
- **GA vs PSO**: Comparación de métodos poblacionales
- **Tiempo vs Calidad**: Trade-off entre eficiencia y precisión

## 🚧 Limitaciones y Consideraciones

### Escalabilidad:
- Los análisis detallados pueden ser computacionalmente intensivos
- Para ciudades > 20, considerar reducir el número de análisis

### Memoria:
- El almacenamiento de poblaciones históricas requiere memoria considerable
- Configurar intervalos de muestreo según recursos disponibles

### Tiempo de Ejecución:
- El análisis completo puede tomar varios minutos
- Posibilidad de ejecutar análisis parciales

## 🔄 Próximas Mejoras

### Funcionalidades Futuras:
- [ ] Análisis de convergencia con criterios de parada adaptativos
- [ ] Comparación con algoritmos exactos (para instancias pequeñas)
- [ ] Análisis de sensibilidad con diseño de experimentos
- [ ] Optimización automática de parámetros
- [ ] Interfaz gráfica para configuración interactiva

### Optimizaciones:
- [ ] Paralelización de múltiples ejecuciones
- [ ] Cacheo de resultados para análisis incrementales
- [ ] Compresión de datos históricos
- [ ] Exportación a formatos adicionales (CSV, JSON)

## 📞 Soporte

Para reportar problemas o solicitar nuevas funcionalidades:
1. Verificar que todas las dependencias estén instaladas
2. Revisar los logs de ejecución en la carpeta de resultados
3. Comprobar la disponibilidad de memoria y espacio en disco

---

*Desarrollado para el análisis completo del Problema del Vendedor Viajero usando Algoritmos Genéticos*
