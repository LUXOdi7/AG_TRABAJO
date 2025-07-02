import random
import math
import matplotlib.pyplot as plt
import os
import datetime # Para generar nombres de archivos unicos con la fecha y hora
import time
import numpy as np
from matplotlib.animation import FuncAnimation
from copy import deepcopy

# --- Configuración del Problema del Vendedor Viajero (TSP) ---

# Ciudades de Lambayeque, Peru
# NOTA: Para TSP con distancias reales, las coordenadas (X,Y) no son directamente usadas para calcular la distancia,
#       sino principalmente para la visualizacion del mapa. Las distancias reales se definen en el diccionario DISTANCES_MATRIX.
#       Puedes ajustar estas coordenadas si quieres una representacion mas fiel en el mapa resultante.
CITIES = {
    'Chiclayo':    (6, 5),
    'Lambayeque':  (5, 6),
    'Ferreñafe':   (7, 7),
    'Monsefu':     (7, 4),
    'Eten':        (8, 3),
    'Reque':       (6.5, 3.5),
    'Olmos':       (1, 9),
    'Motupe':      (3, 8),
    'Pimentel':    (6.5, 5.5),
    'Tuman':       (7.5, 6)
}

NUM_CITIES = len(CITIES)
CITY_NAMES = list(CITIES.keys())

# Matriz de distancias en kilometros entre ciudades (redondeado a 2 decimales).
# ********************************************************************************
DISTANCES_MATRIX = {
    # Chiclayo
    ('Chiclayo', 'Lambayeque'): 4.70,     # 4.7 km :contentReference[oaicite:1]{index=1}
    ('Chiclayo', 'Ferreñafe'): 29.50,     # estimado Google Maps (~30 km)
    ('Chiclayo', 'Monsefu'): 18.50,       # estimado (~18.5 km)
    ('Chiclayo', 'Eten'): 20.00,          # estimado (~20 km)
    ('Chiclayo', 'Reque'): 10.00,         # estimado (~10 km)
    ('Chiclayo', 'Olmos'): 106.00,        # ruta por carretera :contentReference[oaicite:2]{index=2}
    ('Chiclayo', 'Motupe'): 81.00,        # parte de la ruta Chiclayo → Olmos :contentReference[oaicite:3]{index=3}
    ('Chiclayo', 'Pimentel'): 14.00,      # estimado (~14 km costeros)
    ('Chiclayo', 'Tuman'): 32.00,         # estimado (~32 km)

    # Lambayeque
    ('Lambayeque', 'Ferreñafe'): 25.00,   # estimado (~25 km)
    ('Lambayeque', 'Monsefu'): 24.00,     # estimado (~24 km)
    ('Lambayeque', 'Eten'): 26.00,        # estimado (~26 km)
    ('Lambayeque', 'Reque'): 15.00,       # estimado (~15 km)
    ('Lambayeque', 'Olmos'): 101.00,      # Chiclayo a Olmos menos 5 km → ~101 km
    ('Lambayeque', 'Motupe'): 76.00,      # ruta via Motupe (~76 km)
    ('Lambayeque', 'Pimentel'): 18.00,    # estimado (~18 km via Chiclayo)
    ('Lambayeque', 'Tuman'): 28.00,       # estimado (~28 km)

    # Ferreñafe
    ('Ferreñafe', 'Monsefu'): 22.00,      # estimado (~22 km)
    ('Ferreñafe', 'Eten'): 27.00,         # estimado (~27 km)
    ('Ferreñafe', 'Reque'): 28.00,        # estimado (~28 km)
    ('Ferreñafe', 'Olmos'): 120.00,       # estimado carretero (~120 km)
    ('Ferreñafe', 'Motupe'): 95.00,       # estimado (~95 km)
    ('Ferreñafe', 'Pimentel'): 35.00,     # estimado (~35 km)
    ('Ferreñafe', 'Tuman'): 12.00,        # estimado (~12 km)

    # Monsefu
    ('Monsefu', 'Eten'): 8.00,            # estimado (~8 km)
    ('Monsefu', 'Reque'): 18.00,          # estimado (~18 km)
    ('Monsefu', 'Olmos'): 88.00,          # estimado (~88 km)
    ('Monsefu', 'Motupe'): 65.00,         # estimado (~65 km)
    ('Monsefu', 'Pimentel'): 14.00,       # estimado (~14 km)
    ('Monsefu', 'Tuman'): 15.00,          # estimado (~15 km)

    # Eten
    ('Eten', 'Reque'): 5.00,              # estimado (~5 km)
    ('Eten', 'Olmos'): 102.00,            # estimado (~102 km)
    ('Eten', 'Motupe'): 77.00,            # estimado (~77 km)
    ('Eten', 'Pimentel'): 12.00,          # estimado (~12 km)
    ('Eten', 'Tuman'): 25.00,             # estimado (~25 km)

    # Reque
    ('Reque', 'Olmos'): 96.00,            # estimado (~96 km)
    ('Reque', 'Motupe'): 70.00,           # estimado (~70 km)
    ('Reque', 'Pimentel'): 10.00,         # estimado (~10 km)
    ('Reque', 'Tuman'): 20.00,            # estimado (~20 km)

    # Olmos
    ('Olmos', 'Motupe'): 25.00,           # estimado (~25 km)
    ('Olmos', 'Pimentel'): 128.00,        # Chiclayo->Olmos + Chiclayo->Pimentel = 106+14
    ('Olmos', 'Tuman'): 134.00,           # similar agregada

    # Motupe
    ('Motupe', 'Pimentel'): 90.00,        # estimado (~90 km)
    ('Motupe', 'Tuman'): 85.00,           # estimado (~85 km)

    # Pimentel
    ('Pimentel', 'Tuman'): 35.00,         # estimado (~35 km)
}


# Anadir distancias inversas si no estan presentes (asumiendo simetria A-B = B-A)
for (city1, city2), dist in list(DISTANCES_MATRIX.items()):
    if (city2, city1) not in DISTANCES_MATRIX:
        DISTANCES_MATRIX[(city2, city1)] = dist

# --- Algoritmo Genetico: Parametros ---
POPULATION_SIZE = 200     # Tamano de la poblacion en cada generacion
GENERATIONS = 2000        # Numero de generaciones a ejecutar el algoritmo
ELITISM_COUNT = 20        # Numero de los mejores individuos que pasan directamente a la siguiente generacion
MUTATION_PROBABILITY = 0.1 # Probabilidad de que un individuo sufra mutacion

# --- Rutas de Archivos y Carpetas ---
BASE_OUTPUT_FOLDER = "Resultados_TSP" # Nombre de la carpeta base de salida

# --- Funciones Auxiliares ---

def get_distance_from_matrix(city1_name, city2_name):
    """
    Obtiene la distancia entre dos ciudades de la matriz de distancias.
    """
    distance = DISTANCES_MATRIX.get((city1_name, city2_name))
    if distance is None:
        raise ValueError(f"Distancia no definida en DISTANCES_MATRIX entre {city1_name} y {city2_name}.")
    return round(distance, 2) # Redondear a 2 decimales


def get_total_distance(route):
    """
    Calcula la distancia total de una ruta dada usando las distancias de la matriz.
    Una ruta es una lista de nombres de ciudades.
    La ruta comienza y termina en la primera ciudad.
    """
    total_dist = 0
    # Recorrer la ruta, sumando la distancia entre ciudades consecutivas
    for i in range(NUM_CITIES - 1):
        city1_name = route[i]
        city2_name = route[i+1]
        total_dist += get_distance_from_matrix(city1_name, city2_name)
    
    # Sumar la distancia de regreso a la ciudad de inicio (para cerrar el ciclo)
    total_dist += get_distance_from_matrix(route[-1], route[0])
    return total_dist

# --- Operaciones del Algoritmo Genetico ---

def create_individual():
    """
    Crea un individuo (cromosoma) que representa una ruta.
    Un individuo es una permutacion aleatoria de los nombres de las ciudades.
    """
    individual = list(CITY_NAMES) # Inicialmente, en orden alfabetico
    random.shuffle(individual)    # Mezclamos para obtener una permutacion aleatoria
    return individual

def evaluate_fitness(individual):
    """
    Evalua la 'aptitud' de un individuo.
    En el TSP, una menor distancia total significa una mayor aptitud.
    Por convencion, la aptitud se suele definir como 1 / (distancia total).
    Esto es para que el algoritmo busque maximizar la aptitud (mayor valor = mejor).
    """
    distance = get_total_distance(individual)
    # Evitar division por cero si la distancia fuera 0 (no aplicable en TSP real con ciudades distintas)
    return 1 / distance if distance > 0 else float('inf')

def crossover(parent1, parent2):
    """
    Realiza el cruce (crossover) entre dos padres utilizando el Crossover de Orden (OX1).
    Este operador es adecuado para problemas donde el orden de los genes es importante
    (como el TSP, donde la ruta es una permutacion).
    
    Pasos:
    1. Seleccionar un segmento aleatorio del primer padre.
    2. Copiar ese segmento al hijo en la misma posicion.
    3. Rellenar el resto del hijo con los genes del segundo padre,
       manteniendo el orden relativo y sin duplicados.
    """
    size = len(parent1)
    child = [None] * size
    
    # Seleccionar dos puntos de corte aleatorios
    start_index, end_index = sorted(random.sample(range(size), 2))
    
    # Copiar el segmento del primer padre al hijo
    child[start_index:end_index + 1] = parent1[start_index:end_index + 1]
    
    # Rellenar el resto del hijo con los genes del segundo padre
    # Los genes ya copiados del parent1 no deben ser duplicados
    fill_genes = [gene for gene in parent2 if gene not in child]
    
    current_fill_index = 0
    for i in range(size):
        if child[i] is None: # Si la posicion esta vacia
            child[i] = fill_genes[current_fill_index]
            current_fill_index += 1
            
    return child

def mutate(individual, prob=MUTATION_PROBABILITY):
    """
    Realiza una mutacion en el individuo mediante el intercambio de dos genes (Swap Mutation).
    La mutacion introduce variabilidad en la poblacion y ayuda a evitar optimos locales.
    """
    if random.random() < prob: # La mutacion ocurre solo con una cierta probabilidad
        # Seleccionar dos posiciones aleatorias para intercambiar
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def select_parents(population, num_parents_to_select):
    """
    Selecciona padres de la poblacion.
    Aqui se usa un metodo de seleccion por torneo:
    Se eligen aleatoriamente 'k' individuos y se selecciona el mejor de ellos.
    Se repite 'num_parents_to_select' veces.
    """
    selected_parents = []
    tournament_size = 5 # Tamano del torneo (cuantos individuos compiten)
    for _ in range(num_parents_to_select):
        # Seleccionar individuos aleatorios para el torneo
        competitors = random.sample(population, min(tournament_size, len(population)))
        # El ganador es el que tiene la mayor aptitud
        winner = max(competitors, key=evaluate_fitness)
        selected_parents.append(winner)
    return selected_parents

def genetic_algorithm(run_output_folder):
    """
    Implementacion principal del Algoritmo Genetico para el TSP.
    Ademas de encontrar la solucion, registra la evolucion y guarda archivos.
    """
    # Crear la carpeta de salida para esta ejecucion
    if not os.path.exists(run_output_folder):
        os.makedirs(run_output_folder)
        print(f"Carpeta '{run_output_folder}' creada para esta ejecucion.")

    # Generar un nombre de archivo para el log
    log_filename = os.path.join(run_output_folder, f"evolucion_tsp_log.txt")

    # Listas para almacenar datos para los graficos de evolucion
    generations_data = []
    avg_distances = []
    min_distances = []
    
    # Lista para almacenar distancias de poblaciones seleccionadas para histogramas
    distances_for_histograms = []
    generations_for_histograms = []
    
    # Generaciones en las que queremos capturar la distribucion de distancias
    histogram_generations = [0, GENERATIONS // 2, GENERATIONS - 1]
    if GENERATIONS < 3: # Asegurar que al menos la generacion 0 y la final sean capturadas si son pocas
        histogram_generations = sorted(list(set([0, GENERATIONS - 1])))


    # 1. Inicializacion de la poblacion
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_solution_overall = None
    min_distance_overall = float('inf')
    
    print("Iniciando algoritmo genetico para TSP...")

    # Abrir el archivo de log para escribir
    # Usamos encoding='utf-8' para asegurar compatibilidad
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write(f"--- Registro de la Evolucion del Algoritmo Genetico para TSP ---\n")
        log_file.write(f"Parametros: Poblacion={POPULATION_SIZE}, Generaciones={GENERATIONS}, Elitismo={ELITISM_COUNT}, Mutacion={MUTATION_PROBABILITY}\n\n")

        for generation in range(GENERATIONS):
            # 2. Evaluacion de la aptitud de cada individuo
            population.sort(key=evaluate_fitness, reverse=True) # Ordenar por aptitud, de mayor a menor

            # Obtener el mejor individuo de la generacion actual
            current_best_individual = population[0]
            current_min_distance = get_total_distance(current_best_individual)
            
            # Calcular la distancia promedio de la poblacion actual
            all_distances = [get_total_distance(ind) for ind in population]
            current_avg_distance = sum(all_distances) / len(all_distances)

            # Almacenar datos para el grafico
            generations_data.append(generation)
            avg_distances.append(current_avg_distance)
            min_distances.append(current_min_distance)
            
            # Capturar distancias para el histograma en generaciones clave
            if generation in histogram_generations:
                distances_for_histograms.append(all_distances)
                generations_for_histograms.append(generation)


            # Actualizar la mejor solucion global encontrada hasta ahora
            if current_min_distance < min_distance_overall:
                min_distance_overall = current_min_distance
                best_solution_overall = current_best_individual
                log_file.write(f"Generacion {generation}: NUEVA MEJOR! Distancia: {min_distance_overall:.2f} - Ruta: {best_solution_overall}\n")
            else:
                 log_file.write(f"Generacion {generation}: Mejor de la generacion: {current_min_distance:.2f}, Promedio: {current_avg_distance:.2f}\n")

            # Imprimir progreso en consola cada 100 generaciones
            if generation % 100 == 0 or generation == GENERATIONS - 1:
                print(f"Generacion {generation}: Mejor distancia actual: {current_min_distance:.2f}, Promedio de distancia: {current_avg_distance:.2f}")
            
            # Si se encuentra una solucion "optima" (distancia muy baja o que no mejora)
            # Podrias agregar una condicion de parada temprana aqui si lo deseas.

            # 3. Creacion de la nueva poblacion
            new_population = []
            
            # Elitismo: Mantener los mejores individuos de la generacion actual
            new_population.extend(population[:ELITISM_COUNT])
            
            # Mientras la nueva poblacion no alcance el tamano deseado:
            while len(new_population) < POPULATION_SIZE:
                # 4. Seleccion de padres
                # Seleccionamos entre los mejores individuos de la poblacion actual
                parents_for_breeding = select_parents(population[:POPULATION_SIZE // 2], 2)
                parent1, parent2 = parents_for_breeding[0], parents_for_breeding[1]
                
                # 5. Cruce (Crossover)
                child = crossover(parent1, parent2)
                
                # 6. Mutacion
                child = mutate(child)
                
                new_population.append(child)
                
            population = new_population # La nueva poblacion reemplaza a la antigua

    print("\nAlgoritmo genetico finalizado.")
    print(f"Mejor solucion encontrada: {best_solution_overall}")
    print(f"Distancia minima total: {min_distance_overall:.2f}")

    return best_solution_overall, generations_data, avg_distances, min_distances, distances_for_histograms, generations_for_histograms

# --- Visualizacion de la Solucion ---

def plot_solution(solution_route, folder_path):
    """
    Grafica la ruta encontrada por el algoritmo genetico, incluyendo las distancias entre ciudades,
    y la guarda en la carpeta especificada.
    """
    plt.figure(figsize=(12, 10)) # Aumentamos el tamano para mejor visibilidad de las etiquetas
    
    # Extraer coordenadas para graficar
    x_coords = [CITIES[city][0] for city in solution_route]
    y_coords = [CITIES[city][1] for city in solution_route]

    # Conectar las ciudades en el orden de la ruta
    # Se agrega la primera ciudad al final para cerrar el ciclo visualmente
    plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'o-', color='blue', markersize=8, linewidth=2, label='Ruta Optimizada')
    
    # Anotar las ciudades y sus coordenadas (opcional, para referencia)
    for city_name, coords in CITIES.items():
        plt.text(coords[0] + 0.1, coords[1] + 0.1, city_name, fontsize=12, ha='left', va='bottom')
        plt.plot(coords[0], coords[1], 'o', color='red', markersize=10) # Marcar las ciudades
    
    # Marcar la ciudad de inicio/fin
    start_city_coords = CITIES[solution_route[0]]
    plt.plot(start_city_coords[0], start_city_coords[1], 'o', color='green', markersize=12, label='Ciudad de Inicio/Fin')

    # Añadir las distancias entre ciudades sobre las lineas
    for i in range(NUM_CITIES):
        city1_name = solution_route[i]
        # Usamos el modulo para asegurar que la ultima ciudad se conecte con la primera
        city2_name = solution_route[(i + 1) % NUM_CITIES] 
        
        dist = get_distance_from_matrix(city1_name, city2_name)
        
        # Coordenadas de las ciudades
        x1, y1 = CITIES[city1_name]
        x2, y2 = CITIES[city2_name]
        
        # Calcular el punto medio para colocar el texto
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Calcular el angulo de la linea para rotar el texto y que quede alineado
        # math.atan2(dy, dx) devuelve el angulo en radianes
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Ajustar el angulo para que el texto no este "boca abajo"
        if angle > 90 or angle < -90:
            angle += 180
        
        plt.text(mid_x, mid_y, f'{dist:.2f} km', 
                 fontsize=9, color='purple', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                 rotation=angle) # Añadir rotacion y fondo para mejor legibilidad

    plt.title(f'Solucion TSP - Distancia Total: {get_total_distance(solution_route):.2f} km')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True)
    plt.legend()
    
    # Guardar el grafico
    plot_filename = os.path.join(folder_path, f"solucion_tsp.png")
    plt.savefig(plot_filename)
    plt.close() # Cierra la figura para liberar memoria
    print(f"Grafico de la solucion guardado en '{plot_filename}'")

def plot_evolution(generations, avg_distances, min_distances, folder_path):
    """
    Grafica la evolucion de la distancia promedio y minima a lo largo de las generaciones.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(generations, avg_distances, label='Distancia Promedio de la Poblacion', color='blue')
    plt.plot(generations, min_distances, label='Distancia Minima (Mejor Solucion)', color='orange')
    plt.title('Evolucion del Algoritmo Genetico para TSP')
    plt.xlabel('Generacion')
    plt.ylabel('Distancia')
    plt.grid(True)
    plt.legend()
    
    # Guardar el grafico
    evolution_plot_filename = os.path.join(folder_path, f"evolucion_algoritmo.png")
    plt.savefig(evolution_plot_filename)
    plt.close() # Cierra la figura para liberar memoria
    print(f"Grafico de evolucion guardado en '{evolution_plot_filename}'")

def plot_distance_histograms(data_for_hists, generations_for_hists, folder_path):
    """
    Grafica histogramas de la distribucion de distancias en la poblacion en generaciones clave.
    """
    num_plots = len(data_for_hists)
    if num_plots == 0:
        print("No hay datos para generar histogramas de distribucion de distancias.")
        return

    # Ajustar el tamano de la figura dinamicamente
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)
    
    # Si solo hay un subplot, axes no es un array, asi que lo envolvemos
    if num_plots == 1:
        axes = [axes]

    for i, distances in enumerate(data_for_hists):
        ax = axes[i]
        gen = generations_for_hists[i]
        ax.hist(distances, bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'Generacion {gen}')
        ax.set_xlabel('Distancia de Ruta')
        if i == 0: # Solo el primer subplot necesita la etiqueta del eje Y
            ax.set_ylabel('Frecuencia')
        ax.grid(axis='y', alpha=0.75)

    plt.suptitle('Distribucion de Distancias en la Poblacion por Generacion', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout para el suptitle
    
    # Guardar el grafico
    histograms_filename = os.path.join(folder_path, f"distribucion_distancias.png")
    plt.savefig(histograms_filename)
    plt.close()
    print(f"Grafico de distribucion de distancias guardado en '{histograms_filename}'")


# --- Nuevas Funciones de Análisis Avanzado ---

def analyze_fitness_evolution(population_history, folder_path):
    """
    Analizar la evolución del fitness (mejor, promedio, peor) durante las generaciones
    """
    if not population_history:
        return
    
    generations = []
    best_fitness = []
    avg_fitness = []
    worst_fitness = []
    
    for gen, population in enumerate(population_history):
        distances = [get_total_distance(ind) for ind in population]
        
        generations.append(gen)
        best_fitness.append(min(distances))
        avg_fitness.append(np.mean(distances))
        worst_fitness.append(max(distances))
    
    plt.figure(figsize=(12, 8))
    plt.plot(generations, best_fitness, 'g-', label='Mejor Fitness', linewidth=2)
    plt.plot(generations, avg_fitness, 'b-', label='Fitness Promedio', linewidth=2)
    plt.plot(generations, worst_fitness, 'r-', label='Peor Fitness', linewidth=2)
    
    plt.xlabel('Generación')
    plt.ylabel('Distancia (Fitness)')
    plt.title('Evolución del Fitness: Mejor/Promedio/Peor por Generación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fitness_evolution_filename = os.path.join(folder_path, "fitness_evolution_analysis.png")
    plt.savefig(fitness_evolution_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análisis de evolución del fitness guardado en '{fitness_evolution_filename}'")

def calculate_genetic_diversity(population):
    """
    Calcular la diversidad genética de una población
    """
    if len(population) < 2:
        return 0
    
    total_differences = 0
    comparisons = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Contar diferencias en posiciones entre dos individuos
            differences = sum(1 for k in range(len(population[i])) 
                            if population[i][k] != population[j][k])
            total_differences += differences
            comparisons += 1
    
    return total_differences / comparisons if comparisons > 0 else 0

def plot_genetic_diversity_evolution(diversity_history, folder_path):
    """
    Graficar la evolución de la diversidad genética
    """
    if not diversity_history:
        return
    
    plt.figure(figsize=(12, 6))
    generations = list(range(len(diversity_history)))
    
    plt.plot(generations, diversity_history, 'purple', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Generación')
    plt.ylabel('Diversidad Genética')
    plt.title('Evolución de la Diversidad Genética')
    plt.grid(True, alpha=0.3)
    
    # Añadir líneas de referencia
    max_diversity = max(diversity_history) if diversity_history else 0
    min_diversity = min(diversity_history) if diversity_history else 0
    avg_diversity = np.mean(diversity_history) if diversity_history else 0
    
    plt.axhline(y=avg_diversity, color='red', linestyle='--', alpha=0.7, 
                label=f'Diversidad Promedio: {avg_diversity:.2f}')
    plt.legend()
    
    diversity_filename = os.path.join(folder_path, "genetic_diversity_evolution.png")
    plt.savefig(diversity_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de diversidad genética guardado en '{diversity_filename}'")

def create_exploration_exploitation_plot(population_history, folder_path):
    """
    Crear gráfico de exploración vs explotación
    """
    plt.figure(figsize=(14, 10))
    
    generations = []
    fitness_values = []
    colors = []
    
    # Tomar muestras cada cierto número de generaciones para evitar sobrecarga
    sample_interval = max(1, len(population_history) // 20)
    
    for gen_idx in range(0, len(population_history), sample_interval):
        population = population_history[gen_idx]
        
        # Tomar una muestra de la población para reducir puntos
        sample_size = min(50, len(population))
        sampled_population = random.sample(population, sample_size)
        
        for individual in sampled_population:
            generations.append(gen_idx)
            distance = get_total_distance(individual)
            fitness_values.append(distance)
            colors.append(distance)
    
    # Crear gráfico de dispersión
    scatter = plt.scatter(generations, fitness_values, c=colors, cmap='viridis_r', 
                         alpha=0.6, s=30)
    plt.colorbar(scatter, label='Distancia (Fitness)')
    
    plt.xlabel('Generación')
    plt.ylabel('Distancia (Fitness) de Individuos')
    plt.title('Exploración vs Explotación: Distribución de Fitness por Generación')
    plt.grid(True, alpha=0.3)
    
    exploration_filename = os.path.join(folder_path, "exploration_exploitation_analysis.png")
    plt.savefig(exploration_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de exploración vs explotación guardado en '{exploration_filename}'")

def create_fitness_distribution_animation(population_history, folder_path, max_frames=50):
    """
    Crear animación de la distribución del fitness a través de las generaciones
    """
    if len(population_history) < 2:
        return
    
    # Seleccionar frames para la animación
    frame_indices = np.linspace(0, len(population_history) - 1, max_frames, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def animate(frame_idx):
        ax1.clear()
        ax2.clear()
        
        gen_idx = frame_indices[frame_idx]
        population = population_history[gen_idx]
        distances = [get_total_distance(ind) for ind in population]
        
        # Histograma de fitness
        ax1.hist(distances, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Distancia (Fitness)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title(f'Distribución del Fitness - Generación {gen_idx}')
        ax1.grid(True, alpha=0.3)
        
        # Estadísticas acumuladas
        if gen_idx > 0:
            all_gen_data = []
            for i in range(0, gen_idx + 1, max(1, gen_idx // 10)):
                gen_distances = [get_total_distance(ind) for ind in population_history[i]]
                all_gen_data.extend([(i, d) for d in gen_distances])
            
            if all_gen_data:
                gens, fits = zip(*all_gen_data)
                ax2.scatter(gens, fits, alpha=0.3, s=20, c=fits, cmap='viridis_r')
                ax2.set_xlabel('Generación')
                ax2.set_ylabel('Distancia (Fitness)')
                ax2.set_title('Evolución Histórica del Fitness')
                ax2.grid(True, alpha=0.3)
    
    # Crear animación
    anim = FuncAnimation(fig, animate, frames=len(frame_indices), 
                        interval=300, repeat=True)
    
    animation_filename = os.path.join(folder_path, "fitness_distribution_evolution.gif")
    anim.save(animation_filename, writer='pillow', fps=3)
    plt.close()
    print(f"Animación de distribución del fitness guardada en '{animation_filename}'")

def compare_algorithm_performance(folder_path, num_runs=5):
    """
    Comparar el rendimiento del algoritmo genético con otros algoritmos simples
    """
    print("Ejecutando comparación de algoritmos...")
    
    def hill_climbing_simple():
        """Implementación simple de Hill Climbing"""
        current_solution = create_individual()
        current_distance = get_total_distance(current_solution)
        
        distances_history = [current_distance]
        
        for _ in range(500):  # Menos iteraciones para acelerar
            # Generar vecino intercambiando dos ciudades
            neighbor = current_solution.copy()
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
            neighbor_distance = get_total_distance(neighbor)
            
            # Aceptar si es mejor
            if neighbor_distance < current_distance:
                current_solution = neighbor
                current_distance = neighbor_distance
            
            distances_history.append(current_distance)
        
        return current_solution, distances_history
    
    def random_search():
        """Búsqueda aleatoria"""
        best_solution = create_individual()
        best_distance = get_total_distance(best_solution)
        
        distances_history = [best_distance]
        
        for _ in range(500):
            candidate = create_individual()
            candidate_distance = get_total_distance(candidate)
            
            if candidate_distance < best_distance:
                best_solution = candidate
                best_distance = candidate_distance
            
            distances_history.append(best_distance)
        
        return best_solution, distances_history
    
    # Ejecutar comparaciones
    algorithms = {
        'Hill Climbing': hill_climbing_simple,
        'Búsqueda Aleatoria': random_search
    }
    
    results = {}
    
    for alg_name, alg_func in algorithms.items():
        print(f"Ejecutando {alg_name}...")
        alg_distances = []
        alg_histories = []
        
        for run in range(num_runs):
            solution, history = alg_func()
            alg_distances.append(history[-1])
            alg_histories.append(history)
        
        results[alg_name] = {
            'distances': alg_distances,
            'histories': alg_histories
        }
    
    # Agregar resultado del GA (simulado para comparación)
    print("Ejecutando Algoritmo Genético para comparación...")
    ga_distances = []
    ga_histories = []
    
    for run in range(num_runs):
        # Versión simplificada del GA
        population = [create_individual() for _ in range(50)]
        best_distances = []
        
        for generation in range(100):  # Menos generaciones
            population.sort(key=evaluate_fitness, reverse=True)
            current_best = population[0]
            best_distances.append(get_total_distance(current_best))
            
            # Evolución simple
            new_population = population[:5]  # Elitismo
            
            while len(new_population) < 50:
                parents = select_parents(population[:25], 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        ga_distances.append(best_distances[-1])
        ga_histories.append(best_distances)
    
    results['Algoritmo Genético'] = {
        'distances': ga_distances,
        'histories': ga_histories
    }
    
    # Crear gráficos comparativos
    create_algorithm_comparison_plots(results, folder_path)

def create_algorithm_comparison_plots(results, folder_path):
    """
    Crear gráficos de comparación entre algoritmos
    """
    # 1. Gráfico de convergencia
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (alg_name, alg_data) in enumerate(results.items()):
        histories = alg_data['histories']
        
        # Calcular promedio de historias
        max_len = max(len(hist) for hist in histories)
        avg_history = []
        
        for step in range(max_len):
            step_values = []
            for hist in histories:
                if step < len(hist):
                    step_values.append(hist[step])
                else:
                    step_values.append(hist[-1])
            avg_history.append(np.mean(step_values))
        
        plt.plot(range(max_len), avg_history, 
                color=colors[i % len(colors)], label=alg_name, linewidth=2)
    
    plt.xlabel('Iteración/Generación')
    plt.ylabel('Mejor Distancia Encontrada')
    plt.title('Comparación de Convergencia entre Algoritmos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    convergence_filename = os.path.join(folder_path, "algorithm_convergence_comparison.png")
    plt.savefig(convergence_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplot de resultados finales
    plt.figure(figsize=(10, 6))
    
    data = []
    labels = []
    
    for alg_name, alg_data in results.items():
        data.append(alg_data['distances'])
        labels.append(alg_name)
    
    plt.boxplot(data, labels=labels)
    plt.ylabel('Distancia Final')
    plt.title('Distribución de Resultados por Algoritmo')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    boxplot_filename = os.path.join(folder_path, "algorithm_results_boxplot.png")
    plt.savefig(boxplot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficos de comparación guardados en '{convergence_filename}' y '{boxplot_filename}'")

def analyze_parameter_sensitivity(folder_path):
    """
    Análisis de sensibilidad de parámetros del algoritmo genético
    """
    print("Realizando análisis de sensibilidad de parámetros...")
    
    # Parámetros a probar
    population_sizes = [50, 100, 150, 200]
    mutation_rates = [0.05, 0.1, 0.15, 0.2]
    
    results_matrix = np.zeros((len(mutation_rates), len(population_sizes)))
    
    for i, mut_rate in enumerate(mutation_rates):
        for j, pop_size in enumerate(population_sizes):
            print(f"Probando población: {pop_size}, mutación: {mut_rate}")
            
            # Ejecutar GA con estos parámetros (versión corta)
            distances = []
            for run in range(3):  # 3 ejecuciones por combinación
                population = [create_individual() for _ in range(pop_size)]
                
                for generation in range(200):  # Menos generaciones
                    population.sort(key=evaluate_fitness, reverse=True)
                    
                    # Elitismo proporcional
                    elite_size = max(1, pop_size // 10)
                    new_population = population[:elite_size]
                    
                    while len(new_population) < pop_size:
                        parents = select_parents(population[:pop_size//2], 2)
                        child = crossover(parents[0], parents[1])
                        child = mutate(child, mut_rate)
                        new_population.append(child)
                    
                    population = new_population
                
                best_distance = min(get_total_distance(ind) for ind in population)
                distances.append(best_distance)
            
            results_matrix[i, j] = np.mean(distances)
    
    # Crear heatmap
    plt.figure(figsize=(10, 8))
    
    # Usar colores inversos para que menor distancia = mejor (más claro)
    im = plt.imshow(results_matrix, cmap='viridis_r', aspect='auto')
    
    # Configurar etiquetas
    plt.xticks(range(len(population_sizes)), [str(ps) for ps in population_sizes])
    plt.yticks(range(len(mutation_rates)), [f"{mr:.2f}" for mr in mutation_rates])
    plt.xlabel('Tamaño de Población')
    plt.ylabel('Tasa de Mutación')
    plt.title('Sensibilidad de Parámetros: Distancia Final Promedio')
    
    # Añadir colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Distancia Final Promedio')
    
    # Añadir valores en cada celda
    for i in range(len(mutation_rates)):
        for j in range(len(population_sizes)):
            plt.text(j, i, f'{results_matrix[i, j]:.1f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    sensitivity_filename = os.path.join(folder_path, "parameter_sensitivity_heatmap.png")
    plt.savefig(sensitivity_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap de sensibilidad guardado en '{sensitivity_filename}'")

def generate_final_report(folder_path, final_solution, min_distances, avg_distances):
    """
    Generar un reporte final con todas las estadísticas y análisis
    """
    report_filename = os.path.join(folder_path, "reporte_final_analisis.txt")
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("REPORTE FINAL - ANÁLISIS COMPLETO DEL TSP\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. CONFIGURACIÓN DEL PROBLEMA\n")
        f.write("-" * 30 + "\n")
        f.write(f"Número de ciudades: {NUM_CITIES}\n")
        f.write(f"Ciudades: {', '.join(CITY_NAMES)}\n")
        f.write(f"Tamaño de población: {POPULATION_SIZE}\n")
        f.write(f"Número de generaciones: {GENERATIONS}\n")
        f.write(f"Probabilidad de mutación: {MUTATION_PROBABILITY}\n")
        f.write(f"Elitismo: {ELITISM_COUNT} individuos\n\n")
        
        f.write("2. RESULTADOS FINALES\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mejor solución: {final_solution}\n")
        # final_solution ya contiene los nombres de las ciudades directamente
        route_names = final_solution  # No necesitamos hacer conversión
        f.write(f"Ruta: {' -> '.join(route_names)} -> {final_solution[0]}\n")
        f.write(f"Distancia final: {min_distances[-1]:.2f} km\n")
        f.write(f"Mejora total: {min_distances[0] - min_distances[-1]:.2f} km\n")
        f.write(f"Porcentaje de mejora: {((min_distances[0] - min_distances[-1]) / min_distances[0] * 100):.1f}%\n\n")
        
        f.write("3. ESTADÍSTICAS DE EVOLUCIÓN\n")
        f.write("-" * 28 + "\n")
        f.write(f"Distancia inicial (mejor): {min_distances[0]:.2f} km\n")
        f.write(f"Distancia promedio inicial: {avg_distances[0]:.2f} km\n")
        f.write(f"Distancia final (mejor): {min_distances[-1]:.2f} km\n")
        f.write(f"Distancia promedio final: {avg_distances[-1]:.2f} km\n")
        f.write(f"Mejor mejora en una generación: {max(min_distances[i] - min_distances[i+1] for i in range(len(min_distances)-1)):.2f} km\n\n")
        
        f.write("4. ARCHIVOS GENERADOS\n")
        f.write("-" * 18 + "\n")
        f.write("• solucion_tsp.png - Visualización de la mejor ruta\n")
        f.write("• evolucion_algoritmo.png - Evolución del algoritmo\n")
        f.write("• distribucion_distancias.png - Histogramas de distribución\n")
        f.write("• algorithm_convergence_comparison.png - Comparación de convergencia\n")
        f.write("• algorithm_results_boxplot.png - Boxplot de resultados\n")
        f.write("• parameter_sensitivity_heatmap.png - Sensibilidad de parámetros\n")
        f.write("• fitness_evolution_analysis.png - Análisis detallado del fitness\n")
        f.write("• genetic_diversity_evolution.png - Evolución de la diversidad\n")
        f.write("• exploration_exploitation_analysis.png - Análisis de exploración\n")
        f.write("• fitness_distribution_evolution.gif - Animación de evolución\n")
        f.write("• evolucion_tsp_log.txt - Log detallado de la ejecución\n\n")
        
        f.write("5. RECOMENDACIONES\n")
        f.write("-" * 15 + "\n")
        f.write("• Analizar los gráficos de convergencia para identificar si el algoritmo converge prematuramente\n")
        f.write("• Revisar la evolución de la diversidad genética para detectar pérdida de variabilidad\n")
        f.write("• Usar el heatmap de sensibilidad para optimizar parámetros\n")
        f.write("• Comparar resultados con otros algoritmos para validar la eficacia del GA\n")
        f.write("• Considerar ajustar parámetros si la mejora es menor al 10%\n")
    
    print(f"Reporte final generado: {report_filename}")

# --- Ejecucion del Algoritmo ---
if __name__ == "__main__":
    print("=== ANÁLISIS COMPLETO DEL TSP CON ALGORITMO GENÉTICO ===")
    
    # Asegurarse de que la carpeta base exista
    if not os.path.exists(BASE_OUTPUT_FOLDER):
        os.makedirs(BASE_OUTPUT_FOLDER)

    # Generar un nombre de carpeta unica para esta ejecucion (ej. Prueba_YYYYMMDD_HHMMSS)
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_folder = os.path.join(BASE_OUTPUT_FOLDER, f"Análisis_Completo_{current_timestamp}")

    # Ejecutar el algoritmo genetico y obtener los datos de evolucion
    print("\n1. Ejecutando Algoritmo Genético...")
    final_solution, gens, avg_dists, min_dists, distances_for_hists, generations_for_hists = genetic_algorithm(run_output_folder)
    
    # Guardar el grafico de la solucion final
    print("2. Generando gráfico de solución...")
    plot_solution(final_solution, run_output_folder)
    
    # Guardar el grafico de la evolucion del algoritmo
    print("3. Generando gráfico de evolución...")
    plot_evolution(gens, avg_dists, min_dists, run_output_folder)

    # Guardar el grafico de distribucion de distancias
    print("4. Generando histogramas de distribución...")
    plot_distance_histograms(distances_for_hists, generations_for_hists, run_output_folder)

    # --- ANÁLISIS AVANZADO ---
    print("\n=== EJECUTANDO ANÁLISIS AVANZADO ===")
    
    # 5. Comparación de algoritmos
    print("5. Comparando con otros algoritmos...")
    compare_algorithm_performance(run_output_folder, num_runs=5)
    
    # 6. Análisis de sensibilidad de parámetros
    print("6. Analizando sensibilidad de parámetros...")
    analyze_parameter_sensitivity(run_output_folder)
    
    # 7. Ejecutar análisis con datos detallados
    print("7. Ejecutando algoritmo genético con análisis detallado...")
    
    # Ejecutar una versión modificada para capturar datos detallados
    population_history = []
    diversity_history = []
    
    # Re-ejecutar para obtener datos detallados
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    for generation in range(min(500, GENERATIONS)):  # Menos generaciones para acelerar
        population.sort(key=evaluate_fitness, reverse=True)
        
        # Guardar datos cada 10 generaciones
        if generation % 10 == 0:
            population_history.append([ind[:] for ind in population])  # Copia profunda
        
        # Calcular diversidad
        diversity = calculate_genetic_diversity(population)
        diversity_history.append(diversity)
        
        # Evolución estándar
        new_population = population[:ELITISM_COUNT]
        
        while len(new_population) < POPULATION_SIZE:
            parents = select_parents(population[:POPULATION_SIZE // 2], 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # 8. Análisis de evolución del fitness
    print("8. Analizando evolución del fitness...")
    analyze_fitness_evolution(population_history, run_output_folder)
    
    # 9. Análisis de diversidad genética
    print("9. Analizando diversidad genética...")
    plot_genetic_diversity_evolution(diversity_history, run_output_folder)
    
    # 10. Análisis de exploración vs explotación
    print("10. Analizando exploración vs explotación...")
    create_exploration_exploitation_plot(population_history, run_output_folder)
    
    # 11. Animación de distribución del fitness
    print("11. Creando animación de distribución del fitness...")
    create_fitness_distribution_animation(population_history, run_output_folder)
    
    # 12. Generar reporte final
    print("12. Generando reporte final...")
    generate_final_report(run_output_folder, final_solution, min_dists, avg_dists)

    print(f"\n=== ANÁLISIS COMPLETO FINALIZADO ===")
    print(f"Todos los resultados están en: '{run_output_folder}'")
    print("\nArchivos generados:")
    print("- Gráficos básicos: solucion_tsp.png, evolucion_algoritmo.png, distribucion_distancias.png")
    print("- Comparación de algoritmos: algorithm_convergence_comparison.png, algorithm_results_boxplot.png")
    print("- Análisis de parámetros: parameter_sensitivity_heatmap.png")
    print("- Análisis detallado: fitness_evolution_analysis.png, genetic_diversity_evolution.png")
    print("- Exploración: exploration_exploitation_analysis.png")
    print("- Animaciones: fitness_distribution_evolution.gif")
    print("- Reporte: reporte_final_analisis.txt")