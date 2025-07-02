import random
import math
import matplotlib.pyplot as plt
import os
import datetime # Para generar nombres de archivos únicos con la fecha y hora

# --- Configuración del Problema del Vendedor Viajero (TSP) ---
# Definimos las coordenadas de las ciudades. Puedes añadir más o cambiar estas.
# Cada tupla representa (x, y) de una ciudad.
# Ejemplo con 10 ciudades.
CITIES = {
    'A': (0, 0),
    'B': (1, 5),
    'C': (4, 1),
    'D': (6, 4),
    'E': (8, 0),
    'F': (2, 7),
    'G': (7, 2),
    'H': (3, 3),
    'I': (5, 6),
    'J': (9, 3)
}
NUM_CITIES = len(CITIES) # Número total de ciudades
CITY_NAMES = list(CITIES.keys()) # Lista de los nombres de las ciudades

# --- Algoritmo Genético: Parámetros ---
POPULATION_SIZE = 200     # Tamaño de la población en cada generación
GENERATIONS = 2000        # Número de generaciones a ejecutar el algoritmo
ELITISM_COUNT = 20        # Número de los mejores individuos que pasan directamente a la siguiente generación
MUTATION_PROBABILITY = 0.1 # Probabilidad de que un individuo sufra mutación

# --- Rutas de Archivos y Carpetas ---
OUTPUT_FOLDER = "Resultados_TSP" # Nombre de la carpeta de salida

# --- Funciones Auxiliares ---

def calculate_distance(city1, city2):
    """
    Calcula la distancia euclidiana entre dos ciudades.
    city1 y city2 son tuplas (x, y) de las coordenadas.
    """
    return math.dist(city1, city2)

def get_total_distance(route):
    """
    Calcula la distancia total de una ruta dada.
    Una ruta es una lista de nombres de ciudades (ej: ['A', 'B', 'C', ...]).
    La ruta comienza y termina en la primera ciudad.
    """
    total_dist = 0
    # Recorrer la ruta, sumando la distancia entre ciudades consecutivas
    for i in range(NUM_CITIES - 1):
        city1_coords = CITIES[route[i]]
        city2_coords = CITIES[route[i+1]]
        total_dist += calculate_distance(city1_coords, city2_coords)
    
    # Sumar la distancia de regreso a la ciudad de inicio (para cerrar el ciclo)
    total_dist += calculate_distance(CITIES[route[-1]], CITIES[route[0]])
    return total_dist

# --- Operaciones del Algoritmo Genético ---

def create_individual():
    """
    Crea un individuo (cromosoma) que representa una ruta.
    Un individuo es una permutación aleatoria de los nombres de las ciudades.
    """
    individual = list(CITY_NAMES) # Inicialmente, en orden alfabético
    random.shuffle(individual)    # Mezclamos para obtener una permutación aleatoria
    return individual

def evaluate_fitness(individual):
    """
    Evalúa la 'aptitud' de un individuo.
    En el TSP, una menor distancia total significa una mayor aptitud.
    Por convención, la aptitud se suele definir como 1 / (distancia total).
    Esto es para que el algoritmo busque maximizar la aptitud (mayor valor = mejor).
    """
    distance = get_total_distance(individual)
    # Evitar división por cero si la distancia fuera 0 (no aplicable en TSP real con ciudades distintas)
    return 1 / distance if distance > 0 else float('inf')

def crossover(parent1, parent2):
    """
    Realiza el cruce (crossover) entre dos padres utilizando el Crossover de Orden (OX1).
    Este operador es adecuado para problemas donde el orden de los genes es importante
    (como el TSP, donde la ruta es una permutación).
    
    Pasos:
    1. Seleccionar un segmento aleatorio del primer padre.
    2. Copiar ese segmento al hijo en la misma posición.
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
        if child[i] is None: # Si la posición está vacía
            child[i] = fill_genes[current_fill_index]
            current_fill_index += 1
            
    return child

def mutate(individual, prob=MUTATION_PROBABILITY):
    """
    Realiza una mutación en el individuo mediante el intercambio de dos genes (Swap Mutation).
    La mutación introduce variabilidad en la población y ayuda a evitar óptimos locales.
    """
    if random.random() < prob: # La mutación ocurre solo con una cierta probabilidad
        # Seleccionar dos posiciones aleatorias para intercambiar
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def select_parents(population, num_parents_to_select):
    """
    Selecciona padres de la población.
    Aquí se usa un método de selección por torneo:
    Se eligen aleatoriamente 'k' individuos y se selecciona el mejor de ellos.
    Se repite 'num_parents_to_select' veces.
    """
    selected_parents = []
    tournament_size = 5 # Tamaño del torneo (cuántos individuos compiten)
    for _ in range(num_parents_to_select):
        # Seleccionar individuos aleatorios para el torneo
        competitors = random.sample(population, min(tournament_size, len(population)))
        # El ganador es el que tiene la mayor aptitud
        winner = max(competitors, key=evaluate_fitness)
        selected_parents.append(winner)
    return selected_parents

def genetic_algorithm():
    """
    Implementación principal del Algoritmo Genético para el TSP.
    Además de encontrar la solución, registra la evolución y guarda archivos.
    """
    # Crear la carpeta de salida si no existe
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Carpeta '{OUTPUT_FOLDER}' creada.")

    # Generar un nombre de archivo único basado en la fecha y hora
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(OUTPUT_FOLDER, f"evolucion_tsp_{timestamp}.txt")

    # Listas para almacenar datos para el gráfico de evolución
    generations_data = []
    avg_distances = []
    min_distances = []

    # 1. Inicialización de la población
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_solution_overall = None
    min_distance_overall = float('inf')
    
    print("Iniciando algoritmo genético para TSP...")

    # Abrir el archivo de log para escribir
    with open(log_filename, 'w') as log_file:
        log_file.write(f"--- Registro de la Evolución del Algoritmo Genético para TSP ---\n")
        log_file.write(f"Parámetros: Población={POPULATION_SIZE}, Generaciones={GENERATIONS}, Elitismo={ELITISM_COUNT}, Mutación={MUTATION_PROBABILITY}\n\n")

        for generation in range(GENERATIONS):
            # 2. Evaluación de la aptitud de cada individuo
            population.sort(key=evaluate_fitness, reverse=True) # Ordenar por aptitud, de mayor a menor

            # Obtener el mejor individuo de la generación actual
            current_best_individual = population[0]
            current_min_distance = get_total_distance(current_best_individual)
            
            # Calcular la distancia promedio de la población actual
            all_distances = [get_total_distance(ind) for ind in population]
            current_avg_distance = sum(all_distances) / len(all_distances)

            # Almacenar datos para el gráfico
            generations_data.append(generation)
            avg_distances.append(current_avg_distance)
            min_distances.append(current_min_distance)

            # Actualizar la mejor solución global encontrada hasta ahora
            if current_min_distance < min_distance_overall:
                min_distance_overall = current_min_distance
                best_solution_overall = current_best_individual
                log_file.write(f"Generación {generation}: NUEVA MEJOR! Distancia: {min_distance_overall:.2f} - Ruta: {best_solution_overall}\n")
            else:
                 log_file.write(f"Generación {generation}: Mejor de la generación: {current_min_distance:.2f}, Promedio: {current_avg_distance:.2f}\n")

            # Imprimir progreso en consola cada 100 generaciones
            if generation % 100 == 0 or generation == GENERATIONS - 1:
                print(f"Generación {generation}: Mejor distancia actual: {current_min_distance:.2f}, Promedio de distancia: {current_avg_distance:.2f}")
            
            # Si se encuentra una solución "óptima" (distancia muy baja o que no mejora)
            # Podrías agregar una condición de parada temprana aquí si lo deseas.

            # 3. Creación de la nueva población
            new_population = []
            
            # Elitismo: Mantener los mejores individuos de la generación actual
            new_population.extend(population[:ELITISM_COUNT])
            
            # Mientras la nueva población no alcance el tamaño deseado:
            while len(new_population) < POPULATION_SIZE:
                # 4. Selección de padres
                # Seleccionamos entre los mejores individuos de la población actual
                parents_for_breeding = select_parents(population[:POPULATION_SIZE // 2], 2)
                parent1, parent2 = parents_for_breeding[0], parents_for_breeding[1]
                
                # 5. Cruce (Crossover)
                child = crossover(parent1, parent2)
                
                # 6. Mutación
                child = mutate(child)
                
                new_population.append(child)
                
            population = new_population # La nueva población reemplaza a la antigua

    print("\nAlgoritmo genético finalizado.")
    print(f"Mejor solución encontrada: {best_solution_overall}")
    print(f"Distancia mínima total: {min_distance_overall:.2f}")

    return best_solution_overall, generations_data, avg_distances, min_distances

# --- Visualización de la Solución ---

def plot_solution(solution_route, folder_path, timestamp):
    """
    Grafica la ruta encontrada por el algoritmo genético y la guarda en la carpeta especificada.
    """
    plt.figure(figsize=(10, 8))
    
    # Extraer coordenadas para graficar
    x_coords = [CITIES[city][0] for city in solution_route]
    y_coords = [CITIES[city][1] for city in solution_route]

    # Conectar las ciudades en el orden de la ruta
    plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'o-', color='blue', markersize=8, linewidth=2, label='Ruta Optimizada')
    
    # Anotar las ciudades
    for city_name, coords in CITIES.items():
        plt.text(coords[0] + 0.1, coords[1] + 0.1, city_name, fontsize=12, ha='left', va='bottom')
        plt.plot(coords[0], coords[1], 'o', color='red', markersize=10) # Marcar las ciudades
    
    # Marcar la ciudad de inicio/fin
    start_city_coords = CITIES[solution_route[0]]
    plt.plot(start_city_coords[0], start_city_coords[1], 'o', color='green', markersize=12, label='Ciudad de Inicio/Fin')

    plt.title(f'Solución TSP - Distancia: {get_total_distance(solution_route):.2f}')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True)
    plt.legend()
    
    # Guardar el gráfico
    plot_filename = os.path.join(folder_path, f"solucion_tsp_{timestamp}.png")
    plt.savefig(plot_filename)
    plt.close() # Cierra la figura para liberar memoria
    print(f"Gráfico de la solución guardado en '{plot_filename}'")

def plot_evolution(generations, avg_distances, min_distances, folder_path, timestamp):
    """
    Grafica la evolución de la distancia promedio y mínima a lo largo de las generaciones.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(generations, avg_distances, label='Distancia Promedio de la Población', color='blue')
    plt.plot(generations, min_distances, label='Distancia Mínima (Mejor Solución)', color='orange')
    plt.title('Evolución del Algoritmo Genético para TSP')
    plt.xlabel('Generación')
    plt.ylabel('Distancia')
    plt.grid(True)
    plt.legend()
    
    # Guardar el gráfico
    evolution_plot_filename = os.path.join(folder_path, f"evolucion_algoritmo_{timestamp}.png")
    plt.savefig(evolution_plot_filename)
    plt.close() # Cierra la figura para liberar memoria
    print(f"Gráfico de evolución guardado en '{evolution_plot_filename}'")


# --- Ejecución del Algoritmo ---
if __name__ == "__main__":
    # Generar un timestamp único para esta ejecución
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ejecutar el algoritmo genético y obtener los datos de evolución
    final_solution, gens, avg_dists, min_dists = genetic_algorithm()
    
    # Guardar el gráfico de la solución final
    plot_solution(final_solution, OUTPUT_FOLDER, current_timestamp)
    
    # Guardar el gráfico de la evolución del algoritmo
    plot_evolution(gens, avg_dists, min_dists, OUTPUT_FOLDER, current_timestamp)

    print(f"\nTodos los resultados se encuentran en la carpeta '{OUTPUT_FOLDER}'.")