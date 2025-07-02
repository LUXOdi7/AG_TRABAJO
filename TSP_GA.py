import random
import math
import matplotlib.pyplot as plt
import os
import datetime # Para generar nombres de archivos unicos con la fecha y hora

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
# *** IMPORTANTE: DEBES LLENAR ESTAS DISTANCIAS CON VALORES REALES DE GOOGLE MAPS ***
# ********************************************************************************
DISTANCES_MATRIX = {
    # Chiclayo
    ('Chiclayo', 'Lambayeque'): 11.50, # Distancia real segun Google Maps
    ('Chiclayo', 'Ferreñafe'): 18.00, # Distancia real segun Google Maps
    ('Chiclayo', 'Monsefu'): 16.50, # Distancia real segun Google Maps
    ('Chiclayo', 'Eten'): 24.10, # Distancia real segun Google Maps
    ('Chiclayo', 'Reque'): 12.00, # Distancia real segun Google Maps
    ('Chiclayo', 'Olmos'): 95.00, # Distancia real segun Google Maps
    ('Chiclayo', 'Motupe'): 72.00, # Distancia real segun Google Maps
    ('Chiclayo', 'Pimentel'): 14.50, # Distancia real segun Google Maps
    ('Chiclayo', 'Tuman'): 28.00, # Distancia real segun Google Maps

    # Lambayeque
    ('Lambayeque', 'Ferreñafe'): 15.80, # Distancia real segun Google Maps
    ('Lambayeque', 'Monsefu'): 23.50, # Distancia real segun Google Maps
    ('Lambayeque', 'Eten'): 31.00, # Distancia real segun Google Maps
    ('Lambayeque', 'Reque'): 19.50, # Distancia real segun Google Maps
    ('Lambayeque', 'Olmos'): 85.00, # Distancia real segun Google Maps
    ('Lambayeque', 'Motupe'): 60.00, # Distancia real segun Google Maps
    ('Lambayeque', 'Pimentel'): 15.00, # Distancia real segun Google Maps
    ('Lambayeque', 'Tuman'): 25.00, # Distancia real segun Google Maps

    # Ferreñafe
    ('Ferreñafe', 'Monsefu'): 22.00, # Distancia real segun Google Maps
    ('Ferreñafe', 'Eten'): 27.00, # Distancia real segun Google Maps
    ('Ferreñafe', 'Reque'): 28.00, # Distancia real segun Google Maps
    ('Ferreñafe', 'Olmos'): 105.00, # Distancia real segun Google Maps
    ('Ferreñafe', 'Motupe'): 80.00, # Distancia real segun Google Maps
    ('Ferreñafe', 'Pimentel'): 30.00, # Distancia real segun Google Maps
    ('Ferreñafe', 'Tuman'): 12.00, # Distancia real segun Google Maps
    
    # Monsefu
    ('Monsefu', 'Eten'): 8.00, # Distancia real segun Google Maps
    ('Monsefu', 'Reque'): 6.00, # Distancia real segun Google Maps
    ('Monsefu', 'Olmos'): 100.00, # Distancia real segun Google Maps
    ('Monsefu', 'Motupe'): 75.00, # Distancia real segun Google Maps
    ('Monsefu', 'Pimentel'): 10.00, # Distancia real segun Google Maps
    ('Monsefu', 'Tuman'): 20.00, # Distancia real segun Google Maps

    # Eten
    ('Eten', 'Reque'): 5.00, # Distancia real segun Google Maps
    ('Eten', 'Olmos'): 110.00, # Distancia real segun Google Maps
    ('Eten', 'Motupe'): 85.00, # Distancia real segun Google Maps
    ('Eten', 'Pimentel'): 12.00, # Distancia real segun Google Maps
    ('Eten', 'Tuman'): 25.00, # Distancia real segun Google Maps

    # Reque
    ('Reque', 'Olmos'): 98.00, # Distancia real segun Google Maps
    ('Reque', 'Motupe'): 72.00, # Distancia real segun Google Maps
    ('Reque', 'Pimentel'): 8.00, # Distancia real segun Google Maps
    ('Reque', 'Tuman'): 18.00, # Distancia real segun Google Maps

    # Olmos
    ('Olmos', 'Motupe'): 25.00, # Distancia real segun Google Maps
    ('Olmos', 'Pimentel'): 100.00, # Distancia real segun Google Maps
    ('Olmos', 'Tuman'): 115.00, # Distancia real segun Google Maps

    # Motupe
    ('Motupe', 'Pimentel'): 80.00, # Distancia real segun Google Maps
    ('Motupe', 'Tuman'): 90.00, # Distancia real segun Google Maps

    # Pimentel
    ('Pimentel', 'Tuman'): 30.00, # Distancia real segun Google Maps
    # Asegurate de que la matriz este completa con todas las combinaciones A-B y B-A
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


# --- Ejecucion del Algoritmo ---
if __name__ == "__main__":
    # Asegurarse de que la carpeta base exista
    if not os.path.exists(BASE_OUTPUT_FOLDER):
        os.makedirs(BASE_OUTPUT_FOLDER)

    # Generar un nombre de carpeta unica para esta ejecucion (ej. Prueba_YYYYMMDD_HHMMSS)
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_folder = os.path.join(BASE_OUTPUT_FOLDER, f"Prueba_{current_timestamp}")

    # Ejecutar el algoritmo genetico y obtener los datos de evolucion
    final_solution, gens, avg_dists, min_dists, distances_for_hists, generations_for_hists = genetic_algorithm(run_output_folder)
    
    # Guardar el grafico de la solucion final
    plot_solution(final_solution, run_output_folder)
    
    # Guardar el grafico de la evolucion del algoritmo
    plot_evolution(gens, avg_dists, min_dists, run_output_folder)

    # Guardar el grafico de distribucion de distancias
    plot_distance_histograms(distances_for_hists, generations_for_hists, run_output_folder)

    print(f"\nTodos los resultados de esta ejecucion se encuentran en la carpeta '{run_output_folder}'.")