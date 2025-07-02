import random
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
from itertools import product
from copy import deepcopy
from EJECUCION_TSP_GA import (
    CITIES, CITY_NAMES, NUM_CITIES, DISTANCES_MATRIX,
    get_distance_from_matrix, get_total_distance, create_individual,
    evaluate_fitness, crossover, mutate, select_parents
)

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TSPAlgorithms:
    """Implementación de diferentes algoritmos para TSP para comparación"""
    
    @staticmethod
    def hill_climbing(max_iterations=2000):
        """Algoritmo Hill Climbing para TSP"""
        current_solution = create_individual()
        current_distance = get_total_distance(current_solution)
        
        iterations = []
        distances = []
        
        for iteration in range(max_iterations):
            # Generar vecino intercambiando dos ciudades
            neighbor = current_solution.copy()
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
            neighbor_distance = get_total_distance(neighbor)
            
            # Aceptar si es mejor
            if neighbor_distance < current_distance:
                current_solution = neighbor
                current_distance = neighbor_distance
            
            iterations.append(iteration)
            distances.append(current_distance)
        
        return current_solution, iterations, distances
    
    @staticmethod
    def simulated_annealing(max_iterations=2000, initial_temp=1000, cooling_rate=0.995):
        """Algoritmo Simulated Annealing para TSP"""
        current_solution = create_individual()
        current_distance = get_total_distance(current_solution)
        best_solution = current_solution.copy()
        best_distance = current_distance
        
        temperature = initial_temp
        iterations = []
        distances = []
        
        for iteration in range(max_iterations):
            # Generar vecino
            neighbor = current_solution.copy()
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
            neighbor_distance = get_total_distance(neighbor)
            
            # Calcular delta
            delta = neighbor_distance - current_distance
            
            # Aceptar si es mejor o con probabilidad basada en temperatura
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_solution = neighbor
                current_distance = neighbor_distance
                
                if current_distance < best_distance:
                    best_solution = current_solution.copy()
                    best_distance = current_distance
            
            temperature *= cooling_rate
            iterations.append(iteration)
            distances.append(best_distance)
        
        return best_solution, iterations, distances
    
    @staticmethod
    def particle_swarm_optimization(max_iterations=2000, num_particles=50):
        """Algoritmo PSO adaptado para TSP"""
        # Inicializar partículas como permutaciones
        particles = [create_individual() for _ in range(num_particles)]
        velocities = [np.random.random(NUM_CITIES) for _ in range(num_particles)]
        personal_best = particles.copy()
        personal_best_distances = [get_total_distance(p) for p in particles]
        
        global_best = min(particles, key=get_total_distance)
        global_best_distance = get_total_distance(global_best)
        
        iterations = []
        distances = []
        
        w = 0.7  # inercia
        c1 = 1.5  # componente cognitivo
        c2 = 1.5  # componente social
        
        for iteration in range(max_iterations):
            for i in range(num_particles):
                # Actualizar velocidad (adaptado para permutaciones)
                r1, r2 = random.random(), random.random()
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * np.random.random(NUM_CITIES) + 
                               c2 * r2 * np.random.random(NUM_CITIES))
                
                # Aplicar "velocidad" como probabilidad de intercambio
                particle = particles[i].copy()
                for j in range(NUM_CITIES):
                    if random.random() < abs(velocities[i][j]) * 0.1:
                        k = random.randint(0, NUM_CITIES - 1)
                        particle[j], particle[k] = particle[k], particle[j]
                
                particles[i] = particle
                current_distance = get_total_distance(particle)
                
                # Actualizar mejor personal
                if current_distance < personal_best_distances[i]:
                    personal_best[i] = particle.copy()
                    personal_best_distances[i] = current_distance
                    
                    # Actualizar mejor global
                    if current_distance < global_best_distance:
                        global_best = particle.copy()
                        global_best_distance = current_distance
            
            iterations.append(iteration)
            distances.append(global_best_distance)
        
        return global_best, iterations, distances

class TSPAnalyzer:
    """Clase principal para análisis y visualización del TSP"""
    
    def __init__(self, output_folder="Análisis_TSP"):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def compare_algorithms(self, num_runs=10):
        """Comparar diferentes algoritmos de TSP"""
        print("Comparando algoritmos...")
        
        algorithms = {
            'Genetic Algorithm': self._run_genetic_algorithm,
            'Hill Climbing': TSPAlgorithms.hill_climbing,
            'Simulated Annealing': TSPAlgorithms.simulated_annealing,
            'Particle Swarm': TSPAlgorithms.particle_swarm_optimization
        }
        
        results = {}
        
        for alg_name, alg_func in algorithms.items():
            print(f"Ejecutando {alg_name}...")
            alg_results = []
            
            for run in range(num_runs):
                start_time = time.time()
                solution, iterations, distances = alg_func()
                end_time = time.time()
                
                alg_results.append({
                    'solution': solution,
                    'final_distance': distances[-1],
                    'execution_time': end_time - start_time,
                    'iterations': iterations,
                    'distances': distances
                })
            
            results[alg_name] = alg_results
        
        # Generar gráficos comparativos
        self._plot_convergence_comparison(results)
        self._plot_boxplot_comparison(results)
        self._plot_execution_time_comparison(results)
        
        return results
    
    def _run_genetic_algorithm(self):
        """Ejecutar algoritmo genético simplificado para comparación"""
        from EJECUCION_TSP_GA import POPULATION_SIZE, GENERATIONS
        
        population = [create_individual() for _ in range(POPULATION_SIZE)]
        
        iterations = []
        distances = []
        best_solution = None
        best_distance = float('inf')
        
        for generation in range(GENERATIONS):
            population.sort(key=evaluate_fitness, reverse=True)
            current_best = population[0]
            current_distance = get_total_distance(current_best)
            
            if current_distance < best_distance:
                best_distance = current_distance
                best_solution = current_best.copy()
            
            iterations.append(generation)
            distances.append(best_distance)
            
            # Crear nueva población
            new_population = population[:20]  # Elitismo
            
            while len(new_population) < POPULATION_SIZE:
                parents = select_parents(population[:POPULATION_SIZE//2], 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_solution, iterations, distances
    
    def analyze_genetic_algorithm_parameters(self):
        """Análisis de parámetros del algoritmo genético"""
        print("Analizando parámetros del algoritmo genético...")
        
        # Análisis de población vs fitness
        self._analyze_population_size()
        
        # Heatmap de cruce vs mutación
        self._analyze_crossover_mutation_heatmap()
        
        # Análisis de diversidad genética
        self._analyze_genetic_diversity()
    
    def _analyze_population_size(self):
        """Analizar el impacto del tamaño de población"""
        population_sizes = [50, 100, 150, 200, 250, 300]
        results = []
        
        for pop_size in population_sizes:
            print(f"Probando tamaño de población: {pop_size}")
            final_distances = []
            
            for _ in range(5):  # 5 ejecuciones por tamaño
                population = [create_individual() for _ in range(pop_size)]
                
                for generation in range(500):  # Menos generaciones para acelerar
                    population.sort(key=evaluate_fitness, reverse=True)
                    
                    new_population = population[:max(1, pop_size//10)]  # Elitismo proporcional
                    
                    while len(new_population) < pop_size:
                        parents = select_parents(population[:pop_size//2], 2)
                        child = crossover(parents[0], parents[1])
                        child = mutate(child)
                        new_population.append(child)
                    
                    population = new_population
                
                best_distance = min(get_total_distance(ind) for ind in population)
                final_distances.append(best_distance)
            
            results.append({
                'population_size': pop_size,
                'avg_distance': np.mean(final_distances),
                'std_distance': np.std(final_distances)
            })
        
        # Gráfico
        plt.figure(figsize=(10, 6))
        pop_sizes = [r['population_size'] for r in results]
        avg_distances = [r['avg_distance'] for r in results]
        std_distances = [r['std_distance'] for r in results]
        
        plt.errorbar(pop_sizes, avg_distances, yerr=std_distances, 
                    marker='o', capsize=5, capthick=2)
        plt.xlabel('Tamaño de Población')
        plt.ylabel('Distancia Final Promedio')
        plt.title('Impacto del Tamaño de Población en el Rendimiento')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_folder}/population_size_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_crossover_mutation_heatmap(self):
        """Crear heatmap de parámetros de cruce vs mutación"""
        crossover_rates = [0.6, 0.7, 0.8, 0.9]
        mutation_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
        
        results_matrix = np.zeros((len(mutation_rates), len(crossover_rates)))
        
        for i, mut_rate in enumerate(mutation_rates):
            for j, cross_rate in enumerate(crossover_rates):
                print(f"Probando mutación: {mut_rate}, cruce: {cross_rate}")
                
                # Ejecutar GA con estos parámetros
                distances = []
                for _ in range(3):  # 3 ejecuciones por combinación
                    population = [create_individual() for _ in range(100)]
                    
                    for generation in range(300):
                        population.sort(key=evaluate_fitness, reverse=True)
                        
                        new_population = population[:10]  # Elitismo
                        
                        while len(new_population) < 100:
                            if random.random() < cross_rate:
                                parents = select_parents(population[:50], 2)
                                child = crossover(parents[0], parents[1])
                            else:
                                child = random.choice(population[:50]).copy()
                            
                            child = mutate(child, mut_rate)
                            new_population.append(child)
                        
                        population = new_population
                    
                    best_distance = min(get_total_distance(ind) for ind in population)
                    distances.append(best_distance)
                
                results_matrix[i, j] = np.mean(distances)
        
        # Crear heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(results_matrix, 
                   xticklabels=[f"{cr:.1f}" for cr in crossover_rates],
                   yticklabels=[f"{mr:.2f}" for mr in mutation_rates],
                   annot=True, fmt='.1f', cmap='viridis_r')
        plt.xlabel('Tasa de Cruce')
        plt.ylabel('Tasa de Mutación')
        plt.title('Heatmap: Impacto de Parámetros en Distancia Final')
        plt.savefig(f"{self.output_folder}/parameters_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_genetic_diversity(self):
        """Analizar la diversidad genética durante la evolución"""
        population = [create_individual() for _ in range(200)]
        
        generations = []
        diversities = []
        best_fitness = []
        avg_fitness = []
        worst_fitness = []
        
        for generation in range(1000):
            # Calcular diversidad (distancia promedio entre individuos)
            diversity = self._calculate_population_diversity(population)
            diversities.append(diversity)
            
            # Calcular estadísticas de fitness
            fitnesses = [evaluate_fitness(ind) for ind in population]
            distances = [get_total_distance(ind) for ind in population]
            
            best_fitness.append(min(distances))
            avg_fitness.append(np.mean(distances))
            worst_fitness.append(max(distances))
            generations.append(generation)
            
            # Evolucionar población
            population.sort(key=evaluate_fitness, reverse=True)
            new_population = population[:20]  # Elitismo
            
            while len(new_population) < 200:
                parents = select_parents(population[:100], 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Gráfico de diversidad y fitness
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Diversidad
        ax1.plot(generations, diversities, 'b-', linewidth=2)
        ax1.set_xlabel('Generación')
        ax1.set_ylabel('Diversidad Poblacional')
        ax1.set_title('Evolución de la Diversidad Genética')
        ax1.grid(True, alpha=0.3)
        
        # Fitness (mejor, promedio, peor)
        ax2.plot(generations, best_fitness, 'g-', label='Mejor', linewidth=2)
        ax2.plot(generations, avg_fitness, 'b-', label='Promedio', linewidth=2)
        ax2.plot(generations, worst_fitness, 'r-', label='Peor', linewidth=2)
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Distancia')
        ax2.set_title('Evolución del Fitness (Mejor/Promedio/Peor)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/genetic_diversity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_population_diversity(self, population):
        """Calcular la diversidad de una población"""
        total_differences = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Contar diferencias en posiciones
                differences = sum(1 for k in range(len(population[i])) 
                                if population[i][k] != population[j][k])
                total_differences += differences
                comparisons += 1
        
        return total_differences / comparisons if comparisons > 0 else 0
    
    def _plot_convergence_comparison(self, results):
        """Gráfico de convergencia comparativa"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (alg_name, alg_results) in enumerate(results.items()):
            # Promediar las curvas de convergencia
            max_len = max(len(run['distances']) for run in alg_results)
            avg_distances = []
            
            for iter_idx in range(max_len):
                iter_distances = []
                for run in alg_results:
                    if iter_idx < len(run['distances']):
                        iter_distances.append(run['distances'][iter_idx])
                    else:
                        iter_distances.append(run['distances'][-1])
                avg_distances.append(np.mean(iter_distances))
            
            plt.plot(range(max_len), avg_distances, 
                    color=colors[i % len(colors)], label=alg_name, linewidth=2)
        
        plt.xlabel('Iteración/Generación')
        plt.ylabel('Distancia Total')
        plt.title('Comparación de Convergencia entre Algoritmos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_folder}/convergence_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplot_comparison(self, results):
        """Boxplot de distancias finales"""
        plt.figure(figsize=(10, 6))
        
        data = []
        labels = []
        
        for alg_name, alg_results in results.items():
            final_distances = [run['final_distance'] for run in alg_results]
            data.append(final_distances)
            labels.append(alg_name)
        
        plt.boxplot(data, labels=labels)
        plt.ylabel('Distancia Final')
        plt.title('Distribución de Resultados por Algoritmo')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/boxplot_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_time_comparison(self, results):
        """Comparación de tiempos de ejecución"""
        plt.figure(figsize=(10, 6))
        
        alg_names = list(results.keys())
        avg_times = []
        std_times = []
        
        for alg_name in alg_names:
            times = [run['execution_time'] for run in results[alg_name]]
            avg_times.append(np.mean(times))
            std_times.append(np.std(times))
        
        plt.bar(alg_names, avg_times, yerr=std_times, capsize=5)
        plt.ylabel('Tiempo de Ejecución (segundos)')
        plt.title('Comparación de Tiempos de Ejecución')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/execution_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_route_animation(self, num_generations=100):
        """Crear animación de la evolución de la ruta"""
        print("Creando animación de evolución de ruta...")
        
        population = [create_individual() for _ in range(100)]
        best_routes = []
        
        for generation in range(num_generations):
            population.sort(key=evaluate_fitness, reverse=True)
            best_routes.append(population[0].copy())
            
            # Evolucionar población
            new_population = population[:10]
            while len(new_population) < 100:
                parents = select_parents(population[:50], 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                new_population.append(child)
            population = new_population
        
        # Crear animación
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            route = best_routes[frame]
            
            # Extraer coordenadas
            x_coords = [CITIES[city][0] for city in route]
            y_coords = [CITIES[city][1] for city in route]
            
            # Dibujar ruta
            ax.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 
                   'o-', color='blue', markersize=8, linewidth=2)
            
            # Anotar ciudades
            for city_name, coords in CITIES.items():
                ax.text(coords[0] + 0.1, coords[1] + 0.1, city_name, 
                       fontsize=10, ha='left', va='bottom')
                ax.plot(coords[0], coords[1], 'o', color='red', markersize=8)
            
            distance = get_total_distance(route)
            ax.set_title(f'Generación {frame}: Distancia = {distance:.2f} km')
            ax.set_xlabel('Coordenada X')
            ax.set_ylabel('Coordenada Y')
            ax.grid(True, alpha=0.3)
        
        anim = FuncAnimation(fig, animate, frames=num_generations, 
                           interval=200, repeat=True)
        
        # Guardar animación
        anim.save(f"{self.output_folder}/route_evolution.gif", writer='pillow', fps=5)
        plt.close()
        print(f"Animación guardada en {self.output_folder}/route_evolution.gif")
    
    def scalability_analysis(self):
        """Análisis de escalabilidad del algoritmo"""
        print("Realizando análisis de escalabilidad...")
        
        # Generar conjuntos de ciudades de diferentes tamaños
        city_counts = [5, 7, 10]  # Usamos tamaños pequeños para acelerar
        
        results = []
        
        for num_cities in city_counts:
            print(f"Analizando con {num_cities} ciudades...")
            
            # Crear subconjunto de ciudades
            subset_cities = dict(list(CITIES.items())[:num_cities])
            subset_names = list(subset_cities.keys())
            
            times = []
            final_distances = []
            
            for run in range(3):  # 3 ejecuciones por tamaño
                start_time = time.time()
                
                # Ejecutar GA adaptado para el subconjunto
                population = []
                for _ in range(50):  # Población más pequeña
                    individual = list(subset_names)
                    random.shuffle(individual)
                    population.append(individual)
                
                for generation in range(200):  # Menos generaciones
                    population.sort(key=lambda x: self._get_subset_distance(x, subset_cities), reverse=False)
                    
                    new_population = population[:5]  # Elitismo
                    while len(new_population) < 50:
                        parents = random.sample(population[:25], 2)
                        child = self._subset_crossover(parents[0], parents[1])
                        child = self._subset_mutate(child)
                        new_population.append(child)
                    
                    population = new_population
                
                end_time = time.time()
                
                best_solution = min(population, key=lambda x: self._get_subset_distance(x, subset_cities))
                best_distance = self._get_subset_distance(best_solution, subset_cities)
                
                times.append(end_time - start_time)
                final_distances.append(best_distance)
            
            results.append({
                'num_cities': num_cities,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_distance': np.mean(final_distances),
                'std_distance': np.std(final_distances)
            })
        
        # Gráficos de escalabilidad
        self._plot_scalability_results(results)
    
    def _get_subset_distance(self, route, subset_cities):
        """Calcular distancia para subconjunto de ciudades"""
        total_dist = 0
        for i in range(len(route) - 1):
            city1, city2 = route[i], route[i+1]
            if (city1, city2) in DISTANCES_MATRIX:
                total_dist += DISTANCES_MATRIX[(city1, city2)]
            elif (city2, city1) in DISTANCES_MATRIX:
                total_dist += DISTANCES_MATRIX[(city2, city1)]
        
        # Cerrar el ciclo
        if (route[-1], route[0]) in DISTANCES_MATRIX:
            total_dist += DISTANCES_MATRIX[(route[-1], route[0])]
        elif (route[0], route[-1]) in DISTANCES_MATRIX:
            total_dist += DISTANCES_MATRIX[(route[0], route[-1])]
        
        return total_dist
    
    def _subset_crossover(self, parent1, parent2):
        """Crossover para subconjunto"""
        size = len(parent1)
        child = [None] * size
        start_idx, end_idx = sorted(random.sample(range(size), 2))
        
        child[start_idx:end_idx + 1] = parent1[start_idx:end_idx + 1]
        fill_genes = [gene for gene in parent2 if gene not in child]
        
        current_fill_index = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill_genes[current_fill_index]
                current_fill_index += 1
        
        return child
    
    def _subset_mutate(self, individual):
        """Mutación para subconjunto"""
        if random.random() < 0.1:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def _plot_scalability_results(self, results):
        """Gráficos de resultados de escalabilidad"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        city_counts = [r['num_cities'] for r in results]
        avg_times = [r['avg_time'] for r in results]
        std_times = [r['std_time'] for r in results]
        avg_distances = [r['avg_distance'] for r in results]
        std_distances = [r['std_distance'] for r in results]
        
        # Tiempo vs número de ciudades
        ax1.errorbar(city_counts, avg_times, yerr=std_times, 
                    marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Número de Ciudades')
        ax1.set_ylabel('Tiempo de Ejecución (segundos)')
        ax1.set_title('Escalabilidad: Tiempo vs Tamaño del Problema')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Distancia vs número de ciudades
        ax2.errorbar(city_counts, avg_distances, yerr=std_distances, 
                    marker='s', capsize=5, capthick=2, color='red')
        ax2.set_xlabel('Número de Ciudades')
        ax2.set_ylabel('Distancia Final Promedio')
        ax2.set_title('Calidad de Solución vs Tamaño del Problema')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/scalability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def statistical_analysis(self, num_runs=30):
        """Análisis estadístico completo"""
        print(f"Realizando análisis estadístico con {num_runs} ejecuciones...")
        
        final_distances = []
        all_populations = []
        
        for run in range(num_runs):
            if run % 5 == 0:
                print(f"Ejecutando run {run + 1}/{num_runs}")
            
            population = [create_individual() for _ in range(100)]
            generation_populations = []
            
            for generation in range(500):
                population.sort(key=evaluate_fitness, reverse=True)
                
                if generation % 50 == 0:  # Guardar cada 50 generaciones
                    generation_populations.append({
                        'generation': generation,
                        'population': deepcopy(population)
                    })
                
                new_population = population[:10]
                while len(new_population) < 100:
                    parents = select_parents(population[:50], 2)
                    child = crossover(parents[0], parents[1])
                    child = mutate(child)
                    new_population.append(child)
                
                population = new_population
            
            best_distance = min(get_total_distance(ind) for ind in population)
            final_distances.append(best_distance)
            all_populations.extend(generation_populations)
        
        # Histograma de distancias finales
        self._plot_final_distances_histogram(final_distances)
        
        # Gráfico de dispersión exploración vs explotación
        self._plot_exploration_exploitation(all_populations)
    
    def _plot_final_distances_histogram(self, final_distances):
        """Histograma de distancias finales"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(final_distances, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(np.mean(final_distances), color='red', linestyle='--', 
                   label=f'Media: {np.mean(final_distances):.2f}')
        plt.axvline(np.median(final_distances), color='green', linestyle='--', 
                   label=f'Mediana: {np.median(final_distances):.2f}')
        
        plt.xlabel('Distancia Final')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Distancias Finales (Múltiples Ejecuciones)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_folder}/final_distances_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_exploration_exploitation(self, all_populations):
        """Gráfico de dispersión exploración vs explotación"""
        plt.figure(figsize=(12, 8))
        
        generations = []
        fitness_values = []
        colors = []
        
        for pop_data in all_populations:
            gen = pop_data['generation']
            population = pop_data['population']
            
            for individual in population:
                generations.append(gen)
                fitness_values.append(get_total_distance(individual))
                
                # Color basado en la calidad (exploración = alta distancia, explotación = baja)
                colors.append(get_total_distance(individual))
        
        scatter = plt.scatter(generations, fitness_values, c=colors, cmap='viridis_r', 
                             alpha=0.6, s=20)
        plt.colorbar(scatter, label='Distancia (Fitness)')
        
        plt.xlabel('Generación')
        plt.ylabel('Distancia (Fitness) de Individuos')
        plt.title('Exploración vs Explotación: Distribución de Fitness por Generación')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_folder}/exploration_exploitation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_complete_report(self):
        """Generar reporte completo de análisis"""
        print("Generando reporte completo de análisis...")
        
        # 1. Comparación de algoritmos
        algo_results = self.compare_algorithms(num_runs=5)
        
        # 2. Análisis de parámetros GA
        self.analyze_genetic_algorithm_parameters()
        
        # 3. Animación de evolución
        self.create_route_animation(num_generations=50)
        
        # 4. Análisis de escalabilidad
        self.scalability_analysis()
        
        # 5. Análisis estadístico
        self.statistical_analysis(num_runs=20)
        
        print(f"\n¡Análisis completo finalizado!")
        print(f"Todos los gráficos y análisis están en la carpeta: {self.output_folder}")
        
        # Generar resumen de resultados
        self._generate_summary_report(algo_results)
    
    def _generate_summary_report(self, algo_results):
        """Generar reporte resumen en texto"""
        with open(f"{self.output_folder}/resumen_analisis.txt", 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ANÁLISIS COMPLETO - TSP\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("1. COMPARACIÓN DE ALGORITMOS\n")
            f.write("-" * 30 + "\n")
            
            for alg_name, results in algo_results.items():
                distances = [r['final_distance'] for r in results]
                times = [r['execution_time'] for r in results]
                
                f.write(f"\n{alg_name}:\n")
                f.write(f"  - Distancia promedio: {np.mean(distances):.2f} ± {np.std(distances):.2f}\n")
                f.write(f"  - Mejor distancia: {min(distances):.2f}\n")
                f.write(f"  - Tiempo promedio: {np.mean(times):.3f} ± {np.std(times):.3f} segundos\n")
            
            f.write("\n\n2. ARCHIVOS GENERADOS\n")
            f.write("-" * 20 + "\n")
            f.write("- convergence_comparison.png: Comparación de convergencia\n")
            f.write("- boxplot_comparison.png: Distribución de resultados\n")
            f.write("- execution_time_comparison.png: Comparación de tiempos\n")
            f.write("- population_size_analysis.png: Análisis de tamaño de población\n")
            f.write("- parameters_heatmap.png: Heatmap de parámetros\n")
            f.write("- genetic_diversity_analysis.png: Análisis de diversidad\n")
            f.write("- route_evolution.gif: Animación de evolución\n")
            f.write("- scalability_analysis.png: Análisis de escalabilidad\n")
            f.write("- final_distances_histogram.png: Histograma de resultados\n")
            f.write("- exploration_exploitation.png: Exploración vs Explotación\n")
        
        print(f"Reporte resumen guardado en: {self.output_folder}/resumen_analisis.txt")

# Función principal para ejecutar el análisis
def main():
    """Función principal para ejecutar el análisis completo"""
    print("Iniciando análisis completo del TSP...")
    
    # Crear analizador
    analyzer = TSPAnalyzer("Análisis_Completo_TSP")
    
    # Ejecutar análisis completo
    analyzer.generate_complete_report()

if __name__ == "__main__":
    main()
