"""
Script rápido para probar la API del servidor web
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from EJECUCION_TSP_GA import *
    print("✓ Módulo EJECUCION_TSP_GA importado correctamente")
    
    # Probar que create_individual funciona
    individual = create_individual()
    print(f"✓ Individuo creado: {individual[:3]}...")
    print(f"✓ Tipo del primer elemento: {type(individual[0])}")
    
    # Probar que get_total_distance funciona
    distance = get_total_distance(individual)
    print(f"✓ Distancia calculada: {distance}")
    
    # Probar el algoritmo genético básico
    print("\n🧬 Probando algoritmo genético...")
    
    start_city = 'Chiclayo'
    population_size = 10
    generations = 50
    
    print(f"Ciudad de inicio: {start_city}")
    print(f"Tamaño de población: {population_size}")
    print(f"Generaciones: {generations}")
    
    # Crear población inicial
    population = []
    for _ in range(population_size):
        individual = create_individual()
        # Asegurar que comience con la ciudad seleccionada
        if start_city in CITY_NAMES and start_city in individual:
            if individual[0] != start_city:
                # Encontrar la posición de la ciudad de inicio y intercambiar
                city_pos = individual.index(start_city)
                individual[0], individual[city_pos] = individual[city_pos], individual[0]
        population.append(individual)
    
    print(f"✓ Población inicial creada")
    print(f"✓ Primer individuo: {population[0]}")
    print(f"✓ Ciudad de inicio correcta: {population[0][0] == start_city}")
    
    # Ejecutar algunas generaciones
    elitism_count = max(1, population_size // 10)
    
    for generation in range(generations):
        # Evaluar población
        population.sort(key=evaluate_fitness, reverse=True)
        
        # Evolucionar población (excepto en la última generación)
        if generation < generations - 1:
            new_population = population[:elitism_count]
            
            while len(new_population) < population_size:
                parents = select_parents(population[:population_size//2], 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
    
    # Obtener mejor solución
    population.sort(key=evaluate_fitness, reverse=True)
    best_solution = population[0]
    best_distance = get_total_distance(best_solution)
    
    print(f"\n✅ Algoritmo completado exitosamente!")
    print(f"✓ Mejor solución: {best_solution}")
    print(f"✓ Mejor distancia: {best_distance}")
    print(f"✓ Ciudad de inicio: {best_solution[0]}")
    
except ImportError as e:
    print(f"❌ Error al importar EJECUCION_TSP_GA: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error durante la ejecución: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
