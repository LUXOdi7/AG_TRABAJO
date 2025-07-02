"""
Script simplificado para ejecutar análisis específicos del TSP
Permite ejecutar funcionalidades individuales sin el análisis completo
"""

import sys
import os
import argparse

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from EJECUCION_TSP_GA import *
    print("✓ Módulo EJECUCION_TSP_GA importado correctamente")
except ImportError as e:
    print(f"❌ Error al importar EJECUCION_TSP_GA: {e}")
    sys.exit(1)

def ejecutar_analisis_basico():
    """Ejecutar solo el análisis básico del algoritmo genético"""
    print("🔬 EJECUTANDO ANÁLISIS BÁSICO")
    print("-" * 30)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(BASE_OUTPUT_FOLDER, f"Básico_{timestamp}")
    
    # Ejecutar algoritmo genético básico
    final_solution, gens, avg_dists, min_dists, distances_for_hists, generations_for_hists = genetic_algorithm(folder)
    
    # Generar gráficos básicos
    plot_solution(final_solution, folder)
    plot_evolution(gens, avg_dists, min_dists, folder)
    plot_distance_histograms(distances_for_hists, generations_for_hists, folder)
    
    print(f"✓ Análisis básico completado en: {folder}")
    return folder

def ejecutar_comparacion_algoritmos():
    """Ejecutar solo la comparación entre algoritmos"""
    print("🔄 EJECUTANDO COMPARACIÓN DE ALGORITMOS")
    print("-" * 35)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(BASE_OUTPUT_FOLDER, f"Comparación_{timestamp}")
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    compare_algorithm_performance(folder, num_runs=3)
    print(f"✓ Comparación completada en: {folder}")
    return folder

def ejecutar_analisis_parametros():
    """Ejecutar solo el análisis de sensibilidad de parámetros"""
    print("⚙️ EJECUTANDO ANÁLISIS DE PARÁMETROS")
    print("-" * 32)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(BASE_OUTPUT_FOLDER, f"Parámetros_{timestamp}")
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    analyze_parameter_sensitivity(folder)
    print(f"✓ Análisis de parámetros completado en: {folder}")
    return folder

def ejecutar_analisis_diversidad():
    """Ejecutar análisis de diversidad genética"""
    print("🧬 EJECUTANDO ANÁLISIS DE DIVERSIDAD")
    print("-" * 32)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(BASE_OUTPUT_FOLDER, f"Diversidad_{timestamp}")
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Ejecutar GA con captura de diversidad
    population_history = []
    diversity_history = []
    
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    print("Ejecutando algoritmo genético con análisis de diversidad...")
    for generation in range(min(500, GENERATIONS)):
        population.sort(key=evaluate_fitness, reverse=True)
        
        if generation % 10 == 0:
            population_history.append([ind[:] for ind in population])
        
        diversity = calculate_genetic_diversity(population)
        diversity_history.append(diversity)
        
        if generation % 50 == 0:
            print(f"Generación {generation}: Diversidad = {diversity:.2f}")
        
        # Evolución
        new_population = population[:ELITISM_COUNT]
        while len(new_population) < POPULATION_SIZE:
            parents = select_parents(population[:POPULATION_SIZE // 2], 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        population = new_population
    
    # Generar análisis
    analyze_fitness_evolution(population_history, folder)
    plot_genetic_diversity_evolution(diversity_history, folder)
    create_exploration_exploitation_plot(population_history, folder)
    
    print(f"✓ Análisis de diversidad completado en: {folder}")
    return folder

def main():
    """Función principal con menú de opciones"""
    parser = argparse.ArgumentParser(description="Análisis específico del TSP")
    parser.add_argument('--tipo', choices=['basico', 'comparacion', 'parametros', 'diversidad', 'menu'], 
                       default='menu', help='Tipo de análisis a ejecutar')
    
    args = parser.parse_args()
    
    if args.tipo == 'menu':
        print("🎯 ANÁLISIS ESPECÍFICO DEL TSP")
        print("=" * 30)
        print("Selecciona el tipo de análisis:")
        print("1. 📊 Análisis Básico (GA + gráficos básicos)")
        print("2. 🔄 Comparación de Algoritmos")
        print("3. ⚙️ Análisis de Parámetros")
        print("4. 🧬 Análisis de Diversidad Genética")
        print("5. 🚀 Todos los análisis")
        print("0. ❌ Salir")
        
        try:
            opcion = input("\nIngresa tu opción (0-5): ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Análisis cancelado por el usuario")
            return
        
        if opcion == '1':
            ejecutar_analisis_basico()
        elif opcion == '2':
            ejecutar_comparacion_algoritmos()
        elif opcion == '3':
            ejecutar_analisis_parametros()
        elif opcion == '4':
            ejecutar_analisis_diversidad()
        elif opcion == '5':
            print("🚀 EJECUTANDO TODOS LOS ANÁLISIS")
            print("=" * 30)
            folders = []
            folders.append(ejecutar_analisis_basico())
            folders.append(ejecutar_comparacion_algoritmos())
            folders.append(ejecutar_analisis_parametros())
            folders.append(ejecutar_analisis_diversidad())
            
            print("\n🎉 TODOS LOS ANÁLISIS COMPLETADOS")
            print("📁 Resultados en las siguientes carpetas:")
            for folder in folders:
                print(f"   • {folder}")
        elif opcion == '0':
            print("👋 Hasta luego!")
        else:
            print("❌ Opción no válida")
    
    else:
        # Ejecución directa con parámetros
        if args.tipo == 'basico':
            ejecutar_analisis_basico()
        elif args.tipo == 'comparacion':
            ejecutar_comparacion_algoritmos()
        elif args.tipo == 'parametros':
            ejecutar_analisis_parametros()
        elif args.tipo == 'diversidad':
            ejecutar_analisis_diversidad()

if __name__ == "__main__":
    # Asegurar que existe la carpeta base
    if not os.path.exists(BASE_OUTPUT_FOLDER):
        os.makedirs(BASE_OUTPUT_FOLDER)
    
    main()
