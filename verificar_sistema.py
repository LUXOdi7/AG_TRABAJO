"""
Script de verificación para comprobar dependencias y funciones
"""

import sys
import os

def verificar_dependencias():
    """Verificar que todas las dependencias estén instaladas"""
    dependencias = [
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('pandas', 'pd'),
        ('scipy', None)
    ]
    
    print("🔍 VERIFICANDO DEPENDENCIAS")
    print("-" * 25)
    
    dependencias_ok = True
    
    for dep, alias in dependencias:
        try:
            if alias:
                exec(f"import {dep} as {alias}")
            else:
                exec(f"import {dep}")
            print(f"✓ {dep}")
        except ImportError as e:
            print(f"❌ {dep} - Error: {e}")
            dependencias_ok = False
    
    return dependencias_ok

def verificar_funciones():
    """Verificar que todas las funciones estén definidas"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n🔍 VERIFICANDO FUNCIONES TSP_GA")
    print("-" * 30)
    
    try:
        from TSP_GA import (
            # Funciones básicas
            get_distance_from_matrix,
            get_total_distance,
            create_individual,
            evaluate_fitness,
            crossover,
            mutate,
            select_parents,
            genetic_algorithm,
            
            # Funciones de visualización
            plot_solution,
            plot_evolution,
            plot_distance_histograms,
            
            # Funciones de análisis avanzado
            analyze_fitness_evolution,
            calculate_genetic_diversity,
            plot_genetic_diversity_evolution,
            create_exploration_exploitation_plot,
            create_fitness_distribution_animation,
            compare_algorithm_performance,
            analyze_parameter_sensitivity,
            generate_final_report,
            
            # Datos
            CITIES,
            CITY_NAMES,
            DISTANCES_MATRIX,
            NUM_CITIES,
            POPULATION_SIZE,
            GENERATIONS,
            MUTATION_PROBABILITY,
            ELITISM_COUNT
        )
        
        print("✓ Todas las funciones principales")
        print("✓ Todas las funciones de análisis avanzado")
        print("✓ Todas las variables globales")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error al importar funciones: {e}")
        return False

def verificar_analysis_tsp():
    """Verificar funciones de analysis_tsp.py"""
    print("\n🔍 VERIFICANDO ANÁLISIS AVANZADO")
    print("-" * 28)
    
    try:
        from analysis_tsp import TSPAnalyzer, TSPAlgorithms
        print("✓ Clase TSPAnalyzer")
        print("✓ Clase TSPAlgorithms")
        return True
    except ImportError as e:
        print(f"⚠️ analysis_tsp.py no disponible: {e}")
        print("  (Esto es opcional para el análisis básico)")
        return False

def main():
    """Función principal de verificación"""
    print("🧪 VERIFICACIÓN DEL SISTEMA DE ANÁLISIS TSP")
    print("=" * 45)
    
    # Verificar dependencias
    deps_ok = verificar_dependencias()
    
    # Verificar funciones principales
    funcs_ok = verificar_funciones()
    
    # Verificar análisis avanzado (opcional)
    analysis_ok = verificar_analysis_tsp()
    
    print("\n📋 RESUMEN DE VERIFICACIÓN")
    print("-" * 25)
    
    if deps_ok and funcs_ok:
        print("✅ Sistema listo para ejecutar análisis básico")
        print("✅ TSP_GA.py completamente funcional")
        
        if analysis_ok:
            print("✅ Análisis avanzado disponible")
        else:
            print("⚠️ Análisis avanzado limitado")
        
        print("\n🚀 COMANDOS RECOMENDADOS:")
        print("   python TSP_GA.py                    # Análisis completo")
        print("   python analisis_especifico.py       # Análisis selectivo")
        if analysis_ok:
            print("   python analysis_tsp.py              # Análisis avanzado separado")
            print("   python ejecutar_analisis_completo.py # Ambos análisis")
        
    else:
        print("❌ Sistema no está listo")
        
        if not deps_ok:
            print("\n🔧 Para instalar dependencias:")
            print("   pip install -r requirements.txt")
        
        if not funcs_ok:
            print("\n🔧 Revisa el archivo TSP_GA.py")
    
    print(f"\n📁 Directorio de trabajo: {os.getcwd()}")
    print(f"📁 Carpeta de resultados: Resultados_TSP/")

if __name__ == "__main__":
    main()
