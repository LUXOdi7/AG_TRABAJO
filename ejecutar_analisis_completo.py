"""
Script principal para ejecutar análisis completo del TSP
Ejecuta tanto el análisis básico como el avanzado con comparaciones
"""

import sys
import os

# Agregar el directorio actual al path para importar TSP_GA
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el archivo principal
try:
    import TSP_GA
    from analysis_tsp import TSPAnalyzer
    print("✓ Módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error al importar módulos: {e}")
    print("Asegúrate de que estén instaladas todas las dependencias:")
    print("pip install matplotlib numpy seaborn pandas scipy")
    sys.exit(1)

def main():
    """Función principal que ejecuta el análisis completo"""
    print("🔬 INICIANDO ANÁLISIS COMPLETO DEL TSP CON ALGORITMO GENÉTICO")
    print("=" * 60)
    
    try:
        # 1. Ejecutar análisis básico con TSP_GA
        print("\n📊 PARTE 1: Análisis Básico del Algoritmo Genético")
        print("-" * 45)
        
        # Esto ejecutará el código principal de TSP_GA.py
        exec(open('TSP_GA.py').read())
        
        print("\n✓ Análisis básico completado")
        
    except Exception as e:
        print(f"❌ Error en análisis básico: {e}")
    
    try:
        # 2. Ejecutar análisis avanzado
        print("\n🚀 PARTE 2: Análisis Avanzado y Comparativo")
        print("-" * 40)
        
        analyzer = TSPAnalyzer("Análisis_Avanzado_TSP")
        analyzer.generate_complete_report()
        
        print("\n✓ Análisis avanzado completado")
        
    except Exception as e:
        print(f"❌ Error en análisis avanzado: {e}")
        print("Nota: Algunas funciones avanzadas requieren librerías adicionales")
    
    print("\n🎉 ANÁLISIS COMPLETO FINALIZADO")
    print("=" * 35)
    print("\n📁 Revisa las siguientes carpetas para los resultados:")
    print("   • Resultados_TSP/ - Análisis básico")
    print("   • Análisis_Avanzado_TSP/ - Análisis comparativo completo")
    print("\n📈 Tipos de análisis generados:")
    print("   ✓ Evolución del algoritmo genético")
    print("   ✓ Comparación con otros algoritmos (Hill Climbing, PSO, etc.)")
    print("   ✓ Análisis de parámetros (población, mutación)")
    print("   ✓ Análisis de diversidad genética")
    print("   ✓ Análisis de escalabilidad")
    print("   ✓ Distribuciones estadísticas")
    print("   ✓ Animaciones de evolución")

if __name__ == "__main__":
    main()
