"""
Test directo de la lógica del servidor web
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from web_server import TSPWebHandler
    
    print("✓ Módulo web_server importado correctamente")
    
    # Crear instancia del handler
    handler = TSPWebHandler(None, None, None)
    
    print("🧬 Probando execute_genetic_algorithm...")
    
    # Probar con parámetros pequeños para rapidez
    results = handler.execute_genetic_algorithm(
        start_city='Chiclayo',
        population_size=10,
        generations=50,
        mutation_rate=0.1
    )
    
    print("✅ Algoritmo ejecutado exitosamente!")
    print(f"✓ Success: {results.get('success')}")
    print(f"✓ Mejor distancia: {results.get('bestDistance')}")
    print(f"✓ Tiempo de ejecución: {results.get('executionTime'):.2f}s")
    print(f"✓ Mejora: {results.get('improvement'):.2f}%")
    print(f"✓ Ciudad de inicio: {results.get('bestSolution')[0]}")
    print(f"✓ Mejor ruta: {results.get('bestSolution')[:5]}...")
    
    # Verificar que la ciudad de inicio es correcta
    if results.get('bestSolution')[0] == 'Chiclayo':
        print("🎯 ¡Ciudad de inicio correcta!")
    else:
        print(f"⚠️ Ciudad de inicio incorrecta: {results.get('bestSolution')[0]}")
        
    print("\n🎉 ¡Test completado exitosamente!")
    
except Exception as e:
    print(f"❌ Error durante el test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
