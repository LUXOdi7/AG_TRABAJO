"""
Script de prueba para verificar que el servidor web funciona correctamente
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from web_server import TSPWebHandler
    from EJECUCION_TSP_GA import CITY_NAMES
    print("✓ Imports correctos")
    
    # Crear instancia del handler para probar
    class MockHandler(TSPWebHandler):
        def __init__(self):
            pass
        
        def send_json_response(self, data):
            print("📊 Respuesta generada:")
            print(f"   • Éxito: {data.get('success', 'N/A')}")
            if data.get('success'):
                print(f"   • Distancia: {data.get('bestDistance', 'N/A')} km")
                print(f"   • Ruta: {' → '.join(data.get('bestSolution', [])[:3])}...")
                print(f"   • Tiempo: {data.get('executionTime', 'N/A')} seg")
                print(f"   • Mejora: {data.get('improvement', 'N/A')}%")
            else:
                print(f"   • Error: {data.get('error', 'N/A')}")
    
    # Probar ejecución del algoritmo
    print("\n🧬 Probando ejecución del algoritmo...")
    handler = MockHandler()
    
    # Parámetros de prueba
    test_params = {
        'startCity': 'Chiclayo',
        'populationSize': 50,
        'generations': 100,
        'mutationRate': 0.1
    }
    
    print(f"📋 Parámetros de prueba: {test_params}")
    
    try:
        result = handler.execute_genetic_algorithm(
            test_params['startCity'],
            test_params['populationSize'], 
            test_params['generations'],
            test_params['mutationRate']
        )
        
        handler.send_json_response(result)
        print("\n✅ Prueba exitosa - El servidor debería funcionar correctamente")
        
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("Verifica que todos los archivos estén en el directorio correcto")

except Exception as e:
    print(f"❌ Error inesperado: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("🚀 Para iniciar el servidor web, ejecuta:")
print("   python web_server.py")
print("🌐 Luego abre: http://localhost:8000")
print("="*50)
