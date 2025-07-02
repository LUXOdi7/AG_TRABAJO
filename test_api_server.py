"""
Test de la API del servidor web
"""

import sys
import os
import json
import urllib.request
import urllib.parse
import time
import threading

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api():
    """Probar la API del servidor web"""
    
    # Parámetros de prueba
    params = {
        'startCity': 'Chiclayo',
        'populationSize': 10,
        'generations': 100,
        'mutationRate': 0.1
    }
    
    # Convertir a query string
    query_string = urllib.parse.urlencode(params)
    url = f"http://localhost:8001/api/execute?{query_string}"
    
    print(f"🔗 Probando URL: {url}")
    
    try:
        # Hacer petición GET
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                
                print("✅ Respuesta exitosa!")
                print(f"✓ Success: {data.get('success')}")
                print(f"✓ Mejor distancia: {data.get('bestDistance')}")
                print(f"✓ Tiempo de ejecución: {data.get('executionTime'):.2f}s")
                print(f"✓ Mejora: {data.get('improvement'):.2f}%")
                print(f"✓ Mejor ruta: {data.get('bestSolution')[:3]}...")
                
                return True
            else:
                print(f"❌ Error HTTP: {response.status}")
                return False
                
    except Exception as e:
        print(f"❌ Error en la petición: {e}")
        return False

def start_server_and_test():
    """Iniciar servidor y probar API"""
    import subprocess
    import time
    
    print("🚀 Iniciando servidor web...")
    
    # Iniciar servidor en subproceso
    try:
        server_process = subprocess.Popen([
            sys.executable, 'web_server.py', '--no-browser', '--port', '8001'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Esperar a que el servidor inicie
        time.sleep(3)
        
        print("🔍 Probando API...")
        success = test_api()
        
        # Terminar servidor
        server_process.terminate()
        server_process.wait()
        
        return success
        
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        return False

if __name__ == "__main__":
    success = start_server_and_test()
    if success:
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    else:
        print("\n💥 Algunas pruebas fallaron")
