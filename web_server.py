"""
Servidor web para la aplicación del Algoritmo Genético TSP
Permite ejecutar el algoritmo desde una interfaz web
"""

import os
import sys
import json
import base64
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser
import time

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from EJECUCION_TSP_GA import *
    print("✓ Módulo EJECUCION_TSP_GA importado correctamente")
except ImportError as e:
    print(f"❌ Error al importar EJECUCION_TSP_GA: {e}")
    sys.exit(1)

class TSPWebHandler(SimpleHTTPRequestHandler):
    """Manejador HTTP personalizado para la aplicación TSP"""
    
    def do_GET(self):
        """Manejar peticiones GET"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/execute':
            self.handle_execute_algorithm(parsed_path.query)
        else:
            # Servir archivos estáticos
            super().do_GET()
    
    def do_POST(self):
        """Manejar peticiones POST"""
        if self.path == '/api/execute':
            self.handle_execute_algorithm_post()
        else:
            self.send_error(404)
    
    def handle_execute_algorithm(self, query_string):
        """Ejecutar el algoritmo genético con parámetros de la URL"""
        try:
            # Parsear parámetros
            params = parse_qs(query_string)
            
            start_city = params.get('startCity', ['Chiclayo'])[0]
            population_size = int(params.get('populationSize', [100])[0])
            generations = int(params.get('generations', [1000])[0])
            mutation_rate = float(params.get('mutationRate', [0.1])[0])
            
            print(f"🧬 Ejecutando GA: ciudad={start_city}, población={population_size}, generaciones={generations}")
            
            # Ejecutar algoritmo
            results = self.execute_genetic_algorithm(
                start_city, population_size, generations, mutation_rate
            )
            
            # Enviar respuesta JSON
            self.send_json_response(results)
            
        except Exception as e:
            print(f"❌ Error en handle_execute_algorithm: {e}")
            import traceback
            traceback.print_exc()
            self.send_error_response(str(e))
    
    def handle_execute_algorithm_post(self):
        """Manejar petición POST para ejecutar algoritmo"""
        try:
            # Leer datos POST
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            # Ejecutar algoritmo
            results = self.execute_genetic_algorithm(
                params.get('startCity', 'Chiclayo'),
                params.get('populationSize', 100),
                params.get('generations', 1000),
                params.get('mutationRate', 0.1)
            )
            
            # Enviar respuesta JSON
            self.send_json_response(results)
            
        except Exception as e:
            self.send_error_response(str(e))
    
    def execute_genetic_algorithm(self, start_city, population_size, generations, mutation_rate):
        """Ejecutar el algoritmo genético con parámetros dados"""
        
        print(f"🔬 Iniciando GA con: {start_city}, pop={population_size}, gen={generations}, mut={mutation_rate}")
        
        # Configurar parámetros globales temporalmente
        original_pop_size = globals().get('POPULATION_SIZE', 100)
        original_generations = globals().get('GENERATIONS', 2000)
        original_mutation_prob = globals().get('MUTATION_PROBABILITY', 0.1)
        
        try:
            # Actualizar parámetros
            globals()['POPULATION_SIZE'] = population_size
            globals()['GENERATIONS'] = generations
            globals()['MUTATION_PROBABILITY'] = mutation_rate
            
            # Definir elitismo como porcentaje de la población
            elitism_count = max(1, population_size // 10)  # 10% de la población
            
            print(f"📊 Parámetros configurados. Elitismo: {elitism_count}")
            
            # Crear directorio temporal para resultados
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Ejecutar algoritmo genético
                start_time = time.time()
                
                print(f"👥 Creando población inicial...")
                
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
                
                print(f"✓ Población creada. Primer individuo: {population[0][:3]}...")
                
                # Ejecutar evolución
                best_distances = []
                avg_distances = []
                best_solution = None
                best_distance = float('inf')
                
                print(f"🧬 Iniciando evolución...")
                
                for generation in range(generations):
                    # Evaluar población
                    population.sort(key=evaluate_fitness, reverse=True)
                    
                    # Estadísticas de la generación
                    distances = [get_total_distance(ind) for ind in population]
                    current_best = min(distances)
                    current_avg = sum(distances) / len(distances)
                    
                    best_distances.append(current_best)
                    avg_distances.append(current_avg)
                    
                    # Actualizar mejor solución global
                    if current_best < best_distance:
                        best_distance = current_best
                        best_solution = population[0].copy()
                    
                    # Progreso cada 10% de las generaciones
                    if generation % max(1, generations // 10) == 0:
                        print(f"📈 Gen {generation}/{generations}: mejor={current_best:.2f}, promedio={current_avg:.2f}")
                    
                    # Evolucionar población (excepto en la última generación)
                    if generation < generations - 1:
                        new_population = population[:elitism_count]  # Usar variable local
                        
                        while len(new_population) < population_size:
                            parents = select_parents(population[:population_size//2], 2)
                            child = crossover(parents[0], parents[1])
                            child = mutate(child)
                            new_population.append(child)
                        
                        population = new_population
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                print(f"✅ Evolución completada en {execution_time:.2f} segundos")
                
                # La mejor solución ya contiene nombres de ciudades
                best_route_names = best_solution.copy()
                
                # Calcular distancia inicial (para mejora)
                initial_solution = create_individual()
                initial_distance = get_total_distance(initial_solution)
                improvement = ((initial_distance - best_distance) / initial_distance) * 100
                
                print(f"🎯 Mejor distancia: {best_distance:.2f} (mejora: {improvement:.2f}%)")
                
                # Preparar resultados
                results = {
                    'success': True,
                    'bestSolution': best_route_names,
                    'bestDistance': best_distance,
                    'initialDistance': initial_distance,
                    'improvement': improvement,
                    'executionTime': execution_time,
                    'generations': generations,
                    'bestDistances': best_distances,
                    'avgDistances': avg_distances,
                    'parameters': {
                        'startCity': start_city,
                        'populationSize': population_size,
                        'generations': generations,
                        'mutationRate': mutation_rate
                    }
                }
                
                return results
                
        except Exception as e:
            print(f"❌ Error en execute_genetic_algorithm: {e}")
            import traceback
            traceback.print_exc()
            raise
                
        finally:
            # Restaurar parámetros originales
            globals()['POPULATION_SIZE'] = original_pop_size
            globals()['GENERATIONS'] = original_generations
            globals()['MUTATION_PROBABILITY'] = original_mutation_prob
    
    def send_json_response(self, data):
        """Enviar respuesta JSON"""
        response = json.dumps(data, ensure_ascii=False, indent=2)
        response_bytes = response.encode('utf-8')
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        self.wfile.write(response_bytes)
    
    def send_error_response(self, error_message):
        """Enviar respuesta de error"""
        error_data = {
            'success': False,
            'error': error_message
        }
        self.send_json_response(error_data)
    
    def end_headers(self):
        """Agregar headers CORS"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server(port=8000):
    """Iniciar el servidor web"""
    
    # Cambiar al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Configurar servidor
    server_address = ('localhost', port)
    httpd = HTTPServer(server_address, TSPWebHandler)
    
    print(f"🌐 Servidor iniciado en http://localhost:{port}")
    print(f"📁 Sirviendo archivos desde: {script_dir}")
    print("🧬 API disponible en /api/execute")
    print("\n🚀 Para abrir la aplicación, visita: http://localhost:8000")
    print("⏹️  Presiona Ctrl+C para detener el servidor")
    
    # Abrir navegador automáticamente
    def open_browser():
        time.sleep(1)  # Esperar a que el servidor esté listo
        webbrowser.open(f'http://localhost:{port}')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Servidor detenido por el usuario")
        httpd.shutdown()

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Servidor web para Algoritmo Genético TSP")
    parser.add_argument('--port', type=int, default=8000, 
                       help='Puerto del servidor (default: 8000)')
    parser.add_argument('--no-browser', action='store_true',
                       help='No abrir navegador automáticamente')
    
    args = parser.parse_args()
    
    print("🧬 SERVIDOR WEB - ALGORITMO GENÉTICO TSP")
    print("=" * 45)
    
    # Verificar que los archivos web existen
    required_files = ['index.html', 'styles.css', 'script.js']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Archivos faltantes: {', '.join(missing_files)}")
        print("Asegúrate de que index.html, styles.css y script.js estén en el directorio actual")
        return
    
    if args.no_browser:
        print("🌐 Modo sin navegador activado")
    
    # Iniciar servidor
    start_server(args.port)

if __name__ == "__main__":
    main()
