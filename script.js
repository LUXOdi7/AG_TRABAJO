// =========================
// CONFIGURACIÓN GLOBAL
// =========================
const CONFIG = {
    cities: {
        'Chiclayo': { x: 6, y: 5 },
        'Lambayeque': { x: 5, y: 6 },
        'Ferreñafe': { x: 7, y: 7 },
        'Monsefu': { x: 7, y: 4 },
        'Eten': { x: 8, y: 3 },
        'Reque': { x: 6.5, y: 3.5 },
        'Olmos': { x: 1, y: 9 },
        'Motupe': { x: 3, y: 8 },
        'Pimentel': { x: 6.5, y: 5.5 },
        'Tuman': { x: 7.5, y: 6 }
    },
    distances: {
        'Chiclayo-Lambayeque': 4.70, 'Chiclayo-Ferreñafe': 29.50, 'Chiclayo-Monsefu': 18.50,
        'Chiclayo-Eten': 20.00, 'Chiclayo-Reque': 10.00, 'Chiclayo-Olmos': 106.00,
        'Chiclayo-Motupe': 81.00, 'Chiclayo-Pimentel': 14.00, 'Chiclayo-Tuman': 32.00,
        'Lambayeque-Ferreñafe': 25.00, 'Lambayeque-Monsefu': 24.00, 'Lambayeque-Eten': 26.00,
        'Lambayeque-Reque': 15.00, 'Lambayeque-Olmos': 110.00, 'Lambayeque-Motupe': 85.00,
        'Lambayeque-Pimentel': 18.00, 'Lambayeque-Tuman': 28.00, 'Ferreñafe-Monsefu': 35.00,
        'Ferreñafe-Eten': 40.00, 'Ferreñafe-Reque': 38.00, 'Ferreñafe-Olmos': 85.00,
        'Ferreñafe-Motupe': 60.00, 'Ferreñafe-Pimentel': 43.00, 'Ferreñafe-Tuman': 15.00,
        'Monsefu-Eten': 12.00, 'Monsefu-Reque': 8.00, 'Monsefu-Olmos': 125.00,
        'Monsefu-Motupe': 100.00, 'Monsefu-Pimentel': 8.00, 'Monsefu-Tuman': 45.00,
        'Eten-Reque': 6.00, 'Eten-Olmos': 130.00, 'Eten-Motupe': 105.00,
        'Eten-Pimentel': 12.00, 'Eten-Tuman': 50.00, 'Reque-Olmos': 125.00,
        'Reque-Motupe': 100.00, 'Reque-Pimentel': 10.00, 'Reque-Tuman': 42.00,
        'Olmos-Motupe': 25.00, 'Olmos-Pimentel': 120.00, 'Olmos-Tuman': 90.00,
        'Motupe-Pimentel': 95.00, 'Motupe-Tuman': 65.00, 'Pimentel-Tuman': 48.00
    }
};

// Variables globales
let isExecuting = false;
let currentResults = null;
let routeChart = null;
let evolutionChart = null;

// =========================
// INICIALIZACIÓN
// =========================
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Event listeners
    document.getElementById('executeBtn').addEventListener('click', executeGeneticAlgorithm);
    document.getElementById('mutationRate').addEventListener('input', updateMutationValue);
    document.getElementById('downloadBtn').addEventListener('click', downloadResults);
    document.getElementById('newAnalysisBtn').addEventListener('click', resetAnalysis);
    
    // Modal close
    document.querySelector('.close').addEventListener('click', closeErrorModal);
    window.addEventListener('click', function(event) {
        const modal = document.getElementById('errorModal');
        if (event.target === modal) {
            closeErrorModal();
        }
    });
    
    // Inicializar valor de mutación
    updateMutationValue();
    
    console.log('🧬 Aplicación TSP inicializada correctamente');
}

function updateMutationValue() {
    const slider = document.getElementById('mutationRate');
    const valueSpan = document.getElementById('mutationValue');
    valueSpan.textContent = parseFloat(slider.value).toFixed(2);
}

// =========================
// ALGORITMO GENÉTICO
// =========================
async function executeGeneticAlgorithm() {
    if (isExecuting) return;
    
    try {
        isExecuting = true;
        showLoadingPanel();
        
        // Obtener parámetros
        const params = getAlgorithmParameters();
        
        // Validar parámetros
        if (!validateParameters(params)) {
            return;
        }
        
        // Ejecutar algoritmo
        const results = await runGeneticAlgorithmSimulation(params);
        
        // Mostrar resultados
        displayResults(results);
        
    } catch (error) {
        console.error('Error ejecutando algoritmo genético:', error);
        showError('Error al ejecutar el algoritmo genético: ' + error.message);
    } finally {
        isExecuting = false;
        hideLoadingPanel();
    }
}

function getAlgorithmParameters() {
    return {
        startCity: document.getElementById('startCity').value,
        populationSize: parseInt(document.getElementById('populationSize').value),
        generations: parseInt(document.getElementById('generations').value),
        mutationRate: parseFloat(document.getElementById('mutationRate').value)
    };
}

function validateParameters(params) {
    if (params.populationSize < 20 || params.populationSize > 500) {
        showError('El tamaño de población debe estar entre 20 y 500');
        return false;
    }
    
    if (params.generations < 100 || params.generations > 5000) {
        showError('El número de generaciones debe estar entre 100 y 5000');
        return false;
    }
    
    if (params.mutationRate < 0.01 || params.mutationRate > 0.5) {
        showError('La tasa de mutación debe estar entre 0.01 y 0.5');
        return false;
    }
    
    return true;
}

async function runGeneticAlgorithmSimulation(params) {
    updateLoadingStatus('Enviando parámetros al servidor...', 0);
    
    try {
        // Enviar petición al servidor Python
        const response = await fetch('/api/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new Error(`Error del servidor: ${response.status}`);
        }
        
        updateLoadingStatus('Ejecutando algoritmo genético...', 50);
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Error desconocido del servidor');
        }
        
        updateLoadingStatus('Procesando resultados...', 100);
        
        return {
            bestSolution: result.bestSolution,
            bestDistance: result.bestDistance,
            initialDistance: result.initialDistance,
            improvement: result.improvement,
            executionTime: result.executionTime,
            generations: result.generations,
            evolutionData: result.evolutionData || [],
            bestDistances: result.bestDistances,
            avgDistances: result.avgDistances
        };
        
    } catch (error) {
        console.error('Error comunicándose con el servidor:', error);
        throw new Error(`Error de comunicación: ${error.message}`);
    }
}

// =========================
// OPERACIONES DEL ALGORITMO GENÉTICO
// =========================
function initializePopulation(size, startCity) {
    const cities = Object.keys(CONFIG.cities);
    const population = [];
    
    for (let i = 0; i < size; i++) {
        let individual = [...cities];
        
        // Asegurar que comience con la ciudad seleccionada
        const startIndex = individual.indexOf(startCity);
        if (startIndex > 0) {
            [individual[0], individual[startIndex]] = [individual[startIndex], individual[0]];
        }
        
        // Mezclar el resto de ciudades
        for (let j = 1; j < individual.length; j++) {
            const randomIndex = Math.floor(Math.random() * (individual.length - 1)) + 1;
            [individual[j], individual[randomIndex]] = [individual[randomIndex], individual[j]];
        }
        
        population.push(individual);
    }
    
    return population;
}

function evaluatePopulation(population) {
    const evaluated = population.map(individual => ({
        route: individual,
        distance: calculateRouteDistance(individual),
        fitness: 0
    }));
    
    // Calcular fitness (inverso de la distancia)
    evaluated.forEach(individual => {
        individual.fitness = 1 / individual.distance;
    });
    
    // Ordenar por fitness (mejor fitness primero)
    evaluated.sort((a, b) => b.fitness - a.fitness);
    
    return {
        population: evaluated.map(ind => ind.route),
        evaluated: evaluated
    };
}

function getGenerationStats(population) {
    const distances = population.map(calculateRouteDistance);
    const bestDistance = Math.min(...distances);
    const avgDistance = distances.reduce((sum, dist) => sum + dist, 0) / distances.length;
    const worstDistance = Math.max(...distances);
    
    const bestIndex = distances.indexOf(bestDistance);
    const bestSolution = population[bestIndex];
    
    return {
        bestDistance,
        avgDistance,
        worstDistance,
        bestSolution: [...bestSolution]
    };
}

function evolvePopulation(population, mutationRate) {
    const newPopulation = [];
    const eliteCount = Math.floor(population.length * 0.1); // 10% elite
    
    // Agregar élite
    for (let i = 0; i < eliteCount; i++) {
        newPopulation.push([...population[i]]);
    }
    
    // Generar descendencia
    while (newPopulation.length < population.length) {
        const parent1 = selectParent(population);
        const parent2 = selectParent(population);
        
        let child = crossover(parent1, parent2);
        child = mutate(child, mutationRate);
        
        newPopulation.push(child);
    }
    
    return newPopulation;
}

function selectParent(population) {
    // Selección por torneo
    const tournamentSize = 3;
    let best = null;
    let bestDistance = Infinity;
    
    for (let i = 0; i < tournamentSize; i++) {
        const candidate = population[Math.floor(Math.random() * population.length)];
        const distance = calculateRouteDistance(candidate);
        
        if (distance < bestDistance) {
            bestDistance = distance;
            best = candidate;
        }
    }
    
    return [...best];
}

function crossover(parent1, parent2) {
    // Order Crossover (OX)
    const size = parent1.length;
    const child = new Array(size).fill(null);
    
    // Seleccionar segmento aleatorio del padre1
    const start = Math.floor(Math.random() * size);
    const end = Math.floor(Math.random() * (size - start)) + start;
    
    // Copiar segmento del padre1
    for (let i = start; i <= end; i++) {
        child[i] = parent1[i];
    }
    
    // Completar con genes del padre2 en orden
    let parent2Index = 0;
    for (let i = 0; i < size; i++) {
        if (child[i] === null) {
            while (child.includes(parent2[parent2Index])) {
                parent2Index++;
            }
            child[i] = parent2[parent2Index];
            parent2Index++;
        }
    }
    
    return child;
}

function mutate(individual, mutationRate) {
    const mutated = [...individual];
    
    if (Math.random() < mutationRate) {
        // Swap mutation - intercambiar dos ciudades aleatorias (excepto la primera)
        const idx1 = Math.floor(Math.random() * (individual.length - 1)) + 1;
        const idx2 = Math.floor(Math.random() * (individual.length - 1)) + 1;
        
        [mutated[idx1], mutated[idx2]] = [mutated[idx2], mutated[idx1]];
    }
    
    return mutated;
}

function calculateRouteDistance(route) {
    let totalDistance = 0;
    
    for (let i = 0; i < route.length; i++) {
        const city1 = route[i];
        const city2 = route[(i + 1) % route.length]; // Volver al inicio
        
        const distance = getDistanceBetweenCities(city1, city2);
        totalDistance += distance;
    }
    
    return totalDistance;
}

function getDistanceBetweenCities(city1, city2) {
    const key1 = `${city1}-${city2}`;
    const key2 = `${city2}-${city1}`;
    
    return CONFIG.distances[key1] || CONFIG.distances[key2] || 0;
}

// =========================
// INTERFAZ DE USUARIO
// =========================
function showLoadingPanel() {
    document.getElementById('loadingPanel').style.display = 'block';
    document.getElementById('resultsPanel').style.display = 'none';
    document.getElementById('executeBtn').disabled = true;
}

function hideLoadingPanel() {
    document.getElementById('loadingPanel').style.display = 'none';
    document.getElementById('executeBtn').disabled = false;
}

function updateLoadingStatus(message, progress) {
    document.getElementById('loadingStatus').textContent = message;
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressText').textContent = `${Math.round(progress)}%`;
}

function displayResults(results) {
    currentResults = results;
    
    // Mostrar panel de resultados
    document.getElementById('resultsPanel').style.display = 'block';
    
    // Actualizar estadísticas
    document.getElementById('finalDistance').textContent = `${results.bestDistance.toFixed(2)} km`;
    document.getElementById('totalGenerations').textContent = results.generations;
    document.getElementById('executionTime').textContent = `${results.executionTime.toFixed(2)} seg`;
    document.getElementById('improvement').textContent = `${results.improvement.toFixed(1)}%`;
    
    // Mostrar ruta
    displayRoute(results.bestSolution);
    
    // Generar visualizaciones
    generateRouteVisualization(results.bestSolution);
    generateEvolutionChart(results.bestDistances, results.avgDistances);
}

function displayRoute(route) {
    const routeContainer = document.getElementById('routeDisplay');
    routeContainer.innerHTML = '';
    
    route.forEach((city, index) => {
        // Ciudad
        const cityElement = document.createElement('span');
        cityElement.className = 'route-city';
        cityElement.textContent = city;
        routeContainer.appendChild(cityElement);
        
        // Flecha (excepto después de la última ciudad)
        if (index < route.length - 1) {
            const arrowElement = document.createElement('span');
            arrowElement.className = 'route-arrow';
            arrowElement.innerHTML = '→';
            routeContainer.appendChild(arrowElement);
        }
    });
    
    // Flecha de vuelta al inicio
    const returnArrow = document.createElement('span');
    returnArrow.className = 'route-arrow';
    returnArrow.innerHTML = '→';
    routeContainer.appendChild(returnArrow);
    
    const startCity = document.createElement('span');
    startCity.className = 'route-city';
    startCity.style.backgroundColor = '#27ae60';
    startCity.textContent = route[0];
    routeContainer.appendChild(startCity);
}

function generateRouteVisualization(route) {
    const canvas = document.getElementById('routeCanvas');
    const ctx = canvas.getContext('2d');
    
    // Limpiar canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Configuración de dibujo
    const margin = 50;
    const width = canvas.width - 2 * margin;
    const height = canvas.height - 2 * margin;
    
    // Encontrar límites de coordenadas
    const cities = Object.keys(CONFIG.cities);
    const xCoords = cities.map(city => CONFIG.cities[city].x);
    const yCoords = cities.map(city => CONFIG.cities[city].y);
    
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);
    
    // Función para escalar coordenadas
    const scaleX = (x) => margin + ((x - minX) / (maxX - minX)) * width;
    const scaleY = (y) => margin + ((y - minY) / (maxY - minY)) * height;
    
    // Dibujar conexiones de la ruta
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    
    for (let i = 0; i < route.length; i++) {
        const city = route[i];
        const coords = CONFIG.cities[city];
        const x = scaleX(coords.x);
        const y = scaleY(coords.y);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    // Conectar de vuelta al inicio
    const startCoords = CONFIG.cities[route[0]];
    ctx.lineTo(scaleX(startCoords.x), scaleY(startCoords.y));
    ctx.stroke();
    
    // Dibujar ciudades
    cities.forEach((city, index) => {
        const coords = CONFIG.cities[city];
        const x = scaleX(coords.x);
        const y = scaleY(coords.y);
        
        // Círculo de la ciudad
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        
        // Color especial para ciudad de inicio
        if (city === route[0]) {
            ctx.fillStyle = '#27ae60';
        } else {
            ctx.fillStyle = '#e74c3c';
        }
        
        ctx.fill();
        
        // Borde
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Etiqueta de la ciudad
        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(city, x, y - 15);
    });
}

function generateEvolutionChart(bestDistances, avgDistances) {
    const canvas = document.getElementById('evolutionCanvas');
    const ctx = canvas.getContext('2d');
    
    // Destruir gráfico anterior si existe
    if (evolutionChart) {
        evolutionChart.destroy();
    }
    
    // Crear nuevo gráfico
    evolutionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: bestDistances.map((_, index) => index + 1),
            datasets: [{
                label: 'Mejor Distancia',
                data: bestDistances,
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                borderWidth: 2,
                fill: false
            }, {
                label: 'Distancia Promedio',
                data: avgDistances,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Generación'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Distancia (km)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Evolución del Algoritmo Genético'
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

// =========================
// UTILIDADES
// =========================
function downloadResults() {
    if (!currentResults) {
        showError('No hay resultados para descargar');
        return;
    }
    
    const data = {
        timestamp: new Date().toISOString(),
        parameters: getAlgorithmParameters(),
        results: currentResults
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `tsp_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function resetAnalysis() {
    // Limpiar resultados
    currentResults = null;
    
    // Ocultar panel de resultados
    document.getElementById('resultsPanel').style.display = 'none';
    
    // Destruir gráficos
    if (evolutionChart) {
        evolutionChart.destroy();
        evolutionChart = null;
    }
    
    if (routeChart) {
        routeChart.destroy();
        routeChart = null;
    }
    
    // Limpiar canvas
    const routeCanvas = document.getElementById('routeCanvas');
    const routeCtx = routeCanvas.getContext('2d');
    routeCtx.clearRect(0, 0, routeCanvas.width, routeCanvas.height);
    
    console.log('🔄 Análisis reiniciado');
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorModal').style.display = 'block';
}

function closeErrorModal() {
    document.getElementById('errorModal').style.display = 'none';
}

// =========================
// FUNCIONES DE DEPURACIÓN
// =========================
console.log('🚀 Script del Algoritmo Genético TSP cargado');

// Exponer funciones globales para depuración
window.TSP_DEBUG = {
    config: CONFIG,
    calculateDistance: calculateRouteDistance,
    getDistance: getDistanceBetweenCities,
    currentResults: () => currentResults
};
