# 🌐 APLICACIÓN WEB - ALGORITMO GENÉTICO TSP

## 📋 Descripción

Aplicación web interactiva que permite ejecutar y visualizar el Algoritmo Genético para el Problema del Vendedor Viajero (TSP) con las ciudades de Lambayeque, Perú.

## 🚀 Características

### ✨ **Interface Web Moderna**
- **Diseño responsivo** adaptable a diferentes dispositivos
- **Animaciones CSS** suaves y profesionales
- **Tema moderno** con gradientes y efectos de cristal
- **Iconos FontAwesome** para mejor experiencia visual

### ⚙️ **Configuración Interactiva**
- **Selección de ciudad de inicio** mediante dropdown
- **Parámetros ajustables**:
  - Tamaño de población (20-500)
  - Número de generaciones (100-5000)
  - Tasa de mutación (0.01-0.5)

### 📊 **Visualización en Tiempo Real**
- **Progreso animado** con barra de carga y spinner DNA
- **Mapa de ruta** dibujado en canvas con ciudades y conexiones
- **Gráfico de evolución** mostrando convergencia del algoritmo
- **Estadísticas detalladas** de rendimiento

### 💾 **Exportación de Resultados**
- **Descarga JSON** con todos los datos del análisis
- **Resultados completos** incluyendo parámetros y métricas

## 📁 Archivos de la Aplicación

```
AG_TRABAJO/
├── index.html          # 🎨 Interfaz principal de la aplicación
├── styles.css          # 💅 Estilos CSS modernos y responsivos
├── script.js           # ⚡ Lógica JavaScript del frontend
├── web_server.py       # 🐍 Servidor Python con API REST
└── README_WEB.md       # 📖 Esta documentación
```

## 🛠️ Instalación y Uso

### 1️⃣ **Prerrequisitos**
```bash
# Verificar que tienes Python 3.10+
python --version

# Verificar dependencias del proyecto TSP
pip install -r requirements.txt
```

### 2️⃣ **Iniciar la Aplicación**
```bash
# Método 1: Servidor con navegador automático
python web_server.py

# Método 2: Servidor sin abrir navegador
python web_server.py --no-browser

# Método 3: Puerto personalizado
python web_server.py --port 8080
```

### 3️⃣ **Acceder a la Aplicación**
- Abre tu navegador en: `http://localhost:8000`
- La aplicación se abrirá automáticamente si no usas `--no-browser`

## 🎯 Cómo Usar la Aplicación

### **Paso 1: Configurar Parámetros**
1. **Selecciona la ciudad de inicio** del dropdown
2. **Ajusta el tamaño de población** (recomendado: 100-200)
3. **Define el número de generaciones** (recomendado: 1000-2000)
4. **Configura la tasa de mutación** con el slider (recomendado: 0.08-0.15)

### **Paso 2: Ejecutar Algoritmo**
1. Haz clic en **"Ejecutar Algoritmo Genético"**
2. Observa la **animación de progreso** con:
   - Spinner DNA giratorio
   - Barra de progreso
   - Estado actual de la ejecución

### **Paso 3: Analizar Resultados**
1. **Estadísticas**:
   - Distancia final óptima
   - Tiempo de ejecución
   - Porcentaje de mejora vs solución inicial

2. **Visualización de la ruta**:
   - Mapa 2D con ciudades marcadas
   - Conexiones de la ruta óptima
   - Ciudad de inicio destacada en verde

3. **Gráfico de evolución**:
   - Convergencia del mejor resultado
   - Evolución de la distancia promedio

### **Paso 4: Exportar Resultados**
1. **Descargar resultados** en formato JSON
2. **Ejecutar nuevo análisis** para comparar diferentes configuraciones

## 🏗️ Arquitectura Técnica

### **Frontend (Cliente Web)**
```javascript
// Tecnologías utilizadas:
- HTML5 semántico
- CSS3 con variables personalizadas
- JavaScript ES6+ asíncrono
- Chart.js para gráficos
- Canvas API para visualización de mapas
```

### **Backend (Servidor Python)**
```python
# Servidor HTTP personalizado:
- HTTPServer de Python estándar
- API REST en /api/execute
- Integración con EJECUCION_TSP_GA.py
- Respuestas JSON con CORS habilitado
```

### **Comunicación Cliente-Servidor**
```
Cliente Web ←→ HTTP/JSON ←→ Servidor Python ←→ Algoritmo Genético
```

## ⚡ API del Servidor

### **Endpoint**: `POST /api/execute`

**Request Body**:
```json
{
    "startCity": "Chiclayo",
    "populationSize": 100,
    "generations": 1000,
    "mutationRate": 0.1
}
```

**Response Success**:
```json
{
    "success": true,
    "bestSolution": ["Chiclayo", "Lambayeque", "..."],
    "bestDistance": 285.5,
    "improvement": 23.4,
    "executionTime": 12.3,
    "bestDistances": [300, 295, 290, "..."],
    "avgDistances": [450, 420, 390, "..."]
}
```

**Response Error**:
```json
{
    "success": false,
    "error": "Descripción del error"
}
```

## 🎨 Personalización

### **Cambiar Colores del Tema**
Edita las variables CSS en `styles.css`:
```css
:root {
    --primary-color: #2c3e50;     /* Color principal */
    --secondary-color: #3498db;   /* Color secundario */
    --success-color: #27ae60;     /* Color de éxito */
    /* ... más variables */
}
```

### **Agregar Nuevas Ciudades**
Actualiza la configuración en `script.js`:
```javascript
const CONFIG = {
    cities: {
        'NuevaCiudad': { x: 10, y: 8 },
        // ... ciudades existentes
    },
    distances: {
        'Chiclayo-NuevaCiudad': 45,
        // ... distancias existentes
    }
};
```

### **Modificar Parámetros por Defecto**
Cambia los valores en `index.html`:
```html
<input type="number" id="populationSize" value="150" min="20" max="500">
<input type="number" id="generations" value="1500" min="100" max="5000">
```

## 🔧 Solución de Problemas

### **El servidor no inicia**
```bash
# Verificar que el puerto no esté ocupado
netstat -an | findstr :8000

# Usar puerto diferente
python web_server.py --port 8080
```

### **Error de importación de módulos**
```bash
# Verificar que EJECUCION_TSP_GA.py existe
ls EJECUCION_TSP_GA.py

# Reinstalar dependencias
pip install -r requirements.txt
```

### **La visualización no aparece**
1. Abre las **herramientas de desarrollador** (F12)
2. Verifica errores en la **consola**
3. Comprueba que `Chart.js` se carga correctamente

### **El algoritmo no se ejecuta**
1. Verifica la **conexión al servidor** en Network tab
2. Revisa los **parámetros** estén en rangos válidos
3. Consulta la **consola del servidor** para errores Python

## 📈 Optimizaciones Futuras

### **Performance**
- [ ] Implementar WebWorkers para cálculos pesados
- [ ] Cachear resultados frecuentes
- [ ] Optimizar renderizado de gráficos

### **Funcionalidades**
- [ ] Guardado de configuraciones favoritas
- [ ] Comparación de múltiples ejecuciones
- [ ] Exportación de gráficos como imágenes
- [ ] Animación de la evolución de rutas

### **UI/UX**
- [ ] Tema oscuro/claro
- [ ] Tutorial interactivo
- [ ] Tooltips explicativos
- [ ] Modo pantalla completa para visualizaciones

## 🤝 Contribución

Para contribuir al proyecto:

1. **Fork** el repositorio
2. **Crea** una rama para tu feature
3. **Implementa** mejoras o correcciones
4. **Testea** la aplicación web
5. **Envía** un pull request

## 📄 Licencia

Este proyecto es de código abierto para fines educativos y de investigación.

---

**Desarrollado para el análisis y visualización interactiva del Algoritmo Genético aplicado al TSP**

🌐 **URL de la aplicación**: `http://localhost:8000`  
🐍 **Servidor**: `python web_server.py`  
📊 **Tecnologías**: HTML5, CSS3, JavaScript ES6+, Python 3.10+
