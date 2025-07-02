@echo off
echo.
echo 🧬 ALGORITMO GENETICO TSP - APLICACION WEB
echo ==========================================
echo.
echo Iniciando servidor web...
echo.

REM Verificar que Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado o no está en el PATH
    echo.
    echo Instala Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verificar que los archivos necesarios existen
if not exist "index.html" (
    echo ❌ Error: index.html no encontrado
    pause
    exit /b 1
)

if not exist "EJECUCION_TSP_GA.py" (
    echo ❌ Error: EJECUCION_TSP_GA.py no encontrado
    pause
    exit /b 1
)

REM Iniciar servidor
echo ✓ Verificaciones completadas
echo.
echo 🌐 Iniciando servidor en http://localhost:8000
echo 🌟 La aplicación se abrirá automáticamente en tu navegador
echo.
echo ⏹️  Presiona Ctrl+C para detener el servidor
echo.

python web_server.py

echo.
echo 👋 Servidor detenido
pause
