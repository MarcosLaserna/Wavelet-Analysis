@echo off
setlocal EnableExtensions

title Wavelet Archetype Lab

set "ROOT=%~dp0"
set "APP_DIR=%ROOT%App Interactiva"
set "DATA_DIR=%ROOT%Datos"
set "VENV_DIR=%APP_DIR%\.venv"
set "CHECK_ONLY="

if /I "%~1"=="--check" set "CHECK_ONLY=1"

echo.
echo ==========================================
echo   Wavelet Archetype Lab - Lanzador local
echo ==========================================
echo.

if not exist "%APP_DIR%\app.py" (
    echo ERROR: No encuentro la app en:
    echo "%APP_DIR%"
    echo.
    pause
    exit /b 1
)

if not exist "%DATA_DIR%" (
    echo AVISO: No encuentro la carpeta de datos:
    echo "%DATA_DIR%"
    echo.
    echo La app puede abrirse, pero necesitara que indiques la carpeta correcta.
    echo.
)

cd /d "%APP_DIR%"

if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo El entorno .venv existente no funciona en este ordenador.
        echo Creare un entorno nuevo en .venv_app.
        set "VENV_DIR=%APP_DIR%\.venv_app"
    )
) else (
    set "VENV_DIR=%APP_DIR%\.venv_app"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creando entorno virtual local...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo ERROR: No encuentro Python instalado.
            echo Instala Python 3.10 o superior desde https://www.python.org/downloads/
            echo y vuelve a ejecutar este archivo.
            echo.
            pause
            exit /b 1
        )
        python -m venv "%VENV_DIR%"
    )
)

"%VENV_DIR%\Scripts\python.exe" -c "import streamlit, pandas, pywt, plotly, sklearn, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Instalando dependencias de la app...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 (
        echo.
        echo ERROR: No se han podido instalar las dependencias.
        echo Revisa la conexion a internet o ejecuta manualmente:
        echo "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%ROOT%requirements.txt"
        echo.
        pause
        exit /b 1
    )
)

if defined CHECK_ONLY (
    echo OK: lanzador preparado.
    echo App: "%APP_DIR%"
    echo Datos: "%DATA_DIR%"
    echo Python: "%VENV_DIR%\Scripts\python.exe"
    exit /b 0
)

echo Abriendo la app en http://localhost:8501
echo.
start "" "http://localhost:8501"
"%VENV_DIR%\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true

echo.
echo La app se ha cerrado.
pause
