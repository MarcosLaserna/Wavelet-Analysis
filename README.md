# Wavelet Analysis Research Workspace

Repositorio de trabajo para el analisis de coherencia wavelet, arquetipos y
clustering aplicado a indices de energia, activos verdes y bonos.

## Estructura

```text
Investigacion/
  App.bat                 # Lanzador local de la app en Windows
  App Interactiva/        # Aplicacion Streamlit, tests y scripts
  Datos/                  # Excels y CSV de entrada
  Reuniones/              # PDFs preparados para reuniones
  Referencias/            # Bibliografia y material metodologico
```

## Ejecutar la app

En Windows, desde la carpeta principal, haz doble clic en:

```text
App.bat
```

El lanzador abre `http://localhost:8501`, usa un entorno virtual local si esta
disponible y, si hace falta, instala las dependencias de `requirements.txt`.

Tambien se puede ejecutar manualmente:

```powershell
cd "App Interactiva"
.venv\Scripts\streamlit.exe run app.py
```

## Flujo analitico

La app:

1. Carga los Excel desde `Datos/`.
2. Homogeneiza las fechas a frecuencia mensual.
3. Calcula retornos logaritmicos.
4. Estandariza las series.
5. Calcula la matriz de coherencia wavelet media `adj_R2`.
6. Deriva una distancia `1 - adj_R2`.
7. Visualiza arquetipos, K-Means, matrices, series y exportables.

La matriz de coherencia se calcula automaticamente al arrancar la app con los
datos y parametros disponibles.

## Prueba de interfaz

```powershell
cd "App Interactiva"
.venv\Scripts\python.exe tests\smoke_streamlit_app.py
```

La prueba valida carga de datos, calculo automatico, pestañas principales,
metricas y tablas.

## Presentaciones

Los PDFs de reuniones estan en `Reuniones/`. La presentacion mas reciente es:

```text
Reuniones/Quinta Reunion.pdf
```

Los scripts para regenerar presentaciones estan en `App Interactiva/scripts/`.
