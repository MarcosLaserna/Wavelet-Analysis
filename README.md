# Wavelet Analysis

Aplicacion Streamlit para analisis financiero con coherencia wavelet,
arquetipoides, biarquetipos, clustering, redes de activos y exportacion de
graficos/resultados.

## Enlaces

- Repositorio GitHub: https://github.com/MarcosLaserna/Wavelet-Analysis
- App desplegada en Streamlit Cloud: https://wavelet-analysis.streamlit.app/

## Estructura principal

```text
Wavelet-Analysis/
  streamlit_app.py        # Punto de entrada para Streamlit Cloud
  requirements.txt        # Dependencias Python
  runtime.txt             # Version de Python recomendada en Cloud
  App.bat                 # Lanzador local en Windows
  App Interactiva/
    app.py                # Aplicacion principal
    tests/                # Smoke test de la interfaz
  Datos/
    datos_final.xlsx      # Dataset principal usado por la app
    README.md             # Descripcion de datos historicos y complementarios
  Reuniones/              # Material de reuniones
  Referencias/            # Bibliografia y material metodologico
```

## Ejecutar en local

En Windows, desde la carpeta principal, se puede abrir la app con doble clic en:

```text
App.bat
```

O manualmente:

```powershell
cd "App Interactiva"
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

La app se abre en:

```text
http://localhost:8501
```

## Desplegar en Streamlit Cloud

1. Entrar en https://share.streamlit.io/
2. Elegir **New app**.
3. Seleccionar el repositorio:

```text
MarcosLaserna/Wavelet-Analysis
```

4. Usar estos parametros:

```text
Branch: main
Main file path: streamlit_app.py
Python version: definida en runtime.txt
```

5. Pulsar **Deploy**.

Importante: el archivo principal para Cloud es `streamlit_app.py`, no
`App Interactiva/app.py`. Esto evita problemas con espacios en la ruta.

## Flujo analitico

La aplicacion:

1. Carga y combina los datasets disponibles en `Datos/`.
2. Usa `datos_final.xlsx` como base principal y conserva los datos historicos
   y complementarios del proyecto.
3. Homogeneiza precios a frecuencia mensual.
4. Calcula retornos logaritmicos.
5. Calcula la coherencia wavelet media entre pares de activos.
6. Construye la matriz `adj_R2`.
7. Deriva la distancia `1 - adj_R2`.
8. Proyecta la distancia a un h-plot bidimensional mediante PCoA.
9. Calcula arquetipoides sobre las coordenadas del h-plot.
10. Calcula biarquetipos sobre la matriz activo-tiempo de retornos.
11. Genera clustering, matrices, red de activos, series y exportables.

La carpeta `Datos/` conserva tambien los datasets historicos y complementarios
utilizados durante el desarrollo del proyecto.

## Prueba rapida

```powershell
cd "App Interactiva"
.venv\Scripts\python.exe tests\smoke_streamlit_app.py
```

La prueba comprueba que la app carga los datos, calcula resultados iniciales y
renderiza las vistas principales sin errores.
