# Wavelet Archetype Lab (Streamlit)

Aplicación interactiva en **Streamlit** para:

- Calcular **coherencia wavelet** entre activos.
- Construir matriz de distancia `1 - adj_R2`.
- Estimar **arquetipos** (aproximación con NMF).
- Ejecutar **clustering K-Means** con selección automática de `k` por silhouette.
- Visualizar resultados con **Plotly** (ternary, MDS, heatmaps y series temporales).

---

## 1) Requisitos

- Python **3.10+**
- Paquetes listados en `requirements.txt`

Instalación sugerida:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

---

## 2) Estructura esperada de datos

Por defecto la app busca una carpeta `data/` con estos archivos:

- `CSI_New_Energy_index.xlsx`
- `ISE_Clean_Edge_Global_Wind_Energy_index.xlsx`
- `S&P_Global_Clean_energy_index.xlsx`
- `WilderHill.xlsx`
- `DATA revisados Hui.xlsx`
- `eurostat_bonos10.xlsx`

### Hojas relevantes

- `DATA revisados Hui.xlsx`:
  - `SUNIDX`
  - `MVIS GLO URAN PR`
  - `FTSE ENV OPPORT ENE EFF`
  - `ERIXP USD`
- `eurostat_bonos10.xlsx`:
  - `Bond_2001` o `Bond_2015`

> Si cambias la ruta, puedes hacerlo desde el sidebar (“Carpeta con los Excels”).

---

## 3) Ejecución

```bash
streamlit run app.py
```

Luego abre la URL local que muestra Streamlit (normalmente `http://localhost:8501`).

---

## 4) Flujo de la app

1. Carga y homogeneiza series a frecuencia mensual.
2. Calcula retornos logarítmicos.
3. Estandariza datos.
4. Calcula matriz de coherencia wavelet media (`adj_R2`).
5. Genera:
   - Arquetipos (NMF)
   - Clustering K-Means
   - Matrices y series

---

## 5) Consejos de uso

- Selecciona al menos **3 activos**.
- Ajusta escalas wavelet (`min_scale`, `max_scale`, `n_scales`) según granularidad temporal deseada.
- Si el cálculo tarda, reduce número de activos o escalas.
- Si `k` automático no te convence, desactiva “Elegir k automáticamente” y usa `k manual`.

---

## 6) Solución de problemas

### Error: faltan archivos
Verifica que la carpeta de datos tenga los 6 Excels esperados y nombres exactos.

### Dataset vacío
Suele ocurrir cuando las fechas no se solapan entre archivos o por hoja de bonos incorrecta.

### Rendimiento lento
La coherencia wavelet es O(n²) en pares de activos; reduce activos y/o escalas.

---

## 7) Próximas mejoras sugeridas

- Exportación de resultados (CSV/Excel).
- Persistencia de configuraciones de análisis.
- Tests unitarios para funciones de lectura y coherencia.
- Dockerfile para despliegue reproducible.
