from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_FILE = OUT_DIR / "presentacion_wavelet_archetype_lab.pdf"
W, H = 1600, 900

BG = "#f7f4ed"
INK = "#182026"
MUTED = "#68727a"
ACCENT = "#0b7a75"
ACCENT_2 = "#d9673a"
SOFT = "#e8e1d4"
PANEL = "#fffaf1"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(58, True)
F_H1 = font(43, True)
F_H2 = font(30, True)
F_BODY = font(24)
F_SMALL = font(18)
F_TINY = font(14)


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False)


def slide(title: str, kicker: str, page: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 12], fill=ACCENT)
    d.text((86, 54), kicker.upper(), fill=ACCENT, font=F_SMALL)
    d.text((86, 86), title, fill=INK, font=F_H1)
    d.line([86, 154, W - 86, 154], fill=SOFT, width=2)
    d.text((W - 140, H - 54), f"{page:02d}", fill=MUTED, font=F_SMALL)
    return img, d


def paragraph(d: ImageDraw.ImageDraw, x: int, y: int, text: str, width: int = 74, fill: str = INK, leading: int = 35) -> int:
    for line in wrap(text, width):
        d.text((x, y), line, fill=fill, font=F_BODY)
        y += leading
    return y


def bullet(d: ImageDraw.ImageDraw, x: int, y: int, text: str, width: int = 58, fill: str = INK) -> int:
    d.ellipse([x, y + 10, x + 10, y + 20], fill=ACCENT)
    return paragraph(d, x + 28, y, text, width=width, fill=fill, leading=33) + 10


def card(d: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, body: str, accent: str = ACCENT) -> None:
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=10, fill=PANEL, outline=SOFT, width=2)
    d.rectangle([x1, y1, x1 + 8, y2], fill=accent)
    d.text((x1 + 28, y1 + 24), title, fill=INK, font=F_H2)
    paragraph(d, x1 + 28, y1 + 72, body, width=38, fill=MUTED, leading=28)


def run_app_snapshot():
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=60)
    selected_assets = list(app.multiselect[0].options)[:6]
    app.multiselect[0].set_value(selected_assets)
    app.slider[2].set_value(8)
    app.run(timeout=60)
    app.button[0].click()
    app.run(timeout=120)

    if app.exception:
        raise RuntimeError(app.exception)

    adj = app.session_state["adj_r2"].copy()
    return {
        "dataset": app.success[0].value,
        "model": app.success[1].value,
        "tabs": [tab.label for tab in app.tabs],
        "metrics": len(app.metric),
        "dataframes": len(app.dataframe),
        "adj": adj,
    }


def draw_heatmap(d: ImageDraw.ImageDraw, adj, x: int, y: int, size: int) -> None:
    labels = list(adj.index)
    values = adj.values
    n = len(labels)
    cell = size // n

    for i in range(n):
        for j in range(n):
            v = float(values[i, j])
            r = int(245 - 160 * v)
            g = int(245 - 90 * v)
            b = int(245 - 100 * v)
            color = (max(r, 20), max(g, 80), max(b, 80))
            x1 = x + j * cell
            y1 = y + i * cell
            d.rectangle([x1, y1, x1 + cell - 2, y1 + cell - 2], fill=color)
            d.text((x1 + 18, y1 + 20), f"{v:.2f}", fill="white" if v > 0.55 else INK, font=F_SMALL)

    for i, label in enumerate(labels):
        d.text((x - 170, y + i * cell + 18), label[:16], fill=INK, font=F_SMALL)
        d.text((x + i * cell + 10, y + size + 18), label[:8], fill=INK, font=F_TINY)

    d.rectangle([x, y, x + n * cell, y + n * cell], outline=INK, width=2)


def build_pdf() -> Path:
    snap = run_app_snapshot()
    adj = snap["adj"]
    pages: list[Image.Image] = []

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 900], fill=BG)
    d.rectangle([0, 0, 48, H], fill=ACCENT)
    d.text((110, 115), "Wavelet Archetype Lab", fill=INK, font=F_TITLE)
    d.text((112, 190), "Interfaz, pruebas y resultados para la reunion de beca", fill=MUTED, font=F_H2)
    paragraph(
        d,
        112,
        285,
        "La app convierte series financieras en una lectura relacional: coherencia wavelet media, distancia 1 - adj_R2, arquetipos aproximados y clustering K-Means.",
        width=70,
    )
    card(d, (1040, 230, 1450, 390), "Dataset real", snap["dataset"], ACCENT)
    card(d, (1040, 420, 1450, 580), "Prueba de modelo", snap["model"], ACCENT_2)
    d.text((112, 770), "Preparado para demo academica y discusion metodologica", fill=MUTED, font=F_SMALL)
    pages.append(img)

    img, d = slide("La app responde a una pregunta metodologica concreta", "Tesis", 2)
    y = 210
    y = bullet(d, 110, y, "Medir relaciones entre activos cuando la dependencia puede cambiar por escala temporal, no solo por correlacion lineal.")
    y = bullet(d, 110, y, "Transformar coherencia wavelet en una matriz interpretable para distancias, arquetipos y clusters.")
    y = bullet(d, 110, y, "Entregar una interfaz defendible: datos trazables, parametros visibles, resultados exportables y prueba smoke reproducible.")
    card(d, (990, 260, 1455, 560), "Mensaje para el profesor", "No es solo una visualizacion: es un prototipo analitico que conecta lectura financiera, metodo cuantitativo y validacion de interfaz.", ACCENT)
    pages.append(img)

    img, d = slide("Flujo completo: de Excel a evidencias visuales", "Workflow", 3)
    steps = [
        ("1", "Carga", "Excel financieros y bonos Eurostat"),
        ("2", "Normalizacion", "Frecuencia mensual y retornos logaritmicos"),
        ("3", "Wavelet", "Coherencia media adj_R2 por pares"),
        ("4", "Modelos", "NMF como proxy de arquetipos y K-Means"),
        ("5", "Salida", "Tabs, matrices, series y exportacion"),
    ]
    x = 105
    for i, name, body in steps:
        d.rounded_rectangle([x, 285, x + 245, 585], radius=12, fill=PANEL, outline=SOFT, width=2)
        d.ellipse([x + 26, 315, x + 82, 371], fill=ACCENT)
        d.text((x + 45, 327), i, fill="white", font=F_H2)
        d.text((x + 26, 402), name, fill=INK, font=F_H2)
        paragraph(d, x + 26, 452, body, width=19, fill=MUTED, leading=28)
        if i != "5":
            d.line([x + 258, 435, x + 298, 435], fill=ACCENT_2, width=4)
        x += 290
    pages.append(img)

    img, d = slide("La interfaz queda preparada para una demo guiada", "Interfaz", 4)
    card(d, (105, 230, 550, 410), "Sidebar", "Ruta de datos autodetectada, hoja de bonos, escalas wavelet, arquetipos y seleccion de k.", ACCENT)
    card(d, (600, 230, 1045, 410), "Resumen", "Cuatro metricas visibles: observaciones, activos, periodo y hoja Eurostat usada.", ACCENT)
    card(d, (1095, 230, 1490, 410), "Accion", "El boton calcula la coherencia wavelet y desbloquea el analisis.", ACCENT_2)
    card(d, (105, 500, 550, 690), "Tabs", "Arquetipos, K-Means, matrices, series y exportacion quedan separados para explicar paso a paso.", ACCENT)
    card(d, (600, 500, 1045, 690), "Exportables", "Excel con retornos, adj_R2, distancias, pesos, clusters y silhouette.", ACCENT_2)
    card(d, (1095, 500, 1490, 690), "Defensa", "La configuracion se muestra en pantalla para que el resultado sea reproducible.", ACCENT)
    pages.append(img)

    img, d = slide("Prueba smoke ejecutada sobre datos reales", "Validacion", 5)
    y = 220
    y = bullet(d, 115, y, f"Carga inicial sin excepciones: {snap['dataset']}.")
    y = bullet(d, 115, y, "La prueba selecciona 6 activos y reduce escalas a 8 para validar rapido el flujo de interfaz.")
    y = bullet(d, 115, y, "El click en 'Calcular coherencia wavelet' genera adj_R2 y desbloquea las cinco pestañas.")
    y = bullet(d, 115, y, f"Elementos renderizados tras el calculo: {snap['metrics']} metricas y {snap['dataframes']} tablas.")
    d.rounded_rectangle([960, 245, 1455, 610], radius=14, fill="#172126")
    d.text((1005, 292), "Comando de prueba", fill="white", font=F_H2)
    code = ".venv\\Scripts\\python.exe tests\\smoke_streamlit_app.py"
    d.text((1005, 365), code, fill="#e9f2ee", font=F_SMALL)
    d.text((1005, 435), "Salida esperada:", fill="#9ad5c9", font=F_SMALL)
    d.text((1005, 480), "OK - Streamlit smoke test passed", fill="#e9f2ee", font=F_SMALL)
    d.text((1005, 520), snap["model"], fill="#e9f2ee", font=F_SMALL)
    pages.append(img)

    img, d = slide("Ejemplo de resultado: coherencia wavelet media", "Resultado", 6)
    draw_heatmap(d, adj, 420, 250, 420)
    mean_offdiag = (adj.values[np.triu_indices_from(adj.values, k=1)]).mean()
    max_pair = None
    max_value = -1.0
    for i, a in enumerate(adj.index):
        for j, b in enumerate(adj.columns):
            if j <= i:
                continue
            if adj.iloc[i, j] > max_value:
                max_value = float(adj.iloc[i, j])
                max_pair = (a, b)
    card(d, (955, 260, 1465, 420), "Lectura", f"La coherencia media fuera de la diagonal es {mean_offdiag:.2f} en la prueba reducida.", ACCENT)
    card(d, (955, 455, 1465, 620), "Par mas coherente", f"{max_pair[0]} / {max_pair[1]} alcanza {max_value:.2f}.", ACCENT_2)
    d.text((420, 710), "Matriz adj_R2 calculada desde la app con 6 activos y 8 escalas.", fill=MUTED, font=F_SMALL)
    pages.append(img)

    img, d = slide("Mejoras aplicadas antes de la presentacion", "Cambios", 7)
    y = 230
    for item in [
        "Autodeteccion de carpeta de datos: busca data/ dentro del repo y Datos/ en la carpeta padre.",
        "Resumen ejecutivo arriba de la app: observaciones, activos, periodo y hoja de bonos.",
        "Panel de configuracion visible para explicar parametros de la corrida.",
        "Nueva pestaña Exportar con Excel y CSV para respaldar los resultados.",
        "Prueba smoke versionada para repetir la validacion antes de la reunion.",
        "Sustitucion de llamadas Streamlit deprecadas por width='stretch'.",
    ]:
        y = bullet(d, 120, y, item, width=78)
    pages.append(img)

    img, d = slide("Plan de defensa para pasado mañana", "Cierre", 8)
    card(d, (115, 235, 555, 610), "1. Abrir", "Mostrar que los datos cargan y explicar el periodo mensual cubierto.", ACCENT)
    card(d, (605, 235, 1045, 610), "2. Calcular", "Ejecutar coherencia wavelet con pocos activos para demo rapida y despues comentar escala completa.", ACCENT_2)
    card(d, (1095, 235, 1485, 610), "3. Defender", "Pasar por arquetipos, clusters, matrices y exportacion como evidencias del prototipo.", ACCENT)
    paragraph(d, 120, 700, "Siguiente mejora razonable: separar la logica analitica en un modulo testable y dejar app.py solo como capa de interfaz.", width=90, fill=MUTED)
    pages.append(img)

    OUT_DIR.mkdir(exist_ok=True)
    pages[0].save(OUT_FILE, save_all=True, append_images=pages[1:], resolution=120.0)
    return OUT_FILE


if __name__ == "__main__":
    print(build_pdf())
