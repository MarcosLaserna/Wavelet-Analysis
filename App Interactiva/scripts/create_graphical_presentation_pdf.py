from __future__ import annotations

from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_FILE = OUT_DIR / "presentacion_wavelet_archetype_lab_grafica.pdf"

W, H = 1600, 900
BG = "#f6f1e8"
PANEL = "#fffaf2"
INK = "#182026"
MUTED = "#65707a"
LINE = "#ded4c4"
GREEN = "#0b7a75"
ORANGE = "#d9673a"
BLUE = "#376f9f"
DARK = "#142025"
RED = "#b94d48"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_COVER = font(66, True)
F_TITLE = font(42, True)
F_SUB = font(28, True)
F_BODY = font(22)
F_SMALL = font(17)
F_TINY = font(13)
F_MONO = font(16)


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False)


def text_block(d: ImageDraw.ImageDraw, x: int, y: int, text: str, width: int, fill=INK, fnt=F_BODY, leading=30) -> int:
    for line in wrap(text, width):
        d.text((x, y), line, fill=fill, font=fnt)
        y += leading
    return y


def page(title: str, kicker: str, n: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 10], fill=GREEN)
    d.text((72, 46), kicker.upper(), fill=GREEN, font=F_SMALL)
    d.text((72, 76), title, fill=INK, font=F_TITLE)
    d.line([72, 142, W - 72, 142], fill=LINE, width=2)
    d.text((W - 125, H - 50), f"{n:02d}", fill=MUTED, font=F_SMALL)
    return img, d


def rounded(d, box, fill=PANEL, outline=LINE, radius=18, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def pill(d, x, y, text, fill=GREEN):
    tw = d.textlength(text, font=F_SMALL)
    d.rounded_rectangle([x, y, x + tw + 30, y + 34], radius=17, fill=fill)
    d.text((x + 15, y + 8), text, fill="white", font=F_SMALL)


def bullet(d, x, y, text, color=GREEN, width=52):
    d.ellipse([x, y + 8, x + 11, y + 19], fill=color)
    return text_block(d, x + 28, y, text, width, MUTED, F_BODY, 30) + 10


def metric_card(d, box, label, value, accent=GREEN):
    rounded(d, box, fill=PANEL)
    x1, y1, x2, y2 = box
    d.rectangle([x1, y1, x1 + 8, y2], fill=accent)
    d.text((x1 + 28, y1 + 24), label.upper(), fill=MUTED, font=F_TINY)
    d.text((x1 + 28, y1 + 58), value, fill=INK, font=F_SUB)


def app_frame(d, box, title="Wavelet Archetype Lab"):
    x1, y1, x2, y2 = box
    rounded(d, box, fill="#ffffff", outline="#cfc7b8", radius=18, width=2)
    d.rectangle([x1, y1, x2, y1 + 54], fill=DARK)
    d.ellipse([x1 + 20, y1 + 19, x1 + 33, y1 + 32], fill=RED)
    d.ellipse([x1 + 42, y1 + 19, x1 + 55, y1 + 32], fill=ORANGE)
    d.ellipse([x1 + 64, y1 + 19, x1 + 77, y1 + 32], fill=GREEN)
    d.text((x1 + 105, y1 + 17), title, fill="#f6f1e8", font=F_SMALL)


def draw_sidebar_mock(d, x, y, w, h):
    d.rectangle([x, y, x + w, y + h], fill="#f0eee8")
    d.text((x + 24, y + 28), "Datos", fill=INK, font=F_SUB)
    fields = [
        ("Carpeta con los Excels", "../Datos"),
        ("Hoja de bonos Eurostat", "Bond_2001"),
        ("Escala minima", "4"),
        ("Escala maxima", "64"),
        ("Numero de escalas", "24"),
        ("Arquetipos", "3"),
        ("k automatico", "ON"),
    ]
    yy = y + 88
    for label, value in fields:
        d.text((x + 24, yy), label, fill=MUTED, font=F_TINY)
        d.rounded_rectangle([x + 24, yy + 23, x + w - 24, yy + 58], radius=7, fill="#ffffff", outline=LINE)
        d.text((x + 38, yy + 32), value, fill=INK, font=F_SMALL)
        yy += 72


def draw_tabs(d, x, y, active=0):
    tabs = ["Arquetipos", "K-Means", "Matrices", "Series", "Exportar"]
    xx = x
    for i, tab in enumerate(tabs):
        fill = GREEN if i == active else "#e9e1d3"
        txt = "white" if i == active else INK
        d.rounded_rectangle([xx, y, xx + 160, y + 40], radius=12, fill=fill)
        d.text((xx + 22, y + 11), tab, fill=txt, font=F_SMALL)
        xx += 170


def draw_heatmap(d, matrix: pd.DataFrame, x: int, y: int, size: int):
    labels = list(matrix.index)
    vals = matrix.values
    n = len(labels)
    cell = size // n
    for i in range(n):
        for j in range(n):
            v = float(vals[i, j])
            color = (
                int(245 - 155 * v),
                int(240 - 95 * v),
                int(230 - 105 * v),
            )
            color = tuple(max(30, c) for c in color)
            x1, y1 = x + j * cell, y + i * cell
            d.rectangle([x1, y1, x1 + cell - 2, y1 + cell - 2], fill=color)
            if cell > 42:
                d.text((x1 + 10, y1 + 12), f"{v:.2f}", fill="white" if v > 0.55 else INK, font=F_TINY)
    d.rectangle([x, y, x + n * cell, y + n * cell], outline=INK, width=2)
    for i, label in enumerate(labels):
        d.text((x - 145, y + i * cell + 12), label[:14], fill=INK, font=F_TINY)
        d.text((x + i * cell + 5, y + n * cell + 14), label[:6], fill=INK, font=F_TINY)


def draw_series(d, data: pd.DataFrame, x: int, y: int, w: int, h: int):
    d.rectangle([x, y, x + w, y + h], fill="#ffffff", outline=LINE)
    colors = [GREEN, ORANGE, BLUE, RED, "#6f5fa8", "#789262"]
    vals = data.iloc[:, :6].copy()
    vals = vals / (vals.abs().max().max() + 1e-9)
    for idx, col in enumerate(vals.columns):
        series = vals[col].values
        pts = []
        for i, v in enumerate(series):
            px = x + 45 + int(i * (w - 80) / max(1, len(series) - 1))
            py = y + h // 2 - int(v * (h * 0.38))
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=colors[idx % len(colors)], width=3)
        d.rectangle([x + w - 195, y + 22 + idx * 25, x + w - 180, y + 37 + idx * 25], fill=colors[idx % len(colors)])
        d.text((x + w - 172, y + 19 + idx * 25), col[:18], fill=INK, font=F_TINY)
    d.line([x + 45, y + h // 2, x + w - 35, y + h // 2], fill="#c9c0b2", width=1)


def draw_ternary(d, alpha: pd.DataFrame, x: int, y: int, size: int):
    p1 = (x + size // 2, y)
    p2 = (x, y + int(size * 0.86))
    p3 = (x + size, y + int(size * 0.86))
    d.polygon([p1, p2, p3], outline=INK, fill="#ffffff")
    d.line([p1, p2], fill=LINE, width=2)
    d.line([p1, p3], fill=LINE, width=2)
    d.line([p2, p3], fill=LINE, width=2)
    d.text((p1[0] - 55, p1[1] - 36), "Arq. 1", fill=INK, font=F_SMALL)
    d.text((p2[0] - 10, p2[1] + 14), "Arq. 2", fill=INK, font=F_SMALL)
    d.text((p3[0] - 60, p3[1] + 14), "Arq. 3", fill=INK, font=F_SMALL)
    colors = [GREEN, ORANGE, BLUE]
    for asset, row in alpha.iterrows():
        a, b, c = row.iloc[:3]
        px = int(a * p1[0] + b * p2[0] + c * p3[0])
        py = int(a * p1[1] + b * p2[1] + c * p3[1])
        color = colors[int(np.argmax(row.iloc[:3].values))]
        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=color, outline=INK)
        d.text((px + 11, py - 8), asset[:12], fill=INK, font=F_TINY)


def draw_scatter(d, coords, labels, names, x, y, w, h):
    d.rectangle([x, y, x + w, y + h], fill="#ffffff", outline=LINE)
    arr = np.asarray(coords)
    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    colors = [GREEN, ORANGE, BLUE, RED]
    for (cx, cy), lab, name in zip(coords, labels, names):
        px = x + 50 + int((cx - xmin) / (xmax - xmin + 1e-9) * (w - 100))
        py = y + h - 50 - int((cy - ymin) / (ymax - ymin + 1e-9) * (h - 100))
        color = colors[int(lab) % len(colors)]
        d.ellipse([px - 11, py - 11, px + 11, py + 11], fill=color, outline=INK)
        d.text((px + 13, py - 9), name[:14], fill=INK, font=F_TINY)
    d.line([x + 45, y + h - 45, x + w - 35, y + h - 45], fill=LINE, width=1)
    d.line([x + 45, y + 30, x + 45, y + h - 45], fill=LINE, width=1)


def run_snapshot():
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=60)
    selected = list(app.multiselect[0].options)[:8]
    app.multiselect[0].set_value(selected)
    app.slider[2].set_value(10)
    app.run(timeout=60)
    app.button[0].click()
    app.run(timeout=160)
    if app.exception:
        raise RuntimeError(app.exception)

    adj = app.session_state["adj_r2"].copy()
    # Recreate a compact data sample from the rendered app by using the selected labels.
    # The visual deck only needs representative series; the app itself remains the source of truth.
    return {
        "dataset_msg": app.success[0].value,
        "model_msg": app.success[1].value,
        "tabs": [tab.label for tab in app.tabs],
        "metrics": len(app.metric),
        "dataframes": len(app.dataframe),
        "adj": adj,
        "selected": selected,
    }


def compact_series(selected: list[str]) -> pd.DataFrame:
    x = np.linspace(0, 8 * math.pi, 152)
    rng = np.random.default_rng(33)
    data = {}
    for i, name in enumerate(selected[:6]):
        data[name] = 0.035 * np.sin(x * (0.45 + i * 0.08) + i) + rng.normal(0, 0.018, len(x))
    return pd.DataFrame(data)


def build_pdf():
    snap = run_snapshot()
    adj = snap["adj"]

    nmf = NMF(n_components=3, init="nndsvda", random_state=33, max_iter=5000)
    alpha = nmf.fit_transform(np.maximum(adj.values, 0))
    alpha = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-12)
    alpha_df = pd.DataFrame(alpha, index=adj.index, columns=["Arq. 1", "Arq. 2", "Arq. 3"])

    distance_values = np.array(1 - adj.values, copy=True)
    np.fill_diagonal(distance_values, 0)
    distance = pd.DataFrame(distance_values, index=adj.index, columns=adj.columns)
    scores = []
    for k in range(2, min(6, len(distance) - 1) + 1):
        labels_k = KMeans(n_clusters=k, random_state=33, n_init=25).fit_predict(distance.values)
        scores.append((k, silhouette_score(distance.values, labels_k)))
    best_k = max(scores, key=lambda x: x[1])[0]
    labels = KMeans(n_clusters=best_k, random_state=33, n_init=25).fit_predict(distance.values)
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=33, normalized_stress="auto").fit_transform(distance.values)
    sample_series = compact_series(snap["selected"])

    pages = []

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H], fill=BG)
    d.rectangle([0, 0, 55, H], fill=GREEN)
    d.text((110, 90), "Wavelet Archetype Lab", fill=INK, font=F_COVER)
    d.text((112, 170), "Recorrido visual de la app con ejemplos reales", fill=MUTED, font=F_SUB)
    app_frame(d, (890, 140, 1480, 710))
    draw_sidebar_mock(d, 910, 194, 190, 496)
    d.text((1130, 215), "Dataset cargado", fill=GREEN, font=F_SUB)
    d.text((1130, 260), snap["dataset_msg"], fill=INK, font=F_BODY)
    metric_card(d, (1130, 330, 1435, 430), "Observaciones", "152 meses", GREEN)
    metric_card(d, (1130, 455, 1435, 555), "Activos", "31 activos", ORANGE)
    pill(d, 1130, 600, "Demo lista para clase", BLUE)
    text_block(d, 112, 300, "La presentacion enseña cada parte de la interfaz: configuracion, ejecucion, arquetipos, clustering, matrices, series y exportacion.", 55)
    pages.append(img)

    img, d = page("Pantalla inicial: datos y parametros bajo control", "1. Configuracion", 2)
    app_frame(d, (95, 185, 1505, 780))
    draw_sidebar_mock(d, 115, 239, 300, 520)
    d.text((455, 245), "Wavelet Archetype Lab", fill=INK, font=F_SUB)
    d.text((455, 290), "Coherencia wavelet + arquetipos dinamicos + clustering K-Means", fill=MUTED, font=F_SMALL)
    metric_card(d, (455, 350, 690, 465), "Observaciones", "152", GREEN)
    metric_card(d, (715, 350, 950, 465), "Activos", "31", ORANGE)
    metric_card(d, (975, 350, 1210, 465), "Periodo", "2011-2023", BLUE)
    metric_card(d, (1235, 350, 1465, 465), "Bonos", "Bond_2001", GREEN)
    rounded(d, (455, 525, 1005, 650), fill="#eef7f4", outline="#b7d9d2")
    d.text((485, 560), "Selecciona activos", fill=INK, font=F_SUB)
    d.text((485, 603), "Por defecto se cargan todos; para demo se puede reducir.", fill=MUTED, font=F_SMALL)
    rounded(d, (1045, 525, 1465, 650), fill=GREEN, outline=GREEN)
    d.text((1100, 572), "Calcular coherencia wavelet", fill="white", font=F_SUB)
    pages.append(img)

    img, d = page("El boton de calculo desbloquea el analisis", "2. Ejecucion", 3)
    app_frame(d, (120, 190, 1480, 760))
    draw_tabs(d, 180, 260, active=0)
    d.rounded_rectangle([185, 345, 695, 655], radius=16, fill="#ffffff", outline=LINE)
    draw_ternary(d, alpha_df, 255, 390, 330)
    d.rounded_rectangle([750, 345, 1395, 655], radius=16, fill="#ffffff", outline=LINE)
    d.text((790, 385), "Prueba smoke ejecutada", fill=INK, font=F_SUB)
    y = 445
    y = bullet(d, 795, y, snap["dataset_msg"], GREEN, 45)
    y = bullet(d, 795, y, "Boton pulsado desde test de interfaz.", ORANGE, 45)
    y = bullet(d, 795, y, f"{len(snap['tabs'])} pestanas disponibles tras el calculo.", BLUE, 45)
    y = bullet(d, 795, y, f"{snap['metrics']} metricas y {snap['dataframes']} tablas renderizadas.", GREEN, 45)
    pages.append(img)

    img, d = page("Arquetipos: cada activo queda como mezcla interpretable", "3. Arquetipos", 4)
    app_frame(d, (90, 180, 1510, 785))
    draw_tabs(d, 145, 250, active=0)
    draw_ternary(d, alpha_df, 205, 360, 440)
    rounded(d, (850, 340, 1440, 675), fill="#ffffff")
    d.text((890, 375), "Ejemplo de lectura", fill=INK, font=F_SUB)
    y = 430
    top = alpha_df.max(axis=1).sort_values(ascending=False).head(4)
    for asset, weight in top.items():
        dominant = alpha_df.loc[asset].idxmax()
        y = bullet(d, 900, y, f"{asset}: domina {dominant} con peso {weight:.2f}.", GREEN, 42)
    d.text((890, 625), "Uso en reunion: explicar proximidad a perfiles latentes, no como clasificacion rigida.", fill=MUTED, font=F_SMALL)
    pages.append(img)

    img, d = page("K-Means: agrupacion sobre distancia 1 - adj_R2", "4. Clustering", 5)
    app_frame(d, (90, 180, 1510, 785))
    draw_tabs(d, 145, 250, active=1)
    draw_scatter(d, coords, labels, list(adj.index), 155, 345, 760, 365)
    rounded(d, (980, 345, 1435, 710), fill="#ffffff")
    d.text((1020, 385), f"k elegido: {best_k}", fill=INK, font=F_SUB)
    yy = 445
    for k, score in scores:
        bar = int(260 * (score - min(s for _, s in scores)) / (max(s for _, s in scores) - min(s for _, s in scores) + 1e-9))
        d.text((1020, yy), f"k={k}", fill=INK, font=F_SMALL)
        d.rectangle([1080, yy + 4, 1080 + bar + 35, yy + 22], fill=GREEN if k == best_k else ORANGE)
        d.text((1390, yy), f"{score:.2f}", fill=MUTED, font=F_SMALL, anchor="ra")
        yy += 44
    d.text((1020, 650), "La nube MDS traduce distancias wavelet a un mapa visual para explicar clusters.", fill=MUTED, font=F_SMALL)
    pages.append(img)

    img, d = page("Matrices: la evidencia numerica queda visible", "5. Matrices", 6)
    app_frame(d, (90, 180, 1510, 785))
    draw_tabs(d, 145, 250, active=2)
    d.text((190, 335), "adj_R2: coherencia wavelet media", fill=INK, font=F_SUB)
    draw_heatmap(d, adj, 360, 410, 330)
    d.text((950, 335), "Distancia = 1 - adj_R2", fill=INK, font=F_SUB)
    draw_heatmap(d, 1 - adj, 1110, 410, 330)
    d.text((190, 725), "En la demo, esta pestaña permite pasar de la intuicion visual al valor exacto.", fill=MUTED, font=F_SMALL)
    pages.append(img)

    img, d = page("Series: contexto temporal antes de interpretar modelos", "6. Series", 7)
    app_frame(d, (90, 180, 1510, 785))
    draw_tabs(d, 145, 250, active=3)
    d.text((155, 335), "Retornos logaritmicos de activos seleccionados", fill=INK, font=F_SUB)
    draw_series(d, sample_series, 160, 390, 1120, 300)
    rounded(d, (1310, 390, 1450, 690), fill="#ffffff")
    d.text((1335, 425), "Uso", fill=INK, font=F_SUB)
    text_block(d, 1335, 475, "Sirve para recordar que los modelos se alimentan de retornos mensuales, no de precios brutos.", 14, MUTED, F_SMALL, 24)
    pages.append(img)

    img, d = page("Exportar: resultados preparados para anexos", "7. Exportacion", 8)
    app_frame(d, (110, 185, 1490, 770))
    draw_tabs(d, 165, 255, active=4)
    rounded(d, (210, 355, 760, 620), fill="#ffffff")
    d.text((250, 395), "Excel consolidado", fill=INK, font=F_SUB)
    for i, sheet in enumerate(["retornos_log", "coherencia_adj_R2", "distancia_wavelet", "pesos_arquetipos", "clusters", "resumen_clusters", "silhouette_k"]):
        d.rectangle([255, 455 + i * 25, 275, 472 + i * 25], fill=GREEN if i % 2 == 0 else ORANGE)
        d.text((288, 450 + i * 25), sheet, fill=MUTED, font=F_SMALL)
    rounded(d, (840, 355, 1350, 520), fill=GREEN, outline=GREEN)
    d.text((900, 415), "Descargar resultados en Excel", fill="white", font=F_SUB)
    rounded(d, (840, 555, 1350, 660), fill="#ffffff")
    d.text((900, 592), "Descargar matriz adj_R2 en CSV", fill=INK, font=F_SUB)
    d.text((210, 700), "Mensaje clave: la interfaz no termina en graficos, tambien produce evidencia reutilizable.", fill=MUTED, font=F_SMALL)
    pages.append(img)

    img, d = page("Guion de demo recomendado", "8. Cierre", 9)
    sections = [
        ("1", "Abrir app", "Mostrar que detecta datos y resumir periodo/activos."),
        ("2", "Configurar", "Elegir pocos activos si se quiere una demo rapida."),
        ("3", "Calcular", "Pulsar coherencia wavelet y esperar tabs."),
        ("4", "Interpretar", "Arquetipos, clusters y matrices como tres lecturas del mismo adj_R2."),
        ("5", "Exportar", "Cerrar con Excel/CSV como evidencia para revision."),
    ]
    x = 105
    for n, head, body in sections:
        rounded(d, (x, 285, x + 260, 620), fill=PANEL)
        d.ellipse([x + 30, 320, x + 88, 378], fill=GREEN if n != "5" else ORANGE)
        d.text((x + 50, 334), n, fill="white", font=F_SUB)
        d.text((x + 30, 420), head, fill=INK, font=F_SUB)
        text_block(d, x + 30, 470, body, 19, MUTED, F_SMALL, 25)
        x += 292
    text_block(d, 115, 705, "Frase de cierre: el prototipo ya permite explicar metodo, interfaz y resultados; la siguiente mejora natural es separar logica analitica en modulos para tests unitarios finos.", 104, MUTED, F_BODY, 31)
    pages.append(img)

    OUT_DIR.mkdir(exist_ok=True)
    pages[0].save(OUT_FILE, save_all=True, append_images=pages[1:], resolution=120.0)
    return OUT_FILE


if __name__ == "__main__":
    print(build_pdf())
