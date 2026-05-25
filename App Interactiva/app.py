import warnings
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")


EXPECTED_DATA_FILES = (
    "CSI_New_Energy_index.xlsx",
    "ISE_Clean_Edge_Global_Wind_Energy_index.xlsx",
    "S&P_Global_Clean_energy_index.xlsx",
    "WilderHill.xlsx",
    "DATA revisados Hui.xlsx",
    "eurostat_bonos10.xlsx",
)


def has_expected_data_files(path: Path) -> bool:
    return all((path / file_name).exists() for file_name in EXPECTED_DATA_FILES)


def default_data_dir() -> str:
    candidates = [
        Path("data"),
        Path.cwd() / "data",
        Path("Datos"),
        Path.cwd() / "Datos",
        Path.cwd().parent / "Datos",
        Path.cwd().parent,
    ]

    for candidate in candidates:
        if has_expected_data_files(candidate):
            return str(candidate)

    return "data"


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31])

    return output.getvalue()


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Wavelet Archetype Lab",
    layout="wide"
)

st.title("Wavelet Archetype Lab")
st.caption("Coherencia wavelet + arquetipos dinámicos + clustering K-Means")


# ============================================================
# UTILIDADES TEMPORALES
# ============================================================

def to_month_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte cualquier índice de fechas a inicio de mes.
    Evita el problema de mezclar fin de mes con inicio de mes.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index = df.index.to_period("M").to_timestamp()
    return df.sort_index()


# ============================================================
# LECTURA DE EXCELS
# ============================================================

def find_header_row_excel(path, sheet_name=0, keywords=("Exchange Date", "Date")):
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    for i in range(min(len(raw), 100)):
        row_values = raw.iloc[i].astype(str).tolist()
        if any(k in row_values for k in keywords):
            return i

    return 0


def read_market_index_excel(path, label, sheet_name=0):
    """
    Lee índices verdes:
    - CSI_New_Energy_index.xlsx
    - ISE_Clean_Edge_Global_Wind_Energy_index.xlsx
    - S&P_Global_Clean_energy_index.xlsx
    - WilderHill.xlsx

    Usa:
    - Exchange Date
    - Close
    """
    header_row = find_header_row_excel(
        path,
        sheet_name=sheet_name,
        keywords=("Exchange Date",)
    )

    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)

    if "Exchange Date" not in df.columns:
        raise ValueError(f"No encuentro 'Exchange Date' en {path}")

    price_col = "Close" if "Close" in df.columns else df.columns[1]

    out = df[["Exchange Date", price_col]].copy()
    out.columns = ["Date", label]

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[label] = pd.to_numeric(out[label], errors="coerce")

    out = out.dropna().sort_values("Date").set_index("Date")

    monthly = out.resample("ME").last().dropna()
    monthly = to_month_index(monthly)

    returns = np.log(monthly / monthly.shift(1)).dropna()
    returns = to_month_index(returns)

    return returns


def read_hui_sheet(path, sheet_name, label):
    """
    Lee hojas concretas de DATA revisados Hui.xlsx:
    - SUNIDX
    - MVIS GLO URAN PR
    - FTSE ENV OPPORT ENE EFF
    - ERIXP USD
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)

    date_col = df.columns[0]
    price_col = df.columns[1]

    out = df[[date_col, price_col]].copy()
    out.columns = ["Date", label]

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[label] = pd.to_numeric(out[label], errors="coerce")

    out = out.dropna().sort_values("Date").set_index("Date")

    monthly = out.resample("ME").last().dropna()
    monthly = to_month_index(monthly)

    returns = np.log(monthly / monthly.shift(1)).dropna()
    returns = to_month_index(returns)

    return returns


def read_eurostat_bonds(path, sheet_name="Bond_2001", drop_cols=("JP", "US", "TR")):
    """
    Lee eurostat_bonos10.xlsx.

    Mantiene tu lógica de R:
        gross = 1 + yield / 100
        returns = diff(log(gross))
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    if "Time" not in df.columns:
        raise ValueError(f"No encuentro la columna 'Time' en {path}")

    df["Time"] = pd.to_datetime(df["Time"].astype(str), errors="coerce")
    df = df.dropna(subset=["Time"]).set_index("Time")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = to_month_index(df)

    gross = 1 + df / 100.0

    returns = np.log(gross / gross.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    returns = returns.dropna(axis=1, how="any")
    returns = to_month_index(returns)

    return returns


@st.cache_data
def build_dataset(data_dir: str, bond_sheet: str):
    data_dir = Path(data_dir)

    files = {
        "CSI_China": data_dir / "CSI_New_Energy_index.xlsx",
        "Wind": data_dir / "ISE_Clean_Edge_Global_Wind_Energy_index.xlsx",
        "SP_Clean": data_dir / "S&P_Global_Clean_energy_index.xlsx",
        "WilderHill": data_dir / "WilderHill.xlsx",
        "Hui": data_dir / "DATA revisados Hui.xlsx",
        "Bonds": data_dir / "eurostat_bonos10.xlsx",
    }

    missing = [str(v) for v in files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan estos archivos en la carpeta indicada:\n" + "\n".join(missing)
        )

    series = []

    series.append(read_market_index_excel(files["CSI_China"], "CSI_China"))
    series.append(read_market_index_excel(files["Wind"], "Wind"))
    series.append(read_market_index_excel(files["SP_Clean"], "SP_Clean"))
    series.append(read_market_index_excel(files["WilderHill"], "WilderHill"))

    series.append(read_hui_sheet(files["Hui"], "SUNIDX", "Solar"))
    series.append(read_hui_sheet(files["Hui"], "MVIS GLO URAN PR", "Uranium"))
    series.append(read_hui_sheet(files["Hui"], "FTSE ENV OPPORT ENE EFF", "FTSE_Env"))
    series.append(read_hui_sheet(files["Hui"], "ERIXP USD", "ERIXP_EU"))

    bonds = read_eurostat_bonds(files["Bonds"], sheet_name=bond_sheet)
    series.append(bonds)

    dataset = pd.concat(series, axis=1, join="inner")
    dataset = dataset.sort_index()
    dataset = dataset.dropna(axis=0, how="any")
    dataset = dataset.loc[:, dataset.std() > 0]

    return dataset


# ============================================================
# COHERENCIA WAVELET
# ============================================================

def wavelet_coherence_pair(
    x,
    y,
    scales,
    wavelet="cmor1.5-1.0",
    smooth_size=(3, 7)
):
    """
    Coherencia wavelet cuadrática aproximada:

        R²(s,t) = |S(Wx * conj(Wy))|²
                  ---------------------
                  S(|Wx|²) S(|Wy|²)

    donde S es un suavizado local escala-tiempo.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)

    coef_x, _ = pywt.cwt(x, scales, wavelet)
    coef_y, _ = pywt.cwt(y, scales, wavelet)

    wxy = coef_x * np.conj(coef_y)

    s_wxy = (
        uniform_filter(wxy.real, smooth_size)
        + 1j * uniform_filter(wxy.imag, smooth_size)
    )

    s_xx = uniform_filter(np.abs(coef_x) ** 2, smooth_size)
    s_yy = uniform_filter(np.abs(coef_y) ** 2, smooth_size)

    r2 = (np.abs(s_wxy) ** 2) / (s_xx * s_yy + 1e-12)
    r2 = np.clip(r2, 0, 1)

    return r2


@st.cache_data
def compute_wavelet_coherence_matrix(
    data: pd.DataFrame,
    min_scale: int,
    max_scale: int,
    n_scales: int
):
    """
    Construye adj_R2:
        adj_R2[i, j] = media de la coherencia wavelet R² entre activos i y j.
    """
    cols = list(data.columns)
    n = len(cols)

    scales = np.geomspace(min_scale, max_scale, n_scales)
    mat = np.zeros((n, n), dtype=float)

    values = data.values

    total = n * (n - 1) // 2
    counter = 0
    progress = st.progress(0)

    for i in range(n):
        for j in range(i + 1, n):
            r2 = wavelet_coherence_pair(values[:, i], values[:, j], scales)
            mat[i, j] = np.nanmean(r2)
            mat[j, i] = mat[i, j]

            counter += 1
            progress.progress(counter / total)

    np.fill_diagonal(mat, 1.0)

    return pd.DataFrame(mat, index=cols, columns=cols)


def compute_distance_matrix(adj_r2: pd.DataFrame):
    dist = (1 - adj_r2).copy()
    values = dist.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)

    return pd.DataFrame(
        values,
        index=adj_r2.index,
        columns=adj_r2.columns
    )


# ============================================================
# RED DE INTERDEPENDENCIA
# ============================================================

def classify_asset_sector(asset_name: str) -> str:
    green_assets = {
        "CSI_China",
        "Wind",
        "SP_Clean",
        "WilderHill",
        "Solar",
        "Uranium",
        "FTSE_Env",
        "ERIXP_EU",
    }
    return "Green Assets" if asset_name in green_assets else "Conventional"


def build_network_edges(adj_r2: pd.DataFrame, threshold: float):
    rows = []
    assets = list(adj_r2.index)

    for i in range(len(assets) - 1):
        for j in range(i + 1, len(assets)):
            weight = float(adj_r2.iloc[i, j])
            if not np.isnan(weight) and weight > threshold:
                rows.append({
                    "from": assets[i],
                    "to": assets[j],
                    "weight": weight,
                })

    return pd.DataFrame(rows, columns=["from", "to", "weight"])


def weighted_degree(edges_df: pd.DataFrame, assets: list[str]):
    degree = pd.Series(0.0, index=assets)

    for _, edge in edges_df.iterrows():
        degree.loc[edge["from"]] += edge["weight"]
        degree.loc[edge["to"]] += edge["weight"]

    return degree.replace(0, 0.5)


def force_layout(assets: list[str], edges_df: pd.DataFrame, seed: int = 555):
    """
    Layout de fuerzas reproducible inspirado en Fruchterman-Reingold.
    Evita añadir networkx como dependencia solo para esta visualizacion.
    """
    rng = np.random.default_rng(seed)
    n = len(assets)

    if n == 1:
        return {assets[0]: np.array([0.0, 0.0])}

    positions = rng.uniform(-1.0, 1.0, size=(n, 2))
    index = {asset: idx for idx, asset in enumerate(assets)}
    area = 4.0
    k = np.sqrt(area / n)
    temperature = 0.45

    edge_pairs = [
        (index[row["from"]], index[row["to"]], float(row["weight"]))
        for _, row in edges_df.iterrows()
        if row["from"] in index and row["to"] in index
    ]

    for _ in range(450):
        disp = np.zeros_like(positions)

        for i in range(n):
            delta = positions[i] - positions
            distance = np.linalg.norm(delta, axis=1) + 1e-9
            force = (k * k / distance)[:, None] * delta / distance[:, None]
            force[i] = 0
            disp[i] += force.sum(axis=0)

        for source, target, weight in edge_pairs:
            delta = positions[source] - positions[target]
            distance = np.linalg.norm(delta) + 1e-9
            force = (distance * distance / k) * (0.45 + weight)
            vector = (delta / distance) * force
            disp[source] -= vector
            disp[target] += vector

        lengths = np.linalg.norm(disp, axis=1) + 1e-9
        positions += (disp / lengths[:, None]) * np.minimum(lengths, temperature)[:, None]
        positions -= positions.mean(axis=0)
        temperature *= 0.992

    max_abs = np.abs(positions).max()
    if max_abs > 0:
        positions = positions / max_abs

    return {asset: positions[index[asset]] for asset in assets}


def plot_wavelet_network(
    adj_r2: pd.DataFrame,
    threshold: float = 0.42,
    max_edges: int = 80,
):
    assets = list(adj_r2.index)
    all_edges_df = build_network_edges(adj_r2, threshold)
    edges_df = (
        all_edges_df.sort_values("weight", ascending=False)
        .head(max_edges)
        .reset_index(drop=True)
    )
    degree = weighted_degree(edges_df, assets)
    positions = force_layout(assets, edges_df)

    sector_colors = {
        "Green Assets": "green",
        "Conventional": "orange",
    }

    fig = go.Figure()

    if not edges_df.empty:
        min_weight = edges_df["weight"].min()
        max_weight = edges_df["weight"].max()
    else:
        min_weight = max_weight = threshold

    for _, edge in edges_df.iterrows():
        x0, y0 = positions[edge["from"]]
        x1, y1 = positions[edge["to"]]
        weight = float(edge["weight"])
        scaled = (weight - min_weight) / (max_weight - min_weight + 1e-9)
        width = 0.8 + 2.2 * scaled
        alpha = 0.25 + 0.55 * scaled

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=width, color=f"rgba(0, 0, 0, {alpha:.2f})"),
            hoverinfo="text",
            text=f"{edge['from']} → {edge['to']}<br>adj_R2={weight:.3f}",
            showlegend=False,
        ))

        dx = x1 - x0
        dy = y1 - y0
        shrink = 0.08
        fig.add_annotation(
            x=x1 - shrink * dx,
            y=y1 - shrink * dy,
            ax=x0 + shrink * dx,
            ay=y0 + shrink * dy,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=width,
            arrowcolor=f"rgba(0, 0, 0, {alpha:.2f})",
            opacity=alpha,
        )

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    hover_text = []

    min_degree = float(degree.min())
    max_degree = float(degree.max())

    for asset in assets:
        x, y = positions[asset]
        sector = classify_asset_sector(asset)
        deg = float(degree.loc[asset])
        scaled_degree = (deg - min_degree) / (max_degree - min_degree + 1e-9)

        node_x.append(x)
        node_y.append(y)
        node_color.append(sector_colors[sector])
        node_size.append(14 + 28 * scaled_degree)
        node_text.append(asset)
        hover_text.append(
            f"{asset}<br>Sector: {sector}<br>Fuerza ponderada: {deg:.3f}"
        )

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1.5, color="black"),
            opacity=0.92,
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=12, color="black"),
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False,
    ))

    for sector, color in sector_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=color, line=dict(width=1, color="black")),
            name=sector,
        ))

    fig.update_layout(
        title="Red de interdependencia wavelet",
        height=760,
        xaxis=dict(visible=False, range=[-1.25, 1.25]),
        yaxis=dict(visible=False, range=[-1.25, 1.25]),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="v", x=1.02, y=0.95),
    )

    return fig, edges_df, degree, len(all_edges_df)


# ============================================================
# ARQUETIPOS
# ============================================================

def fit_archetype_proxy(adj_r2: pd.DataFrame, n_archetypes: int = 3):
    """
    Aproximación práctica a arquetipos usando NMF sobre adj_R2.

    Cada activo queda expresado como mezcla convexa de arquetipos.
    """
    X = adj_r2.values
    X = np.maximum(X, 0)

    model = NMF(
        n_components=n_archetypes,
        init="nndsvda",
        random_state=33,
        max_iter=5000
    )

    alpha = model.fit_transform(X)
    alpha = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-12)

    alpha_df = pd.DataFrame(
        alpha,
        index=adj_r2.index,
        columns=[f"Arquetipo {i + 1}" for i in range(n_archetypes)]
    )

    return alpha_df


def plot_archetypes(alpha_df: pd.DataFrame):
    plot_df = alpha_df.copy()
    plot_df["Activo"] = plot_df.index
    plot_df["Dominante"] = alpha_df.idxmax(axis=1)

    if alpha_df.shape[1] == 3:
        fig = px.scatter_ternary(
            plot_df,
            a=alpha_df.columns[0],
            b=alpha_df.columns[1],
            c=alpha_df.columns[2],
            color="Dominante",
            hover_name="Activo",
            title="Gráfico dinámico de arquetipos"
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
        fig.update_layout(height=720)
        return fig

    long_df = plot_df.melt(
        id_vars=["Activo", "Dominante"],
        var_name="Arquetipo",
        value_name="Peso"
    )

    fig = px.bar(
        long_df,
        x="Activo",
        y="Peso",
        color="Arquetipo",
        barmode="stack",
        title="Composición arquetípica de activos"
    )
    fig.update_layout(height=650, xaxis_tickangle=-60)

    return fig


# ============================================================
# K-MEANS
# ============================================================

def choose_best_k_silhouette(distance_matrix: pd.DataFrame, max_k=8):
    X = distance_matrix.values
    results = []

    upper = min(max_k, len(distance_matrix) - 1)

    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, random_state=33, n_init=25)
        labels = model.fit_predict(X)

        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = np.nan

        results.append((k, score))

    scores = pd.DataFrame(results, columns=["k", "silhouette"])
    scores = scores.dropna()

    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])

    return best_k, scores


def fit_kmeans(distance_matrix: pd.DataFrame, k: int):
    model = KMeans(n_clusters=k, random_state=33, n_init=25)
    labels = model.fit_predict(distance_matrix.values)
    return labels


def plot_kmeans_mds(distance_matrix: pd.DataFrame, labels):
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=33,
        normalized_stress="auto"
    )

    coords = mds.fit_transform(distance_matrix.values)

    plot_df = pd.DataFrame({
        "Dim 1": coords[:, 0],
        "Dim 2": coords[:, 1],
        "Activo": distance_matrix.index,
        "Cluster": [f"Cluster {x + 1}" for x in labels]
    })

    fig = px.scatter(
        plot_df,
        x="Dim 1",
        y="Dim 2",
        color="Cluster",
        text="Activo",
        hover_name="Activo",
        title="K-Means sobre distancia wavelet: 1 - coherencia"
    )

    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.update_layout(height=720)

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Datos")

data_dir = st.sidebar.text_input(
    "Carpeta con los Excels",
    value=default_data_dir()
)

bond_sheet = st.sidebar.selectbox(
    "Hoja de bonos Eurostat",
    ["Bond_2001", "Bond_2015"],
    index=0
)

st.sidebar.header("Wavelet")

min_scale = st.sidebar.slider("Escala mínima", 2, 24, 4)
max_scale = st.sidebar.slider("Escala máxima", 16, 128, 64)
n_scales = st.sidebar.slider("Número de escalas", 8, 48, 24)

st.sidebar.header("Modelos")

n_archetypes = st.sidebar.slider("Número de arquetipos", 2, 5, 3)

auto_k = st.sidebar.checkbox("Elegir k automáticamente", value=True)
manual_k = st.sidebar.slider("k manual", 2, 8, 3)
max_k = st.sidebar.slider("Máximo k a probar", 3, 12, 8)


# ============================================================
# CARGA DE DATOS
# ============================================================

try:
    data = build_dataset(data_dir, bond_sheet)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Dataset cargado: {data.shape[0]} fechas x {data.shape[1]} activos")

date_min = data.index.min().strftime("%Y-%m")
date_max = data.index.max().strftime("%Y-%m")
summary_cols = st.columns(4)
summary_cols[0].metric("Observaciones mensuales", f"{data.shape[0]}")
summary_cols[1].metric("Activos disponibles", f"{data.shape[1]}")
summary_cols[2].metric("Periodo", f"{date_min} / {date_max}")
summary_cols[3].metric("Hoja de bonos", bond_sheet)

if data.empty:
    st.error(
        "El dataset está vacío. Revisa que las fechas de los Excels se solapen "
        "y que la hoja de bonos elegida sea correcta."
    )
    st.stop()

with st.expander("Ver datos de retornos logarítmicos"):
    st.dataframe(data.round(6))


selected_assets = st.multiselect(
    "Selecciona activos",
    options=list(data.columns),
    default=list(data.columns)
)

if len(selected_assets) < 3:
    st.warning("Selecciona al menos 3 activos.")
    st.stop()

data = data[selected_assets].dropna()

scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    index=data.index,
    columns=data.columns
)

with st.expander("Configuracion del analisis"):
    st.write(
        {
            "activos_seleccionados": len(selected_assets),
            "escala_minima": min_scale,
            "escala_maxima": max_scale,
            "numero_escalas": n_scales,
            "numero_arquetipos": n_archetypes,
            "seleccion_k_automatica": auto_k,
            "maximo_k": max_k if auto_k else manual_k,
        }
    )


# ============================================================
# CÁLCULO WAVELET
# ============================================================

st.subheader("Matriz de coherencia wavelet")

col_a, col_b = st.columns([1, 3])

with col_a:
    calculate = st.button("Recalcular coherencia wavelet")

with col_b:
    st.info(
        "La matriz adj_R2 se calcula como la media de la coherencia wavelet "
        "R² para cada par de activos."
    )

analysis_signature = {
    "assets": tuple(data_scaled.columns),
    "start": str(data_scaled.index.min()),
    "end": str(data_scaled.index.max()),
    "min_scale": min_scale,
    "max_scale": max_scale,
    "n_scales": n_scales,
}

needs_calculation = (
    calculate
    or "adj_r2" not in st.session_state
    or st.session_state.get("analysis_signature") != analysis_signature
)

if needs_calculation:
    st.session_state["adj_r2"] = compute_wavelet_coherence_matrix(
        data_scaled,
        min_scale=min_scale,
        max_scale=max_scale,
        n_scales=n_scales
    )
    st.session_state["analysis_signature"] = analysis_signature

adj_r2 = st.session_state["adj_r2"]
distance_matrix = compute_distance_matrix(adj_r2)

alpha_df = fit_archetype_proxy(adj_r2, n_archetypes=n_archetypes)

if auto_k:
    best_k, scores_df = choose_best_k_silhouette(distance_matrix, max_k=max_k)
    k = best_k
else:
    scores_df = pd.DataFrame(columns=["k", "silhouette"])
    k = manual_k

labels = fit_kmeans(distance_matrix, k)
cluster_df = pd.DataFrame({
    "Activo": distance_matrix.index,
    "Cluster": labels + 1
}).sort_values("Cluster")
resumen = cluster_df.groupby("Cluster")["Activo"].apply(list).reset_index()


# ============================================================
# TABS
# ============================================================

section = st.radio(
    "Vista",
    [
        "1. Arquetipos",
        "2. K-Means",
        "3. Matrices",
        "4. Red",
        "5. Series",
        "6. Exportar",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

if section == "1. Arquetipos":
    st.header("Gráfico dinámico de arquetipos")

    alpha_df = fit_archetype_proxy(adj_r2, n_archetypes=n_archetypes)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plot_archetypes(alpha_df)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Pesos α")
        st.dataframe(alpha_df.round(3))

        dominant_df = pd.DataFrame({
            "Activo": alpha_df.index,
            "Arquetipo dominante": alpha_df.idxmax(axis=1),
            "Peso dominante": alpha_df.max(axis=1)
        }).sort_values("Arquetipo dominante")

        st.subheader("Clasificación arquetípica")
        st.dataframe(dominant_df.round(3))


# ============================================================
# TAB 2: K-MEANS
# ============================================================

elif section == "2. K-Means":
    st.header("Clustering K-Means")

    if auto_k:
        best_k, scores_df = choose_best_k_silhouette(distance_matrix, max_k=max_k)
        k = best_k

        st.success(f"k elegido automáticamente por silhouette: {k}")

        fig_score = px.line(
            scores_df,
            x="k",
            y="silhouette",
            markers=True,
            title="Selección de k por silhouette"
        )
        st.plotly_chart(fig_score, width="stretch")

    else:
        k = manual_k

    labels = fit_kmeans(distance_matrix, k)

    fig_km = plot_kmeans_mds(distance_matrix, labels)
    st.plotly_chart(fig_km, width="stretch")

    cluster_df = pd.DataFrame({
        "Activo": distance_matrix.index,
        "Cluster": labels + 1
    }).sort_values("Cluster")

    st.subheader("Activos por cluster")
    st.dataframe(cluster_df)

    resumen = cluster_df.groupby("Cluster")["Activo"].apply(list).reset_index()

    st.subheader("Resumen por cluster")
    st.dataframe(resumen)


# ============================================================
# TAB 3: MATRICES
# ============================================================

elif section == "3. Matrices":
    st.header("Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("adj_R2: coherencia wavelet media")
        st.dataframe(adj_r2.round(3))

        fig_heat = px.imshow(
            adj_r2,
            text_auto=".2f",
            title="Heatmap de coherencia wavelet"
        )
        st.plotly_chart(fig_heat, width="stretch")

    with col2:
        st.subheader("Distancia = 1 - adj_R2")
        st.dataframe(distance_matrix.round(3))

        fig_dist = px.imshow(
            distance_matrix,
            text_auto=".2f",
            title="Heatmap de distancia wavelet"
        )
        st.plotly_chart(fig_dist, width="stretch")


# ============================================================
# TAB 4: RED
# ============================================================

elif section == "4. Red":
    st.header("Red de interdependencia wavelet")

    upper_values = adj_r2.to_numpy()[np.triu_indices_from(adj_r2.to_numpy(), k=1)]
    suggested_threshold = float(np.quantile(upper_values, 0.85))
    default_threshold = round(max(0.42, min(1.0, suggested_threshold)), 2)

    network_threshold = st.slider(
        "Umbral de coherencia para dibujar aristas",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.01,
        help=(
            "En la Tercera Reunion se usaba 0.42. En este dataset la matriz "
            "es muy densa, asi que la app propone un umbral mas alto para que "
            "el grafo sea legible."
        ),
    )

    max_network_edges = st.slider(
        "Maximo de aristas visibles",
        min_value=10,
        max_value=200,
        value=80,
        step=10,
        help="Si hay muchas relaciones por encima del umbral, se dibujan las mas fuertes.",
    )

    fig_network, edges_df, degree, total_edges = plot_wavelet_network(
        adj_r2,
        threshold=network_threshold,
        max_edges=max_network_edges,
    )

    st.caption(
        f"Relaciones por encima del umbral: {total_edges}. "
        f"Aristas dibujadas: {len(edges_df)}."
    )

    st.plotly_chart(fig_network, width="stretch")

    col_edges, col_nodes = st.columns(2)

    with col_edges:
        st.subheader("Aristas filtradas")
        st.dataframe(
            edges_df.sort_values("weight", ascending=False).round(3),
            width="stretch",
        )

    with col_nodes:
        st.subheader("Fuerza ponderada por activo")
        node_summary = pd.DataFrame({
            "Activo": degree.index,
            "Sector": [classify_asset_sector(asset) for asset in degree.index],
            "Degree": degree.values,
        }).sort_values("Degree", ascending=False)
        st.dataframe(node_summary.round(3), width="stretch")

    st.caption(
        "Estilo basado en la seccion 2.2 de Tercera Reunion: layout de fuerzas, "
        "aristas ponderadas por coherencia, flechas, nodos verdes para activos "
        "verdes y naranjas para convencionales."
    )


# ============================================================
# TAB 5: SERIES
# ============================================================

elif section == "5. Series":
    st.header("Series de retornos logarítmicos")

    assets_to_plot = st.multiselect(
        "Activos a visualizar",
        options=list(data.columns),
        default=list(data.columns[:min(6, len(data.columns))])
    )

    if assets_to_plot:
        fig_series = px.line(
            data[assets_to_plot],
            title="Retornos logarítmicos"
        )
        st.plotly_chart(fig_series, width="stretch")


# ============================================================
# TAB 6: EXPORTAR
# ============================================================

elif section == "6. Exportar":
    st.header("Exportar resultados")

    export_edges_df = build_network_edges(adj_r2, threshold=0.42)
    export_degree = weighted_degree(export_edges_df, list(adj_r2.index))
    export_node_summary = pd.DataFrame({
        "Activo": export_degree.index,
        "Sector": [classify_asset_sector(asset) for asset in export_degree.index],
        "Degree": export_degree.values,
    }).sort_values("Degree", ascending=False)

    export_sheets = {
        "retornos_log": data,
        "coherencia_adj_R2": adj_r2,
        "distancia_wavelet": distance_matrix,
        "pesos_arquetipos": alpha_df,
        "clusters": cluster_df.set_index("Activo"),
        "resumen_clusters": resumen.set_index("Cluster"),
        "red_aristas": export_edges_df.set_index(["from", "to"]) if not export_edges_df.empty else export_edges_df,
        "red_nodos": export_node_summary.set_index("Activo"),
    }

    if auto_k:
        export_sheets["silhouette_k"] = scores_df.set_index("k")

    st.download_button(
        "Descargar resultados en Excel",
        data=dataframe_to_excel_bytes(export_sheets),
        file_name="wavelet_analysis_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        "Descargar matriz adj_R2 en CSV",
        data=adj_r2.to_csv().encode("utf-8"),
        file_name="wavelet_adj_R2.csv",
        mime="text/csv",
    )

    st.caption(
        "Estos archivos sirven como evidencia reproducible para anexos, "
        "revision metodologica o comparaciones posteriores."
    )
