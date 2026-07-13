import warnings
import os
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from archetypes import ADA, BiAA
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score


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


def has_final_data_file(path: Path) -> bool:
    return (path / "datos_final.xlsx").exists()


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
        if has_final_data_file(candidate) or has_expected_data_files(candidate):
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
st.caption("Coherencia wavelet + arquetipoides dinámicos + clustering K-Means")


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


def read_eurostat_bonds(path, sheet_name="Bond_raw_EU", drop_cols=("EA", "EU27_2020", "JP", "US", "TR")):
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
    returns = returns.dropna(axis=1, how="all")
    returns = to_month_index(returns)

    return returns


def read_yahoo_assets(path):
    returns = pd.read_excel(path, sheet_name="Monthly_Log_Returns")

    if "Date" not in returns.columns:
        raise ValueError(f"No encuentro la columna 'Date' en {path}")

    returns["Date"] = pd.to_datetime(returns["Date"], errors="coerce")
    returns = returns.dropna(subset=["Date"]).set_index("Date").sort_index()

    for col in returns.columns:
        returns[col] = pd.to_numeric(returns[col], errors="coerce")

    returns = returns.dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    returns = to_month_index(returns)

    metadata = pd.read_excel(path, sheet_name="Metadata")
    metadata = metadata.rename(columns={
        "Label": "Activo",
        "Group": "Grupo",
        "Sector": "Sector",
        "Yahoo_Ticker": "Ticker",
    })
    metadata = metadata[metadata["Activo"].isin(returns.columns)]

    return returns, metadata


def read_final_dataset(path):
    df = pd.read_excel(path)

    date_col = "Fechas" if "Fechas" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    drop_cols = [col for col in ["N"] if col in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all")
    monthly = df.resample("ME").last()
    monthly = to_month_index(monthly)

    returns = np.log(monthly / monthly.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    returns = returns.loc[:, returns.std(skipna=True) > 0]
    returns = to_month_index(returns)

    return returns


@st.cache_data
def build_dataset(data_dir: str, bond_sheet: str):
    data_dir = Path(data_dir)

    final_path = data_dir / "datos_final.xlsx"
    if final_path.exists():
        return read_final_dataset(final_path)

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

    yahoo_path = data_dir / "Yahoo_Assets.xlsx"
    if yahoo_path.exists():
        yahoo_returns, _ = read_yahoo_assets(yahoo_path)
        series.append(yahoo_returns)

    dataset = pd.concat(series, axis=1, join="outer")
    dataset = dataset.sort_index()
    dataset = dataset.dropna(axis=0, how="all")
    dataset = dataset.loc[:, dataset.std(skipna=True) > 0]

    return dataset


@st.cache_data
def build_asset_catalog(data_dir: str, columns: tuple[str, ...]):
    rows = []

    base_groups = {
        "CSI_China": ("Indices verdes propios", "Green Assets", "Excel"),
        "Wind": ("Indices verdes propios", "Green Assets", "Excel"),
        "SP_Clean": ("Indices verdes propios", "Green Assets", "Excel"),
        "WilderHill": ("Indices verdes propios", "Green Assets", "Excel"),
        "Solar": ("Hui energia", "Green Assets", "Excel"),
        "Uranium": ("Hui energia", "Energia", "Excel"),
        "FTSE_Env": ("Hui energia", "Green Assets", "Excel"),
        "ERIXP_EU": ("Hui energia", "Green Assets", "Excel"),
    }

    for asset in columns:
        if asset in base_groups:
            group, sector, source = base_groups[asset]
        elif asset.isupper() and len(asset) <= 4:
            group, sector, source = ("Bonos Eurostat", "Bonos", "Excel")
        else:
            group, sector, source = ("Acciones renovables", "Green Assets", "datos_final.xlsx")
        rows.append({
            "Activo": asset,
            "Grupo": group,
            "Sector": sector,
            "Fuente": source,
            "Ticker": "",
        })

    yahoo_path = Path(data_dir) / "Yahoo_Assets.xlsx"
    if yahoo_path.exists():
        try:
            _, yahoo_metadata = read_yahoo_assets(yahoo_path)
            yahoo_metadata = yahoo_metadata.assign(Fuente="Yahoo Finance")

            rows_by_asset = {row["Activo"]: row for row in rows}
            for row in yahoo_metadata.to_dict("records"):
                if row["Activo"] in rows_by_asset:
                    rows_by_asset[row["Activo"]].update({
                        "Grupo": row.get("Grupo", "Yahoo Finance"),
                        "Sector": row.get("Sector", "Yahoo Finance"),
                        "Fuente": "Yahoo Finance",
                        "Ticker": row.get("Ticker", ""),
                    })

            rows = list(rows_by_asset.values())
        except Exception:
            pass

    catalog = pd.DataFrame(rows)
    catalog = catalog[catalog["Activo"].isin(columns)]
    return catalog.sort_values(["Grupo", "Activo"]).reset_index(drop=True)


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
    mat = np.full((n, n), np.nan, dtype=float)

    total = n * (n - 1) // 2
    counter = 0
    progress = st.progress(0)

    for i in range(n):
        for j in range(i + 1, n):
            pair = data[[cols[i], cols[j]]].dropna()
            if len(pair) >= max(24, n_scales):
                r2 = wavelet_coherence_pair(
                    pair.iloc[:, 0].values,
                    pair.iloc[:, 1].values,
                    scales,
                )
                mat[i, j] = np.nanmean(r2)
            else:
                mat[i, j] = 0.0
            mat[j, i] = mat[i, j]

            counter += 1
            progress.progress(counter / total)

    np.fill_diagonal(mat, 1.0)
    mat = np.nan_to_num(mat, nan=0.0)

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
    sector_by_asset = {
        "Nvidia": "Tecnologia",
        "Microsoft": "Tecnologia",
        "Apple": "Tecnologia",
        "Google": "Tecnologia",
        "Tesla": "Tecnologia",
        "Bitcoin": "Crypto",
        "Ethereum": "Crypto",
        "Solana": "Crypto",
        "Binance": "Crypto",
        "Exxon": "Energia",
        "Chevron": "Energia",
        "Boeing": "Industrial",
        "Caterpillar": "Industrial",
        "EliLilly": "Farmaceutica y Defensa",
        "J&J": "Farmaceutica y Defensa",
        "Pfizer": "Farmaceutica y Defensa",
        "Merck": "Farmaceutica y Defensa",
        "AbbVie": "Farmaceutica y Defensa",
        "S&P500": "Indices",
        "Nasdaq": "Indices",
        "VIX": "Indices",
        "ORO": "Commodities",
        "Bonos20A": "Bonos",
        "Dolar_IDX": "Divisas",
        "Euro": "Divisas",
        "Libra": "Divisas",
        "Yen": "Divisas",
        "FrancoSuizo": "Divisas",
        "AudDolar": "Divisas",
    }
    if asset_name in sector_by_asset:
        return sector_by_asset[asset_name]

    green_assets = {
        "CSI_China",
        "Wind",
        "SP_Clean",
        "WilderHill",
        "Solar",
        "Uranium",
        "FTSE_Env",
        "ERIXP_EU",
        "SPGBI",
        "GRNENEF",
        "GRNFUEL",
        "GRNGB",
        "GRNPOL",
        "GRNSOLAR",
        "GRNTRN",
        "GRNWIND",
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


def force_layout(
    assets: list[str],
    edges_df: pd.DataFrame,
    initial_positions: np.ndarray | None = None,
    seed: int = 555,
):
    """
    Layout de fuerzas reproducible inspirado en Fruchterman-Reingold.
    Evita añadir networkx como dependencia solo para esta visualizacion.
    """
    rng = np.random.default_rng(seed)
    n = len(assets)

    if n == 1:
        return {assets[0]: np.array([0.0, 0.0])}

    if initial_positions is None:
        positions = rng.uniform(-1.0, 1.0, size=(n, 2))
    else:
        positions = np.asarray(initial_positions, dtype=float).copy()
    index = {asset: idx for idx, asset in enumerate(assets)}
    area = 4.0
    k = np.sqrt(area / n)
    temperature = 0.18

    edge_pairs = [
        (index[row["from"]], index[row["to"]], float(row["weight"]))
        for _, row in edges_df.iterrows()
        if row["from"] in index and row["to"] in index
    ]

    for _ in range(700):
        disp = np.zeros_like(positions)

        for i in range(n):
            delta = positions[i] - positions
            distance = np.linalg.norm(delta, axis=1) + 1e-9
            force = (0.22 * k * k / distance)[:, None] * delta / distance[:, None]
            force[i] = 0
            disp[i] += force.sum(axis=0)

        for source, target, weight in edge_pairs:
            delta = positions[source] - positions[target]
            distance = np.linalg.norm(delta) + 1e-9
            target_distance = 0.08 + 0.72 * (1 - weight) ** 2
            force = (distance - target_distance) * (0.30 + 7.5 * weight ** 2)
            vector = (delta / distance) * force
            disp[source] -= vector
            disp[target] += vector

        lengths = np.linalg.norm(disp, axis=1) + 1e-9
        positions += (disp / lengths[:, None]) * np.minimum(lengths, temperature)[:, None]
        positions -= positions.mean(axis=0)
        temperature *= 0.994

    max_abs = np.abs(positions).max()
    if max_abs > 0:
        positions = positions / max_abs

    return {asset: positions[index[asset]] for asset in assets}


def network_layout(adj_r2: pd.DataFrame, threshold: float = 0.42):
    assets = list(adj_r2.index)
    edges_df = build_network_edges(adj_r2, threshold)
    dist_for_layout = (1 - adj_r2).clip(lower=0, upper=1)
    labels = None

    if len(assets) >= 4:
        k_layout = min(5, len(assets) - 1)
        labels = KMeans(n_clusters=k_layout, random_state=555, n_init=25).fit_predict(
            dist_for_layout.values
        )

    if len(assets) >= 3:
        mds_positions = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=555,
            normalized_stress="auto",
        ).fit_transform(dist_for_layout.values)

        if labels is not None:
            initial_positions = np.zeros_like(mds_positions)
            angles = np.linspace(0, 2 * np.pi, len(set(labels)), endpoint=False)
            centers = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.15
            for cluster_id in sorted(set(labels)):
                idx = np.where(labels == cluster_id)[0]
                local = mds_positions[idx]
                local = local - local.mean(axis=0)
                scale = np.abs(local).max() or 1.0
                initial_positions[idx] = centers[cluster_id] + 0.35 * local / scale
        else:
            initial_positions = mds_positions
    else:
        initial_positions = None

    positions = force_layout(assets, edges_df, initial_positions=initial_positions)
    return positions, edges_df


def plot_wavelet_network(adj_r2: pd.DataFrame, threshold: float = 0.42):
    assets = list(adj_r2.index)
    positions, edges_df = network_layout(adj_r2, threshold=threshold)
    degree = weighted_degree(edges_df, assets)

    sector_colors = {
        "Green Assets": "#2ca02c",
        "Conventional": "#ff7f0e",
        "Tecnologia": "#1f77b4",
        "Crypto": "#d62728",
        "Energia": "#17becf",
        "Industrial": "#8c564b",
        "Farmaceutica y Defensa": "#9467bd",
        "Indices": "#7f7f7f",
        "Commodities": "#bcbd22",
        "Bonos": "#aec7e8",
        "Divisas": "#e377c2",
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
        node_color.append(sector_colors.get(sector, "#ff7f0e"))
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

    return fig, edges_df, degree


# ============================================================
# ARQUETIPOIDES
# ============================================================

def fit_archetype_proxy(adj_r2: pd.DataFrame, n_archetypes: int = 3):
    """
    Archetypoid Analysis formal (ADA).

    Cada activo se representa por su perfil de coherencias wavelet con el resto.
    ADA selecciona arquetipoides reales y minimiza ||X - A B X||^2 bajo
    restricciones convexas en A, con B restringido a observaciones reales.
    """
    assets = list(adj_r2.index)
    n_components = min(n_archetypes, len(assets))
    X = adj_r2.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(X, 0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model = ADA(
        n_components,
        init="furthest_first",
        n_init=5,
        max_iter=500,
        tol=1e-5,
        method="nnls",
        random_state=33,
    )
    alpha = model.fit_transform(X)
    selected = model.B_.argmax(axis=1)
    archetypoids = [assets[i] for i in selected]

    alpha_df = pd.DataFrame(
        alpha,
        index=adj_r2.index,
        columns=[f"Arquetipoide {i + 1}: {name}" for i, name in enumerate(archetypoids)]
    )
    alpha_df.attrs["rss"] = float(model.rss_)
    alpha_df.attrs["reconstruction_error"] = float(model.reconstruction_error_)
    alpha_df.attrs["selected_archetypoids"] = archetypoids

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
            title="Grafico dinamico de arquetipoides"
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
        fig.update_layout(height=720)
        return fig

    long_df = plot_df.melt(
        id_vars=["Activo", "Dominante"],
        var_name="Arquetipoide",
        value_name="Peso"
    )

    fig = px.bar(
        long_df,
        x="Activo",
        y="Peso",
        color="Arquetipoide",
        barmode="stack",
        title="Composicion por arquetipoides"
    )
    fig.update_layout(height=650, xaxis_tickangle=-60)

    return fig


def select_extreme_representatives(distance_matrix: pd.DataFrame, n_components: int):
    assets = list(distance_matrix.index)
    n_components = min(n_components, len(assets))
    dist = distance_matrix.to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)

    selected = [int(np.argmax(dist.mean(axis=1)))]
    while len(selected) < n_components:
        min_dist = dist[:, selected].min(axis=1)
        min_dist[selected] = -1
        selected.append(int(np.argmax(min_dist)))

    return selected


def convex_weights_from_distance(distance_matrix: pd.DataFrame, selected_indices: list[int], prefix: str):
    dist = distance_matrix.to_numpy(copy=True)
    scores = 1 / (dist[:, selected_indices] + 1e-6)
    weights = scores / (scores.sum(axis=1, keepdims=True) + 1e-12)

    for pos, idx in enumerate(selected_indices):
        weights[idx, :] = 0
        weights[idx, pos] = 1

    selected_names = [str(distance_matrix.index[idx]) for idx in selected_indices]
    return pd.DataFrame(
        weights,
        index=distance_matrix.index,
        columns=[f"{prefix} {i + 1}: {name}" for i, name in enumerate(selected_names)],
    )


def compute_time_distance(data: pd.DataFrame):
    complete = data.copy()
    complete = complete.apply(lambda col: col.fillna(col.mean()), axis=0).fillna(0.0)
    values = complete.to_numpy(dtype=float)
    values = (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-9)

    diff = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.mean(diff * diff, axis=2))
    labels = pd.Index([idx.strftime("%Y-%m") for idx in complete.index], name="Fecha")
    return pd.DataFrame(distances, index=labels, columns=labels)


@st.cache_data
def fit_biarchetype_proxy(data: pd.DataFrame, distance_matrix: pd.DataFrame, n_components: int):
    """
    Biarchetype Analysis formal sobre X = activos x tiempo.

    BiAA minimiza ||X - A_row B_row X B_col A_col||^2 con restricciones
    convexas en las dos dimensiones: activos y fechas.
    """
    del distance_matrix

    X_df = data.T.copy()
    X_df = X_df.apply(lambda row: row.fillna(row.mean()), axis=1).fillna(0.0)
    X = X_df.to_numpy(dtype=float)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)

    n_row = min(n_components, X.shape[0])
    n_col = min(n_components, X.shape[1])
    model = BiAA(
        (n_row, n_col),
        init="furthest_first",
        n_init=3,
        max_iter=500,
        tol=1e-5,
        method="pgd",
        random_state=44,
    )
    model.fit(X)

    asset_representatives = model.B_[0].argmax(axis=1)
    time_representatives = model.B_[1].argmax(axis=1)
    asset_names = list(X_df.index)
    time_names = [idx.strftime("%Y-%m") for idx in X_df.columns]

    asset_weights = pd.DataFrame(
        model.coefficients_[0],
        index=asset_names,
        columns=[
            f"Biarquetipo activo {i + 1}: {asset_names[idx]}"
            for i, idx in enumerate(asset_representatives)
        ],
    )
    time_weights = pd.DataFrame(
        model.coefficients_[1],
        index=pd.Index(time_names, name="Fecha"),
        columns=[
            f"Biarquetipo temporal {i + 1}: {time_names[idx]}"
            for i, idx in enumerate(time_representatives)
        ],
    )
    asset_weights.attrs["rss"] = float(model.rss_)
    time_weights.attrs["rss"] = float(model.rss_)

    return asset_weights, time_weights, pd.DataFrame(model.archetypes_)


def plot_biarchetype_assets(asset_weights: pd.DataFrame):
    long_df = (
        asset_weights.reset_index(names="Activo")
        .melt(id_vars="Activo", var_name="Biarquetipo", value_name="Peso")
    )
    fig = px.bar(
        long_df,
        x="Activo",
        y="Peso",
        color="Biarquetipo",
        barmode="stack",
        title="Composicion de activos por biarquetipos",
    )
    fig.update_layout(height=650, xaxis_tickangle=-60)
    return fig


def plot_biarchetype_time(time_weights: pd.DataFrame):
    plot_df = time_weights.copy()
    plot_df["Fecha"] = pd.to_datetime(plot_df.index)
    long_df = plot_df.melt(id_vars="Fecha", var_name="Biarquetipo", value_name="Peso")
    fig = px.area(
        long_df,
        x="Fecha",
        y="Peso",
        color="Biarquetipo",
        title="Composicion temporal por biarquetipos",
    )
    fig.update_layout(height=520, yaxis_range=[0, 1])
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
# EXPORTACION DE GRAFICOS
# ============================================================

def pil_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_wrapped(draw, text, xy, font, fill=(30, 30, 30), max_width=980, line_gap=5):
    x, y = xy
    words = str(text).split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            line = trial
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + line_gap
            line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def hex_to_rgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def new_pdf_page(title: str):
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1240, 88], fill=(23, 50, 77))
    draw.text((58, 28), title, font=pil_font(28, bold=True), fill="white")
    draw.line([58, 118, 1182, 118], fill=(210, 218, 226), width=2)
    return image, draw


def draw_heatmap(draw, matrix: pd.DataFrame, box, title: str):
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 34), title, font=pil_font(22, bold=True), fill=(23, 50, 77))
    max_assets = min(22, len(matrix))
    if len(matrix) > max_assets:
        strength = matrix.sum(axis=1).sort_values(ascending=False).head(max_assets).index
        matrix = matrix.loc[strength, strength]
    labels_short = [str(x)[:9] for x in matrix.index]
    values = matrix.values
    n = len(matrix)
    if n == 0:
        return
    left_label = 135
    top_label = 72
    grid_w = x1 - x0 - left_label
    grid_h = y1 - y0 - top_label
    cell = int(min(grid_w, grid_h) / n)
    gx = x0 + left_label
    gy = y0 + top_label
    for i in range(n):
        draw.text((x0, gy + i * cell + 2), labels_short[i], font=pil_font(10), fill=(40, 40, 40))
        draw.text((gx + i * cell + 2, y0 + 8), labels_short[i], font=pil_font(9), fill=(40, 40, 40))
        for j in range(n):
            v = float(values[i, j])
            if np.isnan(v):
                color = (235, 235, 235)
            else:
                red = int(247 - 210 * v)
                green = int(251 - 120 * v)
                blue = int(255 - 155 * v)
                color = (max(20, red), max(50, green), max(80, blue))
            draw.rectangle(
                [gx + j * cell, gy + i * cell, gx + (j + 1) * cell, gy + (i + 1) * cell],
                fill=color,
                outline=(245, 245, 245),
            )
            if n <= 24:
                text_color = (255, 255, 255) if v > 0.55 else (20, 20, 20)
                draw.text(
                    (gx + j * cell + 3, gy + i * cell + max(2, cell // 4)),
                    f"{v:.2f}",
                    font=pil_font(max(8, min(14, cell // 3)), bold=False),
                    fill=text_color,
                )


def draw_network_pdf(draw, adj_r2: pd.DataFrame, threshold: float, box):
    x0, y0, x1, y1 = box
    positions, edges = network_layout(adj_r2, threshold)
    degree = weighted_degree(edges, list(adj_r2.index))
    sector_colors = {
        "Green Assets": "#2ca02c",
        "Conventional": "#ff7f0e",
        "Tecnologia": "#1f77b4",
        "Crypto": "#d62728",
        "Energia": "#17becf",
        "Industrial": "#8c564b",
        "Farmaceutica y Defensa": "#9467bd",
        "Indices": "#7f7f7f",
        "Commodities": "#bcbd22",
        "Bonos": "#aec7e8",
        "Divisas": "#e377c2",
    }

    def map_xy(pos):
        px = x0 + (pos[0] + 1.15) / 2.3 * (x1 - x0)
        py = y1 - (pos[1] + 1.15) / 2.3 * (y1 - y0)
        return px, py

    if not edges.empty:
        top_edges = edges.sort_values("weight", ascending=False).head(90)
        min_w = float(top_edges["weight"].min())
        max_w = float(top_edges["weight"].max())
        for _, edge in top_edges.sort_values("weight").iterrows():
            p0 = map_xy(positions[edge["from"]])
            p1 = map_xy(positions[edge["to"]])
            scaled = (float(edge["weight"]) - min_w) / (max_w - min_w + 1e-9)
            width = int(1 + 5 * scaled)
            shade = int(170 - 120 * scaled)
            draw.line([p0, p1], fill=(shade, shade, shade), width=width)

    min_d = float(degree.min())
    max_d = float(degree.max())
    for asset, pos in positions.items():
        px, py = map_xy(pos)
        deg = float(degree.loc[asset])
        scaled = (deg - min_d) / (max_d - min_d + 1e-9)
        radius = int(8 + 14 * scaled)
        color = hex_to_rgb(sector_colors.get(classify_asset_sector(asset), "#ff7f0e"))
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline=(0, 0, 0), width=2)
        draw.text((px + radius + 3, py - 6), str(asset)[:12], font=pil_font(11), fill=(20, 20, 20))


def image_bytes(image: Image.Image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def plotly_png_bytes(fig, width=1800, height=1400, scale=2):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


@st.cache_data
def build_graphics_zip_bytes(
    data: pd.DataFrame,
    adj_r2: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    alpha_df: pd.DataFrame,
    bi_asset_weights: pd.DataFrame,
    bi_time_weights: pd.DataFrame,
    cluster_df: pd.DataFrame,
    network_threshold: float,
):
    graphics = {}

    fig_network, _, _ = plot_wavelet_network(adj_r2, threshold=network_threshold)
    exported = plotly_png_bytes(fig_network, width=2200, height=1600, scale=2)
    if exported is None:
        image = Image.new("RGB", (2200, 1600), "white")
        draw = ImageDraw.Draw(image)
        draw_network_pdf(draw, adj_r2, network_threshold, (90, 80, 2110, 1510))
        exported = image_bytes(image)
    graphics["01_red_wavelet.png"] = exported

    fig_heat = px.imshow(adj_r2, text_auto=".2f", title="Heatmap de coherencia wavelet")
    exported = plotly_png_bytes(fig_heat, width=2200, height=1800, scale=2)
    if exported is None:
        image = Image.new("RGB", (2200, 1800), "white")
        draw = ImageDraw.Draw(image)
        draw_heatmap(draw, adj_r2, (80, 130, 2120, 1680), "adj_R2")
        exported = image_bytes(image)
    graphics["02_matriz_coherencia_adj_R2.png"] = exported

    long_alpha = (
        alpha_df.reset_index(names="Activo")
        .melt(id_vars="Activo", var_name="Arquetipoide", value_name="Peso")
    )
    fig_archetypes = px.bar(
        long_alpha,
        x="Activo",
        y="Peso",
        color="Arquetipoide",
        barmode="stack",
        title="Composicion por arquetipoides",
    )
    fig_archetypes.update_layout(height=800, xaxis_tickangle=-60)
    exported = plotly_png_bytes(fig_archetypes, width=2200, height=1600, scale=2)
    if exported is None:
        image = Image.new("RGB", (2200, 1600), "white")
        draw = ImageDraw.Draw(image)
        shown_alpha = alpha_df.head(28)
        colors_archetypes = [
            (31, 119, 180),
            (214, 39, 40),
            (44, 160, 44),
            (148, 103, 189),
            (255, 127, 14),
        ]
        y = 80
        draw.text((80, y), "Composicion por arquetipoides", font=pil_font(34, bold=True), fill=(20, 20, 20))
        y += 70
        for asset, row in shown_alpha.iterrows():
            draw.text((95, y), str(asset)[:24], font=pil_font(22, bold=True), fill=(30, 30, 30))
            draw.rectangle([430, y + 4, 1330, y + 34], outline=(190, 190, 190))
            x_start = 430
            for idx, value in enumerate(row.values):
                segment_w = int(900 * float(value))
                draw.rectangle(
                    [x_start, y + 4, x_start + segment_w, y + 34],
                    fill=colors_archetypes[idx % len(colors_archetypes)],
                )
                x_start += segment_w
            y += 45
        exported = image_bytes(image)
    graphics["03_arquetipoides.png"] = exported

    fig_bi_assets = plot_biarchetype_assets(bi_asset_weights)
    exported = plotly_png_bytes(fig_bi_assets, width=2200, height=1600, scale=2)
    if exported is None:
        exported = image_bytes(Image.new("RGB", (2200, 1600), "white"))
    graphics["04_biarquetipos_activos.png"] = exported

    fig_bi_time = plot_biarchetype_time(bi_time_weights)
    exported = plotly_png_bytes(fig_bi_time, width=2200, height=1400, scale=2)
    if exported is None:
        exported = image_bytes(Image.new("RGB", (2200, 1400), "white"))
    graphics["05_biarquetipos_temporales.png"] = exported

    plot_data = data.iloc[:, :min(8, data.shape[1])].dropna(how="all")
    fig_series = px.line(plot_data, title="Retornos logarítmicos")
    exported = plotly_png_bytes(fig_series, width=2200, height=1600, scale=2)
    if exported is None:
        image = Image.new("RGB", (2200, 1600), "white")
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = 120, 130, 2080, 1250
        draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=2)
        if not plot_data.empty:
            colors_series = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14), (127, 127, 127), (23, 190, 207), (188, 189, 34)]
            min_v = float(plot_data.min().min())
            max_v = float(plot_data.max().max())
            span = max(max_v - min_v, 1e-9)
            for idx, col in enumerate(plot_data.columns):
                series = plot_data[col].interpolate().fillna(0)
                points = []
                for i, value in enumerate(series.values):
                    point_x = x0 + i / max(1, len(series) - 1) * (x1 - x0)
                    point_y = y1 - (float(value) - min_v) / span * (y1 - y0)
                    points.append((point_x, point_y))
                if len(points) >= 2:
                    draw.line(points, fill=colors_series[idx % len(colors_series)], width=4)
                draw.text((140, 1290 + idx * 30), str(col), font=pil_font(20), fill=colors_series[idx % len(colors_series)])
        exported = image_bytes(image)
    graphics["06_series_retornos.png"] = exported

    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in graphics.items():
            archive.writestr(name, content)
    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Datos")

data_dir = st.sidebar.text_input(
    "Carpeta con los Excels",
    value=default_data_dir()
)

bond_sheet = "Bond_raw_EU"
st.sidebar.caption("Bonos Eurostat unificados: historico completo con huecos gestionados por pares.")

st.sidebar.header("Wavelet")

min_scale = st.sidebar.slider("Escala mínima", 2, 24, 4)
max_scale = st.sidebar.slider("Escala máxima", 16, 128, 64)
n_scales = st.sidebar.slider("Número de escalas", 8, 48, 24)

st.sidebar.header("Modelos")

n_archetypes = st.sidebar.slider("Número de arquetipoides", 2, 5, 3)

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
summary_cols = st.columns(3)
summary_cols[0].metric("Observaciones mensuales", f"{data.shape[0]}")
summary_cols[1].metric("Activos disponibles", f"{data.shape[1]}")
summary_cols[2].metric("Periodo", f"{date_min} / {date_max}")

if data.empty:
    st.error(
        "El dataset está vacío. Revisa que las fechas de los Excels se solapen "
        "y que la hoja de bonos elegida sea correcta."
    )
    st.stop()

with st.expander("Ver datos de retornos logarítmicos"):
    st.dataframe(data.round(6))


asset_catalog = build_asset_catalog(data_dir, tuple(data.columns))
all_groups = list(asset_catalog["Grupo"].drop_duplicates())
preferred_groups = [
    "Tecnologia",
    "Crypto",
    "Divisas",
    "Farmaceutica y Defensa",
]
default_groups = [group for group in preferred_groups if group in all_groups]
if not default_groups:
    default_groups = all_groups[:min(3, len(all_groups))]

selected_groups = st.multiselect(
    "Filtra por grupos de activos",
    options=all_groups,
    default=default_groups,
)

filtered_assets = asset_catalog.loc[
    asset_catalog["Grupo"].isin(selected_groups),
    "Activo",
].tolist()

group_assets_signature = tuple(filtered_assets)
if st.session_state.get("group_assets_signature") != group_assets_signature:
    st.session_state["selected_assets"] = filtered_assets.copy()
    st.session_state["group_assets_signature"] = group_assets_signature

selected_assets = st.multiselect(
    "Selecciona activos",
    options=filtered_assets,
    key="selected_assets",
)

if len(selected_assets) < 3:
    st.warning("Selecciona al menos 3 activos.")
    st.stop()

month_options = sorted(pd.Index(data.index.strftime("%Y-%m")).unique().tolist())
selected_month_range = st.select_slider(
    "Rango temporal del analisis",
    options=month_options,
    value=(month_options[0], month_options[-1]),
)
start_month = pd.to_datetime(selected_month_range[0])
end_month = pd.to_datetime(selected_month_range[1])
data = data.loc[start_month:end_month]

if data.shape[0] < 12:
    st.warning("Selecciona al menos 12 meses para que el analisis sea estable.")
    st.stop()

with st.expander("Ver catalogo de activos"):
    st.dataframe(
        asset_catalog[asset_catalog["Activo"].isin(filtered_assets)],
        width="stretch",
    )

data = data[selected_assets].dropna(how="all")
valid_selected_assets = [asset for asset in data.columns if data[asset].notna().sum() >= max(24, n_scales)]
excluded_assets = [asset for asset in selected_assets if asset not in valid_selected_assets]
data = data[valid_selected_assets]

if len(valid_selected_assets) < 3:
    st.warning("Con este rango temporal hay menos de 3 activos con datos suficientes.")
    st.stop()

if excluded_assets:
    st.info(
        "Activos excluidos en este rango por datos insuficientes: "
        + ", ".join(excluded_assets)
    )

data_scaled = data.copy()

with st.expander("Configuracion del analisis"):
    st.write(
        {
            "activos_seleccionados": len(selected_assets),
            "activos_usados": len(valid_selected_assets),
            "rango_temporal": f"{selected_month_range[0]} a {selected_month_range[1]}",
            "observaciones_en_rango": int(data.shape[0]),
            "escala_minima": min_scale,
            "escala_maxima": max_scale,
            "numero_escalas": n_scales,
            "numero_arquetipoides": n_archetypes,
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


TAB_LABELS = [
    "1. Arquetipoides",
    "2. Biarquetipos",
    "3. K-Means",
    "4. Matrices",
    "5. Red",
    "6. Series",
    "7. Exportar"
]

if st.session_state.get("active_tab") not in TAB_LABELS:
    st.session_state["active_tab"] = TAB_LABELS[0]

active_tab = st.radio(
    "Vista",
    options=TAB_LABELS,
    horizontal=True,
    key="active_tab",
    label_visibility="collapsed",
)

alpha_df = fit_archetype_proxy(adj_r2, n_archetypes=n_archetypes)

if auto_k:
    best_k, scores_df = choose_best_k_silhouette(distance_matrix, max_k=max_k)
    k = best_k
else:
    scores_df = pd.DataFrame()
    k = manual_k

labels = fit_kmeans(distance_matrix, k)
cluster_df = pd.DataFrame({
    "Activo": distance_matrix.index,
    "Cluster": labels + 1
}).sort_values("Cluster")
resumen = cluster_df.groupby("Cluster")["Activo"].apply(list).reset_index()

bi_asset_weights, bi_time_weights, bi_time_distance = fit_biarchetype_proxy(
    data,
    distance_matrix,
    n_archetypes,
)

network_threshold = st.session_state.get("network_threshold", 0.42)
edges_df = build_network_edges(adj_r2, network_threshold)
degree = weighted_degree(edges_df, list(adj_r2.index))
node_summary = pd.DataFrame({
    "Activo": degree.index,
    "Sector": [classify_asset_sector(asset) for asset in degree.index],
    "Degree": degree.values,
}).sort_values("Degree", ascending=False)

download_signature = (
    st.session_state.get("analysis_signature"),
    float(st.session_state.get("network_threshold", 0.42)),
)
if st.session_state.get("graphics_zip_signature") != download_signature:
    st.session_state.pop("graphics_zip", None)

if st.sidebar.button("Descargar resultados"):
    with st.sidebar:
        with st.spinner("Preparando gráficos..."):
            st.session_state["graphics_zip"] = build_graphics_zip_bytes(
                data=data,
                adj_r2=adj_r2,
                distance_matrix=distance_matrix,
                alpha_df=alpha_df,
                bi_asset_weights=bi_asset_weights,
                bi_time_weights=bi_time_weights,
                cluster_df=cluster_df,
                network_threshold=float(st.session_state.get("network_threshold", 0.42)),
            )
            st.session_state["graphics_zip_signature"] = download_signature

if "graphics_zip" in st.session_state:
    st.sidebar.download_button(
        "Guardar ZIP",
        data=st.session_state["graphics_zip"],
        file_name="wavelet_graficos_resultados.zip",
        mime="application/zip",
    )


# ============================================================
# TAB 1: ARQUETIPOS
# ============================================================

if active_tab == "1. Arquetipoides":
    st.header("Grafico dinamico de arquetipoides")

    alpha_df = fit_archetype_proxy(adj_r2, n_archetypes=n_archetypes)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plot_archetypes(alpha_df)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Pesos α")
        st.dataframe(alpha_df.round(3))
        if "rss" in alpha_df.attrs:
            st.metric("RSS ADA", f"{alpha_df.attrs['rss']:.4f}")

        dominant_df = pd.DataFrame({
            "Activo": alpha_df.index,
            "Arquetipoide dominante": alpha_df.idxmax(axis=1),
            "Peso dominante": alpha_df.max(axis=1)
        }).sort_values("Arquetipoide dominante")

        st.subheader("Clasificación arquetípica")
        st.dataframe(dominant_df.round(3))


# ============================================================
# TAB 2: BIARQUETIPOS
# ============================================================

if active_tab == "2. Biarquetipos":
    st.header("Biarquetipos: activos y regimenes temporales")

    st.caption(
        "Biarchetype Analysis formal sobre la matriz activo-tiempo de retornos: "
        "se optimizan simultaneamente pesos convexos para activos y fechas."
    )

    col_assets, col_time = st.columns([1, 1])

    with col_assets:
        st.subheader("Pesos por activo")
        fig_bi_assets = plot_biarchetype_assets(bi_asset_weights)
        st.plotly_chart(fig_bi_assets, width="stretch")
        st.dataframe(bi_asset_weights.round(3), width="stretch")

    with col_time:
        st.subheader("Pesos por periodo")
        fig_bi_time = plot_biarchetype_time(bi_time_weights)
        st.plotly_chart(fig_bi_time, width="stretch")
        st.dataframe(bi_time_weights.round(3), width="stretch")

# ============================================================
# TAB 3: K-MEANS
# ============================================================

if active_tab == "3. K-Means":
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
# TAB 4: MATRICES
# ============================================================

if active_tab == "4. Matrices":
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
# TAB 5: RED
# ============================================================

if active_tab == "5. Red":
    st.header("Red de interdependencia wavelet")

    network_threshold = st.slider(
        "Umbral de coherencia para dibujar aristas",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("network_threshold", 0.42)),
        step=0.01,
        key="network_threshold",
        help=(
            "Replica el criterio de la Tercera Reunion: se dibuja una arista "
            "cuando adj_R2 supera el umbral."
        ),
    )

    fig_network, edges_df, degree = plot_wavelet_network(
        adj_r2,
        threshold=network_threshold,
    )

    total_possible_edges = len(adj_r2) * (len(adj_r2) - 1) // 2
    st.caption(
        f"Aristas visibles: {len(edges_df)} de {total_possible_edges} posibles "
        f"(solo adj_R2 > {network_threshold:.2f})."
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
        "aristas ponderadas por coherencia, flechas y nodos coloreados por grupo "
        "o sector."
    )


# ============================================================
# TAB 6: SERIES
# ============================================================

if active_tab == "6. Series":
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
# TAB 7: EXPORTAR
# ============================================================

if active_tab == "7. Exportar":
    st.header("Exportar resultados")

    export_sheets = {
        "retornos_log": data,
        "coherencia_adj_R2": adj_r2,
        "distancia_wavelet": distance_matrix,
        "pesos_arquetipoides": alpha_df,
        "biarquetipos_activos": bi_asset_weights,
        "biarquetipos_temporales": bi_time_weights,
        "clusters": cluster_df.set_index("Activo"),
        "resumen_clusters": resumen.set_index("Cluster"),
        "red_aristas": edges_df.set_index(["from", "to"]) if not edges_df.empty else edges_df,
        "red_nodos": node_summary.set_index("Activo"),
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
