from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Datos"
OUT_FILE = DATA_DIR / "Yahoo_Assets.xlsx"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".yfinance_cache"

START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()


PAPER_ASSETS = [
    ("Paper", "GOLD", "GC=F", "Conventional"),
    ("Paper", "GRNENEF", "ICLN", "Green Assets"),
    ("Paper", "GRNFUEL", "PLUG", "Green Assets"),
    ("Paper", "GRNGB", "VNQ", "Green Assets"),
    ("Paper", "GRNPOL", "PBD", "Green Assets"),
    ("Paper", "GRNSOLAR", "TAN", "Green Assets"),
    ("Paper", "GRNTRN", "DRIV", "Green Assets"),
    ("Paper", "GRNWIND", "FAN", "Green Assets"),
    ("Paper", "HYBOND", "HYG", "Conventional"),
    ("Paper", "IGBOND", "LQD", "Conventional"),
    ("Paper", "MSCI", "URTH", "Conventional"),
    ("Paper", "OIL", "CL=F", "Conventional"),
    ("Paper", "SPGBI", "BGRN", "Green Assets"),
    ("Paper", "TREAS", "IEF", "Conventional"),
]


EXTRA_ASSETS = [
    ("Tecnologia", "Nvidia", "NVDA", "Tecnologia"),
    ("Tecnologia", "Microsoft", "MSFT", "Tecnologia"),
    ("Tecnologia", "Apple", "AAPL", "Tecnologia"),
    ("Tecnologia", "Google", "GOOGL", "Tecnologia"),
    ("Tecnologia", "Tesla", "TSLA", "Tecnologia"),
    ("Crypto", "Bitcoin", "BTC-USD", "Crypto"),
    ("Crypto", "Ethereum", "ETH-USD", "Crypto"),
    ("Crypto", "Solana", "SOL-USD", "Crypto"),
    ("Crypto", "Binance", "BNB-USD", "Crypto"),
    ("Energia", "Exxon", "XOM", "Energia"),
    ("Energia", "Chevron", "CVX", "Energia"),
    ("Industrial", "Boeing", "BA", "Industrial"),
    ("Industrial", "Caterpillar", "CAT", "Industrial"),
    ("Farmaceutica y Defensa", "EliLilly", "LLY", "Farmaceutica y Defensa"),
    ("Farmaceutica y Defensa", "J&J", "JNJ", "Farmaceutica y Defensa"),
    ("Farmaceutica y Defensa", "Pfizer", "PFE", "Farmaceutica y Defensa"),
    ("Farmaceutica y Defensa", "Merck", "MRK", "Farmaceutica y Defensa"),
    ("Farmaceutica y Defensa", "AbbVie", "ABBV", "Farmaceutica y Defensa"),
    ("Indices", "S&P500", "SPY", "Indices"),
    ("Indices", "Nasdaq", "QQQ", "Indices"),
    ("Indices", "VIX", "^VIX", "Indices"),
    ("Indices", "ORO", "GLD", "Commodities"),
    ("Indices", "Bonos20A", "TLT", "Bonos"),
    ("Divisas", "Dolar_IDX", "UUP", "Divisas"),
    ("Divisas", "Euro", "EUR=X", "Divisas"),
    ("Divisas", "Libra", "GBP=X", "Divisas"),
    ("Divisas", "Yen", "JPY=X", "Divisas"),
    ("Divisas", "FrancoSuizo", "CHF=X", "Divisas"),
    ("Divisas", "AudDolar", "AUD=X", "Divisas"),
]


def build_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        PAPER_ASSETS + EXTRA_ASSETS,
        columns=["Group", "Label", "Yahoo_Ticker", "Sector"],
    ).drop_duplicates(subset=["Label"])


def download_adjusted_close(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = {}
    failures = []

    for row in metadata.itertuples(index=False):
        try:
            hist = yf.download(
                row.Yahoo_Ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if hist.empty:
                failures.append((row.Label, row.Yahoo_Ticker, "empty"))
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                if ("Close", row.Yahoo_Ticker) in hist.columns:
                    series = hist[("Close", row.Yahoo_Ticker)]
                elif ("Adj Close", row.Yahoo_Ticker) in hist.columns:
                    series = hist[("Adj Close", row.Yahoo_Ticker)]
                else:
                    failures.append((row.Label, row.Yahoo_Ticker, "missing close column"))
                    continue
            else:
                price_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
                series = hist[price_col]

            series = pd.to_numeric(series.squeeze(), errors="coerce").dropna()
            prices[row.Label] = series
        except Exception as exc:
            failures.append((row.Label, row.Yahoo_Ticker, str(exc)))

    if not prices:
        raise RuntimeError("No se ha podido descargar ningun activo desde Yahoo Finance.")

    price_df = pd.concat(prices, axis=1).sort_index()
    price_df.index.name = "Date"
    failures_df = pd.DataFrame(failures, columns=["Label", "Yahoo_Ticker", "Reason"])

    return price_df, failures_df


def to_monthly_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    monthly = price_df.resample("ME").last().dropna(how="all")
    monthly.index = monthly.index.to_period("M").to_timestamp()
    monthly.index.name = "Date"
    return monthly


def to_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(price_df / price_df.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns.index.name = "Date"
    return returns


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR))

    metadata = build_metadata()
    daily_prices, failures = download_adjusted_close(metadata)
    monthly_prices = to_monthly_prices(daily_prices)
    monthly_returns = to_log_returns(monthly_prices)

    downloaded_metadata = metadata[metadata["Label"].isin(daily_prices.columns)].copy()

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        downloaded_metadata.to_excel(writer, sheet_name="Metadata", index=False)
        daily_prices.to_excel(writer, sheet_name="Daily_Prices")
        monthly_prices.to_excel(writer, sheet_name="Monthly_Prices")
        monthly_returns.to_excel(writer, sheet_name="Monthly_Log_Returns")
        failures.to_excel(writer, sheet_name="Failures", index=False)

    print(f"Guardado: {OUT_FILE}")
    print(f"Activos descargados: {daily_prices.shape[1]}")
    print(f"Fechas diarias: {daily_prices.shape[0]}")
    if not failures.empty:
        print("Fallos:")
        print(failures.to_string(index=False))


if __name__ == "__main__":
    main()
