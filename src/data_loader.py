"""
data_loader.py
Fetches SPY and VIX daily data from yfinance and saves to data/raw/.
"""

import os
import yfinance as yf
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance and lowercase names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    return df


def fetch_spy(start: str = "2015-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """Download SPY daily OHLCV data from yfinance."""
    print(f"Fetching SPY data from {start} to {end}...")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if spy.empty:
        raise RuntimeError("SPY download returned empty data. Check your internet connection.")
    spy = _flatten_columns(spy)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy.index.name = "date"
    return spy


def fetch_vix(start: str = "2015-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """Download VIX daily close data from yfinance."""
    print(f"Fetching VIX data from {start} to {end}...")
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if vix.empty:
        raise RuntimeError("VIX download returned empty data. Check your internet connection.")
    vix = _flatten_columns(vix)
    vix = vix[["close"]].rename(columns={"close": "vix"})
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.index.name = "date"
    return vix


def load_data(start: str = "2015-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """
    Fetch SPY and VIX, merge on date, and return a single DataFrame.
    Saves raw CSVs to data/raw/ for reproducibility.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    spy = fetch_spy(start, end)
    vix = fetch_vix(start, end)

    # Save raw data
    spy.to_csv(os.path.join(DATA_DIR, "spy_raw.csv"))
    vix.to_csv(os.path.join(DATA_DIR, "vix_raw.csv"))
    print(f"Raw data saved to {DATA_DIR}")

    # Merge on date index — inner join keeps only trading days present in both
    df = spy.join(vix, how="inner")

    print(f"Merged dataset: {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.tail())
