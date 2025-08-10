# src/data_loader.py
import pandas as pd
import yfinance as yf
from typing import Optional

TICKER_CSV_NASDAQ = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.csv"
TICKER_CSV_NYSE = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.csv"
TICKER_CSV_AMEX = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.csv"

def get_all_tickers() -> pd.DataFrame:
    """Try to fetch ticker lists; fall back to a small default set."""
    try:
        nasdaq = pd.read_csv(TICKER_CSV_NASDAQ)
        nyse = pd.read_csv(TICKER_CSV_NYSE)
        amex = pd.read_csv(TICKER_CSV_AMEX)
        all_tickers = pd.concat([nasdaq, nyse, amex]).drop_duplicates().reset_index(drop=True)
        if 'Symbol' in all_tickers.columns and 'Name' in all_tickers.columns:
            return all_tickers[['Symbol', 'Name']].sort_values('Symbol').reset_index(drop=True)
        # try to be flexible if column names differ
        cols = all_tickers.columns.tolist()
        return pd.DataFrame({'Symbol': all_tickers[cols[0]], 'Name': all_tickers[cols[1]]})
    except Exception:
        # fallback
        fallback = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT'],
            'Name': ['Apple', 'Microsoft', 'Alphabet', 'Amazon', 'Tesla', 'Meta', 'Nvidia', 'JPMorgan', 'Visa', 'Walmart']
        })
        return fallback

def search_tickers(query: str, ticker_df: pd.DataFrame) -> pd.DataFrame:
    q = query.strip()
    if q == "":
        return ticker_df
    return ticker_df[
        ticker_df['Symbol'].str.contains(q.upper(), na=False) |
        ticker_df['Name'].str.contains(q.title(), na=False)
    ]

def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download price history via yfinance. Returns DataFrame with Open/High/Low/Close/Volume."""
    ticker = ticker.upper()
    print(f"Downloading {ticker} period={period} interval={interval}")
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} with period={period} interval={interval}")
    df = df.reset_index()
    return df
