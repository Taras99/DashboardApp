# src/data_loader.py
from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, List
import numpy as np


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame from yfinance into a flat, consistent format.

    - Flattens MultiIndex columns (e.g. ('Close', 'AAPL') -> 'Close')
    - Ensures a 'Date' column exists (from index or column)
    - Standardizes column names to Open, High, Low, Close, Volume
    """
    df = df.copy()

    # 1. Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # 2. Ensure Date column exists
    if 'Date' not in df.columns and 'date' not in df.columns:
        if df.index.name in ('Date', 'date'):
            df = df.reset_index()
        elif hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index):
            df.index.name = 'Date'
            df = df.reset_index()
        # Handle 'index' column that may come from a prior reset_index()
        if 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
            df = df.rename(columns={'index': 'Date'})

    # Normalize 'date' -> 'Date'
    if 'date' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'date': 'Date'})

    # 3. Standardize column names
    canonical = {'open': 'Open', 'high': 'High', 'low': 'Low',
                 'close': 'Close', 'volume': 'Volume', 'date': 'Date'}
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in canonical and col != canonical[lower]:
            rename_map[col] = canonical[lower]
    if rename_map:
        df = df.rename(columns=rename_map)

    # Remove duplicate columns that may result from flattening
    df = df.loc[:, ~df.columns.duplicated()]

    return df


class ITickerProvider(ABC):
    """Interface for ticker providers"""
    @abstractmethod
    def get_all_tickers(self) -> pd.DataFrame:
        """Get all available tickers"""
        pass
    
    @abstractmethod
    def search_tickers(self, query: str) -> pd.DataFrame:
        """Search tickers by query"""
        pass


class IStockDataProvider(ABC):
    """Interface for stock data providers"""
    @abstractmethod
    def get_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Get stock data for given ticker"""
        pass
    
    @abstractmethod
    def get_stock_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics from stock data"""
        pass


class YahooFinanceTickerProvider(ITickerProvider):
    """Yahoo Finance implementation of ticker provider"""
    
    TICKER_SOURCES = {
        'nasdaq': "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.csv",
        'nyse': "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.csv",
        'amex': "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.csv"
    }
    
    FALLBACK_TICKERS = [
        ('AAPL', 'Apple'), ('MSFT', 'Microsoft'), ('GOOGL', 'Alphabet'),
        ('AMZN', 'Amazon'), ('TSLA', 'Tesla'), ('META', 'Meta'),
        ('NVDA', 'Nvidia'), ('JPM', 'JPMorgan'), ('V', 'Visa'), ('WMT', 'Walmart')
    ]

    def get_all_tickers(self) -> pd.DataFrame:
        """Fetch ticker lists from all exchanges with fallback"""
        try:
            dfs = []
            for exchange, url in self.TICKER_SOURCES.items():
                df = pd.read_csv(url)
                if len(df.columns) >= 2:
                    df.columns = ['Symbol', 'Name'] + list(df.columns[2:])
                dfs.append(df[['Symbol', 'Name']])
            
            combined = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
            return combined.sort_values('Symbol').reset_index(drop=True)
        
        except Exception as e:
            print(f"Error fetching tickers: {e}, using fallback")
            return pd.DataFrame(self.FALLBACK_TICKERS, columns=['Symbol', 'Name'])

    def search_tickers(self, query: str) -> pd.DataFrame:
        """Search tickers by symbol or name"""
        all_tickers = self.get_all_tickers()
        query = query.strip().lower()
        
        if not query:
            return all_tickers
        
        mask = (
            all_tickers['Symbol'].str.lower().str.contains(query) |
            all_tickers['Name'].str.lower().str.contains(query)
        )
        return all_tickers[mask].reset_index(drop=True)


class YahooFinanceDataProvider(IStockDataProvider):
    """Yahoo Finance implementation of stock data provider"""
    
    def get_stock_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Download price history via yfinance"""
        ticker = ticker.upper()
        print(f"Downloading {ticker} period={period} interval={interval}")
        
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                raise ValueError(f"No data for {ticker} with period={period} interval={interval}")

            return normalize_dataframe(df.reset_index())
        
        except Exception as e:
            raise ValueError(f"Failed to download data: {str(e)}")

    def get_stock_statistics(self, data: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Calculate statistics for specified column"""
        if column not in data.columns:
            raise ValueError(f"DataFrame must contain '{column}' column")
        
        series = data[column].dropna()
        mode_values = series.mode()
        
        stats = {
            "mean": series.mean(),
            "median": series.median(),
            "mode": mode_values.iloc[0] if not mode_values.empty else np.nan,
            "min": series.min(),
            "max": series.max(),
            "std_dev": series.std(),
            "variance": series.var(),
            "count": series.count()
        }
        
        return pd.DataFrame([stats])


class StockDataService:
    """Facade for stock data operations"""
    
    def __init__(
        self,
        ticker_provider: Optional[ITickerProvider] = None,
        data_provider: Optional[IStockDataProvider] = None
    ):
        self.ticker_provider = ticker_provider or YahooFinanceTickerProvider()
        self.data_provider = data_provider or YahooFinanceDataProvider()
    
    def get_all_tickers(self) -> pd.DataFrame:
        """Get all available tickers"""
        return self.ticker_provider.get_all_tickers()
    
    def search_tickers(self, query: str) -> pd.DataFrame:
        """Search tickers by query"""
        return self.ticker_provider.search_tickers(query)
    
    def get_stock_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get stock data for given ticker"""
        return self.data_provider.get_stock_data(ticker, period, interval)
    
    def get_stock_statistics(self, data: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Calculate statistics from stock data"""
        return self.data_provider.get_stock_statistics(data, column)
    
    def get_stock_data_with_stats(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        column: str = 'Close'
    ) -> Dict[str, pd.DataFrame]:
        """Get both data and statistics"""
        data = self.get_stock_data(ticker, period, interval)
        stats = self.get_stock_statistics(data, column)
        return {'data': data, 'stats': stats}