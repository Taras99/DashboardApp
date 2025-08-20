import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def _flatten_series_or_df(y):
    """Ensure y is always a 1D array/Series."""
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0]  # first column as Series
    elif isinstance(y, np.ndarray) and y.ndim > 1:
        return y.ravel()
    return y

def _get_dates_from_df(df: pd.DataFrame):
    """Extract dates from DataFrame with proper handling"""
    if 'Date' in df.columns:
        return pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        return pd.to_datetime(df['date'])
    elif df.index.name == 'Date' or df.index.name == 'date':
        return pd.to_datetime(df.index)
    else:
        # Create sequential business days if no dates available
        return pd.date_range(start='2020-01-01', periods=len(df), freq='B')

def plot_price(df: pd.DataFrame, title: str = "Price"):
    """Plot price with proper date handling"""
    dates = _get_dates_from_df(df)
    
    # Determine y-axis and flatten
    if 'Close' in df.columns:
        y = df['Close']
    else:
        y = df.get('price', df.iloc[:, -1])
    y = _flatten_series_or_df(y)

    # Use graph_objects for consistent API
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=y, 
        mode='lines', 
        name='Price',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified'
    )
    return fig

def plot_portfolio_history(history: dict, price_df: pd.DataFrame, title: str = "Portfolio Value"):
    """Plot portfolio value with correct dates aligned to price data"""
    if not history or 'portfolio_valuation' not in history:
        return go.Figure()
    
    # Get dates from price data
    dates = _get_dates_from_df(price_df)
    
    # Ensure we don't exceed available dates
    min_length = min(len(history['portfolio_valuation']), len(dates))
    portfolio_values = history['portfolio_valuation'][:min_length]
    valid_dates = dates[:min_length]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid_dates, 
        y=portfolio_values, 
        mode='lines', 
        name='Portfolio',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title=title, 
        xaxis_title='Date', 
        yaxis_title='Value ($)',
        hovermode='x unified'
    )
    return fig

def plot_trades_on_price(df: pd.DataFrame, history: dict, title="Trades"):
    """Plot trades on price chart with correct date alignment"""
    # Create base price plot using graph_objects
    dates = _get_dates_from_df(df)
    
    if 'Close' in df.columns:
        y = df['Close']
    else:
        y = df.get('price', df.iloc[:, -1])
    y = _flatten_series_or_df(y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=y, 
        mode='lines', 
        name='Price',
        line=dict(color='blue')
    ))
    
    if not history or 'position' not in history or 'price' not in history:
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price ($)')
        return fig
    
    # Ensure we don't exceed available data
    min_length = min(len(history['position']), len(dates), len(history['price']))
    positions = history['position'][:min_length]
    prices = history['price'][:min_length]
    valid_dates = dates[:min_length]
    
    # Find trade entry points (where position changes)
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    for i in range(1, min_length):
        if positions[i] != positions[i-1]:  # Position changed
            if positions[i] == 1:  # Entered long
                buy_dates.append(valid_dates[i])
                buy_prices.append(prices[i])
            elif positions[i] == -1:  # Entered short
                sell_dates.append(valid_dates[i])
                sell_prices.append(prices[i])
    
    # Add trade markers
    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices, mode='markers',
            marker_symbol='triangle-up', marker_size=12,
            marker_color='green', name='Buy Entry',
            hovertemplate='Buy: %{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>'
        ))
    
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices, mode='markers',
            marker_symbol='triangle-down', marker_size=12,
            marker_color='red', name='Sell Entry',
            hovertemplate='Sell: %{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified'
    )
    return fig