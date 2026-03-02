# src/viz/plots.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from datetime import datetime

def _ensure_datetime(dates):
    """Convert input to datetime objects safely"""
    if dates is None:
        return None
    
    if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
        return pd.to_datetime(dates)
    
    if isinstance(dates, (list, tuple, np.ndarray)):
        # Check if first element is already datetime
        if len(dates) > 0 and isinstance(dates[0], (datetime, pd.Timestamp)):
            return pd.to_datetime(dates)
        else:
            # Create sequential dates if not datetime
            return pd.date_range(start='2020-01-01', periods=len(dates), freq='D')
    
    return pd.to_datetime(dates)

def _flatten_series_or_df(y):
    """Ensure y is always a 1D array/Series."""
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0]  # first column as Series
    elif isinstance(y, np.ndarray) and y.ndim > 1:
        return y.ravel()
    elif isinstance(y, list):
        return np.array(y)
    return y

def _get_dates_from_data(data: Any, default_length: int = None):
    """Extract dates from various data formats"""
    if isinstance(data, pd.DataFrame):
        if 'Date' in data.columns:
            return pd.to_datetime(data['Date'])
        elif 'date' in data.columns:
            return pd.to_datetime(data['date'])
        elif data.index.name in ['Date', 'date']:
            return pd.to_datetime(data.index)
        elif hasattr(data.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(data.index):
            return data.index
        else:
            # Create sequential dates if no dates available
            length = len(data) if default_length is None else default_length
            return pd.date_range(start='2020-01-01', periods=length, freq='D')
    
    elif isinstance(data, dict) and 'date' in data:
        return _ensure_datetime(data['date'])
    
    elif default_length is not None:
        return pd.date_range(start='2020-01-01', periods=default_length, freq='D')
    
    return None

def _extract_price_data(df: pd.DataFrame):
    """Extract price data from DataFrame with various column names"""
    price_columns = ['Close', 'close', 'price', 'last', 'Last']
    for col in price_columns:
        if col in df.columns:
            return df[col]
    
    # If no standard price column, try to find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols[0]]
    
    # Last resort: use first column
    return df.iloc[:, 0]

def plot_price(df: pd.DataFrame, title: str = "Price"):
    """Plot price with proper date handling"""
    dates = _get_dates_from_data(df)
    y = _extract_price_data(df)
    y = _flatten_series_or_df(y)

    # Ensure dates and y have same length
    min_length = min(len(dates), len(y))
    dates = dates[:min_length]
    y = y[:min_length]

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
        # Create empty figure
        fig = go.Figure()
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value ($)')
        return fig
    
    portfolio_values = history['portfolio_valuation']
    
    # Try to get dates from various sources
    dates = None
    
    # Option 1: Check if history has dates
    if 'date' in history and len(history['date']) == len(portfolio_values):
        dates = _ensure_datetime(history['date'])
    
    # Option 2: Get dates from price dataframe
    if dates is None:
        dates = _get_dates_from_data(price_df, len(portfolio_values))
    
    # Option 3: Create sequential dates
    if dates is None or len(dates) != len(portfolio_values):
        dates = pd.date_range(start='2020-01-01', periods=len(portfolio_values), freq='D')
    
    # Ensure we don't exceed available data
    min_length = min(len(portfolio_values), len(dates))
    portfolio_values = portfolio_values[:min_length]
    dates = dates[:min_length]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=portfolio_values, 
        mode='lines', 
        name='Portfolio Value',
        line=dict(color='green', width=2)
    ))
    
    # Add initial balance line for reference
    if len(portfolio_values) > 0:
        initial_balance = portfolio_values[0]
        fig.add_trace(go.Scatter(
            x=dates,
            y=[initial_balance] * len(dates),
            mode='lines',
            name='Initial Balance',
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title=title, 
        xaxis_title='Date', 
        yaxis_title='Value ($)',
        hovermode='x unified',
        showlegend=True
    )
    return fig

def plot_trades_on_price(price_df: pd.DataFrame, history: dict, title="Trades on Price"):
    """Plot trades on price chart with correct date alignment"""
    # Create base price plot
    dates = _get_dates_from_data(price_df)
    y = _extract_price_data(price_df)
    y = _flatten_series_or_df(y)
    
    # Ensure consistent lengths
    min_length = min(len(dates), len(y))
    dates = dates[:min_length]
    y = y[:min_length]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=y, 
        mode='lines', 
        name='Price',
        line=dict(color='blue', width=1),
        opacity=0.7
    ))
    
    if not history or 'position' not in history:
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price ($)')
        return fig
    
    positions = history['position']
    trade_prices = history.get('price', [])
    
    # Get dates for trades
    trade_dates = None
    if 'date' in history and len(history['date']) == len(positions):
        trade_dates = _ensure_datetime(history['date'])
    else:
        trade_dates = _get_dates_from_data(price_df, len(positions))
    
    # Ensure consistent lengths and valid data
    min_length = min(len(positions), len(trade_dates), len(trade_prices) if trade_prices else len(positions))
    positions = positions[:min_length]
    trade_dates = trade_dates[:min_length] if trade_dates is not None else dates[:min_length]
    
    if trade_prices:
        trade_prices = trade_prices[:min_length]
    else:
        # Use price data if trade prices not available
        trade_prices = y[:min_length] if len(y) >= min_length else [0] * min_length
    
    # Find trade entry points
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    close_dates, close_prices = [], []
    
    for i in range(1, min_length):
        if positions[i] != positions[i-1]:  # Position changed
            current_date = trade_dates[i] if i < len(trade_dates) else dates[i] if i < len(dates) else None
            current_price = trade_prices[i] if i < len(trade_prices) else y[i] if i < len(y) else 0
            
            if current_date is None:
                continue
                
            if positions[i] == 1:  # Entered long
                buy_dates.append(current_date)
                buy_prices.append(current_price)
            elif positions[i] == -1:  # Entered short
                sell_dates.append(current_date)
                sell_prices.append(current_price)
            elif positions[i] == 0 and positions[i-1] != 0:  # Closed position
                close_dates.append(current_date)
                close_prices.append(current_price)
    
    # Add trade markers
    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices, mode='markers',
            marker_symbol='triangle-up', marker_size=10,
            marker_color='green', name='Buy',
            hovertemplate='Buy: %{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>'
        ))
    
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices, mode='markers',
            marker_symbol='triangle-down', marker_size=10,
            marker_color='red', name='Sell/Short',
            hovertemplate='Sell/Short: %{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>'
        ))
    
    if close_dates:
        fig.add_trace(go.Scatter(
            x=close_dates, y=close_prices, mode='markers',
            marker_symbol='circle', marker_size=8,
            marker_color='orange', name='Close Position',
            hovertemplate='Close: %{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        showlegend=True
    )
    return fig

def plot_agent_performance(history: dict, title: str = "Agent Performance"):
    """Comprehensive performance visualization"""
    if not history:
        return go.Figure()
    
    fig = go.Figure()
    
    # Portfolio value
    if 'portfolio_valuation' in history:
        dates = _get_dates_from_data(history, len(history['portfolio_valuation']))
        portfolio_values = history['portfolio_valuation']
        min_length = min(len(dates), len(portfolio_values))
        fig.add_trace(go.Scatter(
            x=dates[:min_length], y=portfolio_values[:min_length], 
            mode='lines', name='Portfolio Value', line=dict(color='green')
        ))
    
    # Cash balance
    if 'cash_balance' in history:
        dates = _get_dates_from_data(history, len(history['cash_balance']))
        cash_balances = history['cash_balance']
        min_length = min(len(dates), len(cash_balances))
        fig.add_trace(go.Scatter(
            x=dates[:min_length], y=cash_balances[:min_length], 
            mode='lines', name='Cash Balance', line=dict(color='blue')
        ))
    
    # Stock value
    if 'stock_value' in history:
        dates = _get_dates_from_data(history, len(history['stock_value']))
        stock_values = history['stock_value']
        min_length = min(len(dates), len(stock_values))
        fig.add_trace(go.Scatter(
            x=dates[:min_length], y=stock_values[:min_length], 
            mode='lines', name='Stock Value', line=dict(color='orange')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='x unified'
    )
    return fig

def plot_position_changes(history: dict, title: str = "Position Changes"):
    """Plot position changes over time"""
    if not history or 'position' not in history:
        return go.Figure()
    
    positions = history['position']
    dates = _get_dates_from_data(history, len(positions))
    
    min_length = min(len(dates), len(positions))
    dates = dates[:min_length]
    positions = positions[:min_length]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=positions, 
        mode='lines+markers', name='Position',
        line=dict(color='purple'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Position',
        yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Short', 'Neutral', 'Long']),
        hovermode='x unified'
    )
    return fig