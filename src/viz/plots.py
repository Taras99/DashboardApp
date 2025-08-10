# src/viz/plots.py
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

def plot_price(df: pd.DataFrame, title: str = "Price"):
    # Determine x-axis
    if 'Date' in df.columns:
        x = df['Date']
    elif 'date' in df.columns:
        x = df['date']
    else:
        x = df.index

    # Determine y-axis and flatten
    if 'Close' in df.columns:
        y = df['Close']
    else:
        y = df.get('price', df.iloc[:, -1])
    y = _flatten_series_or_df(y)

    fig = px.line(x=x, y=y, labels={'x': 'Date', 'y': 'Price'}, title=title)
    return fig

def plot_portfolio_history(history: dict, title: str = "Portfolio Value"):
    steps = history.get('step', list(range(len(history.get('portfolio_valuation', [])))))
    vals = _flatten_series_or_df(history.get('portfolio_valuation', []))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=vals, mode='lines+markers', name='Portfolio'))
    fig.update_layout(title=title, xaxis_title='Step', yaxis_title='Value')
    return fig

def plot_trades_on_price(df: pd.DataFrame, history: dict, title="Trades"):
    fig = plot_price(df, title=title)
    if history:
        steps = history.get('step', [])
        prices = _flatten_series_or_df(history.get('price', []))
        positions = history.get('position', [])

        buys_x = [steps[i] for i in range(len(positions)) if positions[i] == 1]
        buys_y = [prices[i] for i in range(len(positions)) if positions[i] == 1]
        sells_x = [steps[i] for i in range(len(positions)) if positions[i] == -1]
        sells_y = [prices[i] for i in range(len(positions)) if positions[i] == -1]

        if buys_x:
            fig.add_trace(go.Scatter(
                x=buys_x, y=buys_y, mode='markers',
                marker_symbol='triangle-up', name='Long', marker=dict(size=10)
            ))
        if sells_x:
            fig.add_trace(go.Scatter(
                x=sells_x, y=sells_y, mode='markers',
                marker_symbol='triangle-down', name='Short', marker=dict(size=10)
            ))
    return fig
