import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict


def render_price_metrics(df: pd.DataFrame):
    """Render price metrics (assumes normalized columns with 'Close')."""
    try:
        if 'Close' not in df.columns:
            st.warning("No Close column found")
            return

        close_prices = df['Close'].dropna()

        if len(close_prices) < 2:
            st.warning("Not enough data points to calculate changes")
            return

        latest_close = close_prices.iloc[-1]
        prev_close = close_prices.iloc[-2]

        change = latest_close - prev_close
        pct_change = 100.0 * change / prev_close if prev_close != 0 else 0.0

        cols = st.columns(4)
        cols[0].metric("Last Close", f"${latest_close:.2f}", f"${change:.2f}")
        cols[1].metric("Change (%)", f"{pct_change:.2f}%")
        cols[2].metric("52W High", f"${close_prices.max():.2f}")
        cols[3].metric("52W Low", f"${close_prices.min():.2f}")

    except Exception as e:
        st.error(f"Error rendering price metrics: {str(e)}")


def calculate_performance_metrics(results: Dict) -> Dict:
    """Calculate key performance metrics from simulation results."""
    portfolio_values = results['portfolio_values']

    if len(portfolio_values) < 2:
        return {
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'volatility_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate_pct': 0.0,
            'final_value': portfolio_values[0] if portfolio_values else 0.0
        }

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    annualized_return = (1 + total_return / 100) ** (252 / len(portfolio_values)) - 1 if len(portfolio_values) > 0 else 0

    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0

    positive_returns = len([r for r in returns if r > 0])
    win_rate = positive_returns / len(returns) if len(returns) > 0 else 0

    return {
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return * 100,
        'volatility_pct': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate * 100,
        'final_value': portfolio_values[-1],
        'total_trades': len([a for a in results['actions'] if a != 0])
    }
