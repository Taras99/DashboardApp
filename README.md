# Stock RL Trading Dashboard

An interactive Streamlit dashboard for stock market analysis and algorithmic trading simulation. Load real stock data, configure trading agents, run simulations, and visualize performance metrics.

## Features

- **Live stock data** — fetch OHLCV data from Yahoo Finance for any ticker
- **Multiple trading agents** — choose from rule-based and statistical agents
- **Configurable simulation** — set initial balance, trading fees, position size, shorting, and more
- **Performance metrics** — total return, annualized return, Sharpe ratio, max drawdown, win rate
- **Interactive charts** — portfolio value history and trade entries/exits on price chart

## Project Structure

```
stock-rl-dashboard/
├── streamlit_app.py              # Main Streamlit entry point
├── requirements.txt
├── src/
│   ├── data_loader.py            # yfinance wrapper, ticker search, data normalization
│   ├── agents/
│   │   ├── base.py               # Abstract BaseAgent interface
│   │   ├── statistical_agent.py  # StatisticalMeanAgent, ContinuousTradingAgent
│   │   └── lstm_agent.py         # LSTM-SMA hybrid agent
│   ├── envs/
│   │   └── trading_env_wrapper.py  # Trading environment (gym-compatible)
│   ├── viz/
│   │   └── plots.py              # Plotly charts
│   ├── ui/
│   │   ├── sidebar.py            # Ticker selection and data loading
│   │   ├── simulation.py         # Agent config and simulation runner
│   │   ├── results.py            # Results visualization
│   │   └── metrics.py            # Performance metric calculations
│   └── models/                   # Saved models and scalers
```

## Installation

```bash
pip install -r requirements.txt
```

For LSTM agent support, also install PyTorch:
```bash
pip install torch
```

## Running the App

```bash
python -m streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

## Agents

| Agent | Description |
|---|---|
| **ContinuousTradingAgent** | Rule-based agent using configurable long/short thresholds, stop-loss, and min-profit targets |
| **StatisticalMeanAgent** | Mean-reversion agent using cross-validated statistical thresholds |
| **LSTM-SMA** | Hybrid agent combining LSTM price forecasting with SMA crossover signals |

## Simulation Parameters

| Parameter | Description |
|---|---|
| Initial balance | Starting portfolio cash |
| Trading fee | Fraction of trade value charged as fee |
| Max position size | Maximum % of portfolio in a single position |
| Allow shorting | Enable short selling |
| Interest rate | Daily interest rate applied to short positions |
