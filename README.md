Project Structure

1
stock-rl-dashboard/
├─ requirements.txt
├─ README.md
├─ streamlit_app.py                # main Streamlit UI
├─ src/
│  ├─ __init__.py
│  ├─ data_loader.py               # yfinance wrapper + ticker lookup/search
│  ├─ envs/
│  │   ├─ __init__.py
│  │   ├─ trading_env_wrapper.py   # wraps gym_trading_env.TradingEnv or a fallback env
│  ├─ agents/
│  │   ├─ __init__.py
│  │   ├─ base.py                  # abstract agent interface
│  │   ├─ random_agent.py
│  │   ├─ sma_agent.py
│  │   ├─ trained_agent.py         # load/predict interface for trained RL agent
│  ├─ viz/
│  │   ├─ __init__.py
│  │   ├─ plots.py                 # plotly charts and summary KPI helpers
│  └─ utils.py                     # small helpers (formatting, saving)
└─ models/                         # saved RL models, preprocessing, scalers