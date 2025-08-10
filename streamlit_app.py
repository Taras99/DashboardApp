# streamlit_app.py
import streamlit as st
import pandas as pd
from src.data_loader import get_all_tickers, search_tickers, get_stock_data
from src.envs.trading_env_wrapper import create_trading_env_from_df
#from src.agents.random_agent import RandomAgent
from src.agents.sma_agent import SMAAgent
from src.viz.plots import plot_price, plot_portfolio_history, plot_trades_on_price
from src.utils import ensure_dir
import os

st.set_page_config(page_title="Stock RL Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Stock analysis & RL trading dashboard")

# Sidebar: ticker selection
st.sidebar.header("Data & Ticker")
tickers_df = get_all_tickers()
query = st.sidebar.text_input("Search tickers by symbol or name", "")
results = search_tickers(query, tickers_df)
selected = st.sidebar.selectbox("Select ticker", results['Symbol'].unique())
period = st.sidebar.selectbox("Period", ['1mo','3mo','6mo','1y','5y','10y','max'], index=3)
interval = st.sidebar.selectbox("Interval", ['1d','1wk','1mo','60m','30m','15m','5m'], index=0)

if st.sidebar.button("Load data"):
    try:
        df = get_stock_data(selected, period=period, interval=interval)
        st.session_state['df'] = df
        st.success(f"Loaded {len(df)} rows for {selected}")
    except Exception as e:
        st.error(f"Failed to load data: {e}")

if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader(f"{selected} price")
    fig_price = plot_price(df)
    st.plotly_chart(fig_price, use_container_width=True)

    # quick stats
    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
    change = latest_close - prev_close
    pct_change = 100.0 * change / prev_close if prev_close != 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close", f"${latest_close:.2f}", f"${change:.2f}")
    col2.metric("Change (%)", f"{pct_change:.2f}%")
    col3.metric("Rows", f"{len(df)}")

    st.markdown("---")
    st.subheader("Run simulation")
    initial_balance = st.number_input("Initial balance", value=10000.0, step=1000.0)
    trading_fees = st.number_input("Trading fee (fraction)", value=0.001, step=0.0001, format="%.4f")

    env = create_trading_env_from_df(df, initial_balance=initial_balance, trading_fees=trading_fees)

    agent_choice = st.selectbox("Agent", ["Random", "SMA"])
    if agent_choice == "Random":
        agent = RandomAgent()
    else:
        short = st.number_input("SMA short window", value=5, step=1)
        long = st.number_input("SMA long window", value=20, step=1)
        agent = SMAAgent(short_window=short, long_window=long)

    if st.button("Run simulation"):
        # run episode
        obs = env.reset()
        done = False
        agent.reset()
        iters = 0
        while not done and iters < 10000:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            iters += 1
        st.success("Simulation finished")
        st.write(f"Final portfolio value: ${info['portfolio_valuation']:.2f}")

        # show charts
        hist = getattr(env, "history", None)
        if hist:
            st.plotly_chart(plot_portfolio_history(hist), use_container_width=True)
            st.plotly_chart(plot_trades_on_price(df, hist), use_container_width=True)

    st.markdown("---")
    st.subheader("Raw data preview")
    st.dataframe(df.tail(200))

else:
    st.info("Load a ticker from the sidebar and click 'Load data' to begin.")
