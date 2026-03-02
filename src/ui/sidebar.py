import streamlit as st
from typing import Optional, Tuple
from src.data_loader import StockDataService


def render_sidebar(data_service: StockDataService) -> Tuple[Optional[str], str, str]:
    """Render sidebar controls and return (selected_ticker, period, interval)."""
    st.sidebar.header("Data & Ticker")

    tickers_df = data_service.get_all_tickers()
    query = st.sidebar.text_input("Search tickers by symbol or name", "")
    results = data_service.search_tickers(query)

    selected = st.sidebar.selectbox("Select ticker", results['Symbol'].unique())
    period = st.sidebar.selectbox("Period", ['1mo', '3mo', '6mo', '1y', '5y', '10y', 'max'], index=3)
    interval = st.sidebar.selectbox("Interval", ['1d', '1wk', '1mo', '60m', '30m', '15m', '5m'], index=0)

    if st.sidebar.button("Load data"):
        try:
            result = data_service.get_stock_data_with_stats(
                ticker=selected,
                period=period,
                interval=interval
            )
            st.session_state.df = result['data']
            st.session_state.stats = result['stats']
            st.session_state.selected_ticker = selected
            st.session_state.period = period
            st.session_state.interval = interval
            st.session_state.lstm_trained = False
            st.session_state.stat_agent_trained = False
            st.session_state.simulation_results = None
            st.success(f"Loaded {len(result['data'])} rows for {selected}")
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None, period, interval

    return selected, period, interval
