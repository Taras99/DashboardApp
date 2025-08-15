# streamlit_app.py
import streamlit as st
import pandas as pd
from src.data_loader import StockDataService
from src.envs.trading_env_wrapper import create_trading_env_from_df
from src.agents.sma_agent import SMAAgent
from src.viz.plots import plot_price, plot_portfolio_history, plot_trades_on_price
from typing import Optional, Tuple, Dict

class StockRLDashboard:
    """Main dashboard application class."""
    
    def __init__(self):
        self.data_service = StockDataService()
        self._initialize_session()
        self._setup_page_config()
        
    def _initialize_session(self):
        """Initialize session state variables."""
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'selected_ticker' not in st.session_state:
            st.session_state.selected_ticker = None
        if 'period' not in st.session_state:
            st.session_state.period = '1y'
        if 'interval' not in st.session_state:
            st.session_state.interval = '1d'
        if 'stats' not in st.session_state:
            st.session_state.stats = None
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Stock RL Dashboard", 
            layout="wide", 
            initial_sidebar_state="expanded"
        )
        st.title("📈 Stock analysis & RL trading dashboard")
    
    def _render_sidebar(self) -> Tuple[Optional[str], str, str]:
        """Render sidebar controls and return user selections."""
        st.sidebar.header("Data & Ticker")
        
        # Get tickers and search
        tickers_df = self.data_service.get_all_tickers()
        query = st.sidebar.text_input("Search tickers by symbol or name", "")
        results = self.data_service.search_tickers(query)
        
        # Selection controls
        selected = st.sidebar.selectbox("Select ticker", results['Symbol'].unique())
        period = st.sidebar.selectbox("Period", ['1mo','3mo','6mo','1y','5y','10y','max'], index=3)
        interval = st.sidebar.selectbox("Interval", ['1d','1wk','1mo','60m','30m','15m','5m'], index=0)
        
        # Load data button
        if st.sidebar.button("Load data"):
            try:
                result = self.data_service.get_stock_data_with_stats(
                    ticker=selected, 
                    period=period, 
                    interval=interval
                )
                st.session_state.df = result['data']
                st.session_state.stats = result['stats']
                st.session_state.selected_ticker = selected
                st.session_state.period = period
                st.session_state.interval = interval
                st.success(f"Loaded {len(result['data'])} rows for {selected}")
            except Exception as e:
                st.error(f"Failed to load data: {str(e)}")
                return None, period, interval
        
        return selected, period, interval
    
    def _render_price_metrics(self, df: pd.DataFrame):
        """Render price metrics based on the loaded DataFrame."""
        try:
            # Ensure we're working with single values, not Series
            close_prices = df['Close'].dropna()
            if len(close_prices) < 2:
                st.warning("Not enough data points to calculate changes")
                return
                
            latest_close = close_prices.iloc[-1]
            prev_close = close_prices.iloc[-2]
            
            # Calculate changes
            change = latest_close - prev_close
            pct_change = 100.0 * change / prev_close if prev_close != 0 else 0.0

            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Last Close", f"${latest_close:.2f}", f"${change:.2f}")
            col2.metric("Change (%)", f"{pct_change:.2f}%")
            col3.metric("Rows", f"{len(df)}")

        except Exception as e:
            st.error(f"Error rendering price metrics: {str(e)}")
    
    def _render_simulation_section(self, df: pd.DataFrame):
        """Render the trading simulation section."""
        st.markdown("---")
        st.subheader("Run simulation")
        
        initial_balance = st.number_input("Initial balance", value=10000.0, step=1000.0)
        trading_fees = st.number_input("Trading fee (fraction)", value=0.001, step=0.0001, format="%.4f")

        env = create_trading_env_from_df(
            df, 
            initial_balance=initial_balance, 
            trading_fees=trading_fees
        )

        agent_choice = st.selectbox("Agent", ["SMA"])
        if agent_choice == "SMA":
            short = st.number_input("SMA short window", value=5, step=1)
            long = st.number_input("SMA long window", value=20, step=1)
            agent = SMAAgent(short_window=short, long_window=long)

        if st.button("Run simulation"):
            self._run_simulation(env, agent)
    
    def _run_simulation(self, env, agent):
        """Execute the trading simulation."""
        obs = env.reset()
        done = False
        agent.reset()
        iters = 0
        
        with st.spinner("Running simulation..."):
            while not done and iters < 10000:
                action = agent.act(obs)
                obs, reward,terminated, done, info = env.step(action)
                iters += 1
        
        st.success("Simulation finished")
        st.write(f"Final portfolio value: ${info['portfolio_valuation']:.2f}")

        # Show results
        hist = getattr(env, "history", None)
        if hist:
            st.plotly_chart(plot_portfolio_history(hist), use_container_width=True)
            st.plotly_chart(plot_trades_on_price(st.session_state.df, hist), use_container_width=True)
    
    def _render_data_preview(self, df: pd.DataFrame):
        """Render the raw data preview section."""
        st.markdown("---")
        st.subheader("Raw data preview")
        st.dataframe(df.tail(200))
    
    def run(self):
        """Main method to run the dashboard."""
        selected, period, interval = self._render_sidebar()
        
        if st.session_state.df is not None:
            df = st.session_state.df
            selected = st.session_state.selected_ticker
            
            st.subheader(f"{selected} price")
            fig_price = plot_price(df)
            st.plotly_chart(fig_price, use_container_width=True)
            
            self._render_price_metrics(df)
            self._render_simulation_section(df)
            self._render_data_preview(df)
        else:
            st.info("Load a ticker from the sidebar and click 'Load data' to begin.")

# Main entry point
if __name__ == "__main__":
    dashboard = StockRLDashboard()
    dashboard.run()