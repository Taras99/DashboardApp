import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import StockDataService
from src.envs.trading_env_wrapper import EnhancedTradingEnv
from src.agents.sma_agent import EnhancedSMAAgent
from src.agents.lstm_agent import LSTMSMAAgent
from src.agents.statistical_agent import StatisticalMeanAgent, EnhancedStatisticalAgent  # New imports
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
        if 'lstm_trained' not in st.session_state:
            st.session_state.lstm_trained = False
        if 'stat_agent_trained' not in st.session_state:
            st.session_state.stat_agent_trained = False
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Stock RL Dashboard", 
            layout="wide", 
            initial_sidebar_state="expanded"
        )
        st.title("📈 Stock Analysis & RL Trading Dashboard")
    
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
                st.session_state.lstm_trained = False  # Reset training flags
                st.session_state.stat_agent_trained = False
                st.success(f"Loaded {len(result['data'])} rows for {selected}")
            except Exception as e:
                st.error(f"Failed to load data: {str(e)}")
                return None, period, interval
        
        return selected, period, interval
    
    def _render_price_metrics(self, df: pd.DataFrame):
        """Render price metrics based on the loaded DataFrame."""
        try:
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
            cols = st.columns(4)
            cols[0].metric("Last Close", f"${latest_close:.2f}", f"${change:.2f}")
            cols[1].metric("Change (%)", f"{pct_change:.2f}%")
            cols[2].metric("52W High", f"${close_prices.max():.2f}")
            cols[3].metric("52W Low", f"${close_prices.min():.2f}")

        except Exception as e:
            st.error(f"Error rendering price metrics: {str(e)}")
    
    def _render_simulation_section(self, df: pd.DataFrame):
        """Render the trading simulation section."""
        st.markdown("---")
        st.subheader("Run simulation")
        
        initial_balance = st.number_input("Initial balance", value=10000.0, step=1000.0)
        trading_fees = st.number_input("Trading fee (fraction)", value=0.001, step=0.0001, format="%.4f")
        max_position = st.slider("Max position size (%)", 1, 100, 10) / 100

        env = EnhancedTradingEnv(
            df, 
            initial_balance=initial_balance, 
            trading_fees=trading_fees,
            max_position_pct=max_position
        )

        agent_choice = st.selectbox("Agent", ["SMA", "LSTM-SMA", "Statistical Mean", "Enhanced Statistical"])
        
        if agent_choice == "SMA":
            col1, col2 = st.columns(2)
            short = col1.number_input("SMA short window", value=5, min_value=1, max_value=50)
            long = col2.number_input("SMA long window", value=20, min_value=2, max_value=200)
            confirmation = st.slider("Confirmation threshold (%)", 0.1, 5.0, 1.0) / 100
            
            agent = EnhancedSMAAgent(
                short_window=short,
                long_window=long,
                confirmation_pct=confirmation
            )
            
        elif agent_choice == "LSTM-SMA":
            col1, col2 = st.columns(2)
            short = col1.number_input("SMA short window", value=5, min_value=1, max_value=50)
            long = col2.number_input("SMA long window", value=20, min_value=2, max_value=200)
            
            st.markdown("**LSTM Parameters**")
            col3, col4 = st.columns(2)
            lookback = col3.number_input("Lookback window", value=60, min_value=10, max_value=200)
            horizon = col4.number_input("Prediction horizon", value=30, min_value=5, max_value=90)
            epochs = st.number_input("Training epochs", value=50, min_value=1, max_value=200)
            
            agent = LSTMSMAAgent(
                short_window=short,
                long_window=long,
                lookback=lookback,
                forecast_horizon=horizon
            )
            
            # Train button separate from run
            if st.button("Train LSTM Model"):
                with st.spinner("Training LSTM model..."):
                    train_size = int(0.8 * len(df))
                    close_prices = df['Close'].iloc[:train_size]
                    agent.train(close_prices)
                    st.session_state.lstm_trained = True
                    st.success("LSTM model trained successfully!")
            
            if not st.session_state.lstm_trained:
                st.warning("Please train the LSTM model before running simulation")
                
        elif agent_choice == "Statistical Mean":
            st.markdown("**Statistical Mean Parameters**")
            col1, col2 = st.columns(2)
            long_threshold = col1.number_input("Long threshold (%)", value=10.0, min_value=1.0, max_value=30.0) / 100
            short_threshold = col2.number_input("Short threshold (%)", value=15.0, min_value=1.0, max_value=30.0) / 100
            
            col3, col4 = st.columns(2)
            min_data_points = col3.number_input("Min data points", value=100, min_value=50, max_value=500)
            n_splits = col4.number_input("Cross-validation splits", value=10, min_value=5, max_value=20)
            
            agent = StatisticalMeanAgent(
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                min_data_points=min_data_points,
                n_splits=n_splits
            )
            
            # Statistical agents need sufficient data
            if len(df) < min_data_points:
                st.warning(f"Need at least {min_data_points} data points. Current: {len(df)}")
                
        elif agent_choice == "Enhanced Statistical":
            st.markdown("**Enhanced Statistical Parameters**")
            col1, col2 = st.columns(2)
            long_threshold = col1.number_input("Long threshold (%)", value=10.0, min_value=1.0, max_value=30.0) / 100
            short_threshold = col2.number_input("Short threshold (%)", value=15.0, min_value=1.0, max_value=30.0) / 100
            
            col3, col4 = st.columns(2)
            min_data_points = col3.number_input("Min data points", value=100, min_value=50, max_value=500)
            n_splits = col4.number_input("Cross-validation splits", value=10, min_value=5, max_value=20)
            
            col5, col6 = st.columns(2)
            volatility_filter = col5.checkbox("Volatility filter", value=True)
            trend_confirmation = col6.checkbox("Trend confirmation", value=True)
            
            agent = EnhancedStatisticalAgent(
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                min_data_points=min_data_points,
                n_splits=n_splits,
                volatility_filter=volatility_filter,
                trend_confirmation=trend_confirmation
            )
            
            if len(df) < min_data_points:
                st.warning(f"Need at least {min_data_points} data points. Current: {len(df)}")

        run_disabled = (
            (agent_choice == "LSTM-SMA" and not st.session_state.lstm_trained) or
            (agent_choice in ["Statistical Mean", "Enhanced Statistical"] and len(df) < min_data_points)
        )
        
        if st.button("Run simulation", disabled=run_disabled):
            if agent_choice == "LSTM-SMA" and not st.session_state.lstm_trained:
                st.error("LSTM model must be trained first!")
            elif agent_choice in ["Statistical Mean", "Enhanced Statistical"] and len(df) < min_data_points:
                st.error(f"Need at least {min_data_points} data points for statistical agents!")
            else:
                self._run_simulation(env, agent)
    
    def _run_simulation(self, env, agent):
        """Execute the trading simulation."""
        obs = env.reset()
        done = False
        agent.reset()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        max_steps = len(env.df) - 1
        
        results = {
            'actions': [],
            'rewards': [],
            'portfolio_values': [],
            'positions': []
        }
        
        for step in range(max_steps):
            # Update progress
            progress = int(100 * (step / max_steps))
            progress_bar.progress(progress)
            status_text.text(f"Processing step {step+1}/{max_steps}...")
            
            # Run simulation step
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            
            # Store results
            results['actions'].append(action)
            results['rewards'].append(reward)
            results['portfolio_values'].append(info['portfolio_valuation'])
            results['positions'].append(obs['position'])
            
            if done:
                break
        
        # Store results in session state
        st.session_state.simulation_results = {
            'env_history': getattr(env, 'history', {}),
            'metrics': self._calculate_performance_metrics(results),
            'raw_results': results
        }
        
        # Display completion message
        progress_bar.progress(100)
        status_text.text("Simulation complete!")
        st.success("Trading simulation finished successfully")
        
        # Display results
        self._render_simulation_results()
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate key performance metrics from simulation results."""
        portfolio_values = results['portfolio_values']
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        annualized_return = (1 + total_return/100)**(252/len(portfolio_values)) - 1
        
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        max_drawdown = (np.maximum.accumulate(portfolio_values) - portfolio_values) / \
                      np.maximum.accumulate(portfolio_values)
        max_drawdown = np.max(max_drawdown) * 100 if len(max_drawdown) > 0 else 0
        
        win_rate = len([r for r in returns if r > 0]) / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate * 100,
            'final_value': portfolio_values[-1]
        }
    
    def _render_simulation_results(self):
        """Enhanced results visualization with multiple tabs."""
        if not st.session_state.simulation_results:
            st.warning("No simulation results available")
            return
            
        results = st.session_state.simulation_results
        metrics = results['metrics']
        
        # Display key metrics
        st.subheader("Performance Summary")
        cols = st.columns(4)
        cols[0].metric("Final Value", f"${metrics['final_value']:,.2f}")
        cols[1].metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        cols[2].metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
        cols[3].metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Portfolio Value", "Trades", "Advanced Metrics"])
        
        with tab1:
            st.plotly_chart(
                plot_portfolio_history(results['env_history'], st.session_state.df), 
                use_container_width=True
            )

        with tab2:
            st.plotly_chart(
                plot_trades_on_price(st.session_state.df, results['env_history']), 
                use_container_width=True
            )

        with tab3:
            st.subheader("Risk/Reward Analysis")
            col1, col2 = st.columns(2)
            col1.metric("Volatility", f"{metrics['volatility_pct']:.2f}%")
            col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
            st.subheader("Trade Statistics")
            st.write(f"Win Rate: {metrics['win_rate_pct']:.1f}%")
    
    def run(self):
        """Main method to run the dashboard."""
        selected, period, interval = self._render_sidebar()
        
        if st.session_state.df is not None:
            df = st.session_state.df
            selected = st.session_state.selected_ticker
            
            st.header(f"{selected} Analysis ({period}, {interval})")
            st.plotly_chart(plot_price(df), use_container_width=True)
            
            self._render_price_metrics(df)
            self._render_simulation_section(df)
            
            if st.session_state.simulation_results:
                self._render_simulation_results()
            
            with st.expander("Raw Data Preview"):
                st.dataframe(df.tail(200))
        else:
            st.info("Please select a ticker and load data from the sidebar to begin.")

if __name__ == "__main__":
    dashboard = StockRLDashboard()
    dashboard.run()