import streamlit as st
import pandas as pd
from src.envs.trading_env_wrapper import create_trading_env_from_df
from src.agents.lstm_agent import LSTMSMAAgent
from src.agents.statistical_agent import StatisticalMeanAgent, ContinuousTradingAgent
from src.ui.metrics import calculate_performance_metrics


def render_simulation_section(df: pd.DataFrame):
    """Render the trading simulation section."""
    st.markdown("---")
    st.subheader("Run simulation")

    # Environment parameters
    col1, col2, col3 = st.columns(3)
    initial_balance = col1.number_input("Initial balance", value=10000.0, step=1000.0)
    trading_fees = col2.number_input("Trading fee (fraction)", value=0.001, step=0.0001, format="%.4f")
    max_position = col3.slider("Max position size (%)", 1, 100, 10) / 100

    col4, col5 = st.columns(2)
    allow_shorting = col4.checkbox("Allow Shorting", value=False)
    interest_rate = col5.number_input("Interest Rate", value=0.0003, step=0.0001, format="%.4f")

    env_kwargs = {
        'initial_balance': initial_balance,
        'trading_fees': trading_fees,
        'env_type': 'advanced',
        'max_position_pct': max_position,
        'allow_shorting': allow_shorting,
        'interest_rate': interest_rate
    }

    try:
        env = create_trading_env_from_df(df, **env_kwargs)
    except Exception as e:
        st.error(f"Failed to create environment: {str(e)}")
        return

    # Agent selection
    agent_choice = st.selectbox("Agent", ["ContinuousTradingAgent", "LSTM-SMA", "Statistical Mean"])
    agent = None
    min_data_points = 0

    if agent_choice == "ContinuousTradingAgent":
        st.markdown("**Continuous Trading Agent Parameters**")
        col1, col2 = st.columns(2)
        long_threshold = col1.number_input("Long threshold (%)", value=5.0, min_value=1.0, max_value=20.0) / 100
        short_threshold = col2.number_input("Short threshold (%)", value=5.0, min_value=1.0, max_value=20.0) / 100

        col3, col4 = st.columns(2)
        position_size = col3.number_input("Position size (%)", value=50.0, min_value=10.0, max_value=100.0) / 100
        min_data_points = col4.number_input("Min data points", value=30, min_value=10, max_value=200)

        col5, col6 = st.columns(2)
        stop_loss_pct = col5.number_input("Stop loss (%)", value=10.0, min_value=1.0, max_value=30.0) / 100
        min_profit_pct = col6.number_input("Min profit (%)", value=2.0, min_value=0.5, max_value=20.0) / 100

        allow_loss_selling = st.checkbox("Allow selling at loss", value=False)

        agent = ContinuousTradingAgent(
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            position_size=position_size,
            min_data_points=min_data_points,
            stop_loss_pct=stop_loss_pct,
            min_profit_pct=min_profit_pct,
            allow_loss_selling=allow_loss_selling
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

        if st.button("Train LSTM Model"):
            with st.spinner("Training LSTM model..."):
                try:
                    train_size = int(0.8 * len(df))
                    close_prices = df['Close'].iloc[:train_size] if 'Close' in df.columns else df.iloc[:, 0].iloc[:train_size]
                    agent.train(close_prices)
                    st.session_state.lstm_trained = True
                    st.success("LSTM model trained successfully!")
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

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

        if len(df) < min_data_points:
            st.warning(f"Need at least {min_data_points} data points. Current: {len(df)}")

    if agent is None:
        st.error("Agent not properly initialized. Please check your agent configuration.")
        return

    run_disabled = (
        (agent_choice == "LSTM-SMA" and not st.session_state.lstm_trained) or
        (agent_choice == "Statistical Mean" and len(df) < min_data_points)
    )

    if st.button("Run simulation", disabled=run_disabled):
        if agent_choice == "LSTM-SMA" and not st.session_state.lstm_trained:
            st.error("LSTM model must be trained first!")
        elif agent_choice == "Statistical Mean" and len(df) < min_data_points:
            st.error(f"Need at least {min_data_points} data points for statistical agents!")
        else:
            run_simulation(env, agent, df)


def run_simulation(env, agent, df: pd.DataFrame):
    """Execute the trading simulation."""
    obs = env.reset()

    if hasattr(agent, 'reset'):
        agent.reset()

    if hasattr(agent, 'set_initial_balance'):
        agent.set_initial_balance(env.initial_balance)

    progress_bar = st.progress(0)
    status_text = st.empty()
    max_steps = len(df) - 1

    results = {
        'actions': [],
        'rewards': [],
        'portfolio_values': [],
        'positions': [],
        'prices': [],
        'cash_balances': []
    }

    for step in range(max_steps):
        progress = int(100 * (step / max_steps))
        progress_bar.progress(progress)
        status_text.text(f"Processing step {step + 1}/{max_steps}...")

        try:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            results['actions'].append(action)
            results['rewards'].append(reward)
            results['portfolio_values'].append(info.get('portfolio_valuation', obs.get('portfolio_value', 0)))
            results['positions'].append(obs.get('position', 0))
            results['prices'].append(obs.get('price', 0))
            results['cash_balances'].append(obs.get('cash', 0))

        except Exception as e:
            st.error(f"Error in step {step}: {str(e)}")
            break

        if done:
            break

    st.session_state.simulation_results = {
        'env_history': getattr(env, 'history', {}),
        'metrics': calculate_performance_metrics(results),
        'raw_results': results,
        'environment_type': type(env).__name__
    }

    progress_bar.progress(100)
    status_text.text("Simulation complete!")
    st.success("Trading simulation finished successfully")
    st.rerun()
