import streamlit as st
import pandas as pd
from src.viz.plots import plot_portfolio_history, plot_trades_on_price


def render_simulation_results():
    """Enhanced results visualization with multiple tabs."""
    if not st.session_state.simulation_results:
        st.warning("No simulation results available")
        return

    results = st.session_state.simulation_results
    metrics = results['metrics']

    st.subheader("Performance Summary")
    cols = st.columns(4)
    cols[0].metric("Final Value", f"${metrics['final_value']:,.2f}")
    cols[1].metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
    cols[2].metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
    cols[3].metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

    cols2 = st.columns(3)
    cols2[0].metric("Volatility", f"{metrics['volatility_pct']:.2f}%")
    cols2[1].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    cols2[2].metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")

    tab1, tab2, tab3 = st.tabs(["Portfolio Value", "Trades", "Detailed Analysis"])

    with tab1:
        try:
            st.plotly_chart(
                plot_portfolio_history(results['env_history'], st.session_state.df),
                use_container_width=True,
                key="portfolio_history_chart"
            )
        except Exception as e:
            st.error(f"Error plotting portfolio history: {str(e)}")

    with tab2:
        try:
            st.plotly_chart(
                plot_trades_on_price(st.session_state.df, results['env_history']),
                use_container_width=True,
                key="trades_on_price_chart"
            )
        except Exception as e:
            st.error(f"Error plotting trades: {str(e)}")

    with tab3:
        st.subheader("Trade Analysis")
        st.write(f"Total Trades Executed: {metrics['total_trades']}")

        positions = results['raw_results']['positions']
        position_counts = pd.Series(positions).value_counts()
        st.write("Position Distribution:")
        st.write(f"- Long: {position_counts.get(1, 0)} steps")
        st.write(f"- Short: {position_counts.get(-1, 0)} steps")
        st.write(f"- Neutral: {position_counts.get(0, 0)} steps")

        if results['raw_results']['cash_balances']:
            st.subheader("Cash Balance Over Time")
            cash_df = pd.DataFrame({
                'step': range(len(results['raw_results']['cash_balances'])),
                'cash': results['raw_results']['cash_balances']
            })
            st.line_chart(cash_df.set_index('step'))
