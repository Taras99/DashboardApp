import streamlit as st
from src.data_loader import StockDataService
from src.viz.plots import plot_price
from src.ui.sidebar import render_sidebar
from src.ui.metrics import render_price_metrics
from src.ui.simulation import render_simulation_section
from src.ui.results import render_simulation_results


class StockRLDashboard:
    """Main dashboard application class."""

    def __init__(self):
        self.data_service = StockDataService()
        self._initialize_session()
        self._setup_page_config()

    def _initialize_session(self):
        defaults = {
            'df': None, 'selected_ticker': None, 'period': '1y',
            'interval': '1d', 'stats': None, 'lstm_trained': False,
            'stat_agent_trained': False, 'simulation_results': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _setup_page_config(self):
        st.set_page_config(
            page_title="Stock RL Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def run(self):
        st.title("Stock Analysis & RL Trading Dashboard")

        selected, period, interval = render_sidebar(self.data_service)

        if st.session_state.df is not None:
            df = st.session_state.df
            selected = st.session_state.selected_ticker

            st.header(f"{selected} Analysis ({period}, {interval})")

            try:
                st.plotly_chart(plot_price(df), use_container_width=True, key="price_chart")
            except Exception as e:
                st.error(f"Error plotting price data: {str(e)}")

            render_price_metrics(df)
            render_simulation_section(df)

            if st.session_state.simulation_results:
                render_simulation_results()

            with st.expander("Raw Data Preview"):
                st.dataframe(df.tail(200))
        else:
            st.info("Please select a ticker and load data from the sidebar to begin.")


if __name__ == "__main__":
    dashboard = StockRLDashboard()
    dashboard.run()
