from .base import BaseAgent
from ..models.lstm_predictor import LSTMPredictor
import pandas as pd
import numpy as np

class LSTMSMAAgent(BaseAgent):
    def __init__(self, short_window=5, long_window=20, 
                 lookback=60, forecast_horizon=30):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.predictor = LSTMPredictor(lookback, forecast_horizon)
        self.trained = False
        self.price_history = []
        self.predicted_prices = []
        
    def train(self, historical_data, epochs=50):
        """Train LSTM on historical data"""
        # Convert input to DataFrame if it isn't already
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
            
        # Safely extract Close prices and convert to list
        close_prices = historical_data['Close'] if 'Close' in historical_data else historical_data.iloc[:, 0]
        self.price_history = close_prices.values.tolist()
        
        # Ensure we have a DataFrame with just Close column for training
        train_df = pd.DataFrame({'Close': self.price_history})
        
        # Train the predictor
        self.predictor.train(train_df, epochs=epochs)
        self.trained = True
        
    def act(self, observation):
        if not self.trained:
            return 0
            
        # Get current price and update price history
        price = self._extract_price(observation)
        if price is None:
            return 0
            
        self.price_history.append(price)
        
        # Generate predictions periodically
        if len(self.price_history) >= self.predictor.lookback and len(self.price_history) % 30 == 0:
            recent_prices = pd.Series(self.price_history[-self.predictor.lookback:])
            self.predicted_prices = self.predictor.predict(recent_prices)
        
        # Use SMA strategy on predicted prices
        if len(self.predicted_prices) >= self.long_window:
            combined_prices = self.price_history + self.predicted_prices.tolist()
            price_series = pd.Series(combined_prices)
            
            short_ma = price_series.rolling(self.short_window).mean().iloc[-1]
            long_ma = price_series.rolling(self.long_window).mean().iloc[-1]
            
            if short_ma > long_ma:
                return 1  # Buy
            elif short_ma < long_ma:
                return -1  # Sell
                
        return 0  # Hold