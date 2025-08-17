import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class LSTMPredictor:
    def __init__(self, lookback=60, forecast_horizon=30):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            Input(shape=(self.lookback, 1)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(self.forecast_horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def prepare_data(self, data):
        """Convert time series into supervised learning problem"""
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            values = data['Close'].values
        elif isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.array(data)
            
        # Ensure proper shape and type
        values = np.array(values, dtype=np.float64).reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data)-self.lookback-self.forecast_horizon):
            X.append(scaled_data[i:(i+self.lookback), 0])
            y.append(scaled_data[(i+self.lookback):(i+self.lookback+self.forecast_horizon), 0])
            
        return np.array(X), np.array(y)
    
    def train(self, train_data, epochs=50, batch_size=32, verbose=0):
        X, y = self.prepare_data(train_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        history = self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose
        )
        return history
    
    def predict(self, recent_data):
        """Predict future prices given recent history"""
        # Convert input to numpy array
        if isinstance(recent_data, pd.Series):
            values = recent_data.values
        else:
            values = np.array(recent_data)
            
        values = values.astype(np.float64).reshape(-1, 1)
        scaled_data = self.scaler.transform(values)
        X = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
        pred = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(pred).flatten()