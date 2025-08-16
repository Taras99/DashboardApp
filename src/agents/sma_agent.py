from .base import BaseAgent
from collections import deque
from typing import Deque, Union
import pandas as pd
import numpy as np

class SMAAgent(BaseAgent):
    def __init__(self, short_window: int = 5, long_window: int = 20, 
                 position_size: float = 0.1, stop_loss_pct: float = 0.05):
        assert short_window < long_window
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size  # Fraction of capital to risk per trade
        self.stop_loss_pct = stop_loss_pct  # Percentage stop loss
        self.prices: Deque[float] = deque(maxlen=long_window)
        self.position = 0
        self.entry_price = 0

    def reset(self):
        """Reset agent state between episodes"""
        self.prices.clear()
        self.position = 0
        self.entry_price = 0

    def act(self, observation: dict) -> int:
        """
        Generate trading action based on enhanced SMA strategy
        
        Args:
            observation: Dictionary containing 'price' and other market data
            
        Returns:
            int: Trading action (-1, 0, 1)
        """
        price = self._extract_price(observation)
        if price is None:
            return 0
            
        self.prices.append(price)
        
        # Check for stop loss first
        if self.position != 0:
            if (self.position > 0 and price <= self.entry_price * (1 - self.stop_loss_pct)) or \
               (self.position < 0 and price >= self.entry_price * (1 + self.stop_loss_pct)):
                action = -self.position  # Exit position
                self.position = 0
                return action
        
        # Wait until we have enough data
        if len(self.prices) < self.long_window:
            return 0
            
        # Calculate moving averages
        short_ma = np.mean(list(self.prices)[-self.short_window:])
        long_ma = np.mean(self.prices)
        
        # Generate signals with confirmation
        if short_ma > long_ma * 1.01:  # 1% threshold for confirmation
            if self.position <= 0:  # Only enter if not already long
                self.position = self.position_size  # Fractional position
                self.entry_price = price
                return 1  # Buy signal
        elif short_ma < long_ma * 0.99:  # 1% threshold for confirmation
            if self.position >= 0:  # Only enter if not already short
                self.position = -self.position_size  # Fractional position
                self.entry_price = price
                return -1  # Sell signal
                
        return 0  # Hold

    def _extract_price(self, observation: dict) -> Union[float, None]:
        """Same as before"""
        try:
            price_data = observation['price']
            if isinstance(price_data, (float, int)):
                return float(price_data)
            elif isinstance(price_data, pd.Series):
                return float(price_data.iloc[-1])  # Get most recent price
            elif isinstance(price_data, np.ndarray):
                return float(price_data.item()) if price_data.size == 1 else float(price_data[-1])
            else:
                return float(price_data)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            print(f"Warning: Failed to extract price - {str(e)}")
            return None
        
class EnhancedSMAAgent(SMAAgent):
    def __init__(self, short_window: int = 5, long_window: int = 20,
                 confirmation_pct: float = 0.01, volatility_threshold: float = 0.02):
        super().__init__(short_window, long_window)
        self.confirmation_pct = confirmation_pct
        self.volatility_threshold = volatility_threshold
        self.price_history = deque(maxlen=100)  # For volatility calculation

    def act(self, observation: dict) -> int:
        price = self._extract_price(observation)
        if price is None:
            return 0
            
        self.price_history.append(price)
        self.prices.append(price)
        
        # Wait for sufficient data
        if len(self.prices) < self.long_window:
            return 0
            
        # Calculate volatility safely
        if len(self.price_history) > 1:
            try:
                # Convert deque to numpy array first
                price_array = np.array(self.price_history)
                # Calculate returns safely
                returns = np.diff(price_array) / price_array[:-1]
                volatility = np.std(returns)
                
                # Avoid trading in high volatility if filter enabled
                if hasattr(self, 'volatility_filter') and self.volatility_filter:
                    if volatility > self.volatility_threshold:
                        return 0
            except (ValueError, ZeroDivisionError) as e:
                print(f"Volatility calculation error: {str(e)}")
                volatility = 0
        
        # Calculate moving averages
        short_ma = np.mean(list(self.prices)[-self.short_window:])
        long_ma = np.mean(self.prices)
        
        # Generate signals with confirmation
        if short_ma > long_ma * (1 + self.confirmation_pct):
            if self.position <= 0:  # Only enter if not already long
                return 1
        elif short_ma < long_ma * (1 - self.confirmation_pct):
            if self.position >= 0:  # Only enter if not already short
                return -1
                
        return 0  # Hold