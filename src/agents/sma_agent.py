# src/agents/sma_agent.py
from .base import BaseAgent
from collections import deque
from typing import Deque, Union
import pandas as pd
import numpy as np

class SMAAgent(BaseAgent):
    def __init__(self, short_window: int = 5, long_window: int = 20):
        assert short_window < long_window
        self.short_window = short_window
        self.long_window = long_window
        self.prices: Deque[float] = deque(maxlen=long_window)
        self.position = 0

    def reset(self):
        """Reset agent state between episodes"""
        self.prices.clear()
        self.position = 0

    def act(self, observation: dict) -> int:
        """
        Generate trading action based on SMA strategy
        
        Args:
            observation: Dictionary containing 'price' and other market data
            
        Returns:
            int: Trading action (-1, 0, 1)
        """
        # Safely extract price from observation
        price = self._extract_price(observation)
        if price is None:  # If price extraction failed
            return 0  # Return neutral position
            
        self.prices.append(price)
        
        # Wait until we have enough data
        if len(self.prices) < self.long_window:
            return 0
            
        # Calculate moving averages
        short_ma = sum(list(self.prices)[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices) / self.long_window
        
        # Generate signals
        if short_ma > long_ma and self.position <= 0:
            self.position = 1
            return 1  # Buy signal
        elif short_ma < long_ma and self.position >= 0:
            self.position = -1
            return -1  # Sell signal
        return 0  # Hold

    def _extract_price(self, observation: dict) -> Union[float, None]:
        """
        Safely extract price from observation dictionary
        
        Args:
            observation: Dictionary containing market data
            
        Returns:
            float: Extracted price or None if extraction fails
        """
        try:
            price_data = observation['price']
            
            # Handle all possible input types
            if isinstance(price_data, (float, int)):
                return float(price_data)
            elif isinstance(price_data, pd.Series):
                return float(price_data.iloc[0])  # Correct way to get first element
            elif isinstance(price_data, np.ndarray):
                return float(price_data.item()) if price_data.size == 1 else float(price_data[0])
            else:
                # Final attempt at conversion
                return float(price_data)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            print(f"Warning: Failed to extract price - {str(e)}")
            return None