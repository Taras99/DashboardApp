from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseAgent(ABC):
    @abstractmethod
    def act(self, observation) -> int:
        """Return action given observation. Actions expected to be -1,0,1"""
        raise NotImplementedError

    def reset(self):
        """Optional reset between episodes"""
        pass

    def _extract_price(self, observation: dict) -> float:
        """Extract price from observation dictionary."""
        try:
            price_data = observation['price']
            if isinstance(price_data, (float, int)):
                return float(price_data)
            elif isinstance(price_data, pd.Series):
                return float(price_data.iloc[-1])
            elif isinstance(price_data, np.ndarray):
                return float(price_data.item()) if price_data.size == 1 else float(price_data[-1])
            else:
                return float(price_data)
        except (KeyError, ValueError, TypeError):
            return None
