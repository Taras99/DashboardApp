# src/agents/sma_agent.py
from .base import BaseAgent
from collections import deque
from typing import Deque

class SMAAgent(BaseAgent):
    def __init__(self, short_window: int = 5, long_window: int = 20):
        assert short_window < long_window
        self.short_window = short_window
        self.long_window = long_window
        self.prices = Deque(maxlen=long_window)
        self.position = 0

    def reset(self):
        self.prices.clear()
        self.position = 0

    def act(self, observation):
        price = float(observation['price'])
        self.prices.append(price)
        if len(self.prices) < self.long_window:
            return 0
        short_ma = sum(list(self.prices)[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices) / self.long_window
        if short_ma > long_ma and self.position <= 0:
            self.position = 1
            return 1
        elif short_ma < long_ma and self.position >= 0:
            self.position = -1
            return -1
        else:
            return 0
