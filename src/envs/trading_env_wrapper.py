# src/envs/trading_env_wrapper.py
from typing import Optional
import pandas as pd
import numpy as np

try:
    from gym_trading_env.environments import TradingEnv  # your package
    _HAS_EXTERNAL_ENV = True
except Exception:
    TradingEnv = None
    _HAS_EXTERNAL_ENV = False

class SimpleTradingEnv:
    """
    Minimal fallback trading environment (not gym-compliant fully) to allow simulation and agent debugging.
    It assumes df has a 'Close' column and index is increasing time.
    Actions: -1 (short), 0 (flat), 1 (long)
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, trading_fees: float = 0.0):
        df = df.copy().reset_index(drop=True)
        if 'Close' not in df.columns:
            raise ValueError("df must include 'Close' column for SimpleTradingEnv")
        self.df = df
        self.initial_balance = initial_balance
        self.trading_fees = trading_fees
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.position = 0
        self.balance = self.initial_balance
        self.shares = 0.0
        self.history = {'step': [], 'price': [], 'position': [], 'portfolio_valuation': []}
        return self._obs()

    def _obs(self):
        row = self.df.iloc[self.step_idx]
        return {'price': row['Close'], 'position': self.position}

    def step(self, action: int):
        price = float(self.df.iloc[self.step_idx]['Close'])
        # settle P&L for position change
        if action != self.position:
            # close existing
            if self.position != 0:
                # realize P&L from existing position
                pnl = (price - self.entry_price) * self.position * self.shares
                self.balance += pnl
                # remove shares
                self.shares = 0
            # open new if not neutral
            if action != 0:
                # invest some proportion of balance into position (simple: full invest)
                invest = self.balance
                self.shares = invest / price
                self.entry_price = price
                # subtract fees (simple)
                self.balance -= invest * self.trading_fees
            self.position = action

        # portfolio valuation
        if self.position == 0:
            portfolio_val = self.balance
        else:
            portfolio_val = self.balance + self.shares * price

        self.history['step'].append(self.step_idx)
        self.history['price'].append(price)
        self.history['position'].append(self.position)
        self.history['portfolio_valuation'].append(portfolio_val)

        self.step_idx += 1
        done = self.step_idx >= len(self.df)-1
        reward = (self.history['portfolio_valuation'][-1] / self.initial_balance - 1) * 100  # percent return

        info = {'portfolio_valuation': portfolio_val}
        obs = self._obs() if not done else None
        return obs, reward, done, info

def create_trading_env_from_df(df: pd.DataFrame, initial_balance: float = 10000.0, trading_fees: float = 0.001):
    """Factory: try to create external TradingEnv; otherwise fallback to SimpleTradingEnv."""
    if _HAS_EXTERNAL_ENV and TradingEnv is not None:
        # Attempt to adapt df to the external env expected format
        try:
            # Example adaptation — your gym_trading_env may expect columns named 'Close' -> 'price', etc.
            df2 = df.copy().reset_index(drop=True)
            if 'Close' in df2.columns:
                df2 = df2.rename(columns={'Close': 'price'})
            # Instantiate external env (adjust params as needed)
            env = TradingEnv(
                name="StockTradingEnv",
                df=df2,
                windows=None,
                positions=[-1,0,1],
                initial_position=0,
                trading_fees=trading_fees,
                borrow_interest_rate=0.0003,
                reward_function=None
            )
            return env
        except Exception as e:
            print("Failed to instantiate external TradingEnv:", str(e))
            print("Falling back to SimpleTradingEnv.")
    # fallback
    return SimpleTradingEnv(df=df, initial_balance=initial_balance, trading_fees=trading_fees)
