# src/envs/trading_env_wrapper.py
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import warnings

try:
    from gym_trading_env.environments import TradingEnv
    _HAS_EXTERNAL_ENV = True
except ImportError:
    TradingEnv = None
    _HAS_EXTERNAL_ENV = False

class SimpleTradingEnv:
    """
    Robust trading environment that handles both single-valued and MultiIndex DataFrames.
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, trading_fees: float = 0.0):
        self.df = df.copy().reset_index(drop=True)
        self._validate_dataframe()
        self.initial_balance = initial_balance
        self.trading_fees = trading_fees
        self.reset()

    def _validate_dataframe(self):
        """Ensure DataFrame has required structure."""
        if not any('Close' in col for col in self._get_column_names()):
            raise ValueError("DataFrame must contain 'Close' column")
        if len(self.df) < 2:
            raise ValueError("DataFrame must contain at least 2 rows")

    def _get_column_names(self):
        """Get column names accounting for MultiIndex"""
        if isinstance(self.df.columns, pd.MultiIndex):
            return [col[0] for col in self.df.columns]
        return self.df.columns.tolist()

    def _get_price(self, idx: int) -> float:
        """Safely extract price from DataFrame row."""
        if isinstance(self.df.columns, pd.MultiIndex):
            close_col = [col for col in self.df.columns if col[0] == 'Close'][0]
            price_data = self.df.iloc[idx][close_col]
        else:
            price_data = self.df.iloc[idx]['Close']
        
        if isinstance(price_data, pd.Series):
            return float(price_data.iloc[0])
        return float(price_data)

    def reset(self) -> Dict[str, float]:
        """Reset environment to initial state."""
        self.step_idx = 0
        self.position = 0
        self.balance = self.initial_balance
        self.shares = 0.0
        self.entry_price = 0.0
        self.history = {
            'step': [], 
            'price': [], 
            'position': [], 
            'portfolio_valuation': []
        }
        return self._obs()

    def _obs(self) -> Dict[str, float]:
        """Generate observation dictionary."""
        return {
            'price': self._get_price(self.step_idx),
            'position': float(self.position)
        }

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """Execute one environment step with guaranteed 4-value return"""
        try:
            # Initialize default return values
            obs = {}
            reward = 0.0
            done = False
            info = {'portfolio_valuation': self.balance}
            
            # Check if episode is done
            if self.step_idx >= len(self.df) - 1:
                return self._obs(), reward, True, info

            # Get current price safely
            price = self._get_price(self.step_idx)
            
            # Handle position changes
            if action != self.position:
                if self.position != 0:  # Close existing position
                    pnl = (price - self.entry_price) * self.position * self.shares
                    self.balance += pnl
                    self.shares = 0.0

                if action != 0:  # Open new position
                    invest = self.balance
                    self.shares = invest / price
                    self.entry_price = price
                    self.balance -= invest * self.trading_fees
                self.position = action

            # Calculate portfolio value
            portfolio_val = self.balance + (self.shares * price * self.position if self.position != 0 else 0)

            # Update history
            self.history['step'].append(self.step_idx)
            self.history['price'].append(price)
            self.history['position'].append(self.position)
            self.history['portfolio_valuation'].append(portfolio_val)

            self.step_idx += 1
            done = self.step_idx >= len(self.df) - 1
            reward = (portfolio_val / self.initial_balance - 1) * 100  # percent return
            info['portfolio_valuation'] = portfolio_val
            print(self._obs(), " ",reward," ",done,info)  # Debugging output
            return (
                self._obs(),  # observation
                float(reward),  # reward
                done,  # done flag
                info  # info dictionary
            )

        except Exception as e:
            print(f"Error in step(): {str(e)}")
            # Return safe default values that maintain the expected structure
            return {}, 0.0, True, {'error': str(e)}

def _prepare_for_external_env(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert DataFrame to external env format"""
    df_prepared = df.copy().reset_index(drop=True)
    
    # Handle MultiIndex columns
    if isinstance(df_prepared.columns, pd.MultiIndex):
        # Flatten MultiIndex to single level
        df_prepared.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                             for col in df_prepared.columns]
    
    # Standardize column names
    column_mapping = {
        'Close': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume',
        'Close_AAPL': 'close',
        'Open_AAPL': 'open',
        'High_AAPL': 'high',
        'Low_AAPL': 'low',
        'Volume_AAPL': 'volume'
    }
    
    df_prepared = df_prepared.rename(columns={
        k: v for k, v in column_mapping.items() if k in df_prepared.columns
    })
    
    # Ensure required columns exist
    required_cols = ['close', 'open', 'high', 'low', 'volume']
    for col in required_cols:
        if col not in df_prepared.columns:
            df_prepared[col] = df_prepared.get('close', 1.0)
    
    return df_prepared
def create_trading_env_from_df(
    df: pd.DataFrame, 
    initial_balance: float = 10000.0, 
    trading_fees: float = 0.001,
    ticker: str = "AAPL"
) -> SimpleTradingEnv:
    """Create trading environment with robust reward handling."""
    if not _HAS_EXTERNAL_ENV or TradingEnv is None:
        return SimpleTradingEnv(df=df, initial_balance=initial_balance, trading_fees=trading_fees)

    try:
        df_prepared = _prepare_for_external_env(df, ticker)
        
        def pnl_reward(history):
            """Calculate reward from portfolio values without dtype warnings"""
            # Convert to pandas Series if needed
            if isinstance(history["portfolio_valuation"], np.ndarray):
                vals = pd.Series(history["portfolio_valuation"], dtype='float64')
            else:
                vals = history["portfolio_valuation"].astype('float64')
            
            # Calculate percentage change with explicit dtype handling
            if len(vals) >= 2:
                pct_changes = vals.pct_change()
                # Explicitly handle NA values without downcasting
                filled_changes = pct_changes.replace([np.inf, -np.inf], np.nan).fillna(0)
                return float(filled_changes.iloc[-1])
            return 0.0  # Default reward for first step
        
        return TradingEnv(
            name=f"{ticker}TradingEnv",
            df=df_prepared,
            windows=None,
            positions=[-1, 0, 1],
            initial_position=0,
            trading_fees=trading_fees,
            borrow_interest_rate=0.0003,
            reward_function=pnl_reward
        )
    except Exception as e:
        warnings.warn(f"External TradingEnv failed: {str(e)}. Using SimpleTradingEnv.")
        return SimpleTradingEnv(df=df, initial_balance=initial_balance, trading_fees=trading_fees)
    

class EnhancedTradingEnv(SimpleTradingEnv):
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 trading_fees: float = 0.001, max_position_pct: float = 0.1):
        super().__init__(df, initial_balance, trading_fees)
        self.max_position_pct = max_position_pct  # Max % of capital per trade
        self.drawdown_threshold = 0.2  # Max allowed drawdown
        self.peak_balance = initial_balance

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        try:
            if self.step_idx >= len(self.df) - 1:
                return self._obs(), 0.0, True, {'portfolio_valuation': self.balance}
            
            price = self._get_price(self.step_idx)
            prev_portfolio_val = self.balance + (self.shares * price * self.position if self.position != 0 else 0)
            
            # Handle position changes with risk management
            if action != self.position:
                # Close existing position
                if self.position != 0:
                    pnl = (price - self.entry_price) * self.position * self.shares
                    self.balance += pnl
                    self.shares = 0.0
                
                # Open new position with position sizing
                if action != 0:
                    max_invest = self.balance * self.max_position_pct
                    self.shares = max_invest / price
                    self.entry_price = price
                    self.balance -= max_invest * self.trading_fees
                
                self.position = action
            
            # Calculate new portfolio value
            portfolio_val = self.balance + (self.shares * price * self.position if self.position != 0 else 0)
            self.peak_balance = max(self.peak_balance, portfolio_val)
            
            # Calculate drawdown
            current_drawdown = (self.peak_balance - portfolio_val) / self.peak_balance
            if current_drawdown >= self.drawdown_threshold:
                return self._obs(), -100.0, True, {'portfolio_valuation': portfolio_val, 'termination': 'drawdown'}
            
            # Update history
            self.history['step'].append(self.step_idx)
            self.history['price'].append(price)
            self.history['position'].append(self.position)
            self.history['portfolio_valuation'].append(portfolio_val)
            
            self.step_idx += 1
            done = self.step_idx >= len(self.df) - 1
            
            # Risk-adjusted reward calculation
            reward = self._calculate_reward(prev_portfolio_val, portfolio_val)
            
            return self._obs(), reward, done, {'portfolio_valuation': portfolio_val}
            
        except Exception as e:
            return self._obs(), 0.0, True, {'error': str(e)}

    def _calculate_reward(self, prev_val: float, current_val: float) -> float:
        """Calculate risk-adjusted reward"""
        raw_return = (current_val - prev_val) / prev_val if prev_val != 0 else 0
        
        # Add penalty for large positions
        position_penalty = abs(self.position) * 0.01  # 1% penalty per position unit
        
        # Add volatility component
        price_changes = np.diff([x for x in self.history['price'][-10:] if x is not None])
        volatility = np.std(price_changes) if len(price_changes) > 1 else 0
        volatility_penalty = volatility * 0.5
        
        return raw_return - position_penalty - volatility_penalty