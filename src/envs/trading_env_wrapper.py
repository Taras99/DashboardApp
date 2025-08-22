# src/envs/trading_env_wrapper.py
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import warnings
import datetime
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
    

class AdvancedTradingEnv:
    """
    Advanced trading environment with better portfolio management and realistic trading mechanics.
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 trading_fees: float = 0.001, max_position_pct: float = 0.1,
                 allow_shorting: bool = False, interest_rate: float = 0.0003):
        
        self.df = df.copy()
        self._validate_dataframe()
        self.initial_balance = initial_balance
        self.trading_fees = trading_fees
        self.max_position_pct = max_position_pct
        self.allow_shorting = allow_shorting
        self.interest_rate = interest_rate
        self.reset()

    def _validate_dataframe(self):
        """Ensure DataFrame has required structure."""
        required_columns = ['Close', 'Open', 'High', 'Low']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        if len(self.df) < 2:
            raise ValueError("DataFrame must contain at least 2 rows")

    def _get_price(self, idx: int, price_type: str = 'Close') -> float:
        """Safely extract price from DataFrame row."""
        try:
            price_data = self.df.iloc[idx][price_type]
            if isinstance(price_data, pd.Series):
                return float(price_data.iloc[0])
            return float(price_data)
        except (KeyError, IndexError, ValueError):
            return float(self.df.iloc[idx]['Close'])

    def _get_current_date(self) -> datetime:
        """Get current date from DataFrame."""
        if 'Date' in self.df.columns:
            return pd.to_datetime(self.df.iloc[self.step_idx]['Date'])
        elif self.df.index.name == 'Date':
            return pd.to_datetime(self.df.index[self.step_idx])
        else:
            return datetime.now()

    def reset(self):
        """Reset environment to initial state."""
        self.step_idx = 0
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.balance = self.initial_balance  # CASH balance
        self.shares_held = 0.0  # NUMBER of shares currently held
        self.entry_price = 0.0  # Average entry price
        self.peak_balance = self.initial_balance
        self.total_trades = 0
        self.total_commission = 0.0
        
        self.history = {
            'step': [], 
            'date': [],
            'price': [], 
            'position': [], 
            'portfolio_valuation': [],
            'cash_balance': [],
            'stock_value': [],
            'shares_held': [],
            'action_taken': []
        }
        return self._obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one environment step with proper portfolio tracking."""
        try:
            if self.step_idx >= len(self.df) - 1:
                final_obs = self._obs()
                return final_obs, 0.0, True, {'portfolio_valuation': final_obs['portfolio_value']}

            current_price = self._get_price(self.step_idx)
            
            # Store previous portfolio value for reward calculation
            prev_portfolio_val = self._calculate_portfolio_value(current_price)
            
            # Handle position changes
            position_changed = False
            if action != self.position:
                position_changed = True
                self.total_trades += 1
                
                # CLOSE EXISTING POSITION (sell stocks)
                if self.position != 0:
                    if self.position == 1:  # Close long position (sell stocks)
                        # Calculate profit from selling
                        sale_value = self.shares_held * current_price
                        fees = self._apply_trading_fees(sale_value)
                        self.balance += sale_value - fees  # Add cash from sale
                        self.shares_held = 0.0
                        
                    elif self.position == -1:  # Close short position (buy back)
                        # Calculate cost to buy back shares
                        buy_cost = self.shares_held * current_price
                        fees = self._apply_trading_fees(buy_cost)
                        self.balance -= (buy_cost + fees)  # Subtract cash to buy back
                        self.shares_held = 0.0
                    
                    self.entry_price = 0.0

                # OPEN NEW POSITION (buy stocks)
                if action != 0:
                    max_invest = self.balance * self.max_position_pct
                    
                    if action == 1:  # Open long position (buy stocks)
                        self.shares_held = max_invest / current_price
                        fees = self._apply_trading_fees(max_invest)
                        self.balance -= (max_invest + fees)  # Subtract cash spent
                        self.entry_price = current_price
                        
                    elif action == -1 and self.allow_shorting:  # Open short position
                        # For shorting, we borrow shares and sell them
                        self.shares_held = max_invest / current_price
                        short_proceeds = self.shares_held * current_price
                        fees = self._apply_trading_fees(short_proceeds)
                        self.balance += (short_proceeds - fees)  # Add cash from short sale
                        self.entry_price = current_price
                        
                    else:  # Invalid short position when not allowed
                        action = 0
                        self.shares_held = 0.0
                
                self.position = action

            # Calculate CURRENT portfolio value (cash + stock value)
            portfolio_val = self._calculate_portfolio_value(current_price)
            
            # Update peak balance
            self.peak_balance = max(self.peak_balance, portfolio_val)
            
            # Update history with ACTUAL values
            current_date = self._get_current_date()
            stock_value = self.shares_held * current_price * (1 if self.position == 1 else -1 if self.position == -1 else 0)
            
            self.history['step'].append(self.step_idx)
            self.history['date'].append(current_date)
            self.history['price'].append(current_price)
            self.history['position'].append(self.position)
            self.history['portfolio_valuation'].append(portfolio_val)
            self.history['cash_balance'].append(self.balance)
            self.history['stock_value'].append(stock_value)
            self.history['shares_held'].append(self.shares_held)
            self.history['action_taken'].append(action if position_changed else 0)

            # Move to next step
            self.step_idx += 1
            done = self.step_idx >= len(self.df) - 1
            
            # Calculate reward based on actual portfolio change
            reward = self._calculate_reward(prev_portfolio_val, portfolio_val)
            
            info = {
                'portfolio_valuation': portfolio_val,
                'cash_balance': self.balance,
                'shares_held': self.shares_held,
                'stock_value': stock_value,
                'current_price': current_price,
                'position': self.position
            }
            
            return self._obs(), reward, done, info
            
        except Exception as e:
            print(f"Error in step(): {str(e)}")
            return self._obs(), 0.0, True, {'error': str(e)}

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate actual portfolio value: CASH + STOCK_VALUE."""
        if self.position == 1:  # Long
            stock_value = self.shares_held * current_price
            return self.balance + stock_value
            
        elif self.position == -1:  # Short  
            # For short positions, we owe shares so stock value is negative
            stock_value = self.shares_held * current_price * -1
            return self.balance + stock_value
            
        else:  # Neutral
            return self.balance

    def _obs(self) -> Dict[str, Any]:
        """Generate observation with actual portfolio breakdown."""
        current_price = self._get_price(self.step_idx)
        portfolio_val = self._calculate_portfolio_value(current_price)
        
        return {
            'price': current_price,
            'position': float(self.position),
            'cash': float(self.balance),
            'shares_held': float(self.shares_held),
            'stock_value': float(portfolio_val - self.balance),
            'portfolio_value': float(portfolio_val),
            'entry_price': float(self.entry_price),
            'step': self.step_idx,
            'date': self._get_current_date()
        }