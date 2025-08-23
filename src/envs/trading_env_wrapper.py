# src/envs/trading_env_wrapper.py
from typing import Optional, Dict, Any, Tuple, Union, List
import pandas as pd
import numpy as np
import warnings
import datetime
from datetime import datetime as dt
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
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 trading_fees: float = 0.0, allow_shorting: bool = False):
        self.df = df.copy()
        self._validate_dataframe()
        self.initial_balance = initial_balance
        self.trading_fees = trading_fees
        self.allow_shorting = allow_shorting
        self.reset()

    def _validate_dataframe(self):
        """Ensure DataFrame has required structure."""
        if not any('close' in col.lower() or 'Close' in col for col in self._get_column_names()):
            raise ValueError("DataFrame must contain 'Close' column")
        if len(self.df) < 2:
            raise ValueError("DataFrame must contain at least 2 rows")

    def _get_column_names(self) -> List[str]:
        """Get column names accounting for MultiIndex"""
        if isinstance(self.df.columns, pd.MultiIndex):
            return [col[0] for col in self.df.columns]
        return self.df.columns.tolist()

    def _get_price(self, idx: int, price_type: str = 'Close') -> float:
        """Safely extract price from DataFrame row."""
        try:
            # Try to find the column case-insensitively
            col_names = self._get_column_names()
            close_cols = [col for col in col_names if price_type.lower() in col.lower()]
            
            if not close_cols:
                # Fallback to any column with price data
                price_cols = [col for col in col_names if any(x in col.lower() for x in ['close', 'price', 'last'])]
                if not price_cols:
                    raise ValueError(f"No {price_type} column found")
                target_col = price_cols[0]
            else:
                target_col = close_cols[0]
            
            # Get the actual column name (for MultiIndex)
            if isinstance(self.df.columns, pd.MultiIndex):
                actual_col = [col for col in self.df.columns if col[0] == target_col][0]
            else:
                actual_col = target_col
            
            price_data = self.df.iloc[idx][actual_col]
            
            if isinstance(price_data, pd.Series):
                return float(price_data.iloc[0])
            return float(price_data)
            
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Error getting price at index {idx}: {str(e)}")

    def reset(self) -> Dict[str, float]:
        """Reset environment to initial state."""
        self.step_idx = 0
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.balance = self.initial_balance
        self.shares = 0.0
        self.entry_price = 0.0
        self.history = {
            'step': [], 
            'price': [], 
            'position': [], 
            'portfolio_valuation': [],
            'action_taken': [],
            'cash_balance': [],
            'shares_held': []
        }
        return self._obs()

    def _obs(self) -> Dict[str, float]:
        """Generate observation dictionary."""
        try:
            price = self._get_price(self.step_idx)
            portfolio_val = self._calculate_portfolio_value(price)
            
            return {
                'price': price,
                'position': float(self.position),
                'portfolio_value': float(portfolio_val),
                'cash': float(self.balance),
                'shares_held': float(self.shares)
            }
        except Exception as e:
            return {
                'price': 0.0,
                'position': float(self.position),
                'portfolio_value': float(self.balance),
                'cash': float(self.balance),
                'shares_held': 0.0
            }

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        if self.position == 1:  # Long
            return self.balance + (self.shares * current_price)
        elif self.position == -1:  # Short
            return self.balance - (self.shares * current_price)  # Negative value for short
        else:  # Neutral
            return self.balance

    def _apply_trading_fees(self, amount: float) -> float:
        """Calculate trading fees."""
        return abs(amount) * self.trading_fees

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """Execute one environment step with guaranteed 4-value return"""
        try:
            # Check if episode is done
            if self.step_idx >= len(self.df) - 1:
                obs = self._obs()
                return obs, 0.0, True, {'portfolio_valuation': obs.get('portfolio_value', self.balance)}

            # Get current price
            current_price = self._get_price(self.step_idx)
            prev_portfolio_val = self._calculate_portfolio_value(current_price)
            
            # Handle position changes
            position_changed = False
            if action != self.position:
                position_changed = True
                
                # Close existing position
                if self.position != 0:
                    if self.position == 1:  # Close long
                        sale_value = self.shares * current_price
                        fees = self._apply_trading_fees(sale_value)
                        self.balance += sale_value - fees
                    elif self.position == -1:  # Close short
                        buy_cost = self.shares * current_price
                        fees = self._apply_trading_fees(buy_cost)
                        self.balance -= (buy_cost + fees)
                    
                    self.shares = 0.0
                    self.entry_price = 0.0

                # Open new position
                if action != 0:
                    max_invest = self.balance
                    
                    if action == 1:  # Long
                        self.shares = max_invest / current_price
                        fees = self._apply_trading_fees(max_invest)
                        self.balance -= (max_invest + fees)
                        self.entry_price = current_price
                        
                    elif action == -1 and self.allow_shorting:  # Short
                        self.shares = max_invest / current_price
                        short_proceeds = self.shares * current_price
                        fees = self._apply_trading_fees(short_proceeds)
                        self.balance += (short_proceeds - fees)
                        self.entry_price = current_price
                    else:
                        # Invalid short position when not allowed
                        action = 0
                
                self.position = action

            # Calculate new portfolio value (using the same price for consistency)
            portfolio_val = self._calculate_portfolio_value(current_price)
            
            # Update history - CRITICAL: Update history BEFORE moving to next step
            self.history['step'].append(self.step_idx)
            self.history['price'].append(current_price)
            self.history['position'].append(self.position)
            self.history['portfolio_valuation'].append(portfolio_val)
            self.history['action_taken'].append(action if position_changed else 0)
            self.history['cash_balance'].append(self.balance)
            self.history['shares_held'].append(self.shares)

            # Calculate reward
            reward = (portfolio_val / prev_portfolio_val - 1) * 100 if prev_portfolio_val > 0 else 0.0
            
            # Move to next step
            self.step_idx += 1
            done = self.step_idx >= len(self.df) - 1
            
            info = {
                'portfolio_valuation': portfolio_val,
                'cash_balance': self.balance,
                'shares_held': self.shares,
                'current_price': current_price,
                'position': self.position,
                'step': self.step_idx
            }
            
            return self._obs(), float(reward), done, info

        except Exception as e:
            print(f"Error in step(): {str(e)}")
            # Return safe default values
            obs = self._obs()
            return obs, 0.0, True, {'error': str(e), 'portfolio_valuation': obs.get('portfolio_value', self.balance)}

def _prepare_for_external_env(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert DataFrame to external env format"""
    df_prepared = df.copy()
    
    # Handle MultiIndex columns by flattening
    if isinstance(df_prepared.columns, pd.MultiIndex):
        df_prepared.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                             for col in df_prepared.columns]
    
    # Standardize column names (case-insensitive)
    column_mapping = {}
    for col in df_prepared.columns:
        col_lower = col.lower()
        if 'close' in col_lower:
            column_mapping[col] = 'close'
        elif 'open' in col_lower:
            column_mapping[col] = 'open'
        elif 'high' in col_lower:
            column_mapping[col] = 'high'
        elif 'low' in col_lower:
            column_mapping[col] = 'low'
        elif 'volume' in col_lower:
            column_mapping[col] = 'volume'
    
    df_prepared = df_prepared.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['close', 'open', 'high', 'low']
    for col in required_cols:
        if col not in df_prepared.columns:
            if col == 'close' and len(df_prepared.columns) > 0:
                df_prepared['close'] = df_prepared.iloc[:, 0]
            else:
                df_prepared[col] = df_prepared.get('close', 1.0)
    
    # Add volume if missing
    if 'volume' not in df_prepared.columns:
        df_prepared['volume'] = 1.0
    
    return df_prepared

def create_trading_env_from_df(
    df: pd.DataFrame, 
    initial_balance: float = 10000.0, 
    trading_fees: float = 0.001,
    ticker: str = "AAPL",
    env_type: str = "simple",  # "simple" or "advanced"
    **kwargs
) -> Union[SimpleTradingEnv, Any]:
    """Create trading environment with robust reward handling."""
    
    if env_type.lower() == "advanced":
        return AdvancedTradingEnv(
            df=df, 
            initial_balance=initial_balance, 
            trading_fees=trading_fees,
            **kwargs
        )
    
    if not _HAS_EXTERNAL_ENV or TradingEnv is None:
        return SimpleTradingEnv(
            df=df, 
            initial_balance=initial_balance, 
            trading_fees=trading_fees,
            allow_shorting=kwargs.get('allow_shorting', False)
        )

    try:
        df_prepared = _prepare_for_external_env(df, ticker)
        
        def pnl_reward(history):
            """Calculate reward from portfolio values without dtype warnings"""
            if isinstance(history, dict) and "portfolio_valuation" in history:
                vals = pd.Series(history["portfolio_valuation"]).astype('float64')
                if len(vals) >= 2:
                    pct_changes = vals.pct_change().fillna(0)
                    pct_changes = pct_changes.replace([np.inf, -np.inf], 0)
                    return float(pct_changes.iloc[-1])
            return 0.0
        
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
        return SimpleTradingEnv(
            df=df, 
            initial_balance=initial_balance, 
            trading_fees=trading_fees,
            allow_shorting=kwargs.get('allow_shorting', False)
        )

class AdvancedTradingEnv(SimpleTradingEnv):
    """
    Advanced trading environment with better portfolio management and realistic trading mechanics.
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 trading_fees: float = 0.001, max_position_pct: float = 0.1,
                 allow_shorting: bool = False, interest_rate: float = 0.0003):
        
        super().__init__(df, initial_balance, trading_fees, allow_shorting)
        self.max_position_pct = max_position_pct
        self.interest_rate = interest_rate
        self.peak_balance = initial_balance
        self.total_trades = 0
        self.total_commission = 0.0

    def reset(self):
        """Reset environment to initial state."""
        super().reset()
        self.peak_balance = self.initial_balance
        self.total_trades = 0
        self.total_commission = 0.0
        
        # Enhanced history tracking
        self.history.update({
            'date': [],
            'stock_value': [],
            'max_drawdown': [],
            'daily_return': []
        })
        return self._obs()

    def _get_current_date(self) -> datetime:
        """Get current date from DataFrame."""
        try:
            if 'date' in self.df.columns or 'Date' in self.df.columns:
                date_col = 'date' if 'date' in self.df.columns else 'Date'
                return pd.to_datetime(self.df.iloc[self.step_idx][date_col])
            elif hasattr(self.df.index, 'name') and self.df.index.name in ['date', 'Date']:
                return pd.to_datetime(self.df.index[self.step_idx])
            elif hasattr(self.df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(self.df.index):
                return pd.to_datetime(self.df.index[self.step_idx])
            else:
                return dt.now()
        except:
            return dt.now()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one environment step with proper portfolio tracking."""
        try:
            if self.step_idx >= len(self.df) - 1:
                final_obs = self._obs()
                return final_obs, 0.0, True, {'portfolio_valuation': final_obs['portfolio_value']}

            current_price = self._get_price(self.step_idx)
            prev_portfolio_val = self._calculate_portfolio_value(current_price)
            
            # Handle position changes with max position limit
            position_changed = False
            if action != self.position:
                position_changed = True
                self.total_trades += 1
                
                # Close existing position
                if self.position != 0:
                    close_value = self.shares * current_price
                    fees = self._apply_trading_fees(close_value)
                    self.total_commission += fees
                    
                    if self.position == 1:  # Close long
                        self.balance += close_value - fees
                    elif self.position == -1:  # Close short
                        self.balance -= (close_value + fees)
                    
                    self.shares = 0.0
                    self.entry_price = 0.0

                # Open new position with position sizing
                if action != 0:
                    max_invest = self.balance * self.max_position_pct
                    
                    if action == 1:  # Long
                        self.shares = max_invest / current_price
                        fees = self._apply_trading_fees(max_invest)
                        self.total_commission += fees
                        self.balance -= (max_invest + fees)
                        self.entry_price = current_price
                        
                    elif action == -1 and self.allow_shorting:  # Short
                        self.shares = max_invest / current_price
                        short_proceeds = self.shares * current_price
                        fees = self._apply_trading_fees(short_proceeds)
                        self.total_commission += fees
                        self.balance += (short_proceeds - fees)
                        self.entry_price = current_price
                        
                    else:  # Invalid short position
                        action = 0
                
                self.position = action

            # Calculate portfolio value and update peak
            portfolio_val = self._calculate_portfolio_value(current_price)
            self.peak_balance = max(self.peak_balance, portfolio_val)
            
            # Calculate drawdown
            drawdown = (portfolio_val - self.peak_balance) / self.peak_balance if self.peak_balance > 0 else 0
            
            # Calculate stock value for proper visualization
            if self.position == 1:  # Long
                stock_value = self.shares * current_price
            elif self.position == -1:  # Short
                stock_value = -self.shares * current_price  # Negative for short
            else:
                stock_value = 0.0
            
            # Update history - CRITICAL: Update BEFORE moving to next step
            current_date = self._get_current_date()
            
            self.history['step'].append(self.step_idx)
            self.history['date'].append(current_date)
            self.history['price'].append(current_price)
            self.history['position'].append(self.position)
            self.history['portfolio_valuation'].append(portfolio_val)
            self.history['cash_balance'].append(self.balance)
            self.history['shares_held'].append(self.shares)
            self.history['stock_value'].append(stock_value)
            self.history['action_taken'].append(action if position_changed else 0)
            self.history['max_drawdown'].append(drawdown)

            # Calculate daily return
            daily_return = (portfolio_val / prev_portfolio_val - 1) if prev_portfolio_val > 0 else 0
            self.history['daily_return'].append(daily_return)

            # Move to next step
            self.step_idx += 1
            done = self.step_idx >= len(self.df) - 1
            
            # Risk-adjusted reward
            reward = daily_return * 100  # Percentage return
            if drawdown < -0.05:  # Penalize large drawdowns
                reward *= (1 + drawdown)  # Reduce reward by drawdown percentage

            info = {
                'portfolio_valuation': portfolio_val,
                'cash_balance': self.balance,
                'shares_held': self.shares,
                'stock_value': stock_value,
                'current_price': current_price,
                'position': self.position,
                'drawdown': drawdown,
                'daily_return': daily_return,
                'total_trades': self.total_trades,
                'total_commission': self.total_commission
            }
            
            return self._obs(), float(reward), done, info
            
        except Exception as e:
            print(f"Error in AdvancedTradingEnv step(): {str(e)}")
            obs = self._obs()
            return obs, 0.0, True, {'error': str(e), 'portfolio_valuation': obs.get('portfolio_value', self.balance)}