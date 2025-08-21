from .base import BaseAgent
from typing import List, Tuple
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta

class StatisticalMeanAgent(BaseAgent):
    def __init__(self, 
                 long_threshold: float = 0.10,    # 10% below mean for long
                 short_threshold: float = 0.15,   # 15% above mean for short  
                 min_data_points: int = 100,      # Minimum data to start trading
                 n_splits: int = 10,             # Number of cross-validation splits
                 trading_frequency: str = '1W'):  # Trading frequency: '1W', '2W', '1M'
        
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.min_data_points = min_data_points
        self.n_splits = n_splits
        self.trading_frequency = trading_frequency
        
        self.price_history = []
        self.current_position = 0
        self.entry_price = 0
        self.entry_prices = []  # Track multiple entry points
        self.shares_held = 0
        self.cross_val_means = []
        self.last_trade_date = None
        self.available_cash = 10000.0  # Starting cash
        self.total_invested = 0.0
        
    def reset(self):
        """Reset agent state"""
        self.price_history = []
        self.current_position = 0
        self.entry_price = 0
        self.entry_prices = []
        self.shares_held = 0
        self.cross_val_means = []
        self.last_trade_date = None
        self.available_cash = 10000.0
        self.total_invested = 0.0
        
    def _calculate_cross_val_means(self, data: List[float]) -> List[float]:
        """Calculate means for n equal parts of the dataset"""
        if len(data) < self.min_data_points:
            return []
            
        # Split data into n equal parts
        split_size = len(data) // self.n_splits
        means = []
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            if end_idx > len(data):
                end_idx = len(data)
                
            split_data = data[start_idx:end_idx]
            if split_data:  # Ensure we have data in this split
                means.append(np.mean(split_data))
                
        return means
        
    def _get_current_mean(self) -> float:
        """Get weighted mean of cross-validation means"""
        if not self.cross_val_means:
            return 0
            
        # Recent splits get higher weight
        weights = np.linspace(0.5, 1.5, len(self.cross_val_means))
        weighted_mean = np.average(self.cross_val_means, weights=weights)
        return weighted_mean
        
    def _calculate_trading_signals(self, current_price: float, current_date: datetime) -> dict:
        """Calculate trading signals with frequency control"""
        if not self.cross_val_means:
            return {'buy': False, 'sell': False, 'hold': True}
            
        current_mean = self._get_current_mean()
        
        # Check if we should trade based on frequency
        should_trade = self._should_trade_today(current_date)
        
        signals = {
            'buy': False,
            'sell': False, 
            'hold': True,
            'current_price': current_price,
            'current_mean': current_mean,
            'deviation_pct': (current_price - current_mean) / current_mean * 100,
            'should_trade': should_trade
        }
        
        # Only generate signals on trading days
        if not should_trade:
            return signals
            
        # Buy signal: price is significantly below mean AND we have cash
        if (current_price <= current_mean * (1 - self.long_threshold) and 
            self.available_cash > current_price):
            signals['buy'] = True
            signals['hold'] = False
            
        # NEVER SELL - Only buy more when prices are low
        # We hold through downturns and only add more positions
        
        return signals
        
    def _should_trade_today(self, current_date: datetime) -> bool:
        """Check if today is a trading day based on frequency"""
        if self.last_trade_date is None:
            return True
            
        if self.trading_frequency == '1W':
            return (current_date - self.last_trade_date).days >= 7
        elif self.trading_frequency == '2W':
            return (current_date - self.last_trade_date).days >= 14
        elif self.trading_frequency == '1M':
            return (current_date - self.last_trade_date).days >= 30
        else:
            return True  # Daily trading by default
            
    def _extract_price_and_date(self, observation: dict) -> Tuple[float, datetime]:
        """Extract both price and date from observation"""
        try:
            price = None
            date = datetime.now()  # Default to current date
            
            # Extract price
            price_data = observation.get('price')
            if isinstance(price_data, (float, int)):
                price = float(price_data)
            elif isinstance(price_data, pd.Series):
                price = float(price_data.iloc[-1])
            elif isinstance(price_data, np.ndarray):
                price = float(price_data.item()) if price_data.size == 1 else float(price_data[-1])
            else:
                price = float(price_data)
                
            # Extract date if available
            if 'date' in observation:
                date = pd.to_datetime(observation['date'])
            elif 'Date' in observation:
                date = pd.to_datetime(observation['Date'])
                
            return price, date
            
        except (KeyError, ValueError, TypeError):
            return None, datetime.now()
        
    def act(self, observation: dict) -> int:
        """
        Generate trading action - Only buy, never sell
        Returns: 1 (buy), 0 (hold), -1 is disabled
        """
        price, current_date = self._extract_price_and_date(observation)
        if price is None:
            return 0
            
        # Add to price history
        self.price_history.append(price)
        
        # Wait for sufficient data
        if len(self.price_history) < self.min_data_points:
            return 0
            
        # Update cross-validation means periodically
        if len(self.price_history) % 50 == 0:  # Update every 50 steps
            self.cross_val_means = self._calculate_cross_val_means(self.price_history)
            
        # Calculate trading signals
        signals = self._calculate_trading_signals(price, current_date)
        
        # Execute BUY only (never sell)
        if signals['buy'] and signals['should_trade']:
            # Calculate how many shares to buy (use 20% of available cash)
            investment_amount = min(self.available_cash * 0.2, self.available_cash)
            shares_to_buy = investment_amount / price
            
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.entry_prices.append(price)
                self.available_cash -= investment_amount
                self.total_invested += investment_amount
                self.last_trade_date = current_date
                self.current_position = 1  # Always long
                return 1  # Buy signal
                
        return 0  # Hold - never sell
        
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        stock_value = self.shares_held * current_price
        return stock_value + self.available_cash
        
    def get_average_entry_price(self) -> float:
        """Calculate average entry price across all purchases"""
        if not self.entry_prices:
            return 0
        return sum(self.entry_prices) / len(self.entry_prices)
        
    def get_performance_stats(self, current_price: float) -> dict:
        """Get performance statistics"""
        total_value = self.get_portfolio_value(current_price)
        avg_entry = self.get_average_entry_price()
        
        return {
            'total_value': total_value,
            'available_cash': self.available_cash,
            'shares_held': self.shares_held,
            'total_invested': self.total_invested,
            'average_entry_price': avg_entry,
            'current_unrealized_pnl': (current_price - avg_entry) * self.shares_held if avg_entry > 0 else 0,
            'current_price': current_price
        }


class EnhancedStatisticalAgent(StatisticalMeanAgent):
    def __init__(self, 
                 long_threshold: float = 0.10,
                 short_threshold: float = 0.15,
                 min_data_points: int = 100,
                 n_splits: int = 10,
                 trading_frequency: str = '1W',
                 volatility_filter: bool = True,
                 max_cash_usage: float = 0.2):  # Max 20% of cash per trade
        
        super().__init__(long_threshold, short_threshold, min_data_points, n_splits, trading_frequency)
        
        self.volatility_filter = volatility_filter
        self.max_cash_usage = max_cash_usage
        self.volatility_history = deque(maxlen=50)
        
    def _calculate_volatility(self) -> float:
        """Calculate recent volatility"""
        if len(self.price_history) < 2:
            return 0
            
        returns = np.diff(self.price_history[-50:]) / np.array(self.price_history[-51:-1])
        return np.std(returns) if len(returns) > 0 else 0
        
    def _calculate_trading_signals(self, current_price: float, current_date: datetime) -> dict:
        """Enhanced signals with volatility filtering"""
        signals = super()._calculate_trading_signals(current_price, current_date)
        
        if self.volatility_filter and signals['buy']:
            volatility = self._calculate_volatility()
            # Avoid buying during extremely high volatility
            if volatility > 0.08:  # 8% daily volatility threshold
                signals['buy'] = False
                signals['hold'] = True
                
        return signals
        
    def act(self, observation: dict) -> int:
        price, current_date = self._extract_price_and_date(observation)
        if price is None:
            return 0
            
        self.price_history.append(price)
        
        if len(self.price_history) < self.min_data_points:
            return 0
            
        if len(self.price_history) % 50 == 0:
            self.cross_val_means = self._calculate_cross_val_means(self.price_history)
            
        signals = self._calculate_trading_signals(price, current_date)
        
        # Enhanced buy logic with cash management
        if signals['buy'] and signals['should_trade']:
            # Use smaller position size during high volatility
            volatility = self._calculate_volatility()
            cash_percentage = self.max_cash_usage
            
            if volatility > 0.05:  # Reduce position size in high vol
                cash_percentage *= 0.5
                
            investment_amount = min(self.available_cash * cash_percentage, self.available_cash)
            shares_to_buy = investment_amount / price
            
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.entry_prices.append(price)
                self.available_cash -= investment_amount
                self.total_invested += investment_amount
                self.last_trade_date = current_date
                self.current_position = 1
                return 1  # Buy
                
        return 0  # Hold - never sell
    
from .base import BaseAgent
from typing import List, Tuple
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta

class StatisticalMeanAgent(BaseAgent):
    def __init__(self, 
                 long_threshold: float = 0.10,    # 10% below mean for long
                 min_data_points: int = 100,      # Minimum data to start trading
                 n_splits: int = 10):             # Number of cross-validation splits
        
        self.long_threshold = long_threshold
        self.min_data_points = min_data_points
        self.n_splits = n_splits
        
        self.price_history = []
        self.shares_held = 0.0
        self.entry_prices = []
        self.available_cash = 10000.0  # Fixed initial balance
        self.total_invested = 0.0
        self.cross_val_means = []
        self.last_action = 0
        
    def reset(self):
        """Reset agent state"""
        self.price_history = []
        self.shares_held = 0.0
        self.entry_prices = []
        self.available_cash = 10000.0  # Reset to fixed amount
        self.total_invested = 0.0
        self.cross_val_means = []
        self.last_action = 0
        
    def set_initial_balance(self, initial_balance: float):
        """Set initial balance separately"""
        self.available_cash = initial_balance
        
    def _calculate_cross_val_means(self) -> List[float]:
        """Calculate means for n equal parts of the dataset"""
        if len(self.price_history) < self.min_data_points:
            return []
            
        # Split data into n equal parts
        split_size = len(self.price_history) // self.n_splits
        means = []
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            if end_idx > len(self.price_history):
                end_idx = len(self.price_history)
                
            split_data = self.price_history[start_idx:end_idx]
            if split_data:  # Ensure we have data in this split
                means.append(np.mean(split_data))
                
        return means
        
    def _get_current_mean(self) -> float:
        """Get weighted mean of cross-validation means"""
        if not self.cross_val_means:
            return 0
            
        # Recent splits get higher weight
        weights = np.linspace(0.5, 1.5, len(self.cross_val_means))
        weighted_mean = np.average(self.cross_val_means, weights=weights)
        return weighted_mean
        
    def _should_buy(self, current_price: float) -> bool:
        """Determine if we should buy based on statistical mean"""
        if not self.cross_val_means or self.available_cash < 100:
            return False
            
        current_mean = self._get_current_mean()
        if current_mean == 0:
            return False
            
        # Buy if price is significantly below mean AND we have cash
        price_ratio = current_price / current_mean
        return price_ratio <= (1 - self.long_threshold)
        
    def _calculate_shares_to_buy(self, current_price: float) -> float:
        """Calculate how many shares to buy"""
        if self.available_cash < 100:
            return 0
            
        # Use 50% of available cash per trade
        investment_amount = self.available_cash * 0.5
        return investment_amount / current_price
        
    def act(self, observation: dict) -> int:
        """
        Generate trading action - Only buy, never sell
        Returns: 1 (buy), 0 (hold)
        """
        price = self._extract_price(observation)
        if price is None:
            return 0
            
        # Add to price history
        self.price_history.append(price)
        
        # Wait for sufficient data
        if len(self.price_history) < self.min_data_points:
            return 0
            
        # Update cross-validation means periodically
        if len(self.price_history) % 30 == 0:  # Update every 30 steps
            self.cross_val_means = self._calculate_cross_val_means()
            
        # Check if we should buy
        should_buy = self._should_buy(price)
        
        if should_buy:
            shares_to_buy = self._calculate_shares_to_buy(price)
            
            if shares_to_buy > 0:
                # Execute buy
                investment_amount = shares_to_buy * price
                self.shares_held += shares_to_buy
                self.entry_prices.append(price)
                self.available_cash -= investment_amount
                self.total_invested += investment_amount
                self.last_action = 1
                return 1  # Buy signal
                
        self.last_action = 0
        return 0  # Hold
        
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        stock_value = self.shares_held * current_price
        return stock_value + self.available_cash
        
    def get_performance_stats(self, current_price: float) -> dict:
        """Get detailed performance statistics"""
        total_value = self.get_portfolio_value(current_price)
        avg_entry = np.mean(self.entry_prices) if self.entry_prices else 0
        
        return {
            'total_value': total_value,
            'available_cash': self.available_cash,
            'shares_held': self.shares_held,
            'total_invested': self.total_invested,
            'average_entry_price': avg_entry,
            'unrealized_pnl': (current_price - avg_entry) * self.shares_held if avg_entry > 0 else 0,
            'total_return_pct': (total_value / 10000.0 - 1) * 100  # Based on fixed 10k
        }

    def _extract_price(self, observation: dict) -> float:
        """Extract price from observation"""
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


class BullMarketStatisticalAgent(StatisticalMeanAgent):
    def __init__(self, 
                 long_threshold: float = 0.05,    # Only 5% below mean (more sensitive)
                 min_data_points: int = 20,       # Start trading sooner
                 n_splits: int = 3):              # Fewer splits for faster adaptation
        
        # Call parent constructor without initial_balance
        super().__init__(
            long_threshold=long_threshold,
            min_data_points=min_data_points,
            n_splits=n_splits
        )
        self.position_size = 0.8  # Use 80% of cash
        
    def _should_buy(self, current_price: float) -> bool:
        """More sensitive buying for bull markets"""
        if not self.cross_val_means or self.available_cash < 100:
            return False
            
        current_mean = self._get_current_mean()
        if current_mean == 0:
            return False
            
        # More sensitive: buy if not extremely overvalued
        price_ratio = current_price / current_mean
        return price_ratio <= 1.15  # Buy if not more than 15% above mean
        
    def _calculate_shares_to_buy(self, current_price: float) -> float:
        """More aggressive position sizing"""
        if self.available_cash < 100:
            return 0
            
        investment_amount = self.available_cash * self.position_size
        return investment_amount / current_price


class DebugAgent(BaseAgent):
    """Simple agent that buys on first few steps for testing"""
    def __init__(self):
        self.buy_counter = 0
        
    def reset(self):
        self.buy_counter = 0
        
    def act(self, observation):
        self.buy_counter += 1
        # Buy on first 5 steps to test
        if self.buy_counter <= 5:
            return 1
        return 0
        
    def _extract_price(self, observation: dict) -> float:
        """Extract price from observation"""
        try:
            return float(observation['price'])
        except (KeyError, ValueError, TypeError):
            return None