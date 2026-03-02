# src/agents/statistical_agent.py   
from .base import BaseAgent
from typing import List
import pandas as pd
import numpy as np

class StatisticalMeanAgent(BaseAgent):
    def __init__(self, 
                 long_threshold: float = 0.07,    # 7% below mean for buy
                 short_threshold: float = 0.07,   # 7% above mean for sell
                 min_data_points: int = 20,       # Faster startup
                 n_splits: int = 5,              # Faster calculation
                 position_size: float = 0.3,      # 30% of cash per trade
                 min_trade_amount: float = 100.0): # Minimum trade size
        
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.min_data_points = min_data_points
        self.n_splits = n_splits
        self.position_size = position_size
        self.min_trade_amount = min_trade_amount
        
        self.price_history = []
        self.shares_held = 0.0
        self.entry_prices = []
        self.available_cash = 10000.0
        self.total_invested = 0.0
        self.cross_val_means = []
        self.last_action = 0
        self.current_position = 0  # 0: neutral, 1: long, -1: short
        
    def reset(self):
        """Reset agent state"""
        self.price_history = []
        self.shares_held = 0.0
        self.entry_prices = []
        self.available_cash = 10000.0
        self.total_invested = 0.0
        self.cross_val_means = []
        self.last_action = 0
        self.current_position = 0
        
    def set_initial_balance(self, initial_balance: float):
        """Set initial balance"""
        self.available_cash = initial_balance
        
    def _calculate_cross_val_means(self) -> List[float]:
        """Calculate rolling means for recent data"""
        if len(self.price_history) < self.min_data_points:
            return []
            
        # Use recent data for means calculation
        recent_data = self.price_history[-self.min_data_points:]
        split_size = len(recent_data) // self.n_splits
        means = []
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            split_data = recent_data[start_idx:end_idx]
            if split_data:
                means.append(np.mean(split_data))
                
        return means if means else [np.mean(recent_data)]
        
    def _get_current_mean(self) -> float:
        """Get current mean value"""
        if not self.cross_val_means:
            return 0
        return np.mean(self.cross_val_means)
        
    def _calculate_trading_signals(self, current_price: float) -> dict:
        """Calculate both buy and sell signals continuously"""
        if not self.cross_val_means:
            return {'buy': False, 'sell': False, 'hold': True}
            
        current_mean = self._get_current_mean()
        if current_mean == 0:
            return {'buy': False, 'sell': False, 'hold': True}
            
        price_ratio = current_price / current_mean
        signals = {
            'buy': False,
            'sell': False,
            'hold': True,
            'current_price': current_price,
            'current_mean': current_mean,
            'deviation_pct': (price_ratio - 1) * 100
        }
        
        # BUY signal: price significantly below mean AND we have cash
        if (price_ratio <= (1 - self.long_threshold) and 
            self.available_cash >= self.min_trade_amount):
            signals['buy'] = True
            signals['hold'] = False
            
        # SELL signal: price significantly above mean AND we have shares
        elif (price_ratio >= (1 + self.short_threshold) and 
              self.shares_held > 0 and self.current_position == 1):
            signals['sell'] = True
            signals['hold'] = False
            
        return signals
        
    def _calculate_position_size(self, current_price: float, signal_type: str) -> float:
        """Calculate how many shares to buy/sell"""
        if signal_type == 'buy':
            if self.available_cash < self.min_trade_amount:
                return 0
            investment = min(self.available_cash * self.position_size, self.available_cash)
            return investment / current_price
            
        elif signal_type == 'sell':
            if self.shares_held <= 0:
                return 0
            # Sell all shares for now
            return self.shares_held
            
        return 0
        
    def act(self, observation: dict) -> int:
        """
        Continuous trading based on latest price points
        Returns: 1 (buy), -1 (sell), 0 (hold)
        """
        price = self._extract_price(observation)
        if price is None:
            return 0
            
        # Add to price history
        self.price_history.append(price)
        
        # Update means periodically
        if len(self.price_history) % 20 == 0 or len(self.price_history) < self.min_data_points:
            self.cross_val_means = self._calculate_cross_val_means()
            
        # Wait for sufficient data
        if len(self.price_history) < self.min_data_points:
            return 0
            
        # Calculate trading signals based on LATEST price
        signals = self._calculate_trading_signals(price)
        
        # EXECUTE BUY
        if signals['buy'] and self.available_cash >= self.min_trade_amount:
            shares_to_buy = self._calculate_position_size(price, 'buy')
            if shares_to_buy > 0:
                investment = shares_to_buy * price
                self.shares_held += shares_to_buy
                self.entry_prices.append(price)
                self.available_cash -= investment
                self.total_invested += investment
                self.current_position = 1
                self.last_action = 1
                return 1  # Buy signal
                
        # EXECUTE SELL
        elif signals['sell'] and self.shares_held > 0:
            shares_to_sell = self._calculate_position_size(price, 'sell')
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price
                self.shares_held -= shares_to_sell
                self.available_cash += proceeds
                self.current_position = 0
                self.last_action = -1
                return -1  # Sell signal
                
        # HOLD - no action
        self.last_action = 0
        return 0
        
    def get_current_value(self, current_price: float) -> float:
        """Get current portfolio value"""
        return self.available_cash + (self.shares_held * current_price)
        
    def get_status(self, current_price: float) -> dict:
        """Get current agent status"""
        return {
            'cash': self.available_cash,
            'shares': self.shares_held,
            'position': self.current_position,
            'portfolio_value': self.get_current_value(current_price),
            'last_action': self.last_action,
            'data_points': len(self.price_history)
        }



class ContinuousTradingAgent(StatisticalMeanAgent):
    """Continuous trading with profit protection"""
    def __init__(self, 
                 long_threshold: float = 0.05,    # 5% below mean
                 short_threshold: float = 0.05,   # 5% above mean  
                 position_size: float = 0.5,      # 50% of cash per trade
                 min_data_points: int = 30,
                 stop_loss_pct: float = 0.10,     # 10% max loss before forced sell
                 min_profit_pct: float = 0.02,    # 2% minimum profit to sell
                 allow_loss_selling: bool = False): # Whether to allow selling at loss
        
        super().__init__(
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            position_size=position_size,
            min_data_points=min_data_points
        )
        
        self.stop_loss_pct = stop_loss_pct
        self.min_profit_pct = min_profit_pct
        self.allow_loss_selling = allow_loss_selling
        self.entry_price = 0.0  # Track average entry price
        
    def _calculate_average_entry_price(self) -> float:
        """Calculate weighted average entry price"""
        if not self.entry_prices or self.shares_held <= 0:
            return 0.0
            
        # For simplicity, use simple average
        # In production, you might want weighted average by purchase size
        return np.mean(self.entry_prices)
        
    def _calculate_profit_loss(self, current_price: float) -> float:
        """Calculate current profit/loss percentage"""
        avg_entry = self._calculate_average_entry_price()
        if avg_entry == 0:
            return 0.0
            
        return (current_price - avg_entry) / avg_entry
        
    def _should_sell_based_on_profit(self, current_price: float) -> bool:
        """Determine if selling is allowed based on profit/loss"""
        if self.shares_held <= 0:
            return False
            
        profit_pct = self._calculate_profit_loss(current_price)
        
        # 1. STOP LOSS: Force sell if loss exceeds threshold
        if profit_pct <= -self.stop_loss_pct:
            print(f"STOP LOSS triggered: {profit_pct:.1%} loss")
            return True
            
        # 2. MIN PROFIT: Only sell if we have minimum profit
        if profit_pct >= self.min_profit_pct:
            print(f"Profit target reached: {profit_pct:.1%} gain")
            return True
            
        # 3. ALLOW LOSS SELLING: If explicitly enabled
        if self.allow_loss_selling and profit_pct < 0:
            print(f"Selling at loss (allowed): {profit_pct:.1%}")
            return True
            
        # 4. PREVENT SELLING AT LOSS
        print(f"Prevented selling: {profit_pct:.1%} (below {self.min_profit_pct:.1%} profit)")
        return False
        
    def _calculate_trading_signals(self, current_price: float) -> dict:
        """More sensitive signals with profit protection"""
        signals = super()._calculate_trading_signals(current_price)
        
        # Override sell signal with profit check
        if signals['sell']:
            signals['sell'] = self._should_sell_based_on_profit(current_price)
        
        # Additional: Buy on any dip when we're not invested
        if (not signals['buy'] and self.current_position == 0 and 
            self.available_cash >= self.min_trade_amount):
            recent_avg = np.mean(self.price_history[-20:]) if len(self.price_history) >= 20 else 0
            if recent_avg > 0 and current_price < recent_avg * 0.98:
                signals['buy'] = True
                signals['hold'] = False
                
        return signals
        
    def act(self, observation: dict) -> int:
        """Enhanced act method with profit protection"""
        price = self._extract_price(observation)
        if price is None:
            return 0
            
        self.price_history.append(price)
        
        if len(self.price_history) % 20 == 0 or len(self.price_history) < self.min_data_points:
            self.cross_val_means = self._calculate_cross_val_means()
            
        if len(self.price_history) < self.min_data_points:
            return 0
            
        signals = self._calculate_trading_signals(price)
        
        # EXECUTE BUY
        if signals['buy'] and self.available_cash >= self.min_trade_amount:
            shares_to_buy = self._calculate_position_size(price, 'buy')
            if shares_to_buy > 0:
                investment = shares_to_buy * price
                self.shares_held += shares_to_buy
                self.entry_prices.append(price)  # Track entry price
                self.available_cash -= investment
                self.total_invested += investment
                self.current_position = 1
                self.last_action = 1
                print(f"BUY: {shares_to_buy:.2f} shares at ${price:.2f}")
                return 1
                
        # EXECUTE SELL (with profit protection)
        elif signals['sell'] and self.shares_held > 0:
            profit_pct = self._calculate_profit_loss(price)
            shares_to_sell = self._calculate_position_size(price, 'sell')
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price
                self.shares_held -= shares_to_sell
                self.available_cash += proceeds
                
                # Remove sold shares from entry prices (FIFO)
                shares_remaining = self.shares_held
                self.entry_prices = self.entry_prices[-int(shares_remaining):] if shares_remaining > 0 else []
                
                self.current_position = 0 if self.shares_held <= 0 else 1
                self.last_action = -1
                print(f"SELL: {shares_to_sell:.2f} shares at ${price:.2f} ({profit_pct:+.1%})")
                return -1
                
        self.last_action = 0
        return 0
        
    def get_status(self, current_price: float) -> dict:
        """Enhanced status with profit information"""
        status = super().get_status(current_price)
        profit_pct = self._calculate_profit_loss(current_price)
        
        status.update({
            'profit_pct': profit_pct * 100,  # as percentage
            'stop_loss_pct': self.stop_loss_pct * 100,
            'min_profit_pct': self.min_profit_pct * 100,
            'average_entry_price': self._calculate_average_entry_price(),
            'current_profit': profit_pct * self.total_invested if self.total_invested > 0 else 0
        })
        return status


class BullMarketStatisticalAgent(StatisticalMeanAgent):
    def __init__(self,
                 long_threshold: float = 0.05,    # Only 5% below mean (more sensitive)
                 min_data_points: int = 20,       # Start trading sooner
                 n_splits: int = 3):              # Fewer splits for faster adaptation

        super().__init__(
            long_threshold=long_threshold,
            min_data_points=min_data_points,
            n_splits=n_splits,
            position_size=0.8  # Use 80% of cash
        )

    def _calculate_trading_signals(self, current_price: float) -> dict:
        """More sensitive buying for bull markets — buy if not extremely overvalued."""
        signals = super()._calculate_trading_signals(current_price)

        # Override buy signal: allow buying up to 15% above mean
        if (not signals['buy'] and self.available_cash >= self.min_trade_amount
                and self.cross_val_means):
            current_mean = self._get_current_mean()
            if current_mean > 0:
                price_ratio = current_price / current_mean
                if price_ratio <= 1.15:
                    signals['buy'] = True
                    signals['hold'] = False

        return signals

