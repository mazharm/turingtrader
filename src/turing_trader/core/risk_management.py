"""
Risk Management Module

This module handles all aspects of risk management:
- Position sizing based on risk tolerance
- Risk parameters for different risk profiles (Low, Medium, High)
- Maximum loss control per trade and per day
- Stop loss and take profit calculations
"""
import datetime
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

class RiskLevel(Enum):
    """Risk tolerance levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RiskManager:
    """
    Risk management system responsible for position sizing
    and risk parameters across different strategies
    """
    
    def __init__(self, ib_client, 
                risk_level: RiskLevel = RiskLevel.MEDIUM,
                max_daily_drawdown_pct: float = 2.0,
                max_position_pct: float = 10.0):
        """
        Initialize the risk manager
        
        Args:
            ib_client: Interactive Brokers client
            risk_level: Risk tolerance level
            max_daily_drawdown_pct: Maximum daily drawdown percentage
            max_position_pct: Maximum percentage of account for a single position
        """
        self.logger = logging.getLogger(__name__)
        self.ib_client = ib_client
        self.risk_level = risk_level
        
        # Set default risk parameters
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_position_pct = max_position_pct
        self.max_options_allocation_pct = 50.0  # Max % of account in options
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.starting_equity = 0.0
        self.current_equity = 0.0
        self.positions_risk = {}
        
        # Initialize risk parameters based on selected risk level
        self._configure_risk_level(risk_level)
    
    def _configure_risk_level(self, risk_level: RiskLevel) -> None:
        """
        Configure risk parameters based on risk level
        
        Args:
            risk_level: Selected risk level
        """
        if risk_level == RiskLevel.LOW:
            self.max_daily_drawdown_pct = 1.0  # 1% max daily loss
            self.max_position_pct = 5.0  # 5% max position size
            self.max_options_allocation_pct = 25.0  # 25% max in options
            self.stop_loss_pct = 25.0  # 25% stop loss on option positions
            self.take_profit_pct = 50.0  # 50% take profit target
            
        elif risk_level == RiskLevel.MEDIUM:
            self.max_daily_drawdown_pct = 2.0  # 2% max daily loss
            self.max_position_pct = 10.0  # 10% max position size
            self.max_options_allocation_pct = 50.0  # 50% max in options
            self.stop_loss_pct = 35.0  # 35% stop loss on option positions
            self.take_profit_pct = 70.0  # 70% take profit target
            
        elif risk_level == RiskLevel.HIGH:
            self.max_daily_drawdown_pct = 4.0  # 4% max daily loss
            self.max_position_pct = 20.0  # 20% max position size
            self.max_options_allocation_pct = 75.0  # 75% max in options
            self.stop_loss_pct = 50.0  # 50% stop loss on option positions
            self.take_profit_pct = 100.0  # 100% take profit target
            
        self.logger.info(f"Risk parameters configured for {risk_level.value} risk profile")
        self.logger.debug(f"Max daily drawdown: {self.max_daily_drawdown_pct}%, "
                         f"Max position size: {self.max_position_pct}%")
    
    def update_account_equity(self, current_equity: float) -> None:
        """
        Update current account equity
        
        Args:
            current_equity: Current account equity value
        """
        # Record starting equity at first update of the day
        if self.starting_equity == 0.0:
            self.starting_equity = current_equity
            
        self.current_equity = current_equity
        
        # Calculate daily P&L
        if self.starting_equity > 0:
            self.daily_pnl = current_equity - self.starting_equity
            daily_pnl_pct = (self.daily_pnl / self.starting_equity) * 100
            
            self.logger.debug(f"Updated equity: ${current_equity:.2f}, "
                             f"Daily P&L: ${self.daily_pnl:.2f} ({daily_pnl_pct:.2f}%)")
            
            # Check if we've exceeded daily drawdown limit
            if daily_pnl_pct <= -self.max_daily_drawdown_pct:
                self.logger.warning(f"RISK ALERT: Daily drawdown limit exceeded: {daily_pnl_pct:.2f}%")
    
    def reset_daily_tracking(self) -> None:
        """Reset daily tracking parameters (call at start of trading day)"""
        self.daily_pnl = 0.0
        self.starting_equity = 0.0
        self.logger.info("Daily risk tracking parameters reset")
    
    def calculate_position_size(self, 
                              allocation_factor: float, 
                              option_type: str = 'call',
                              option_price: float = None) -> int:
        """
        Calculate appropriate position size based on risk parameters
        
        Args:
            allocation_factor: Target allocation factor (0.0-1.0)
            option_type: Type of option ('call' or 'put')
            option_price: Option price (if known)
            
        Returns:
            int: Number of contracts to trade
        """
        if self.current_equity <= 0:
            return 0
            
        # Calculate maximum amount to allocate based on risk level
        max_allocation = self.current_equity * (self.max_position_pct / 100)
        
        # Apply allocation factor
        target_allocation = max_allocation * allocation_factor
        
        # For options, we need to determine number of contracts
        if option_price:
            # With known option price
            contract_value = option_price * 100  # Each option is for 100 shares
            num_contracts = int(target_allocation / contract_value)
            return max(1, num_contracts)  # At least 1 contract
        else:
            # Without known option price, estimate conservatively
            # Assume average option price of ~2% of underlying for SPY (adjust as needed)
            estimated_price = self.estimate_option_price(option_type)
            if estimated_price > 0:
                contract_value = estimated_price * 100
                num_contracts = int(target_allocation / contract_value)
                return max(1, num_contracts)
            else:
                # If we can't estimate, use a conservative default
                return 1
    
    def estimate_option_price(self, option_type: str) -> float:
        """
        Estimate option price based on option type and current market
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            float: Estimated option price
        """
        # In a real implementation, would use volatility and pricing models
        # This is a simplified placeholder
        
        # For SPY options, mid-range strikes often cost around 1-3% of underlying
        # Higher for longer-dated options
        # This is very approximate and should be replaced with real market data
        
        # Use a conservative estimate of 2% of underlying
        spy_price = 400.0  # Placeholder - would normally get real price
        return spy_price * 0.02
    
    def check_daily_risk_limit(self) -> bool:
        """
        Check if daily risk limit has been reached
        
        Returns:
            bool: True if trading should continue, False if risk limit reached
        """
        if self.starting_equity <= 0:
            return True
            
        daily_pnl_pct = (self.daily_pnl / self.starting_equity) * 100
        
        if daily_pnl_pct <= -self.max_daily_drawdown_pct:
            self.logger.warning(f"Daily risk limit reached: {daily_pnl_pct:.2f}% loss")
            return False
            
        return True
    
    def get_stop_loss_price(self, entry_price: float) -> float:
        """
        Calculate stop loss price based on entry price and risk parameters
        
        Args:
            entry_price: Position entry price
            
        Returns:
            float: Stop loss price
        """
        stop_loss_factor = self.stop_loss_pct / 100
        return entry_price * (1 - stop_loss_factor)
    
    def get_take_profit_price(self, entry_price: float) -> float:
        """
        Calculate take profit price based on entry price and risk parameters
        
        Args:
            entry_price: Position entry price
            
        Returns:
            float: Take profit price
        """
        take_profit_factor = self.take_profit_pct / 100
        return entry_price * (1 + take_profit_factor)
    
    def register_position(self, 
                        symbol: str, 
                        quantity: int, 
                        entry_price: float, 
                        strategy_type: str) -> Dict:
        """
        Register a new position for risk tracking
        
        Args:
            symbol: Symbol or identifier for the position
            quantity: Position quantity
            entry_price: Entry price
            strategy_type: Strategy type identifier
            
        Returns:
            dict: Risk parameters for the position
        """
        position_id = f"{symbol}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        stop_price = self.get_stop_loss_price(entry_price)
        take_profit = self.get_take_profit_price(entry_price)
        
        position_risk = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_time': datetime.datetime.now(),
            'stop_loss': stop_price,
            'take_profit': take_profit,
            'strategy': strategy_type,
            'risk_level': self.risk_level.value
        }
        
        self.positions_risk[position_id] = position_risk
        self.logger.info(f"Registered position {position_id} for risk management")
        self.logger.debug(f"Stop loss: ${stop_price:.2f}, Take profit: ${take_profit:.2f}")
        
        return position_risk
    
    def get_current_risk_metrics(self) -> Dict:
        """
        Get current risk metrics
        
        Returns:
            dict: Risk metrics summary
        """
        return {
            'risk_level': self.risk_level.value,
            'max_daily_drawdown_pct': self.max_daily_drawdown_pct,
            'max_position_pct': self.max_position_pct,
            'current_equity': self.current_equity,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.starting_equity * 100) if self.starting_equity > 0 else 0,
            'active_positions': len(self.positions_risk),
            'risk_limit_reached': not self.check_daily_risk_limit()
        }