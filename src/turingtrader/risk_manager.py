"""Risk management module for the TuringTrader."""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Enumeration of risk levels for the trading strategy."""
    
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    AGGRESSIVE = 4


class RiskManager:
    """Risk management system for the TuringTrader.
    
    This class handles risk assessment and position sizing based on
    user-defined risk profiles and market conditions.
    """
    
    def __init__(
        self,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        max_position_size_pct: float = 0.05,
        max_daily_loss_pct: float = 0.02,
        stop_loss_pct: float = 0.15,
        account_balance: Optional[float] = None,
    ):
        """Initialize the risk manager.
        
        Args:
            risk_level (RiskLevel): Risk level for the strategy
            max_position_size_pct (float): Maximum position size as % of account
            max_daily_loss_pct (float): Maximum daily loss as % of account
            stop_loss_pct (float): Stop loss percentage for positions
            account_balance (float, optional): Initial account balance
        """
        self.risk_level = risk_level
        self._configure_risk_profile(risk_level)
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.stop_loss_pct = stop_loss_pct
        self.account_balance = account_balance or 0.0
        self.daily_pnl = 0.0
        
    def _configure_risk_profile(self, risk_level: RiskLevel) -> None:
        """Configure risk parameters based on risk level.
        
        Args:
            risk_level (RiskLevel): Risk level to configure for
        """
        # Base risk parameters
        if risk_level == RiskLevel.LOW:
            self.max_position_size_pct = 0.02
            self.max_daily_loss_pct = 0.01
            self.stop_loss_pct = 0.10
            self.max_options_per_trade = 1
            self.preferred_delta = 0.30
            self.max_dte = 14  # days to expiration
            
        elif risk_level == RiskLevel.MEDIUM:
            self.max_position_size_pct = 0.05
            self.max_daily_loss_pct = 0.02
            self.stop_loss_pct = 0.15
            self.max_options_per_trade = 2
            self.preferred_delta = 0.40
            self.max_dte = 30
            
        elif risk_level == RiskLevel.HIGH:
            self.max_position_size_pct = 0.10
            self.max_daily_loss_pct = 0.04
            self.stop_loss_pct = 0.25
            self.max_options_per_trade = 5
            self.preferred_delta = 0.50
            self.max_dte = 45
            
        elif risk_level == RiskLevel.AGGRESSIVE:
            self.max_position_size_pct = 0.15
            self.max_daily_loss_pct = 0.06
            self.stop_loss_pct = 0.35
            self.max_options_per_trade = 10
            self.preferred_delta = 0.60
            self.max_dte = 60
    
    def set_risk_level(self, risk_level: RiskLevel) -> None:
        """Set a new risk level and reconfigure parameters.
        
        Args:
            risk_level (RiskLevel): New risk level to set
        """
        self.risk_level = risk_level
        self._configure_risk_profile(risk_level)
        logger.info(f"Risk level set to {risk_level.name}")
    
    def update_account_balance(self, balance: float) -> None:
        """Update the current account balance.
        
        Args:
            balance (float): New account balance
        """
        old_balance = self.account_balance
        self.account_balance = balance
        logger.info(f"Account balance updated: {old_balance} -> {balance}")
        
        # Reset daily P&L when account balance is updated
        # (typically done at the start of the trading day)
        if old_balance != balance:
            self.daily_pnl = 0.0
    
    def calculate_position_size(
        self, 
        price: float, 
        volatility_factor: float = 1.0
    ) -> Tuple[int, float]:
        """Calculate the position size for a trade.
        
        Args:
            price (float): Price of the security
            volatility_factor (float): Adjustment factor based on market volatility
            
        Returns:
            Tuple[int, float]: (Number of contracts, Percentage of account)
        """
        if self.account_balance <= 0 or price <= 0:
            return 0, 0.0
        
        # Adjust position size based on risk level and volatility
        adjusted_max_size_pct = self.max_position_size_pct * volatility_factor
        
        # Calculate maximum dollar amount for the position
        max_position_value = self.account_balance * adjusted_max_size_pct
        
        # Calculate number of contracts (options contracts are for 100 shares)
        contract_value = price * 100  # 1 option contract = 100 shares
        num_contracts = int(max_position_value / contract_value)
        
        # Calculate actual percentage of account
        actual_pct = (num_contracts * contract_value) / self.account_balance
        
        logger.info(
            f"Position size calculation: {num_contracts} contracts "
            f"({actual_pct:.2%} of account)"
        )
        
        return num_contracts, actual_pct
    
    def update_daily_pnl(self, trade_pnl: float) -> bool:
        """Update and check daily P&L against max loss limit.
        
        Args:
            trade_pnl (float): P&L from the most recent trade
            
        Returns:
            bool: True if trading can continue, False if max loss exceeded
        """
        self.daily_pnl += trade_pnl
        daily_loss_pct = abs(self.daily_pnl) / self.account_balance if self.daily_pnl < 0 else 0
        
        if daily_loss_pct > self.max_daily_loss_pct:
            logger.warning(
                f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.max_daily_loss_pct:.2%}"
            )
            return False
        
        logger.info(f"Current daily P&L: {self.daily_pnl}, {self.daily_pnl / self.account_balance:.2%}")
        return True
    
    def should_close_position(self, entry_price: float, current_price: float, is_long: bool) -> bool:
        """Determine if a position should be closed based on stop loss.
        
        Args:
            entry_price (float): Price at which the position was entered
            current_price (float): Current market price
            is_long (bool): Whether the position is long (True) or short (False)
            
        Returns:
            bool: True if position should be closed, False otherwise
        """
        if entry_price <= 0:
            return False
        
        price_change_pct = (current_price - entry_price) / entry_price
        
        # For short positions, the loss is when price increases
        if not is_long:
            price_change_pct = -price_change_pct
        
        # Close position if loss exceeds stop loss percentage
        if price_change_pct < -self.stop_loss_pct:
            logger.warning(
                f"Stop loss triggered: {price_change_pct:.2%} < -{self.stop_loss_pct:.2%}"
            )
            return True
        
        return False
    
    def adjust_for_volatility(self, is_high_volatility: bool) -> Dict[str, Any]:
        """Adjust risk parameters based on volatility.
        
        Args:
            is_high_volatility (bool): Whether market volatility is high
            
        Returns:
            Dict[str, Any]: Adjusted risk parameters
        """
        # In high volatility, reduce position size but allow more aggressive trades
        if is_high_volatility:
            position_size_factor = 0.7  # Reduce position size
            delta_adjustment = 0.1  # Increase delta (more aggressive options)
        else:
            position_size_factor = 0.5  # Further reduce in low volatility
            delta_adjustment = -0.1  # Decrease delta (more conservative options)
            
        adjusted_params = {
            "position_size_factor": position_size_factor,
            "preferred_delta": self.preferred_delta + delta_adjustment,
        }
        
        logger.info(
            f"Risk parameters adjusted for {'high' if is_high_volatility else 'low'} volatility: "
            f"{adjusted_params}"
        )
        
        return adjusted_params
    
    def get_risk_profile(self) -> Dict[str, Any]:
        """Get the current risk profile settings.
        
        Returns:
            Dict[str, Any]: Current risk profile parameters
        """
        return {
            "risk_level": self.risk_level.name,
            "max_position_size_pct": self.max_position_size_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "preferred_delta": self.preferred_delta,
            "max_options_per_trade": self.max_options_per_trade,
            "max_dte": self.max_dte,
        }