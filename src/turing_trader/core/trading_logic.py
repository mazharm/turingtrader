"""
Trading Logic and Execution Module

This module implements the core trading logic:
- Strategy selection based on volatility conditions
- Signal generation and trade execution
- Position management and adjustment
- Stop loss and take profit management
"""
import datetime
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from ibapi.contract import Contract
from ibapi.order import Order

class RiskLevel(Enum):
    """Risk tolerance levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class StrategyType(Enum):
    """Available strategy types"""
    LONG_CALLS = "long_calls"
    LONG_PUTS = "long_puts"
    CALL_SPREAD = "call_spread"
    PUT_SPREAD = "put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

class TradingStrategy:
    """
    Abstract base class for trading strategies
    """
    
    def __init__(self, ib_client, volatility_analyzer, risk_manager):
        """
        Initialize the trading strategy
        
        Args:
            ib_client: Interactive Brokers client
            volatility_analyzer: Volatility analysis module
            risk_manager: Risk management module
        """
        self.ib_client = ib_client
        self.volatility_analyzer = volatility_analyzer
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
    def generate_signal(self) -> Dict:
        """Generate trading signal - to be implemented by subclasses"""
        raise NotImplementedError
        
    def execute_signal(self, signal: Dict) -> bool:
        """Execute a trading signal - to be implemented by subclasses"""
        raise NotImplementedError

class VolatilityStrategy(TradingStrategy):
    """
    Volatility-based trading strategy implementation
    """
    
    def __init__(self, ib_client, volatility_analyzer, risk_manager,
                risk_level: RiskLevel = RiskLevel.MEDIUM, 
                underlying: str = "SPY"):
        """
        Initialize the volatility strategy
        
        Args:
            ib_client: Interactive Brokers client
            volatility_analyzer: Volatility analysis module
            risk_manager: Risk management module
            risk_level: Risk tolerance level
            underlying: Underlying symbol (SPY or SPX)
        """
        super().__init__(ib_client, volatility_analyzer, risk_manager)
        self.risk_level = risk_level
        self.underlying = underlying
        
        # Strategy parameters
        self.min_volatility_threshold = 15.0  # Minimum VIX for trading
        self.min_volatility_signal = 1  # Minimum signal strength to trade
        
        # Adjust parameters based on risk level
        self._adjust_risk_parameters()
        
    def _adjust_risk_parameters(self):
        """Adjust strategy parameters based on risk level"""
        if self.risk_level == RiskLevel.LOW:
            self.min_volatility_threshold = 20.0
            self.min_volatility_signal = 2
            self.position_size_factor = 0.05  # 5% of available capital
            self.stop_loss_pct = 0.25  # 25% loss triggers stop
            self.take_profit_pct = 0.50  # 50% gain triggers take profit
            
        elif self.risk_level == RiskLevel.MEDIUM:
            self.min_volatility_threshold = 15.0
            self.min_volatility_signal = 1
            self.position_size_factor = 0.10  # 10% of available capital
            self.stop_loss_pct = 0.35  # 35% loss triggers stop
            self.take_profit_pct = 0.70  # 70% gain triggers take profit
            
        elif self.risk_level == RiskLevel.HIGH:
            self.min_volatility_threshold = 10.0
            self.min_volatility_signal = 1
            self.position_size_factor = 0.20  # 20% of available capital
            self.stop_loss_pct = 0.50  # 50% loss triggers stop
            self.take_profit_pct = 1.00  # 100% gain triggers take profit
            
        self.logger.info(f"Risk parameters adjusted for {self.risk_level.value} risk profile")
    
    def generate_signal(self) -> Dict:
        """
        Generate trading signal based on volatility conditions
        
        Returns:
            dict: Trading signal with action and parameters
        """
        # Get current volatility metrics
        vol_metrics = self.volatility_analyzer.get_option_volatility_metrics(self.underlying)
        
        # Default signal
        signal = {
            'action': 'WAIT',
            'timestamp': datetime.datetime.now(),
            'underlying': self.underlying,
            'strategy': None,
            'reason': 'Default wait state'
        }
        
        # Check if volatility is above minimum threshold for trading
        vix = vol_metrics.get('vix')
        if vix is None or vix < self.min_volatility_threshold:
            signal['reason'] = f'Volatility too low for trading, VIX: {vix}'
            return signal
            
        # Check volatility signal strength
        if vol_metrics.get('signal_strength', 0) < self.min_volatility_signal:
            signal['reason'] = f'Volatility signal too weak: {vol_metrics.get("signal_strength", 0)}'
            return signal
            
        # Determine best strategy based on volatility conditions
        if vol_metrics.get('volatility_spike', False):
            # On volatility spike, use directional strategies
            if vol_metrics.get('signal') == 'BUY':
                signal['action'] = 'BUY'
                signal['strategy'] = StrategyType.LONG_CALLS
                signal['reason'] = 'Volatility spike detected with bullish signal'
            else:
                signal['action'] = 'BUY'
                signal['strategy'] = StrategyType.LONG_PUTS
                signal['reason'] = 'Volatility spike detected with bearish/neutral signal'
                
        elif vol_metrics.get('high_vol_environment', False):
            # In high volatility environment, use non-directional strategies
            signal['action'] = 'BUY'
            signal['strategy'] = StrategyType.STRADDLE
            signal['reason'] = 'High volatility environment, non-directional strategy'
            
        self.logger.debug(f"Generated signal: {signal['action']} - {signal['strategy']}")
        return signal
    
    def _select_option_strike(self, current_price: float, strategy_type: StrategyType) -> float:
        """
        Select appropriate option strike based on strategy
        
        Args:
            current_price: Current price of underlying
            strategy_type: Strategy being executed
            
        Returns:
            float: Selected strike price
        """
        # Round price to nearest strike increment
        if self.underlying == "SPX":
            # SPX options have 5-point strike increments
            increment = 5.0
        else:
            # SPY options typically have $1 strike increments
            increment = 1.0
            
        atm_strike = round(current_price / increment) * increment
        
        if strategy_type in [StrategyType.LONG_CALLS, StrategyType.LONG_PUTS, 
                            StrategyType.STRADDLE, StrategyType.STRANGLE]:
            # For these strategies, use at-the-money options
            return atm_strike
            
        elif strategy_type == StrategyType.CALL_SPREAD:
            # For call spreads, use slightly out-of-the-money
            return atm_strike + increment
            
        elif strategy_type == StrategyType.PUT_SPREAD:
            # For put spreads, use slightly out-of-the-money
            return atm_strike - increment
            
        return atm_strike
    
    def _select_option_expiry(self) -> str:
        """
        Select appropriate option expiration date
        
        Returns:
            str: Option expiration date in YYYYMMDD format
        """
        # Choose expiration based on risk level
        today = datetime.date.today()
        
        if self.risk_level == RiskLevel.LOW:
            # For low risk, use shorter-term options (7-14 days)
            target_days = 10
        elif self.risk_level == RiskLevel.MEDIUM:
            # For medium risk, use medium-term options (14-30 days)
            target_days = 21
        else:
            # For high risk, use slightly longer-term options (30-45 days)
            target_days = 35
            
        # Find the next expiration date after our target
        target_date = today + datetime.timedelta(days=target_days)
        
        # Format as YYYYMMDD
        return target_date.strftime("%Y%m%d")
    
    def execute_signal(self, signal: Dict) -> bool:
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal with action and parameters
            
        Returns:
            bool: True if execution successful
        """
        if signal['action'] != 'BUY' or signal['strategy'] is None:
            return False
            
        try:
            # Get current underlying price
            quote = self._get_current_price(self.underlying)
            if quote is None:
                self.logger.error(f"Unable to get price for {self.underlying}")
                return False
                
            # Select option parameters
            strike = self._select_option_strike(quote, signal['strategy'])
            expiry = self._select_option_expiry()
            
            # Execute based on strategy type
            if signal['strategy'] == StrategyType.LONG_CALLS:
                return self._execute_long_call(quote, strike, expiry)
                
            elif signal['strategy'] == StrategyType.LONG_PUTS:
                return self._execute_long_put(quote, strike, expiry)
                
            elif signal['strategy'] == StrategyType.STRADDLE:
                return self._execute_straddle(quote, strike, expiry)
                
            else:
                self.logger.warning(f"Strategy not implemented: {signal['strategy']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing signal: {str(e)}")
            return False
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        # Implementation depends on IB client interface
        # This is a simplified version
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.currency = "USD"
            contract.exchange = "SMART"
            
            # Request market data
            req_id = self.ib_client.request_market_data(contract)
            
            # Wait for data (in a real implementation, would use callbacks)
            time.sleep(1)
            
            # Get latest price
            if req_id in self.ib_client.market_data:
                latest_price = self.ib_client.market_data[req_id].get(4, None)  # 4 is last price
                return latest_price
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def _execute_long_call(self, price: float, strike: float, expiry: str) -> bool:
        """Execute a long call option trade"""
        try:
            # Calculate position size based on risk parameters
            position_size = self.risk_manager.calculate_position_size(
                self.position_size_factor, option_type='call')
                
            # Create option contract
            contract = self.ib_client.create_option_contract(
                symbol=self.underlying,
                expiry=expiry,
                strike=strike,
                option_type='CALL'
            )
            
            # Create order
            order = Order()
            order.action = "BUY"
            order.orderType = "MKT"  # Market order
            order.totalQuantity = position_size
            
            # Submit order
            order_id = self.ib_client.place_order(contract, order)
            
            self.logger.info(f"Executed long call: {self.underlying} {expiry} {strike} call, "
                           f"qty: {position_size}, order_id: {order_id}")
            
            # Set stop loss and take profit
            # In a real implementation, would add bracket orders or trailing stops
            
            return True
        except Exception as e:
            self.logger.error(f"Error executing long call: {str(e)}")
            return False
    
    def _execute_long_put(self, price: float, strike: float, expiry: str) -> bool:
        """Execute a long put option trade"""
        try:
            # Calculate position size based on risk parameters
            position_size = self.risk_manager.calculate_position_size(
                self.position_size_factor, option_type='put')
                
            # Create option contract
            contract = self.ib_client.create_option_contract(
                symbol=self.underlying,
                expiry=expiry,
                strike=strike,
                option_type='PUT'
            )
            
            # Create order
            order = Order()
            order.action = "BUY"
            order.orderType = "MKT"  # Market order
            order.totalQuantity = position_size
            
            # Submit order
            order_id = self.ib_client.place_order(contract, order)
            
            self.logger.info(f"Executed long put: {self.underlying} {expiry} {strike} put, "
                           f"qty: {position_size}, order_id: {order_id}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error executing long put: {str(e)}")
            return False
    
    def _execute_straddle(self, price: float, strike: float, expiry: str) -> bool:
        """Execute a straddle (long call + long put at same strike)"""
        try:
            # Execute both legs
            call_success = self._execute_long_call(price, strike, expiry)
            put_success = self._execute_long_put(price, strike, expiry)
            
            return call_success and put_success
            
        except Exception as e:
            self.logger.error(f"Error executing straddle: {str(e)}")
            return False