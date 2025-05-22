"""
Interactive Brokers connection and management module.
This module provides the core functionality to connect to the Interactive Brokers API
and handle order submissions, account management, and market data requests.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Import IB-insync for Interactive Brokers API
try:
    from ib_insync import IB, Contract, Option, Stock, Order, MarketOrder, LimitOrder
    from ib_insync import util as ib_util
except ImportError:
    logging.error("ib_insync package not found. Please install with: pip install ib_insync")
    raise

from .config import Config, IBKRConfig


class IBConnector:
    """Interactive Brokers API connector for algorithmic trading."""
    
    def __init__(self, config: Optional[Config] = None, ibkr_config: Optional[IBKRConfig] = None):
        """
        Initialize the Interactive Brokers connector.
        
        Args:
            config: Overall configuration object
            ibkr_config: IBKR-specific configuration (used if config is None)
        """
        self.logger = logging.getLogger(__name__)
        
        if config is not None:
            self.config = config
            self.ibkr_config = config.ibkr
        elif ibkr_config is not None:
            self.ibkr_config = ibkr_config
            self.config = None
        else:
            self.config = Config()
            self.ibkr_config = self.config.ibkr
            
        self.ib = IB()
        self.connected = False
        self._last_error = None
        self._account = None
        
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS or Gateway.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to IB on {self.ibkr_config.host}:{self.ibkr_config.port}")
            self.ib.connect(
                self.ibkr_config.host,
                self.ibkr_config.port,
                clientId=self.ibkr_config.client_id,
                timeout=self.ibkr_config.timeout,
                readonly=self.ibkr_config.read_only
            )
            self.connected = self.ib.isConnected()
            
            if self.connected:
                self.logger.info("Successfully connected to Interactive Brokers")
                self._account = self.ib.managedAccounts()[0] if self.ib.managedAccounts() else None
                self.logger.info(f"Using account: {self._account}")
                # Subscribe to account updates
                if self._account:
                    self.ib.reqAccountUpdates(True, self._account)
            else:
                self.logger.error("Failed to connect to Interactive Brokers")
                
            return self.connected
        
        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error connecting to Interactive Brokers: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from Interactive Brokers")
    
    def check_connection(self) -> bool:
        """
        Check if connection is active, reconnect if necessary.
        
        Returns:
            bool: True if connected, False if connection failed
        """
        if not self.connected or not self.ib.isConnected():
            self.logger.warning("Not connected to IB, attempting to reconnect")
            return self.connect()
        return True
    
    def get_account_summary(self) -> Dict[str, float]:
        """
        Get account summary information.
        
        Returns:
            Dict with account values including:
            - NetLiquidation
            - TotalCashValue
            - AvailableFunds
            - BuyingPower
            - MaintMarginReq
        """
        if not self.check_connection() or not self._account:
            self.logger.error("Cannot get account summary - not connected or no account")
            return {}
        
        try:
            account_values = self.ib.accountSummary(self._account)
            summary = {}
            
            for av in account_values:
                if av.currency == 'USD' and av.tag in [
                    'NetLiquidation', 'TotalCashValue', 'AvailableFunds',
                    'BuyingPower', 'MaintMarginReq'
                ]:
                    summary[av.tag] = float(av.value)
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error getting account summary: {e}")
            return {}
    
    def get_cash_balance(self) -> float:
        """
        Get current cash balance in the account.
        
        Returns:
            float: Cash balance in USD
        """
        summary = self.get_account_summary()
        return summary.get('TotalCashValue', 0.0)
    
    def get_account_value(self) -> float:
        """
        Get current account net liquidation value.
        
        Returns:
            float: Account value in USD
        """
        summary = self.get_account_summary()
        return summary.get('NetLiquidation', 0.0)
    
    def get_market_data(self, symbol: str, sec_type: str = 'STK',
                       exchange: str = 'SMART', currency: str = 'USD',
                       duration: str = '1 D', bar_size: str = '1 min') -> List[Dict]:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Ticker symbol
            sec_type: Security type ('STK', 'OPT', 'FUT', etc.)
            exchange: Exchange name
            currency: Currency code
            duration: Time duration (e.g., '1 D', '1 W', '1 M')
            bar_size: Bar size (e.g., '1 min', '5 mins', '1 hour')
            
        Returns:
            List of bars with OHLCV data
        """
        if not self.check_connection():
            return []
        
        try:
            contract = Contract(symbol=symbol, secType=sec_type, 
                              exchange=exchange, currency=currency)
                              
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            # Convert to dictionaries
            return [
                {
                    'date': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                for bar in bars
            ]
        
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return []
    
    def get_vix_data(self, duration: str = '1 W') -> List[Dict]:
        """
        Get VIX data as a volatility indicator.
        
        Args:
            duration: Time duration (e.g., '1 D', '1 W', '1 M')
            
        Returns:
            List of VIX data points
        """
        return self.get_market_data('VIX', 'IND', 'CBOE', 'USD', duration)
    
    def create_option_contract(self, 
                             symbol: str, 
                             expiry: str,  # Format: YYYYMMDD
                             strike: float, 
                             option_type: str,  # 'C' or 'P'
                             exchange: str = 'SMART',
                             currency: str = 'USD') -> Option:
        """
        Create an option contract.
        
        Args:
            symbol: Underlying symbol
            expiry: Option expiry date in YYYYMMDD format
            strike: Option strike price
            option_type: Option type ('C' for call, 'P' for put)
            exchange: Exchange name
            currency: Currency code
            
        Returns:
            Option contract
        """
        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=option_type,
            exchange=exchange,
            currency=currency
        )
        
        return contract
    
    def get_option_chain(self, symbol: str, exchange: str = 'SMART') -> Dict:
        """
        Get the full option chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            exchange: Exchange name
            
        Returns:
            Dictionary with option chain data
        """
        if not self.check_connection():
            return {}
        
        try:
            underlying = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(underlying)
            
            chains = self.ib.reqSecDefOptParams(
                underlying.symbol, 
                '', 
                underlying.secType, 
                underlying.conId
            )
            
            if not chains:
                self.logger.warning(f"No option chains found for {symbol}")
                return {}
            
            # Process the option chain data
            chain_data = {}
            
            for chain in chains:
                expirations = chain.expirations
                strikes = chain.strikes
                
                # Get current date for filtering
                today = datetime.now().date()
                
                # Filter and organize by expiration
                for expiry in expirations:
                    # Convert to date object
                    try:
                        exp_date = datetime.strptime(expiry, '%Y%m%d').date()
                        # Skip if already expired
                        if exp_date < today:
                            continue
                            
                        days_to_expiry = (exp_date - today).days
                        
                        if expiry not in chain_data:
                            chain_data[expiry] = {
                                'days_to_expiry': days_to_expiry,
                                'strikes': [],
                                'calls': {},
                                'puts': {}
                            }
                            
                        chain_data[expiry]['strikes'] = sorted(strikes)
                    except ValueError:
                        self.logger.error(f"Invalid expiry format: {expiry}")
            
            # Sort expirations by date
            sorted_expirations = sorted(chain_data.keys())
            
            # Get details for specific strikes/expirations
            for expiry in sorted_expirations[:5]:  # Limit to first 5 expirations
                for right in ['C', 'P']:
                    for strike in chain_data[expiry]['strikes'][::5]:  # Sample every 5th strike
                        contract = Option(
                            symbol, 
                            expiry, 
                            strike, 
                            right, 
                            exchange
                        )
                        
                        try:
                            self.ib.qualifyContracts(contract)
                            ticker = self.ib.reqMktData(contract)
                            time.sleep(0.1)  # Avoid rate limiting
                            
                            # Allow some time for data to arrive
                            timeout = time.time() + 2
                            while time.time() < timeout and not ticker.marketPrice():
                                self.ib.sleep(0.1)
                                
                            if right == 'C':
                                chain_data[expiry]['calls'][strike] = {
                                    'bid': ticker.bid,
                                    'ask': ticker.ask,
                                    'last': ticker.last,
                                    'volume': ticker.volume,
                                    'iv': ticker.impliedVolatility
                                }
                            else:
                                chain_data[expiry]['puts'][strike] = {
                                    'bid': ticker.bid,
                                    'ask': ticker.ask,
                                    'last': ticker.last,
                                    'volume': ticker.volume,
                                    'iv': ticker.impliedVolatility
                                }
                                
                            self.ib.cancelMktData(contract)
                            
                        except Exception as e:
                            self.logger.error(f"Error fetching option data for {symbol} {expiry} {strike} {right}: {e}")
            
            return chain_data
            
        except Exception as e:
            self.logger.error(f"Error getting option chain for {symbol}: {e}")
            return {}
    
    def submit_order(self, contract: Contract, order: Order) -> Any:
        """
        Submit an order to Interactive Brokers.
        
        Args:
            contract: Contract to trade
            order: Order specifications
            
        Returns:
            Trade object if successful, None otherwise
        """
        if not self.check_connection():
            return None
            
        try:
            trade = self.ib.placeOrder(contract, order)
            self.logger.info(f"Order submitted: {order.action} {order.totalQuantity} {contract.symbol}")
            
            # Wait for order to be acknowledged
            timeout = time.time() + 5
            while time.time() < timeout and not trade.isDone():
                self.ib.waitOnUpdate(0.1)
                
            return trade
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return None
    
    def market_order(self, 
                    contract: Contract, 
                    quantity: int, 
                    action: str = 'BUY') -> Any:
        """
        Submit a market order.
        
        Args:
            contract: Contract to trade
            quantity: Number of contracts/shares
            action: 'BUY' or 'SELL'
            
        Returns:
            Trade object if successful, None otherwise
        """
        order = MarketOrder(action=action, totalQuantity=quantity)
        return self.submit_order(contract, order)
    
    def limit_order(self, 
                   contract: Contract, 
                   quantity: int, 
                   limit_price: float,
                   action: str = 'BUY') -> Any:
        """
        Submit a limit order.
        
        Args:
            contract: Contract to trade
            quantity: Number of contracts/shares
            limit_price: Limit price
            action: 'BUY' or 'SELL'
            
        Returns:
            Trade object if successful, None otherwise
        """
        order = LimitOrder(action=action, totalQuantity=quantity, lmtPrice=limit_price)
        return self.submit_order(contract, order)
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns:
            bool: True if all positions closed successfully
        """
        if not self.check_connection():
            return False
            
        try:
            positions = self.ib.positions()
            
            if not positions:
                self.logger.info("No positions to close")
                return True
                
            all_closed = True
            
            for position in positions:
                contract = position.contract
                pos_size = position.position
                
                if pos_size > 0:
                    action = 'SELL'
                    quantity = abs(pos_size)
                elif pos_size < 0:
                    action = 'BUY'
                    quantity = abs(pos_size)
                else:
                    continue  # Skip zero positions
                
                trade = self.market_order(contract, quantity, action)
                
                if trade is None:
                    all_closed = False
                    self.logger.error(f"Failed to close position: {contract.symbol}")
                else:
                    self.logger.info(f"Closed position: {action} {quantity} {contract.symbol}")
            
            return all_closed
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open
        """
        if not self.check_connection():
            return False
            
        try:
            contract = Stock('SPY', 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Check market hours
            details = self.ib.reqContractDetails(contract)[0]
            trading_hours = details.tradingHours
            
            # Parse trading hours (format: 20220520:0930-1600;...)
            if trading_hours:
                # Get current date and time in US Eastern time
                now = datetime.now(ib_util.datetime_to_timezone(datetime.now(), 'US/Eastern'))
                today_str = now.strftime('%Y%m%d')
                
                # Look for today's hours
                for schedule in trading_hours.split(';'):
                    if schedule.startswith(today_str):
                        times = schedule.split(':')[1]
                        for time_range in times.split(','):
                            if '-' in time_range:
                                start_str, end_str = time_range.split('-')
                                
                                # Convert to datetime objects
                                start_time = datetime.strptime(f"{today_str} {start_str}", '%Y%m%d %H%M')
                                end_time = datetime.strptime(f"{today_str} {end_str}", '%Y%m%d %H%M')
                                
                                # Check if current time is within range
                                current_time = datetime.strptime(
                                    f"{today_str} {now.strftime('%H%M')}", 
                                    '%Y%m%d %H%M'
                                )
                                
                                if start_time <= current_time <= end_time:
                                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error


# Simple test function
def test_connection(host: str = "127.0.0.1", port: int = 7497) -> None:
    """Test connection to Interactive Brokers."""
    logging.basicConfig(level=logging.INFO)
    
    ibkr_config = IBKRConfig(host=host, port=port)
    connector = IBConnector(ibkr_config=ibkr_config)
    
    if connector.connect():
        print("Connected successfully to IB")
        
        # Test account info
        account_value = connector.get_account_value()
        cash = connector.get_cash_balance()
        print(f"Account value: ${account_value:,.2f}")
        print(f"Cash balance: ${cash:,.2f}")
        
        # Test market data
        spy_data = connector.get_market_data('SPY', duration='1 D')
        print(f"SPY data: {len(spy_data)} bars")
        if spy_data:
            print(f"Latest SPY price: ${spy_data[-1]['close']:.2f}")
        
        # Check market status
        is_open = connector.is_market_open()
        print(f"Market is {'open' if is_open else 'closed'}")
        
        connector.disconnect()
        print("Disconnected from IB")
    else:
        print(f"Failed to connect to IB: {connector.get_last_error()}")


if __name__ == "__main__":
    # Run connection test when module is executed directly
    test_connection()