"""
Interactive Brokers API Client

This module provides integration with Interactive Brokers API for:
- Authentication and connection management
- Market data subscription for SPX/SPY options
- Order execution and management
- Account and position monitoring
"""
import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable

# Import IB API modules
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, BarData
from ibapi.utils import iswrapper

class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"

class IBClient(EWrapper, EClient):
    """Interactive Brokers API client implementation"""
    
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        """
        Initialize the IB API client
        
        Args:
            host: TWS/IB Gateway host (default: localhost)
            port: TWS/IB Gateway port (default: 7497 for paper trading)
            client_id: Unique client ID for this connection
        """
        EClient.__init__(self, self)
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Callbacks and data storage
        self.next_valid_order_id = None
        self.market_data = {}
        self.positions = {}
        self.account_info = {}
        self.option_chains = {}
        self.volatility_data = {}
        
        # Callback registries
        self.market_data_callbacks = {}
        self.order_callbacks = {}
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to IB API at {self.host}:{self.port}")
            super().connect(self.host, self.port, self.client_id)
            
            # Start processing messages from API
            thread = threading.Thread(target=self.run)
            thread.start()
            
            # Wait for nextValidId callback
            timeout = 10  # seconds
            start_time = time.time()
            while self.next_valid_order_id is None and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.next_valid_order_id is None:
                self.logger.error("Failed to receive valid order ID from IB API")
                return False
                
            self.logger.info(f"Connected to IB API, client ID: {self.client_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IB API: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers API"""
        self.logger.info("Disconnecting from IB API")
        self.done = True
        super().disconnect()
    
    # Contract creation helpers
    def create_option_contract(self, symbol: str, expiry: str, 
                              strike: float, option_type: OptionType) -> Contract:
        """
        Create an option contract for SPX/SPY
        
        Args:
            symbol: Underlying symbol (e.g., 'SPX' or 'SPY')
            expiry: Option expiry date in YYYYMMDD format
            strike: Option strike price
            option_type: OptionType.CALL or OptionType.PUT
            
        Returns:
            Contract: IB API Contract object
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = option_type.value
        contract.multiplier = "100"
        
        # Use index options for SPX
        if symbol == "SPX":
            contract.exchange = "CBOE"
            
        return contract
    
    def get_option_chain(self, symbol: str) -> Dict:
        """
        Request option chain data for a symbol
        
        Args:
            symbol: Underlying symbol (e.g., 'SPX' or 'SPY')
            
        Returns:
            dict: Option chain data if available
        """
        # Implementation depends on specific IB API approach
        # This is a simplified placeholder
        pass
    
    # Order management methods
    def place_order(self, contract: Contract, order: Order, callback: Callable = None) -> int:
        """
        Submit an order to Interactive Brokers
        
        Args:
            contract: Contract object
            order: Order object
            callback: Optional callback function when order status changes
            
        Returns:
            int: Order ID
        """
        order_id = self.next_valid_order_id
        self.next_valid_order_id += 1
        
        if callback:
            self.order_callbacks[order_id] = callback
            
        self.placeOrder(order_id, contract, order)
        self.logger.info(f"Placed order {order_id} for {contract.symbol}")
        return order_id
    
    def cancel_order(self, order_id: int) -> None:
        """
        Cancel an existing order
        
        Args:
            order_id: Order ID to cancel
        """
        self.cancelOrder(order_id)
        self.logger.info(f"Cancelled order {order_id}")
    
    def request_market_data(self, contract: Contract, callback: Callable = None) -> int:
        """
        Request real-time market data for a contract
        
        Args:
            contract: Contract to request data for
            callback: Optional callback for market data updates
            
        Returns:
            int: Request ID
        """
        req_id = self.next_valid_order_id
        self.next_valid_order_id += 1
        
        if callback:
            self.market_data_callbacks[req_id] = callback
            
        # Request market data
        self.reqMktData(req_id, contract, "", False, False, [])
        return req_id
    
    def request_account_updates(self) -> None:
        """Request account and position updates"""
        self.reqAccountUpdates(True, "")
    
    def liquidate_all_positions(self) -> List[int]:
        """
        Liquidate all open positions
        
        Returns:
            list: Order IDs of liquidation orders
        """
        order_ids = []
        
        for position_key, position in self.positions.items():
            symbol, contract_id = position_key
            quantity = position['position']
            
            if quantity == 0:
                continue
                
            # Create an opposite order to close position
            contract = self.create_contract_from_position(position)
            order = Order()
            order.action = "SELL" if quantity > 0 else "BUY"
            order.orderType = "MKT"
            order.totalQuantity = abs(quantity)
            
            order_id = self.place_order(contract, order)
            order_ids.append(order_id)
            
        self.logger.info(f"Liquidation orders placed: {len(order_ids)}")
        return order_ids
    
    # IB API Callback implementations
    @iswrapper
    def nextValidId(self, order_id: int):
        """Called by IB with the next valid order ID"""
        self.next_valid_order_id = order_id
        self.logger.debug(f"Next valid order ID: {order_id}")
    
    @iswrapper
    def error(self, req_id: TickerId, error_code: int, error_string: str):
        """Called when IB API reports an error"""
        self.logger.error(f"IB API Error {error_code}: {error_string} (req_id: {req_id})")
    
    @iswrapper
    def position(self, account: str, contract: Contract, position: float, avg_cost: float):
        """Called when position information is received"""
        key = (contract.symbol, contract.conId)
        self.positions[key] = {
            'account': account,
            'contract': contract,
            'position': position,
            'avg_cost': avg_cost
        }
        self.logger.debug(f"Position update: {contract.symbol} {position} @ {avg_cost}")
    
    @iswrapper
    def tickPrice(self, req_id: TickerId, tick_type: int, price: float, attrib):
        """Called when price data is received"""
        if req_id in self.market_data:
            self.market_data[req_id][tick_type] = price
            
        if req_id in self.market_data_callbacks:
            self.market_data_callbacks[req_id](tick_type, price)
    
    @iswrapper
    def tickOptionComputation(self, req_id: TickerId, tick_type: int, 
                              implied_vol: float, delta: float, opt_price: float, 
                              pv_dividend: float, gamma: float, vega: float, 
                              theta: float, und_price: float):
        """Called when option computation data is received"""
        if req_id not in self.volatility_data:
            self.volatility_data[req_id] = {}
            
        self.volatility_data[req_id] = {
            'implied_vol': implied_vol,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'underlying_price': und_price
        }
        
        # Call registered callbacks
        if req_id in self.market_data_callbacks:
            self.market_data_callbacks[req_id]('volatility', self.volatility_data[req_id])
            
    @iswrapper
    def orderStatus(self, order_id: int, status: str, filled: float, 
                   remaining: float, avg_fill_price: float, 
                   perm_id: int, parent_id: int, 
                   last_fill_price: float, client_id: int, 
                   why_held: str, mkt_cap_price: float):
        """Called when order status changes"""
        self.logger.debug(f"Order {order_id} status: {status}")
        
        # Call registered callbacks
        if order_id in self.order_callbacks:
            self.order_callbacks[order_id](order_id, status, filled, remaining)

# Missing import
import threading