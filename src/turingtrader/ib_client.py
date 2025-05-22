"""Interactive Brokers API client for TuringTrader."""

from typing import Dict, Any, Optional, List
from ib_insync import IB, Contract, Option, Order, Trade
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IBClient:
    """Interactive Brokers API client for the TuringTrader.
    
    This class handles all communication with Interactive Brokers,
    including connection, order submission, and position management.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 is for TWS paper trading, 7496 for live
        client_id: int = 1,
    ):
        """Initialize the IB client.
        
        Args:
            host (str): IB Gateway/TWS hostname
            port (int): IB Gateway/TWS port
            client_id (int): Client ID for this connection
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(f"Connected to IB on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")
    
    def is_connected(self) -> bool:
        """Check if the client is connected to IB.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self._connected and self.ib.isConnected()
    
    def get_account_summary(self) -> Dict[str, float]:
        """Get account summary.
        
        Returns:
            Dict[str, float]: Account summary data
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        summary = {}
        try:
            account_summary = self.ib.accountSummary()
            for item in account_summary:
                if item.tag in ["TotalCashValue", "AvailableFunds", "NetLiquidation"]:
                    summary[item.tag] = float(item.value)
            return summary
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}
    
    def is_market_open(self) -> bool:
        """Check if the market is open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        if not self.is_connected():
            return False
        
        # Use S&P 500 ETF (SPY) as reference
        spy = Contract()
        spy.symbol = "SPY"
        spy.secType = "STK"
        spy.exchange = "SMART"
        spy.currency = "USD"
        
        try:
            self.ib.qualifyContracts(spy)
            details = self.ib.reqContractDetails(spy)[0]
            trading_hours = details.tradingHours
            # Parse trading hours string to determine if market is open
            # This is a simplified check - would need enhancement for production
            return "OPEN" in trading_hours
        except Exception as e:
            logger.error(f"Failed to check if market is open: {e}")
            return False
    
    def get_sp500_options(self, days_to_expiry: int = 30) -> List[Contract]:
        """Get a list of available S&P500 options.
        
        Args:
            days_to_expiry (int): Target days to expiry
            
        Returns:
            List[Contract]: List of option contracts
        """
        # Create SPX index contract
        spx = Contract()
        spx.symbol = "SPX"
        spx.secType = "IND"
        spx.exchange = "CBOE"
        spx.currency = "USD"
        
        try:
            # Get contract details and available expirations
            self.ib.qualifyContracts(spx)
            chains = self.ib.reqSecDefOptParams(
                spx.symbol, spx.exchange, spx.secType, spx.conId
            )
            
            if not chains:
                logger.error("No option chains found for SPX")
                return []
            
            chain = next(c for c in chains)
            
            # Get the current price of the underlying
            market_data = self.ib.reqMktData(spx)
            self.ib.sleep(1)  # Wait for data to arrive
            current_price = market_data.last if market_data.last > 0 else market_data.close
            
            # Find appropriate expiration date
            import datetime
            today = datetime.date.today()
            target_date = today + datetime.timedelta(days=days_to_expiry)
            
            # Find closest expiration
            expirations = sorted(chain.expirations)
            closest_expiry = min(expirations, key=lambda x: abs(
                datetime.datetime.strptime(x, "%Y%m%d").date() - target_date
            ))
            
            # Select a range of strikes around the current price
            strikes = sorted(chain.strikes)
            atm_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price))
            selected_strikes = strikes[max(0, atm_index - 5):min(len(strikes), atm_index + 6)]
            
            # Create option contracts
            option_contracts = []
            for strike in selected_strikes:
                for right in ["C", "P"]:
                    option = Option(
                        "SPX",
                        closest_expiry,
                        strike,
                        right,
                        "SMART",
                        tradingClass="SPX",
                    )
                    option_contracts.append(option)
            
            # Qualify the contracts
            qualified_contracts = []
            for batch in [option_contracts[i:i+10] for i in range(0, len(option_contracts), 10)]:
                self.ib.qualifyContracts(*batch)
                qualified_contracts.extend(batch)
                
            return qualified_contracts
        
        except Exception as e:
            logger.error(f"Failed to get S&P500 options: {e}")
            return []
    
    def get_option_price(self, contract: Contract) -> Dict[str, Any]:
        """Get current market data for an option.
        
        Args:
            contract (Contract): Option contract
            
        Returns:
            Dict[str, Any]: Market data for the option
        """
        if not self.is_connected():
            return {}
        
        try:
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data to arrive
            
            return {
                "bid": ticker.bid,
                "ask": ticker.ask,
                "last": ticker.last,
                "volume": ticker.volume,
                "open_interest": ticker.openInterest,
                "implied_volatility": ticker.impliedVolatility
            }
        except Exception as e:
            logger.error(f"Failed to get option price: {e}")
            return {}
    
    def place_order(
        self, 
        contract: Contract, 
        action: str, 
        quantity: int, 
        order_type: str = "MKT",
        limit_price: Optional[float] = None
    ) -> Optional[Trade]:
        """Place an order for a contract.
        
        Args:
            contract (Contract): Contract to trade
            action (str): "BUY" or "SELL"
            quantity (int): Number of contracts
            order_type (str): Order type, e.g., "MKT", "LMT"
            limit_price (float, optional): Limit price for limit orders
            
        Returns:
            Optional[Trade]: Trade object if successful, None otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        
        if order_type == "LMT" and limit_price is not None:
            order.lmtPrice = limit_price
        
        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Placed order: {action} {quantity} {contract.symbol}")
            return trade
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions.
        
        Returns:
            Dict[str, Dict]: Dictionary of positions
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        positions = {}
        try:
            for position in self.ib.positions():
                symbol = position.contract.symbol
                positions[symbol] = {
                    "contract": position.contract,
                    "position": position.position,
                    "avg_cost": position.avgCost,
                }
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    def close_all_positions(self) -> bool:
        """Close all open positions.
        
        Returns:
            bool: True if all positions closed successfully, False otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return False
        
        try:
            positions = self.get_positions()
            for symbol, data in positions.items():
                action = "SELL" if data["position"] > 0 else "BUY"
                quantity = abs(data["position"])
                
                if quantity > 0:
                    self.place_order(data["contract"], action, quantity)
                    logger.info(f"Closed position: {action} {quantity} {symbol}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False