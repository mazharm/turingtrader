"""Main trading agent for TuringTrader."""

from typing import Dict, Any, Optional, List
import logging
import time
import datetime
import os
import json
from pathlib import Path

from .ib_client import IBClient
from .volatility import VolatilityAnalyzer
from .risk_manager import RiskManager, RiskLevel
from .options_trader import OptionsTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("turingtrader.log"),
    ],
)

logger = logging.getLogger(__name__)


class TuringTrader:
    """Main trading agent for TuringTrader.
    
    This is the main class that integrates all components and implements
    the trading strategy.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 for TWS paper trading, 7496 for live
        client_id: int = 1,
        risk_level: str = "MEDIUM",
        data_dir: Optional[str] = None,
    ):
        """Initialize the TuringTrader.
        
        Args:
            host (str): IB Gateway/TWS hostname
            port (int): IB Gateway/TWS port
            client_id (int): Client ID for IB connection
            risk_level (str): Risk level ("LOW", "MEDIUM", "HIGH", "AGGRESSIVE")
            data_dir (str, optional): Directory for saving trading data
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Initialize components
        self.ib_client = IBClient(host, port, client_id)
        
        # Parse risk level
        try:
            risk_enum = RiskLevel[risk_level.upper()]
        except (KeyError, AttributeError):
            logger.warning(f"Invalid risk level '{risk_level}', defaulting to MEDIUM")
            risk_enum = RiskLevel.MEDIUM
            
        self.risk_manager = RiskManager(risk_level=risk_enum)
        self.volatility_analyzer = VolatilityAnalyzer()
        self.options_trader = OptionsTrader(self.ib_client, self.risk_manager)
        
        # Set up data directory for storing trading data
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / ".turingtrader" / "data"
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track state
        self.is_connected = False
        self.daily_trades = []
        self.daily_pnl = 0.0
        
    def connect(self) -> bool:
        """Connect to Interactive Brokers.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        result = self.ib_client.connect()
        self.is_connected = result
        
        if result:
            # Update account balance
            account_summary = self.ib_client.get_account_summary()
            if account_summary:
                balance = account_summary.get("TotalCashValue", 0.0)
                self.risk_manager.update_account_balance(balance)
                logger.info(f"Connected to IB, account balance: ${balance:.2f}")
            else:
                logger.warning("Connected to IB but failed to get account balance")
        
        return result
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        self.ib_client.disconnect()
        self.is_connected = False
        logger.info("Disconnected from IB")
    
    def set_risk_level(self, risk_level: str) -> bool:
        """Set the risk level for trading.
        
        Args:
            risk_level (str): Risk level ("LOW", "MEDIUM", "HIGH", "AGGRESSIVE")
            
        Returns:
            bool: True if risk level was set successfully, False otherwise
        """
        try:
            risk_enum = RiskLevel[risk_level.upper()]
            self.risk_manager.set_risk_level(risk_enum)
            logger.info(f"Risk level set to {risk_level}")
            return True
        except (KeyError, AttributeError):
            logger.error(f"Invalid risk level: {risk_level}")
            return False
    
    def start_day(self) -> bool:
        """Initialize for the trading day.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.is_connected:
            if not self.connect():
                logger.error("Failed to connect to IB")
                return False
        
        # Check if the market is open
        if not self.ib_client.is_market_open():
            logger.info("Market is not open, waiting...")
            return False
        
        # Reset daily tracking
        self.daily_trades = []
        self.daily_pnl = 0.0
        
        # Update account balance
        account_summary = self.ib_client.get_account_summary()
        if account_summary:
            balance = account_summary.get("TotalCashValue", 0.0)
            self.risk_manager.update_account_balance(balance)
            logger.info(f"Starting day with account balance: ${balance:.2f}")
        else:
            logger.warning("Failed to get account balance")
        
        # Ensure no positions are carried over
        positions = self.ib_client.get_positions()
        if positions:
            logger.warning(f"Found {len(positions)} open positions at start of day")
            self.options_trader.close_all_positions()
            
        return True
    
    def end_day(self) -> Dict[str, Any]:
        """End the trading day, closing all positions.
        
        Returns:
            Dict[str, Any]: Trading day summary
        """
        # Close all positions
        if self.is_connected:
            self.options_trader.close_all_positions()
        
        # Get final account balance
        final_balance = 0.0
        if self.is_connected:
            account_summary = self.ib_client.get_account_summary()
            if account_summary:
                final_balance = account_summary.get("TotalCashValue", 0.0)
        
        # Create summary
        summary = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "trades": len(self.daily_trades),
            "pnl": self.daily_pnl,
            "final_balance": final_balance,
        }
        
        # Save summary to data directory
        summary_file = self.data_dir / f"summary_{summary['date']}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Trading day ended with P&L: ${self.daily_pnl:.2f}")
        return summary
    
    def run_once(self) -> Dict[str, Any]:
        """Execute a single trading cycle.
        
        Returns:
            Dict[str, Any]: Results of this trading cycle
        """
        if not self.is_connected:
            if not self.connect():
                return {"status": "error", "reason": "Failed to connect to IB"}
        
        # Check if market is open
        if not self.ib_client.is_market_open():
            return {"status": "idle", "reason": "Market closed"}
        
        # Analyze market volatility
        is_high_vol, vol_metrics = self.volatility_analyzer.is_high_volatility()
        vol_trend = self.volatility_analyzer.get_volatility_trend()
        
        logger.info(
            f"Volatility analysis: high={is_high_vol}, "
            f"VIX={vol_metrics['vix']:.2f}, trend={vol_trend['trend']}"
        )
        
        # Get options market data
        market_analysis = self.options_trader.analyze_market()
        if not market_analysis.get("tradeable", False):
            reason = market_analysis.get("reason", "Unknown reason")
            logger.info(f"Market not tradeable: {reason}")
            return {"status": "idle", "reason": reason}
        
        # Generate trade plan
        options_data = market_analysis.get("options_data", [])
        trade_plan = self.options_trader.generate_trade_plan(
            is_high_volatility=is_high_vol,
            volatility_metrics={**vol_metrics, **vol_trend},
            options_data=options_data
        )
        
        if not trade_plan.get("tradeable", False):
            reason = trade_plan.get("reason", "No tradeable plan")
            logger.info(f"No trades to execute: {reason}")
            return {"status": "idle", "reason": reason}
        
        # Execute trades if conditions are right
        if is_high_vol:
            logger.info("High volatility detected, executing trades")
            success = self.options_trader.execute_trades(trade_plan)
            
            if success:
                # Track this trade
                trade_info = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "volatility": vol_metrics,
                    "strategy": trade_plan["strategy"],
                    "trade_details": trade_plan["trade_details"],
                }
                self.daily_trades.append(trade_info)
                
                return {
                    "status": "traded",
                    "strategy": trade_plan["strategy"],
                    "details": trade_plan["trade_details"],
                }
            else:
                return {"status": "error", "reason": "Failed to execute trades"}
        else:
            return {
                "status": "idle", 
                "reason": "Low volatility, waiting for better conditions"
            }
    
    def run_trading_day(self) -> Dict[str, Any]:
        """Run trading for a full trading day.
        
        Returns:
            Dict[str, Any]: Summary of the trading day
        """
        # Start the trading day
        if not self.start_day():
            return {"status": "error", "reason": "Failed to start trading day"}
        
        # Trading loop
        trading_cycles = 0
        max_cycles = 10  # Limit number of trading cycles per day
        
        try:
            while trading_cycles < max_cycles:
                # Check if market is still open
                if not self.ib_client.is_market_open():
                    logger.info("Market closed, ending trading day")
                    break
                
                # Run a single trading cycle
                result = self.run_once()
                trading_cycles += 1
                
                if result["status"] == "traded":
                    # Wait longer between trades
                    logger.info(f"Trade executed, waiting 30 minutes ({result['strategy']})")
                    time.sleep(1800)  # 30 minutes
                else:
                    # Wait shorter time between checks
                    logger.info(f"No trade, waiting 5 minutes: {result.get('reason', 'Unknown')}")
                    time.sleep(300)  # 5 minutes
        finally:
            # Always end the trading day properly
            summary = self.end_day()
            summary["cycles"] = trading_cycles
            return summary
    
    def backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest of the strategy on historical data.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Placeholder for backtesting functionality
        # In a real implementation, this would use historical data
        # and simulate the trading strategy
        
        return {
            "status": "not_implemented",
            "message": "Backtesting functionality not implemented yet"
        }


def main() -> None:
    """Main entry point for the TuringTrader agent."""
    # Parse command-line arguments (simplified here)
    import argparse
    parser = argparse.ArgumentParser(description="TuringTrader options trading agent")
    parser.add_argument("--host", default="127.0.0.1", help="IB Gateway/TWS hostname")
    parser.add_argument("--port", type=int, default=7497, help="IB Gateway/TWS port")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID for IB connection")
    parser.add_argument(
        "--risk-level", 
        default="MEDIUM", 
        choices=["LOW", "MEDIUM", "HIGH", "AGGRESSIVE"], 
        help="Risk level"
    )
    parser.add_argument("--data-dir", help="Directory for saving trading data")
    args = parser.parse_args()
    
    # Create and run the trading agent
    trader = TuringTrader(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        risk_level=args.risk_level,
        data_dir=args.data_dir,
    )
    
    try:
        # Connect to IB
        if not trader.connect():
            logger.error("Failed to connect to IB, exiting")
            return
        
        # Run trading day
        summary = trader.run_trading_day()
        logger.info(f"Trading day complete: {summary}")
        
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    except Exception as e:
        logger.exception(f"Error during trading: {e}")
    finally:
        # Ensure we disconnect properly
        trader.disconnect()


if __name__ == "__main__":
    main()