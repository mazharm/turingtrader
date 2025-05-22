"""
TuringTrader Main Application

This is the main entry point for the TuringTrader application:
- Initializes all components
- Sets up the trading environment
- Handles the trading loop
- Manages cleanup and shutdown
"""
import sys
import time
import logging
import argparse
import datetime
import signal
import threading
from typing import Dict, Optional

from turing_trader.api.ib_client import IBClient
from turing_trader.core.volatility_analysis import VolatilityAnalyzer
from turing_trader.core.risk_management import RiskManager, RiskLevel
from turing_trader.core.cash_management import CashManager
from turing_trader.core.trading_logic import VolatilityStrategy
from turing_trader.config.settings import Settings, RiskLevel as ConfigRiskLevel
from turing_trader.utils.logging import setup_logging
from turing_trader.utils.reporting import PerformanceReporter

class TuringTrader:
    """Main Turing Trader application class"""
    
    def __init__(self, settings_file: Optional[str] = None, risk_level: str = "medium"):
        """
        Initialize the TuringTrader
        
        Args:
            settings_file: Optional path to settings file
            risk_level: Risk level (low, medium, high)
        """
        # Setup logging
        self.logger = setup_logging(log_level=logging.INFO, component="main")
        self.logger.info("Initializing TuringTrader")
        
        # Load settings
        self.settings = Settings(settings_file)
        
        # Override risk level if provided
        if risk_level:
            try:
                self.settings.risk_level = ConfigRiskLevel(risk_level.lower())
                self.logger.info(f"Risk level set to: {self.settings.risk_level.value}")
            except ValueError:
                self.logger.warning(f"Invalid risk level: {risk_level}, using default")
        
        # Application state
        self.running = False
        self.trading_active = False
        self.components_initialized = False
        self.shutdown_event = threading.Event()
        
        # Component instances (initialized in setup)
        self.ib_client = None
        self.volatility_analyzer = None
        self.risk_manager = None
        self.cash_manager = None
        self.trading_strategy = None
        self.performance_reporter = None
        
    def setup(self) -> bool:
        """
        Setup all components and establish connections
        
        Returns:
            bool: True if setup successful
        """
        try:
            self.logger.info("Setting up TuringTrader components")
            
            # Initialize IB client
            self.ib_client = IBClient(
                host=self.settings.ib_host,
                port=self.settings.ib_port,
                client_id=self.settings.ib_client_id
            )
            
            # Connect to Interactive Brokers
            if not self.ib_client.connect():
                self.logger.error("Failed to connect to Interactive Brokers")
                return False
                
            self.logger.info("Connected to Interactive Brokers")
                
            # Initialize components
            self.volatility_analyzer = VolatilityAnalyzer(
                lookback_period=self.settings.volatility_lookback_period,
                vix_threshold=self.settings.min_vix_threshold
            )
            
            # Map configuration risk level to component risk level
            risk_map = {
                ConfigRiskLevel.LOW: RiskLevel.LOW,
                ConfigRiskLevel.MEDIUM: RiskLevel.MEDIUM,
                ConfigRiskLevel.HIGH: RiskLevel.HIGH
            }
            component_risk_level = risk_map[self.settings.risk_level]
            
            self.risk_manager = RiskManager(
                ib_client=self.ib_client,
                risk_level=component_risk_level
            )
            
            self.cash_manager = CashManager(
                ib_client=self.ib_client,
                market_open_time=self.settings.market_open_time,
                market_close_time=self.settings.market_close_time,
                liquidation_time=self.settings.liquidation_time
            )
            
            self.trading_strategy = VolatilityStrategy(
                ib_client=self.ib_client,
                volatility_analyzer=self.volatility_analyzer,
                risk_manager=self.risk_manager,
                risk_level=component_risk_level,
                underlying=self.settings.underlying_symbol
            )
            
            self.performance_reporter = PerformanceReporter()
            
            # Register signal handlers for clean shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            self.components_initialized = True
            self.logger.info("TuringTrader components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during setup: {str(e)}")
            self.cleanup()
            return False
    
    def start(self) -> None:
        """Start the trader"""
        if not self.components_initialized and not self.setup():
            self.logger.error("Failed to initialize components")
            return
            
        self.running = True
        self.logger.info("TuringTrader starting")
        
        try:
            # Request initial account data
            self.ib_client.request_account_updates()
            
            # Initial cash position check
            status = self.cash_manager.verify_cash_position()
            self.logger.info(f"Initial cash position: {status['is_cash_only']}")
            
            if not status['is_cash_only']:
                self.logger.warning("Not in cash position at startup, liquidating positions")
                self.cash_manager.liquidate_all_positions()
            
            # Schedule end-of-day liquidation
            self.cash_manager.schedule_liquidation()
            
            # Enter main trading loop
            self.trading_loop()
            
        except Exception as e:
            self.logger.error(f"Error during trader execution: {str(e)}")
        finally:
            self.cleanup()
            
    def trading_loop(self) -> None:
        """Main trading loop"""
        self.logger.info("Entering trading loop")
        self.trading_active = True
        
        # Parameters for trading loop
        check_interval = 60  # seconds between loop iterations
        market_check_counter = 0
        volatility_check_counter = 0
        
        while self.running and not self.shutdown_event.is_set():
            try:
                now = datetime.datetime.now()
                current_time = now.time()
                
                # Market session check (every 5 minutes)
                if market_check_counter == 0:
                    is_market_open = (
                        self.settings.market_open_time <= current_time < 
                        self.settings.market_close_time
                    )
                    
                    # Reset counters for different market sessions
                    if is_market_open:
                        if not self.trading_active:
                            self.logger.info("Market open, activating trading")
                            self.cash_manager.handle_market_open()
                            self.trading_active = True
                    else:
                        if self.trading_active:
                            self.logger.info("Market closed, deactivating trading")
                            self.cash_manager.handle_market_close()
                            self.trading_active = False
                
                # Only perform trading operations during market hours
                if self.trading_active:
                    # Check volatility and generate signals (every 5 minutes)
                    if volatility_check_counter == 0:
                        # Update volatility
                        vix_value = self._get_vix_value()
                        if vix_value is not None:
                            self.volatility_analyzer.update_vix(vix_value)
                            
                            # Generate trading signal
                            signal = self.trading_strategy.generate_signal()
                            
                            self.logger.info(f"Signal: {signal['action']} - {signal['reason']}")
                            
                            # Execute signal if appropriate
                            if signal['action'] == 'BUY' and self.settings.enable_auto_trading:
                                self.trading_strategy.execute_signal(signal)
                                
                # Update account data (every 10 minutes)
                if market_check_counter % 10 == 0:
                    self.ib_client.request_account_updates()
                
                # Check if we've reached daily risk limit
                if not self.risk_manager.check_daily_risk_limit():
                    self.logger.warning("Daily risk limit reached, stopping trading for today")
                    self.trading_active = False
                    
                # Update counters
                market_check_counter = (market_check_counter + 1) % 5
                volatility_check_counter = (volatility_check_counter + 1) % 5
                
                # Sleep until next iteration
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                # Brief pause to avoid rapid error loops
                time.sleep(5)
    
    def _get_vix_value(self) -> Optional[float]:
        """
        Get current VIX value
        
        Returns:
            float: VIX value or None if unavailable
        """
        # In a real implementation, would get this from market data
        # This is a placeholder - would normally use IB client to get actual VIX
        return 20.0  # Placeholder value
    
    def signal_handler(self, sig, frame) -> None:
        """Handle termination signals"""
        self.logger.info(f"Received signal {sig}, initiating shutdown")
        self.shutdown_event.set()
        self.running = False
    
    def cleanup(self) -> None:
        """Clean up resources and connections"""
        self.logger.info("Cleaning up resources")
        
        # Liquidate positions if needed
        if self.components_initialized and self.cash_manager:
            status = self.cash_manager.verify_cash_position()
            if not status['is_cash_only']:
                self.logger.info("Liquidating positions before exit")
                self.cash_manager.liquidate_all_positions()
        
        # Disconnect from IB
        if self.components_initialized and self.ib_client:
            self.logger.info("Disconnecting from Interactive Brokers")
            self.ib_client.disconnect()
            
        self.logger.info("TuringTrader shutdown complete")


def main():
    """Application entry point"""
    parser = argparse.ArgumentParser(description="TuringTrader - AI driven algorithmic trader")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--risk", type=str, default="medium", 
                      choices=["low", "medium", "high"],
                      help="Risk level (low, medium, high)")
    parser.add_argument("--generate-config", type=str, 
                      help="Generate example config file at specified path")
    args = parser.parse_args()
    
    # Generate example config if requested
    if args.generate_config:
        settings = Settings()
        if settings.create_example_config(args.generate_config):
            print(f"Example configuration created at: {args.generate_config}")
            return 0
        else:
            print(f"Failed to create example configuration")
            return 1
    
    # Create and start trader
    trader = TuringTrader(settings_file=args.config, risk_level=args.risk)
    trader.start()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())