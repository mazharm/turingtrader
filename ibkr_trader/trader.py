"""
Main trader module for the TuringTrader algorithm.
This is the core controller that integrates all components and runs the trading strategy.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from ib_insync import IB, Option, MarketOrder

from .config import Config
from .ib_connector import IBConnector
from .volatility_analyzer import VolatilityAnalyzer
from .risk_manager import RiskManager
from .options_strategy import OptionsStrategy


class TuringTrader:
    """Main trader class that orchestrates the entire trading strategy."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the TuringTrader.
        
        Args:
            config_path: Path to configuration file
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = Config(config_path)
        
        # Initialize components
        self.ib_connector = IBConnector(self.config)
        self.volatility_analyzer = VolatilityAnalyzer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.options_strategy = OptionsStrategy(
            self.config, 
            self.volatility_analyzer, 
            self.risk_manager
        )
        
        # Trading state
        self.is_trading_day = False
        self.market_open = False
        self.in_trading_window = False
        self.day_trade_count = 0
        self.daily_pnl = 0.0
        self.account_value = 0.0
        self.cash_balance = 0.0
        self.last_check_time = None
        self.last_trade_time = None
        self.trading_period_minutes = self.config.trading.trading_period_minutes
        self.day_start_offset = self.config.trading.day_start_offset_hours
        self.day_end_offset = self.config.trading.day_end_offset_hours
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers.
        
        Returns:
            bool: True if connection is successful
        """
        return self.ib_connector.connect()
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        self.ib_connector.disconnect()
    
    def update_account_info(self) -> None:
        """Update account information."""
        account_value = self.ib_connector.get_account_value()
        cash_balance = self.ib_connector.get_cash_balance()
        
        if account_value > 0:
            self.account_value = account_value
            self.cash_balance = cash_balance
            
            # Update risk manager
            self.risk_manager.update_account_value(account_value)
            
            self.logger.info(f"Account value: ${account_value:.2f}, Cash: ${cash_balance:.2f}")
    
    def check_market_status(self) -> bool:
        """
        Check if market is open and we're in valid trading hours.
        
        Returns:
            bool: True if we should be trading now
        """
        # First check if market is open
        self.market_open = self.ib_connector.is_market_open()
        
        # Default state
        self.in_trading_window = False
        
        if not self.market_open:
            self.logger.info("Market is currently closed")
            return False
        
        # Check if we're within our trading window
        now = datetime.now()
        
        # Get market hours (assuming 9:30 AM to 4:00 PM Eastern Time)
        # In a real implementation, fetch these from the IB API
        market_open_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Calculate our trading window
        start_offset_seconds = self.day_start_offset * 3600
        end_offset_seconds = self.day_end_offset * 3600
        
        trading_start = market_open_time + timedelta(seconds=start_offset_seconds)
        trading_end = market_close_time - timedelta(seconds=end_offset_seconds)
        
        # Check if we're in the trading window
        self.in_trading_window = trading_start <= now <= trading_end
        
        if not self.in_trading_window:
            if now < trading_start:
                self.logger.info(f"Before trading window, will start at {trading_start.strftime('%H:%M:%S')}")
            else:
                self.logger.info(f"After trading window, ended at {trading_end.strftime('%H:%M:%S')}")
        
        # Determine if this is a new trading day
        if self.last_check_time is None or self.last_check_time.date() != now.date():
            self.is_trading_day = True
            self.day_trade_count = 0
            self.daily_pnl = 0.0
            # Reset daily state
            self.options_strategy.reset_daily_state()
            self.logger.info(f"New trading day: {now.date()}")
        
        self.last_check_time = now
        
        return self.market_open and self.in_trading_window
    
    def fetch_market_data(self) -> Dict:
        """
        Fetch current market data including VIX and underlying price.
        
        Returns:
            Dict with market data
        """
        self.logger.info("Fetching market data")
        
        # Get VIX data
        vix_data = self.ib_connector.get_vix_data()
        
        # Get underlying data
        index_symbol = self.config.trading.index_symbol
        underlying_data = self.ib_connector.get_market_data(index_symbol)
        
        # Get current price
        current_price = 0.0
        if underlying_data:
            current_price = underlying_data[-1]['close']
        
        return {
            'vix_data': vix_data,
            'underlying_data': underlying_data,
            'current_price': current_price
        }
    
    def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """
        Analyze current market conditions.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dict with analysis results
        """
        self.logger.info("Analyzing market conditions")
        
        vix_data = market_data.get('vix_data', [])
        underlying_data = market_data.get('underlying_data', [])
        
        # Analyze VIX
        vix_analysis = self.volatility_analyzer.analyze_vix(vix_data)
        
        # Calculate historical volatility
        if underlying_data:
            prices = [bar['close'] for bar in underlying_data]
            hist_vol = self.volatility_analyzer.calculate_historical_volatility(prices)
            vix_analysis['historical_volatility'] = hist_vol
        
        return {'vix_analysis': vix_analysis}
    
    def fetch_option_chain(self) -> Dict:
        """
        Fetch current option chain for the index.
        
        Returns:
            Dict with option chain data
        """
        self.logger.info(f"Fetching option chain for {self.config.trading.index_symbol}")
        
        option_chain = self.ib_connector.get_option_chain(self.config.trading.index_symbol)
        
        return option_chain
    
    def execute_trade(self, trade_decision: Dict) -> Dict:
        """
        Execute a trade based on the trade decision.
        
        Args:
            trade_decision: Trade decision dictionary
            
        Returns:
            Dict with trade execution results
        """
        action = trade_decision.get('action')

        if action == 'buy': # Handle single leg option buy
            try:
                # Extract trade details
                symbol = trade_decision.get('symbol')
                expiry = trade_decision.get('expiry')
                strike = trade_decision.get('strike')
                option_type = trade_decision.get('option_type', 'call')
                quantity = trade_decision.get('quantity', 1)
                
                # Convert option_type to IB format ('C' or 'P')
                right = 'C' if option_type.lower() == 'call' else 'P'
                
                # Create option contract
                contract = self.ib_connector.create_option_contract(
                    symbol=symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=right
                )
                
                # Create market order
                trade = self.ib_connector.market_order(
                    contract=contract,
                    quantity=quantity,
                    action='BUY'
                )
                
                if trade is None or not trade.orderStatus:
                    self.logger.error(f"Failed to execute single option trade for {symbol}")
                    return {'executed': False, 'reason': 'order_failed_or_no_status'}
                    
                # Update state
                self.day_trade_count += 1
                self.last_trade_time = datetime.now()
                
                avg_price = trade.orderStatus.avgFillPrice if trade.orderStatus else 0.0
                if not avg_price: # Fallback if avgFillPrice is not available
                    avg_price = trade_decision.get('price', 0.0) 
                
                self.risk_manager.add_position(
                    symbol=f"{symbol}_{right}_{expiry}_{strike}",
                    quantity=quantity,
                    entry_price=avg_price,
                    position_type='option',
                    option_data={
                        'underlying': symbol,
                        'expiry': expiry,
                        'strike': strike,
                        'option_type': option_type,
                        'contract': contract # Storing the qualified contract might be useful
                    }
                )
                
                self.logger.info(f"Executed trade: BUY {quantity} {symbol} {option_type} "
                               f"@ {strike} exp:{expiry} AvgPrice: {avg_price}")
                               
                return {
                    'executed': True,
                    'trade_id': trade.order.orderId if trade.order else None,
                    'avg_price': avg_price,
                    'quantity': quantity,
                    'symbol': symbol,
                    'option_type': option_type,
                    'strike': strike,
                    'expiry': expiry
                }
            
            except Exception as e:
                self.logger.error(f"Error executing single option trade: {e}", exc_info=True)
                return {'executed': False, 'reason': str(e)}

        elif action == 'iron_condor':
            try:
                symbol = trade_decision.get('symbol')
                expiry = trade_decision.get('expiry')
                short_call_strike = trade_decision.get('short_call_strike')
                short_put_strike = trade_decision.get('short_put_strike')
                long_call_strike = trade_decision.get('long_call_strike')
                long_put_strike = trade_decision.get('long_put_strike')
                quantity = trade_decision.get('quantity')
                net_credit_estimate = trade_decision.get('net_credit') # Estimated credit

                # For Iron Condors, we are selling the spread, so we expect a credit.
                # The IB API for BAG orders might require a positive limit price for a credit.
                # We place a limit order at the estimated net credit.
                # A market order for complex spreads is often not advisable due to slippage.
                # If net_credit_estimate is positive, it's a credit.
                # IB's limit orders for combos: BUY for a debit, SELL for a credit.
                # Since we are *receiving* a credit, we are *selling* the condor structure.
                
                trade = self.ib_connector.submit_iron_condor_order(
                    symbol=symbol,
                    expiry=expiry,
                    short_put_strike=short_put_strike,
                    short_call_strike=short_call_strike,
                    long_put_strike=long_put_strike,
                    long_call_strike=long_call_strike,
                    quantity=quantity,
                    # For a credit, the price is positive. We are SELLING the condor.
                    # The submit_iron_condor_order in ib_connector uses 'BUY' for LimitOrder by default
                    # This needs to be 'SELL' if we expect a credit.
                    # Let's assume submit_iron_condor_order handles the action based on price (e.g. positive for credit = SELL)
                    # Or, more explicitly, we might need to adjust ib_connector or pass 'SELL' action here.
                    # For now, assuming ib_connector's submit_iron_condor_order is set up to sell for a credit.
                    # If it uses a BUY order with a positive price, that would be a DEBIT.
                    # Let's assume the `submit_iron_condor_order` is designed to take the net_credit as the limit price
                    # and handles the BUY/SELL action appropriately (e.g. if price > 0, it's a credit, so SELL action).
                    # Re-checking ib_connector.py: submit_iron_condor_order uses LimitOrder('BUY', ...)
                    # This is problematic if we want a credit. A BUY limit order for a spread means we are willing to PAY up to limit_price.
                    # To receive a credit, we need a SELL limit order.
                    # Fix: Change the approach to explicitly use 'SELL' action for credit trades in the BAG order
                    # The limit price should be positive for a credit when using SELL
                    action="SELL",  # SELL to receive credit for the iron condor
                    limit_price=abs(net_credit_estimate)  # Price must be positive for IB limit orders
                )

                if trade is None or not trade.orderStatus:
                    self.logger.error(f"Failed to execute iron condor trade for {symbol}")
                    return {'executed': False, 'reason': 'iron_condor_order_failed_or_no_status'}

                self.day_trade_count += 1 # Counts as one complex trade
                self.last_trade_time = datetime.now()

                avg_fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else 0.0
                if not avg_fill_price: # Fallback
                    avg_fill_price = net_credit_estimate # Use estimate if fill not available

                # Construct a symbol for the condor
                condor_symbol = f"{symbol}_IC_{expiry}_{long_put_strike:.0f}P_{short_put_strike:.0f}P_{short_call_strike:.0f}C_{long_call_strike:.0f}C"
                
                self.risk_manager.add_position(
                    symbol=condor_symbol,
                    quantity=quantity,
                    entry_price=avg_fill_price, # This is the net credit received per spread
                    position_type='iron_condor',
                    option_data={ # Store all relevant details for the condor
                        'underlying': symbol,
                        'expiry': expiry,
                        'short_put_strike': short_put_strike,
                        'short_call_strike': short_call_strike,
                        'long_put_strike': long_put_strike,
                        'long_call_strike': long_call_strike,
                        'net_credit_received': avg_fill_price,
                        'max_risk_per_spread': trade_decision.get('max_risk'), # From options_strategy
                        'contract': trade.contract # The BAG contract
                    }
                )

                self.logger.info(f"Executed Iron Condor: {quantity} {condor_symbol} for a credit of {avg_fill_price:.2f} per spread")

                return {
                    'executed': True,
                    'trade_id': trade.order.orderId if trade.order else None,
                    'avg_price_credit': avg_fill_price,
                    'quantity': quantity,
                    'details': trade_decision 
                }

            except Exception as e:
                self.logger.error(f"Error executing iron condor trade: {e}", exc_info=True)
                return {'executed': False, 'reason': str(e)}

        else:
            self.logger.info(f"No trade to execute or unknown action: {action}")
            return {'executed': False, 'reason': f'no_trade_action_or_unknown: {action}'}
    
    def close_all_positions(self) -> Dict:
        """
        Close all open positions.
        
        Returns:
            Dict with results
        """
        self.logger.info("Closing all positions")
        
        # Get current positions
        positions = self.risk_manager.get_open_positions()
        
        if not positions:
            self.logger.info("No positions to close")
            return {'closed': True, 'count': 0}
        
        # Create a dictionary of current market prices
        market_prices = {}
        
        # Close all positions through IB
        result = self.ib_connector.close_all_positions()
        
        if result:
            self.logger.info(f"Successfully closed all {len(positions)} positions")
        else:
            self.logger.error("Failed to close some positions")
        
        # Update risk manager with position closures
        closed_positions = self.risk_manager.close_all_positions(market_prices)
        
        return {
            'closed': result,
            'count': len(positions),
            'positions': closed_positions
        }
    
    def run_trading_cycle(self) -> Dict:
        """
        Run a single trading decision cycle.
        
        Returns:
            Dict with cycle results
        """
        self.logger.info("Running trading cycle")
        
        # Check if we should be trading
        if not self.check_market_status():
            return {'action': 'none', 'reason': 'outside_trading_hours'}
        
        # Update account info
        self.update_account_info()
        
        # Fetch market data
        market_data = self.fetch_market_data()
        
        # Check if it's end of day
        should_close = self.options_strategy.should_close_positions()
        
        if should_close:
            self.logger.info("End of trading day, closing all positions")
            close_result = self.close_all_positions()
            return {'action': 'close_all', 'result': close_result}
        
        # Analyze market conditions
        analysis = self.analyze_market_conditions(market_data)
        
        # Get VIX analysis
        vix_analysis = analysis.get('vix_analysis', {})
        
        # Check if we should trade based on volatility
        should_trade = self.volatility_analyzer.should_trade_today(vix_analysis)
        
        if not should_trade:
            self.logger.info("Volatility conditions not favorable for trading")
            return {'action': 'none', 'reason': 'unfavorable_volatility'}
        
        # Check if we already have open positions
        open_positions = self.risk_manager.get_open_positions()
        
        if open_positions:
            # We already have positions, just hold them
            self.logger.info(f"Already have {len(open_positions)} open positions, holding")
            return {'action': 'hold', 'positions': len(open_positions)}
        
        # Fetch option chain
        option_chain = self.fetch_option_chain()
        
        # Generate trade decision
        current_price = market_data.get('current_price', 0.0)
        
        trade_decision = self.options_strategy.generate_trade_decision(
            vix_analysis,
            option_chain,
            current_price,
            self.account_value
        )
        
        # Execute trade if appropriate
        if trade_decision.get('action') == 'buy':
            execution_result = self.execute_trade(trade_decision)
            trade_decision['execution'] = execution_result
        
        return {'action': 'decision', 'trade_decision': trade_decision}
    
    def run_trading_loop(self, max_cycles: int = -1, interval_seconds: int = 60) -> None:
        """
        Run the trading loop for continuous trading.
        
        Args:
            max_cycles: Maximum number of cycles to run (-1 for unlimited)
            interval_seconds: Seconds to wait between cycles
        """
        self.logger.info(f"Starting trading loop with {max_cycles} max cycles, "
                       f"{interval_seconds}s interval")
        
        if not self.connect():
            self.logger.error("Failed to connect to Interactive Brokers")
            return
        
        try:
            cycle_count = 0
            
            while max_cycles == -1 or cycle_count < max_cycles:
                cycle_count += 1
                self.logger.info(f"--- Trading cycle {cycle_count} ---")
                
                try:
                    result = self.run_trading_cycle()
                    self.logger.info(f"Cycle result: {result.get('action', 'none')}")
                    
                    # If we're outside trading hours and this is continuous mode, wait longer
                    if result.get('reason') == 'outside_trading_hours' and max_cycles == -1:
                        wait_time = 300  # 5 minutes
                        self.logger.info(f"Outside trading hours, waiting {wait_time} seconds")
                    else:
                        wait_time = interval_seconds
                    
                    # Sleep between cycles
                    time.sleep(wait_time)
                    
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}")
                    time.sleep(interval_seconds)
        
        finally:
            # Make sure we close all positions before exiting
            self.logger.info("Trading loop ended, closing all positions")
            self.close_all_positions()
            self.disconnect()
    
    def run_backtest(self, start_date: str, end_date: str, risk_level: int = 5) -> Dict:
        """
        Run a backtest over a specified date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            risk_level: Risk level (1-10)
            
        Returns:
            Dict with backtest results
        """
        self.logger.info(f"Running backtest from {start_date} to {end_date} with risk level {risk_level}")
        
        # Adjust risk level
        self.config.risk.adjust_for_risk_level(risk_level)
        
        # Let backtest engine run the simulation
        # This would integrate with the backtesting module
        from backtesting.backtest_engine import BacktestEngine
        
        backtest_engine = BacktestEngine(
            self.config,
            self.volatility_analyzer,
            self.risk_manager,
            self.options_strategy
        )
        
        results = backtest_engine.run_backtest(start_date, end_date)
        
        return results


def setup_logging(log_level: str = 'INFO') -> None:
    """Set up logging configuration."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


if __name__ == "__main__":
    # Example of how to use the trader
    setup_logging('INFO')
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing TuringTrader")
    
    # Create and run trader
    trader = TuringTrader()
    
    if trader.connect():
        logger.info("Connected to Interactive Brokers")
        
        # Run a single trading cycle
        result = trader.run_trading_cycle()
        logger.info(f"Trading cycle result: {result}")
        
        # Disconnect
        trader.disconnect()
        logger.info("Disconnected from Interactive Brokers")
    else:
        logger.error("Failed to connect to Interactive Brokers")