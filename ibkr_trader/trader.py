"""
Main trader module for the TuringTrader algorithm.
This is the core controller that integrates all components and runs the trading strategy.

Live-trading safety model
-------------------------
Every trading cycle passes through ordered gates before any order is sent:
  1. Kill switch  - if the configured kill-switch file exists, flatten and halt
  2. Connection   - reconnect (with backoff) before doing anything else
  3. Market hours - IB contract hours with a timezone-aware clock fallback
  4. End of day   - mandatory flatten inside the configured close-out window
  5. Monitoring   - open positions are marked to market and closed on
                    stop-loss / profit-target *before* new entries are considered
  6. Risk gates   - daily loss limit, daily trade cap, remaining risk budget
Positions are registered with the risk manager using FILLED quantity and
actual fill prices, never the requested quantity or estimated prices.
"""

import logging
import os
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from zoneinfo import ZoneInfo

from .config import Config
from .ib_connector import IBConnector
from .volatility_analyzer import VolatilityAnalyzer
from .risk_manager import RiskManager, CREDIT_SPREAD_TYPES
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

        # Market timezone (exchange clock, not machine clock)
        self.market_tz = ZoneInfo(self.config.trading.market_timezone)

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

        # Loop robustness state
        self._consecutive_errors = 0
        self._shutdown_requested = False

    def now_market_time(self) -> datetime:
        """Current time on the exchange clock."""
        return datetime.now(self.market_tz)

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

        # Times below are on the exchange clock, never the machine clock.
        now = self.now_market_time()

        # Determine if this is a new trading day (do this even when the market
        # is closed so daily state resets before the next open).
        if self.last_check_time is None or self.last_check_time.date() != now.date():
            self.is_trading_day = True
            self.day_trade_count = 0
            self.daily_pnl = 0.0
            self.options_strategy.reset_daily_state()
            self.logger.info(f"New trading day: {now.date()}")
        self.last_check_time = now

        if not self.market_open:
            self.logger.info("Market is currently closed")
            return False

        # Regular session bounds on the exchange clock
        market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Calculate our trading window
        trading_start = market_open_time + timedelta(hours=self.day_start_offset)
        trading_end = market_close_time - timedelta(hours=self.day_end_offset)

        # Check if we're in the trading window
        self.in_trading_window = trading_start <= now <= trading_end

        if not self.in_trading_window:
            if now < trading_start:
                self.logger.info(f"Before trading window, will start at {trading_start.strftime('%H:%M:%S')}")
            else:
                self.logger.info(f"After trading window, ended at {trading_end.strftime('%H:%M:%S')}")

        return self.market_open and self.in_trading_window

    def check_kill_switch(self) -> bool:
        """
        Check for the kill-switch file. If present, flatten everything and halt.

        Returns:
            bool: True if the kill switch is active
        """
        kill_file = self.config.trading.kill_switch_file
        if kill_file and os.path.exists(kill_file):
            if not self.risk_manager.trading_halted:
                self.logger.critical(f"KILL SWITCH detected ({kill_file}): flattening all positions")
                self.close_all_positions()
            self.risk_manager.halt_trading(f"kill switch file present: {kill_file}")
            return True
        return False

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

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, trade_decision: Dict) -> Dict:
        """
        Execute a trade based on the trade decision.

        Entries use limit orders walked toward a worst-acceptable price;
        market orders are never used to open a position. Positions are
        registered with the risk manager using the actually-filled quantity.

        Args:
            trade_decision: Trade decision dictionary

        Returns:
            Dict with trade execution results
        """
        action = trade_decision.get('action')

        # Pre-trade risk gate (daily loss limit, trade cap, risk budget)
        proposed_risk = trade_decision.get('max_loss', 0.0) or 0.0
        allowed, reason = self.risk_manager.can_open_new_position(
            max_daily_trades=self.config.trading.max_daily_trades,
            proposed_risk=proposed_risk
        )
        if not allowed:
            self.logger.warning(f"Trade blocked by risk gate: {reason}")
            return {'executed': False, 'reason': f'risk_gate: {reason}'}

        if action == 'buy':
            return self._execute_single_option_buy(trade_decision)
        elif action == 'iron_condor':
            return self._execute_credit_spread_entry(trade_decision, kind='iron_condor')
        elif action == 'vertical_spread':
            return self._execute_credit_spread_entry(trade_decision, kind='vertical_spread')
        else:
            self.logger.info(f"No trade to execute or unknown action: {action}")
            return {'executed': False, 'reason': f'no_trade_action_or_unknown: {action}'}

    def _execute_single_option_buy(self, trade_decision: Dict) -> Dict:
        """Buy a single option using a protected limit order."""
        try:
            symbol = trade_decision.get('symbol')
            expiry = trade_decision.get('expiry')
            strike = trade_decision.get('strike')
            option_type = trade_decision.get('option_type', 'call')
            quantity = int(trade_decision.get('quantity', 1))
            est_price = float(trade_decision.get('price', 0.0) or 0.0)

            if quantity <= 0:
                return {'executed': False, 'reason': 'invalid_quantity'}
            if est_price <= 0:
                # Refuse to buy options without a price reference; a market
                # order on an illiquid option can fill absurdly far away.
                self.logger.error("No price estimate for option buy; refusing market order")
                return {'executed': False, 'reason': 'no_price_reference'}

            right = 'C' if option_type.lower() == 'call' else 'P'
            contract = self.ib_connector.create_option_contract(
                symbol=symbol, expiry=expiry, strike=strike, option_type=right
            )

            result = self.ib_connector.execute_limit_order_with_price_walk(
                contract=contract,
                action='BUY',
                quantity=quantity,
                initial_price=est_price,
                worst_price=est_price * 1.05,  # Pay at most 5% over the estimate
                max_attempts=self.config.trading.order_max_attempts,
                fill_timeout_seconds=self.config.trading.order_fill_timeout_seconds,
            )

            filled_qty = result['filled_quantity']
            if filled_qty <= 0:
                return {'executed': False, 'reason': 'order_not_filled'}

            avg_price = result['avg_price']
            self.day_trade_count += 1
            self.last_trade_time = self.now_market_time()

            self.risk_manager.add_position(
                symbol=f"{symbol}_{right}_{expiry}_{strike}",
                quantity=filled_qty,
                entry_price=avg_price,
                position_type='option',
                option_data={
                    'underlying': symbol,
                    'expiry': expiry,
                    'strike': strike,
                    'option_type': option_type,
                    'contract': contract,
                }
            )

            self.logger.info(f"Executed trade: BUY {filled_qty} {symbol} {option_type} "
                             f"@ {strike} exp:{expiry} AvgPrice: {avg_price}")

            return {
                'executed': True,
                'avg_price': avg_price,
                'quantity': filled_qty,
                'requested_quantity': quantity,
                'symbol': symbol,
                'option_type': option_type,
                'strike': strike,
                'expiry': expiry,
            }

        except Exception as e:
            self.logger.error(f"Error executing single option trade: {e}", exc_info=True)
            return {'executed': False, 'reason': str(e)}

    def _execute_credit_spread_entry(self, trade_decision: Dict, kind: str) -> Dict:
        """
        Open an iron condor or vertical credit spread by SELLING the combo with
        a limit order walked down from slightly above the credit estimate to
        the minimum acceptable credit (95% of estimate).
        """
        try:
            symbol = trade_decision.get('symbol')
            expiry = trade_decision.get('expiry')
            quantity = int(trade_decision.get('quantity', 0) or 0)
            net_credit_estimate = float(trade_decision.get('net_credit', 0.0) or 0.0)

            if quantity <= 0:
                self.logger.error(f"Invalid quantity for {kind} trade: {quantity}")
                return {'executed': False, 'reason': 'invalid_quantity'}
            if net_credit_estimate <= 0:
                self.logger.error(f"Invalid credit estimate for {kind} trade: {net_credit_estimate}")
                return {'executed': False, 'reason': 'invalid_credit_estimate'}

            # Build the qualified combo contract
            if kind == 'iron_condor':
                short_call_strike = trade_decision.get('short_call_strike')
                short_put_strike = trade_decision.get('short_put_strike')
                long_call_strike = trade_decision.get('long_call_strike')
                long_put_strike = trade_decision.get('long_put_strike')

                bag_contract = self.ib_connector.create_iron_condor_contract(
                    symbol, expiry, short_put_strike, short_call_strike,
                    long_put_strike, long_call_strike,
                    exchange=self.config.ibkr.default_options_exchange,
                )
                position_symbol = (f"{symbol}_IC_{expiry}_{long_put_strike:.0f}P_{short_put_strike:.0f}P_"
                                   f"{short_call_strike:.0f}C_{long_call_strike:.0f}C")
                option_data = {
                    'underlying': symbol,
                    'expiry': expiry,
                    'short_put_strike': short_put_strike,
                    'short_call_strike': short_call_strike,
                    'long_put_strike': long_put_strike,
                    'long_call_strike': long_call_strike,
                    'max_risk_per_spread': trade_decision.get('max_risk_per_spread'),
                    'estimated_credit': net_credit_estimate,
                }
            else:  # vertical_spread
                spread_type = trade_decision.get('spread_type')
                option_type = trade_decision.get('option_type')
                short_strike = trade_decision.get('short_strike')
                long_strike = trade_decision.get('long_strike')
                right = 'C' if str(option_type).lower() == 'call' else 'P'

                bag_contract = self.ib_connector.create_vertical_spread_contract(
                    symbol, expiry, short_strike, long_strike, right,
                    exchange=self.config.ibkr.default_options_exchange,
                )
                if bag_contract is None:
                    return {'executed': False, 'reason': 'could_not_build_contract'}
                position_symbol = (f"{symbol}_{spread_type}_{expiry}_"
                                   f"{long_strike:.0f}{right}_{short_strike:.0f}{right}")
                option_data = {
                    'underlying': symbol,
                    'spread_type': spread_type,
                    'option_type': option_type,
                    'expiry': expiry,
                    'short_strike': short_strike,
                    'long_strike': long_strike,
                    'width': trade_decision.get('width'),
                    'max_risk_per_spread': trade_decision.get('max_risk_per_spread'),
                    'estimated_credit': net_credit_estimate,
                }

            # Walk the credit down from 102% of the estimate to 95% (worst acceptable)
            result = self.ib_connector.execute_limit_order_with_price_walk(
                contract=bag_contract,
                action='SELL',
                quantity=quantity,
                initial_price=net_credit_estimate * 1.02,
                worst_price=net_credit_estimate * 0.95,
                max_attempts=self.config.trading.order_max_attempts,
                fill_timeout_seconds=self.config.trading.order_fill_timeout_seconds,
            )

            filled_qty = result['filled_quantity']
            if filled_qty <= 0:
                self.logger.warning(f"Failed to fill {kind} order after price walk")
                return {'executed': False, 'reason': 'order_not_filled_after_retries'}

            avg_fill_price = result['avg_price']
            self.day_trade_count += 1
            self.last_trade_time = self.now_market_time()

            fill_quality = (avg_fill_price / net_credit_estimate) * 100
            option_data['net_credit_received'] = avg_fill_price
            option_data['contract'] = bag_contract
            option_data['fill_quality'] = fill_quality

            self.risk_manager.add_position(
                symbol=position_symbol,
                quantity=filled_qty,
                entry_price=avg_fill_price,
                position_type=kind,
                option_data=option_data,
            )

            self.logger.info(f"Executed {kind}: {filled_qty} {position_symbol} "
                             f"for a credit of ${avg_fill_price:.2f} per spread "
                             f"(fill quality {fill_quality:.1f}% of estimate)")

            return {
                'executed': True,
                'avg_price_credit': avg_fill_price,
                'quantity': filled_qty,
                'requested_quantity': quantity,
                'symbol': position_symbol,
                'fill_quality_pct': fill_quality,
                'details': trade_decision,
            }

        except Exception as e:
            self.logger.error(f"Error executing {kind} trade: {e}", exc_info=True)
            return {'executed': False, 'reason': str(e)}

    # ------------------------------------------------------------------
    # Position monitoring and exits
    # ------------------------------------------------------------------

    def monitor_open_positions(self) -> List[Dict]:
        """
        Mark open positions to market and close any that hit their stop-loss
        or profit target. This runs every cycle BEFORE new entries.

        Returns:
            List of close results for positions that were exited
        """
        closed = []
        positions = self.risk_manager.get_open_positions()
        if not positions:
            return closed

        for symbol in list(positions.keys()):
            position = positions.get(symbol)
            if position is None:
                continue

            contract = (position.get('option_data') or {}).get('contract')
            if contract is None:
                self.logger.warning(f"No contract stored for {symbol}; cannot monitor")
                continue

            current_cost = self.ib_connector.get_combo_close_price(contract)
            if current_cost is None:
                self.logger.warning(f"No market quote for {symbol}; skipping monitor this cycle")
                continue

            update = self.risk_manager.update_position(symbol, current_cost)
            status = update.get('status')

            if status in ('stop_loss', 'take_profit'):
                self.logger.info(f"{symbol} hit {status} "
                                 f"(cost to close: {current_cost:.2f}); closing")
                close_result = self._close_position(symbol, position, current_cost, reason=status)
                closed.append(close_result)

        return closed

    def _close_position(self, symbol: str, position: Dict,
                        current_cost: float, reason: str) -> Dict:
        """
        Close a single position. Credit spreads are bought back with a limit
        order walked up; if the walk fails the remainder is closed at market
        (getting flat matters more than price once a stop has triggered).
        """
        contract = (position.get('option_data') or {}).get('contract')
        quantity = int(position.get('quantity', 0))
        position_type = position.get('type')

        if contract is None or quantity <= 0:
            self.risk_manager.close_position(symbol, current_cost)
            return {'symbol': symbol, 'closed': True, 'reason': reason,
                    'exit_price': current_cost, 'note': 'book-only close'}

        close_action = 'BUY' if position_type in CREDIT_SPREAD_TYPES else 'SELL'

        result = self.ib_connector.execute_limit_order_with_price_walk(
            contract=contract,
            action=close_action,
            quantity=quantity,
            initial_price=current_cost,
            worst_price=current_cost * (1.15 if close_action == 'BUY' else 0.85),
            max_attempts=self.config.trading.order_max_attempts,
            fill_timeout_seconds=self.config.trading.order_fill_timeout_seconds,
        )

        remaining = quantity - result['filled_quantity']
        exit_price = result['avg_price'] if result['filled_quantity'] > 0 else current_cost

        if remaining > 0:
            self.logger.warning(f"{symbol}: {remaining} contracts unfilled on limit close; "
                                f"escalating to market order")
            market_trade = self.ib_connector.market_order(contract, remaining, close_action)
            if market_trade is not None and self.ib_connector.wait_for_fill(market_trade, 30.0):
                mkt_price = market_trade.orderStatus.avgFillPrice or current_cost
                filled_before = result['filled_quantity']
                if filled_before > 0:
                    exit_price = ((exit_price * filled_before) + (mkt_price * remaining)) / quantity
                else:
                    exit_price = mkt_price
                remaining = 0
            else:
                self.logger.error(f"FAILED to fully close {symbol}: {remaining} contracts remain. "
                                  f"Manual intervention may be required.")

        book = self.risk_manager.close_position(symbol, exit_price)
        return {
            'symbol': symbol,
            'closed': remaining == 0,
            'reason': reason,
            'exit_price': exit_price,
            'pnl': book.get('pnl', 0.0),
        }

    def close_all_positions(self) -> Dict:
        """
        Close all open positions (end of day / shutdown / kill switch).

        Uses the connector's flatten (which fixes contract exchanges and waits
        for fills), then books the closures in the risk manager using the best
        available prices.

        Returns:
            Dict with results
        """
        self.logger.info("Closing all positions")

        positions = self.risk_manager.get_open_positions()

        # Gather current quotes for bookkeeping before flattening
        market_prices: Dict[str, float] = {}
        for symbol, position in positions.items():
            contract = (position.get('option_data') or {}).get('contract')
            if contract is not None:
                price = self.ib_connector.get_combo_close_price(contract)
                if price is not None:
                    market_prices[symbol] = price

        # Flatten everything at the broker
        report = self.ib_connector.flatten_all_positions()

        if report['all_closed']:
            self.logger.info("Successfully flattened all broker positions")
        else:
            self.logger.error(f"Failed to close some positions: {report['failed']}")

        # Book closures in the risk manager with the quotes we captured
        closed_positions = self.risk_manager.close_all_positions(market_prices)

        return {
            'closed': report['all_closed'],
            'count': len(closed_positions),
            'positions': closed_positions,
            'broker_report': report,
        }

    # ------------------------------------------------------------------
    # Trading cycle and loop
    # ------------------------------------------------------------------

    def run_trading_cycle(self) -> Dict:
        """
        Run a single trading decision cycle.

        Returns:
            Dict with cycle results
        """
        self.logger.info("Running trading cycle")

        # Gate 1: kill switch
        if self.check_kill_switch():
            return {'action': 'halt', 'reason': 'kill_switch'}

        # Gate 2: connection (reconnect if needed before making decisions)
        if not self.ib_connector.check_connection():
            return {'action': 'none', 'reason': 'disconnected'}

        # Gate 3: market hours
        if not self.check_market_status():
            return {'action': 'none', 'reason': 'outside_trading_hours'}

        # Update account info
        self.update_account_info()

        # Gate 4: end of day - mandatory flatten
        if self.options_strategy.should_close_positions(self.now_market_time()):
            open_positions = self.risk_manager.get_open_positions()
            if open_positions:
                self.logger.info("End of trading day, closing all positions")
                close_result = self.close_all_positions()
                return {'action': 'close_all', 'result': close_result}
            return {'action': 'none', 'reason': 'end_of_day'}

        # Gate 5: monitor open positions for stop-loss / profit-target
        exits = self.monitor_open_positions()
        if exits:
            return {'action': 'managed_exits', 'exits': exits}

        # If positions remain open, hold (one position at a time by design)
        open_positions = self.risk_manager.get_open_positions()
        if open_positions:
            self.logger.info(f"Already have {len(open_positions)} open positions, holding")
            return {'action': 'hold', 'positions': len(open_positions)}

        # Gate 6: risk gates for NEW entries
        allowed, reason = self.risk_manager.can_open_new_position(
            max_daily_trades=self.config.trading.max_daily_trades
        )
        if not allowed:
            self.logger.info(f"Not entering new trades: {reason}")
            return {'action': 'none', 'reason': f'risk_gate: {reason}'}

        # Fetch and analyze market data
        market_data = self.fetch_market_data()
        analysis = self.analyze_market_conditions(market_data)
        vix_analysis = analysis.get('vix_analysis', {})

        # Check if we should trade based on volatility
        if not self.volatility_analyzer.should_trade_today(vix_analysis):
            self.logger.info("Volatility conditions not favorable for trading")
            return {'action': 'none', 'reason': 'unfavorable_volatility'}

        # Fetch option chain and generate a trade decision
        option_chain = self.fetch_option_chain()
        current_price = market_data.get('current_price', 0.0)

        trade_decision = self.options_strategy.generate_trade_decision(
            vix_analysis,
            option_chain,
            current_price,
            self.account_value
        )

        # Execute trade if appropriate
        action = trade_decision.get('action', 'none')
        if action in ('buy', 'iron_condor', 'vertical_spread'):
            execution_result = self.execute_trade(trade_decision)
            trade_decision['execution'] = execution_result

        return {'action': 'decision', 'trade_decision': trade_decision}

    def _request_shutdown(self, signum, frame) -> None:
        """Signal handler: finish the current cycle, then flatten and exit."""
        self.logger.warning(f"Shutdown signal received ({signum}); "
                            f"will flatten positions and stop")
        self._shutdown_requested = True

    def run_trading_loop(self, max_cycles: int = -1, interval_seconds: int = 60) -> None:
        """
        Run the trading loop for continuous trading.

        Robustness behavior:
        - Reconnects with exponential backoff if the IB connection drops
        - Halts trading after config.trading.max_consecutive_errors in a row
        - SIGINT/SIGTERM cause a graceful flatten-and-exit
        - Always attempts to flatten all positions on the way out

        Args:
            max_cycles: Maximum number of cycles to run (-1 for unlimited)
            interval_seconds: Seconds to wait between cycles
        """
        self.logger.info(f"Starting trading loop with {max_cycles} max cycles, "
                         f"{interval_seconds}s interval")

        # Graceful shutdown on Ctrl+C / termination
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._request_shutdown)
            except (ValueError, OSError):
                pass  # Not on main thread, or unsupported signal on this OS

        if not self.connect():
            self.logger.error("Failed to connect to Interactive Brokers")
            return

        reconnect_backoff = 5  # seconds, doubles up to a cap

        try:
            cycle_count = 0

            while (max_cycles == -1 or cycle_count < max_cycles) and not self._shutdown_requested:
                cycle_count += 1
                self.logger.info(f"--- Trading cycle {cycle_count} ---")

                try:
                    # Reconnect with backoff if the connection dropped
                    if not self.ib_connector.check_connection():
                        self.logger.error(f"Connection lost; retrying in {reconnect_backoff}s")
                        time.sleep(reconnect_backoff)
                        reconnect_backoff = min(reconnect_backoff * 2, 300)
                        continue
                    reconnect_backoff = 5

                    result = self.run_trading_cycle()
                    self.logger.info(f"Cycle result: {result.get('action', 'none')}")
                    self._consecutive_errors = 0

                    if result.get('action') == 'halt':
                        self.logger.warning("Trading halted; loop will idle until halt clears")

                    # If we're outside trading hours and this is continuous mode, wait longer
                    if result.get('reason') == 'outside_trading_hours' and max_cycles == -1:
                        wait_time = 300  # 5 minutes
                        self.logger.info(f"Outside trading hours, waiting {wait_time} seconds")
                    else:
                        wait_time = interval_seconds

                    # Sleep between cycles. When connected, pump the ib_insync
                    # event loop so order/account updates keep flowing.
                    self._sleep_between_cycles(wait_time)

                except Exception as e:
                    self._consecutive_errors += 1
                    self.logger.error(f"Error in trading cycle ({self._consecutive_errors} in a row): {e}",
                                      exc_info=True)

                    max_errors = self.config.trading.max_consecutive_errors
                    if max_errors > 0 and self._consecutive_errors >= max_errors:
                        self.logger.critical(f"{self._consecutive_errors} consecutive errors; "
                                             f"flattening and halting trading")
                        try:
                            self.close_all_positions()
                        except Exception as close_err:
                            self.logger.error(f"Error flattening after repeated failures: {close_err}")
                        self.risk_manager.halt_trading(
                            f"{self._consecutive_errors} consecutive cycle errors")

                    self._sleep_between_cycles(interval_seconds)

        finally:
            # Make sure we close all positions before exiting
            self.logger.info("Trading loop ended, closing all positions")
            try:
                self.close_all_positions()
            except Exception as e:
                self.logger.error(f"Error closing positions on shutdown: {e}", exc_info=True)
            self.disconnect()

    def _sleep_between_cycles(self, seconds: float) -> None:
        """
        Wait between cycles in one-second slices so a shutdown request is
        honored promptly. Uses ib.sleep when connected to keep processing
        broker messages; otherwise falls back to time.sleep.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self._shutdown_requested:
            if self.ib_connector.connected and self.ib_connector.ib.isConnected():
                self.ib_connector.ib.sleep(1)
            else:
                time.sleep(1)

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
