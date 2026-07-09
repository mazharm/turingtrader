"""
Interactive Brokers connection and management module.
This module provides the core functionality to connect to the Interactive Brokers API
and handle order submissions, account management, and market data requests.
"""

import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from zoneinfo import ZoneInfo

# Import IB-insync for Interactive Brokers API
try:
    from ib_insync import IB, Contract, Option, Stock, Order, MarketOrder, LimitOrder
    from ib_insync import ComboLeg, Bag, util as ib_util
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

            # Batch market data requests: request everything first, wait once
            # for the data to stream in (pumping the event loop), then read.
            # The serial request->wait->cancel pattern took minutes per chain.
            def _clean(value):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return 0.0
                return value

            requested = []  # (expiry, right, strike, contract, ticker)
            max_data_lines = 80  # Stay under IB's concurrent market data line limits

            for expiry in sorted_expirations[:5]:  # Limit to first 5 expirations
                if len(requested) >= max_data_lines:
                    break
                for right in ['C', 'P']:
                    for strike in chain_data[expiry]['strikes'][::5]:  # Sample every 5th strike
                        if len(requested) >= max_data_lines:
                            break
                        contract = Option(symbol, expiry, strike, right, exchange)
                        try:
                            qualified = self.ib.qualifyContracts(contract)
                            if not qualified:
                                continue
                            # Generic tick 101 adds open interest
                            ticker = self.ib.reqMktData(contract, '101', False, False)
                            requested.append((expiry, right, strike, contract, ticker))
                        except Exception as e:
                            self.logger.debug(f"Skipping {symbol} {expiry} {strike} {right}: {e}")

            # Give the data time to arrive while processing network messages
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                self.ib.sleep(0.25)
                if all(t.bid is not None or t.last is not None for *_, t in requested):
                    break

            for expiry, right, strike, contract, ticker in requested:
                try:
                    greeks = ticker.modelGreeks
                    option_data = {
                        'bid': _clean(ticker.bid),
                        'ask': _clean(ticker.ask),
                        'last': _clean(ticker.last),
                        'volume': _clean(ticker.volume),
                        'iv': _clean(ticker.impliedVolatility) or
                              (_clean(greeks.impliedVol) if greeks else 0.0),
                        'delta': _clean(greeks.delta) if greeks else 0.0,
                        'open_interest': _clean(ticker.callOpenInterest if right == 'C'
                                                else ticker.putOpenInterest),
                    }
                    side = 'calls' if right == 'C' else 'puts'
                    chain_data[expiry][side][strike] = option_data
                except Exception as e:
                    self.logger.error(f"Error reading option data for {symbol} {expiry} {strike} {right}: {e}")
                finally:
                    try:
                        self.ib.cancelMktData(contract)
                    except Exception:
                        pass

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

        Kept for backward compatibility; delegates to flatten_all_positions(),
        which fixes the exchange on position contracts and waits for fills.

        Returns:
            bool: True if all positions closed successfully
        """
        return self.flatten_all_positions()['all_closed']
    
    @staticmethod
    def _parse_ib_hours(hours_str: str, tz: ZoneInfo, now: datetime) -> Optional[bool]:
        """
        Parse an IB tradingHours/liquidHours string and report whether `now`
        falls inside a session. Handles both IB formats:
          old: '20090507:0700-1830,1830-2330;20090508:CLOSED'
          new: '20180323:0400-20180323:2000;20180326:0400-20180326:2000'

        Returns True/False, or None if nothing matched today's date (parse failure).
        """
        today_str = now.strftime('%Y%m%d')
        found_today = False

        for schedule in hours_str.split(';'):
            schedule = schedule.strip()
            if not schedule or not schedule.startswith(today_str):
                continue
            found_today = True

            if schedule.endswith('CLOSED'):
                continue

            # Strip the leading 'YYYYMMDD:' then examine each comma-separated range
            ranges = schedule[len(today_str) + 1:]
            for time_range in ranges.split(','):
                if '-' not in time_range:
                    continue
                start_str, end_str = time_range.split('-', 1)

                try:
                    # New format has full datetimes on both sides ('20180323:0400')
                    if ':' in start_str:
                        start_dt = datetime.strptime(start_str, '%Y%m%d:%H%M')
                        end_dt = datetime.strptime(end_str, '%Y%m%d:%H%M')
                    else:
                        start_dt = datetime.strptime(f"{today_str}{start_str}", '%Y%m%d%H%M')
                        end_dt = datetime.strptime(f"{today_str}{end_str}", '%Y%m%d%H%M')
                except ValueError:
                    continue

                start_dt = start_dt.replace(tzinfo=tz)
                end_dt = end_dt.replace(tzinfo=tz)

                if start_dt <= now <= end_dt:
                    return True

        return False if found_today else None

    def is_market_open(self) -> bool:
        """
        Check if the market is currently open (regular trading hours).

        Uses IB contract details when available, falling back to a
        US-equity-market clock check (Mon-Fri 9:30-16:00 ET) so that a
        parsing problem cannot silently disable trading.

        Returns:
            bool: True if market is open
        """
        eastern = ZoneInfo('America/New_York')

        if self.check_connection():
            try:
                contract = Stock('SPY', 'SMART', 'USD')
                self.ib.qualifyContracts(contract)

                details_list = self.ib.reqContractDetails(contract)
                if details_list:
                    details = details_list[0]
                    # Prefer liquidHours (regular session); tradingHours includes extended hours
                    hours_str = details.liquidHours or details.tradingHours
                    tz_id = details.timeZoneId or 'America/New_York'
                    try:
                        tz = ZoneInfo(tz_id)
                    except Exception:
                        tz = eastern

                    if hours_str:
                        now = datetime.now(tz)
                        result = self._parse_ib_hours(hours_str, tz, now)
                        if result is not None:
                            return result
                        self.logger.warning("Could not find today's session in IB hours string; "
                                            "falling back to clock check")
            except Exception as e:
                self.logger.error(f"Error checking market status via IB: {e}; "
                                  f"falling back to clock check")

        # Fallback: regular US equity hours in Eastern time.
        # NOTE: does not account for exchange holidays; the strategy simply
        # finds no tradable market data on those days.
        now_et = datetime.now(eastern)
        if now_et.weekday() >= 5:  # Saturday/Sunday
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et <= market_close
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error

    def wait_for_fill(self, trade: Any, timeout_seconds: float) -> bool:
        """
        Pump the ib_insync event loop until the trade fills, terminally fails,
        or the timeout elapses. NEVER use time.sleep() for this — it starves
        the event loop and order status would never update.

        Returns:
            bool: True if the order is completely filled
        """
        if trade is None:
            return False
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if trade.orderStatus.status == 'Filled':
                return True
            if trade.isDone():  # Cancelled / rejected / inactive
                return False
            self.ib.sleep(0.25)
        return trade.orderStatus.status == 'Filled'

    def cancel_and_wait(self, trade: Any, timeout_seconds: float = 10.0) -> int:
        """
        Cancel a working order and wait for the cancellation to settle, so that
        the final filled quantity is known before any re-submission.

        Returns:
            int: Quantity filled before the cancel took effect
        """
        if trade is None:
            return 0
        try:
            if not trade.isDone():
                self.ib.cancelOrder(trade.order)
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline and not trade.isDone():
                    self.ib.sleep(0.25)
        except Exception as e:
            self.logger.error(f"Error cancelling order {trade.order.orderId}: {e}")
        return int(trade.orderStatus.filled or 0)

    def execute_limit_order_with_price_walk(self,
                                            contract: Contract,
                                            action: str,
                                            quantity: int,
                                            initial_price: float,
                                            worst_price: float,
                                            max_attempts: int = 3,
                                            fill_timeout_seconds: float = 15.0) -> Dict:
        """
        Work a limit order toward the market: submit at initial_price, and if it
        does not fill within the timeout, cancel and re-submit the REMAINING
        quantity at a price stepped toward worst_price. Never crosses beyond
        worst_price. Partial fills are accounted for across attempts.

        For SELL (credit) orders prices walk DOWN from initial to worst.
        For BUY (debit) orders prices walk UP from initial to worst.

        Returns:
            Dict with:
              filled_quantity: total contracts filled
              avg_price: volume-weighted average fill price (0 if none)
              complete: True if the full quantity filled
              trades: list of ib_insync Trade objects created
        """
        result = {'filled_quantity': 0, 'avg_price': 0.0, 'complete': False, 'trades': []}

        if not self.check_connection() or quantity <= 0:
            return result

        action = action.upper()
        if action not in ('BUY', 'SELL'):
            self.logger.error(f"Invalid order action: {action}")
            return result

        # Build the price ladder from initial to worst (inclusive), monotonic
        # in the direction that improves fill probability.
        if max_attempts < 1:
            max_attempts = 1
        if max_attempts == 1:
            prices = [initial_price]
        else:
            step = (worst_price - initial_price) / (max_attempts - 1)
            prices = [initial_price + step * i for i in range(max_attempts)]

        remaining = quantity
        total_value = 0.0
        total_filled = 0

        for attempt, raw_price in enumerate(prices, start=1):
            # Options tick in cents; keep prices sane and positive
            price = max(0.01, round(raw_price, 2))

            order = LimitOrder(action, remaining, price)
            order.tif = 'DAY'

            try:
                trade = self.ib.placeOrder(contract, order)
            except Exception as e:
                self.logger.error(f"Error placing order attempt {attempt}: {e}")
                break

            result['trades'].append(trade)
            self.logger.info(f"Order attempt {attempt}/{max_attempts}: {action} {remaining} "
                             f"{contract.symbol} @ ${price:.2f}")

            filled_now = self.wait_for_fill(trade, fill_timeout_seconds)

            if not filled_now:
                filled_qty = self.cancel_and_wait(trade)
            else:
                filled_qty = int(trade.orderStatus.filled or 0)

            if filled_qty > 0:
                fill_price = trade.orderStatus.avgFillPrice or price
                total_value += fill_price * filled_qty
                total_filled += filled_qty
                remaining -= filled_qty
                self.logger.info(f"Attempt {attempt} filled {filled_qty} @ ${fill_price:.2f}, "
                                 f"{remaining} remaining")

            if remaining <= 0:
                break

        result['filled_quantity'] = total_filled
        result['avg_price'] = (total_value / total_filled) if total_filled > 0 else 0.0
        result['complete'] = remaining <= 0

        if not result['complete'] and total_filled > 0:
            self.logger.warning(f"Partial fill: {total_filled}/{quantity} contracts. "
                                f"Position tracking must use the FILLED quantity.")
        elif total_filled == 0:
            self.logger.warning(f"Order not filled after {max_attempts} attempts")

        return result

    def get_combo_close_price(self, bag_contract: Contract,
                              timeout_seconds: float = 5.0) -> Optional[float]:
        """
        Get the current cost (debit, per spread) to BUY BACK a combo position.
        Used for monitoring credit spreads against stop-loss/profit targets.

        Returns:
            float mid/ask-based estimate, or None if no usable quote arrived
        """
        if not self.check_connection():
            return None

        ticker = None
        try:
            ticker = self.ib.reqMktData(bag_contract, '', False, False)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                self.ib.sleep(0.25)
                bid, ask = ticker.bid, ticker.ask
                if bid is not None and ask is not None and not math.isnan(bid) and not math.isnan(ask) and ask > 0:
                    return (bid + ask) / 2.0
                last = ticker.last
                if last is not None and not math.isnan(last) and last > 0:
                    return last
            return None
        except Exception as e:
            self.logger.error(f"Error getting combo price: {e}")
            return None
        finally:
            if ticker is not None:
                try:
                    self.ib.cancelMktData(bag_contract)
                except Exception:
                    pass

    def flatten_all_positions(self) -> Dict:
        """
        Close every open position in the account and report fill details.

        Contracts returned by ib.positions() carry no exchange, so one is
        assigned before ordering (a close order without an exchange is
        rejected by IB).

        Returns:
            Dict with:
              all_closed: True if every close order filled
              closed: list of {symbol, conId, action, quantity, fill_price}
              failed: list of {symbol, conId, reason}
        """
        report = {'all_closed': True, 'closed': [], 'failed': []}

        if not self.check_connection():
            report['all_closed'] = False
            return report

        try:
            # Cancel any working orders first so they can't race our closes
            open_trades = self.ib.openTrades()
            for open_trade in open_trades:
                try:
                    self.ib.cancelOrder(open_trade.order)
                except Exception as e:
                    self.logger.warning(f"Could not cancel open order {open_trade.order.orderId}: {e}")
            if open_trades:
                self.ib.sleep(1.0)

            positions = self.ib.positions()
            if not positions:
                self.logger.info("No positions to flatten")
                return report

            for position in positions:
                contract = position.contract
                pos_size = position.position
                if pos_size == 0:
                    continue

                # Positions come back without an exchange; orders need one.
                if not contract.exchange:
                    contract.exchange = 'SMART'

                action = 'SELL' if pos_size > 0 else 'BUY'
                quantity = int(abs(pos_size))

                self.logger.info(f"Flattening: {action} {quantity} {contract.localSymbol or contract.symbol}")

                order = MarketOrder(action, quantity)
                order.tif = 'DAY'
                try:
                    trade = self.ib.placeOrder(contract, order)
                except Exception as e:
                    report['all_closed'] = False
                    report['failed'].append({'symbol': contract.localSymbol or contract.symbol,
                                             'conId': contract.conId, 'reason': str(e)})
                    continue

                filled = self.wait_for_fill(trade, timeout_seconds=30.0)
                if filled:
                    report['closed'].append({
                        'symbol': contract.localSymbol or contract.symbol,
                        'conId': contract.conId,
                        'action': action,
                        'quantity': quantity,
                        'fill_price': trade.orderStatus.avgFillPrice or 0.0,
                    })
                else:
                    report['all_closed'] = False
                    report['failed'].append({
                        'symbol': contract.localSymbol or contract.symbol,
                        'conId': contract.conId,
                        'reason': f"close order status: {trade.orderStatus.status}",
                    })
                    self.logger.error(f"POSITION NOT CLOSED: {contract.localSymbol or contract.symbol} "
                                      f"status={trade.orderStatus.status} — manual intervention may be required")

            return report

        except Exception as e:
            self.logger.error(f"Error flattening positions: {e}", exc_info=True)
            report['all_closed'] = False
            return report
        
    def create_iron_condor_contract(self,
                              symbol: str,
                              expiry: str,  # Format: YYYYMMDD
                              short_put_strike: float,
                              short_call_strike: float,
                              long_put_strike: float,
                              long_call_strike: float,
                              exchange: str = 'SMART',
                              currency: str = 'USD') -> Bag:
        """
        Create an iron condor combination contract.
        
        Args:
            symbol: Underlying symbol
            expiry: Option expiry date in YYYYMMDD format
            short_put_strike: Strike price for short put
            short_call_strike: Strike price for short call
            long_put_strike: Strike price for long put
            long_call_strike: Strike price for long call
            exchange: Exchange name
            currency: Currency code
            
        Returns:
            Bag contract representing the iron condor
        """
        # Create combo legs for iron condor
        legs = []
        
        # Long put leg (buy)
        legs.append(ComboLeg(
            conId=0,  # Will be filled by IB
            ratio=1,
            action='BUY',
            exchange=exchange,
            designatedLocation='',  # Local Symbol
            openClose=0  # Open position
        ))
        
        # Short put leg (sell)
        legs.append(ComboLeg(
            conId=0,  # Will be filled by IB
            ratio=1,
            action='SELL',
            exchange=exchange,
            designatedLocation='',
            openClose=0
        ))
        
        # Short call leg (sell)
        legs.append(ComboLeg(
            conId=0,  # Will be filled by IB
            ratio=1,
            action='SELL',
            exchange=exchange,
            designatedLocation='',
            openClose=0
        ))
        
        # Long call leg (buy)
        legs.append(ComboLeg(
            conId=0,  # Will be filled by IB
            ratio=1,
            action='BUY',
            exchange=exchange,
            designatedLocation='',
            openClose=0
        ))
        
        # Create individual option contracts to get contract IDs
        long_put = Option(symbol, expiry, long_put_strike, 'P', exchange, currency)
        short_put = Option(symbol, expiry, short_put_strike, 'P', exchange, currency)
        short_call = Option(symbol, expiry, short_call_strike, 'C', exchange, currency)
        long_call = Option(symbol, expiry, long_call_strike, 'C', exchange, currency)
        
        # Qualify contracts to get contract IDs
        self.ib.qualifyContracts(long_put, short_put, short_call, long_call)

        # An unqualified leg would produce a conId of 0 and the whole combo
        # order would be rejected; fail fast instead.
        for leg_name, leg_contract in [('long put', long_put), ('short put', short_put),
                                       ('short call', short_call), ('long call', long_call)]:
            if not leg_contract.conId:
                raise ValueError(f"Could not qualify {leg_name} leg "
                                 f"({symbol} {expiry} {leg_contract.strike} {leg_contract.right})")

        # Update combo legs with contract IDs
        legs[0].conId = long_put.conId
        legs[1].conId = short_put.conId
        legs[2].conId = short_call.conId
        legs[3].conId = long_call.conId
        
        # Create bag (combination) contract
        bag = Bag(symbol, exchange, currency)
        bag.comboLegs = legs
        
        # Set a descriptive name
        bag.symbol = symbol
        
        return bag
        
    def close_vertical_spread(self,
                        bag_contract: Bag,
                        quantity: int,
                        limit_price: Optional[float] = None,
                        action: str = 'BUY',  # 'BUY' to close a short spread, 'SELL' to close a long spread
                        max_attempts: int = 3,
                        price_increment_pct: float = 5.0
                        ) -> Any:
        """
        Close a vertical spread position by submitting an opposing order.
        
        Args:
            bag_contract: The qualified Bag contract of the vertical spread to close
            quantity: Number of spreads to close
            limit_price: Limit price for closing. For a short spread (credit spread), this is the debit to pay
            action: Order action ('BUY' to close a short spread, 'SELL' to close a long spread)
            max_attempts: Maximum number of attempts to fill the order
            price_increment_pct: Percentage to increase the limit price by on each retry
            
        Returns:
            Trade object if successful, None otherwise
        """
        if not self.check_connection():
            return None
            
        try:
            if limit_price is None or limit_price <= 0:
                self.logger.warning("No valid limit price provided for closing vertical spread. Using market order.")
                order = MarketOrder(action, quantity)
                
                # Submit market order
                trade = self.ib.placeOrder(bag_contract, order)
                
                self.logger.info(f"Market order submitted to close vertical spread: {action} {quantity} {bag_contract.symbol}")
                
                # Wait for order to be acknowledged
                timeout = time.time() + 10 
                while time.time() < timeout:
                    self.ib.sleep(0.5)
                    if trade.isDone():
                        break
                
                return trade
            
            # For limit orders, implement price improvement with multiple attempts
            filled = False
            attempt = 1
            current_limit = limit_price
            last_trade = None
            
            while not filled and attempt <= max_attempts:
                # Create a limit order with the current price
                order = LimitOrder(action, quantity, current_limit)
                order.tif = 'GTC'  # Good Till Cancelled
                
                # Submit the order
                trade = self.ib.placeOrder(bag_contract, order)
                last_trade = trade
                
                self.logger.info(f"Limit order attempt {attempt}/{max_attempts} to close vertical spread: "
                               f"{action} {quantity} {bag_contract.symbol} @ ${current_limit:.2f}")
                
                # Wait for the order to potentially fill
                wait_time = 5 if attempt < max_attempts else 10  # Wait longer on final attempt
                timeout = time.time() + wait_time
                
                while time.time() < timeout:
                    self.ib.sleep(0.5)
                    if trade.orderStatus.status == 'Filled':
                        filled = True
                        self.logger.info(f"Vertical spread close order filled: {trade.orderStatus.filled} @ {trade.orderStatus.avgFillPrice}")
                        break
                
                if filled:
                    break
                
                # If not filled and not the last attempt, cancel and retry with higher price
                if attempt < max_attempts:
                    self.ib.cancelOrder(order)
                    
                    # For BUY orders (closing a short spread), increase the price
                    # For SELL orders (closing a long spread), decrease the price
                    if action == 'BUY':
                        # Increase the limit price by the specified percentage
                        price_increment = current_limit * (price_increment_pct / 100.0)
                        current_limit += price_increment
                    else:  # 'SELL'
                        # Decrease the limit price by the specified percentage
                        price_decrement = current_limit * (price_increment_pct / 100.0)
                        current_limit -= price_decrement
                    
                    self.logger.info(f"Adjusting limit price to ${current_limit:.2f} for next attempt")
                
                attempt += 1
            
            # If we couldn't fill after all attempts, return the last trade
            if not filled:
                self.logger.warning(f"Failed to fill vertical spread close order after {max_attempts} attempts")
            
            return last_trade
                
        except Exception as e:
            self.logger.error(f"Error closing vertical spread: {e}", exc_info=True)
            return None

    def submit_iron_condor_order(self,
                               symbol: str,
                               expiry: str,
                               short_put_strike: float,
                               short_call_strike: float,
                               long_put_strike: float,
                               long_call_strike: float,
                               quantity: int,
                               limit_price: Optional[float] = None, # This is the NET CREDIT desired
                               action: str = 'SELL', # 'SELL' opens the condor for a credit
                               exchange: str = 'SMART', # Default to SMART for underlying, options might use specific exchanges
                               currency: str = 'USD') -> Any:
        """
        Submit an iron condor order.
        For an iron condor, we are typically SELLING the spread to receive a net credit.
        
        Args:
            symbol: Underlying symbol
            expiry: Option expiry date in YYYYMMDD format
            short_put_strike: Strike price for short put
            short_call_strike: Strike price for short call
            long_put_strike: Strike price for long put
            long_call_strike: Strike price for long call
            quantity: Number of iron condor spreads
            limit_price: Limit price for the entire spread (NET CREDIT desired). 
                         IB requires a positive price for limit orders.
                         This price represents the credit you want to receive per spread.
            exchange: The exchange for the combo. BOX, CBOE, or other specific option exchanges might be better.
                      'SMART' might not be ideal for complex option spreads. Consider 'BOX' or allowing override.
            currency: Currency code
            
        Returns:
            Trade object if successful, None otherwise
        """
        if not self.check_connection():
            return None
            
        try:
            # Create iron condor contract
            # Ensure the exchange used for leg creation is appropriate (e.g., 'BOX', 'CBOE2')
            # Using 'SMART' for individual legs might be okay if IB can resolve them,
            # but the BAG contract itself should ideally be routed to an options exchange.
            bag_contract = self.create_iron_condor_contract(
                symbol, expiry, short_put_strike, short_call_strike,
                long_put_strike, long_call_strike, exchange=self.ibkr_config.default_options_exchange, currency=currency
            )
            
            # Qualify the bag contract itself
            self.ib.qualifyContracts(bag_contract)

            order_action = action.upper()
            if order_action not in ('SELL', 'BUY'):
                self.logger.error(f"Invalid iron condor action: {action}")
                return None

            # Create order: To receive a credit, this must be a SELL order.
            # The limit_price should be the positive credit amount.
            if limit_price is not None:
                if limit_price <= 0:
                    self.logger.warning("Limit price for credit must be positive. Using abs(limit_price).")
                lmt_price = abs(limit_price) # IB expects positive price for limit orders
                order = LimitOrder(order_action, quantity, lmt_price)
                order.tif = 'DAY'  # This is an intraday strategy; never leave GTC orders behind
            else:
                # Market orders for complex spreads are generally not recommended due to slippage.
                # Consider raising an error or requiring a limit price.
                self.logger.warning("Submitting Iron Condor as a Market Order. This is risky.")
                order = MarketOrder(order_action, quantity)
                
            # Submit order
            trade = self.ib.placeOrder(bag_contract, order) # Use placeOrder directly
            
            self.logger.info(f"Order submitted: {order.action} {order.totalQuantity} {bag_contract.symbol} Iron Condor")

            # Wait for order to be acknowledged and potentially fill
            timeout = time.time() + 10 # Increased timeout for complex orders
            while time.time() < timeout:
                self.ib.sleep(0.5) # ib_insync's sleep processes messages
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                self.logger.info(f"Iron Condor order filled: {trade.orderStatus.filled} @ {trade.orderStatus.avgFillPrice}")
            elif trade.orderStatus.status == 'Submitted' or trade.orderStatus.status == 'PendingSubmit':
                 self.logger.info(f"Iron Condor order submitted/pending, current status: {trade.orderStatus.status}")
            else:
                self.logger.warning(f"Iron Condor order status: {trade.orderStatus.status}, Filled: {trade.orderStatus.filled}")

            return trade
            
        except Exception as e:
            self.logger.error(f"Error submitting iron condor order: {e}", exc_info=True)
            return None
            
    def create_vertical_spread_contract(self,
                            symbol: str,
                            expiry: str,
                            short_strike: float,
                            long_strike: float,
                            right: str,  # 'C' for call, 'P' for put
                            exchange: str = 'SMART',
                            currency: str = 'USD') -> Optional[Contract]:
        """
        Create and qualify a two-leg vertical spread BAG contract
        (leg 0 = short/SELL, leg 1 = long/BUY).

        Returns:
            Qualified Bag contract, or None if the legs could not be qualified
        """
        # Validate spread geometry
        if right == 'C':
            if long_strike <= short_strike:
                self.logger.error("Bear call spread requires long_strike > short_strike")
                return None
        elif right == 'P':
            if short_strike <= long_strike:
                self.logger.error("Bull put spread requires short_strike > long_strike")
                return None
        else:
            self.logger.error(f"Invalid option type: {right}, must be 'C' or 'P'")
            return None

        short_contract = Option(symbol, expiry, short_strike, right, exchange)
        long_contract = Option(symbol, expiry, long_strike, right, exchange)
        self.ib.qualifyContracts(short_contract, long_contract)

        if not short_contract.conId or not long_contract.conId:
            self.logger.error(f"Could not qualify vertical spread legs "
                              f"({symbol} {expiry} {short_strike}/{long_strike} {right})")
            return None

        legs = [
            ComboLeg(conId=short_contract.conId, ratio=1, action='SELL',
                     exchange=exchange, openClose=0, designatedLocation='', exemptCode=-1),
            ComboLeg(conId=long_contract.conId, ratio=1, action='BUY',
                     exchange=exchange, openClose=0, designatedLocation='', exemptCode=-1),
        ]

        bag_contract = Contract()
        bag_contract.symbol = symbol
        bag_contract.secType = 'BAG'
        bag_contract.currency = currency
        bag_contract.exchange = exchange
        bag_contract.comboLegs = legs
        return bag_contract

    def submit_vertical_spread_order(self,
                            symbol: str,
                            expiry: str,
                            short_strike: float,
                            long_strike: float,
                            right: str,  # 'C' for call, 'P' for put
                            quantity: int,
                            action: str = 'SELL',  # 'SELL' for credit spreads, 'BUY' for debit spreads
                            limit_price: Optional[float] = None,
                            exchange: str = 'SMART',
                            currency: str = 'USD') -> Any:
        """
        Submit a vertical spread order.
        
        Args:
            symbol: Underlying symbol
            expiry: Option expiry date in YYYYMMDD format
            short_strike: Strike price for short leg
            long_strike: Strike price for long leg
            right: Option type ('C' for call, 'P' for put)
            quantity: Number of spreads
            action: 'SELL' for credit spreads (bull put, bear call), 'BUY' for debit spreads
            limit_price: Limit price for the spread. For credit spreads, this is the net credit desired.
            exchange: Exchange name
            currency: Currency code
            
        Returns:
            Trade object if successful, None otherwise
        """
        if not self.check_connection():
            return None

        try:
            # Create and qualify the spread contract
            bag_contract = self.create_vertical_spread_contract(
                symbol, expiry, short_strike, long_strike, right, exchange, currency
            )
            if bag_contract is None:
                return None

            # Create order
            if limit_price is not None:
                if limit_price <= 0 and action == 'SELL':
                    self.logger.warning("Limit price for credit spread must be positive")
                    limit_price = abs(limit_price)
                order = LimitOrder(action, quantity, limit_price)
                order.tif = 'DAY'  # Intraday strategy; never leave GTC orders behind
            else:
                self.logger.warning("Submitting vertical spread as Market Order (risky)")
                order = MarketOrder(action, quantity)
                
            # Submit order
            trade = self.ib.placeOrder(bag_contract, order)
            
            self.logger.info(f"Vertical spread order submitted: {action} {quantity} {symbol} {right} spread @ {short_strike}/{long_strike}")
            
            # Wait for acknowledgement
            timeout = time.time() + 10
            while time.time() < timeout:
                self.ib.sleep(0.5)
                if trade.isDone():
                    break
                    
            return trade
            
        except Exception as e:
            self.logger.error(f"Error submitting vertical spread order: {e}", exc_info=True)
            return None

    def close_iron_condor(self,
                        bag_contract: 'Bag', # Pass the qualified Bag contract of the open position
                        quantity: int,
                        limit_price: Optional[float] = None, # This is the NET DEBIT to pay
                        max_attempts: int = 3,
                        price_increment_pct: float = 5.0 # Percentage to increase price by on retries
                        ) -> Any:
        """
        Close an iron condor position by submitting an opposing order.
        To close a short iron condor (established for a credit), we BUY it back, paying a debit.
        
        Args:
            bag_contract: The qualified Bag contract of the iron condor to close.
            quantity: Number of iron condor spreads to close.
            limit_price: Limit price for closing (NET DEBIT to pay).
                         IB requires a positive price.
            max_attempts: Maximum number of attempts to fill the order.
            price_increment_pct: Percentage to increase the limit price by on each retry.
                         
        Returns:
            Trade object if successful, None otherwise
        """
        if not self.check_connection():
            return None
            
        try:
            # Create order: To close a short condor (sold for credit), we BUY it back.
            order_action = 'BUY' # Buying back the condor
            
            if limit_price is None or limit_price <= 0:
                self.logger.warning("No valid limit price provided for closing iron condor. Using market order.")
                order = MarketOrder(order_action, quantity)
                
                # Submit market order
                trade = self.ib.placeOrder(bag_contract, order)
                
                self.logger.info(f"Market order submitted to close Iron Condor: {order_action} {quantity} {bag_contract.symbol}")
                
                # Wait for order to be acknowledged
                timeout = time.time() + 10 
                while time.time() < timeout:
                    self.ib.sleep(0.5)
                    if trade.isDone():
                        break
                
                return trade
            
            # For limit orders, implement price improvement with multiple attempts
            filled = False
            attempt = 1
            current_limit = limit_price
            last_trade = None
            
            while not filled and attempt <= max_attempts:
                # Create a limit order with the current price
                order = LimitOrder(order_action, quantity, current_limit)
                order.tif = 'GTC'  # Good Till Cancelled
                
                # Submit the order
                trade = self.ib.placeOrder(bag_contract, order)
                last_trade = trade
                
                self.logger.info(f"Limit order attempt {attempt}/{max_attempts} to close Iron Condor: "
                               f"{order_action} {quantity} {bag_contract.symbol} @ ${current_limit:.2f}")
                
                # Wait for the order to potentially fill
                wait_time = 5 if attempt < max_attempts else 10  # Wait longer on final attempt
                timeout = time.time() + wait_time
                
                while time.time() < timeout:
                    self.ib.sleep(0.5)
                    if trade.orderStatus.status == 'Filled':
                        filled = True
                        self.logger.info(f"Iron Condor close order filled: {trade.orderStatus.filled} @ {trade.orderStatus.avgFillPrice}")
                        break
                
                if filled:
                    break
                
                # If not filled and not the last attempt, cancel and retry with higher price
                if attempt < max_attempts:
                    self.ib.cancelOrder(order)
                    
                    # Increase the limit price by the specified percentage
                    price_increment = current_limit * (price_increment_pct / 100.0)
                    current_limit += price_increment
                    
                    self.logger.info(f"Increasing limit price to ${current_limit:.2f} for next attempt")
                
                attempt += 1
            
            # If we couldn't fill after all attempts, return the last trade
            if not filled:
                self.logger.warning(f"Failed to fill Iron Condor close order after {max_attempts} attempts")
            
            return last_trade
                
        except Exception as e:
            self.logger.error(f"Error closing iron condor: {e}", exc_info=True)
            return None


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