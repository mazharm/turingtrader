"""
Backtesting engine for the TuringTrader algorithm.
"""

import logging
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Local imports
from ibkr_trader.config import Config
from ibkr_trader.volatility_analyzer import VolatilityAnalyzer
from ibkr_trader.risk_manager import RiskManager
from ibkr_trader.options_strategy import OptionsStrategy
from historical_data.data_fetcher import HistoricalDataFetcher


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    a = 0.4361836
    b = -0.1201676
    c = 0.9372980
    k = 1.0 / (1.0 + 0.33267 * abs(x))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * (a * k + b * k * k + c * k * k * k)
    return cdf if x >= 0 else 1.0 - cdf


class BacktestEngine:
    """Engine for backtesting the TuringTrader algorithm."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        volatility_analyzer: Optional[VolatilityAnalyzer] = None,
        risk_manager: Optional[RiskManager] = None,
        options_strategy: Optional[OptionsStrategy] = None,
        initial_balance: float = 100000.0,
        data_fetcher: Optional[Any] = None,  # Added data_fetcher argument
        holding_days: int = 1
    ):
        """
        Initialize the backtesting engine.

        Args:
            config: Configuration object
            volatility_analyzer: Volatility analyzer instance
            risk_manager: Risk manager instance
            options_strategy: Options strategy instance
            initial_balance: Initial account balance
            data_fetcher: Data fetcher instance (e.g., HistoricalDataFetcher or MockDataFetcher)
            holding_days: Trading days a position is held before it is closed.
                Positions always settle at expiry if that comes first. The
                default of 1 preserves the strategy's daily open/close cycle.
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config = config or Config()
        self.volatility_analyzer = volatility_analyzer or VolatilityAnalyzer(self.config)
        self.risk_manager = risk_manager or RiskManager(self.config)
        self.options_strategy = options_strategy or OptionsStrategy(
            self.config, 
            self.volatility_analyzer, 
            self.risk_manager
        )
        
        # Initialize data fetcher - allow injection
        self.data_fetcher = data_fetcher or HistoricalDataFetcher()
        
        # Backtest state
        self.initial_balance = initial_balance
        self.holding_days = max(1, holding_days)
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_returns = []
        self.daily_values = []
        self.positions = {}
        self.trades = []
        self.trade_history = []
        self._current_day_index = 0

    def _reset_backtest_state(self) -> None:
        """Reset the backtest state."""
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.daily_returns = []
        self.daily_values = []
        self.positions = {}
        self.trades = []
        self.trade_history = []
        self._current_day_index = 0
        
        # Reset component states
        self.risk_manager.reset_daily_metrics()
        self.risk_manager.update_account_value(self.initial_balance)
        self.options_strategy.reset_daily_state()
    
    def run_backtest(self, start_date: str, end_date: str, risk_level: int = 5, use_cache: bool = True) -> Dict:
        """
        Run a backtest over a specified date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            risk_level: Risk level (1-10)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dict with backtest results
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date} with risk level {risk_level}")
        
        # Reset state for fresh backtest
        self._reset_backtest_state()
        
        # Set risk level
        self.config.risk.adjust_for_risk_level(risk_level)
        
        try:
            # Parse dates
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Fetch historical data
            self.logger.info("Fetching historical market data...")
            
            # Fetch S&P500 (underlying) data
            underlying_data = self.data_fetcher.fetch_data(
                self.config.trading.index_symbol,
                start_date,
                end_date,
                use_cache=use_cache
            )
            
            # Fetch VIX data
            vix_data = self.data_fetcher.fetch_data(
                'VIX',
                start_date,
                end_date,
                use_cache=use_cache
            )
            
            # Align dates between VIX and underlying
            common_dates = sorted(set(underlying_data.index) & set(vix_data.index))
            
            if not common_dates:
                raise ValueError("No overlapping dates between underlying and VIX data")
            
            self.logger.info(f"Backtesting {len(common_dates)} trading days")
            
            # Process each trading day
            for day_index, date in enumerate(common_dates):
                self._current_day_index = day_index
                self.logger.debug(f"Processing date: {date.strftime('%Y-%m-%d')}")
                
                # Reset daily state
                self.risk_manager.reset_daily_metrics()
                self.options_strategy.reset_daily_state()
                
                # Get data for this day
                daily_vix = vix_data.loc[date]
                daily_underlying = underlying_data.loc[date]
                
                # Convert to format expected by the strategy
                vix_data_point = [{
                    'date': date.strftime('%Y-%m-%d'),
                    'close': daily_vix['close']
                }]
                
                # Previous days for VIX trend
                prev_dates = [d for d in common_dates if d < date][-5:]
                for prev_date in prev_dates:
                    vix_data_point.insert(0, {
                        'date': prev_date.strftime('%Y-%m-%d'),
                        'close': vix_data.loc[prev_date, 'close']
                    })
                
                # Analyze VIX
                vix_analysis = self.volatility_analyzer.analyze_vix(vix_data_point)
                
                # Determine if we should trade today
                should_trade = self.volatility_analyzer.should_trade_today(vix_analysis)
                
                # Process the trading day
                self._process_trading_day(
                    date,
                    daily_underlying['close'],
                    vix_analysis,
                    should_trade
                )
                
                # Record daily values
                self.daily_values.append({
                    'date': date,
                    'balance': self.current_balance,
                    'return': (self.current_balance / self.initial_balance) - 1.0
                })
                
                # Update peak balance
                if self.current_balance > self.peak_balance:
                    self.peak_balance = self.current_balance

            # Liquidate anything still open on the final day so its entry
            # credit doesn't count as pure profit with no exit cost.
            if self.positions and common_dates:
                final_date = common_dates[-1]
                self._close_positions(
                    final_date,
                    underlying_data.loc[final_date, 'close'],
                    vix_analysis,
                    force_all=True
                )

            # Calculate performance metrics
            results = self._calculate_performance_metrics()
            
            self.logger.info(f"Backtest completed. Final balance: ${self.current_balance:.2f}, "
                           f"Return: {((self.current_balance / self.initial_balance) - 1.0) * 100:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during backtest: {e}")
            return {'error': str(e)}
    
    def _process_trading_day(self, 
                           date: datetime,
                           current_price: float,
                           vix_analysis: Dict,
                           should_trade: bool) -> None:
        """
        Process a single trading day in the backtest.
        
        Args:
            date: Trading date
            current_price: Current price of the underlying
            vix_analysis: VIX analysis results
            should_trade: Whether we should trade today
        """
        # Close any existing positions (end of previous day)
        self._close_positions(date, current_price, vix_analysis)
        
        # If we shouldn't trade today, we're done
        if not should_trade:
            return
        
        # Simulate option chain data
        option_chain = self._simulate_option_chain(date, current_price, vix_analysis)
        
        # Generate a trade decision
        trade_decision = self.options_strategy.generate_trade_decision(
            vix_analysis,
            option_chain,
            current_price,
            self.current_balance
        )
        
        # Execute the trade for any actionable signal
        action = trade_decision.get('action', 'none')
        if action in ('buy', 'iron_condor', 'vertical_spread'):
            # Simulate trade execution
            self._execute_trade(date, trade_decision, current_price)
    
    def _close_positions(self, date: datetime, current_price: float,
                         vix_analysis: Optional[Dict] = None,
                         force_all: bool = False) -> None:
        """
        Close positions that have reached their holding period or expiry.

        Args:
            date: Trading date
            current_price: Current price of the underlying
            vix_analysis: VIX analysis for the closing day (prices exit legs)
            force_all: Close everything regardless of age (end of backtest)
        """
        if not self.positions:
            return

        base_iv = (vix_analysis or {}).get('current_vix', 20.0) / 100.0

        for symbol, position in list(self.positions.items()):
            if not force_all:
                age = self._current_day_index - position.get('entry_day_index', self._current_day_index - self.holding_days)
                expired = self._remaining_days_to_expiry(position, date) <= 0
                if age < self.holding_days and not expired:
                    continue

            strategy = position.get('strategy', 'single_option')

            if strategy == 'iron_condor':
                pnl = self._close_iron_condor(position, current_price, base_iv, date)
            elif strategy == 'vertical_spread':
                pnl = self._close_vertical_spread(position, current_price, base_iv, date)
            else:
                pnl = self._close_single_option(position, current_price)

            self.current_balance += pnl

            trade_record = {
                'date': date,
                'symbol': symbol,
                'action': 'close',
                'quantity': position['quantity'],
                'pnl': pnl,
                'balance': self.current_balance
            }
            self.trade_history.append(trade_record)
            self.logger.debug(f"Closed position {symbol} with P&L: ${pnl:.2f}")
            del self.positions[symbol]

    def _remaining_days_to_expiry(self, position: Dict, close_date: datetime) -> int:
        """Days from close_date to the position's expiry (0 when at/past expiry)."""
        expiry_str = str(position.get('expiry', ''))
        expiry_dt = None
        for fmt in ('%Y%m%d', '%Y-%m-%d'):
            try:
                expiry_dt = datetime.strptime(expiry_str, fmt)
                break
            except ValueError:
                continue
        if expiry_dt is None:
            # Unknown expiry format: assume the shortest simulated tenor
            return 7
        return max(0, (expiry_dt - close_date).days)

    def _simulated_leg_price(self,
                             underlying: float,
                             strike: float,
                             base_iv: float,
                             days_to_expiry: int,
                             option_type: str) -> float:
        """
        Model price of a single option leg. Must mirror the pricing in
        _simulate_option_chain so entry and exit are marked with the same
        model; at expiry it collapses to intrinsic value.
        """
        if days_to_expiry <= 0:
            if option_type == 'call':
                return max(0.0, underlying - strike)
            return max(0.0, strike - underlying)

        t = days_to_expiry / 365.0
        if days_to_expiry <= 7:
            expiry_iv_factor = 1.05
        elif days_to_expiry <= 30:
            expiry_iv_factor = 1.0
        else:
            expiry_iv_factor = 0.95

        abs_moneyness = abs((strike / underlying) - 1.0)
        smile_factor = 1.0 + abs_moneyness * 0.3
        adjusted_iv = max(0.01, base_iv) * expiry_iv_factor * smile_factor

        vol_t = adjusted_iv * math.sqrt(t)
        d1 = math.log(underlying / strike) / vol_t + 0.5 * vol_t
        d2 = d1 - vol_t

        call_price = max(0.01, underlying * _norm_cdf(d1) - strike * _norm_cdf(d2))
        if option_type == 'call':
            return call_price
        return max(0.01, call_price - underlying + strike)  # Put-call parity

    def _close_iron_condor(self, position: Dict, current_price: float,
                           base_iv: float = 0.20, close_date: Optional[datetime] = None) -> float:
        """
        P&L for buying back an iron condor: credit received minus the model
        cost to close all four legs with the remaining time value. Settling at
        intrinsic value before expiry would award weeks of unearned theta and
        produce impossible returns.
        """
        net_credit = position['net_credit']
        quantity = position['quantity']

        dte = self._remaining_days_to_expiry(position, close_date) if close_date else 0
        short_call_px = self._simulated_leg_price(current_price, position['short_call_strike'], base_iv, dte, 'call')
        short_put_px = self._simulated_leg_price(current_price, position['short_put_strike'], base_iv, dte, 'put')
        long_call_px = self._simulated_leg_price(current_price, position['long_call_strike'], base_iv, dte, 'call')
        long_put_px = self._simulated_leg_price(current_price, position['long_put_strike'], base_iv, dte, 'put')

        cost_to_close = (short_call_px + short_put_px) - (long_call_px + long_put_px)
        pnl_per_spread = net_credit - cost_to_close

        return pnl_per_spread * quantity * 100

    def _close_vertical_spread(self, position: Dict, current_price: float,
                               base_iv: float = 0.20, close_date: Optional[datetime] = None) -> float:
        """
        P&L for buying back a vertical credit spread: credit received minus
        the model cost to close both legs with the remaining time value.
        """
        net_credit = position['net_credit']
        quantity = position['quantity']
        spread_type = position.get('spread_type', 'bull_put')
        option_type = 'put' if spread_type == 'bull_put' else 'call'

        dte = self._remaining_days_to_expiry(position, close_date) if close_date else 0
        short_px = self._simulated_leg_price(current_price, position['short_strike'], base_iv, dte, option_type)
        long_px = self._simulated_leg_price(current_price, position['long_strike'], base_iv, dte, option_type)

        cost_to_close = short_px - long_px
        pnl_per_spread = net_credit - cost_to_close

        return pnl_per_spread * quantity * 100

    def _close_single_option(self, position: Dict, current_price: float) -> float:
        """Calculate P&L for closing a single option position."""
        underlying_change = (current_price / position['underlying_price']) - 1.0

        if position.get('option_type', 'call') == 'call':
            price_multiplier = 2.5 if underlying_change > 0 else 1.5
        else:
            price_multiplier = 2.5 if underlying_change < 0 else 1.5
            underlying_change = -underlying_change

        option_change = underlying_change * price_multiplier
        exit_price = max(0.01, position['entry_price'] * (1 + option_change))
        return (exit_price - position['entry_price']) * position['quantity'] * 100

    def _execute_trade(self, date: datetime, trade_decision: Dict, current_price: float) -> None:
        """
        Execute a trade in the backtest.

        Args:
            date: Trading date
            trade_decision: Trade decision details
            current_price: Current price of the underlying
        """
        action = trade_decision.get('action', 'buy')

        if action == 'iron_condor':
            self._execute_iron_condor(date, trade_decision, current_price)
        elif action == 'vertical_spread':
            self._execute_vertical_spread(date, trade_decision, current_price)
        else:
            self._execute_single_option(date, trade_decision, current_price)

    def _execute_iron_condor(self, date: datetime, trade_decision: Dict, current_price: float) -> None:
        """Execute an iron condor trade in the backtest."""
        symbol = trade_decision.get('symbol', self.options_strategy.index_symbol)
        expiry = trade_decision.get('expiry', '')
        quantity = trade_decision.get('quantity', 1)
        net_credit = trade_decision.get('net_credit', 0.0)
        short_call = trade_decision.get('short_call_strike', 0)
        short_put = trade_decision.get('short_put_strike', 0)
        long_call = trade_decision.get('long_call_strike', 0)
        long_put = trade_decision.get('long_put_strike', 0)
        max_risk = trade_decision.get('max_risk_per_spread', trade_decision.get('max_risk', 0))

        # Margin required is the max risk per spread (already in per-contract dollars)
        margin_required = max_risk * quantity if max_risk > 0 else net_credit * 3 * quantity * 100

        if margin_required > self.current_balance * 0.5:
            # Reduce quantity to fit within 50% of balance
            per_contract_margin = max_risk if max_risk > 0 else net_credit * 3 * 100
            quantity = max(1, int(self.current_balance * 0.5 / per_contract_margin))
            margin_required = max_risk * quantity if max_risk > 0 else net_credit * 3 * quantity * 100

        if quantity <= 0 or margin_required > self.current_balance:
            return

        position_id = f"{symbol}_IC_{expiry}_{short_put}P_{short_call}C_{date.strftime('%Y%m%d')}"
        max_width = min(short_call - short_put, long_call - short_call) if short_call > short_put else 5.0

        position = {
            'strategy': 'iron_condor',
            'symbol': symbol,
            'expiry': expiry,
            'quantity': quantity,
            'net_credit': net_credit,
            'short_call_strike': short_call,
            'short_put_strike': short_put,
            'long_call_strike': long_call,
            'long_put_strike': long_put,
            'max_width': max_width,
            'entry_date': date,
            'entry_day_index': self._current_day_index,
            'underlying_price': current_price
        }

        self.positions[position_id] = position

        # P&L (credit minus cost to close) is booked when the position is
        # closed; adding the credit here as well would double-count it.
        credit_received = net_credit * quantity * 100

        trade_record = {
            'date': date,
            'symbol': position_id,
            'action': 'iron_condor',
            'quantity': quantity,
            'price': net_credit,
            'cost': -credit_received,
            'balance': self.current_balance
        }
        self.trade_history.append(trade_record)

        self.logger.debug(f"Executed Iron Condor: {quantity}x {position_id} credit: ${net_credit:.2f}")

    def _execute_vertical_spread(self, date: datetime, trade_decision: Dict, current_price: float) -> None:
        """Execute a vertical spread trade in the backtest."""
        symbol = trade_decision.get('symbol', self.options_strategy.index_symbol)
        spread_type = trade_decision.get('spread_type', 'bull_put')
        expiry = trade_decision.get('expiry', '')
        quantity = trade_decision.get('quantity', 1)
        net_credit = trade_decision.get('net_credit', 0.0)
        short_strike = trade_decision.get('short_strike', 0)
        long_strike = trade_decision.get('long_strike', 0)
        max_risk = trade_decision.get('max_risk_per_spread', trade_decision.get('max_risk', 0))

        # Margin required is the max risk per spread (already in per-contract dollars)
        margin_required = max_risk * quantity if max_risk > 0 else net_credit * 3 * quantity * 100

        if margin_required > self.current_balance * 0.5:
            per_contract_margin = max_risk if max_risk > 0 else net_credit * 3 * 100
            quantity = max(1, int(self.current_balance * 0.5 / per_contract_margin))
            margin_required = max_risk * quantity if max_risk > 0 else net_credit * 3 * quantity * 100

        if quantity <= 0 or margin_required > self.current_balance:
            return

        position_id = f"{symbol}_{spread_type}_{expiry}_{short_strike}_{long_strike}_{date.strftime('%Y%m%d')}"

        position = {
            'strategy': 'vertical_spread',
            'spread_type': spread_type,
            'symbol': symbol,
            'expiry': expiry,
            'quantity': quantity,
            'net_credit': net_credit,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'entry_date': date,
            'entry_day_index': self._current_day_index,
            'underlying_price': current_price
        }

        self.positions[position_id] = position

        # P&L (credit minus cost to close) is booked when the position is
        # closed; adding the credit here as well would double-count it.
        credit_received = net_credit * quantity * 100

        trade_record = {
            'date': date,
            'symbol': position_id,
            'action': 'vertical_spread',
            'quantity': quantity,
            'price': net_credit,
            'cost': -credit_received,
            'balance': self.current_balance
        }
        self.trade_history.append(trade_record)

        self.logger.debug(f"Executed {spread_type}: {quantity}x {position_id} credit: ${net_credit:.2f}")

    def _execute_single_option(self, date: datetime, trade_decision: Dict, current_price: float) -> None:
        """Execute a single option trade in the backtest."""
        symbol = trade_decision.get('symbol')
        expiry = trade_decision.get('expiry')
        strike = trade_decision.get('strike')
        option_type = trade_decision.get('option_type', 'call')
        quantity = trade_decision.get('quantity', 1)
        price = trade_decision.get('price', 0.0)

        cost = price * quantity * 100
        if cost > self.current_balance:
            quantity = int(self.current_balance / (price * 100))
            cost = price * quantity * 100
            if quantity <= 0:
                return

        # P&L (exit minus entry) is booked when the position is closed;
        # subtracting the cost here as well would double-charge the entry.
        position_id = f"{symbol}_{option_type}_{expiry}_{strike}_{date.strftime('%Y%m%d')}"

        position = {
            'strategy': 'single_option',
            'symbol': symbol,
            'expiry': expiry,
            'strike': strike,
            'option_type': option_type,
            'quantity': quantity,
            'entry_price': price,
            'entry_date': date,
            'entry_day_index': self._current_day_index,
            'underlying_price': current_price
        }
        self.positions[position_id] = position

        trade_record = {
            'date': date,
            'symbol': position_id,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'balance': self.current_balance
        }
        self.trade_history.append(trade_record)

        self.logger.debug(f"Executed trade: BUY {quantity} {symbol} {option_type} @ ${strike} "
                        f"for ${price:.2f} per contract, total cost: ${cost:.2f}")
    
    def _simulate_option_chain(self, date: datetime, current_price: float, vix_analysis: Dict) -> Dict:
        """
        Simulate an option chain based on market conditions.
        
        Args:
            date: Trading date
            current_price: Current price of the underlying
            vix_analysis: VIX analysis results
            
        Returns:
            Dict with simulated option chain data
        """
        # Get the VIX value (implied volatility)
        vix_value = vix_analysis.get('current_vix', 20.0)
        
        # Calculate base IV (convert VIX to decimal)
        base_iv = vix_value / 100
        
        # Generate expiration dates: short-dated dailies (SPY lists daily
        # expirations) plus weekly options out to 8 weeks
        expiry_dates = []
        for d in (1, 2, 3):
            expiry = date + timedelta(days=d)
            expiry_dates.append((expiry.strftime('%Y%m%d'), d))
        for i in range(1, 9):  # Next 8 weeks
            expiry = date + timedelta(days=i*7)
            expiry_str = expiry.strftime('%Y%m%d')
            expiry_dates.append((expiry_str, i*7))
        
        # Generate strikes (around current price)
        strike_range = 0.20  # 20% up and down
        min_strike = current_price * (1 - strike_range)
        max_strike = current_price * (1 + strike_range)
        
        strikes = []
        strike = min_strike
        while strike <= max_strike:
            strikes.append(round(strike, 1))
            strike += current_price * 0.01  # 1% steps
        
        # Build the option chain
        option_chain = {}
        
        for expiry_str, days_to_expiry in expiry_dates:
            # Calculate time to expiry in years
            t = days_to_expiry / 365.0
            
            # Adjust IV based on time to expiry (term structure)
            if days_to_expiry <= 7:  # Weekly options
                expiry_iv_factor = 1.05
            elif days_to_expiry <= 30:  # Monthly options
                expiry_iv_factor = 1.0
            else:  # Longer dated
                expiry_iv_factor = 0.95
            
            calls = {}
            puts = {}
            
            for strike in strikes:
                # Black-Scholes-style option pricing approximation
                abs_moneyness = abs((strike / current_price) - 1.0)

                # Volatility smile (mild skew)
                smile_factor = 1.0 + abs_moneyness * 0.3
                adjusted_iv = base_iv * expiry_iv_factor * smile_factor

                vol_t = adjusted_iv * math.sqrt(t) if t > 0 else 0.01

                # d1 approximation for Black-Scholes
                d1 = math.log(current_price / strike) / vol_t + 0.5 * vol_t

                norm_cdf = _norm_cdf

                d2 = d1 - vol_t

                nd1 = norm_cdf(d1)
                nd2 = norm_cdf(d2)

                # Option prices (no risk-free rate for simplicity)
                call_price = max(0.01, round(current_price * nd1 - strike * nd2, 2))
                put_price = max(0.01, round(call_price - current_price + strike, 2))  # Put-call parity

                # Bid-ask spread (wider for far OTM), calibrated to SPY's
                # highly liquid market: ~1% half-spread near the money,
                # ~2% a few percent out
                spread_half = 0.01 + abs_moneyness * 0.15
                call_bid = max(0.01, round(call_price * (1 - spread_half), 2))
                call_ask = max(0.02, round(call_price * (1 + spread_half), 2))
                put_bid = max(0.01, round(put_price * (1 - spread_half), 2))
                put_ask = max(0.02, round(put_price * (1 + spread_half), 2))

                # Delta
                call_delta = max(0.01, min(0.99, nd1))
                put_delta = max(0.01, min(0.99, 1.0 - nd1))  # Put delta magnitude = 1 - N(d1)
                
                calls[strike] = {
                    'bid': call_bid,
                    'ask': call_ask,
                    'last': call_price,
                    'iv': adjusted_iv,
                    'delta': call_delta,
                    'volume': int(1000 * max(0, 1 - abs_moneyness) ** 2),  # Higher volume for ATM
                    'open_interest': int(5000 * max(0, 1 - abs_moneyness) ** 2)
                }
                
                puts[strike] = {
                    'bid': put_bid,
                    'ask': put_ask,
                    'last': put_price,
                    'iv': adjusted_iv,
                    'delta': -put_delta,  # Put delta is negative
                    'volume': int(1000 * max(0, 1 - abs_moneyness) ** 2),
                    'open_interest': int(5000 * max(0, 1 - abs_moneyness) ** 2)
                }
            
            option_chain[expiry_str] = {
                'days_to_expiry': days_to_expiry,
                'calls': calls,
                'puts': puts,
                'strikes': strikes
            }
        
        return option_chain
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics from the backtest results.
        
        Returns:
            Dict with performance metrics
        """
        if not self.daily_values:
            return {'error': 'No data available for performance calculation'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.daily_values)
        
        # Calculate daily returns
        df['daily_return'] = df['balance'].pct_change()
        
        # Calculate metrics
        total_days = len(df)
        trading_days = len([t for t in self.trade_history if t['action'] in ('buy', 'iron_condor', 'vertical_spread')])
        total_return = ((self.current_balance / self.initial_balance) - 1.0) * 100
        
        # Annualized return (assuming 252 trading days per year)
        years = total_days / 252
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Volatility (annualized standard deviation of returns)
        daily_volatility = df['daily_return'].std()
        annualized_volatility = daily_volatility * np.sqrt(252) * 100 if not np.isnan(daily_volatility) else 0
        
        # Maximum drawdown
        df['peak'] = df['balance'].cummax()
        df['drawdown'] = (df['peak'] - df['balance']) / df['peak']
        max_drawdown = df['drawdown'].max() * 100
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 0 else 0
        
        # Win rate
        trades_df = pd.DataFrame([t for t in self.trade_history if t['action'] in ('sell', 'close')])
        if len(trades_df) > 0:
            trades_df['win'] = trades_df['pnl'] > 0
            win_rate = trades_df['win'].mean() * 100
            avg_win = trades_df.loc[trades_df['win'], 'pnl'].mean() if any(trades_df['win']) else 0
            avg_loss = trades_df.loc[~trades_df['win'], 'pnl'].mean() if any(~trades_df['win']) else 0
            profit_factor = abs(trades_df.loc[trades_df['win'], 'pnl'].sum() / 
                              trades_df.loc[~trades_df['win'], 'pnl'].sum()) if any(~trades_df['win']) and trades_df.loc[~trades_df['win'], 'pnl'].sum() != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calculate monthly returns
        df['month'] = df['date'].dt.to_period('M')
        monthly_returns = df.groupby('month')['balance'].last().pct_change() * 100
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'annualized_volatility_pct': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_days': total_days,
            'trading_days': trading_days,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'monthly_returns': monthly_returns.to_dict(),
            'trades': len(trades_df),
            'daily_values': self.daily_values
        }


# Module import guard
if __name__ == "__main__":
    import math
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = BacktestEngine(initial_balance=100000.0)
    
    # Run a short backtest
    results = engine.run_backtest(
        start_date='2022-01-01',
        end_date='2022-06-30',
        risk_level=5
    )
    
    # Print results
    print("\nBacktest Results:")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")
    print(f"Annualized Volatility: {results['annualized_volatility_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")