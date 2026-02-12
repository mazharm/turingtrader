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


class BacktestEngine:
    """Engine for backtesting the TuringTrader algorithm."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        volatility_analyzer: Optional[VolatilityAnalyzer] = None,
        risk_manager: Optional[RiskManager] = None,
        options_strategy: Optional[OptionsStrategy] = None,
        initial_balance: float = 100000.0,
        data_fetcher: Optional[Any] = None  # Added data_fetcher argument
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
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_returns = []
        self.daily_values = []
        self.positions = {}
        self.trades = []
        self.trade_history = []
    
    def _reset_backtest_state(self) -> None:
        """Reset the backtest state."""
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.daily_returns = []
        self.daily_values = []
        self.positions = {}
        self.trades = []
        self.trade_history = []
        
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
            for date in common_dates:
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
        self._close_positions(date, current_price)
        
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
    
    def _close_positions(self, date: datetime, current_price: float) -> None:
        """
        Close all positions at the end of day.

        Args:
            date: Trading date
            current_price: Current price of the underlying
        """
        if not self.positions:
            return

        self.logger.debug(f"Closing {len(self.positions)} positions on {date.strftime('%Y-%m-%d')}")

        for symbol, position in list(self.positions.items()):
            strategy = position.get('strategy', 'single_option')

            if strategy == 'iron_condor':
                pnl = self._close_iron_condor(position, current_price)
            elif strategy == 'vertical_spread':
                pnl = self._close_vertical_spread(position, current_price)
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

        self.positions = {}

    def _close_iron_condor(self, position: Dict, current_price: float) -> float:
        """Calculate P&L for closing an iron condor position."""
        short_put = position['short_put_strike']
        short_call = position['short_call_strike']
        long_put = position['long_put_strike']
        long_call = position['long_call_strike']
        net_credit = position['net_credit']
        quantity = position['quantity']
        max_width = position.get('max_width', short_call - short_put)

        # Determine payout based on where price ended
        if short_put <= current_price <= short_call:
            # Price within short strikes - max profit (keep full credit)
            pnl_per_spread = net_credit
        elif current_price < long_put:
            # Price below long put - max loss on put side
            put_width = short_put - long_put
            pnl_per_spread = net_credit - put_width
        elif current_price > long_call:
            # Price above long call - max loss on call side
            call_width = long_call - short_call
            pnl_per_spread = net_credit - call_width
        elif current_price < short_put:
            # Price between long put and short put - partial loss on put side
            intrusion = short_put - current_price
            pnl_per_spread = net_credit - intrusion
        else:
            # Price between short call and long call - partial loss on call side
            intrusion = current_price - short_call
            pnl_per_spread = net_credit - intrusion

        return pnl_per_spread * quantity * 100

    def _close_vertical_spread(self, position: Dict, current_price: float) -> float:
        """Calculate P&L for closing a vertical spread position."""
        short_strike = position['short_strike']
        long_strike = position['long_strike']
        net_credit = position['net_credit']
        quantity = position['quantity']
        spread_type = position.get('spread_type', 'bull_put')

        if spread_type == 'bull_put':
            # Bull put spread: profit if price stays above short put
            if current_price >= short_strike:
                pnl_per_spread = net_credit  # Keep full credit
            elif current_price <= long_strike:
                width = short_strike - long_strike
                pnl_per_spread = net_credit - width  # Max loss
            else:
                intrusion = short_strike - current_price
                pnl_per_spread = net_credit - intrusion
        elif spread_type == 'bear_call':
            # Bear call spread: profit if price stays below short call
            if current_price <= short_strike:
                pnl_per_spread = net_credit  # Keep full credit
            elif current_price >= long_strike:
                width = long_strike - short_strike
                pnl_per_spread = net_credit - width  # Max loss
            else:
                intrusion = current_price - short_strike
                pnl_per_spread = net_credit - intrusion
        else:
            pnl_per_spread = net_credit * 0.5  # Fallback

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

        position_id = f"{symbol}_IC_{expiry}_{short_put}P_{short_call}C"
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
            'underlying_price': current_price
        }

        self.positions[position_id] = position

        # Credit received (added to balance)
        credit_received = net_credit * quantity * 100
        self.current_balance += credit_received

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

        position_id = f"{symbol}_{spread_type}_{expiry}_{short_strike}_{long_strike}"

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
            'underlying_price': current_price
        }

        self.positions[position_id] = position

        # Credit received
        credit_received = net_credit * quantity * 100
        self.current_balance += credit_received

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

        self.current_balance -= cost
        position_id = f"{symbol}_{option_type}_{expiry}_{strike}"

        position = {
            'strategy': 'single_option',
            'symbol': symbol,
            'expiry': expiry,
            'strike': strike,
            'option_type': option_type,
            'quantity': quantity,
            'entry_price': price,
            'entry_date': date,
            'cost': cost,
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
        
        # Generate expiration dates (weekly options)
        expiry_dates = []
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

                # Standard normal CDF approximation (Abramowitz & Stegun)
                def norm_cdf(x):
                    a = 0.4361836
                    b = -0.1201676
                    c = 0.9372980
                    k = 1.0 / (1.0 + 0.33267 * abs(x))
                    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * (a * k + b * k * k + c * k * k * k)
                    return cdf if x >= 0 else 1.0 - cdf

                d2 = d1 - vol_t

                nd1 = norm_cdf(d1)
                nd2 = norm_cdf(d2)

                # Option prices (no risk-free rate for simplicity)
                call_price = max(0.01, round(current_price * nd1 - strike * nd2, 2))
                put_price = max(0.01, round(call_price - current_price + strike, 2))  # Put-call parity

                # Bid-ask spread (wider for far OTM)
                spread_half = 0.04 + abs_moneyness * 0.15
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