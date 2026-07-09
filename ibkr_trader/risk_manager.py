"""
Risk management module for the TuringTrader algorithm.
Controls position sizing, risk limits, and monitors overall portfolio risk.
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

from .config import Config, RiskParameters

# Position types that represent short premium (credit) spreads: entry price is a
# credit received, and the position profits as the cost to close falls.
CREDIT_SPREAD_TYPES = ('iron_condor', 'vertical_spread', 'spread')


class RiskManager:
    """
    Risk manager for the trading algorithm.
    Handles position sizing, maximum risk, and other risk controls.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration object
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration
        self.config = config or Config()
        self.risk_params = self.config.risk
        
        # Internal state
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_drawdown = 0.0
        self.starting_balance = 0.0
        self.current_positions = {}
        self._position_history = []

        # Trading halt state: once halted (daily loss limit, kill switch, or
        # repeated errors), no new positions may be opened until reset.
        self.trading_halted = False
        self.halt_reason = ""

        # Initialize risk limits
        self._initialize_risk_limits()
    
    def _initialize_risk_limits(self) -> None:
        """Initialize risk limits from configuration."""
        self.max_daily_risk_amount = 0.0  # Will be set when account value is known
        self.max_position_size = 0.0      # Will be set when account value is known
        self.max_delta_exposure = self.risk_params.max_delta_exposure
        self.stop_loss_pct = self.risk_params.stop_loss_pct
        self.target_profit_pct = self.risk_params.target_profit_pct
    
    def update_account_value(self, account_value: float) -> None:
        """
        Update account value and risk limits.
        
        Args:
            account_value: Current account value (USD)
        """
        if account_value <= 0:
            self.logger.error(f"Invalid account value: {account_value}")
            return
            
        # Set starting balance if not set
        if self.starting_balance == 0.0:
            self.starting_balance = account_value
            
        # Update risk limits based on account value
        self.max_daily_risk_amount = account_value * (self.risk_params.max_daily_risk_pct / 100.0)
        self.max_position_size = account_value * (self.risk_params.max_position_size_pct / 100.0)
        
        # Update maximum drawdown
        current_drawdown = (1.0 - (account_value / self.starting_balance)) * 100.0
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        self.logger.info(f"Updated account value: ${account_value:.2f}, "
                       f"max position: ${self.max_position_size:.2f}, "
                       f"max daily risk: ${self.max_daily_risk_amount:.2f}")
    
    def update_risk_level(self, level: int) -> None:
        """
        Update the risk level.
        
        Args:
            level: Risk level (1-10)
        """
        self.risk_params.adjust_for_risk_level(level)
        self._initialize_risk_limits()
        self.logger.info(f"Updated risk level to {level}")
    
    def calculate_position_size(self, 
                              price: float, 
                              volatility: float, 
                              account_value: float,
                              vol_multiplier: float = 1.0) -> int:
        """
        Calculate position size based on price, volatility, and account value.
        
        Args:
            price: Price of the instrument
            volatility: Current volatility (percentage)
            account_value: Current account value (USD)
            vol_multiplier: Multiplier based on volatility analysis
            
        Returns:
            int: Number of contracts/shares to trade
        """
        if price <= 0 or account_value <= 0:
            return 0
            
        # Update account value and limits if needed
        if self.max_daily_risk_amount == 0.0:
            self.update_account_value(account_value)
            
        # Apply additional safety factor that reduces all position sizes
        global_safety_factor = 0.65  # Reduce all positions by 35%, further reduced from 0.75
        
        # Base position size as percentage of max position size - reduced multiplier for safety
        base_size = self.max_position_size * (vol_multiplier * 0.75) * global_safety_factor
        
        # More conservative volatility-based scaling with finer gradations
        if volatility > 60:
            vol_factor = 0.08  # Extreme volatility crisis - even tinier positions, reduced from 0.1
        elif volatility > 50:
            vol_factor = 0.12  # Extremely high volatility - drastically reduced, from 0.15
        elif volatility > 40:
            vol_factor = 0.20  # Very high volatility - significantly reduced, from 0.25
        elif volatility > 35:
            vol_factor = 0.30  # High volatility - substantially reduced, from 0.35
        elif volatility > 30:
            vol_factor = 0.40  # Above average volatility - moderately reduced, from 0.45
        elif volatility > 25:
            vol_factor = 0.50  # Slightly elevated volatility - somewhat reduced, from 0.6
        elif volatility > 20:
            vol_factor = 0.60  # Normal volatility - mild reduction, from 0.7
        elif volatility > 15:
            vol_factor = 0.65  # Optimal volatility range - slight reduction, from 0.8
        elif volatility > 12:
            vol_factor = 0.55  # Lower volatility - reduced as premium may be inadequate, from 0.7
        else:
            vol_factor = 0.30  # Very low volatility - substantially reduced due to minimal premium, from 0.4
            
        # Calculate dollar amount with more conservative position cap
        position_value = min(base_size * vol_factor, account_value * 0.05)  # Further reduced from 0.06
        
        # Convert to quantity
        quantity = int(position_value / price)
        
        # Apply an absolute maximum based on price
        if price > 0:
            # Limit by absolute dollar amount
            max_dollar_amount = 5000.0  # Maximum $5,000 per position
            max_quantity_by_dollars = int(max_dollar_amount / price)
            quantity = min(quantity, max_quantity_by_dollars)
        
        self.logger.info(f"Calculated position size: {quantity} units "
                       f"(${position_value:.2f}, {vol_multiplier:.2f} vol mult, "
                       f"{vol_factor:.2f} vol factor, {global_safety_factor:.2f} safety)")
                       
        return max(0, quantity)  # Allow zero quantity for extreme cases
    
    def calculate_option_quantity(self, 
                                option_price: float,
                                delta: float,
                                account_value: float,
                                vol_multiplier: float = 1.0) -> int:
        """
        Calculate option quantity based on price, delta, and account value.
        
        Args:
            option_price: Price of the option contract
            delta: Option delta (absolute value)
            account_value: Current account value (USD)
            vol_multiplier: Multiplier based on volatility analysis
            
        Returns:
            int: Number of option contracts to trade
        """
        if option_price <= 0 or account_value <= 0:
            return 0
            
        # Ensure delta is positive for calculation
        abs_delta = abs(delta)
        
        # Update account value and limits if needed
        if self.max_daily_risk_amount == 0.0:
            self.update_account_value(account_value)
            
        # Apply global safety factor for additional risk reduction
        global_safety_factor = 0.55  # Reduce all option positions by 45%, further reduced from 0.65
        
        # Base position value as percentage of max position size - further reduced for safety
        base_size = (self.max_position_size * vol_multiplier) * 0.5 * global_safety_factor  # Further reduced from 0.6
        
        # Much more conservative delta-based adjustment with finer gradations
        if abs_delta > 0.8:  # Deep ITM options
            delta_factor = min(0.1, self.max_delta_exposure / (100 * abs_delta))  # Extremely small
        elif abs_delta > 0.6:  # Moderately ITM options
            delta_factor = min(0.15, self.max_delta_exposure / (100 * abs_delta))  # Very small
        elif abs_delta > 0.4:  # ATM options
            delta_factor = min(0.25, self.max_delta_exposure / (100 * abs_delta))  # Small
        elif abs_delta > 0.3:  # Slightly OTM options
            delta_factor = min(0.35, self.max_delta_exposure / (100 * abs_delta))  # Moderate
        elif abs_delta > 0.2:  # OTM options - our sweet spot for most trades
            delta_factor = min(0.45, self.max_delta_exposure / (100 * abs_delta))  # Optimal range
        elif abs_delta > 0.1:  # Further OTM options
            delta_factor = min(0.35, self.max_delta_exposure / (100 * abs_delta))  # Moderate
        elif abs_delta > 0.05:  # Very OTM options
            delta_factor = min(0.25, self.max_delta_exposure / (100 * abs_delta))  # Small again
        else:  # Extremely OTM options
            delta_factor = 0.1  # Very small due to high gamma risk and low probability
        
        # Calculate dollar amount with maximum position cap - significantly reduced cap
        position_value = min(base_size * delta_factor, account_value * 0.03)  # Further reduced from 0.04
        
        # Options have multiplier (usually 100)
        contract_value = option_price * 100
        
        # Calculate number of contracts with better risk control
        if contract_value > 0:
            # Limit the number of contracts based on absolute risk - more conservative
            # Use a lower percentage of daily risk for each individual position
            daily_risk_allocation = self.risk_params.max_daily_risk_pct / 100
            position_risk_allocation = daily_risk_allocation * 0.5  # Only use 50% of daily risk on a single position
            max_contracts_by_risk = int((account_value * position_risk_allocation) / contract_value)
            
            # Limit based on absolute value
            max_dollar_exposure = 1500.0  # Maximum $1,500 per option position, reduced from $2,000
            max_contracts_by_dollars = int(max_dollar_exposure / contract_value)
            
            # Take the minimum of all constraints
            quantity = min(
                int(position_value / contract_value),
                max_contracts_by_risk,
                max_contracts_by_dollars,
                8  # Hard cap on number of contracts, reduced from 10
            )
        else:
            quantity = 0
            
        self.logger.info(f"Calculated option quantity: {quantity} contracts "
                       f"(${position_value:.2f}, delta: {abs_delta:.2f}, "
                       f"contract value: ${contract_value:.2f}, safety: {global_safety_factor:.2f})")
        
        # For expensive options, we no longer force a minimum of 1 contract
        # This allows the algorithm to skip trades when they're too expensive
                
        return quantity  # Allow zero quantity for better risk management
    
    def update_daily_pnl(self, trade_pnl: float) -> bool:
        """
        Update daily P&L and check if daily risk limit is exceeded.
        
        Args:
            trade_pnl: P&L from the latest trade (positive or negative)
            
        Returns:
            bool: True if still within daily risk limit, False otherwise
        """
        self.daily_pnl += trade_pnl
        self.daily_trades += 1

        # Check if we've exceeded max daily loss; halt new trades if so.
        if self.max_daily_risk_amount > 0 and self.daily_pnl < -self.max_daily_risk_amount:
            self.logger.warning(f"Daily risk limit exceeded: ${self.daily_pnl:.2f} loss "
                             f"exceeds ${self.max_daily_risk_amount:.2f} limit")
            self.halt_trading(
                f"daily loss ${-self.daily_pnl:.2f} exceeds limit ${self.max_daily_risk_amount:.2f}"
            )
            return False

        return True

    def halt_trading(self, reason: str) -> None:
        """Halt opening of new positions until reset_daily_metrics() or resume_trading()."""
        if not self.trading_halted:
            self.logger.warning(f"TRADING HALTED: {reason}")
        self.trading_halted = True
        self.halt_reason = reason

    def resume_trading(self) -> None:
        """Clear a trading halt."""
        if self.trading_halted:
            self.logger.info(f"Trading resumed (was halted: {self.halt_reason})")
        self.trading_halted = False
        self.halt_reason = ""

    def can_open_new_position(self, max_daily_trades: int = 0,
                              proposed_risk: float = 0.0) -> Tuple[bool, str]:
        """
        Pre-trade gate: check every risk limit that should block a new position.

        Args:
            max_daily_trades: Maximum number of new positions per day (0 = no cap)
            proposed_risk: Max loss (USD) of the proposed trade

        Returns:
            (allowed, reason) - reason is empty when allowed
        """
        if self.trading_halted:
            return False, f"trading halted: {self.halt_reason}"

        if self.max_daily_risk_amount > 0 and self.daily_pnl < -self.max_daily_risk_amount:
            return False, (f"daily loss ${-self.daily_pnl:.2f} exceeds "
                           f"limit ${self.max_daily_risk_amount:.2f}")

        if max_daily_trades > 0 and self.daily_trades >= max_daily_trades:
            return False, f"daily trade cap reached ({self.daily_trades}/{max_daily_trades})"

        if proposed_risk > 0 and self.max_daily_risk_amount > 0:
            # Remaining risk budget = limit - realized losses - risk already open
            realized_loss = max(0.0, -self.daily_pnl)
            open_risk = self.get_total_open_risk()
            remaining = self.max_daily_risk_amount - realized_loss - open_risk
            if proposed_risk > remaining:
                return False, (f"proposed risk ${proposed_risk:.2f} exceeds remaining "
                               f"daily budget ${remaining:.2f} "
                               f"(open risk ${open_risk:.2f}, realized loss ${realized_loss:.2f})")

        return True, ""

    def get_total_open_risk(self) -> float:
        """Sum of max potential loss (USD) across all open positions."""
        total = 0.0
        for position in self.current_positions.values():
            entry_price = position.get('entry_price', 0.0)
            quantity = position.get('quantity', 0)
            if position.get('type') in CREDIT_SPREAD_TYPES:
                risk_points = self._max_risk_points(entry_price, position.get('option_data'))
                total += risk_points * quantity * 100
            elif position.get('type') == 'option':
                # Long option: max loss is the premium paid
                total += entry_price * quantity * 100
            else:
                total += entry_price * quantity
        return total

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics for a new trading day."""
        self.logger.info(f"Resetting daily metrics. Previous P&L: ${self.daily_pnl:.2f}, "
                       f"trades: {self.daily_trades}")

        self.daily_pnl = 0.0
        self.daily_trades = 0
        # A new day clears a daily-loss halt (but an explicit kill switch is
        # re-detected by the trader every cycle, so this stays safe).
        self.resume_trading()
    
    def _max_risk_points(self, entry_credit: float, option_data: Optional[Dict]) -> float:
        """
        Determine max risk of a credit spread in per-share points.

        Prefers strike widths from option_data; falls back to a dollar
        max_risk_per_spread (per contract) divided by the option multiplier.
        """
        option_data = option_data or {}

        # Iron condor: risk = narrower wing width - credit
        try:
            if all(k in option_data for k in ('short_call_strike', 'long_call_strike',
                                              'short_put_strike', 'long_put_strike')):
                call_width = float(option_data['long_call_strike']) - float(option_data['short_call_strike'])
                put_width = float(option_data['short_put_strike']) - float(option_data['long_put_strike'])
                width = min(call_width, put_width)
                if width > 0:
                    return max(0.0, width - entry_credit)

            # Vertical spread: risk = width - credit
            if 'width' in option_data and option_data['width']:
                width = abs(float(option_data['width']))
                if width > 0:
                    return max(0.0, width - entry_credit)
            if 'short_strike' in option_data and 'long_strike' in option_data:
                width = abs(float(option_data['short_strike']) - float(option_data['long_strike']))
                if width > 0:
                    return max(0.0, width - entry_credit)
        except (TypeError, ValueError):
            pass

        # Fallback: dollar risk per contract -> points
        max_risk_dollars = option_data.get('max_risk_per_spread', 0) or 0
        if max_risk_dollars > 0:
            return max_risk_dollars / 100.0

        return 0.0

    def add_position(self, symbol: str, quantity: int, entry_price: float,
                   position_type: str = 'option', option_data: Optional[Dict] = None) -> None:
        """
        Add a new position to tracking.
        
        Args:
            symbol: Symbol or identifier (e.g., SPY_C_20231215_450 or SPY_IC_20231215_430P_440P_460C_470C)
            quantity: Number of shares/contracts/spreads
            entry_price: Entry price per share/contract. For spreads, this is the net credit or debit.
                         Positive for credits (e.g., Iron Condor sold), negative for debits.
            position_type: Type of position ('stock', 'option', 'iron_condor', 'spread', etc.)
            option_data: Additional details (e.g., legs for spreads, delta, contract object)
        """
        if symbol in self.current_positions:
            # Handle existing position (e.g., averaging down, partial close then re-open)
            # For simplicity, current implementation overwrites, which might not be ideal for all scenarios.
            self.logger.warning(f"Position already exists for {symbol}, overwriting with new details.")
            
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price, # For condors, this is net credit (positive)
            'current_price': entry_price, # Will be updated with current market value of the spread
            'type': position_type,
            'entry_time': datetime.now(),
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'option_data': option_data or {}
        }

        # Specific handling for credit spreads (iron condors, verticals) for SL/TP.
        # All thresholds are expressed in per-share points (same units as quoted
        # option/spread prices), matching entry_price and the market values
        # passed to update_position().
        if position_type in CREDIT_SPREAD_TYPES:
            max_risk_points = self._max_risk_points(entry_price, option_data)
            net_credit_received = entry_price  # Should be positive

            if max_risk_points > 0 and net_credit_received > 0:
                # Stop loss: close when the debit to buy the spread back has grown
                # to entry credit + X% of the max possible loss. P&L at that point
                # is -X% of max risk.
                stop_loss_factor = self.risk_params.condor_stop_loss_factor_of_max_risk / 100.0
                position['stop_loss_value'] = net_credit_received + (max_risk_points * stop_loss_factor)
                # Target profit: close when the spread can be bought back for
                # (100 - Y)% of the credit, retaining Y% of it as profit.
                profit_factor = self.risk_params.condor_profit_target_factor_of_credit / 100.0
                position['take_profit_value'] = net_credit_received * (1 - profit_factor)
            else:
                self.logger.warning(f"Max risk or net credit not properly defined for {position_type} {symbol}, SL/TP may not be effective.")
                position['stop_loss_value'] = None
                position['take_profit_value'] = None
        else: # For single options or other types
            # Stop loss and take profit for long positions (buying options)
            if entry_price > 0: # Assuming long option if entry_price is positive cost
                position['stop_loss'] = entry_price * (1 - self.stop_loss_pct/100)
                position['take_profit'] = entry_price * (1 + self.target_profit_pct/100)
            # Add logic for short single options if applicable (negative entry_price for credit)
            
        self.current_positions[symbol] = position
        self._position_history.append({
            'action': 'open',
            'time': datetime.now(),
            'position': position.copy()
        })
        
        self.logger.info(f"Added position: {quantity} {symbol} @ {entry_price:.2f} ({position_type})")
    
    def update_position(self, symbol: str, current_market_value: float) -> Dict:
        """
        Update a position with its current market value and P&L.
        
        Args:
            symbol: Symbol or identifier
            current_market_value: Current market value of the position.
                                  For options, this is the option price.
                                  For spreads like Iron Condors, this is the current cost to close (debit) or credit if market moved favorably.
                                  A positive value means it costs that much to buy back/close.
                                  A negative value means you would receive that much credit to close (unlikely for a condor initially sold for credit).
            
        Returns:
            Dict with position info and status
        """
        if symbol not in self.current_positions:
            self.logger.warning(f"No position found for {symbol} to update.")
            return {'symbol': symbol, 'status': 'not_found'}
            
        position = self.current_positions[symbol]
        position['current_price'] = current_market_value # This is the current value/cost_to_close of the spread/option
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        position_type = position['type']
        
        # Calculate P&L
        if position_type in CREDIT_SPREAD_TYPES:
            # Entry price was net credit (positive). Current market value is cost to close (debit, positive).
            # P&L = (Credit Received - Debit to Close) * Quantity * Multiplier (usually 100 for options)
            position['pnl'] = (entry_price - current_market_value) * quantity * 100
            initial_investment_basis = entry_price * quantity * 100 # The credit received
            if initial_investment_basis != 0:
                 # P&L % relative to initial credit. Can be > 100% if it becomes a large loss.
                position['pnl_pct'] = (position['pnl'] / abs(initial_investment_basis)) * 100
            else:
                position['pnl_pct'] = 0
        elif position_type == 'option':
            # Assumes long option (bought, entry_price is cost)
            # P&L = (Current Price - Entry Price) * Quantity * Multiplier
            position['pnl'] = (current_market_value - entry_price) * quantity * 100
            initial_cost = entry_price * quantity * 100
            if initial_cost != 0:
                position['pnl_pct'] = (position['pnl'] / initial_cost) * 100
            else:
                position['pnl_pct'] = 0
        else: # Simple stock-like P&L
            position['pnl'] = (current_market_value - entry_price) * quantity
            if entry_price != 0:
                position['pnl_pct'] = ((current_market_value / entry_price) - 1.0) * 100
            else:
                position['pnl_pct'] = 0
            
        # Determine status based on stop loss and take profit
        status = 'open'
        if position_type in CREDIT_SPREAD_TYPES:
            stop_loss_val = position.get('stop_loss_value')
            take_profit_val = position.get('take_profit_value')
            # current_market_value is the debit to close.
            # If debit_to_close >= stop_loss_val (e.g. stop_loss_val is a higher debit or smaller credit than entry)
            # If debit_to_close <= take_profit_val (e.g. take_profit_val is a smaller debit, meaning profit taken)
            if stop_loss_val is not None and current_market_value >= stop_loss_val: # Cost to close is too high
                status = 'stop_loss'
            elif take_profit_val is not None and current_market_value <= take_profit_val: # Cost to close is favorably low
                status = 'take_profit'
        elif position_type == 'option': # Assuming long option
            if current_market_value <= position.get('stop_loss', float('-inf')):
                status = 'stop_loss'
            elif current_market_value >= position.get('take_profit', float('inf')):
                status = 'take_profit'
        # Add other types if necessary
                
        position['status'] = status
        self.logger.debug(f"Updated position {symbol}: P&L ${position['pnl']:.2f} ({position['pnl_pct']:.2f}%), Status: {status}")
        return {'symbol': symbol, 'position': position, 'status': status}
    
    def close_position(self, symbol: str, exit_price: float) -> Dict:
        """
        Close a position and calculate final P&L.
        
        Args:
            symbol: Symbol or identifier
            exit_price: Exit price per share/contract. For spreads, this is the net debit paid or credit received to close.
                        For an Iron Condor sold for credit, exit_price is the debit paid to close.
        Returns:
            Dict with position details and final P&L
        """
        if symbol not in self.current_positions:
            self.logger.warning(f"No position found for {symbol} to close.")
            return {'symbol': symbol, 'status': 'not_found', 'pnl': 0.0}
            
        position = self.current_positions.pop(symbol) # Remove from current positions
        entry_price = position['entry_price']
        quantity = position['quantity']
        position_type = position['type']
        
        final_pnl = 0.0
        final_pnl_pct = 0.0

        if position_type in CREDIT_SPREAD_TYPES:
            # entry_price was net credit (e.g., +1.50)
            # exit_price is net debit to close (e.g., +0.30 to buy back for profit, or +2.00 to buy back for loss)
            # P&L = (Credit Received - Debit Paid) * Quantity * 100
            final_pnl = (entry_price - exit_price) * quantity * 100
            initial_investment_basis = entry_price * quantity * 100 # The credit received
            if initial_investment_basis != 0:
                final_pnl_pct = (final_pnl / abs(initial_investment_basis)) * 100
        elif position_type == 'option':
            # Assumes long option (bought, entry_price is cost)
            final_pnl = (exit_price - entry_price) * quantity * 100
            initial_cost = entry_price * quantity * 100
            if initial_cost != 0:
                final_pnl_pct = (final_pnl / initial_cost) * 100
        else:
            final_pnl = (exit_price - entry_price) * quantity
            if entry_price != 0:
                final_pnl_pct = ((exit_price / entry_price) - 1.0) * 100
            
        # Record trade completion
        self._position_history.append({
            'action': 'close',
            'time': datetime.now(),
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': final_pnl,
            'pnl_pct': final_pnl_pct,
            'position_type': position_type,
            'option_data': position.get('option_data')
        })
        
        # Update daily P&L
        self.update_daily_pnl(final_pnl)
        
        self.logger.info(f"Closed position: {quantity} {symbol} @ exit {exit_price:.2f}, "
                       f"Entry: {entry_price:.2f}, P&L: ${final_pnl:.2f} ({final_pnl_pct:.2f}%)")
                       
        return {
            'symbol': symbol,
            'position_details': position, # The state of position before closing
            'exit_price': exit_price,
            'pnl': final_pnl,
            'pnl_pct': final_pnl_pct
        }
    
    def close_all_positions(self, market_data: Dict[str, float]) -> List[Dict]:
        """
        Close all open positions at current market prices.
        
        Args:
            market_data: Dictionary of current prices by symbol
            
        Returns:
            List of closed position details
        """
        results = []
        
        if not self.current_positions:
            self.logger.info("No positions to close")
            return results
            
        self.logger.info(f"Closing all positions: {len(self.current_positions)}")
        
        # Make a copy of keys since we'll modify the dictionary during iteration
        for symbol in list(self.current_positions.keys()):
            exit_price = market_data.get(symbol)
            
            if exit_price is None:
                self.logger.warning(f"No market data for {symbol}, using last known price")
                position = self.current_positions[symbol]
                exit_price = position.get('current_price', position['entry_price'])
                
            result = self.close_position(symbol, exit_price)
            results.append(result)
            
        return results
    
    def get_open_positions(self) -> Dict[str, Dict]:
        """
        Get all currently open positions.
        
        Returns:
            Dictionary of open positions
        """
        return self.current_positions
    
    def get_position_history(self) -> List[Dict]:
        """
        Get position history.
        
        Returns:
            List of historical position actions
        """
        return self._position_history
    
    def get_portfolio_stats(self) -> Dict:
        """
        Get portfolio statistics.
        
        Returns:
            Dictionary with portfolio stats
        """
        total_pnl = sum(p['pnl'] for p in self._position_history if 'action' in p and p['action'] == 'close')
        trade_count = sum(1 for p in self._position_history if 'action' in p and p['action'] == 'close')
        
        winning_trades = sum(1 for p in self._position_history 
                            if 'action' in p and p['action'] == 'close' and p.get('pnl', 0) > 0)
                            
        losing_trades = sum(1 for p in self._position_history 
                           if 'action' in p and p['action'] == 'close' and p.get('pnl', 0) <= 0)
        
        win_rate = (winning_trades / trade_count) * 100 if trade_count > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'trade_count': trade_count,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown
        }
    
    def should_close_for_day(self, current_time: datetime = None) -> bool:
        """
        Determine if we should close all positions for the day.

        Args:
            current_time: Current datetime. Naive datetimes are interpreted as
                market time (for tests/backtests); when None, the current time
                in the configured market timezone is used.

        Returns:
            bool: True if we should close all positions
        """
        market_tz = ZoneInfo(getattr(self.config.trading, 'market_timezone', 'America/New_York'))

        if current_time is None:
            current_time = datetime.now(market_tz)
        elif current_time.tzinfo is not None:
            current_time = current_time.astimezone(market_tz)

        # Get configured end time offset
        end_offset_hours = self.config.trading.day_end_offset_hours

        # Determine market close time (4:00 PM in the market timezone)
        market_close = time(16, 0)

        # Calculate cutoff time using timedelta for correct arithmetic
        market_close_dt = datetime.combine(current_time.date(), market_close)
        cutoff_dt = market_close_dt - timedelta(minutes=int(end_offset_hours * 60))
        cutoff_time = cutoff_dt.time()

        # Check if current time is past cutoff
        return current_time.time() >= cutoff_time


if __name__ == "__main__":
    # Test risk manager
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    config.risk.adjust_for_risk_level(5)  # Medium risk
    
    risk_manager = RiskManager(config)
    
    # Test position sizing
    account_value = 100000
    risk_manager.update_account_value(account_value)
    
    # Test stock position sizing
    stock_price = 150.0
    volatility = 25.0
    vol_mult = 0.8
    
    stock_quantity = risk_manager.calculate_position_size(stock_price, volatility, account_value, vol_mult)
    print(f"Stock position size: {stock_quantity} shares")
    
    # Test option position sizing
    option_price = 3.50
    delta = 0.45
    
    option_quantity = risk_manager.calculate_option_quantity(option_price, delta, account_value, vol_mult)
    print(f"Option position size: {option_quantity} contracts")
    
    # Test position tracking
    risk_manager.add_position("SPY", stock_quantity, stock_price, "stock")
    risk_manager.add_position("SPY_CALL", option_quantity, option_price, "option", 
                            {"delta": delta, "expiry": "20221216"})
    
    # Update positions
    risk_manager.update_position("SPY", 153.0)
    risk_manager.update_position("SPY_CALL", 4.25)
    
    # Close positions
    risk_manager.close_position("SPY", 155.0)
    risk_manager.close_position("SPY_CALL", 5.0)
    
    # Show stats
    print("Portfolio stats:", risk_manager.get_portfolio_stats())