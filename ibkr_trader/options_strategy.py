"""
S&P500 options trading strategy module for the TuringTrader algorithm.
Handles option selection, strategy decisions, and trade execution logic.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from .config import Config
from .volatility_analyzer import VolatilityAnalyzer
from .risk_manager import RiskManager


class OptionsStrategy:
    """Options trading strategy implementation."""
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        volatility_analyzer: Optional[VolatilityAnalyzer] = None,
        risk_manager: Optional[RiskManager] = None
    ):
        """
        Initialize the options strategy.
        
        Args:
            config: Configuration object
            volatility_analyzer: Volatility analyzer instance
            risk_manager: Risk manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or Config()
        self.volatility_analyzer = volatility_analyzer or VolatilityAnalyzer(self.config)
        self.risk_manager = risk_manager or RiskManager(self.config)
        
        # Default parameters
        self.index_symbol = self.config.trading.index_symbol
        
        # State tracking
        self.current_vix = 0.0
        self.vix_signal = 'none'
        self.volatility_state = 'unknown'
        self.last_trade_time = None
        self.todays_trades = []
    
    def select_options_for_volatility(
        self, 
        option_chain: Dict, 
        vix_analysis: Dict, 
        current_price: float
    ) -> List[Dict]:
        """
        Select appropriate options based on current volatility conditions.
        
        Args:
            option_chain: Option chain data
            vix_analysis: VIX analysis data
            current_price: Current price of the underlying
            
        Returns:
            List of selected option contracts
        """
        self.logger.info("Selecting options for current volatility conditions")
        
        # Update state
        self.current_vix = vix_analysis.get('current_vix', 0.0)
        self.vix_signal = vix_analysis.get('signal', 'none')
        self.volatility_state = vix_analysis.get('volatility_state', 'unknown')
        
        if not option_chain:
            self.logger.error("No option chain data provided")
            return []
            
        # If volatility signal doesn't suggest trading, return empty list
        if self.vix_signal not in ['buy', 'strong_buy']:
            self.logger.info(f"VIX signal {self.vix_signal} doesn't suggest trading")
            return []
            
        # Analyze option chain for opportunities
        chain_analysis = self.volatility_analyzer.analyze_option_chain(option_chain)
        opportunities = chain_analysis.get('opportunities', [])
        
        if not opportunities:
            self.logger.info("No suitable options found in the chain")
            return []
            
        # Filter opportunities based on current volatility conditions
        selected = []
        
        # Get DTE (days to expiration) range based on volatility
        min_dte, max_dte = self._get_dte_range()
        
        # Get strike distance based on volatility
        strike_pct = self._get_strike_distance()
        
        for opt in opportunities:
            # Check if days to expiry is within our range
            dte = opt.get('days_to_expiry', 0)
            if not (min_dte <= dte <= max_dte):
                continue
                
            # Check if strike price is within our target range from current price
            strike = float(opt.get('strike', 0))
            strike_distance = abs((strike / current_price) - 1.0) * 100
            
            if strike_distance > strike_pct:
                continue
                
            # Ensure option has enough liquidity (has bid and ask)
            bid = opt.get('bid', 0)
            ask = opt.get('ask', 0)
            
            if bid <= 0 or ask <= 0 or (ask - bid) / bid > 0.15:  # Max 15% spread
                continue
                
            # Calculate score based on IV, DTE, and strike distance
            score = self._calculate_option_score(opt, current_price)
            opt['score'] = score
            
            selected.append(opt)
            
        # Sort by score (descending)
        selected.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Take top N options
        top_n = min(5, len(selected))
        top_options = selected[:top_n]
        
        self.logger.info(f"Selected {len(top_options)} options out of {len(opportunities)} opportunities")
        
        # Log selected options
        for i, opt in enumerate(top_options):
            self.logger.info(f"Option {i+1}: {opt.get('type', '')} @ ${opt.get('strike', 0)}, "
                           f"expiry in {opt.get('days_to_expiry', 0)} days, "
                           f"IV: {opt.get('iv', 0)*100:.1f}%, "
                           f"score: {opt.get('score', 0):.2f}")
        
        return top_options
    
    def _get_dte_range(self) -> Tuple[int, int]:
        """
        Get appropriate days to expiration range based on volatility state.
        
        Returns:
            Tuple of (min_dte, max_dte)
        """
        if self.volatility_state == 'extreme':
            # In extreme volatility, use shorter-dated options
            return 3, 14
        elif self.volatility_state == 'high':
            # In high volatility, use medium-dated options
            return 7, 28
        elif self.volatility_state == 'normal':
            # In normal volatility, use longer-dated options
            return 14, 45
        else:  # Low volatility or unknown
            # Default range
            return 7, 30
    
    def _get_strike_distance(self) -> float:
        """
        Get appropriate strike distance percentage based on volatility state.
        
        Returns:
            float: Strike distance percentage
        """
        if self.volatility_state == 'extreme':
            # In extreme volatility, use strikes further from the money
            return 10.0
        elif self.volatility_state == 'high':
            # In high volatility, use strikes moderately away from the money
            return 7.5
        elif self.volatility_state == 'normal':
            # In normal volatility, use strikes closer to the money
            return 5.0
        else:  # Low volatility or unknown
            # Default
            return 3.0
    
    def _calculate_option_score(self, option: Dict, current_price: float) -> float:
        """
        Calculate a score for an option based on its characteristics.
        Higher score = better option for current conditions.
        
        Args:
            option: Option data
            current_price: Current price of the underlying
            
        Returns:
            float: Score from 0.0 to 100.0
        """
        # Extract option parameters
        iv = option.get('iv', 0) * 100  # Convert to percentage
        dte = option.get('days_to_expiry', 30)
        strike = float(option.get('strike', current_price))
        option_type = option.get('type', 'call')
        bid = option.get('bid', 0)
        ask = option.get('ask', 0)
        
        # Calculate mid price
        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        
        # Calculate score components
        
        # 1. IV Score (higher IV = higher score, max 30 points)
        # But not too high (>100% IV might be problematic)
        iv_score = min(30, iv / 2) if iv <= 60 else max(0, 30 - (iv - 60) / 3)
        
        # 2. DTE Score (sweet spot around 14-28 days, max 25 points)
        if dte < 5:
            dte_score = dte * 3  # Quickly increasing
        elif 5 <= dte < 14:
            dte_score = 15 + (dte - 5)  # Gradually increasing
        elif 14 <= dte <= 28:
            dte_score = 25  # Optimal range
        else:
            dte_score = max(0, 25 - (dte - 28) / 3)  # Gradually decreasing
            
        # 3. Strike Distance Score (based on volatility state, max 25 points)
        # Calculate moneyness (how far strike is from current price)
        moneyness = abs((strike / current_price) - 1.0) * 100
        
        # Ideal moneyness depends on volatility state
        if self.volatility_state == 'extreme':
            ideal_moneyness = 8.0
        elif self.volatility_state == 'high':
            ideal_moneyness = 5.0
        elif self.volatility_state == 'normal':
            ideal_moneyness = 3.0
        else:  # Low volatility or unknown
            ideal_moneyness = 2.0
            
        # Score based on how close to ideal moneyness
        strike_score = max(0, 25 - abs(moneyness - ideal_moneyness) * 2)
        
        # 4. Liquidity Score (based on bid-ask spread, max 20 points)
        if mid_price > 0:
            spread_pct = (ask - bid) / mid_price
            liquidity_score = max(0, 20 - spread_pct * 100)
        else:
            liquidity_score = 0
            
        # Calculate total score (max 100)
        total_score = iv_score + dte_score + strike_score + liquidity_score
        
        # Log score components for debugging
        self.logger.debug(f"Option score components - IV: {iv_score:.1f}, DTE: {dte_score:.1f}, "
                        f"Strike: {strike_score:.1f}, Liquidity: {liquidity_score:.1f}, "
                        f"Total: {total_score:.1f}")
                        
        return total_score
    
    def generate_trade_decision(
        self,
        vix_analysis: Dict, 
        option_chain: Dict,
        current_price: float,
        account_value: float
    ) -> Dict:
        """
        Generate a trading decision based on current market conditions.
        
        Args:
            vix_analysis: VIX analysis data
            option_chain: Option chain data
            current_price: Current price of the underlying
            account_value: Current account value
            
        Returns:
            Dictionary with trade decision details
        """
        self.logger.info("Generating trade decision")
        
        # Should we trade today based on volatility?
        should_trade = self.volatility_analyzer.should_trade_today(vix_analysis)
        
        if not should_trade:
            self.logger.info("Decision: No trade - volatility conditions unfavorable")
            return {
                'action': 'none',
                'reason': 'unfavorable_volatility',
                'vix': vix_analysis.get('current_vix', 0.0),
                'vix_signal': vix_analysis.get('signal', 'none')
            }
            
        # Select options
        selected_options = self.select_options_for_volatility(option_chain, vix_analysis, current_price)
        
        if not selected_options:
            self.logger.info("Decision: No trade - no suitable options found")
            return {
                'action': 'none',
                'reason': 'no_suitable_options',
                'vix': vix_analysis.get('current_vix', 0.0),
                'vix_signal': vix_analysis.get('signal', 'none')
            }
            
        # Get the best option
        best_option = selected_options[0]
        
        # Get position size multiplier based on volatility
        position_size_multiplier = self.volatility_analyzer.get_position_size_multiplier(vix_analysis)
        
        # Calculate quantity
        option_price = (best_option.get('bid', 0) + best_option.get('ask', 0)) / 2
        delta = 0.5  # Default if not available
        
        if option_price <= 0:
            self.logger.error("Invalid option price")
            return {
                'action': 'none',
                'reason': 'invalid_price',
                'vix': vix_analysis.get('current_vix', 0.0)
            }
            
        # Calculate quantity
        quantity = self.risk_manager.calculate_option_quantity(
            option_price, delta, account_value, position_size_multiplier
        )
        
        if quantity <= 0:
            self.logger.info("Decision: No trade - quantity calculation resulted in zero")
            return {
                'action': 'none',
                'reason': 'zero_quantity',
                'vix': vix_analysis.get('current_vix', 0.0)
            }
            
        # Determine trade action based on option type and current volatility trend
        option_type = best_option.get('type', 'call')
        vix_change = vix_analysis.get('vix_change_1d', 0)
        
        # Get trade action:
        # - If volatility is rising strongly, buy calls
        # - If volatility is rising moderately, buy puts
        # - Otherwise, default to calls for high volatility
        if vix_change > 1.0 and vix_analysis.get('current_vix', 0) > 20:
            preferred_type = 'call'
        elif vix_change > 0:
            preferred_type = 'put'
        else:
            preferred_type = 'call'
            
        # Try to find an option of the preferred type in our top selections
        selected_option = None
        for opt in selected_options:
            if opt.get('type') == preferred_type:
                selected_option = opt
                break
                
        # If no option of preferred type, use the best one we have
        if selected_option is None:
            self.logger.info(f"No {preferred_type} option found in selections, using best available")
            selected_option = best_option
            
        # Create trade decision
        trade_decision = {
            'action': 'buy',
            'option_type': selected_option.get('type'),
            'symbol': self.index_symbol,
            'expiry': selected_option.get('expiry'),
            'strike': selected_option.get('strike'),
            'quantity': quantity,
            'price': option_price,
            'reason': 'volatility_opportunity',
            'vix': vix_analysis.get('current_vix', 0.0),
            'vix_signal': vix_analysis.get('signal', 'none'),
            'score': selected_option.get('score', 0)
        }
        
        self.logger.info(f"Decision: Buy {quantity} {self.index_symbol} "
                       f"{selected_option.get('type', '')} options @ "
                       f"strike ${selected_option.get('strike', 0)}, "
                       f"expiry: {selected_option.get('expiry', '')}")
                       
        self.last_trade_time = datetime.now()
        self.todays_trades.append(trade_decision)
        
        return trade_decision
    
    def should_close_positions(self, current_time: Optional[datetime] = None) -> bool:
        """
        Determine if positions should be closed based on time of day.
        
        Args:
            current_time: Current time, uses system time if None
            
        Returns:
            bool: True if positions should be closed
        """
        return self.risk_manager.should_close_for_day(current_time)
    
    def reset_daily_state(self) -> None:
        """Reset daily state for new trading day."""
        self.todays_trades = []
        self.last_trade_time = None
        self.risk_manager.reset_daily_metrics()
        self.logger.info("Daily state reset for new trading day")


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    
    # Create dependencies
    config = Config()
    volatility_analyzer = VolatilityAnalyzer(config)
    risk_manager = RiskManager(config)
    
    # Create strategy
    strategy = OptionsStrategy(config, volatility_analyzer, risk_manager)
    
    # Test with sample data
    vix_analysis = {
        'current_vix': 25.0,
        'vix_change_1d': 2.0,
        'volatility_state': 'high',
        'signal': 'buy'
    }
    
    # Mock option chain with some sample data
    option_chain = {
        '20221216': {
            'days_to_expiry': 14,
            'calls': {
                400: {'bid': 5.0, 'ask': 5.5, 'iv': 0.3},
                410: {'bid': 3.5, 'ask': 4.0, 'iv': 0.35},
                420: {'bid': 2.0, 'ask': 2.5, 'iv': 0.4}
            },
            'puts': {
                400: {'bid': 4.0, 'ask': 4.5, 'iv': 0.32},
                390: {'bid': 2.5, 'ask': 3.0, 'iv': 0.37},
                380: {'bid': 1.5, 'ask': 2.0, 'iv': 0.42}
            },
            'strikes': [380, 390, 400, 410, 420]
        },
        '20221223': {
            'days_to_expiry': 21,
            'calls': {
                400: {'bid': 7.0, 'ask': 7.5, 'iv': 0.28},
                410: {'bid': 5.0, 'ask': 5.5, 'iv': 0.33},
                420: {'bid': 3.0, 'ask': 3.5, 'iv': 0.38}
            },
            'puts': {
                400: {'bid': 6.0, 'ask': 6.5, 'iv': 0.30},
                390: {'bid': 4.0, 'ask': 4.5, 'iv': 0.35},
                380: {'bid': 2.5, 'ask': 3.0, 'iv': 0.40}
            },
            'strikes': [380, 390, 400, 410, 420]
        }
    }
    
    # Test generating trade decision
    current_price = 405.0
    account_value = 100000.0
    
    # Update risk manager
    risk_manager.update_account_value(account_value)
    
    # Generate trade decision
    decision = strategy.generate_trade_decision(vix_analysis, option_chain, current_price, account_value)
    print("Trade decision:", decision)