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
try:
    from .market_data_store import MarketDataStore
except ImportError:
    MarketDataStore = None


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
        
        # Initialize market data store if available
        self.market_data_store = MarketDataStore(self.config) if MarketDataStore else None
        
        # Default parameters
        self.index_symbol = self.config.trading.index_symbol
        
        # Iron condor parameters
        self.strike_width_pct = self.config.vol_harvesting.strike_width_pct
        self.target_short_delta = self.config.vol_harvesting.target_short_delta
        self.target_long_delta = self.config.vol_harvesting.target_long_delta
        self.min_dte = self.config.vol_harvesting.min_dte
        self.max_dte = self.config.vol_harvesting.max_dte
        
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
        open_interest = option.get('open_interest', 0)
        volume = option.get('volume', 0)
        delta = abs(option.get('delta', 0))
        
        # Calculate mid price
        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        
        # Calculate score components
        
        # 1. IV Score (higher IV = higher score, but with a more optimal curve, max 25 points)
        # Prefer IV in the 30-60% range, penalize very low or very high IV
        if iv < 15:
            # Too low IV doesn't provide enough premium
            iv_score = iv * 0.5
        elif 15 <= iv <= 30:
            # Growing score as IV approaches optimal range
            iv_score = 7.5 + (iv - 15) * 0.7
        elif 30 < iv <= 60:
            # Optimal IV range
            iv_score = 18 + (iv - 30) * 0.23
        elif 60 < iv <= 80:
            # Still good but starting to get risky
            iv_score = 25 - (iv - 60) * 0.25
        else:
            # Very high IV is often a warning sign
            iv_score = max(0, 20 - (iv - 80) * 0.4)
        
        # 2. DTE Score (sweet spot around 21-45 days, max 20 points)
        # Adjusted to prefer slightly longer-dated options
        if dte < 7:
            # Too close to expiration - high gamma risk
            dte_score = max(0, dte * 1.5)
        elif 7 <= dte < 21:
            # Approaching ideal range
            dte_score = 10 + (dte - 7) * 0.5
        elif 21 <= dte <= 45:
            # Optimal range - best theta decay vs. capital efficiency
            dte_score = 20
        elif 45 < dte <= 60:
            # Still good but capital tied up longer
            dte_score = 20 - (dte - 45) * 0.3
        else:
            # Too far from expiration - reduced time decay benefits
            dte_score = max(0, 15 - (dte - 60) * 0.1)
            
        # 3. Strike Distance Score (based on volatility state, max 20 points)
        # Calculate moneyness (how far strike is from current price)
        moneyness = abs((strike / current_price) - 1.0) * 100
        
        # Ideal moneyness depends on volatility state
        if self.volatility_state == 'extreme':
            # Further OTM in extreme volatility
            ideal_moneyness = 10.0
        elif self.volatility_state == 'high':
            # Further OTM in high volatility
            ideal_moneyness = 7.0
        elif self.volatility_state == 'normal':
            # Moderately OTM in normal volatility
            ideal_moneyness = 4.0
        else:  # Low volatility or unknown
            # Closer to ATM in low volatility
            ideal_moneyness = 2.5
            
        # Score based on how close to ideal moneyness
        strike_score = max(0, 20 - abs(moneyness - ideal_moneyness) * 1.8)
        
        # 4. Liquidity Score (improved calculation, max 15 points)
        liquidity_score = 0
        if mid_price > 0:
            # Bid-ask spread component
            spread_pct = (ask - bid) / mid_price
            spread_score = max(0, 10 - spread_pct * 100)
            
            # Volume component
            volume_score = min(5, volume / 200)
            
            liquidity_score = spread_score + volume_score
        
        # 5. Open Interest Score (new component, max 10 points)
        # Higher open interest indicates more market participants and better liquidity
        oi_score = min(10, open_interest / 500)
        
        # 6. Risk-Reward Score (new component, max 10 points)
        risk_reward_score = 0
        if delta > 0:
            # For OTM options (which is what we mostly trade)
            # Lower delta (more OTM) should have better risk-reward for selling
            if delta < 0.2:
                risk_reward_score = 10
            elif delta < 0.3:
                risk_reward_score = 8
            elif delta < 0.4:
                risk_reward_score = 5
            elif delta < 0.5:
                risk_reward_score = 3
            else:
                risk_reward_score = 1
        
        # Calculate total score (max 100)
        total_score = iv_score + dte_score + strike_score + liquidity_score + oi_score + risk_reward_score
        
        # Log score components for debugging
        self.logger.debug(f"Option score components - IV: {iv_score:.1f}, DTE: {dte_score:.1f}, "
                        f"Strike: {strike_score:.1f}, Liquidity: {liquidity_score:.1f}, "
                        f"OI: {oi_score:.1f}, Risk/Reward: {risk_reward_score:.1f}, "
                        f"Total: {total_score:.1f}")
                        
        return total_score
    
    def _evaluate_trade(self, trade_data: Dict) -> float:
        """
        Evaluate a potential trade based on risk/reward and market conditions.
        
        Args:
            trade_data: Dictionary containing trade information
                - net_credit: Net credit received for the trade
                - max_loss: Maximum possible loss
                - prob_profit: Probability of profit (0-100)
                - days_to_expiry: Days to expiration
                - width: Width of the spread in points
                - iv: Implied volatility
                - iv_hv_ratio: IV to HV ratio
        
        Returns:
            float: Trade score from 0.0 to 100.0 (higher is better)
        """
        # Extract trade parameters
        net_credit = trade_data.get('net_credit', 0)
        max_loss = trade_data.get('max_loss', float('inf'))
        prob_profit = trade_data.get('prob_profit', 0)
        days_to_expiry = trade_data.get('days_to_expiry', 30)
        width = trade_data.get('width', 0)
        iv = trade_data.get('iv', 0) * 100  # Convert to percentage
        iv_hv_ratio = trade_data.get('iv_hv_ratio', 1.0)
        
        # If critical data is missing or invalid, return low score
        if net_credit <= 0 or max_loss <= 0 or width <= 0:
            return 0
        
        # Calculate trade metrics
        return_on_risk = (net_credit / max_loss) * 100
        risk_reward_ratio = max_loss / net_credit
        daily_return_rate = return_on_risk / days_to_expiry
        
        # 1. Return on Risk Score (max 30 points)
        # Higher return on risk is better
        if return_on_risk < 5:
            ror_score = return_on_risk * 2  # Low returns score poorly
        elif 5 <= return_on_risk < 15:
            ror_score = 10 + (return_on_risk - 5) * 1.5  # Moderate returns
        else:
            ror_score = 25 + min(5, (return_on_risk - 15) * 0.5)  # High returns (capped)
        
        # 2. Probability of Profit Score (max 25 points)
        # Higher probability of profit is better
        pop_score = min(25, prob_profit * 0.25)
        
        # 3. Risk-Reward Ratio Score (max 20 points)
        # Lower risk-reward ratio is better (we want to risk less to make more)
        if risk_reward_ratio > 10:
            rr_score = 0  # Too much risk
        elif 5 < risk_reward_ratio <= 10:
            rr_score = 5  # High risk
        elif 3 < risk_reward_ratio <= 5:
            rr_score = 10  # Moderate risk
        elif 2 < risk_reward_ratio <= 3:
            rr_score = 15  # Good risk-reward
        else:
            rr_score = 20  # Excellent risk-reward
        
        # 4. Daily Return Rate Score (max 15 points)
        # Higher daily return rate is better
        daily_score = min(15, daily_return_rate * 3)
        
        # 5. IV/HV Ratio Score (max 10 points)
        # Higher IV/HV ratio is better for selling options
        if iv_hv_ratio < 1.0:
            iv_hv_score = 0  # IV less than HV is not good for selling
        elif 1.0 <= iv_hv_ratio < 1.2:
            iv_hv_score = iv_hv_ratio * 5  # Adequate
        elif 1.2 <= iv_hv_ratio < 1.5:
            iv_hv_score = 6 + (iv_hv_ratio - 1.2) * 10  # Good
        else:
            iv_hv_score = 9 + min(1, (iv_hv_ratio - 1.5) * 2)  # Excellent (capped)
        
        # Calculate total score
        total_score = ror_score + pop_score + rr_score + daily_score + iv_hv_score
        
        # Log score components
        self.logger.debug(f"Trade evaluation - Return on Risk: {ror_score:.1f}, Prob of Profit: {pop_score:.1f}, "
                        f"Risk-Reward: {rr_score:.1f}, Daily Return: {daily_score:.1f}, "
                        f"IV/HV: {iv_hv_score:.1f}, Total: {total_score:.1f}")
        
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
        
        # Get more detailed volatility analysis
        vol_harvest_analysis = self.volatility_analyzer.analyze_volatility_for_harvesting(
            self.index_symbol, vix_analysis, option_chain
        )
        
        # Determine which strategy to use based on market conditions
        current_vix = vix_analysis.get('current_vix', 0)
        vix_change_1d = vix_analysis.get('vix_change_1d', 0)
        vix_change_5d = vix_analysis.get('vix_change_5d', 0)
        vix_trend = vix_analysis.get('vix_trend', 'flat')
        vix_regime = vix_analysis.get('vix_regime', 'normal')
        iv_hv_ratio = vol_harvest_analysis.get('iv_hv_ratio', 0)
        volatility_state = vix_analysis.get('volatility_state', 'normal')
        
        # Enhanced decision metrics for strategy selection
        # ------------------------------------------------
        # 1. Extremely high volatility environment with decreasing VIX: Bull Put Spreads
        # 2. High volatility environment with rising VIX: Bear Call Spreads
        # 3. Normal volatility with high IV/HV ratio: Iron Condors
        # 4. Stable, low volatility environments: Iron Condors with narrower width
        # 5. Default: No trade if conditions aren't favorable
        
        self.logger.info(f"Market conditions - VIX: {current_vix:.1f}, Change 1d: {vix_change_1d:.1f}, "
                       f"Change 5d: {vix_change_5d:.1f}, IV/HV ratio: {iv_hv_ratio:.2f}, "
                       f"State: {volatility_state}, Trend: {vix_trend}, Regime: {vix_regime}")
        
        # Strategy selection based on volatility environment
        strategy = None  # Start with no strategy - must meet criteria to trade
        spread_type = None
        min_credit_threshold = 0.0  # Base minimum credit
        min_return_on_risk = 0.0    # Base minimum return on risk
        
        # Set minimum requirements based on volatility regime
        if vix_regime == 'crisis':
            # In crisis regime, require substantial premium
            min_credit_threshold = 0.50  # Minimum $0.50 per spread
            min_return_on_risk = 20.0    # Minimum 20% return on risk
        elif vix_regime == 'high_volatility':
            min_credit_threshold = 0.35  # Minimum $0.35 per spread
            min_return_on_risk = 15.0    # Minimum 15% return on risk
        elif vix_regime == 'normal':
            min_credit_threshold = 0.25  # Minimum $0.25 per spread
            min_return_on_risk = 12.0    # Minimum 12% return on risk
        else:  # low_volatility
            min_credit_threshold = 0.20  # Minimum $0.20 per spread
            min_return_on_risk = 10.0    # Minimum 10% return on risk
        
        # Strategy decision tree with enhanced selectivity
        if volatility_state in ('extreme', 'high'):
            # Extreme volatility (VIX > 25)
            if vix_trend in ('down', 'strong_down'):
                # VIX declining - bullish bias
                strategy = "vertical_spread"
                spread_type = "bull_put"
                min_credit_threshold *= 1.2  # Higher minimum for extreme vol
                self.logger.info(f"High volatility with {vix_trend} VIX trend - selecting bull put spreads")
            elif vix_trend in ('up', 'strong_up'):
                # VIX rising - bearish bias
                strategy = "vertical_spread"
                spread_type = "bear_call"
                min_credit_threshold *= 1.2  # Higher minimum for extreme vol
                self.logger.info(f"High volatility with {vix_trend} VIX trend - selecting bear call spreads")
            elif iv_hv_ratio > 1.4:
                # Very high IV/HV ratio - premium selling opportunity but with caution
                strategy = "vertical_spread"
                # In flat high volatility, slight bias to put side as markets often recover
                spread_type = "bull_put"
                self.logger.info(f"High volatility with flat VIX and high IV/HV - selecting bull put spreads")
            else:
                # Not enough edge in high volatility without clear direction or premium
                strategy = None
                self.logger.info(f"High volatility without clear direction or sufficient premium - no trade")
        
        elif volatility_state == 'normal':
            # Normal volatility (VIX 18-25)
            if iv_hv_ratio > 1.5:
                # Very high IV/HV ratio - excellent premium harvesting opportunity
                strategy = "iron_condor"
                self.logger.info(f"Normal volatility with very high IV/HV ratio ({iv_hv_ratio:.2f}) - selecting iron condors")
            elif iv_hv_ratio > 1.3:
                # Good IV/HV ratio - still good for iron condors
                strategy = "iron_condor"
                self.logger.info(f"Normal volatility with good IV/HV ratio ({iv_hv_ratio:.2f}) - selecting iron condors")
            elif iv_hv_ratio > 1.2 and vix_trend in ('up', 'strong_up'):
                # Rising volatility with decent IV/HV - vertical spreads
                strategy = "vertical_spread"
                spread_type = "bear_call"  # Bearish bias when volatility is rising
                self.logger.info(f"Normal volatility with rising VIX - selecting bear call spreads")
            elif iv_hv_ratio > 1.2 and vix_trend in ('down', 'strong_down'):
                # Falling volatility with decent IV/HV - vertical spreads
                strategy = "vertical_spread"
                spread_type = "bull_put"  # Bullish bias when volatility is falling
                self.logger.info(f"Normal volatility with falling VIX - selecting bull put spreads")
            else:
                # Not enough premium edge
                strategy = None
                self.logger.info(f"Normal volatility without sufficient premium (IV/HV: {iv_hv_ratio:.2f}) - no trade")
        
        elif volatility_state == 'low':
            # Low volatility (VIX 13-18)
            if iv_hv_ratio > 1.4:
                # Very high IV/HV ratio despite low vol - good for iron condors
                strategy = "iron_condor"
                self.logger.info(f"Low volatility with high IV/HV ratio ({iv_hv_ratio:.2f}) - selecting iron condors")
            elif vix_trend in ('up', 'strong_up') and iv_hv_ratio > 1.25:
                # Rising volatility from low levels with decent premium
                strategy = "vertical_spread"
                spread_type = "bear_call"  # Bearish bias with rising volatility
                self.logger.info(f"Low but rising volatility - selecting bear call spreads")
            else:
                # Not enough edge in low volatility
                strategy = None
                self.logger.info(f"Low volatility without sufficient edge - no trade")
        
        else:  # very_low
            # Very low volatility (VIX < 13)
            if iv_hv_ratio > 1.5:
                # Extremely high IV/HV ratio despite very low vol - rare opportunity
                strategy = "iron_condor"
                self.logger.info(f"Very low volatility with very high IV/HV ratio ({iv_hv_ratio:.2f}) - selecting iron condors")
            else:
                # Generally avoid trading in very low volatility
                strategy = None
                self.logger.info(f"Very low volatility - insufficient premium for trading")
        
        # Execute the selected strategy with minimum requirements
        if strategy == "iron_condor":
            trade_decision = self.generate_iron_condor_trade(
                option_chain, current_price, account_value, vix_analysis,
                min_credit=min_credit_threshold, min_return_on_risk=min_return_on_risk
            )
            
            # Verify the trade meets our minimum requirements
            if trade_decision['action'] != 'none':
                # Evaluate the trade quality
                trade_data = {
                    'net_credit': trade_decision.get('net_credit', 0),
                    'max_loss': trade_decision.get('max_loss', float('inf')),
                    'prob_profit': trade_decision.get('prob_profit', 0),
                    'days_to_expiry': trade_decision.get('days_to_expiry', 30),
                    'width': trade_decision.get('width', 0),
                    'iv': vol_harvest_analysis.get('implied_volatility', 0),
                    'iv_hv_ratio': iv_hv_ratio
                }
                
                trade_score = self._evaluate_trade(trade_data)
                trade_decision['trade_score'] = trade_score
                
                # Only execute trades with a minimum score
                if trade_score < 60:
                    self.logger.info(f"Iron condor trade score too low ({trade_score:.1f}) - no trade")
                    trade_decision = {'action': 'none', 'reason': 'low_trade_score'}
                else:
                    self.logger.info(f"Iron condor trade score: {trade_score:.1f} - executing trade")
            
            return trade_decision
            
        elif strategy == "vertical_spread" and spread_type is not None:
            trade_decision = self.generate_vertical_spread_trade(
                option_chain, current_price, account_value, vix_analysis, spread_type,
                min_credit=min_credit_threshold, min_return_on_risk=min_return_on_risk
            )
            
            # Verify the trade meets our minimum requirements
            if trade_decision['action'] != 'none':
                # Evaluate the trade quality
                trade_data = {
                    'net_credit': trade_decision.get('net_credit', 0),
                    'max_loss': trade_decision.get('max_loss', float('inf')),
                    'prob_profit': trade_decision.get('prob_profit', 0),
                    'days_to_expiry': trade_decision.get('days_to_expiry', 30),
                    'width': trade_decision.get('width', 0),
                    'iv': vol_harvest_analysis.get('implied_volatility', 0),
                    'iv_hv_ratio': iv_hv_ratio
                }
                
                trade_score = self._evaluate_trade(trade_data)
                trade_decision['trade_score'] = trade_score
                
                # Only execute trades with a minimum score
                if trade_score < 65:  # Slightly higher threshold for vertical spreads
                    self.logger.info(f"Vertical spread trade score too low ({trade_score:.1f}) - no trade")
                    trade_decision = {'action': 'none', 'reason': 'low_trade_score'}
                else:
                    self.logger.info(f"Vertical spread trade score: {trade_score:.1f} - executing trade")
            
            return trade_decision
        else:
            # No suitable strategy selected
            self.logger.info("No suitable strategy selected based on current market conditions")
            return {
                'action': 'none',
                'reason': 'unfavorable_market_conditions'
            }
            return trade_decision
                       
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

    def find_iron_condor_legs(
        self,
        option_chain: Dict,
        current_price: float,
        expiry_target: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Find appropriate legs for an iron condor spread.
        
        Args:
            option_chain: Option chain data
            current_price: Current price of the underlying
            expiry_target: Target expiry date (if None, will select based on DTE range)
            
        Returns:
            Dictionary with iron condor leg details or None if no suitable legs found
        """
        if not option_chain:
            self.logger.warning("No option chain data provided")
            return None
            
        self.logger.info(f"Finding iron condor legs at price {current_price}")
        
        # Find expiry date within target range
        target_expiry = None
        if expiry_target and expiry_target in option_chain:
            target_expiry = expiry_target
        else:
            # Filter by DTE range - optimized to focus on the sweet spot for theta decay
            valid_expiries = [
                exp for exp, data in option_chain.items()
                if self.min_dte <= data.get('days_to_expiry', 0) <= self.max_dte
            ]
            
            if valid_expiries:
                # Choose based on a combination of IV and optimal DTE (closer to optimal DTE range)
                best_expiry = None
                best_score = 0
                
                for exp in valid_expiries:
                    exp_data = option_chain[exp]
                    days_to_expiry = exp_data.get('days_to_expiry', 0)
                    calls = exp_data.get('calls', {})
                    puts = exp_data.get('puts', {})
                    
                    # Calculate average IV
                    call_ivs = [opt.get('iv', 0) for opt in calls.values() if opt.get('iv', 0) > 0]
                    put_ivs = [opt.get('iv', 0) for opt in puts.values() if opt.get('iv', 0) > 0]
                    
                    if call_ivs and put_ivs:
                        avg_iv = (sum(call_ivs) + sum(put_ivs)) / (len(call_ivs) + len(put_ivs))
                        
                        # Score based on optimal DTE (25-35 days is ideal for iron condors)
                        dte_score = 0
                        if 25 <= days_to_expiry <= 35:
                            dte_score = 1.0  # Ideal range
                        elif 20 <= days_to_expiry < 25 or 35 < days_to_expiry <= 40:
                            dte_score = 0.8  # Good range
                        else:
                            dte_score = 0.6  # Acceptable range
                            
                        # Combine IV and DTE scores, weighting DTE more
                        total_score = (avg_iv * 0.4) + (dte_score * 0.6)
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_expiry = exp
                
                target_expiry = best_expiry
        
        if not target_expiry:
            self.logger.warning("No suitable expiry date found for iron condor")
            return None
            
        # Get expiry data
        expiry_data = option_chain[target_expiry]
        days_to_expiry = expiry_data.get('days_to_expiry', 0)
        calls = expiry_data.get('calls', {})
        puts = expiry_data.get('puts', {})
        
        # Get available strikes
        strikes = sorted(set(list(calls.keys()) + list(puts.keys())))
        if not strikes:
            self.logger.warning("No valid strikes found")
            return None
            
        # Find short legs (inner legs) based on delta target and liquidity
        # First find the ATM strike (closest to current price)
        atm_strike = min(strikes, key=lambda x: abs(x - current_price))
        atm_index = strikes.index(atm_strike)
        
        # Find short call (OTM call with delta closest to target) with good liquidity
        short_call_candidates = []
        
        for i in range(atm_index, len(strikes)):
            strike = strikes[i]
            if strike > current_price and strike in calls:  # OTM call
                call = calls[strike]
                delta = call.get('delta', abs(1 - (strike / current_price)))
                bid = call.get('bid', 0)
                ask = call.get('ask', 0)
                volume = call.get('volume', 0) if call.get('volume') is not None else 0
                open_interest = call.get('open_interest', 0) if call.get('open_interest') is not None else 0
                
                # Calculate bid-ask spread percentage
                spread_pct = (ask - bid) / bid if bid > 0 else float('inf')
                
                # Only consider strikes with reasonable delta, liquidity and bid price
                if (delta <= self.target_short_delta and 
                    delta >= self.target_short_delta * 0.7 and  # Ensure delta is not too small
                    bid >= 0.15 and 
                    volume >= 10 and 
                    open_interest >= 50 and
                    spread_pct < 0.15):
                    
                    # Calculate a score based on how close delta is to target, premium, and liquidity
                    delta_score = 1.0 - abs(delta - self.target_short_delta) / self.target_short_delta
                    premium_score = min(1.0, bid / 3.0)  # Scale to max 1.0 at premium of $3.00
                    liquidity_score = min(1.0, volume / 500) * 0.5 + min(1.0, open_interest / 1000) * 0.5
                    
                    score = delta_score * 0.5 + premium_score * 0.3 + liquidity_score * 0.2
                    
                    short_call_candidates.append({
                        'strike': strike,
                        'delta': delta,
                        'bid': bid,
                        'ask': ask,
                        'option': call,
                        'score': score
                    })
        
        # Find short put (OTM put with delta closest to target) with good liquidity
        short_put_candidates = []
        
        for i in range(atm_index, -1, -1):
            strike = strikes[i]
            if strike < current_price and strike in puts:  # OTM put
                put = puts[strike]
                delta = put.get('delta', abs(1 - (strike / current_price)))
                bid = put.get('bid', 0)
                ask = put.get('ask', 0)
                volume = put.get('volume', 0) if put.get('volume') is not None else 0
                open_interest = put.get('open_interest', 0) if put.get('open_interest') is not None else 0
                
                # Calculate bid-ask spread percentage
                spread_pct = (ask - bid) / bid if bid > 0 else float('inf')
                
                # Only consider strikes with reasonable delta, liquidity and bid price
                if (delta <= self.target_short_delta and 
                    delta >= self.target_short_delta * 0.7 and  # Ensure delta is not too small
                    bid >= 0.15 and 
                    volume >= 10 and 
                    open_interest >= 50 and
                    spread_pct < 0.15):
                    
                    # Calculate a score based on how close delta is to target, premium, and liquidity
                    delta_score = 1.0 - abs(delta - self.target_short_delta) / self.target_short_delta
                    premium_score = min(1.0, bid / 3.0)  # Scale to max 1.0 at premium of $3.00
                    liquidity_score = min(1.0, volume / 500) * 0.5 + min(1.0, open_interest / 1000) * 0.5
                    
                    score = delta_score * 0.5 + premium_score * 0.3 + liquidity_score * 0.2
                    
                    short_put_candidates.append({
                        'strike': strike,
                        'delta': delta,
                        'bid': bid,
                        'ask': ask,
                        'option': put,
                        'score': score
                    })
        
        # If we don't have good candidates, return None
        if not short_call_candidates or not short_put_candidates:
            self.logger.warning("Could not find appropriate short legs with good liquidity for iron condor")
            return None
        
        # Select best short call and put candidates based on score
        short_call_candidates.sort(key=lambda x: x['score'], reverse=True)
        short_put_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        short_call = short_call_candidates[0]
        short_put = short_put_candidates[0]
        
        short_call_strike = short_call['strike']
        short_put_strike = short_put['strike']
        short_call_delta = short_call['delta']
        short_put_delta = short_put['delta']
        
        # Calculate strike width based on a combination of fixed percentage and volatility
        # Use a wider width in higher volatility environments for better protection
        iv_avg = (expiry_data.get('avg_iv_calls', 0) + expiry_data.get('avg_iv_puts', 0)) / 2
        volatility_factor = min(1.5, max(1.0, iv_avg / 25.0))  # Scale based on IV, capped at 1.5x
        
        # Adjust strike width for better risk control
        adjusted_strike_width = current_price * (self.strike_width_pct / 100) * volatility_factor
        
        # For better protection, ensure strike width is a minimum percentage of underlying price
        min_width = current_price * 0.02  # Minimum 2% width
        adjusted_strike_width = max(adjusted_strike_width, min_width)
        
        # Find long call strikes with good liquidity
        long_call_candidates = []
        
        # Try to find strikes within a range of the target width
        target_long_call_strike = short_call_strike + adjusted_strike_width
        potential_long_calls = [s for s in strikes if s > short_call_strike]
        
        if not potential_long_calls:
            self.logger.warning("No potential long call strikes found")
            return None
            
        for strike in potential_long_calls:
            if abs(strike - target_long_call_strike) / target_long_call_strike > 0.2:
                # Skip strikes that are too far from target (more than 20% deviation)
                continue
                
            if strike in calls:
                call = calls[strike]
                bid = call.get('bid', 0)
                ask = call.get('ask', 0)
                volume = call.get('volume', 0) if call.get('volume') is not None else 0
                
                # Skip strikes with poor liquidity or zero bid
                if bid <= 0 or volume < 5:
                    continue
                    
                # Calculate a score based on proximity to target strike and liquidity
                proximity_score = 1.0 - abs(strike - target_long_call_strike) / adjusted_strike_width
                liquidity_score = min(1.0, volume / 200)
                
                # Calculate debit as percentage of credit from short call
                debit_pct = (ask / short_call['bid']) if short_call['bid'] > 0 else float('inf')
                
                # Prefer strikes that use less of the credit from the short leg
                cost_efficiency = max(0, 1.0 - debit_pct)
                
                score = proximity_score * 0.5 + liquidity_score * 0.2 + cost_efficiency * 0.3
                
                long_call_candidates.append({
                    'strike': strike,
                    'option': call,
                    'bid': bid,
                    'ask': ask,
                    'score': score,
                    'width': strike - short_call_strike
                })
        
        # Find long put strikes with good liquidity
        long_put_candidates = []
        
        # Try to find strikes within a range of the target width
        target_long_put_strike = short_put_strike - adjusted_strike_width
        potential_long_puts = [s for s in strikes if s < short_put_strike]
        
        if not potential_long_puts:
            self.logger.warning("No potential long put strikes found")
            return None
            
        for strike in potential_long_puts:
            if abs(strike - target_long_put_strike) / target_long_put_strike > 0.2:
                # Skip strikes that are too far from target (more than 20% deviation)
                continue
                
            if strike in puts:
                put = puts[strike]
                bid = put.get('bid', 0)
                ask = put.get('ask', 0)
                volume = put.get('volume', 0) if put.get('volume') is not None else 0
                
                # Skip strikes with poor liquidity or zero bid
                if bid <= 0 or volume < 5:
                    continue
                    
                # Calculate a score based on proximity to target strike and liquidity
                proximity_score = 1.0 - abs(strike - target_long_put_strike) / adjusted_strike_width
                liquidity_score = min(1.0, volume / 200)
                
                # Calculate debit as percentage of credit from short put
                debit_pct = (ask / short_put['bid']) if short_put['bid'] > 0 else float('inf')
                
                # Prefer strikes that use less of the credit from the short leg
                cost_efficiency = max(0, 1.0 - debit_pct)
                
                score = proximity_score * 0.5 + liquidity_score * 0.2 + cost_efficiency * 0.3
                
                long_put_candidates.append({
                    'strike': strike,
                    'option': put,
                    'bid': bid,
                    'ask': ask,
                    'score': score,
                    'width': short_put_strike - strike
                })
        
        # If we can't find good long leg candidates, return None
        if not long_call_candidates or not long_put_candidates:
            self.logger.warning("Could not find appropriate long legs with good liquidity for iron condor")
            return None
            
        # Select best long call and put candidates based on score
        long_call_candidates.sort(key=lambda x: x['score'], reverse=True)
        long_put_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        long_call = long_call_candidates[0]
        long_put = long_put_candidates[0]
        
        long_call_strike = long_call['strike']
        long_put_strike = long_put['strike']
        
        # Calculate mid prices for the options
        # Use more conservative pricing - for credits, use bid prices; for debits, use ask prices
        short_call_price = short_call['bid']  # Use bid for credit
        short_put_price = short_put['bid']    # Use bid for credit
        long_call_price = long_call['ask']    # Use ask for debit
        long_put_price = long_put['ask']      # Use ask for debit
        
        # Calculate credit received - ensure positive
        net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
        
        # If net credit is too small, reject the iron condor
        min_credit_threshold = 0.20  # Minimum $0.20 credit
        if net_credit < min_credit_threshold:
            self.logger.warning(f"Iron condor credit {net_credit:.2f} is below minimum threshold {min_credit_threshold:.2f}")
            return None
        
        # Calculate max risk (width between strikes minus credit received)
        call_spread_width = long_call_strike - short_call_strike
        put_spread_width = short_put_strike - long_put_strike
        
        # Use smaller of the two widths for max risk calculation to be conservative
        max_width = min(call_spread_width, put_spread_width)
        max_risk = max_width - net_credit
        
        if max_risk <= 0:
            self.logger.warning("Invalid risk calculation for iron condor")
            return None
        
        # Calculate credit-to-risk ratio
        credit_to_risk_ratio = net_credit / max_width
        
        # Only proceed if credit-to-risk ratio meets minimum threshold
        min_credit_risk_ratio = 0.15  # Require at least 15% credit to width ratio
        if credit_to_risk_ratio < min_credit_risk_ratio:
            self.logger.warning(f"Iron condor credit-to-risk ratio {credit_to_risk_ratio:.2f} below threshold {min_credit_risk_ratio:.2f}")
            return None
        
        # Calculate risk metrics
        return_on_risk = net_credit / max_risk
        
        # Calculate probability of profit (approximate using delta)
        prob_profit = (1 - short_call_delta) * (1 - short_put_delta) * 100
        
        # Create iron condor details
        iron_condor = {
            'strategy': 'iron_condor',
            'expiry': target_expiry,
            'days_to_expiry': days_to_expiry,
            'short_call_strike': short_call_strike,
            'short_put_strike': short_put_strike,
            'long_call_strike': long_call_strike,
            'long_put_strike': long_put_strike,
            'short_call_price': short_call_price,
            'short_put_price': short_put_price,
            'long_call_price': long_call_price,
            'long_put_price': long_put_price,
            'net_credit': net_credit,
            'max_risk': max_risk,
            'max_width': max_width,
            'credit_to_risk_ratio': credit_to_risk_ratio,
            'return_on_risk': return_on_risk,
            'short_call_delta': short_call_delta,
            'short_put_delta': short_put_delta,
            'prob_profit': prob_profit
        }
        
        self.logger.info(f"Found iron condor: {target_expiry} expiry, "
                       f"short put @ {short_put_strike}, short call @ {short_call_strike}, "
                       f"credit: {net_credit:.2f}, RoR: {return_on_risk:.2f}, "
                       f"credit/risk ratio: {credit_to_risk_ratio:.2f}, prob profit: {prob_profit:.2f}%")
                       
        return iron_condor
        
    def find_vertical_spread_legs(
        self,
        option_chain: Dict,
        current_price: float,
        spread_type: str = 'bull_put',  # 'bull_put', 'bear_call', 'bull_call', 'bear_put'
        expiry_target: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Find appropriate legs for a vertical spread.
        
        Args:
            option_chain: Option chain data
            current_price: Current price of the underlying
            spread_type: Type of vertical spread to find
            expiry_target: Target expiry date (if None, will select based on DTE range)
            
        Returns:
            Dictionary with vertical spread leg details or None if no suitable legs found
        """
        if not option_chain:
            self.logger.warning("No option chain data provided")
            return None
            
        self.logger.info(f"Finding {spread_type} vertical spread legs at price {current_price}")
        
        # Find expiry date within target range - similar to iron condor but with different optimal DTE
        target_expiry = None
        if expiry_target and expiry_target in option_chain:
            target_expiry = expiry_target
        else:
            # Filter by DTE range - vertical spreads can work with slightly shorter DTEs
            valid_expiries = [
                exp for exp, data in option_chain.items()
                if self.min_dte <= data.get('days_to_expiry', 0) <= self.max_dte
            ]
            
            if valid_expiries:
                # Choose based on a combination of IV and optimal DTE
                best_expiry = None
                best_score = 0
                
                for exp in valid_expiries:
                    exp_data = option_chain[exp]
                    days_to_expiry = exp_data.get('days_to_expiry', 0)
                    
                    # Different optimal DTE ranges based on spread type
                    dte_score = 0
                    if spread_type in ['bull_put', 'bear_call']:  # Credit spreads
                        # Shorter-term is better for credit spreads to benefit from faster theta decay
                        if 18 <= days_to_expiry <= 30:
                            dte_score = 1.0  # Ideal range
                        elif 14 <= days_to_expiry < 18 or 30 < days_to_expiry <= 35:
                            dte_score = 0.8  # Good range
                        else:
                            dte_score = 0.6  # Acceptable range
                    else:  # Debit spreads
                        # Longer-term is better for debit spreads to allow time for directional move
                        if 25 <= days_to_expiry <= 45:
                            dte_score = 1.0  # Ideal range
                        elif 20 <= days_to_expiry < 25 or 45 < days_to_expiry <= 60:
                            dte_score = 0.8  # Good range
                        else:
                            dte_score = 0.6  # Acceptable range
                    
                    # Get option chain for this expiry
                    if spread_type.endswith('call'):
                        options = exp_data.get('calls', {})
                    else:  # put spreads
                        options = exp_data.get('puts', {})
                    
                    # Calculate average IV for this expiry's options
                    ivs = [opt.get('iv', 0) for opt in options.values() if opt.get('iv', 0) > 0]
                    avg_iv = sum(ivs) / len(ivs) if ivs else 0
                    
                    # Combine IV and DTE scores, weighting DTE more for verticala
                    total_score = (avg_iv * 0.3) + (dte_score * 0.7)
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_expiry = exp
                
                target_expiry = best_expiry
        
        if not target_expiry:
            self.logger.warning("No suitable expiry date found for vertical spread")
            return None
            
        # Get expiry data
        expiry_data = option_chain[target_expiry]
        days_to_expiry = expiry_data.get('days_to_expiry', 0)
        
        # Get appropriate option type based on spread type
        if spread_type.endswith('call'):
            options = expiry_data.get('calls', {})
            option_type = 'call'
        else:  # put spreads
            options = expiry_data.get('puts', {})
            option_type = 'put'
        
        # Get available strikes
        strikes = sorted(list(options.keys()))
        if not strikes:
            self.logger.warning(f"No valid {option_type} strikes found")
            return None
        
        # Find ATM strike
        atm_strike = min(strikes, key=lambda x: abs(x - current_price))
        atm_index = strikes.index(atm_strike)
        
        # Different logic based on spread type
        if spread_type == 'bull_put':  # Sell higher put, buy lower put (credit spread)
            # For bull put spread, we want to find strikes below the current price
            # that have reasonable premium and liquidity
            
            # First find the short leg (higher strike, closer to ATM)
            short_candidates = []
            
            # Look at strikes below current price for put spreads
            for i in range(atm_index, -1, -1):
                strike = strikes[i]
                if strike >= current_price:
                    continue  # Skip ITM puts
                
                option = options[strike]
                delta = abs(option.get('delta', 0))
                bid = option.get('bid', 0)
                ask = option.get('ask', 0)
                volume = option.get('volume', 0) if option.get('volume') is not None else 0
                
                # For bull put, we want short strike with delta around 0.20-0.35
                if 0.15 <= delta <= 0.35 and bid >= 0.15 and volume >= 10:
                    spread_pct = (ask - bid) / bid if bid > 0 else float('inf')
                    
                    # Calculate a score based on premium and liquidity
                    premium_score = min(1.0, bid / 2.0)  # Scale to max 1.0 at premium of $2.00
                    liquidity_score = min(1.0, volume / 300)
                    delta_score = 1.0 - abs(delta - 0.25) / 0.25  # Optimal delta around 0.25
                    
                    score = premium_score * 0.5 + liquidity_score * 0.3 + delta_score * 0.2
                    
                    short_candidates.append({
                        'strike': strike,
                        'delta': delta,
                        'bid': bid,
                        'ask': ask,
                        'option': option,
                        'score': score
                    })
            
            if not short_candidates:
                self.logger.warning("No suitable short leg found for bull put spread")
                return None
            
            # Select best short candidate
            short_candidates.sort(key=lambda x: x['score'], reverse=True)
            short_leg = short_candidates[0]
            short_strike = short_leg['strike']
            
            # Now find the long leg (lower strike)
            long_candidates = []
            
            # Target width based on risk management
            # For vertical spreads, we typically want narrower width than iron condors
            target_width_pct = 4.0  # 4% of underlying price
            target_width = current_price * (target_width_pct / 100)
            target_long_strike = short_strike - target_width
            
            # Find long leg strikes (must be below short strike)
            for strike in [s for s in strikes if s < short_strike]:
                option = options[strike]
                delta = abs(option.get('delta', 0))
                bid = option.get('bid', 0)
                ask = option.get('ask', 0)
                
                # Calculate width and cost efficiency
                width = short_strike - strike
                
                # Debit to pay for long leg
                long_cost = ask
                
                # Net credit for the spread
                net_credit = short_leg['bid'] - long_cost
                
                # Credit-to-width ratio
                credit_width_ratio = net_credit / width if width > 0 else 0
                
                # Calculate a score based on width and credit
                width_score = 1.0 - abs(width - target_width) / target_width if target_width > 0 else 0
                credit_score = min(1.0, credit_width_ratio * 5)  # Scale to max 1.0 at 20% credit-to-width
                
                score = width_score * 0.4 + credit_score * 0.6
                
                long_candidates.append({
                    'strike': strike,
                    'delta': delta,
                    'bid': bid,
                    'ask': ask,
                    'option': option,
                    'width': width,
                    'net_credit': net_credit,
                    'credit_width_ratio': credit_width_ratio,
                    'score': score
                })
            
            if not long_candidates:
                self.logger.warning("No suitable long leg found for bull put spread")
                return None
            
            # Select best long candidate
            long_candidates.sort(key=lambda x: x['score'], reverse=True)
            long_leg = long_candidates[0]
            long_strike = long_leg['strike']
            
            # Calculate spread details
            width = short_strike - long_strike
            net_credit = short_leg['bid'] - long_leg['ask']  # Use bid-ask for conservative estimate
            max_risk = width - net_credit
            
            # Probability metrics (approximate)
            prob_profit = (1 - short_leg['delta']) * 100  # Probability OTM at expiration
            
            # Return spread details
            spread = {
                'strategy': 'vertical_spread',
                'spread_type': spread_type,
                'option_type': option_type,
                'expiry': target_expiry,
                'days_to_expiry': days_to_expiry,
                'short_strike': short_strike,
                'long_strike': long_strike,
                'short_price': short_leg['bid'],  # Credit received
                'long_price': long_leg['ask'],    # Debit paid
                'width': width,
                'net_credit': net_credit,
                'max_risk': max_risk * 100,  # Per contract (multiply by 100)
                'max_profit': net_credit * 100,  # Per contract
                'credit_to_width_ratio': net_credit / width if width > 0 else 0,
                'return_on_risk': net_credit / max_risk if max_risk > 0 else 0,
                'prob_profit': prob_profit
            }
            
            self.logger.info(f"Found {spread_type} spread: {target_expiry} expiry, "
                           f"short {option_type} @ {short_strike}, long {option_type} @ {long_strike}, "
                           f"width: {width}, credit: {net_credit:.2f}, max risk: {max_risk * 100:.2f}")
            
            return spread
            
        elif spread_type == 'bear_call':  # Sell lower call, buy higher call (credit spread)
            # For bear call spread, we want to find strikes above the current price
            # that have reasonable premium and liquidity
            
            # First find the short leg (lower strike, closer to ATM)
            short_candidates = []
            
            # Look at strikes above current price for call spreads
            for i in range(atm_index, len(strikes)):
                strike = strikes[i]
                if strike <= current_price:
                    continue  # Skip ITM calls
                
                option = options[strike]
                delta = abs(option.get('delta', 0))
                bid = option.get('bid', 0)
                ask = option.get('ask', 0)
                volume = option.get('volume', 0) if option.get('volume') is not None else 0
                
                # For bear call, we want short strike with delta around 0.20-0.35
                if 0.15 <= delta <= 0.35 and bid >= 0.15 and volume >= 10:
                    spread_pct = (ask - bid) / bid if bid > 0 else float('inf')
                    
                    # Calculate a score based on premium and liquidity
                    premium_score = min(1.0, bid / 2.0)  # Scale to max 1.0 at premium of $2.00
                    liquidity_score = min(1.0, volume / 300)
                    delta_score = 1.0 - abs(delta - 0.25) / 0.25  # Optimal delta around 0.25
                    
                    score = premium_score * 0.5 + liquidity_score * 0.3 + delta_score * 0.2
                    
                    short_candidates.append({
                        'strike': strike,
                        'delta': delta,
                        'bid': bid,
                        'ask': ask,
                        'option': option,
                        'score': score
                    })
            
            if not short_candidates:
                self.logger.warning("No suitable short leg found for bear call spread")
                return None
            
            # Select best short candidate
            short_candidates.sort(key=lambda x: x['score'], reverse=True)
            short_leg = short_candidates[0]
            short_strike = short_leg['strike']
            
            # Now find the long leg (higher strike)
            long_candidates = []
            
            # Target width based on risk management
            target_width_pct = 4.0  # 4% of underlying price
            target_width = current_price * (target_width_pct / 100)
            target_long_strike = short_strike + target_width
            
            # Find long leg strikes (must be above short strike)
            for strike in [s for s in strikes if s > short_strike]:
                option = options[strike]
                delta = abs(option.get('delta', 0))
                bid = option.get('bid', 0)
                ask = option.get('ask', 0)
                
                # Calculate width and cost efficiency
                width = strike - short_strike
                
                # Debit to pay for long leg
                long_cost = ask
                
                # Net credit for the spread
                net_credit = short_leg['bid'] - long_cost
                
                # Credit-to-width ratio
                credit_width_ratio = net_credit / width if width > 0 else 0
                
                # Calculate a score based on width and credit
                width_score = 1.0 - abs(width - target_width) / target_width if target_width > 0 else 0
                credit_score = min(1.0, credit_width_ratio * 5)  # Scale to max 1.0 at 20% credit-to-width
                
                score = width_score * 0.4 + credit_score * 0.6
                
                long_candidates.append({
                    'strike': strike,
                    'delta': delta,
                    'bid': bid,
                    'ask': ask,
                    'option': option,
                    'width': width,
                    'net_credit': net_credit,
                    'credit_width_ratio': credit_width_ratio,
                    'score': score
                })
            
            if not long_candidates:
                self.logger.warning("No suitable long leg found for bear call spread")
                return None
            
            # Select best long candidate
            long_candidates.sort(key=lambda x: x['score'], reverse=True)
            long_leg = long_candidates[0]
            long_strike = long_leg['strike']
            
            # Calculate spread details
            width = long_strike - short_strike
            net_credit = short_leg['bid'] - long_leg['ask']  # Use bid-ask for conservative estimate
            max_risk = width - net_credit
            
            # Probability metrics (approximate)
            prob_profit = (1 - short_leg['delta']) * 100  # Probability OTM at expiration
            
            # Return spread details
            spread = {
                'strategy': 'vertical_spread',
                'spread_type': spread_type,
                'option_type': option_type,
                'expiry': target_expiry,
                'days_to_expiry': days_to_expiry,
                'short_strike': short_strike,
                'long_strike': long_strike,
                'short_price': short_leg['bid'],  # Credit received
                'long_price': long_leg['ask'],    # Debit paid
                'width': width,
                'net_credit': net_credit,
                'max_risk': max_risk * 100,  # Per contract (multiply by 100)
                'max_profit': net_credit * 100,  # Per contract
                'credit_to_width_ratio': net_credit / width if width > 0 else 0,
                'return_on_risk': net_credit / max_risk if max_risk > 0 else 0,
                'prob_profit': prob_profit
            }
            
            self.logger.info(f"Found {spread_type} spread: {target_expiry} expiry, "
                           f"short {option_type} @ {short_strike}, long {option_type} @ {long_strike}, "
                           f"width: {width}, credit: {net_credit:.2f}, max risk: {max_risk * 100:.2f}")
            
            return spread
        
        # TODO: Add logic for debit spreads (bull_call and bear_put) if needed
        
        return None
        
    def generate_vertical_spread_trade(
        self, 
        option_chain: Dict, 
        current_price: float,
        account_value: float,
        vix_analysis: Dict,
        spread_type: str = 'bull_put'  # 'bull_put', 'bear_call'
    ) -> Dict:
        """
        Generate a vertical spread trade decision.
        
        Args:
            option_chain: Option chain data
            current_price: Current price of the underlying
            account_value: Current account value
            vix_analysis: VIX analysis data
            spread_type: Type of vertical spread to generate
            
        Returns:
            Dictionary with trade decision
        """
        # Find vertical spread legs
        vertical_spread = self.find_vertical_spread_legs(option_chain, current_price, spread_type)
        
        if not vertical_spread:
            self.logger.info(f"No suitable {spread_type} vertical spread found")
            return {
                'action': 'none',
                'reason': f'no_suitable_{spread_type}_spread'
            }
        
        # Get position size multiplier
        vol_harvest_analysis = self.volatility_analyzer.analyze_volatility_for_harvesting(
            self.index_symbol, vix_analysis, option_chain
        )
        
        # More conservative position sizing for vertical spreads
        position_size_multiplier = 0.0
        if vol_harvest_analysis['signal'] == 'strong_volatility_harvest':
            position_size_multiplier = 0.8  # High volatility good for premium collection
        elif vol_harvest_analysis['signal'] == 'volatility_harvest':
            position_size_multiplier = 0.6
        else:
            position_size_multiplier = 0.4
            
        # Additional checks for vertical spreads
        # Minimum credit requirements (different for each spread type)
        min_credit = 0.20  # Minimum $0.20 credit per spread
        if vertical_spread['net_credit'] < min_credit:
            self.logger.info(f"Vertical spread net credit too low: {vertical_spread['net_credit']:.2f} < {min_credit:.2f}")
            return {
                'action': 'none',
                'reason': 'insufficient_credit',
                'net_credit': vertical_spread['net_credit'],
                'threshold': min_credit
            }
            
        # Minimum return on risk
        min_return_on_risk = 0.15  # At least 15% return on risk
        if vertical_spread['return_on_risk'] < min_return_on_risk:
            self.logger.info(f"Vertical spread return on risk too low: {vertical_spread['return_on_risk']:.2f} < {min_return_on_risk:.2f}")
            return {
                'action': 'none',
                'reason': 'insufficient_return_on_risk',
                'return_on_risk': vertical_spread['return_on_risk'],
                'threshold': min_return_on_risk
            }
            
        # Minimum probability of profit
        min_prob_profit = 60.0  # At least 60% probability of profit
        if vertical_spread['prob_profit'] < min_prob_profit:
            self.logger.info(f"Vertical spread probability of profit too low: {vertical_spread['prob_profit']:.2f}% < {min_prob_profit:.2f}%")
            return {
                'action': 'none',
                'reason': 'insufficient_probability',
                'prob_profit': vertical_spread['prob_profit'],
                'threshold': min_prob_profit
            }
            
        # Calculate quantity based on risk management
        # For vertical spreads, max loss is width - credit received
        max_risk_per_spread = vertical_spread['max_risk']  # Already in dollars (width - credit) * 100
        
        # Risk allocation - use a percentage of the max daily risk
        # More conservative allocation for vertical spreads
        risk_allocation_pct = 0.7  # Use 70% of max daily risk for vertical spreads
        max_risk_amount = account_value * (self.risk_manager.risk_params.max_daily_risk_pct / 100.0) * position_size_multiplier * risk_allocation_pct
        
        # Calculate quantity
        quantity = int(max_risk_amount / max_risk_per_spread) if max_risk_per_spread > 0 else 1
        quantity = max(1, quantity)  # At least 1 contract
        
        # Additional cap based on account size
        max_quantity_by_account = int(account_value * 0.005 / max_risk_per_spread)
        quantity = min(quantity, max_quantity_by_account)
        
        # Absolute maximum contracts
        absolute_max_contracts = 10
        quantity = min(quantity, absolute_max_contracts)
        
        # Calculate total credit and max loss
        total_credit = vertical_spread['net_credit'] * quantity * 100
        max_loss = max_risk_per_spread * quantity
        
        # Build trade decision
        trade_decision = {
            'action': 'vertical_spread',
            'spread_type': spread_type,
            'symbol': self.index_symbol,
            'option_type': vertical_spread['option_type'],
            'expiry': vertical_spread['expiry'],
            'short_strike': vertical_spread['short_strike'],
            'long_strike': vertical_spread['long_strike'],
            'quantity': quantity,
            'net_credit': vertical_spread['net_credit'],
            'total_credit': total_credit,
            'width': vertical_spread['width'],
            'max_risk_per_spread': max_risk_per_spread,
            'max_loss': max_loss,
            'return_on_risk': vertical_spread['return_on_risk'],
            'credit_to_width_ratio': vertical_spread['credit_to_width_ratio'],
            'prob_profit': vertical_spread['prob_profit'],
            'days_to_expiry': vertical_spread['days_to_expiry'],
            'vix': vix_analysis.get('current_vix', 0.0),
            'iv_hv_ratio': vol_harvest_analysis.get('iv_hv_ratio', 0.0),
            'position_size_multiplier': position_size_multiplier,
            'reason': vol_harvest_analysis.get('signal', 'volatility_harvest')
        }
        
        self.logger.info(f"Generated {spread_type} spread: {quantity} contracts, "
                       f"width: {vertical_spread['width']}, "
                       f"credit: ${vertical_spread['net_credit']:.2f}, "
                       f"total credit: ${total_credit:.2f}, "
                       f"max loss: ${max_loss:.2f}, "
                       f"prob profit: {vertical_spread['prob_profit']:.2f}%")
                       
        self.last_trade_time = datetime.now()
        self.todays_trades.append(trade_decision)
        
        return trade_decision
    
    def generate_iron_condor_trade(
        self, 
        option_chain: Dict, 
        current_price: float,
        account_value: float,
        vix_analysis: Dict,
        min_credit: float = 0.10,
        min_return_on_risk: float = 8.0
    ) -> Dict:
        """
        Generate an iron condor trade decision.
        
        Args:
            option_chain: Option chain data
            current_price: Current price of the underlying
            account_value: Current account value
            vix_analysis: VIX analysis data
            min_credit: Minimum credit required per spread
            min_return_on_risk: Minimum return on risk percentage
            
        Returns:
            Dictionary with trade decision
        """
        # Find iron condor legs
        iron_condor = self.find_iron_condor_legs(
            option_chain, 
            current_price,
            min_credit=min_credit,
            min_return_on_risk=min_return_on_risk
        )
        
        if not iron_condor:
            self.logger.info("No suitable iron condor found")
            return {
                'action': 'none',
                'reason': 'no_suitable_iron_condor'
            }
        
        # Get position size multiplier
        vol_harvest_analysis = self.volatility_analyzer.analyze_volatility_for_harvesting(
            self.index_symbol, vix_analysis, option_chain
        )
        
        # More conservative position sizing with further reduction
        position_size_multiplier = 0.0
        if vol_harvest_analysis['signal'] == 'strong_volatility_harvest':
            position_size_multiplier = 0.6  # Reduced from 0.7
        elif vol_harvest_analysis['signal'] == 'volatility_harvest':
            position_size_multiplier = 0.35  # Reduced from 0.4
        else:
            position_size_multiplier = 0.15  # Reduced from 0.2
        
        # Apply an additional safety factor based on the volatility regime
        volatility_state = vix_analysis.get('volatility_state', 'normal')
        if volatility_state == 'extreme':
            position_size_multiplier *= 0.5  # Further reduce by 50% in extreme volatility
        elif volatility_state == 'high':
            position_size_multiplier *= 0.7  # Further reduce by 30% in high volatility
        
        # Additional check for return on risk - higher threshold for better risk/reward
        if iron_condor['return_on_risk'] < min_return_on_risk:
            self.logger.info(f"Iron condor return on risk too low: {iron_condor['return_on_risk']:.2f}% < {min_return_on_risk:.2f}%")
            return {
                'action': 'none',
                'reason': 'insufficient_return_on_risk',
                'return_on_risk': iron_condor['return_on_risk'],
                'threshold': min_return_on_risk
            }
            
        # Additional check for minimum credit - ensure enough premium is collected
        if iron_condor['net_credit'] < min_credit:
            self.logger.info(f"Iron condor net credit too low: {iron_condor['net_credit']:.2f} < {min_credit:.2f}")
            return {
                'action': 'none',
                'reason': 'insufficient_credit',
                'net_credit': iron_condor['net_credit'],
                'threshold': min_credit
            }
        
        # Check for credit-to-risk ratio - ensure the credit is a meaningful percentage of the risk
        min_credit_risk_ratio = 0.15  # Minimum 15% of width
        current_credit_risk_ratio = iron_condor.get('credit_to_risk_ratio', 0)
        
        # Adjust minimum credit risk ratio based on volatility state
        volatility_state = vix_analysis.get('volatility_state', 'normal')
        if volatility_state == 'extreme':
            min_credit_risk_ratio = 0.20  # Require higher ratio in extreme volatility
        elif volatility_state == 'high':
            min_credit_risk_ratio = 0.17  # Require higher ratio in high volatility
            
        if current_credit_risk_ratio < min_credit_risk_ratio:
            self.logger.info(f"Iron condor credit-to-risk ratio too low: {current_credit_risk_ratio:.2f} < {min_credit_risk_ratio:.2f}")
            return {
                'action': 'none',
                'reason': 'insufficient_credit_risk_ratio',
                'credit_risk_ratio': current_credit_risk_ratio,
                'threshold': min_credit_risk_ratio
            }
            
        # Check probability of profit - ensure reasonable odds of success
        min_prob_profit = 65  # Minimum 65% probability of profit
        if iron_condor.get('prob_profit', 0) < min_prob_profit:
            self.logger.info(f"Iron condor probability of profit too low: {iron_condor.get('prob_profit', 0):.2f}% < {min_prob_profit:.2f}%")
            return {
                'action': 'none',
                'reason': 'insufficient_probability_of_profit',
                'prob_profit': iron_condor.get('prob_profit', 0),
                'threshold': min_prob_profit
            }
        
        # Risk only what we're willing to lose (max risk * quantity)
        # More conservative risk allocation - additional 0.7 factor (further reduced from 0.8)
        max_risk_amount = account_value * (self.risk_manager.risk_params.max_daily_risk_pct / 100.0) * position_size_multiplier * 0.7
        
        # Calculate quantity
        max_risk_per_spread = iron_condor['max_risk'] * 100  # Convert to dollars (per contract)
        quantity = int(max_risk_amount / max_risk_per_spread) if max_risk_per_spread > 0 else 1
        quantity = max(1, quantity)  # At least 1 contract
        
        # Additional cap on quantity based on account size - reduced from 0.005 to 0.004
        max_quantity_by_account = int(account_value * 0.004 / max_risk_per_spread)
        quantity = min(quantity, max_quantity_by_account)
        
        # Absolute maximum contracts for risk control
        absolute_max_contracts = 8  # Reduced from default
        quantity = min(quantity, absolute_max_contracts)
        
        # Calculate total credit and max loss
        total_credit = iron_condor['net_credit'] * quantity * 100
        max_loss = max_risk_per_spread * quantity
        
        # Build enhanced trade decision with more details
        trade_decision = {
            'action': 'iron_condor',
            'symbol': self.index_symbol,
            'expiry': iron_condor['expiry'],
            'short_call_strike': iron_condor['short_call_strike'],
            'short_put_strike': iron_condor['short_put_strike'],
            'long_call_strike': iron_condor['long_call_strike'],
            'long_put_strike': iron_condor['long_put_strike'],
            'quantity': quantity,
            'net_credit': iron_condor['net_credit'],
            'total_credit': total_credit,
            'max_risk_per_spread': max_risk_per_spread,
            'max_loss': max_loss,
            'return_on_risk': iron_condor['return_on_risk'],
            'credit_to_risk_ratio': iron_condor.get('credit_to_risk_ratio', 0),
            'prob_profit': iron_condor.get('prob_profit', 0),
            'days_to_expiry': iron_condor['days_to_expiry'],
            'short_call_delta': iron_condor['short_call_delta'],
            'short_put_delta': iron_condor['short_put_delta'],
            'vix': vix_analysis.get('current_vix', 0.0),
            'iv_hv_ratio': vol_harvest_analysis.get('iv_hv_ratio', 0.0),
            'position_size_multiplier': position_size_multiplier,
            'reason': vol_harvest_analysis.get('signal', 'volatility_harvest')
        }
        
        self.logger.info(f"Generated iron condor trade: {quantity} contracts, "
                       f"width: {iron_condor['long_call_strike'] - iron_condor['short_call_strike']}, "
                       f"credit: {iron_condor['net_credit']:.2f}, "
                       f"total credit: ${total_credit:.2f}, "
                       f"max risk: ${max_loss:.2f}, "
                       f"prob profit: {iron_condor.get('prob_profit', 0):.2f}%")
                       
        self.last_trade_time = datetime.now()
        self.todays_trades.append(trade_decision)
        
        return trade_decision


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