"""
Volatility analyzer for the TuringTrader algorithm.
This module provides tools for analyzing market volatility
and determining trading decisions based on volatility levels.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Config
try:
    from .market_data_store import MarketDataStore
except ImportError:
    MarketDataStore = None


class VolatilityAnalyzer:
    """
    Analyzer for market volatility to determine trading actions.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the volatility analyzer.
        
        Args:
            config: Configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or Config()
        
        # Volatility thresholds
        self.min_volatility_threshold = self.config.risk.min_volatility_threshold
        self.min_volatility_change = self.config.risk.min_volatility_change
        
        # Load volatility harvesting configurations
        self.iv_hv_ratio_threshold = self.config.vol_harvesting.iv_hv_ratio_threshold
        self.min_iv_threshold = self.config.vol_harvesting.min_iv_threshold
        self.use_adaptive_thresholds = self.config.vol_harvesting.use_adaptive_thresholds
        
        # Historical data
        self.historical_volatility = []
        self.vix_history = []
        self.iv_hv_ratios = []
        
        # Initialize market data store if available
        self.market_data_store = MarketDataStore(config) if MarketDataStore else None
        
        # Initialize adaptive thresholds
        self.adaptive_iv_threshold = self.min_iv_threshold
        self.adaptive_iv_hv_ratio = self.iv_hv_ratio_threshold
        
    def calculate_historical_volatility(self, prices: List[float], 
                                      window: int = 20, 
                                      trading_days: int = 252) -> float:
        """
        Calculate historical volatility based on price history.
        
        Args:
            prices: List of closing prices
            window: Rolling window size for calculation
            trading_days: Number of trading days in a year
            
        Returns:
            float: Annualized volatility as a percentage
        """
        if len(prices) < window + 1:
            self.logger.warning(f"Insufficient price data for volatility calculation. "
                           f"Need at least {window + 1} points, got {len(prices)}")
            return 0.0
        
        try:
            # Convert to numpy array
            price_array = np.array(prices)
            
            # Calculate daily returns
            returns = np.diff(np.log(price_array))
            
            # Calculate rolling standard deviation
            if len(returns) >= window:
                vol = np.std(returns[-window:])
                
                # Annualize the volatility
                annualized_vol = vol * math.sqrt(trading_days) * 100
                
                self.historical_volatility.append(annualized_vol)
                return annualized_vol
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility: {e}")
            return 0.0
    
    def calculate_implied_volatility(self, 
                                   option_price: float,
                                   strike_price: float,
                                   current_price: float,
                                   time_to_expiry: float,  # in years
                                   risk_free_rate: float = 0.03,
                                   is_call: bool = True) -> float:
        """
        Calculate implied volatility using the Black-Scholes model.
        
        Args:
            option_price: Current option price
            strike_price: Option strike price
            current_price: Current underlying price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            is_call: Whether the option is a call (True) or put (False)
            
        Returns:
            float: Implied volatility as a percentage
        """
        if option_price <= 0 or time_to_expiry <= 0:
            return 0.0
            
        try:
            # Initial volatility guess
            sigma = 0.3
            
            # Maximum iterations and precision
            max_iterations = 100
            precision = 0.00001
            
            for i in range(max_iterations):
                # Black-Scholes option price
                if is_call:
                    price = self._black_scholes(current_price, strike_price, time_to_expiry, 
                                             risk_free_rate, sigma, True)
                else:
                    price = self._black_scholes(current_price, strike_price, time_to_expiry, 
                                             risk_free_rate, sigma, False)
                
                # Vega calculation
                d1 = (math.log(current_price / strike_price) + 
                     (risk_free_rate + 0.5 * sigma**2) * time_to_expiry) / (sigma * math.sqrt(time_to_expiry))
                vega = current_price * math.sqrt(time_to_expiry) * norm.pdf(d1)
                
                # Update volatility estimate
                diff = option_price - price
                if abs(diff) < precision:
                    return sigma * 100  # Convert to percentage
                
                # Update sigma
                sigma = sigma + diff / vega
                
                # Check for non-convergence
                if sigma <= 0.001:
                    return 0.0
                if sigma > 5:  # Cap at 500% volatility
                    return 500.0
            
            # If we didn't converge, return our best guess
            return sigma * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {e}")
            return 0.0
    
    def _black_scholes(self, 
                     s: float, 
                     k: float, 
                     t: float, 
                     r: float, 
                     sigma: float, 
                     is_call: bool) -> float:
        """
        Calculate option price using Black-Scholes model.
        
        Args:
            s: Current stock price
            k: Strike price
            t: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility
            is_call: Whether the option is a call (True) or put (False)
            
        Returns:
            float: Option price
        """
        if t <= 0:
            # For expired options, return intrinsic value
            if is_call:
                return max(0, s - k)
            else:
                return max(0, k - s)
                
        d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        
        if is_call:
            price = s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
        else:
            price = k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
            
        return price
    
    def analyze_vix(self, vix_data: List[Dict]) -> Dict:
        """
        Analyze VIX data to determine market volatility state.
        
        Args:
            vix_data: List of VIX data points with 'date' and 'close' values
            
        Returns:
            Dictionary with volatility analysis
        """
        if not vix_data:
            self.logger.warning("No VIX data provided for analysis")
            return {
                'current_vix': 0.0,
                'vix_change_1d': 0.0,
                'vix_change_5d': 0.0,
                'volatility_state': 'unknown',
                'signal': 'none',
                'vix_regime': 'unknown',
                'vix_trend': 'flat'
            }
            
        try:
            # Extract VIX closing values
            vix_values = [item['close'] for item in vix_data]
            
            # Store in history
            self.vix_history.extend(vix_values)
            
            # Keep only the most recent 100 values
            if len(self.vix_history) > 100:
                self.vix_history = self.vix_history[-100:]
            
            # Current VIX
            current_vix = vix_values[-1]
            
            # Calculate changes
            vix_change_1d = 0.0 if len(vix_values) < 2 else current_vix - vix_values[-2]
            vix_change_5d = 0.0 if len(vix_values) < 6 else current_vix - vix_values[-6]
            vix_change_10d = 0.0 if len(vix_values) < 11 else current_vix - vix_values[-11]
            
            vix_change_pct_1d = 0.0 if len(vix_values) < 2 else 100 * (vix_change_1d / vix_values[-2])
            vix_change_pct_5d = 0.0 if len(vix_values) < 6 else 100 * (vix_change_5d / vix_values[-6])
            vix_change_pct_10d = 0.0 if len(vix_values) < 11 else 100 * (vix_change_10d / vix_values[-11])
            
            # Calculate simple moving averages
            vix_sma_5 = np.mean(vix_values[-5:]) if len(vix_values) >= 5 else current_vix
            vix_sma_10 = np.mean(vix_values[-10:]) if len(vix_values) >= 10 else current_vix
            vix_sma_20 = np.mean(vix_values[-20:]) if len(vix_values) >= 20 else current_vix
            
            # Enhanced volatility state determination
            if current_vix > 35:
                volatility_state = 'extreme'
            elif current_vix > 25:
                volatility_state = 'high'
            elif current_vix > 18:
                volatility_state = 'normal'
            elif current_vix > 13:
                volatility_state = 'low'
            else:
                volatility_state = 'very_low'
            
            # Determine VIX regime (longer-term view)
            if vix_sma_20 > 30:
                vix_regime = 'crisis'
            elif vix_sma_20 > 22:
                vix_regime = 'high_volatility'
            elif vix_sma_20 > 16:
                vix_regime = 'normal'
            else:
                vix_regime = 'low_volatility'
                
            # Determine VIX trend
            # Look at both short-term and medium-term trends
            if vix_sma_5 > vix_sma_10 and vix_change_pct_5d > 8:
                vix_trend = 'strong_up'
            elif vix_sma_5 > vix_sma_10:
                vix_trend = 'up'
            elif vix_sma_5 < vix_sma_10 and vix_change_pct_5d < -8:
                vix_trend = 'strong_down'
            elif vix_sma_5 < vix_sma_10:
                vix_trend = 'down'
            else:
                vix_trend = 'flat'
            
            # Enhanced signal determination based on VIX state, regime, and trend
            signal = 'none'
            
            # Base signal on volatility level, trend, and the relationship between short and longer-term VIX
            if volatility_state in ('extreme', 'high'):
                # In high volatility environments, we look for stabilization or decline
                if vix_trend == 'strong_down':
                    signal = 'strong_buy'  # Volatility dropping rapidly - good time to sell premium
                elif vix_trend == 'down':
                    signal = 'buy'  # Volatility declining - favorable for selling premium
                elif vix_trend == 'flat' and current_vix > vix_sma_10:
                    signal = 'hold'  # Elevated but stable volatility - cautious premium selling
                else:
                    signal = 'cash'  # Volatility still increasing - stay in cash
                    
            elif volatility_state == 'normal':
                # In normal volatility, we want confirmed trends
                if vix_trend in ('up', 'strong_up') and vix_change_pct_1d > 5:
                    signal = 'buy'  # Volatility increasing from normal levels - good for premium selling
                elif vix_trend == 'flat' and current_vix >= self.min_volatility_threshold:
                    signal = 'hold'  # Stable, sufficient volatility - cautious premium selling
                else:
                    signal = 'cash'  # Not enough volatility movement to justify trading
                    
            elif volatility_state in ('low', 'very_low'):
                # In low volatility, we need strong upward moves
                if vix_trend == 'strong_up' and vix_change_pct_1d > 8:
                    signal = 'buy'  # Sharp volatility spike from low levels - opportunity for premium selling
                elif vix_trend == 'up' and vix_change_pct_5d > 15:
                    signal = 'hold'  # Sustained volatility increase - cautious premium selling
                else:
                    signal = 'cash'  # Volatility too low for effective premium selling
            
            # Return enhanced analysis
            return {
                'current_vix': current_vix,
                'vix_change_1d': vix_change_1d,
                'vix_change_pct_1d': vix_change_pct_1d,
                'vix_change_5d': vix_change_5d,
                'vix_change_pct_5d': vix_change_pct_5d,
                'vix_change_10d': vix_change_10d,
                'vix_change_pct_10d': vix_change_pct_10d,
                'vix_sma_5': vix_sma_5,
                'vix_sma_10': vix_sma_10,
                'vix_sma_20': vix_sma_20,
                'volatility_state': volatility_state,
                'vix_regime': vix_regime,
                'vix_trend': vix_trend,
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing VIX data: {e}")
            return {
                'current_vix': 0.0,
                'vix_change_1d': 0.0,
                'vix_change_5d': 0.0,
                'volatility_state': 'error',
                'signal': 'none',
                'error': str(e)
            }
    
    def calculate_probability_of_profit(self, 
                                  current_price: float, 
                                  strike_price: float, 
                                  days_to_expiry: float, 
                                  volatility: float, 
                                  option_type: str = 'call',
                                  position_type: str = 'short') -> float:
        """
        Calculate probability of profit for an option trade based on volatility.
        
        Args:
            current_price: Current price of the underlying
            strike_price: Strike price of the option
            days_to_expiry: Days to expiration
            volatility: Implied volatility (decimal)
            option_type: 'call' or 'put'
            position_type: 'long' or 'short'
        
        Returns:
            float: Probability of profit as a percentage (0-100)
        """
        if days_to_expiry <= 0 or volatility <= 0:
            return 0.0
            
        try:
            # Convert inputs to the right format
            s = current_price
            k = strike_price
            t = days_to_expiry / 365.0  # Convert to years
            sigma = volatility  # Should be in decimal format (e.g., 0.25 for 25%)
            
            # Calculate probability based on log-normal distribution
            if option_type.lower() == 'call':
                if position_type.lower() == 'long':
                    # For long calls, need price to rise above strike + premium
                    # This is simplified and doesn't account for premium
                    ln_ratio = math.log(s / k)
                    probability = norm.cdf((ln_ratio + (sigma**2 / 2) * t) / (sigma * math.sqrt(t)))
                    return probability * 100
                else:  # short call
                    # For short calls, need price to stay below strike
                    ln_ratio = math.log(s / k)
                    probability = norm.cdf(-(ln_ratio + (sigma**2 / 2) * t) / (sigma * math.sqrt(t)))
                    return probability * 100
            else:  # put
                if position_type.lower() == 'long':
                    # For long puts, need price to fall below strike - premium
                    # This is simplified and doesn't account for premium
                    ln_ratio = math.log(s / k)
                    probability = norm.cdf(-(ln_ratio + (sigma**2 / 2) * t) / (sigma * math.sqrt(t)))
                    return probability * 100
                else:  # short put
                    # For short puts, need price to stay above strike
                    ln_ratio = math.log(s / k)
                    probability = norm.cdf((ln_ratio + (sigma**2 / 2) * t) / (sigma * math.sqrt(t)))
                    return probability * 100
                    
        except Exception as e:
            self.logger.error(f"Error calculating probability of profit: {e}")
            return 0.0
            
    def calculate_probability_of_profit_spread(self,
                                           current_price: float,
                                           short_strike: float,
                                           long_strike: float,
                                           days_to_expiry: float,
                                           volatility: float,
                                           spread_type: str) -> float:
        """
        Calculate probability of profit for a vertical spread or iron condor.
        
        Args:
            current_price: Current price of the underlying
            short_strike: Strike price of the short option
            long_strike: Strike price of the long option
            days_to_expiry: Days to expiration
            volatility: Implied volatility (decimal)
            spread_type: 'bull_put', 'bear_call', or 'iron_condor'
        
        Returns:
            float: Probability of profit as a percentage (0-100)
        """
        if days_to_expiry <= 0 or volatility <= 0:
            return 0.0
            
        try:
            if spread_type == 'bull_put':
                # For bull put spread, price needs to stay above short put strike
                return self.calculate_probability_of_profit(
                    current_price, short_strike, days_to_expiry, volatility, 'put', 'short')
                    
            elif spread_type == 'bear_call':
                # For bear call spread, price needs to stay below short call strike
                return self.calculate_probability_of_profit(
                    current_price, short_strike, days_to_expiry, volatility, 'call', 'short')
                    
            elif spread_type == 'iron_condor':
                # For iron condor, price needs to stay between short put and short call
                put_prob = self.calculate_probability_of_profit(
                    current_price, short_strike, days_to_expiry, volatility, 'put', 'short')
                call_prob = self.calculate_probability_of_profit(
                    current_price, long_strike, days_to_expiry, volatility, 'call', 'short')
                
                # The combined probability - this is a simplification as it doesn't account for correlation
                # In reality, these aren't independent events
                combined_prob = put_prob * call_prob / 100
                return combined_prob
                
            else:
                self.logger.warning(f"Unknown spread type: {spread_type}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating probability of profit for spread: {e}")
            return 0.0
        """
        Calculate a quality score for an option opportunity.
        Higher score means better opportunity.
        
        Args:
            iv: Implied volatility (decimal)
            days_to_expiry: Days to expiration
            volume: Trading volume
            bid: Bid price
            ask: Ask price (optional)
            open_interest: Open interest (optional)
            spread_pct: Bid-ask spread percentage (optional)
            
        Returns:
            Quality score (0-100)
        """
        # IV score (higher IV is better, but with diminishing returns)
        iv_pct = iv * 100  # Convert to percentage
        iv_score = min(45, iv_pct * 1.2)  # Cap at 45, reduced weighting to emphasize other factors
        
        # Days to expiry score (20-40 days is optimal for iron condors)
        dte_score = 0
        if 20 <= days_to_expiry <= 40:
            dte_score = 25  # Optimal range - reduced from 30
        elif 15 <= days_to_expiry < 20 or 40 < days_to_expiry <= 50:
            dte_score = 15  # Good range - reduced from 20
        else:
            dte_score = 5   # Acceptable range - reduced from 10
            
        # Volume score (higher volume means better liquidity)
        volume_score = min(15, volume / 80)  # Increased weight from 10 to 15
        
        # Bid price score (higher premium is better)
        bid_score = min(15, bid * 4)  # Increased weight from 10 to 15
        
        # Add open interest score if available
        oi_score = 0
        if open_interest is not None:
            oi_score = min(10, open_interest / 100)  # New factor
        
        # Add spread quality score if available
        spread_score = 0
        if spread_pct is not None and spread_pct > 0:
            # Lower spread_pct is better (tighter spread)
            if spread_pct < 0.05:
                spread_score = 15  # Excellent spread
            elif spread_pct < 0.10:
                spread_score = 10  # Good spread
            elif spread_pct < 0.15:
                spread_score = 5   # Acceptable spread
        
        # Total score
        total_score = iv_score + dte_score + volume_score + bid_score + oi_score + spread_score
        
        # Return normalized score (0-100)
        return min(100, total_score)
        
    def analyze_option_chain(self, chain_data: Dict) -> Dict:
        """
        Analyze option chain to find implied volatility and trading opportunities.
        
        Args:
            chain_data: Option chain data from IBConnector.get_option_chain()
            
        Returns:
            Dictionary with analysis results
        """
        if not chain_data:
            self.logger.warning("No option chain data provided")
            return {}
            
        result = {
            'avg_iv_calls': 0.0,
            'avg_iv_puts': 0.0,
            'skew': 0.0,
            'opportunities': []
        }
        
        try:
            all_call_ivs = []
            all_put_ivs = []
            
            # Process each expiration
            for expiry, expiry_data in chain_data.items():
                calls = expiry_data.get('calls', {})
                puts = expiry_data.get('puts', {})
                
                # Calculate average IV for calls and puts
                call_ivs = [option.get('iv', 0) for strike, option in calls.items() 
                            if option.get('iv', 0) > 0]
                put_ivs = [option.get('iv', 0) for strike, option in puts.items() 
                           if option.get('iv', 0) > 0]
                
                all_call_ivs.extend(call_ivs)
                all_put_ivs.extend(put_ivs)
                
                # Days to expiry
                days_to_expiry = expiry_data.get('days_to_expiry', 0)
                
                # Find potential opportunities (high IV, reasonable time to expiry)
                # Use 20-60 days for better spread of opportunities while still focusing on optimal theta decay
                if 20 <= days_to_expiry <= 60:
                    # Focus on opportunities with higher implied volatility
                    # Adjust the min_volatility_threshold to ensure we're targeting higher IV options
                    min_iv_threshold = self.min_volatility_threshold / 100
                    
                    # For calls: Only consider those with enough premium, volume, and better spreads
                    for strike, option in calls.items():
                        iv = option.get('iv', 0)
                        bid = option.get('bid', 0)
                        ask = option.get('ask', 0)
                        volume = option.get('volume', 0) if option.get('volume') is not None else 0
                        open_interest = option.get('open_interest', 0) if option.get('open_interest') is not None else 0
                        
                        # Higher threshold for IV to ensure we're capturing true high-volatility opportunities
                        # Also check for reasonable bid price, volume, and bid-ask spread
                        spread_pct = (ask - bid) / bid if bid > 0 else float('inf')
                        
                        if (iv > min_iv_threshold * 1.1 and  # Increased minimum IV by 10%
                            bid >= 0.15 and                   # Increased minimum bid
                            volume >= 15 and                  # Increased minimum volume
                            open_interest >= 50 and           # Added minimum open interest
                            spread_pct < 0.15):               # Added maximum bid-ask spread percentage
                            
                            result['opportunities'].append({
                                'type': 'call',
                                'expiry': expiry,
                                'strike': strike,
                                'iv': iv,
                                'days_to_expiry': days_to_expiry,
                                'bid': bid,
                                'ask': ask,
                                'volume': volume,
                                'open_interest': open_interest,
                                'spread_pct': spread_pct,
                                'quality_score': self._calculate_opportunity_score(
                                    iv, days_to_expiry, volume, bid, 
                                    ask=ask, open_interest=open_interest, spread_pct=spread_pct
                                )
                            })
                    
                    # For puts: Similar but more stringent criteria as calls
                    for strike, option in puts.items():
                        iv = option.get('iv', 0)
                        bid = option.get('bid', 0)
                        ask = option.get('ask', 0)
                        volume = option.get('volume', 0) if option.get('volume') is not None else 0
                        open_interest = option.get('open_interest', 0) if option.get('open_interest') is not None else 0
                        
                        spread_pct = (ask - bid) / bid if bid > 0 else float('inf')
                        
                        if (iv > min_iv_threshold * 1.1 and  # Increased minimum IV by 10%
                            bid >= 0.15 and                   # Increased minimum bid
                            volume >= 15 and                  # Increased minimum volume
                            open_interest >= 50 and           # Added minimum open interest
                            spread_pct < 0.15):               # Added maximum bid-ask spread percentage
                            result['opportunities'].append({
                                'type': 'put',
                                'expiry': expiry,
                                'strike': strike,
                                'iv': iv,
                                'days_to_expiry': days_to_expiry,
                                'bid': bid,
                                'ask': ask,
                                'volume': volume,
                                'open_interest': open_interest,
                                'spread_pct': spread_pct,
                                'quality_score': self._calculate_opportunity_score(
                                    iv, days_to_expiry, volume, bid, 
                                    ask=ask, open_interest=open_interest, spread_pct=spread_pct
                                )
                            })
            
            # Calculate overall average IVs
            if all_call_ivs:
                result['avg_iv_calls'] = np.mean(all_call_ivs) * 100  # Convert to percentage
            
            if all_put_ivs:
                result['avg_iv_puts'] = np.mean(all_put_ivs) * 100  # Convert to percentage
            
            # Calculate put-call skew
            if all_call_ivs and all_put_ivs:
                result['skew'] = result['avg_iv_puts'] / result['avg_iv_calls'] - 1.0
            
            # Sort opportunities by quality score (best first) instead of just IV
            result['opportunities'].sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing option chain: {e}")
            return {'error': str(e)}
    
    def should_trade_today(self, vix_analysis: Dict) -> bool:
        """
        Determine if we should trade today based on volatility.
        
        Args:
            vix_analysis: Result from analyze_vix()
            
        Returns:
            bool: True if we should trade, False otherwise
        """
        signal = vix_analysis.get('signal', 'none')
        return signal in ['buy', 'strong_buy']
    
    def get_position_size_multiplier(self, vix_analysis: Dict) -> float:
        """
        Calculate position size multiplier based on volatility.
        Higher volatility = larger position size.
        
        Args:
            vix_analysis: Result from analyze_vix()
            
        Returns:
            float: Position size multiplier (0.0 to 1.0)
        """
        signal = vix_analysis.get('signal', 'none')
        
        if signal == 'strong_buy':
            return 1.0
        elif signal == 'buy':
            return 0.7
        elif signal == 'hold':
            return 0.3
        else:
            return 0.0


    def get_iv_hv_ratio(self, symbol: str, expiry: Optional[str] = None) -> float:
        """
        Calculate IV/HV ratio for a symbol.
        
        Args:
            symbol: Ticker symbol
            expiry: Option expiry (if None, uses nearest expiry)
            
        Returns:
            float: IV/HV ratio (IV/HV)
        """
        if self.market_data_store:
            return self.market_data_store.get_iv_hv_ratio(symbol, expiry)
        
        return 1.0  # Default if no market data store
    
    def update_adaptive_thresholds(self, current_vix: float) -> None:
        """
        Update adaptive thresholds based on current market conditions.
        
        Args:
            current_vix: Current VIX value
        """
        if not self.use_adaptive_thresholds:
            return
            
        # Add to history for tracking VIX changes
        self.vix_history.append(current_vix)
        if len(self.vix_history) > 30:  # Keep last 30 days
            self.vix_history.pop(0)
            
        # Calculate VIX statistics for more nuanced adaptation
        vix_history = np.array(self.vix_history)
        vix_mean = np.mean(vix_history) if len(vix_history) > 0 else current_vix
        vix_std = np.std(vix_history) if len(vix_history) > 1 else 0
        vix_percentile = (current_vix - vix_mean) / vix_std if vix_std > 0 else 0
        
        # Update IV threshold based on VIX - more conservative approach with finer gradations
        # Significantly enhanced to be more selective in all volatility ranges
        if current_vix > 40:
            # Severe volatility crisis - extremely conservative
            self.adaptive_iv_threshold = self.min_iv_threshold * 2.0  # Highest threshold
        elif current_vix > 35:
            # Extreme volatility - significantly increase threshold
            self.adaptive_iv_threshold = self.min_iv_threshold * 1.8  # Increased from 1.7
        elif current_vix > 30:
            # Very high volatility - substantially increase threshold
            self.adaptive_iv_threshold = self.min_iv_threshold * 1.6  # Increased from 1.5
        elif current_vix > 25:
            # High volatility - moderately increase threshold
            self.adaptive_iv_threshold = self.min_iv_threshold * 1.4  # Increased from 1.3
        elif current_vix > 20:
            # Above average volatility - slightly increase threshold
            self.adaptive_iv_threshold = self.min_iv_threshold * 1.2  # Increased from 1.15
        elif current_vix > 18:
            # Slightly above normal volatility
            self.adaptive_iv_threshold = self.min_iv_threshold * 1.1
        elif current_vix > 15:
            # Normal volatility - use standard threshold
            self.adaptive_iv_threshold = self.min_iv_threshold
        elif current_vix > 12:
            # Slightly below normal volatility - slightly decrease threshold
            self.adaptive_iv_threshold = self.min_iv_threshold * 0.95
        else:
            # Very low volatility - decrease threshold but maintain selectivity
            self.adaptive_iv_threshold = self.min_iv_threshold * 0.9
        
        # Update IV/HV ratio threshold based on VIX volatility
        # Become more selective as VIX becomes more volatile
        if len(self.vix_history) >= 5:
            # Calculate recent VIX volatility (standard deviation)
            recent_vix_std = np.std(self.vix_history[-5:])
            
            # Calculate VIX trend (positive = rising, negative = falling)
            vix_trend = 0
            if len(self.vix_history) >= 10:
                vix_5d_avg = np.mean(self.vix_history[-5:])
                vix_10d_avg = np.mean(self.vix_history[-10:])
                vix_trend = (vix_5d_avg / vix_10d_avg) - 1.0  # Normalized trend
            
            # Adjust IV/HV threshold based on VIX volatility and trend
            if recent_vix_std > 4.0:  # Extremely volatile VIX
                # Be extremely selective in chaotic markets
                self.adaptive_iv_hv_ratio = self.iv_hv_ratio_threshold * 1.5  # Increased from 1.4
            elif recent_vix_std > 3.0:  # Very volatile VIX
                # Be much more selective in unstable markets
                self.adaptive_iv_hv_ratio = self.iv_hv_ratio_threshold * 1.4  # No change
            elif recent_vix_std > 2.0:  # Moderately volatile VIX
                # Be more selective in variable markets
                self.adaptive_iv_hv_ratio = self.iv_hv_ratio_threshold * 1.3  # Increased from 1.25
            elif recent_vix_std > 1.0:  # Slightly volatile VIX
                # Be somewhat more selective in normal markets with some variability
                self.adaptive_iv_hv_ratio = self.iv_hv_ratio_threshold * 1.15  # Increased from 1.1
            elif recent_vix_std < 0.5 and current_vix < 15:  # Very stable low VIX
                # Be slightly less selective in stable, low volatility markets
                self.adaptive_iv_hv_ratio = self.iv_hv_ratio_threshold * 0.95  # Slightly less reduction
            else:
                # Standard markets
                self.adaptive_iv_hv_ratio = self.iv_hv_ratio_threshold
                
            # Further adjust based on VIX trend (rising VIX = be more selective)
            if vix_trend > 0.05:  # Strongly rising VIX
                self.adaptive_iv_hv_ratio *= 1.1  # Additional 10% increase in selectivity
            elif vix_trend < -0.05:  # Strongly falling VIX
                # Don't reduce below original threshold
                self.adaptive_iv_hv_ratio = max(self.iv_hv_ratio_threshold, self.adaptive_iv_hv_ratio * 0.95)
        
        self.logger.debug(f"Updated adaptive thresholds - IV: {self.adaptive_iv_threshold:.1f}%, "
                        f"IV/HV ratio: {self.adaptive_iv_hv_ratio:.2f}, "
                        f"VIX: {current_vix:.1f}, VIX percentile: {vix_percentile:.2f}")
    
    def analyze_volatility_for_harvesting(self, 
                                        symbol: str,
                                        vix_analysis: Dict, 
                                        option_chain: Optional[Dict] = None) -> Dict:
        """
        Analyze volatility for the enhanced volatility-harvesting system.
        
        Args:
            symbol: Ticker symbol
            vix_analysis: VIX analysis data
            option_chain: Option chain data (optional)
            
        Returns:
            Dictionary with volatility harvesting analysis
        """
        result = {
            'signal': 'none',
            'strategy': 'none',
            'iv_hv_ratio': 0.0,
            'historical_volatility': 0.0,
            'implied_volatility': 0.0,
            'vix': vix_analysis.get('current_vix', 0.0),
            'adaptive_iv_threshold': self.adaptive_iv_threshold,
            'adaptive_iv_hv_ratio': self.adaptive_iv_hv_ratio
        }
        
        try:
            # Update adaptive thresholds
            current_vix = vix_analysis.get('current_vix', 0.0)
            self.update_adaptive_thresholds(current_vix)
            
            # Calculate historical volatility
            hv = 0.0
            if self.market_data_store:
                hv = self.market_data_store.calculate_historical_volatility(symbol)
            elif len(self.historical_volatility) > 0:
                hv = self.historical_volatility[-1]
            
            result['historical_volatility'] = hv
            
            # Calculate implied volatility
            iv = 0.0
            if option_chain:
                # Extract all IVs from the option chain
                all_ivs = []
                for expiry_data in option_chain.values():
                    for call in expiry_data.get('calls', {}).values():
                        if call.get('iv', 0) > 0:
                            all_ivs.append(call['iv'])
                    for put in expiry_data.get('puts', {}).values():
                        if put.get('iv', 0) > 0:
                            all_ivs.append(put['iv'])
                
                if all_ivs:
                    iv = np.mean(all_ivs) * 100  # Convert to percentage
            
            result['implied_volatility'] = iv
            
            # Calculate IV/HV ratio
            iv_hv_ratio = 0.0
            if hv > 0:
                iv_hv_ratio = iv / hv
            else:
                # Use the market data store if available
                iv_hv_ratio = self.get_iv_hv_ratio(symbol)
                
            result['iv_hv_ratio'] = iv_hv_ratio
            self.iv_hv_ratios.append(iv_hv_ratio)
            
            # Keep only the most recent 20 ratios
            if len(self.iv_hv_ratios) > 20:
                self.iv_hv_ratios = self.iv_hv_ratios[-20:]
            
            # Determine signal based on IV/HV ratio and other factors
            # Added additional check for minimum IV to ensure sufficiently high implied volatility
            if iv_hv_ratio >= self.adaptive_iv_hv_ratio and iv >= self.adaptive_iv_threshold:
                # Check VIX state and trend
                vix_signal = vix_analysis.get('signal', 'none')
                volatility_state = vix_analysis.get('volatility_state', 'unknown')
                
                # More conservative approach in extreme volatility states
                if volatility_state == 'extreme' and iv_hv_ratio < self.adaptive_iv_hv_ratio * 1.3:
                    # In extreme volatility, need much higher IV/HV ratio to generate signal
                    result['signal'] = 'monitor'
                elif vix_signal in ['buy', 'strong_buy']:
                    # Strong signal - only if VIX suggests favorable conditions
                    result['signal'] = 'strong_volatility_harvest'
                    result['strategy'] = 'iron_condor'
                elif vix_signal == 'hold' and iv_hv_ratio >= self.adaptive_iv_hv_ratio * 1.2:
                    # Elevated IV/HV ratio, even if VIX isn't rising
                    result['signal'] = 'volatility_harvest'
                    result['strategy'] = 'iron_condor'
                else:
                    result['signal'] = 'monitor'
            else:
                result['signal'] = 'wait'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in volatility harvesting analysis: {e}")
            result['error'] = str(e)
            return result


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    analyzer = VolatilityAnalyzer()
    
    # Test historical volatility calculation
    prices = [100, 102, 104, 103, 105, 107, 108, 109, 111, 110, 
              112, 111, 113, 114, 116, 115, 117, 119, 120, 122, 121]
    hist_vol = analyzer.calculate_historical_volatility(prices)
    print(f"Historical volatility: {hist_vol:.2f}%")
    
    # Test VIX analysis
    vix_data = [
        {'date': '2022-01-01', 'close': 17.0},
        {'date': '2022-01-02', 'close': 17.5},
        {'date': '2022-01-03', 'close': 18.0},
        {'date': '2022-01-04', 'close': 19.0},
        {'date': '2022-01-05', 'close': 20.0},
        {'date': '2022-01-06', 'close': 22.0}
    ]
    vix_analysis = analyzer.analyze_vix(vix_data)
    print("VIX Analysis:", vix_analysis)
    print(f"Should trade today: {analyzer.should_trade_today(vix_analysis)}")
    print(f"Position size multiplier: {analyzer.get_position_size_multiplier(vix_analysis):.2f}")
    
    # Test volatility harvesting analysis
    test_symbol = "SPY"
    vh_analysis = analyzer.analyze_volatility_for_harvesting(test_symbol, vix_analysis)
    print(f"Volatility Harvesting Analysis: {vh_analysis}")