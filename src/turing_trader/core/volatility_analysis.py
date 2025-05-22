"""
Volatility Analysis Module

This module provides tools for analyzing market volatility:
- Historical Volatility (HV) calculations
- Implied Volatility (IV) analysis
- VIX index monitoring
- Volatility-based trading signals
"""
import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

class VolatilityAnalyzer:
    """
    Analyzes market volatility and generates trading signals
    based on volatility conditions
    """
    
    def __init__(self, lookback_period: int = 20, vix_threshold: float = 20.0):
        """
        Initialize the volatility analyzer
        
        Args:
            lookback_period: Number of days to calculate historical volatility
            vix_threshold: Baseline VIX threshold for volatility signals
        """
        self.logger = logging.getLogger(__name__)
        self.lookback_period = lookback_period
        self.vix_threshold = vix_threshold
        
        # Data storage
        self.historical_prices = {}
        self.historical_volatility = {}
        self.implied_volatility = {}
        self.vix_data = []
    
    def calculate_historical_volatility(self, 
                                       prices: pd.Series, 
                                       window: int = None) -> float:
        """
        Calculate historical volatility from price series
        
        Args:
            prices: Series of historical prices
            window: Lookback period (defaults to self.lookback_period)
            
        Returns:
            float: Annualized historical volatility
        """
        if window is None:
            window = self.lookback_period
            
        if len(prices) < window + 1:
            self.logger.warning(f"Insufficient data for volatility calculation, need {window+1}, got {len(prices)}")
            return None
            
        # Calculate daily returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate volatility (standard deviation of returns)
        daily_vol = returns.rolling(window=window).std().iloc[-1]
        
        # Annualize (multiply by sqrt of trading days)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    def update_historical_prices(self, symbol: str, new_prices: pd.Series) -> None:
        """
        Update the historical price database for a symbol
        
        Args:
            symbol: Ticker symbol
            new_prices: Series of new price data to add
        """
        if symbol not in self.historical_prices:
            self.historical_prices[symbol] = new_prices
        else:
            # Merge new with existing, avoid duplicates
            combined = pd.concat([self.historical_prices[symbol], new_prices])
            self.historical_prices[symbol] = combined[~combined.index.duplicated(keep='last')]
        
        # Recalculate historical volatility
        self.historical_volatility[symbol] = self.calculate_historical_volatility(self.historical_prices[symbol])
        self.logger.debug(f"Updated historical volatility for {symbol}: {self.historical_volatility[symbol]:.2%}")
    
    def update_implied_volatility(self, symbol: str, iv_data: Dict[str, float]) -> None:
        """
        Update implied volatility data for a symbol
        
        Args:
            symbol: Ticker symbol
            iv_data: Dictionary of implied volatility data from options
        """
        self.implied_volatility[symbol] = iv_data
        self.logger.debug(f"Updated implied volatility for {symbol}")
    
    def update_vix(self, vix_value: float, timestamp: datetime.datetime = None) -> None:
        """
        Update VIX index value
        
        Args:
            vix_value: Current VIX index value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        self.vix_data.append((timestamp, vix_value))
        
        # Trim old data
        if len(self.vix_data) > 1000:  # Keep reasonable history
            self.vix_data = self.vix_data[-1000:]
            
        self.logger.debug(f"Updated VIX: {vix_value:.2f}")
    
    def get_current_vix(self) -> float:
        """
        Get the most recent VIX value
        
        Returns:
            float: Latest VIX value or None if no data
        """
        if not self.vix_data:
            return None
        return self.vix_data[-1][1]
    
    def is_high_volatility_environment(self) -> bool:
        """
        Determine if we're in a high volatility environment
        
        Returns:
            bool: True if in high volatility environment
        """
        # Get latest VIX
        vix = self.get_current_vix()
        if vix is None:
            return False
            
        # Check if VIX above threshold
        return vix > self.vix_threshold
    
    def detect_volatility_spike(self, 
                              window: int = 5, 
                              threshold_pct: float = 10.0) -> bool:
        """
        Detect if there's been a recent spike in volatility
        
        Args:
            window: Window to look for spike
            threshold_pct: Minimum percent increase to qualify as spike
            
        Returns:
            bool: True if volatility spike detected
        """
        if len(self.vix_data) < window + 1:
            return False
            
        # Get recent VIX values
        recent_vix = [v[1] for v in self.vix_data[-window-1:]]
        
        # Calculate percent change from window start to now
        start_vix = recent_vix[0]
        current_vix = recent_vix[-1]
        
        pct_change = (current_vix - start_vix) / start_vix * 100
        
        return pct_change > threshold_pct
    
    def get_volatility_signal(self) -> Dict:
        """
        Generate a volatility-based trading signal
        
        Returns:
            dict: Trading signal information
        """
        vix = self.get_current_vix()
        if vix is None:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Insufficient data'}
            
        # Check for volatility spike
        spike_detected = self.detect_volatility_spike()
        
        # Determine market condition
        high_vol_environment = self.is_high_volatility_environment()
        
        # Generate signal
        if spike_detected:
            return {
                'signal': 'BUY' if high_vol_environment else 'NEUTRAL',
                'strength': 2 if high_vol_environment else 1,
                'reason': f'Volatility spike detected, VIX: {vix:.2f}'
            }
        elif high_vol_environment:
            return {
                'signal': 'BUY',
                'strength': 1,
                'reason': f'Elevated volatility, VIX: {vix:.2f}'
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'strength': 0,
                'reason': f'Normal volatility, VIX: {vix:.2f}'
            }
    
    def get_option_volatility_metrics(self, symbol: str) -> Dict:
        """
        Get consolidated volatility metrics for options trading
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            dict: Volatility metrics including HV, IV, and signals
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.datetime.now(),
            'historical_volatility': None,
            'implied_volatility': None,
            'vix': self.get_current_vix(),
            'signal': 'NEUTRAL',
            'high_vol_environment': self.is_high_volatility_environment(),
            'volatility_spike': self.detect_volatility_spike()
        }
        
        # Add historical volatility if available
        if symbol in self.historical_volatility:
            result['historical_volatility'] = self.historical_volatility[symbol]
            
        # Add implied volatility if available
        if symbol in self.implied_volatility:
            result['implied_volatility'] = self.implied_volatility[symbol]
            
        # Get trading signal
        signal = self.get_volatility_signal()
        result['signal'] = signal['signal']
        result['signal_strength'] = signal['strength']
        result['signal_reason'] = signal['reason']
        
        return result