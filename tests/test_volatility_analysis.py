"""
Tests for the Volatility Analysis module
"""
import datetime
import pytest
import pandas as pd
import numpy as np
from turing_trader.core.volatility_analysis import VolatilityAnalyzer

def test_volatility_analyzer_initialization():
    """Test VolatilityAnalyzer initialization"""
    analyzer = VolatilityAnalyzer(lookback_period=20, vix_threshold=15.0)
    
    assert analyzer.lookback_period == 20
    assert analyzer.vix_threshold == 15.0
    assert analyzer.historical_prices == {}
    assert analyzer.historical_volatility == {}
    assert analyzer.implied_volatility == {}
    assert analyzer.vix_data == []

def test_historical_volatility_calculation():
    """Test historical volatility calculation"""
    analyzer = VolatilityAnalyzer(lookback_period=10)
    
    # Create a price series with known volatility pattern
    dates = pd.date_range(start='2023-01-01', periods=30)
    prices = pd.Series(
        index=dates, 
        data=[100 * (1 + 0.01 * np.sin(i)) for i in range(30)]
    )
    
    # Calculate volatility
    vol = analyzer.calculate_historical_volatility(prices, window=10)
    
    # Volatility should be a positive number
    assert vol > 0
    
    # Test with insufficient data
    short_prices = prices.iloc[:5]
    vol_short = analyzer.calculate_historical_volatility(short_prices, window=10)
    assert vol_short is None

def test_update_vix():
    """Test VIX data updates"""
    analyzer = VolatilityAnalyzer()
    
    # Add some VIX values
    analyzer.update_vix(20.5)
    analyzer.update_vix(21.0)
    analyzer.update_vix(19.8)
    
    # Check that values were stored
    assert len(analyzer.vix_data) == 3
    assert analyzer.get_current_vix() == 19.8
    
    # Check high volatility detection
    analyzer = VolatilityAnalyzer(vix_threshold=20.0)
    analyzer.update_vix(21.5)
    assert analyzer.is_high_volatility_environment() == True
    
    analyzer.update_vix(18.5)
    assert analyzer.is_high_volatility_environment() == False

def test_volatility_spike_detection():
    """Test volatility spike detection"""
    analyzer = VolatilityAnalyzer()
    
    # Add rising VIX values
    base_time = datetime.datetime.now()
    analyzer.update_vix(20.0, base_time - datetime.timedelta(minutes=50))
    analyzer.update_vix(21.0, base_time - datetime.timedelta(minutes=40))
    analyzer.update_vix(22.0, base_time - datetime.timedelta(minutes=30))
    analyzer.update_vix(23.0, base_time - datetime.timedelta(minutes=20))
    analyzer.update_vix(24.0, base_time - datetime.timedelta(minutes=10))
    analyzer.update_vix(25.0, base_time)
    
    # Should detect a 25% spike (20.0 -> 25.0)
    assert analyzer.detect_volatility_spike(window=5, threshold_pct=10.0) == True
    
    # Should not detect spike with higher threshold
    assert analyzer.detect_volatility_spike(window=5, threshold_pct=30.0) == False

def test_get_volatility_signal():
    """Test volatility signal generation"""
    analyzer = VolatilityAnalyzer(vix_threshold=20.0)
    
    # No VIX data yet
    signal = analyzer.get_volatility_signal()
    assert signal['signal'] == 'NEUTRAL'
    
    # Normal volatility, below threshold
    analyzer.update_vix(15.0)
    signal = analyzer.get_volatility_signal()
    assert signal['signal'] == 'NEUTRAL'
    
    # High volatility
    analyzer.update_vix(25.0)
    signal = analyzer.get_volatility_signal()
    assert signal['signal'] == 'BUY'
    
    # Add volatility spike
    base_time = datetime.datetime.now()
    analyzer.vix_data = []  # Clear existing data
    analyzer.update_vix(15.0, base_time - datetime.timedelta(minutes=10))
    analyzer.update_vix(20.0, base_time - datetime.timedelta(minutes=5))
    analyzer.update_vix(22.0, base_time)
    
    signal = analyzer.get_volatility_signal()
    assert signal['signal'] == 'BUY'
    assert signal['strength'] > 0