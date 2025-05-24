"""
Mock data generator for testing the TuringTrader algorithm evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_market_data(start_date: str, end_date: str, symbol: str = 'SPY') -> pd.DataFrame:
    """
    Generate mock market data for testing.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbol: Market symbol ('SPY' or 'VIX')
        
    Returns:
        DataFrame with mock market data
    """
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Create base dataframe
    df = pd.DataFrame(index=date_range)
    
    # Generate random price data
    np.random.seed(42 if symbol == 'SPY' else 24)  # Different seeds for different symbols
    
    # Starting price
    if symbol == 'SPY':
        start_price = 450.0
        volatility = 0.01
    elif symbol == 'VIX':
        start_price = 15.0
        volatility = 0.03
    else:
        start_price = 100.0
        volatility = 0.02
    
    # Generate price series using random walk
    price_changes = np.random.normal(0, volatility, len(date_range))
    prices = start_price * (1 + np.cumsum(price_changes))
    
    # Ensure positive prices
    prices = np.maximum(prices, start_price * 0.5)
    
    # Add price data
    df['open'] = prices * (1 + np.random.normal(0, 0.002, len(date_range)))
    df['high'] = prices * (1 + np.random.uniform(0.001, 0.015, len(date_range)))
    df['low'] = prices * (1 - np.random.uniform(0.001, 0.015, len(date_range)))
    df['close'] = prices
    df['volume'] = np.random.randint(1000000, 10000000, len(date_range))
    
    # Make sure high/low are actually high/low
    df['high'] = np.maximum(np.maximum(df['high'], df['open']), df['close'])
    df['low'] = np.minimum(np.minimum(df['low'], df['open']), df['close'])
    
    return df


class MockDataFetcher:
    """Mock data fetcher for testing."""
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the mock data fetcher.
        
        Args:
            data_dir: Directory for storing cached data (not used in mock)
        """
        self.data_dir = data_dir
    
    def fetch_data(self, 
                 symbol: str, 
                 start_date,
                 end_date,
                 interval: str = '1d',
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch mock market data.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (ignored in mock)
            use_cache: Whether to use cached data (ignored in mock)
            
        Returns:
            DataFrame with mock market data
        """
        # Create mock data
        if symbol.upper() in ['SPY', '^SPX', 'SPX']:
            return generate_mock_market_data(start_date, end_date, 'SPY')
        elif symbol.upper() in ['VIX', '^VIX']:
            return generate_mock_market_data(start_date, end_date, 'VIX')
        else:
            # Return empty dataframe for unknown symbols
            return pd.DataFrame()
    
    def fetch_vix_data(self, start_date, end_date, use_cache: bool = True) -> pd.DataFrame:
        """Fetch mock VIX data."""
        return self.fetch_data('^VIX', start_date, end_date, use_cache=use_cache)
    
    def fetch_sp500_data(self, start_date, end_date, use_cache: bool = True) -> pd.DataFrame:
        """Fetch mock S&P500 data."""
        return self.fetch_data('SPY', start_date, end_date, use_cache=use_cache)
    
    def fetch_option_chain(self, symbol: str, date=None) -> dict:
        """
        Fetch mock option chain data.
        
        Args:
            symbol: Ticker symbol
            date: Date for option chain
            
        Returns:
            Dict with mock option chain data
        """
        # Create a simple mock option chain
        if date is None:
            date = datetime.now()
            
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # Current price of underlying
        if symbol.upper() in ['SPY', '^SPX', 'SPX']:
            underlying_price = 450.0
        else:
            underlying_price = 100.0
            
        # Create expiration dates
        expirations = []
        for i in range(1, 5):
            exp_date = date + timedelta(days=i*7)
            expirations.append(exp_date.strftime('%Y-%m-%d'))
            
        # Create option chain dictionary
        option_chain = {}
        
        for exp_date in expirations:
            exp_datetime = pd.to_datetime(exp_date)
            days_to_expiry = (exp_datetime - date).days
            
            # Create strike prices
            strikes = []
            for i in range(-10, 11):
                strike = underlying_price * (1 + i * 0.01)
                strikes.append(round(strike, 1))
                
            # Create call options
            calls = {}
            for strike in strikes:
                delta = max(0, min(1, 0.5 - (strike - underlying_price) / (underlying_price * 0.1)))
                calls[strike] = {
                    'strike': strike,
                    'bid': max(0.01, underlying_price * 0.05 * delta),
                    'ask': max(0.01, underlying_price * 0.05 * delta) + 0.05,
                    'delta': delta,
                    'gamma': 0.01,
                    'theta': -0.01,
                    'vega': 0.1,
                    'open_interest': int(np.random.randint(100, 1000)),
                    'volume': int(np.random.randint(10, 100))
                }
                
            # Create put options
            puts = {}
            for strike in strikes:
                delta = -max(0, min(1, 0.5 - (underlying_price - strike) / (underlying_price * 0.1)))
                puts[strike] = {
                    'strike': strike,
                    'bid': max(0.01, underlying_price * 0.05 * abs(delta)),
                    'ask': max(0.01, underlying_price * 0.05 * abs(delta)) + 0.05,
                    'delta': delta,
                    'gamma': 0.01,
                    'theta': -0.01,
                    'vega': 0.1,
                    'open_interest': int(np.random.randint(100, 1000)),
                    'volume': int(np.random.randint(10, 100))
                }
                
            # Add to option chain
            option_chain[exp_date] = {
                'days_to_expiry': days_to_expiry,
                'calls': calls,
                'puts': puts,
                'strikes': strikes
            }
            
        return option_chain
    
    def get_cached_symbols(self) -> list:
        """Get list of cached symbols (mock implementation)."""
        return ['SPY', 'VIX']