"""
Enhanced mock data generator for realistic backtesting in the TuringTrader algorithm evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os


class RealisticMockDataFetcher:
    """Mock data fetcher that generates realistic market data for backtesting."""
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the realistic mock data fetcher.
        
        Args:
            data_dir: Directory for storing cached data (not used in mock)
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Properties for enhanced backtesting
        self.iv_hv_ratio_min = 1.2  # Minimum IV/HV ratio for backtesting
        self.iv_hv_ratio_max = 2.5  # Maximum IV/HV ratio for backtesting
        self.base_vix_level = 25.0  # Base VIX level for more realistic volatility
    
    def fetch_data(self, 
                 symbol: str, 
                 start_date,
                 end_date,
                 interval: str = '1d',
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch realistic mock market data.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (ignored in mock)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with mock market data
        """
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Check for cached data if use_cache is enabled
        cache_file = os.path.join(self.data_dir, f"{symbol}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_realistic_mock.csv")
        
        if use_cache and os.path.exists(cache_file):
            self.logger.info(f"Loading cached realistic mock data for {symbol} from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Generate realistic data
        if symbol.upper() in ['SPY', '^SPX', 'SPX']:
            data = self._generate_realistic_spy_data(start_date, end_date)
        elif symbol.upper() in ['VIX', '^VIX']:
            data = self._generate_realistic_vix_data(start_date, end_date)
        else:
            # Return empty dataframe for unknown symbols
            self.logger.warning(f"No mock data available for symbol: {symbol}")
            return pd.DataFrame()
            
        # Save to cache
        data.to_csv(cache_file)
        self.logger.info(f"Generated realistic mock data for {symbol} with {len(data)} rows")
        
        return data
    
    def _generate_realistic_spy_data(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate realistic SPY data with price movements that mimic actual stock market behavior.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with mock SPY data
        """
        # Generate date range of business days
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Create base dataframe
        df = pd.DataFrame(index=date_range)
        
        # Starting price - realistic SPY price
        start_price = 450.0
        
        # Generate price series using more realistic parameters
        # SPY has historically shown annual volatility of ~15-20%
        # This translates to daily volatility of around 1%
        daily_volatility = 0.01
        daily_drift = 0.0003  # Small upward bias (~7-8% annually)
        
        # Generate daily returns with slight upward bias and occasional larger movements
        num_days = len(date_range)
        random_factor = np.random.normal(daily_drift, daily_volatility, num_days)
        
        # Add some occasional jumps (big up or down days)
        for i in range(num_days // 20):  # Every ~20 trading days
            jump_idx = np.random.randint(0, num_days)
            random_factor[jump_idx] = random_factor[jump_idx] * 3
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + random_factor)
        prices = start_price * cumulative_returns
        
        # Add price data with typical open/high/low/close relationships
        df['open'] = prices * (1 + np.random.normal(0, 0.002, num_days))
        # High is higher than both open and close
        df['close'] = prices
        df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0.001, 0.003, num_days)))
        # Low is lower than both open and close
        df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0.001, 0.003, num_days)))
        
        # Add volume with higher volume on bigger price movement days
        base_volume = 50000000  # SPY average volume
        volume_factor = 1 + 5 * np.abs(random_factor)  # More volume on bigger move days
        df['volume'] = base_volume * volume_factor
        df['volume'] = df['volume'].astype(int)
        
        # Add adjusted close (same as close for simplicity)
        df['adj close'] = df['close']
        
        return df
    
    def _generate_realistic_vix_data(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate realistic VIX data with mean-reverting behavior and occasional spikes.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with mock VIX data
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Create base dataframe
        df = pd.DataFrame(index=date_range)
        
        # VIX parameters - adjusted for more trading signals
        mean_level = self.base_vix_level  # Use the base VIX level
        reversion_strength = 0.05  # Mean reversion strength
        daily_volatility = 0.06    # Increased daily volatility for more signal opportunities
        
        # Generate VIX levels with mean-reverting behavior and occasional spikes
        num_days = len(date_range)
        vix_levels = np.zeros(num_days)
        
        # Initial VIX level
        vix_levels[0] = mean_level
        
        # Generate the rest of the series
        for i in range(1, num_days):
            # Mean reversion component
            mean_reversion = reversion_strength * (mean_level - vix_levels[i-1])
            
            # Random component
            random_shock = np.random.normal(0, daily_volatility * vix_levels[i-1])
            
            # Next value
            vix_levels[i] = vix_levels[i-1] + mean_reversion + random_shock
            
            # Ensure VIX is always positive and add occasional spikes
            vix_levels[i] = max(12.0, vix_levels[i])
        
        # Add occasional volatility spikes (market panic events)
        for i in range(num_days // 30):  # More frequent spikes (every ~30 days)
            spike_idx = np.random.randint(0, num_days)
            spike_magnitude = np.random.uniform(1.5, 2.5)  # 50-150% increase
            vix_levels[spike_idx] = vix_levels[spike_idx] * spike_magnitude
            
            # Usually spikes are followed by a gradual decrease
            decay_period = min(20, num_days - spike_idx - 1)
            decay_factor = np.linspace(0.8, 0.1, decay_period)
            
            for j in range(decay_period):
                if spike_idx + j + 1 < num_days:
                    extra = vix_levels[spike_idx] - vix_levels[spike_idx - 1]
                    vix_levels[spike_idx + j + 1] += extra * decay_factor[j]
        
        # Create OHLC data for VIX (less important, as usually only close is used)
        df['close'] = vix_levels
        df['open'] = df['close'].shift(1).fillna(df['close'])
        
        # Add some intraday noise to create high and low
        daily_range_factor = 0.03  # 3% intraday range on average
        df['high'] = df['close'] * (1 + np.random.uniform(0, daily_range_factor, num_days))
        df['low'] = df['close'] * (1 - np.random.uniform(0, daily_range_factor, num_days))
        
        # Volume is less relevant for VIX, but add it for consistency
        df['volume'] = np.random.randint(1000000, 5000000, num_days)
        
        # Add adjusted close (same as close for indices)
        df['adj close'] = df['close']
        
        return df
    
    def fetch_vix_data(self, start_date, end_date, use_cache: bool = True) -> pd.DataFrame:
        """Fetch realistic mock VIX data."""
        return self.fetch_data('^VIX', start_date, end_date, use_cache=use_cache)
    
    def fetch_sp500_data(self, start_date, end_date, use_cache: bool = True) -> pd.DataFrame:
        """Fetch realistic mock S&P500 data."""
        return self.fetch_data('SPY', start_date, end_date, use_cache=use_cache)
    
    def fetch_option_chain(self, symbol: str, date=None) -> dict:
        """
        Fetch realistic mock option chain data.
        
        Args:
            symbol: Ticker symbol
            date: Date for option chain
            
        Returns:
            Dict with mock option chain data
        """
        # Create a more realistic mock option chain
        if date is None:
            date = datetime.now()
            
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # Get current price of underlying from our mock data
        today = date.strftime('%Y-%m-%d')
        yesterday = (date - timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            # Try to get recent price data
            price_data = self.fetch_data(symbol, yesterday, today)
            if not price_data.empty:
                underlying_price = price_data['close'].iloc[-1]
            else:
                # Fallback prices if no data
                underlying_price = 450.0 if symbol.upper() in ['SPY', '^SPX', 'SPX'] else 100.0
        except:
            # Fallback prices if error
            underlying_price = 450.0 if symbol.upper() in ['SPY', '^SPX', 'SPX'] else 100.0
            
        # For VIX, get current level to inform implied volatility
        try:
            vix_data = self.fetch_data('^VIX', yesterday, today)
            if not vix_data.empty:
                current_vix = vix_data['close'].iloc[-1]
            else:
                current_vix = 15.0
        except:
            current_vix = 15.0
            
        # Create realistic expiration dates (weekly for SPY)
        expirations = []
        current_date = date
        
        # Add weekly expirations (every Friday)
        for i in range(12):  # 12 weeks out
            # Find next Friday
            days_to_friday = (4 - current_date.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7  # If today is Friday, go to next Friday
                
            friday = current_date + timedelta(days=days_to_friday)
            expirations.append(friday.strftime('%Y-%m-%d'))
            current_date = friday
            
        # Add monthly expirations (3rd Friday of month)
        current_date = date
        for i in range(6):  # 6 months out
            # Move to next month
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1, day=1)
                
            # Find 3rd Friday
            first_day = next_month.replace(day=1)
            friday_offset = (4 - first_day.weekday()) % 7  # Days until first Friday
            third_friday = first_day + timedelta(days=friday_offset + 14)  # +14 days to get to 3rd Friday
            
            # Only add if not already in the list
            if third_friday.strftime('%Y-%m-%d') not in expirations:
                expirations.append(third_friday.strftime('%Y-%m-%d'))
                
            current_date = next_month
            
        # Create option chain dictionary
        option_chain = {}
        
        for exp_date in expirations:
            exp_datetime = pd.to_datetime(exp_date)
            days_to_expiry = (exp_datetime - date).days
            days_to_expiry = max(1, days_to_expiry)
            
            # Get annualized time to expiry
            t = days_to_expiry / 365.0
            
            # Implied volatility tends to be higher for nearer-term options and lower for longer term
            # Also, VIX level influences the IV
            base_iv = max(current_vix / 100.0 * 1.4, 0.30)  # Convert VIX to decimal, increase by 40%, minimum 30%
            # IV skew - higher for OTM puts, lower for OTM calls
            
            # Create strike prices around current price
            strikes = []
            num_strikes = 15  # Number of strikes on each side
            step_size = round(underlying_price * 0.01, 1)  # 1% steps
            
            for i in range(-num_strikes, num_strikes + 1):
                strike = round(underlying_price + i * step_size, 1)
                strikes.append(strike)
                
            # Create call options
            calls = {}
            for strike in strikes:
                # Calculate distance from current price
                moneyness = strike / underlying_price - 1
                
                # IV skew - higher for OTM puts, lower for OTM calls
                strike_iv = base_iv * (1 - 0.1 * moneyness)  # Simple skew model
                # Ensure IV is within reasonable bounds
                strike_iv = max(0.05, min(1.0, strike_iv))
                
                # Option pricing approximation
                d1 = (np.log(underlying_price / strike) + (0.025 + 0.5 * strike_iv**2) * t) / (strike_iv * np.sqrt(t))
                d2 = d1 - strike_iv * np.sqrt(t)
                
                from scipy.stats import norm
                call_price = underlying_price * norm.cdf(d1) - strike * np.exp(-0.025 * t) * norm.cdf(d2)
                
                # Calculate greeks
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (underlying_price * strike_iv * np.sqrt(t))
                theta = -underlying_price * strike_iv * norm.pdf(d1) / (2 * np.sqrt(t)) - 0.025 * strike * np.exp(-0.025 * t) * norm.cdf(d2)
                vega = underlying_price * np.sqrt(t) * norm.pdf(d1) * 0.01  # For 1% change in IV
                
                # Add bid-ask spread
                spread_pct = 0.05 + 0.15 * (1 - delta)  # Wider spreads for OTM options
                bid = max(0.01, call_price * (1 - spread_pct))
                ask = call_price * (1 + spread_pct)
                
                calls[float(strike)] = {
                    'strike': strike,
                    'bid': round(bid, 2),
                    'ask': round(ask, 2),
                    'last': round((bid + ask) / 2, 2),
                    'volume': int(np.random.randint(10, 1000) * (1.5 - abs(moneyness))),
                    'open_interest': int(np.random.randint(100, 5000) * (1.5 - abs(moneyness))),
                    'delta': round(delta, 3),
                    'gamma': round(gamma, 4),
                    'theta': round(theta, 3),
                    'vega': round(vega, 3),
                    'iv': round(strike_iv, 3)
                }
                
            # Create put options
            puts = {}
            for strike in strikes:
                # Calculate distance from current price
                moneyness = 1 - strike / underlying_price
                
                # IV skew - higher for OTM puts
                strike_iv = base_iv * (1 + 0.2 * moneyness)  # Put skew is typically steeper
                # Ensure IV is within reasonable bounds
                strike_iv = max(0.05, min(1.0, strike_iv))
                
                # Option pricing approximation
                d1 = (np.log(underlying_price / strike) + (0.025 + 0.5 * strike_iv**2) * t) / (strike_iv * np.sqrt(t))
                d2 = d1 - strike_iv * np.sqrt(t)
                
                from scipy.stats import norm
                call_price = underlying_price * norm.cdf(d1) - strike * np.exp(-0.025 * t) * norm.cdf(d2)
                put_price = call_price + strike * np.exp(-0.025 * t) - underlying_price
                
                # Calculate greeks
                delta = norm.cdf(d1) - 1
                gamma = norm.pdf(d1) / (underlying_price * strike_iv * np.sqrt(t))
                theta = -underlying_price * strike_iv * norm.pdf(d1) / (2 * np.sqrt(t)) + 0.025 * strike * np.exp(-0.025 * t) * norm.cdf(-d2)
                vega = underlying_price * np.sqrt(t) * norm.pdf(d1) * 0.01  # For 1% change in IV
                
                # Add bid-ask spread
                spread_pct = 0.05 + 0.15 * (1 - abs(delta))  # Wider spreads for OTM options
                bid = max(0.01, put_price * (1 - spread_pct))
                ask = put_price * (1 + spread_pct)
                
                puts[float(strike)] = {
                    'strike': strike,
                    'bid': round(bid, 2),
                    'ask': round(ask, 2),
                    'last': round((bid + ask) / 2, 2),
                    'volume': int(np.random.randint(10, 1000) * (1.5 - abs(moneyness))),
                    'open_interest': int(np.random.randint(100, 5000) * (1.5 - abs(moneyness))),
                    'delta': round(delta, 3),
                    'gamma': round(gamma, 4),
                    'theta': round(theta, 3),
                    'vega': round(vega, 3),
                    'iv': round(strike_iv, 3)
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
    
    def clear_cache(self, symbol=None) -> None:
        """
        Clear cached data.
        
        Args:
            symbol: Symbol to clear cache for (None for all)
        """
        if not os.path.exists(self.data_dir):
            return
            
        files = os.listdir(self.data_dir)
        
        for file in files:
            if file.endswith('realistic_mock.csv'):
                if symbol is None or file.startswith(f"{symbol}_"):
                    os.remove(os.path.join(self.data_dir, file))
                    
    def get_iv_hv_ratio(self, symbol: str, date=None, expiry=None) -> float:
        """
        Generate synthetic IV/HV ratio for backtesting.
        
        Args:
            symbol: Ticker symbol
            date: Date to get ratio for (ignored in mock)
            expiry: Option expiry (ignored in mock)
            
        Returns:
            Float IV/HV ratio
        """
        # Generate a realistic but elevated IV/HV ratio for testing
        # Randomly generate between the min and max values set in the constructor
        return np.random.uniform(self.iv_hv_ratio_min, self.iv_hv_ratio_max)
                    self.logger.info(f"Removed mock cache file: {file}")