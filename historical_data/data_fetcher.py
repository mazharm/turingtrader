"""
Historical data fetching module for the TuringTrader algorithm.
"""

import logging
import os
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# Try to import yfinance for data fetching
try:
    import yfinance as yf
except ImportError:
    logging.warning("yfinance package not found. Please install with: pip install yfinance")


class HistoricalDataFetcher:
    """Fetch and manage historical market data."""
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the data fetcher.
        
        Args:
            data_dir: Directory for storing cached data
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_data(self, 
                 symbol: str, 
                 start_date: Union[str, datetime],
                 end_date: Union[str, datetime],
                 interval: str = '1d',
                 use_cache: bool = True,
                 max_retries: int = 3,
                 retry_delay: int = 2) -> pd.DataFrame:
        """
        Fetch historical market data for a symbol.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval ('1d', '1wk', etc.)
            use_cache: Whether to use cached data if available
            max_retries: Maximum number of retries for failed fetches
            retry_delay: Delay between retries (in seconds)
            
        Returns:
            DataFrame with historical data
        """
        # Convert dates to strings if they are datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Check if we can use cached data
        cache_file = os.path.join(self.data_dir, f"{symbol}_{start_date}_{end_date}_{interval}.csv")
        
        if use_cache and os.path.exists(cache_file):
            self.logger.info(f"Loading cached data for {symbol} from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Fetch data using yfinance with retries
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} (attempt {attempt+1}/{max_retries})")
                data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
                
                # Check if we got valid data
                if data.empty:
                    self.logger.warning(f"No data found for {symbol} (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        # All retries failed
                        break
                
                # Rename columns to lowercase
                data.columns = [c.lower() for c in data.columns]
                
                # Save to cache
                data.to_csv(cache_file)
                
                self.logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
                return data
                
            except Exception as e:
                self.logger.warning(f"Error fetching data for {symbol} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
        # All retries failed, try loading from cache as fallback
        self.logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
        if os.path.exists(cache_file):
            self.logger.info(f"Falling back to cached data for {symbol}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
        return pd.DataFrame()
    
    def fetch_vix_data(self, 
                      start_date: Union[str, datetime],
                      end_date: Union[str, datetime],
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch VIX data.
        
        Args:
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with VIX data
        """
        return self.fetch_data('^VIX', start_date, end_date, use_cache=use_cache)
    
    def fetch_sp500_data(self, 
                        start_date: Union[str, datetime],
                        end_date: Union[str, datetime],
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch S&P 500 data.
        
        Args:
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with S&P 500 data
        """
        return self.fetch_data('SPY', start_date, end_date, use_cache=use_cache)
    
    def fetch_option_chain(self, 
                         symbol: str, 
                         date: Optional[Union[str, datetime]] = None,
                         max_retries: int = 3,
                         retry_delay: int = 2) -> Dict:
        """
        Fetch option chain data for a symbol.
        
        Args:
            symbol: Ticker symbol
            date: Specific date for options (None for current options)
            max_retries: Maximum number of retries for failed fetches
            retry_delay: Delay between retries (in seconds)
            
        Returns:
            Dictionary with option chain data
        """
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching option chain for {symbol} (attempt {attempt+1}/{max_retries})")
                ticker = yf.Ticker(symbol)
                
                # Get option expiration dates
                expirations = ticker.options
                
                if not expirations:
                    self.logger.warning(f"No options data found for {symbol} (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        # All retries failed
                        return {}
                    
                # If date is specified, use it if it's in the list
                if date:
                    if isinstance(date, datetime):
                        date_str = date.strftime('%Y-%m-%d')
                    else:
                        date_str = date
                        
                    # Find the closest expiration date
                    closest = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(date_str, '%Y-%m-%d')).days))
                    expirations = [closest]
                
                # Fetch option chains for each expiration
                option_chain = {}
                
                for exp_date in expirations:
                    try:
                        options = ticker.option_chain(exp_date)
                        calls = options.calls
                        puts = options.puts
                        
                        # Convert to standardized format
                        exp_date_fmt = exp_date.replace('-', '')
                        
                        # Calculate days to expiry
                        days_to_expiry = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days
                        days_to_expiry = max(0, days_to_expiry)
                        
                        calls_dict = {}
                        for _, row in calls.iterrows():
                            calls_dict[float(row['strike'])] = {
                                'bid': row['bid'],
                                'ask': row['ask'],
                                'last': row['lastPrice'],
                                'volume': row['volume'],
                                'open_interest': row['openInterest'],
                                'iv': row['impliedVolatility']
                            }
                            
                        puts_dict = {}
                        for _, row in puts.iterrows():
                            puts_dict[float(row['strike'])] = {
                                'bid': row['bid'],
                                'ask': row['ask'],
                                'last': row['lastPrice'],
                                'volume': row['volume'],
                                'open_interest': row['openInterest'],
                                'iv': row['impliedVolatility']
                            }
                        
                        # Get unique strikes
                        all_strikes = sorted(list(set(list(calls_dict.keys()) + list(puts_dict.keys()))))
                        
                        option_chain[exp_date_fmt] = {
                            'days_to_expiry': days_to_expiry,
                            'calls': calls_dict,
                            'puts': puts_dict,
                            'strikes': all_strikes
                        }
                    except Exception as e:
                        self.logger.warning(f"Error fetching option chain for expiry {exp_date}: {e}")
                        # Continue with other expiry dates
                
                if option_chain:
                    self.logger.info(f"Successfully fetched option chain for {symbol} with {len(option_chain)} expiry dates")
                    return option_chain
                else:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"No valid option data found for {symbol} (attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {}
                    
            except Exception as e:
                self.logger.warning(f"Error fetching option chain for {symbol} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
        # All retries failed
        self.logger.error(f"Failed to fetch option chain for {symbol} after {max_retries} attempts")
        return {}
    
    def get_cached_symbols(self) -> List[str]:
        """
        Get list of symbols with cached data.
        
        Returns:
            List of cached symbols
        """
        if not os.path.exists(self.data_dir):
            return []
            
        files = os.listdir(self.data_dir)
        symbols = set()
        
        for file in files:
            if file.endswith('.csv'):
                # Extract symbol from filename (format: SYMBOL_START_END_INTERVAL.csv)
                parts = file.split('_')
                if len(parts) >= 4:
                    symbols.add(parts[0])
        
        return list(symbols)
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            symbol: Symbol to clear cache for (None for all)
        """
        if not os.path.exists(self.data_dir):
            return
            
        files = os.listdir(self.data_dir)
        
        for file in files:
            if file.endswith('.csv'):
                if symbol is None or file.startswith(f"{symbol}_"):
                    os.remove(os.path.join(self.data_dir, file))
                    self.logger.info(f"Removed cache file: {file}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the data fetcher
    fetcher = HistoricalDataFetcher()
    
    # Fetch S&P 500 data for the last 3 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    spy_data = fetcher.fetch_sp500_data(start_date, end_date)
    print(f"Fetched {len(spy_data)} rows of SPY data")
    if not spy_data.empty:
        print(spy_data.head())
        
    # Fetch VIX data
    vix_data = fetcher.fetch_vix_data(start_date, end_date)
    print(f"Fetched {len(vix_data)} rows of VIX data")
    if not vix_data.empty:
        print(vix_data.head())