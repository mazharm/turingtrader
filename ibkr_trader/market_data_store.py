"""
Market data store module for the TuringTrader algorithm.
Handles storing and retrieving market data from databases, with backfilling capabilities.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import pandas as pd
    import numpy as np
    import psycopg2
    import redis
    from psycopg2 import pool
    from psycopg2.extras import execute_values
except ImportError as e:
    logging.error(f"Required package missing: {e}")
    logging.error("Install with: pip install pandas numpy psycopg2-binary redis")
    raise

from .config import Config
from .database_config import PostgresConfig, RedisConfig


class MarketDataStore:
    """
    Store and retrieve market data from TimescaleDB and Redis.
    Handles caching and backfilling of data.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the market data store.
        
        Args:
            config: Configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or Config()
        
        # Initialize connections
        self._pg_pool = None
        self._redis_client = None
        
        try:
            # Create PostgreSQL connection pool
            self._pg_pool = pool.ThreadedConnectionPool(
                minconn=self.config.postgres.min_connections,
                maxconn=self.config.postgres.max_connections,
                dsn=self.config.postgres.get_connection_string()
            )
            self.logger.info(f"Connected to PostgreSQL at {self.config.postgres.host}:{self.config.postgres.port}")
            
            # Initialize Redis connection
            self._redis_client = redis.from_url(self.config.redis.get_connection_string())
            self._redis_client.ping()  # Test connection
            self.logger.info(f"Connected to Redis at {self.config.redis.host}:{self.config.redis.port}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database connections: {e}")
            
            # Clean up if partial initialization
            if self._pg_pool:
                self._pg_pool.closeall()
                self._pg_pool = None
    
    def __del__(self):
        """Clean up database connections."""
        if self._pg_pool:
            self._pg_pool.closeall()
        
    def _get_pg_conn(self):
        """Get a PostgreSQL connection from the pool."""
        if not self._pg_pool:
            raise RuntimeError("PostgreSQL connection pool not initialized")
        return self._pg_pool.getconn()
    
    def _put_pg_conn(self, conn):
        """Return a PostgreSQL connection to the pool."""
        if self._pg_pool:
            self._pg_pool.putconn(conn)
    
    def store_market_data(self, data: List[Dict], symbol: str, source: str = 'IBKR') -> bool:
        """
        Store market data in the database.
        
        Args:
            data: List of market data points with date/time, OHLCV
            symbol: Ticker symbol
            source: Data source (e.g., 'IBKR', 'Yahoo')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not data:
            return True  # Nothing to store
        
        try:
            # Convert to standardized format
            records = []
            
            for bar in data:
                # Handle different date formats
                if isinstance(bar.get('date'), str):
                    try:
                        dt = datetime.fromisoformat(bar['date'].replace('Z', '+00:00'))
                    except ValueError:
                        dt = pd.to_datetime(bar['date']).to_pydatetime()
                else:
                    dt = bar.get('date')
                
                records.append((
                    dt,
                    symbol,
                    bar.get('open', None),
                    bar.get('high', None),
                    bar.get('low', None),
                    bar.get('close', None),
                    bar.get('volume', None),
                    source
                ))
            
            # Get DB connection
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    # Insert multiple rows efficiently
                    query = """
                        INSERT INTO market_data 
                        (time, symbol, open, high, low, close, volume, source)
                        VALUES %s
                        ON CONFLICT (time, symbol) DO UPDATE
                        SET open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            source = EXCLUDED.source
                    """
                    execute_values(cur, query, records)
                conn.commit()
                
                # Update Redis cache with latest close
                if data and 'close' in data[-1]:
                    latest = data[-1]
                    key = f"quote:{symbol}:last"
                    self._redis_client.hset(
                        key,
                        mapping={
                            'price': str(latest['close']),
                            'time': str(latest['date']),
                            'source': source
                        }
                    )
                    self._redis_client.expire(key, self.config.redis.default_expiry)
                
                return True
            
            finally:
                self._put_pg_conn(conn)
                
        except Exception as e:
            self.logger.error(f"Error storing market data for {symbol}: {e}")
            return False
    
    def store_options_data(self, data: Dict, symbol: str) -> bool:
        """
        Store options data in the database.
        
        Args:
            data: Options chain data
            symbol: Underlying symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not data:
            return True  # Nothing to store
        
        try:
            # Get timestamp for this data point
            timestamp = datetime.now()
            
            # Convert chain data to records
            records = []
            
            for expiry, expiry_data in data.items():
                # Parse expiry date
                try:
                    expiry_date = datetime.strptime(expiry, '%Y%m%d').date()
                except ValueError:
                    self.logger.error(f"Invalid expiry format: {expiry}")
                    continue
                
                # Process calls
                for strike, call_data in expiry_data.get('calls', {}).items():
                    records.append((
                        timestamp,
                        symbol,
                        expiry_date,
                        float(strike),
                        'C',  # Call
                        call_data.get('bid', None),
                        call_data.get('ask', None),
                        call_data.get('last', None),
                        call_data.get('volume', None),
                        call_data.get('open_interest', None),
                        call_data.get('iv', None)
                    ))
                
                # Process puts
                for strike, put_data in expiry_data.get('puts', {}).items():
                    records.append((
                        timestamp,
                        symbol,
                        expiry_date,
                        float(strike),
                        'P',  # Put
                        put_data.get('bid', None),
                        put_data.get('ask', None),
                        put_data.get('last', None),
                        put_data.get('volume', None),
                        put_data.get('open_interest', None),
                        put_data.get('iv', None)
                    ))
            
            # Store in database
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO options_data 
                        (time, symbol, expiry, strike, option_type, bid, ask, last, volume, open_interest, implied_volatility)
                        VALUES %s
                    """
                    execute_values(cur, query, records)
                conn.commit()
                
                # Cache IV data in Redis
                self._cache_iv_data(symbol, data)
                
                return True
            
            finally:
                self._put_pg_conn(conn)
                
        except Exception as e:
            self.logger.error(f"Error storing options data for {symbol}: {e}")
            return False
    
    def _cache_iv_data(self, symbol: str, options_data: Dict) -> None:
        """Cache key implied volatility data in Redis."""
        try:
            # Extract average IVs by expiry
            for expiry, expiry_data in options_data.items():
                # Get all IVs for calls and puts
                call_ivs = [opt.get('iv', 0) for opt in expiry_data.get('calls', {}).values() if opt.get('iv', 0) > 0]
                put_ivs = [opt.get('iv', 0) for opt in expiry_data.get('puts', {}).values() if opt.get('iv', 0) > 0]
                
                # Calculate averages if we have data
                if call_ivs:
                    avg_call_iv = sum(call_ivs) / len(call_ivs)
                    # Cache call IV
                    self._redis_client.hset(
                        f"iv:{symbol}:{expiry}",
                        mapping={
                            'call': str(avg_call_iv),
                            'time': datetime.now().isoformat()
                        }
                    )
                
                if put_ivs:
                    avg_put_iv = sum(put_ivs) / len(put_ivs)
                    # Cache put IV
                    self._redis_client.hset(
                        f"iv:{symbol}:{expiry}",
                        mapping={
                            'put': str(avg_put_iv),
                            'time': datetime.now().isoformat()
                        }
                    )
                
                # Set expiry
                self._redis_client.expire(f"iv:{symbol}:{expiry}", self.config.redis.default_expiry)
                
        except Exception as e:
            self.logger.error(f"Error caching IV data: {e}")
    
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime = None,
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Get market data for a symbol between dates.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date (defaults to now)
            use_cache: Whether to use Redis cache for recent data
            
        Returns:
            DataFrame with market data
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Try to get from cache for very recent data
        latest_data = None
        if use_cache and (datetime.now() - end_date).total_seconds() < 300:  # Last 5 minutes
            try:
                cache_key = f"quote:{symbol}:last"
                cached_data = self._redis_client.hgetall(cache_key)
                
                if cached_data and b'price' in cached_data:
                    price = float(cached_data[b'price'])
                    time_str = cached_data[b'time'].decode('utf-8')
                    time_obj = pd.to_datetime(time_str)
                    
                    latest_data = pd.DataFrame([{
                        'time': time_obj,
                        'symbol': symbol,
                        'close': price
                    }])
            except Exception as e:
                self.logger.warning(f"Error getting cached market data: {e}")
        
        # Get data from database
        try:
            conn = self._get_pg_conn()
            query = """
                SELECT time, open, high, low, close, volume
                FROM market_data
                WHERE symbol = %s AND time BETWEEN %s AND %s
                ORDER BY time
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            self._put_pg_conn(conn)
            
            # If we got cached data and it's newer than our query result
            if latest_data is not None and (not df.empty and latest_data['time'].iloc[0] > df['time'].iloc[-1]):
                df = pd.concat([df, latest_data[['time', 'open', 'high', 'low', 'close', 'volume']]])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            if conn:
                self._put_pg_conn(conn)
            return pd.DataFrame()
    
    def get_options_data(self, symbol: str, expiry: Optional[str] = None, 
                        strike_min: Optional[float] = None, strike_max: Optional[float] = None) -> Dict:
        """
        Get options data for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiry: Optional expiry date (YYYYMMDD format)
            strike_min: Minimum strike price
            strike_max: Maximum strike price
            
        Returns:
            Dictionary with options chain data
        """
        result = {}
        
        try:
            # Build query conditions
            conditions = ["symbol = %s"]
            params = [symbol]
            
            if expiry:
                try:
                    expiry_date = datetime.strptime(expiry, '%Y%m%d').date()
                    conditions.append("expiry = %s")
                    params.append(expiry_date)
                except ValueError:
                    self.logger.error(f"Invalid expiry format: {expiry}")
            
            if strike_min is not None:
                conditions.append("strike >= %s")
                params.append(float(strike_min))
                
            if strike_max is not None:
                conditions.append("strike <= %s")
                params.append(float(strike_max))
            
            # Get latest timestamp for each expiry/strike/type combination
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    # First get the latest timestamp for our data
                    latest_query = f"""
                        SELECT MAX(time) FROM options_data
                        WHERE {" AND ".join(conditions)}
                    """
                    cur.execute(latest_query, params)
                    latest_time = cur.fetchone()[0]
                    
                    if not latest_time:
                        self.logger.warning(f"No options data found for {symbol}")
                        return {}
                    
                    # Now get the options data at that timestamp
                    query = f"""
                        SELECT time, expiry, strike, option_type, bid, ask, last, 
                               volume, open_interest, implied_volatility
                        FROM options_data
                        WHERE {" AND ".join(conditions)} AND time = %s
                        ORDER BY expiry, strike, option_type
                    """
                    cur.execute(query, params + [latest_time])
                    
                    # Process results
                    for row in cur.fetchall():
                        (time, expiry_date, strike, option_type, 
                         bid, ask, last, volume, open_interest, iv) = row
                        
                        # Convert expiry to string format
                        expiry_str = expiry_date.strftime('%Y%m%d')
                        
                        # Initialize expiry dict if not exists
                        if expiry_str not in result:
                            result[expiry_str] = {
                                'days_to_expiry': (expiry_date - datetime.now().date()).days,
                                'calls': {},
                                'puts': {},
                                'strikes': []
                            }
                            
                        # Add strike if not in list
                        if strike not in result[expiry_str]['strikes']:
                            result[expiry_str]['strikes'].append(strike)
                            
                        # Add option data
                        option_data = {
                            'bid': bid,
                            'ask': ask,
                            'last': last,
                            'volume': volume,
                            'open_interest': open_interest,
                            'iv': iv
                        }
                        
                        if option_type == 'C':
                            result[expiry_str]['calls'][strike] = option_data
                        else:
                            result[expiry_str]['puts'][strike] = option_data
                
                # Sort strikes for each expiry
                for expiry_data in result.values():
                    expiry_data['strikes'].sort()
                
                return result
                
            finally:
                self._put_pg_conn(conn)
                
        except Exception as e:
            self.logger.error(f"Error getting options data for {symbol}: {e}")
            return {}
    
    def calculate_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """
        Calculate historical volatility for a symbol.
        
        Args:
            symbol: Ticker symbol
            days: Number of days to use for calculation
            
        Returns:
            float: Annualized historical volatility percentage
        """
        try:
            # Get market data for the past days + 10 (buffer)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+10)
            
            df = self.get_market_data(symbol, start_date, end_date)
            
            if len(df) < days:
                self.logger.warning(f"Insufficient data to calculate historical volatility for {symbol}")
                return 0.0
                
            # Calculate daily log returns
            df['return'] = np.log(df['close'] / df['close'].shift(1))
            df = df.dropna()
            
            # Calculate standard deviation of returns
            std_dev = df['return'].iloc[-days:].std()
            
            # Annualize volatility (252 trading days)
            annualized_vol = std_dev * np.sqrt(252) * 100
            
            return annualized_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility for {symbol}: {e}")
            return 0.0
    
    def get_iv_hv_ratio(self, symbol: str, expiry: str = None) -> float:
        """
        Calculate IV/HV ratio for a symbol.
        
        Args:
            symbol: Ticker symbol
            expiry: Option expiry (if None, uses nearest expiry)
            
        Returns:
            float: IV/HV ratio
        """
        try:
            # Get HV
            hv = self.calculate_historical_volatility(symbol)
            if hv <= 0:
                return 0.0
                
            # Get current option chain if no specific expiry
            options_data = {}
            if expiry is None:
                options_data = self.get_options_data(symbol)
                if not options_data:
                    return 0.0
                
                # Find nearest expiry with 14-45 DTE
                valid_expiries = [exp for exp, data in options_data.items() 
                                if 14 <= data.get('days_to_expiry', 0) <= 45]
                
                if not valid_expiries:
                    # If no valid expiry, use the nearest one
                    valid_expiries = list(options_data.keys())
                    
                if valid_expiries:
                    expiry = min(valid_expiries, key=lambda x: options_data[x]['days_to_expiry'])
            
            # Try to get IV from Redis cache
            iv = 0.0
            try:
                iv_data = self._redis_client.hgetall(f"iv:{symbol}:{expiry}")
                if b'call' in iv_data and b'put' in iv_data:
                    call_iv = float(iv_data[b'call'])
                    put_iv = float(iv_data[b'put'])
                    iv = (call_iv + put_iv) / 2
            except Exception as e:
                self.logger.error(f"Error getting IV from cache: {e}")
            
            # If no IV in cache, get from options data
            if iv <= 0 and options_data and expiry in options_data:
                calls = options_data[expiry].get('calls', {})
                puts = options_data[expiry].get('puts', {})
                
                call_ivs = [opt.get('iv', 0) for opt in calls.values() if opt.get('iv', 0) > 0]
                put_ivs = [opt.get('iv', 0) for opt in puts.values() if opt.get('iv', 0) > 0]
                
                if call_ivs and put_ivs:
                    avg_call_iv = sum(call_ivs) / len(call_ivs)
                    avg_put_iv = sum(put_ivs) / len(put_ivs)
                    iv = (avg_call_iv + avg_put_iv) / 2
            
            # Convert IV to percentage
            iv = iv * 100
            
            # Calculate ratio
            if hv > 0 and iv > 0:
                return iv / hv
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating IV/HV ratio for {symbol}: {e}")
            return 0.0
    
    def backfill_market_data(self, symbol: str, start_date: datetime, end_date: datetime = None, 
                            source: str = 'YF', save: bool = True) -> pd.DataFrame:
        """
        Backfill missing market data from an external source.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date (defaults to now)
            source: Data source ('YF' for Yahoo Finance)
            save: Whether to save data to database
            
        Returns:
            DataFrame with backfilled data
        """
        if end_date is None:
            end_date = datetime.now()
            
        try:
            # Import yfinance if needed
            import yfinance as yf
            
            # Get existing data from DB
            existing_data = self.get_market_data(symbol, start_date, end_date)
            
            # Download data from Yahoo Finance
            yf_data = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1))
            
            if yf_data.empty:
                self.logger.warning(f"No data available from Yahoo Finance for {symbol}")
                return existing_data
                
            # Convert to our format
            df = yf_data.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Save to database if requested
            if save:
                records = df.to_dict('records')
                self.store_market_data(records, symbol, source)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error backfilling market data for {symbol}: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test market data store
    logging.basicConfig(level=logging.INFO)
    
    # Create store
    store = MarketDataStore()
    
    # Test storing market data
    test_data = [
        {
            'date': datetime.now() - timedelta(minutes=5),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        },
        {
            'date': datetime.now(),
            'open': 100.5,
            'high': 102.0,
            'low': 100.0,
            'close': 101.5,
            'volume': 1500
        }
    ]
    
    print("Storing test market data...")
    result = store.store_market_data(test_data, 'TEST', 'test')
    print(f"Store result: {result}")
    
    # Test retrieving market data
    print("Retrieving test market data...")
    data = store.get_market_data('TEST', datetime.now() - timedelta(hours=1), datetime.now())
    print(f"Retrieved {len(data)} records")
    print(data.head())
    
    # Test HV calculation
    print("Calculating historical volatility for SPY...")
    hv = store.calculate_historical_volatility('SPY')
    print(f"SPY historical volatility: {hv:.2f}%")
    
    # Test backfilling
    print("Backfilling SPY data for last 30 days...")
    backfilled = store.backfill_market_data('SPY', datetime.now() - timedelta(days=30))
    print(f"Backfilled {len(backfilled)} records")