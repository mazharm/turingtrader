"""
Data processing module for the TuringTrader algorithm.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

from .data_fetcher import HistoricalDataFetcher


class DataProcessor:
    """Process and prepare historical market data for analysis and backtesting."""
    
    def __init__(self, data_fetcher: Optional[HistoricalDataFetcher] = None):
        """
        Initialize the data processor.
        
        Args:
            data_fetcher: Historical data fetcher instance
        """
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = data_fetcher or HistoricalDataFetcher()
    
    def calculate_historical_volatility(self, 
                                       data: pd.DataFrame, 
                                       window: int = 20, 
                                       trading_days: int = 252,
                                       price_col: str = 'close') -> pd.Series:
        """
        Calculate historical volatility from price data.
        
        Args:
            data: DataFrame with historical prices
            window: Rolling window size
            trading_days: Number of trading days in a year
            price_col: Column name for price data
            
        Returns:
            Series with historical volatility values
        """
        if data.empty:
            return pd.Series()
            
        try:
            # Calculate log returns
            log_returns = np.log(data[price_col] / data[price_col].shift(1))
            
            # Calculate rolling standard deviation of returns
            rolling_std = log_returns.rolling(window=window).std()
            
            # Annualize the volatility
            volatility = rolling_std * np.sqrt(trading_days) * 100
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility: {e}")
            return pd.Series()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for market data.
        
        Args:
            data: DataFrame with historical prices
            
        Returns:
            DataFrame with added technical indicators
        """
        if data.empty:
            return data
            
        try:
            df = data.copy()
            
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Historical volatility
            df['hist_vol_20'] = self.calculate_historical_volatility(df, window=20)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    def merge_market_data(self, 
                        underlying_data: pd.DataFrame, 
                        vix_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge underlying and VIX data.
        
        Args:
            underlying_data: DataFrame with underlying price data
            vix_data: DataFrame with VIX data
            
        Returns:
            DataFrame with merged data
        """
        if underlying_data.empty or vix_data.empty:
            if underlying_data.empty:
                self.logger.error("Underlying data is empty")
            if vix_data.empty:
                self.logger.error("VIX data is empty")
            return pd.DataFrame()
            
        try:
            # Resample both to ensure same frequency
            underlying = underlying_data.copy()
            vix = vix_data.copy()
            
            # Rename VIX columns to avoid conflicts
            vix_columns = {col: f"vix_{col}" for col in vix.columns}
            vix = vix.rename(columns=vix_columns)
            
            # Merge on date index
            merged = pd.merge(underlying, vix, left_index=True, right_index=True, how='inner')
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging market data: {e}")
            return pd.DataFrame()
    
    def prepare_backtest_data(self, 
                            symbol: str, 
                            start_date: Union[str, pd.Timestamp],
                            end_date: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """
        Prepare a complete dataset for backtesting.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with prepared data
        """
        try:
            # Fetch underlying data
            underlying_data = self.data_fetcher.fetch_data(symbol, start_date, end_date)
            
            # Fetch VIX data
            vix_data = self.data_fetcher.fetch_vix_data(start_date, end_date)
            
            # Apply technical indicators
            underlying_with_indicators = self.calculate_technical_indicators(underlying_data)
            
            # Merge datasets
            merged_data = self.merge_market_data(underlying_with_indicators, vix_data)
            
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error preparing backtest data: {e}")
            return pd.DataFrame()
    
    def generate_option_prices(self, 
                             underlying_price: float, 
                             volatility: float,
                             days_to_expiry: int,
                             strike_pct_range: float = 0.2,
                             strike_steps: int = 10,
                             risk_free_rate: float = 0.03) -> Dict:
        """
        Generate synthetic option prices based on Black-Scholes model.
        
        Args:
            underlying_price: Price of the underlying
            volatility: Implied volatility (percentage)
            days_to_expiry: Days to expiration
            strike_pct_range: Strike price range as percentage of underlying price
            strike_steps: Number of strike prices to generate
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Dictionary with synthetic option chain
        """
        import math
        from scipy.stats import norm
        
        # Convert days to years
        t = days_to_expiry / 365.0
        
        # Convert volatility from percentage to decimal
        sigma = volatility / 100.0
        
        # Generate strike prices
        min_strike = underlying_price * (1 - strike_pct_range)
        max_strike = underlying_price * (1 + strike_pct_range)
        strikes = np.linspace(min_strike, max_strike, strike_steps)
        
        # Generate option prices
        calls = {}
        puts = {}
        
        for strike in strikes:
            # Black-Scholes formula components
            d1 = (math.log(underlying_price / strike) + (risk_free_rate + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
            d2 = d1 - sigma * math.sqrt(t)
            
            # Call price
            call_price = underlying_price * norm.cdf(d1) - strike * math.exp(-risk_free_rate * t) * norm.cdf(d2)
            
            # Put price using put-call parity
            put_price = call_price + strike * math.exp(-risk_free_rate * t) - underlying_price
            
            # Calculate option greeks
            call_delta = norm.cdf(d1)
            put_delta = call_delta - 1
            
            gamma = norm.pdf(d1) / (underlying_price * sigma * math.sqrt(t))
            
            call_theta = -(underlying_price * sigma * norm.pdf(d1)) / (2 * math.sqrt(t)) - risk_free_rate * strike * math.exp(-risk_free_rate * t) * norm.cdf(d2)
            put_theta = call_theta + risk_free_rate * strike * math.exp(-risk_free_rate * t)
            
            vega = underlying_price * math.sqrt(t) * norm.pdf(d1) / 100  # Vega per 1% change in IV
            
            # Add bid-ask spread
            call_bid = max(0.01, call_price * 0.95)
            call_ask = call_price * 1.05
            put_bid = max(0.01, put_price * 0.95)
            put_ask = put_price * 1.05
            
            # Store in dictionaries
            calls[float(strike)] = {
                'bid': round(call_bid, 2),
                'ask': round(call_ask, 2),
                'last': round(call_price, 2),
                'iv': sigma,
                'delta': round(call_delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(call_theta, 4),
                'vega': round(vega, 4)
            }
            
            puts[float(strike)] = {
                'bid': round(put_bid, 2),
                'ask': round(put_ask, 2),
                'last': round(put_price, 2),
                'iv': sigma,
                'delta': round(put_delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(put_theta, 4),
                'vega': round(vega, 4)
            }
        
        # Create the structure similar to real option chain
        expiry_date = (pd.Timestamp.now() + pd.Timedelta(days=days_to_expiry)).strftime('%Y%m%d')
        
        return {
            expiry_date: {
                'days_to_expiry': days_to_expiry,
                'calls': calls,
                'puts': puts,
                'strikes': [float(strike) for strike in strikes]
            }
        }
    
    def generate_historical_options_data(self,
                                       underlying_data: pd.DataFrame,
                                       vix_data: pd.DataFrame,
                                       dte_values: List[int] = [7, 14, 30, 60]) -> Dict:
        """
        Generate synthetic options data for historical backtesting.
        
        Args:
            underlying_data: DataFrame with underlying price data
            vix_data: DataFrame with VIX data
            dte_values: List of days-to-expiry values
            
        Returns:
            Dictionary with historical options data by date
        """
        if underlying_data.empty or vix_data.empty:
            return {}
            
        # Merge the data
        merged = self.merge_market_data(underlying_data, vix_data)
        
        if merged.empty:
            return {}
            
        historical_options = {}
        
        # For each date, generate option prices
        for date, row in merged.iterrows():
            underlying_price = row['close']
            vix_value = row['vix_close']  # Use VIX as implied volatility
            
            # Generate options for each DTE
            date_options = {}
            for dte in dte_values:
                option_chain = self.generate_option_prices(
                    underlying_price=underlying_price,
                    volatility=vix_value,
                    days_to_expiry=dte
                )
                date_options.update(option_chain)
            
            historical_options[date] = date_options
        
        return historical_options


# Module import guard
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the data processor
    fetcher = HistoricalDataFetcher()
    processor = DataProcessor(fetcher)
    
    # Fetch some test data
    import datetime as dt
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=90)
    
    spy_data = fetcher.fetch_sp500_data(start_date, end_date)
    vix_data = fetcher.fetch_vix_data(start_date, end_date)
    
    # Calculate volatility
    if not spy_data.empty:
        volatility = processor.calculate_historical_volatility(spy_data)
        print("Historical Volatility (last 5 days):")
        print(volatility.tail())
    
    # Calculate technical indicators
    if not spy_data.empty:
        with_indicators = processor.calculate_technical_indicators(spy_data)
        print("\nTechnical Indicators (last day):")
        print(with_indicators.iloc[-1][['close', 'sma_20', 'ema_12', 'rsi', 'hist_vol_20']])
    
    # Generate sample option prices
    if not spy_data.empty and not spy_data.empty:
        current_price = spy_data['close'].iloc[-1]
        current_vix = vix_data['close'].iloc[-1]
        
        print(f"\nGenerating option prices for: Price=${current_price:.2f}, VIX={current_vix:.2f}%")
        options = processor.generate_option_prices(
            underlying_price=current_price,
            volatility=current_vix,
            days_to_expiry=30
        )
        
        # Show some sample options
        exp_date = list(options.keys())[0]
        strikes = options[exp_date]['strikes']
        middle_strike_idx = len(strikes) // 2
        
        print(f"\nSample Call Option (Strike: ${strikes[middle_strike_idx]:.2f}, Expiry: {exp_date}):")
        print(options[exp_date]['calls'][strikes[middle_strike_idx]])
        
        print(f"\nSample Put Option (Strike: ${strikes[middle_strike_idx]:.2f}, Expiry: {exp_date}):")
        print(options[exp_date]['puts'][strikes[middle_strike_idx]])