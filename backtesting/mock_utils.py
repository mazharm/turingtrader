\
# filepath: c:\\Users\\mazharm\\code\\turingtrader\\backtesting\\mock_utils.py
\"\"\"
Utilities for mock data generation for backtesting.
\"\"\"
import logging
import random
from datetime import timedelta

import numpy as np
import pandas as pd


class MockDataFetcher:
    \"\"\"Mock data fetcher to avoid database dependencies for testing.\"\"\"

    def __init__(self, seed: int = 42):
        \"\"\"
        Initialize mock data fetcher.
        Args:
            seed: Random seed for reproducibility.
        \"\"\"
        self.logger = logging.getLogger(__name__)
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    def fetch_data(self, symbol: str, start_date_str: str, end_date_str: str, days_limit: Optional[int] = None) -> pd.DataFrame:
        \"\"\"
        Generate mock historical price data for backtesting.

        Args:
            symbol: The symbol to fetch data for (e.g., 'SPY', 'VIX').
            start_date_str: Start date string (YYYY-MM-DD).
            end_date_str: End date string (YYYY-MM-DD).
            days_limit: Optional limit for the number of days of data to generate from start_date.

        Returns:
            pd.DataFrame: A DataFrame with mock historical data.
        \"\"\"
        self.logger.debug(f"Generating mock data for {symbol} from {start_date_str} to {end_date_str} (limit: {days_limit} days)")
        
        start_date = pd.to_datetime(start_date_str)
        if days_limit is not None:
            end_date = start_date + timedelta(days=days_limit -1) # -1 because date_range is inclusive
        else:
            end_date = pd.to_datetime(end_date_str)

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        if not len(dates):
             return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])


        if symbol == 'VIX':
            initial_price = 20.0
            volatility = 0.15  # Higher volatility for VIX
        else:  # Assume SPY or other index
            initial_price = 400.0
            volatility = 0.01

        prices = [initial_price]
        for i in range(1, len(dates)):
            if symbol == 'VIX':
                mean_reversion = 0.05 * (20.0 - prices[-1])
                spike = 0.0
                if random.random() < 0.05:  # 5% chance of spike
                    spike = random.uniform(3.0, 10.0) * random.choice([-1,1]) # Spike can be up or down
                change = mean_reversion + spike + np.random.normal(0, volatility * prices[-1])
                new_price = max(5.0, prices[-1] + change) # VIX usually has a floor
            else: # SPY
                trend = 0.0002  # Slight upward bias
                change = trend * prices[-1] + np.random.normal(0, volatility * prices[-1])
                new_price = max(0.1, prices[-1] + change)
            prices.append(new_price)

        df = pd.DataFrame({
            'open': [p * (1 - volatility * random.uniform(0, 0.1)) for p in prices], # open slightly different from close
            'high': [p * (1 + volatility * random.uniform(0, 0.5)) for p in prices],
            'low': [p * (1 - volatility * random.uniform(0, 0.5)) for p in prices],
            'close': prices,
            'volume': [int(1e6 * random.uniform(0.5, 1.5)) for _ in prices]
        }, index=dates)
        
        # Ensure high is highest and low is lowest
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        self.logger.info(f"Generated {len(df)} rows of mock data for {symbol}")
        return df

    def fetch_vix_data(self, start_date: str, end_date: str, days_limit: Optional[int] = None) -> pd.DataFrame:
        \"\"\"Fetch mock VIX data.\"\"\"
        return self.fetch_data('VIX', start_date, end_date, days_limit=days_limit)

    def fetch_sp500_data(self, start_date: str, end_date: str, days_limit: Optional[int] = None) -> pd.DataFrame:
        \"\"\"Fetch mock S&P500 data.\"\"\"
        return self.fetch_data('SPY', start_date, end_date, days_limit=days_limit)
