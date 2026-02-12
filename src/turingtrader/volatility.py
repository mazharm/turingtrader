"""Volatility assessment module for TuringTrader."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """Analyzer for market volatility to determine trading opportunities.
    
    This class assesses market volatility using various measures and indicators
    to help determine when to trade options based on volatility conditions.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        volatility_threshold: float = 0.15,
        vix_threshold: float = 20.0,
    ):
        """Initialize the volatility analyzer.
        
        Args:
            lookback_period (int): Number of days to use for historical calculations
            volatility_threshold (float): Threshold above which volatility is considered high
            vix_threshold (float): VIX threshold above which volatility is considered high
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.vix_threshold = vix_threshold
        self.data_cache = {}
    
    def fetch_historical_data(self, ticker: str = "^GSPC", period: str = "60d") -> pd.DataFrame:
        """Fetch historical price data for volatility calculation.
        
        Args:
            ticker (str): Ticker symbol to fetch data for
            period (str): Period to fetch data for (e.g., '60d', '1y')
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        try:
            data = yf.download(ticker, period=period)
            if data.empty:
                logger.error(f"No data found for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"Fetched historical data for {ticker}: {len(data)} rows")
            # Cache the data for future use
            self.data_cache[ticker] = data
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def calculate_historical_volatility(
        self, data: Optional[pd.DataFrame] = None, ticker: str = "^GSPC"
    ) -> float:
        """Calculate historical volatility.
        
        Args:
            data (pd.DataFrame, optional): Price data with 'Close' column
            ticker (str): Ticker symbol to use if data not provided
            
        Returns:
            float: Historical volatility (annualized standard deviation)
        """
        if data is None:
            if ticker in self.data_cache:
                data = self.data_cache[ticker]
            else:
                data = self.fetch_historical_data(ticker)
        
        if data.empty:
            logger.error("No data available for volatility calculation")
            return 0.0
        
        try:
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate rolling volatility (annualized)
            rolling_vol = returns.rolling(
                window=self.lookback_period
            ).std() * np.sqrt(252)
            
            # Get the most recent volatility value
            current_vol = rolling_vol.iloc[-1]
            
            logger.info(f"Current historical volatility: {current_vol:.4f}")
            return current_vol
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return 0.0
    
    def get_current_vix(self) -> float:
        """Get the current VIX (CBOE Volatility Index) value.
        
        Returns:
            float: Current VIX value
        """
        try:
            vix_data = self.fetch_historical_data("^VIX", period="5d")
            if vix_data.empty:
                logger.error("No VIX data available")
                return 0.0
            
            current_vix = vix_data['Close'].iloc[-1]
            logger.info(f"Current VIX: {current_vix:.2f}")
            return current_vix
        except Exception as e:
            logger.error(f"Error getting current VIX: {e}")
            return 0.0
    
    def is_high_volatility(self) -> Tuple[bool, Dict[str, Any]]:
        """Determine if current market conditions represent high volatility.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: (Is high volatility, Volatility metrics)
        """
        # Get S&P500 historical volatility
        sp500_data = self.fetch_historical_data("^GSPC")
        hist_vol = self.calculate_historical_volatility(sp500_data)
        
        # Get current VIX
        vix = self.get_current_vix()
        
        # Calculate volatility range for recent period
        if not sp500_data.empty:
            returns = sp500_data['Close'].pct_change().dropna()
            vol_range = returns.rolling(window=self.lookback_period).max() - \
                        returns.rolling(window=self.lookback_period).min()
            current_range = vol_range.iloc[-1] if not vol_range.empty else 0.0
        else:
            current_range = 0.0
        
        # Determine if volatility is high based on multiple factors
        is_high = (
            hist_vol > self.volatility_threshold or
            vix > self.vix_threshold
        )
        
        metrics = {
            "historical_volatility": hist_vol,
            "vix": vix,
            "volatility_range": current_range,
            "is_high_volatility": is_high,
        }
        
        return is_high, metrics
    
    def get_volatility_trend(self) -> Dict[str, Any]:
        """Get the trend of market volatility.
        
        Returns:
            Dict[str, Any]: Volatility trend metrics
        """
        vix_data = self.fetch_historical_data("^VIX", period="60d")
        if vix_data.empty:
            return {"trend": "unknown", "confidence": 0.0}
        
        # Calculate short and long-term moving averages
        vix_data['vix_ma5'] = vix_data['Close'].rolling(window=5).mean()
        vix_data['vix_ma20'] = vix_data['Close'].rolling(window=20).mean()
        
        # Determine trend direction
        if vix_data['vix_ma5'].iloc[-1] > vix_data['vix_ma20'].iloc[-1]:
            trend = "increasing"
            # Calculate confidence based on the distance between MAs
            confidence = min(1.0, (vix_data['vix_ma5'].iloc[-1] / vix_data['vix_ma20'].iloc[-1] - 1) * 5)
        else:
            trend = "decreasing"
            # Calculate confidence based on the distance between MAs
            confidence = min(1.0, (1 - vix_data['vix_ma5'].iloc[-1] / vix_data['vix_ma20'].iloc[-1]) * 5)
        
        return {
            "trend": trend,
            "confidence": confidence,
            "vix_current": vix_data['Close'].iloc[-1],
            "vix_ma5": vix_data['vix_ma5'].iloc[-1],
            "vix_ma20": vix_data['vix_ma20'].iloc[-1],
        }