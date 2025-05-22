"""Tests for the volatility module."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
import datetime

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.turingtrader.volatility import VolatilityAnalyzer


class TestVolatilityAnalyzer(unittest.TestCase):
    """Test cases for the VolatilityAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = VolatilityAnalyzer(
            lookback_period=20,
            volatility_threshold=0.15,
            vix_threshold=20.0
        )
        
        # Create sample data for testing
        dates = pd.date_range(start='2022-01-01', periods=60)
        close_prices = np.linspace(4000, 4200, 60) + np.random.normal(0, 50, 60)
        
        self.sample_data = pd.DataFrame({
            'Close': close_prices,
            'Open': close_prices - np.random.normal(0, 10, 60),
            'High': close_prices + np.random.normal(10, 5, 60),
            'Low': close_prices - np.random.normal(10, 5, 60),
            'Volume': np.random.normal(1000000, 200000, 60)
        }, index=dates)
        
        self.vix_data = pd.DataFrame({
            'Close': np.linspace(15, 25, 60) + np.random.normal(0, 3, 60),
            'Open': np.linspace(15, 25, 60) + np.random.normal(0, 1, 60),
            'High': np.linspace(15, 25, 60) + np.random.normal(2, 1, 60),
            'Low': np.linspace(15, 25, 60) - np.random.normal(2, 1, 60),
            'Volume': np.random.normal(500000, 100000, 60)
        }, index=dates)
    
    @patch('src.turingtrader.volatility.yf.download')
    def test_fetch_historical_data(self, mock_download):
        """Test fetching historical data."""
        mock_download.return_value = self.sample_data
        
        data = self.analyzer.fetch_historical_data(ticker="^GSPC", period="60d")
        
        self.assertFalse(data.empty)
        self.assertEqual(len(data), 60)
        mock_download.assert_called_once_with("^GSPC", period="60d")
    
    @patch('src.turingtrader.volatility.yf.download')
    def test_fetch_historical_data_empty(self, mock_download):
        """Test fetching historical data with empty result."""
        mock_download.return_value = pd.DataFrame()
        
        data = self.analyzer.fetch_historical_data(ticker="^GSPC", period="60d")
        
        self.assertTrue(data.empty)
    
    def test_calculate_historical_volatility(self):
        """Test calculating historical volatility."""
        # Store the data in the cache
        self.analyzer.data_cache["^GSPC"] = self.sample_data
        
        vol = self.analyzer.calculate_historical_volatility(self.sample_data)
        
        # Ensure the result is a float
        self.assertIsInstance(vol, float)
        # Volatility should be a positive number
        self.assertGreater(vol, 0)
    
    @patch('src.turingtrader.volatility.VolatilityAnalyzer.fetch_historical_data')
    def test_get_current_vix(self, mock_fetch):
        """Test getting current VIX value."""
        mock_fetch.return_value = self.vix_data
        
        vix = self.analyzer.get_current_vix()
        
        # Ensure the result is a float
        self.assertIsInstance(vix, float)
        # Check that the value is reasonable
        self.assertGreater(vix, 0)
        mock_fetch.assert_called_once_with("^VIX", period="5d")
    
    @patch('src.turingtrader.volatility.VolatilityAnalyzer.calculate_historical_volatility')
    @patch('src.turingtrader.volatility.VolatilityAnalyzer.get_current_vix')
    @patch('src.turingtrader.volatility.VolatilityAnalyzer.fetch_historical_data')
    def test_is_high_volatility(self, mock_fetch, mock_vix, mock_hist_vol):
        """Test determining if volatility is high."""
        mock_fetch.return_value = self.sample_data
        mock_vix.return_value = 25.0  # High VIX
        mock_hist_vol.return_value = 0.18  # High historical volatility
        
        is_high, metrics = self.analyzer.is_high_volatility()
        
        self.assertTrue(is_high)
        self.assertEqual(metrics["vix"], 25.0)
        self.assertEqual(metrics["historical_volatility"], 0.18)
    
    @patch('src.turingtrader.volatility.VolatilityAnalyzer.fetch_historical_data')
    def test_get_volatility_trend(self, mock_fetch):
        """Test getting volatility trend."""
        mock_fetch.return_value = self.vix_data
        
        trend = self.analyzer.get_volatility_trend()
        
        self.assertIn("trend", trend)
        self.assertIn(trend["trend"], ["increasing", "decreasing"])
        self.assertIn("confidence", trend)
        self.assertTrue(0 <= trend["confidence"] <= 1)


if __name__ == "__main__":
    unittest.main()