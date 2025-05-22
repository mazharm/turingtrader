"""
Tests for the volatility analyzer module.
"""

import unittest
from ibkr_trader.volatility_analyzer import VolatilityAnalyzer
from ibkr_trader.config import Config


class TestVolatilityAnalyzer(unittest.TestCase):
    """Test cases for the volatility analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.analyzer = VolatilityAnalyzer(self.config)
    
    def test_calculate_historical_volatility(self):
        """Test historical volatility calculation."""
        # Test with a simple price series
        prices = [100, 102, 104, 103, 105, 107, 108, 109, 111, 110,
                 112, 111, 113, 114, 116, 115, 117, 119, 120, 122, 121]
        
        vol = self.analyzer.calculate_historical_volatility(prices)
        
        # The result should be a non-negative value
        self.assertTrue(vol >= 0)
    
    def test_analyze_vix(self):
        """Test VIX analysis."""
        # Sample VIX data
        vix_data = [
            {'date': '2022-01-01', 'close': 17.0},
            {'date': '2022-01-02', 'close': 17.5},
            {'date': '2022-01-03', 'close': 18.0},
            {'date': '2022-01-04', 'close': 19.0},
            {'date': '2022-01-05', 'close': 20.0},
            {'date': '2022-01-06', 'close': 22.0}
        ]
        
        analysis = self.analyzer.analyze_vix(vix_data)
        
        # Check that analysis contains expected keys
        self.assertIn('current_vix', analysis)
        self.assertIn('vix_change_1d', analysis)
        self.assertIn('volatility_state', analysis)
        self.assertIn('signal', analysis)
        
        # Check the current VIX value
        self.assertEqual(analysis['current_vix'], 22.0)
        
        # Check that the change is calculated correctly
        self.assertEqual(analysis['vix_change_1d'], 2.0)
    
    def test_should_trade_today(self):
        """Test whether we should trade based on VIX."""
        # Test with low volatility (shouldn't trade)
        low_vol = {
            'current_vix': 12.0,
            'signal': 'cash'
        }
        self.assertFalse(self.analyzer.should_trade_today(low_vol))
        
        # Test with high volatility (should trade)
        high_vol = {
            'current_vix': 25.0,
            'signal': 'buy'
        }
        self.assertTrue(self.analyzer.should_trade_today(high_vol))
    
    def test_get_position_size_multiplier(self):
        """Test position size multiplier calculation."""
        # Test with different signals
        strong_buy = {'signal': 'strong_buy'}
        buy = {'signal': 'buy'}
        hold = {'signal': 'hold'}
        cash = {'signal': 'none'}
        
        # Check that multipliers are in the expected range [0, 1]
        self.assertEqual(self.analyzer.get_position_size_multiplier(strong_buy), 1.0)
        self.assertEqual(self.analyzer.get_position_size_multiplier(buy), 0.7)
        self.assertEqual(self.analyzer.get_position_size_multiplier(hold), 0.3)
        self.assertEqual(self.analyzer.get_position_size_multiplier(cash), 0.0)


if __name__ == '__main__':
    unittest.main()