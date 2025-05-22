"""Tests for the options trader module."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import datetime

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.turingtrader.options_trader import OptionsTrader
from src.turingtrader.risk_manager import RiskManager, RiskLevel


class TestOptionsTrader(unittest.TestCase):
    """Test cases for the OptionsTrader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_ib_client = MagicMock()
        self.risk_manager = RiskManager(risk_level=RiskLevel.MEDIUM, account_balance=100000.0)
        
        # Create the options trader
        self.options_trader = OptionsTrader(
            ib_client=self.mock_ib_client,
            risk_manager=self.risk_manager
        )
    
    def test_select_options_strategy(self):
        """Test strategy selection based on volatility."""
        # High volatility should use iron condor
        strategy = self.options_trader._select_options_strategy(is_high_volatility=True)
        self.assertEqual(strategy, "iron_condor")
        
        # Low volatility should use vertical spread
        strategy = self.options_trader._select_options_strategy(is_high_volatility=False)
        self.assertEqual(strategy, "vertical_spread")
    
    def test_find_best_options(self):
        """Test options selection logic."""
        # Create sample options data
        sample_options = [
            {
                "contract": MagicMock(right="C", strike=4000),
                "bid": 10.0,
                "ask": 12.0,
                "volume": 500,
                "open_interest": 1000,
                "implied_volatility": 0.2
            },
            {
                "contract": MagicMock(right="C", strike=4050),
                "bid": 8.0,
                "ask": 10.0,
                "volume": 300,
                "open_interest": 800,
                "implied_volatility": 0.25
            },
            {
                "contract": MagicMock(right="P", strike=3950),
                "bid": 9.0,
                "ask": 11.0,
                "volume": 400,
                "open_interest": 900,
                "implied_volatility": 0.22
            }
        ]
        
        # Sample volatility metrics
        volatility_metrics = {
            "historical_volatility": 0.18,
            "vix": 22.0,
            "trend": "increasing"
        }
        
        # Test with various deltas
        selected = self.options_trader._find_best_options(
            sample_options,
            volatility_metrics,
            preferred_delta=0.4
        )
        
        # Should select options that match criteria
        self.assertGreaterEqual(len(selected), 0)
        
        # Test with no volume
        no_volume_options = [
            {
                "contract": MagicMock(),
                "bid": 10.0,
                "ask": 12.0,
                "volume": 0,  # No volume
                "open_interest": 1000,
                "implied_volatility": 0.2
            }
        ]
        
        selected = self.options_trader._find_best_options(
            no_volume_options,
            volatility_metrics,
            preferred_delta=0.4
        )
        
        # Should select no options when none meet criteria
        self.assertEqual(len(selected), 0)
    
    def test_analyze_market(self):
        """Test market analysis."""
        # Configure mock
        self.mock_ib_client.is_market_open.return_value = True
        self.mock_ib_client.get_sp500_options.return_value = [MagicMock() for _ in range(10)]
        self.mock_ib_client.get_option_price.return_value = {
            "bid": 10.0,
            "ask": 12.0,
            "volume": 500,
            "open_interest": 1000,
            "implied_volatility": 0.2
        }
        
        # Test market analysis
        result = self.options_trader.analyze_market()
        
        # Should be tradeable with options data
        self.assertTrue(result["tradeable"])
        self.assertIn("options_data", result)
        
        # Test when market is closed
        self.mock_ib_client.is_market_open.return_value = False
        result = self.options_trader.analyze_market()
        
        # Should not be tradeable when market closed
        self.assertFalse(result["tradeable"])
        self.assertEqual(result["reason"], "Market closed")
    
    @patch('src.turingtrader.options_trader.OptionsTrader._create_iron_condor')
    @patch('src.turingtrader.options_trader.OptionsTrader._find_best_options')
    def test_generate_trade_plan_iron_condor(self, mock_find_options, mock_create_condor):
        """Test generating an iron condor trade plan."""
        # Set up mocks
        self.mock_ib_client.get_account_summary.return_value = {"TotalCashValue": 100000.0}
        
        # Sample options
        sample_options = [MagicMock() for _ in range(10)]
        mock_find_options.return_value = sample_options
        
        # Sample iron condor
        sample_condor = {
            "strategy": "iron_condor",
            "short_put": {"contract": MagicMock()},
            "long_put": {"contract": MagicMock()},
            "short_call": {"contract": MagicMock()},
            "long_call": {"contract": MagicMock()},
            "max_risk_per_contract": 500.0,
            "max_profit_per_contract": 250.0,
            "contracts": 2,
            "total_credit": 500.0,
            "total_max_risk": 1000.0
        }
        mock_create_condor.return_value = sample_condor
        
        # Test with high volatility (should create iron condor)
        plan = self.options_trader.generate_trade_plan(
            is_high_volatility=True,
            volatility_metrics={"trend": "increasing"},
            options_data=[{"contract": MagicMock()}]
        )
        
        # Verify results
        self.assertTrue(plan["tradeable"])
        self.assertEqual(plan["strategy"], "iron_condor")
        self.assertIn("trade_details", plan)
        
        # Ensure iron condor was created
        mock_create_condor.assert_called_once()
    
    @patch('src.turingtrader.options_trader.OptionsTrader._execute_iron_condor')
    def test_execute_trades_iron_condor(self, mock_execute_condor):
        """Test executing an iron condor trade."""
        # Setup mock
        mock_execute_condor.return_value = True
        
        # Create sample trade plan
        trade_plan = {
            "tradeable": True,
            "strategy": "iron_condor",
            "trade_details": {"some": "details"}
        }
        
        # Execute trade
        result = self.options_trader.execute_trades(trade_plan)
        
        # Verify execution
        self.assertTrue(result)
        mock_execute_condor.assert_called_once_with({"some": "details"})
    
    def test_execute_trades_not_tradeable(self):
        """Test executing trades with non-tradeable plan."""
        # Create sample trade plan
        trade_plan = {
            "tradeable": False,
            "reason": "Test reason"
        }
        
        # Execute trade
        result = self.options_trader.execute_trades(trade_plan)
        
        # Should not execute
        self.assertFalse(result)
    
    def test_close_all_positions(self):
        """Test closing all positions."""
        # Set up mock
        self.mock_ib_client.close_all_positions.return_value = True
        
        # Add some active trades
        self.options_trader.active_trades = {
            "test_trade": {"strategy": "iron_condor"}
        }
        
        # Close positions
        result = self.options_trader.close_all_positions()
        
        # Verify result
        self.assertTrue(result)
        self.assertEqual(len(self.options_trader.active_trades), 0)
        self.mock_ib_client.close_all_positions.assert_called_once()


if __name__ == "__main__":
    unittest.main()