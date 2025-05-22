"""Tests for the risk manager module."""

import unittest
import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.turingtrader.risk_manager import RiskManager, RiskLevel


class TestRiskManager(unittest.TestCase):
    """Test cases for the RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            risk_level=RiskLevel.MEDIUM,
            account_balance=100000.0
        )
    
    def test_init_default_params(self):
        """Test that default parameters are set correctly."""
        self.assertEqual(self.risk_manager.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(self.risk_manager.account_balance, 100000.0)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
    
    def test_configure_risk_profile(self):
        """Test risk profile configuration."""
        # Test low risk profile
        low_risk = RiskManager(risk_level=RiskLevel.LOW)
        self.assertEqual(low_risk.max_position_size_pct, 0.02)
        self.assertEqual(low_risk.max_daily_loss_pct, 0.01)
        self.assertEqual(low_risk.stop_loss_pct, 0.10)
        
        # Test high risk profile
        high_risk = RiskManager(risk_level=RiskLevel.HIGH)
        self.assertEqual(high_risk.max_position_size_pct, 0.10)
        self.assertEqual(high_risk.max_daily_loss_pct, 0.04)
        self.assertEqual(high_risk.stop_loss_pct, 0.25)
    
    def test_set_risk_level(self):
        """Test changing risk level."""
        self.risk_manager.set_risk_level(RiskLevel.AGGRESSIVE)
        
        self.assertEqual(self.risk_manager.risk_level, RiskLevel.AGGRESSIVE)
        self.assertEqual(self.risk_manager.max_position_size_pct, 0.15)
        self.assertEqual(self.risk_manager.max_daily_loss_pct, 0.06)
    
    def test_update_account_balance(self):
        """Test updating account balance."""
        original_balance = self.risk_manager.account_balance
        self.risk_manager.daily_pnl = 500.0  # Set some daily P&L
        
        self.risk_manager.update_account_balance(110000.0)
        
        self.assertEqual(self.risk_manager.account_balance, 110000.0)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)  # Should be reset
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Test with normal price
        num_contracts, actual_pct = self.risk_manager.calculate_position_size(
            price=400.0,  # $400 per option
            volatility_factor=1.0
        )
        
        expected_contracts = int((100000.0 * 0.05) / (400.0 * 100))  # 5% of account / contract value
        self.assertEqual(num_contracts, expected_contracts)
        
        # Test with volatility factor
        num_contracts_vol, actual_pct_vol = self.risk_manager.calculate_position_size(
            price=400.0,
            volatility_factor=0.5  # Reduce position size due to volatility
        )
        
        self.assertLess(num_contracts_vol, num_contracts)
    
    def test_update_daily_pnl_within_limit(self):
        """Test updating daily P&L within the loss limit."""
        result = self.risk_manager.update_daily_pnl(-1000.0)  # 1% loss
        
        self.assertTrue(result)
        self.assertEqual(self.risk_manager.daily_pnl, -1000.0)
    
    def test_update_daily_pnl_exceeding_limit(self):
        """Test updating daily P&L exceeding the loss limit."""
        # 2% limit, set a 3% loss
        result = self.risk_manager.update_daily_pnl(-3000.0)
        
        self.assertFalse(result)
        self.assertEqual(self.risk_manager.daily_pnl, -3000.0)
    
    def test_should_close_position(self):
        """Test stop loss check for positions."""
        # Test position that hasn't hit stop loss
        should_close = self.risk_manager.should_close_position(
            entry_price=100.0,
            current_price=95.0,  # 5% loss
            is_long=True
        )
        self.assertFalse(should_close)
        
        # Test position that has hit stop loss
        should_close = self.risk_manager.should_close_position(
            entry_price=100.0,
            current_price=80.0,  # 20% loss
            is_long=True
        )
        self.assertTrue(should_close)
        
        # Test short position
        should_close = self.risk_manager.should_close_position(
            entry_price=100.0,
            current_price=120.0,  # 20% move against short position
            is_long=False
        )
        self.assertTrue(should_close)
    
    def test_adjust_for_volatility(self):
        """Test risk parameter adjustment for volatility."""
        # High volatility
        params = self.risk_manager.adjust_for_volatility(True)
        self.assertEqual(params["position_size_factor"], 0.7)
        self.assertGreater(params["preferred_delta"], self.risk_manager.preferred_delta)
        
        # Low volatility
        params = self.risk_manager.adjust_for_volatility(False)
        self.assertEqual(params["position_size_factor"], 0.5)
        self.assertLess(params["preferred_delta"], self.risk_manager.preferred_delta)
    
    def test_get_risk_profile(self):
        """Test getting the risk profile."""
        profile = self.risk_manager.get_risk_profile()
        
        self.assertEqual(profile["risk_level"], "MEDIUM")
        self.assertEqual(profile["max_position_size_pct"], 0.05)
        self.assertEqual(profile["max_daily_loss_pct"], 0.02)
        self.assertEqual(profile["stop_loss_pct"], 0.15)


if __name__ == "__main__":
    unittest.main()