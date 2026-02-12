"""
Cash Management Module

This module handles daily cash management operations:
- Ensures all positions are liquidated by market close
- Monitors cash balances and available buying power
- Schedules liquidation events for end-of-day processing
"""
import datetime
import logging
import time
import threading
from typing import Dict, List, Optional, Callable

class CashManager:
    """
    Manages cash positions and end-of-day liquidation
    """
    
    def __init__(self, ib_client, 
                market_open_time: datetime.time = datetime.time(9, 30),
                market_close_time: datetime.time = datetime.time(16, 0),
                liquidation_time: datetime.time = datetime.time(15, 45),
                liquidation_callback: Optional[Callable] = None):
        """
        Initialize Cash Manager
        
        Args:
            ib_client: Interactive Brokers client instance
            market_open_time: Market opening time (default: 9:30 AM ET)
            market_close_time: Market closing time (default: 4:00 PM ET)
            liquidation_time: Time to begin liquidation (default: 3:45 PM ET)
            liquidation_callback: Optional callback function after liquidation
        """
        self.logger = logging.getLogger(__name__)
        self.ib_client = ib_client
        self.market_open_time = market_open_time
        self.market_close_time = market_close_time
        self.liquidation_time = liquidation_time
        self.liquidation_callback = liquidation_callback
        
        # Internal state
        self.cash_balance = 0.0
        self.positions_value = 0.0
        self.account_value = 0.0
        self.pending_orders = []
        self.liquidation_active = False
        self.liquidation_thread = None
    
    def update_account_info(self, account_info: Dict) -> None:
        """
        Update account information
        
        Args:
            account_info: Dictionary containing account information
        """
        if 'TotalCashValue' in account_info:
            self.cash_balance = float(account_info['TotalCashValue'])
            
        if 'NetLiquidation' in account_info:
            self.account_value = float(account_info['NetLiquidation'])
            
        if 'StockMarketValue' in account_info:
            self.positions_value = float(account_info['StockMarketValue'])
            
        self.logger.debug(f"Account info updated - Cash: ${self.cash_balance:.2f}, "
                         f"Positions: ${self.positions_value:.2f}, "
                         f"Total: ${self.account_value:.2f}")
    
    def is_cash_only(self) -> bool:
        """
        Check if portfolio is currently in cash-only position
        
        Returns:
            bool: True if no positions are held (cash only)
        """
        # Allow small position values (< 1% of account) to handle rounding errors
        return self.positions_value < (self.account_value * 0.01)
    
    def get_available_cash(self) -> float:
        """
        Get available cash for trading
        
        Returns:
            float: Available cash balance
        """
        return self.cash_balance
    
    def schedule_liquidation(self) -> None:
        """
        Schedule end-of-day liquidation based on market hours
        """
        # Stop any existing liquidation thread
        if self.liquidation_thread and self.liquidation_thread.is_alive():
            self.liquidation_active = False
            self.liquidation_thread.join(1.0)
        
        # Start new liquidation thread
        self.liquidation_active = True
        self.liquidation_thread = threading.Thread(
            target=self._liquidation_monitor_task,
            daemon=True
        )
        self.liquidation_thread.start()
        self.logger.info(f"Liquidation scheduled for {self.liquidation_time}")
    
    def _liquidation_monitor_task(self) -> None:
        """Background task to monitor time and trigger liquidation"""
        while self.liquidation_active:
            now = datetime.datetime.now().time()
            
            # Check if we've reached liquidation time
            if now >= self.liquidation_time and now < self.market_close_time:
                self.logger.info(f"Liquidation time reached ({now}), initiating liquidation")
                self.liquidate_all_positions()
                
                # Execute callback if provided
                if self.liquidation_callback:
                    self.liquidation_callback()
                    
                self.liquidation_active = False
                break
                
            # Sleep for 15 seconds before checking again
            time.sleep(15)
    
    def liquidate_all_positions(self) -> bool:
        """
        Liquidate all open positions
        
        Returns:
            bool: True if liquidation orders were placed successfully
        """
        self.logger.info("Initiating liquidation of all positions")
        
        # Request fresh position data
        self.ib_client.request_account_updates()
        
        # Force a small delay to ensure we have the latest positions
        time.sleep(1)
        
        # Place liquidation orders
        try:
            order_ids = self.ib_client.liquidate_all_positions()
            self.pending_orders = order_ids
            
            if not order_ids:
                self.logger.info("No positions to liquidate")
                return True
                
            self.logger.info(f"Placed {len(order_ids)} liquidation orders")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during liquidation: {str(e)}")
            return False
    
    def verify_cash_position(self) -> Dict:
        """
        Verify current cash position status
        
        Returns:
            dict: Cash position status information
        """
        is_cash = self.is_cash_only()
        
        return {
            'is_cash_only': is_cash,
            'cash_balance': self.cash_balance,
            'positions_value': self.positions_value,
            'account_value': self.account_value,
            'liquidation_scheduled': self.liquidation_active,
            'liquidation_time': self.liquidation_time.strftime('%H:%M:%S') if self.liquidation_time else None,
            'pending_liquidation_orders': len(self.pending_orders)
        }
        
    def handle_market_open(self) -> None:
        """Handle market open procedures"""
        self.logger.info("Market open procedures initiated")
        
        # Verify we're in cash position at start of day
        status = self.verify_cash_position()
        if not status['is_cash_only']:
            self.logger.warning("WARNING: Not in cash-only position at market open")
            # Force liquidation if we somehow have positions at open
            self.liquidate_all_positions()
        
        # Schedule end-of-day liquidation
        self.schedule_liquidation()
        
    def handle_market_close(self) -> None:
        """Handle market close procedures"""
        self.logger.info("Market close procedures initiated")
        
        # Verify we're in cash position at end of day
        status = self.verify_cash_position()
        if not status['is_cash_only']:
            self.logger.warning("WARNING: Not in cash-only position at market close")
            # For safety, attempt one final liquidation
            # This might not execute if market is already closed
            self.liquidate_all_positions()
        else:
            self.logger.info("Successfully ended trading day in cash-only position")