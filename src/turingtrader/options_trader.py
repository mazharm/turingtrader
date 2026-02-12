"""S&P500 options trading module for TuringTrader."""

from typing import Dict, Any, Optional, List, Tuple
from ib_insync import Contract, Trade
import pandas as pd
import numpy as np
import logging
import datetime

from .ib_client import IBClient
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OptionsTrader:
    """S&P500 options trading logic for the TuringTrader.
    
    This class implements options trading strategies for S&P500 options,
    focusing on strategies that capitalize on market volatility.
    """
    
    def __init__(
        self,
        ib_client: IBClient,
        risk_manager: RiskManager,
    ):
        """Initialize the options trader.
        
        Args:
            ib_client (IBClient): Interactive Brokers client
            risk_manager (RiskManager): Risk manager
        """
        self.ib_client = ib_client
        self.risk_manager = risk_manager
        self.active_trades = {}  # Track active trades
        
    def _select_options_strategy(self, is_high_volatility: bool) -> str:
        """Select options strategy based on market conditions.
        
        Args:
            is_high_volatility (bool): Whether current volatility is high
            
        Returns:
            str: Selected strategy name
        """
        if is_high_volatility:
            # In high volatility, prefer strategies that profit from volatility contraction
            return "iron_condor"  # Neutral strategy that benefits from volatility contraction
        else:
            # In low volatility, stay conservative with defined risk strategies
            return "vertical_spread"  # Defined risk directional strategy
    
    def _find_best_options(
        self, 
        options_data: List[Dict[str, Any]], 
        volatility_metrics: Dict[str, Any],
        preferred_delta: float
    ) -> List[Dict[str, Any]]:
        """Find the best options to trade based on current conditions.
        
        Args:
            options_data (List[Dict[str, Any]]): Options market data
            volatility_metrics (Dict[str, Any]): Current volatility metrics
            preferred_delta (float): Target delta for options selection
            
        Returns:
            List[Dict[str, Any]]: Selected options for trading
        """
        # Sort options by their liquidity (volume and open interest)
        sorted_options = sorted(
            [opt for opt in options_data if opt["volume"] > 0],
            key=lambda x: x["volume"] * x["open_interest"] if x["open_interest"] > 0 else 0,
            reverse=True
        )
        
        if not sorted_options:
            logger.warning("No liquid options found")
            return []
        
        # Filter options based on delta (approximate, actual delta would come from IB)
        # This is a simplified approximation - production system would use actual delta
        selected_options = []
        
        for option in sorted_options:
            # Skip options with no market data
            if option["bid"] <= 0 or option["ask"] <= 0:
                continue
                
            # Estimated delta (simplified approximation for this example)
            estimated_delta = min(0.99, max(0.01, option["implied_volatility"] / 3))
            
            # Select options close to preferred delta
            if abs(estimated_delta - preferred_delta) < 0.15:
                option["estimated_delta"] = estimated_delta
                selected_options.append(option)
                
                # Limit selection to a reasonable number
                if len(selected_options) >= 5:
                    break
        
        return selected_options
    
    def _create_iron_condor(
        self, 
        options: List[Dict[str, Any]], 
        account_balance: float
    ) -> Optional[Dict[str, Any]]:
        """Create an iron condor options strategy.
        
        Args:
            options (List[Dict[str, Any]]): Available options
            account_balance (float): Current account balance
            
        Returns:
            Optional[Dict[str, Any]]: Iron condor strategy details
        """
        # Separate puts and calls
        puts = [opt for opt in options if opt["contract"].right == "P"]
        calls = [opt for opt in options if opt["contract"].right == "C"]
        
        if len(puts) < 2 or len(calls) < 2:
            logger.warning("Not enough options to create iron condor")
            return None
        
        # Sort by strike
        puts = sorted(puts, key=lambda x: x["contract"].strike)
        calls = sorted(calls, key=lambda x: x["contract"].strike)
        
        # Select put spread (short higher strike, long lower strike)
        short_put = puts[-2]  # Second highest strike put
        long_put = puts[-3] if len(puts) > 2 else puts[-1]  # Lower strike put
        
        # Select call spread (short lower strike, long higher strike)
        short_call = calls[1]  # Second lowest strike call
        long_call = calls[2] if len(calls) > 2 else calls[0]  # Higher strike call
        
        # Calculate max risk: width of wider spread minus credit received
        put_spread_width = abs(short_put["contract"].strike - long_put["contract"].strike)
        call_spread_width = abs(short_call["contract"].strike - long_call["contract"].strike)
        max_width = max(put_spread_width, call_spread_width)
        
        # Credit received (for each spread)
        put_credit = short_put["bid"] - long_put["ask"]
        call_credit = short_call["bid"] - long_call["ask"]
        total_credit = put_credit + call_credit
        
        # Max risk per contract
        max_risk_per_contract = (max_width - total_credit) * 100  # 100 shares per contract
        
        # Determine number of contracts based on risk management
        risk_factor = self.risk_manager.max_position_size_pct
        max_contracts = max(1, int(account_balance * risk_factor / max_risk_per_contract))
        
        return {
            "strategy": "iron_condor",
            "short_put": short_put,
            "long_put": long_put,
            "short_call": short_call,
            "long_call": long_call,
            "max_risk_per_contract": max_risk_per_contract,
            "max_profit_per_contract": total_credit * 100,
            "contracts": max_contracts,
            "total_credit": total_credit * max_contracts * 100,
            "total_max_risk": max_risk_per_contract * max_contracts,
        }
    
    def _create_vertical_spread(
        self, 
        options: List[Dict[str, Any]], 
        account_balance: float,
        is_bullish: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Create a vertical spread options strategy.
        
        Args:
            options (List[Dict[str, Any]]): Available options
            account_balance (float): Current account balance
            is_bullish (bool): Whether to create a bullish (call) or bearish (put) spread
            
        Returns:
            Optional[Dict[str, Any]]: Vertical spread strategy details
        """
        # Filter by option type
        option_type = "C" if is_bullish else "P"
        filtered_options = [opt for opt in options if opt["contract"].right == option_type]
        
        if len(filtered_options) < 2:
            logger.warning(f"Not enough {option_type} options to create vertical spread")
            return None
        
        # Sort by strike
        filtered_options = sorted(filtered_options, key=lambda x: x["contract"].strike)
        
        if is_bullish:  # Bull Call Spread (buy lower strike, sell higher strike)
            long_option = filtered_options[0]  # Lowest strike
            short_option = filtered_options[1]  # Next strike up
            
            # Debit paid
            debit = long_option["ask"] - short_option["bid"]
            
            # Max profit: difference in strikes minus debit paid
            max_profit_per_contract = (
                (short_option["contract"].strike - long_option["contract"].strike) - debit
            ) * 100
            
            # Max risk: debit paid
            max_risk_per_contract = debit * 100
            
        else:  # Bear Put Spread (buy higher strike, sell lower strike)
            short_option = filtered_options[0]  # Lowest strike
            long_option = filtered_options[1]  # Next strike up
            
            # Debit paid
            debit = long_option["ask"] - short_option["bid"]
            
            # Max profit: difference in strikes minus debit paid
            max_profit_per_contract = (
                (long_option["contract"].strike - short_option["contract"].strike) - debit
            ) * 100
            
            # Max risk: debit paid
            max_risk_per_contract = debit * 100
        
        # Determine number of contracts based on risk management
        risk_factor = self.risk_manager.max_position_size_pct
        max_contracts = max(1, int(account_balance * risk_factor / max_risk_per_contract))
        
        return {
            "strategy": "vertical_spread",
            "direction": "bullish" if is_bullish else "bearish",
            "long_option": long_option,
            "short_option": short_option,
            "max_risk_per_contract": max_risk_per_contract,
            "max_profit_per_contract": max_profit_per_contract,
            "contracts": max_contracts,
            "total_debit": debit * max_contracts * 100,
            "total_max_risk": max_risk_per_contract * max_contracts,
            "total_max_profit": max_profit_per_contract * max_contracts,
        }
    
    def analyze_market(self) -> Dict[str, Any]:
        """Analyze market conditions for trading opportunities.
        
        Returns:
            Dict[str, Any]: Market analysis results
        """
        # Check if the market is open
        if not self.ib_client.is_market_open():
            return {"tradeable": False, "reason": "Market closed"}
            
        # Get SPX options with appropriate expiration
        options_contracts = self.ib_client.get_sp500_options(
            days_to_expiry=self.risk_manager.max_dte
        )
        
        if not options_contracts:
            return {"tradeable": False, "reason": "No suitable options available"}
        
        # Get market data for the options
        options_data = []
        for contract in options_contracts:
            price_data = self.ib_client.get_option_price(contract)
            if price_data:
                options_data.append({**price_data, "contract": contract})
        
        if not options_data:
            return {"tradeable": False, "reason": "No options market data available"}
        
        return {
            "tradeable": True,
            "options_data": options_data,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    
    def generate_trade_plan(
        self,
        is_high_volatility: bool,
        volatility_metrics: Dict[str, Any],
        options_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a trading plan based on market conditions.
        
        Args:
            is_high_volatility (bool): Whether current volatility is high
            volatility_metrics (Dict[str, Any]): Volatility metrics
            options_data (List[Dict[str, Any]]): Options market data
            
        Returns:
            Dict[str, Any]: Trading plan details
        """
        # Get account balance
        account_summary = self.ib_client.get_account_summary()
        account_balance = account_summary.get("TotalCashValue", 0.0)
        
        # Adjust risk parameters based on volatility
        adjusted_risk = self.risk_manager.adjust_for_volatility(is_high_volatility)
        preferred_delta = adjusted_risk["preferred_delta"]
        
        # Select strategy based on volatility
        strategy_type = self._select_options_strategy(is_high_volatility)
        
        # Find best options based on adjusted parameters
        best_options = self._find_best_options(options_data, volatility_metrics, preferred_delta)
        
        if not best_options:
            return {"tradeable": False, "reason": "No suitable options found"}
        
        # Generate strategy-specific trade plan
        if strategy_type == "iron_condor":
            trade_details = self._create_iron_condor(best_options, account_balance)
        elif strategy_type == "vertical_spread":
            # Determine direction based on volatility trend
            is_bullish = volatility_metrics.get("trend") == "decreasing"
            trade_details = self._create_vertical_spread(
                best_options, account_balance, is_bullish=is_bullish
            )
        else:
            trade_details = None
        
        if not trade_details:
            return {"tradeable": False, "reason": f"Failed to create {strategy_type} strategy"}
        
        return {
            "tradeable": True,
            "strategy": strategy_type,
            "is_high_volatility": is_high_volatility,
            "trade_details": trade_details,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    
    def execute_trades(self, trade_plan: Dict[str, Any]) -> bool:
        """Execute trades based on the trade plan.
        
        Args:
            trade_plan (Dict[str, Any]): Trade plan to execute
            
        Returns:
            bool: True if trades executed successfully, False otherwise
        """
        if not trade_plan.get("tradeable", False):
            logger.info(f"No tradeable plan: {trade_plan.get('reason', 'Unknown reason')}")
            return False
        
        strategy = trade_plan.get("strategy")
        trade_details = trade_plan.get("trade_details", {})
        
        if strategy == "iron_condor":
            return self._execute_iron_condor(trade_details)
        elif strategy == "vertical_spread":
            return self._execute_vertical_spread(trade_details)
        
        logger.error(f"Unsupported strategy: {strategy}")
        return False
    
    def _execute_iron_condor(self, trade_details: Dict[str, Any]) -> bool:
        """Execute an iron condor strategy.
        
        Args:
            trade_details (Dict[str, Any]): Iron condor trade details
            
        Returns:
            bool: True if executed successfully, False otherwise
        """
        try:
            contracts = trade_details.get("contracts", 1)
            
            # Sell put spread
            short_put = trade_details["short_put"]["contract"]
            long_put = trade_details["long_put"]["contract"]
            self.ib_client.place_order(short_put, "SELL", contracts)
            self.ib_client.place_order(long_put, "BUY", contracts)
            
            # Sell call spread
            short_call = trade_details["short_call"]["contract"]
            long_call = trade_details["long_call"]["contract"]
            self.ib_client.place_order(short_call, "SELL", contracts)
            self.ib_client.place_order(long_call, "BUY", contracts)
            
            logger.info(
                f"Executed iron condor: {contracts} contracts, "
                f"max risk: ${trade_details['total_max_risk']:.2f}"
            )
            
            # Store trade details for later reference
            trade_id = f"ic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_trades[trade_id] = {
                "strategy": "iron_condor",
                "details": trade_details,
                "entry_time": datetime.datetime.now().isoformat(),
            }
            
            return True
        except Exception as e:
            logger.error(f"Failed to execute iron condor: {e}")
            return False
    
    def _execute_vertical_spread(self, trade_details: Dict[str, Any]) -> bool:
        """Execute a vertical spread strategy.
        
        Args:
            trade_details (Dict[str, Any]): Vertical spread trade details
            
        Returns:
            bool: True if executed successfully, False otherwise
        """
        try:
            contracts = trade_details.get("contracts", 1)
            
            # Execute long option
            long_option = trade_details["long_option"]["contract"]
            self.ib_client.place_order(long_option, "BUY", contracts)
            
            # Execute short option
            short_option = trade_details["short_option"]["contract"]
            self.ib_client.place_order(short_option, "SELL", contracts)
            
            logger.info(
                f"Executed {trade_details['direction']} vertical spread: {contracts} contracts, "
                f"max risk: ${trade_details['total_max_risk']:.2f}"
            )
            
            # Store trade details for later reference
            trade_id = f"vs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_trades[trade_id] = {
                "strategy": "vertical_spread",
                "direction": trade_details["direction"],
                "details": trade_details,
                "entry_time": datetime.datetime.now().isoformat(),
            }
            
            return True
        except Exception as e:
            logger.error(f"Failed to execute vertical spread: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all open options positions.
        
        Returns:
            bool: True if all positions closed successfully, False otherwise
        """
        result = self.ib_client.close_all_positions()
        
        if result:
            # Clear active trades tracking
            self.active_trades = {}
            logger.info("All positions closed successfully")
        else:
            logger.error("Failed to close all positions")
            
        return result