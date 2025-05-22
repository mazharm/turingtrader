#!/usr/bin/env python3
"""Backtesting script for TuringTrader."""

import argparse
import logging
import datetime
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path so we can import our package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.turingtrader.volatility import VolatilityAnalyzer
from src.turingtrader.risk_manager import RiskLevel, RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest.log"),
    ],
)

logger = logging.getLogger(__name__)


class OptionsBacktester:
    """Backtester for the TuringTrader options strategy."""
    
    def __init__(
        self,
        risk_level: str = "MEDIUM",
        initial_capital: float = 100000.0,
        data_dir: str = None,
    ):
        """Initialize the backtester.
        
        Args:
            risk_level (str): Risk level ("LOW", "MEDIUM", "HIGH", "AGGRESSIVE")
            initial_capital (float): Initial capital for backtesting
            data_dir (str, optional): Directory for saving backtest data
        """
        try:
            self.risk_level = RiskLevel[risk_level.upper()]
        except (KeyError, AttributeError):
            logger.warning(f"Invalid risk level '{risk_level}', defaulting to MEDIUM")
            self.risk_level = RiskLevel.MEDIUM
            
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager(risk_level=self.risk_level, account_balance=initial_capital)
        self.volatility_analyzer = VolatilityAnalyzer()
        
        # Set up data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / ".turingtrader" / "backtest"
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for backtesting.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Historical market data
        """
        # Convert string dates to datetime
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        # Add a buffer before the start date for volatility calculation
        buffer_start = start - datetime.timedelta(days=60)
        buffer_start_str = buffer_start.strftime("%Y-%m-%d")
        
        # Fetch S&P500 data
        logger.info(f"Fetching historical data from {buffer_start_str} to {end_date}")
        sp500_data = self.volatility_analyzer.fetch_historical_data(
            "^GSPC", period=f"{buffer_start_str} {end_date}"
        )
        
        # Fetch VIX data
        vix_data = self.volatility_analyzer.fetch_historical_data(
            "^VIX", period=f"{buffer_start_str} {end_date}"
        )
        
        if sp500_data.empty or vix_data.empty:
            logger.error("Failed to fetch historical data")
            return pd.DataFrame()
        
        # Combine data
        data = pd.DataFrame({
            "sp500_close": sp500_data["Close"],
            "sp500_high": sp500_data["High"],
            "sp500_low": sp500_data["Low"],
            "sp500_volume": sp500_data["Volume"],
            "vix_close": vix_data["Close"],
        })
        
        # Calculate historical volatility
        returns = sp500_data["Close"].pct_change().dropna()
        data["hist_vol_20d"] = returns.rolling(window=20).std() * np.sqrt(252)
        data["daily_return"] = returns
        
        return data
    
    def simulate_option_strategy(
        self,
        data: pd.DataFrame,
        strategy: str = "iron_condor",
        is_high_vol: bool = True
    ) -> pd.DataFrame:
        """Simulate option strategy returns based on market conditions.
        
        Args:
            data (pd.DataFrame): Market data
            strategy (str): Strategy to simulate (e.g., "iron_condor")
            is_high_vol (bool): Whether volatility is high
            
        Returns:
            pd.DataFrame: Data with simulated strategy returns
        """
        # This is a simplified simulation that models options returns
        # based on market conditions. A real implementation would use
        # actual options pricing models and historical options data.
        
        # Create a copy of the data
        sim_data = data.copy()
        
        # Initialize columns
        sim_data["is_trading_day"] = ~sim_data.index.weekday.isin([5, 6])  # Not weekend
        sim_data["is_high_vol"] = sim_data["vix_close"] > 20.0
        sim_data["strategy"] = None
        sim_data["trade_return"] = 0.0
        
        # Simulate daily trading decisions and returns
        for i in range(20, len(sim_data)):  # Start after vol calculation window
            if not sim_data["is_trading_day"].iloc[i]:
                continue
                
            day_data = sim_data.iloc[i]
            prev_day = sim_data.iloc[i-1]
            
            # Only trade on high volatility days if the strategy requires it
            if is_high_vol and not day_data["is_high_vol"]:
                continue
            
            # Simulate the options strategy return based on:
            # 1. Volatility (VIX)
            # 2. S&P500 daily move
            # 3. Current strategy
            
            # Iron Condor: profits when S&P500 stays within a range (low movement)
            if strategy == "iron_condor":
                # Calculate daily move magnitude as percentage
                day_move = abs(day_data["daily_return"])
                
                # Iron condors profit when market stays within a range (simplified)
                if day_move < 0.005:  # Small move (less than 0.5%)
                    sim_data.at[sim_data.index[i], "trade_return"] = 0.01  # Good profit
                    sim_data.at[sim_data.index[i], "strategy"] = "iron_condor_win"
                elif day_move < 0.01:  # Medium move (0.5% - 1%)
                    sim_data.at[sim_data.index[i], "trade_return"] = 0.005  # Small profit
                    sim_data.at[sim_data.index[i], "strategy"] = "iron_condor_small_win"
                else:  # Large move (> 1%)
                    sim_data.at[sim_data.index[i], "trade_return"] = -0.02  # Loss
                    sim_data.at[sim_data.index[i], "strategy"] = "iron_condor_loss"
                    
            # Vertical Spread: profits from directional movement
            elif strategy == "vertical_spread":
                # Simplified: alternate between bullish and bearish based on previous day's move
                is_bullish = prev_day["daily_return"] < 0  # Contrary to previous move
                
                if (is_bullish and day_data["daily_return"] > 0) or \
                   (not is_bullish and day_data["daily_return"] < 0):
                    # Direction was correct
                    win_size = min(0.03, abs(day_data["daily_return"]) * 3)  # Scale with move size
                    sim_data.at[sim_data.index[i], "trade_return"] = win_size
                    sim_data.at[sim_data.index[i], "strategy"] = "vertical_spread_win"
                else:
                    # Direction was wrong
                    loss_size = min(0.015, abs(day_data["daily_return"]) * 1.5)
                    sim_data.at[sim_data.index[i], "trade_return"] = -loss_size
                    sim_data.at[sim_data.index[i], "strategy"] = "vertical_spread_loss"
                    
        return sim_data
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        strategies: List[str] = ["iron_condor", "vertical_spread"],
        high_vol_only: bool = True
    ) -> Dict[str, Any]:
        """Run backtest of the strategy on historical data.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            strategies (List[str]): List of strategies to test
            high_vol_only (bool): Whether to only trade on high volatility days
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        # Fetch historical data
        data = self.fetch_historical_data(start_date, end_date)
        if data.empty:
            return {"status": "error", "message": "Failed to fetch historical data"}
        
        # Filter to the actual backtest period
        data = data[data.index >= start_date]
        
        results = {}
        
        # Run backtest for each strategy
        for strategy in strategies:
            logger.info(f"Running backtest for {strategy} strategy")
            
            # Simulate strategy
            sim_data = self.simulate_option_strategy(data, strategy, high_vol_only)
            
            # Calculate portfolio value over time
            sim_data["portfolio_return"] = sim_data["trade_return"].fillna(0)
            sim_data["portfolio_value"] = self.initial_capital * (1 + sim_data["portfolio_return"]).cumprod()
            
            # Calculate backtest metrics
            trades = len(sim_data[sim_data["strategy"].notna()])
            winning_trades = len(sim_data[sim_data["trade_return"] > 0])
            win_rate = winning_trades / trades if trades > 0 else 0
            
            final_value = sim_data["portfolio_value"].iloc[-1]
            total_return = (final_value / self.initial_capital) - 1
            
            # Calculate other metrics
            daily_returns = sim_data["portfolio_return"].fillna(0)
            annual_return = ((1 + total_return) ** (252 / len(sim_data))) - 1
            annual_volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            max_drawdown = (sim_data["portfolio_value"] / sim_data["portfolio_value"].cummax() - 1).min()
            
            # Store results
            results[strategy] = {
                "initial_capital": self.initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "trades": trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "portfolio_values": sim_data["portfolio_value"].tolist(),
                "trade_dates": [str(d.date()) for d in sim_data.index],
                "high_vol_days": int(sim_data["is_high_vol"].sum()),
            }
            
        # Save results
        results_file = self.data_dir / f"backtest_results_{start_date}_{end_date}.json"
        with open(results_file, "w") as f:
            # Convert array data to lists for JSON serialization
            json_results = results.copy()
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Backtest complete, results saved to {results_file}")
        return results
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None) -> None:
        """Generate a performance report from backtest results.
        
        Args:
            results (Dict[str, Any]): Backtest results
            output_file (str, optional): Output file for the report
        """
        if not results:
            logger.error("No results to generate report from")
            return
        
        # Set up plotting
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 18))
        
        # Plot portfolio values
        plt.subplot(3, 1, 1)
        for strategy, result in results.items():
            plt.plot(result["trade_dates"], result["portfolio_values"], label=strategy)
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        
        # Plot strategy metrics comparison
        plt.subplot(3, 1, 2)
        metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "win_rate"]
        metric_values = {m: [results[s][m] for s in results.keys()] for m in metrics}
        
        bar_width = 0.2
        index = np.arange(len(metrics))
        
        for i, (strategy, result) in enumerate(results.items()):
            plt.bar(
                index + i * bar_width,
                [result[m] for m in metrics],
                bar_width,
                label=strategy
            )
        
        plt.title("Strategy Metrics Comparison")
        plt.xticks(index + bar_width * (len(results) / 2), metrics)
        plt.legend()
        
        # Plot drawdowns
        plt.subplot(3, 1, 3)
        for strategy, result in results.items():
            portfolio_values = np.array(result["portfolio_values"])
            drawdowns = 1 - portfolio_values / np.maximum.accumulate(portfolio_values)
            plt.plot(result["trade_dates"], drawdowns, label=f"{strategy} Drawdown")
        plt.title("Portfolio Drawdowns")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if output_file:
            plt.tight_layout()
            plt.savefig(output_file)
            logger.info(f"Performance report saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()


def main() -> None:
    """Main entry point for the backtest script."""
    parser = argparse.ArgumentParser(description="Backtest TuringTrader options strategy")
    parser.add_argument(
        "--start-date", 
        required=True,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=datetime.datetime.now().strftime("%Y-%m-%d"),
        help="End date for backtest (YYYY-MM-DD, defaults to today)"
    )
    parser.add_argument(
        "--risk-level",
        default="MEDIUM",
        choices=["LOW", "MEDIUM", "HIGH", "AGGRESSIVE"],
        help="Risk level for backtesting"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtest"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["iron_condor", "vertical_spread"],
        help="Strategies to backtest"
    )
    parser.add_argument(
        "--high-vol-only",
        action="store_true",
        help="Only trade on high volatility days"
    )
    parser.add_argument(
        "--data-dir",
        help="Directory for saving backtest data"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for performance report (e.g., 'report.png')"
    )
    args = parser.parse_args()
    
    # Create and run backtester
    backtester = OptionsBacktester(
        risk_level=args.risk_level,
        initial_capital=args.initial_capital,
        data_dir=args.data_dir
    )
    
    results = backtester.run_backtest(
        args.start_date,
        args.end_date,
        args.strategies,
        args.high_vol_only
    )
    
    # Generate report
    if results:
        backtester.generate_report(results, args.output_file)


if __name__ == "__main__":
    main()