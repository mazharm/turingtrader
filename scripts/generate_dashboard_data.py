#!/usr/bin/env python3
"""
Generate static JSON data for the GitHub Pages dashboard.

Runs the existing backtesting framework and writes JSON to dashboard/public/data/.
These JSON files are consumed by the React SPA at runtime.

Usage:
    python scripts/generate_dashboard_data.py [--test-mode] [--period DAYS] [--initial-investment AMOUNT]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr_trader.config import Config, RiskParameters
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.mock_data import MockDataFetcher
from backtesting.realistic_mock_data import RealisticMockDataFetcher
from historical_data.data_fetcher import HistoricalDataFetcher


def get_risk_parameters(level: int) -> dict:
    """Get the computed risk parameters for a given level."""
    rp = RiskParameters()
    rp.adjust_for_risk_level(level)
    return {
        "risk_level": rp.risk_level,
        "max_daily_risk_pct": round(rp.max_daily_risk_pct, 2),
        "min_volatility_threshold": round(rp.min_volatility_threshold, 1),
        "max_position_size_pct": round(rp.max_position_size_pct, 1),
        "max_delta_exposure": round(rp.max_delta_exposure, 1),
        "stop_loss_pct": round(rp.stop_loss_pct, 1),
        "target_profit_pct": round(rp.target_profit_pct, 1),
        "min_volatility_change": round(rp.min_volatility_change, 2),
        "condor_stop_loss_factor_of_max_risk": round(rp.condor_stop_loss_factor_of_max_risk, 1),
        "condor_profit_target_factor_of_credit": round(rp.condor_profit_target_factor_of_credit, 1),
    }


def serialize_daily_values(daily_values: list) -> list:
    """Convert daily_values to JSON-serializable format."""
    result = []
    for dv in daily_values:
        date = dv["date"]
        if hasattr(date, "strftime"):
            date = date.strftime("%Y-%m-%d")
        result.append({
            "date": str(date),
            "balance": round(dv["balance"], 2),
        })
    return result


def run_backtests(args) -> dict:
    """Run backtests for all risk levels and return results."""
    config = Config()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.period)).strftime("%Y-%m-%d")

    # Select data fetcher
    if args.test_mode:
        data_fetcher = RealisticMockDataFetcher()
        logging.info("Using realistic mock data")
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        data_fetcher = HistoricalDataFetcher(data_dir=data_dir)
        # Test connectivity
        try:
            test_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            test_data = data_fetcher.fetch_data("SPY", test_start, end_date, use_cache=False, max_retries=2)
            if test_data.empty:
                raise ValueError("Empty data returned")
            logging.info("Using real Yahoo Finance data")
        except Exception as e:
            logging.warning(f"Yahoo Finance unavailable ({e}), falling back to realistic mock data")
            data_fetcher = RealisticMockDataFetcher()

    engine = BacktestEngine(
        config=config,
        initial_balance=args.initial_investment,
        data_fetcher=data_fetcher,
    )
    analyzer = PerformanceAnalyzer()

    results_by_risk = {}
    for level in range(1, 11):
        logging.info(f"Running backtest for risk level {level}...")
        raw = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            risk_level=level,
            use_cache=True,
        )
        if "error" in raw:
            logging.error(f"Risk level {level} failed: {raw['error']}")
            continue
        enriched = analyzer.analyze_results(raw, risk_level=level)
        results_by_risk[level] = enriched

    return results_by_risk, start_date, end_date


def build_summary(results_by_risk: dict, initial_investment: float, start_date: str, end_date: str) -> dict:
    """Build summary.json content."""
    analyzer = PerformanceAnalyzer()
    comparison = analyzer.compare_risk_levels(results_by_risk)

    risk_levels = []
    for level in sorted(results_by_risk.keys()):
        r = results_by_risk[level]
        risk_levels.append({
            "risk_level": level,
            "total_return_pct": round(r.get("total_return_pct", 0), 2),
            "annualized_return_pct": round(r.get("annualized_return_pct", 0), 2),
            "annualized_volatility_pct": round(r.get("annualized_volatility_pct", 0), 2),
            "max_drawdown_pct": round(r.get("max_drawdown_pct", 0), 2),
            "sharpe_ratio": round(r.get("sharpe_ratio", 0), 2),
            "sortino_ratio": round(r.get("sortino_ratio", 0), 2),
            "calmar_ratio": round(r.get("calmar_ratio", 0), 2),
            "win_rate": round(r.get("win_rate", 0), 1),
            "profit_factor": round(r.get("profit_factor", 0), 2),
            "total_trades": r.get("trades", 0),
            "best_day_pct": round(r.get("best_day_pct", 0), 2),
            "worst_day_pct": round(r.get("worst_day_pct", 0), 2),
            "profitable_months_pct": round(r.get("profitable_months_pct", 0), 1),
            "final_balance": round(r.get("final_balance", 0), 2),
            "max_winning_streak": r.get("max_winning_streak", 0),
            "max_losing_streak": r.get("max_losing_streak", 0),
        })

    return {
        "generated_at": datetime.now().isoformat() + "Z",
        "start_date": start_date,
        "end_date": end_date,
        "initial_investment": initial_investment,
        "period_days": (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days,
        "optimal_risk_level": comparison.get("optimal_risk_level"),
        "best_return_risk_level": comparison.get("best_return_risk_level"),
        "risk_levels": risk_levels,
    }


def build_risk_level_file(level: int, results: dict) -> dict:
    """Build risk_level_{N}.json content."""
    return {
        "risk_level": level,
        "metrics": {
            "total_return_pct": round(results.get("total_return_pct", 0), 2),
            "annualized_return_pct": round(results.get("annualized_return_pct", 0), 2),
            "annualized_volatility_pct": round(results.get("annualized_volatility_pct", 0), 2),
            "max_drawdown_pct": round(results.get("max_drawdown_pct", 0), 2),
            "sharpe_ratio": round(results.get("sharpe_ratio", 0), 2),
            "sortino_ratio": round(results.get("sortino_ratio", 0), 2),
            "calmar_ratio": round(results.get("calmar_ratio", 0), 2),
            "win_rate": round(results.get("win_rate", 0), 1),
            "profit_factor": round(results.get("profit_factor", 0), 2),
            "total_trades": results.get("trades", 0),
            "best_day_pct": round(results.get("best_day_pct", 0), 2),
            "worst_day_pct": round(results.get("worst_day_pct", 0), 2),
            "best_month_pct": round(results.get("best_month_pct", 0), 2),
            "worst_month_pct": round(results.get("worst_month_pct", 0), 2),
            "profitable_months_pct": round(results.get("profitable_months_pct", 0), 1),
            "max_winning_streak": results.get("max_winning_streak", 0),
            "max_losing_streak": results.get("max_losing_streak", 0),
            "initial_balance": results.get("initial_balance", 100000),
            "final_balance": round(results.get("final_balance", 0), 2),
        },
        "daily_values": serialize_daily_values(results.get("daily_values", [])),
        "risk_parameters": get_risk_parameters(level),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate dashboard JSON data")
    parser.add_argument("--test-mode", action="store_true", help="Use realistic mock data")
    parser.add_argument("--period", type=int, default=365, help="Backtest period in days (default: 365)")
    parser.add_argument("--initial-investment", type=float, default=100000.0, help="Initial investment (default: 100000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for JSON files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dashboard", "public", "data"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Run backtests
    results_by_risk, start_date, end_date = run_backtests(args)

    if not results_by_risk:
        logging.error("No backtest results generated")
        return 1

    # Write summary.json
    summary = build_summary(results_by_risk, args.initial_investment, start_date, end_date)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote {summary_path}")

    # Write risk_level_{N}.json files
    for level, results in sorted(results_by_risk.items()):
        data = build_risk_level_file(level, results)
        file_path = os.path.join(output_dir, f"risk_level_{level}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Wrote {file_path}")

    logging.info(f"Done. {len(results_by_risk)} risk level files + summary.json written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
