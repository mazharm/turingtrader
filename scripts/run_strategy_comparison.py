#!/usr/bin/env python3
"""
Run several strategy variants over the same historical data and emit JSON
for the dashboard's strategy-comparison page.

Each variant is the same volatility-harvesting engine with different option
tenor and holding period — the two dials that change the theta-vs-friction
economics identified in the honest backtest review.

Usage:
    # Run one variant (parallel-friendly):
    python scripts/run_strategy_comparison.py --strategy daily-1dte

    # Merge all variant files into the strategies.json index:
    python scripts/run_strategy_comparison.py --merge

Data files land in dashboard/public/data/ by default:
    strategy_{slug}.json   per-variant metrics + equity curves per risk level
    strategies.json        index consumed by the SPA
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr_trader.config import Config
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.realistic_mock_data import RealisticMockDataFetcher
from historical_data.data_fetcher import HistoricalDataFetcher
from scripts.generate_dashboard_data import describe_data_source, serialize_daily_values

STRATEGY_VARIANTS = [
    {
        "slug": "daily-1dte",
        "name": "0-1 DTE Daily Settlement",
        "description": "Sells the shortest-dated spreads available and lets them "
                       "settle at expiry the next day. Maximum theta capture per "
                       "day of risk; full exposure to overnight moves.",
        "min_dte": 1,
        "max_dte": 3,
        "holding_days": 1,
    },
    {
        "slug": "baseline-30dte-1day",
        "name": "30 DTE, 1-Day Hold (baseline)",
        "description": "The original strategy: sells ~3-6 week spreads and buys "
                       "them back the next day. One day of slow theta against "
                       "entry friction.",
        "min_dte": 21,
        "max_dte": 45,
        "holding_days": 1,
    },
    {
        "slug": "swing-30dte-5day",
        "name": "30 DTE, 5-Day Hold",
        "description": "Sells ~3-6 week spreads and holds them for a trading "
                       "week, amortizing entry friction over five days of theta.",
        "min_dte": 21,
        "max_dte": 45,
        "holding_days": 5,
    },
    {
        "slug": "theta-7dte-expiry",
        "name": "7 DTE Held to Expiry",
        "description": "Sells one-week spreads and holds them to expiration, "
                       "collecting the entire premium when they finish out of "
                       "the money.",
        "min_dte": 5,
        "max_dte": 9,
        "holding_days": 10,  # expiry (~7 days) is reached first
    },
]


def make_data_fetcher(source: str, data_dir: str):
    """Create the data fetcher; 'auto' prefers real data with mock fallback."""
    if source == "mock":
        return RealisticMockDataFetcher(seed=20260708)

    fetcher = HistoricalDataFetcher(data_dir=data_dir)
    if source == "real":
        return fetcher

    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        test = fetcher.fetch_data("SPY", start, end, use_cache=False, max_retries=2)
        if not test.empty:
            return fetcher
    except Exception:
        pass

    logging.warning("Yahoo Finance unavailable, using realistic mock data")
    return RealisticMockDataFetcher(seed=20260708)


def run_variant(variant: dict, args) -> dict:
    """Backtest one strategy variant across all requested risk levels."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    data_fetcher = make_data_fetcher(args.data_source, data_dir)
    analyzer = PerformanceAnalyzer()

    risk_levels_out = []
    daily_values_by_level = {}

    for level in args.risk_levels:
        logging.info(f"[{variant['slug']}] risk level {level}...")
        config = Config()
        config.risk.adjust_for_risk_level(level)
        config.vol_harvesting.min_dte = variant["min_dte"]
        config.vol_harvesting.max_dte = variant["max_dte"]

        engine = BacktestEngine(
            config=config,
            initial_balance=args.initial_investment,
            data_fetcher=data_fetcher,
            holding_days=variant["holding_days"],
        )

        raw = engine.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            risk_level=level,
            use_cache=True,
        )
        if "error" in raw:
            logging.error(f"[{variant['slug']}] risk level {level} failed: {raw['error']}")
            continue

        r = analyzer.analyze_results(raw, risk_level=level)
        risk_levels_out.append({
            "risk_level": level,
            "total_return_pct": round(r.get("total_return_pct", 0), 2),
            "annualized_return_pct": round(r.get("annualized_return_pct", 0), 2),
            "annualized_volatility_pct": round(r.get("annualized_volatility_pct", 0), 2),
            "max_drawdown_pct": round(r.get("max_drawdown_pct", 0), 2),
            "sharpe_ratio": round(r.get("sharpe_ratio", 0), 2),
            "sortino_ratio": round(r.get("sortino_ratio", 0), 2),
            "win_rate": round(r.get("win_rate", 0), 1),
            "profit_factor": round(r.get("profit_factor", 0), 2),
            "total_trades": r.get("trades", 0),
            "final_balance": round(r.get("final_balance", 0), 2),
        })
        daily_values_by_level[str(level)] = serialize_daily_values(r.get("daily_values", []))

    return {
        "slug": variant["slug"],
        "name": variant["name"],
        "description": variant["description"],
        "params": {
            "min_dte": variant["min_dte"],
            "max_dte": variant["max_dte"],
            "holding_days": variant["holding_days"],
        },
        "generated_at": datetime.now().isoformat() + "Z",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_investment": args.initial_investment,
        "data_source": describe_data_source(data_fetcher),
        "risk_levels": risk_levels_out,
        "daily_values_by_level": daily_values_by_level,
    }


def merge_index(output_dir: str) -> dict:
    """Build strategies.json from the per-variant files present on disk."""
    strategies = []
    meta = {}
    for variant in STRATEGY_VARIANTS:
        path = os.path.join(output_dir, f"strategy_{variant['slug']}.json")
        if not os.path.exists(path):
            logging.warning(f"Missing {path}; skipping {variant['slug']}")
            continue
        with open(path) as f:
            data = json.load(f)
        if not meta:
            meta = {k: data[k] for k in
                    ("generated_at", "start_date", "end_date", "initial_investment", "data_source")}
        best = max(data["risk_levels"], key=lambda r: r["sharpe_ratio"], default=None)
        strategies.append({
            "slug": data["slug"],
            "name": data["name"],
            "description": data["description"],
            "params": data["params"],
            "best_sharpe_level": best["risk_level"] if best else None,
            "risk_levels": data["risk_levels"],
        })

    return {**meta, "strategies": strategies}


def main():
    parser = argparse.ArgumentParser(description="Run strategy-variant comparison backtests")
    parser.add_argument("--strategy", type=str, help="Variant slug to run (default: all sequentially)")
    parser.add_argument("--merge", action="store_true", help="Only merge existing variant files into strategies.json")
    parser.add_argument("--data-source", choices=["auto", "real", "mock"], default="real")
    parser.add_argument("--period", type=int, default=365, help="Backtest period in days")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--initial-investment", type=float, default=100000.0)
    parser.add_argument("--risk-levels", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ["ibkr_trader", "backtesting", "historical_data", "urllib3", "yfinance"]:
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    args.risk_levels = [int(x) for x in args.risk_levels.split(",")]
    args.end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    args.start_date = args.start_date or (
        datetime.strptime(args.end_date, "%Y-%m-%d") - timedelta(days=args.period)
    ).strftime("%Y-%m-%d")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard", "public", "data"
    )
    os.makedirs(output_dir, exist_ok=True)

    if not args.merge:
        variants = STRATEGY_VARIANTS
        if args.strategy:
            variants = [v for v in STRATEGY_VARIANTS if v["slug"] == args.strategy]
            if not variants:
                known = ", ".join(v["slug"] for v in STRATEGY_VARIANTS)
                print(f"Unknown strategy '{args.strategy}'. Known: {known}")
                return 1

        for variant in variants:
            result = run_variant(variant, args)
            path = os.path.join(output_dir, f"strategy_{variant['slug']}.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
            logging.info(f"Wrote {path}")

    index = merge_index(output_dir)
    if index.get("strategies"):
        index_path = os.path.join(output_dir, "strategies.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        logging.info(f"Wrote {index_path} ({len(index['strategies'])} strategies)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
