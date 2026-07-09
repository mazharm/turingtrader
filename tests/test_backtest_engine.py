"""
Tests for BacktestEngine mechanics: holding periods, expiry settlement,
and realized-PnL accounting.
"""

import pytest
from datetime import datetime

from ibkr_trader.config import Config
from backtesting.backtest_engine import BacktestEngine
from backtesting.realistic_mock_data import RealisticMockDataFetcher


def _run(holding_days: int, min_dte: int = 21, max_dte: int = 45):
    config = Config()
    config.risk.adjust_for_risk_level(5)
    config.vol_harvesting.min_dte = min_dte
    config.vol_harvesting.max_dte = max_dte

    engine = BacktestEngine(
        config=config,
        initial_balance=100000.0,
        data_fetcher=RealisticMockDataFetcher(seed=20260708),
        holding_days=holding_days,
    )
    results = engine.run_backtest(
        start_date='2023-01-02',
        end_date='2023-03-31',
        risk_level=5,
        use_cache=False,
    )
    return engine, results


def _entry_date_from_position_id(position_id: str) -> datetime:
    return datetime.strptime(position_id.rsplit('_', 1)[-1], '%Y%m%d')


class TestHoldingPeriod:
    def test_default_engine_closes_next_trading_day(self):
        engine, results = _run(holding_days=1)
        closes = [t for t in engine.trade_history if t['action'] == 'close']
        assert closes, "expected the strategy to trade on seeded mock data"

        last_day = max(t['date'] for t in engine.trade_history)
        for close in closes:
            entry = _entry_date_from_position_id(close['symbol'])
            held = (close['date'] - entry).days
            # Next trading day: 1 calendar day, up to 3-4 over weekends/holidays,
            # except the forced end-of-backtest liquidation
            if close['date'] != last_day:
                assert 1 <= held <= 4, (
                    f"1-day hold closed after {held} calendar days: {close['symbol']}"
                )

    def test_multi_day_hold_keeps_positions_open(self):
        engine, results = _run(holding_days=5)
        closes = [t for t in engine.trade_history if t['action'] == 'close']
        assert closes, "expected the strategy to trade on seeded mock data"

        last_day = max(t['date'] for t in engine.trade_history)
        regular_closes = [c for c in closes if c['date'] != last_day]
        assert regular_closes, "expected at least one close before the final day"
        for close in regular_closes:
            entry = _entry_date_from_position_id(close['symbol'])
            held = (close['date'] - entry).days
            # 5 trading days spans at least 6 calendar days (one weekend)
            assert held >= 5, (
                f"5-day hold closed after only {held} calendar days: {close['symbol']}"
            )

    def test_short_dte_settles_at_expiry_before_holding_period(self):
        engine, results = _run(holding_days=10, min_dte=1, max_dte=3)
        closes = [t for t in engine.trade_history if t['action'] == 'close']
        if not closes:
            pytest.skip("no short-DTE trades generated on this data")

        for close in closes:
            entry = _entry_date_from_position_id(close['symbol'])
            held = (close['date'] - entry).days
            # Expiry (1-3 days out) must force the close well before 10 days
            assert held <= 5, (
                f"short-DTE position outlived its expiry: held {held} days"
            )

    def test_no_positions_left_open_at_end(self):
        engine, _ = _run(holding_days=5)
        assert engine.positions == {}, "end of backtest must liquidate everything"


class TestRealizedPnlAccounting:
    def test_balance_equals_initial_plus_closed_pnl(self):
        engine, results = _run(holding_days=1)
        closes = [t for t in engine.trade_history if t['action'] == 'close']
        realized = sum(t['pnl'] for t in closes)
        assert engine.current_balance == pytest.approx(100000.0 + realized), (
            "balance must move only by realized close PnL (no double-counted credits)"
        )
