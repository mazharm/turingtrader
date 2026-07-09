"""
Tests for risk management (ibkr_trader.risk_manager).

These cover the bugs that would have destroyed live P&L:
- Iron condor stop-loss threshold was BELOW the entry credit (fired instantly)
  and mixed dollars with per-share points (then never fired)
- Vertical spread P&L used long-position sign (profit/loss inverted)
- Daily loss limit was computed but never halted trading
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from ibkr_trader.config import Config
from ibkr_trader.risk_manager import RiskManager


@pytest.fixture
def rm():
    config = Config()
    config.risk.adjust_for_risk_level(5)
    manager = RiskManager(config)
    manager.update_account_value(100000.0)
    return manager


def make_condor(rm, credit=1.50, wing_width=5.0, quantity=2):
    """Open a standard test condor: credit 1.50, wings 5 wide -> risk 3.50/share."""
    rm.add_position(
        symbol="SPY_IC_TEST",
        quantity=quantity,
        entry_price=credit,
        position_type="iron_condor",
        option_data={
            'short_put_strike': 395.0,
            'long_put_strike': 395.0 - wing_width,
            'short_call_strike': 415.0,
            'long_call_strike': 415.0 + wing_width,
            'max_risk_per_spread': (wing_width - credit) * 100,  # dollars/contract
        },
    )
    return rm.current_positions["SPY_IC_TEST"]


class TestCondorStopLossTakeProfit:
    def test_stop_loss_is_above_entry_credit(self, rm):
        """The SL threshold is a debit LARGER than the credit received."""
        position = make_condor(rm)
        assert position['stop_loss_value'] > position['entry_price']

    def test_take_profit_is_below_entry_credit(self, rm):
        position = make_condor(rm)
        assert 0 < position['take_profit_value'] < position['entry_price']

    def test_thresholds_match_risk_parameters(self, rm):
        credit, width = 1.50, 5.0
        position = make_condor(rm, credit=credit, wing_width=width)
        max_risk_points = width - credit  # 3.50

        sl_factor = rm.risk_params.condor_stop_loss_factor_of_max_risk / 100.0
        tp_factor = rm.risk_params.condor_profit_target_factor_of_credit / 100.0

        assert position['stop_loss_value'] == pytest.approx(credit + max_risk_points * sl_factor)
        assert position['take_profit_value'] == pytest.approx(credit * (1 - tp_factor))

    def test_no_status_change_at_entry_price(self, rm):
        """At entry the position must be 'open', not instantly stopped out."""
        make_condor(rm, credit=1.50)
        update = rm.update_position("SPY_IC_TEST", 1.50)
        assert update['status'] == 'open'

    def test_stop_loss_triggers_when_cost_rises(self, rm):
        position = make_condor(rm, credit=1.50)
        breach = position['stop_loss_value'] + 0.01
        update = rm.update_position("SPY_IC_TEST", breach)
        assert update['status'] == 'stop_loss'

    def test_take_profit_triggers_when_cost_falls(self, rm):
        position = make_condor(rm, credit=1.50)
        target = position['take_profit_value'] - 0.01
        update = rm.update_position("SPY_IC_TEST", target)
        assert update['status'] == 'take_profit'


class TestCreditSpreadPnL:
    def test_condor_profit_when_cost_falls(self, rm):
        make_condor(rm, credit=1.50, quantity=2)
        update = rm.update_position("SPY_IC_TEST", 0.50)
        # (1.50 - 0.50) * 2 contracts * 100 = +$200
        assert update['position']['pnl'] == pytest.approx(200.0)

    def test_condor_loss_when_cost_rises(self, rm):
        make_condor(rm, credit=1.50, quantity=2)
        update = rm.update_position("SPY_IC_TEST", 2.50)
        assert update['position']['pnl'] == pytest.approx(-200.0)

    def test_vertical_spread_pnl_sign(self, rm):
        """Credit vertical: buying back cheaper is PROFIT (sign was inverted before)."""
        rm.add_position(
            symbol="SPY_bull_put_TEST",
            quantity=1,
            entry_price=1.00,
            position_type="vertical_spread",
            option_data={'short_strike': 400.0, 'long_strike': 395.0, 'width': 5.0},
        )
        update = rm.update_position("SPY_bull_put_TEST", 0.40)
        # (1.00 - 0.40) * 1 * 100 = +$60
        assert update['position']['pnl'] == pytest.approx(60.0)

        result = rm.close_position("SPY_bull_put_TEST", 0.40)
        assert result['pnl'] == pytest.approx(60.0)

    def test_vertical_spread_has_sl_tp(self, rm):
        rm.add_position(
            symbol="SPY_bear_call_TEST",
            quantity=1,
            entry_price=1.20,
            position_type="vertical_spread",
            option_data={'short_strike': 420.0, 'long_strike': 425.0, 'width': 5.0},
        )
        position = rm.current_positions["SPY_bear_call_TEST"]
        assert position['stop_loss_value'] > 1.20
        assert position['take_profit_value'] < 1.20


class TestDailyLossHalt:
    def test_loss_within_limit_keeps_trading(self, rm):
        assert rm.update_daily_pnl(-100.0) is True
        assert not rm.trading_halted

    def test_loss_beyond_limit_halts(self, rm):
        big_loss = -(rm.max_daily_risk_amount + 1.0)
        assert rm.update_daily_pnl(big_loss) is False
        assert rm.trading_halted

        allowed, reason = rm.can_open_new_position()
        assert not allowed
        assert 'halted' in reason

    def test_new_day_clears_halt(self, rm):
        rm.update_daily_pnl(-(rm.max_daily_risk_amount + 1.0))
        assert rm.trading_halted
        rm.reset_daily_metrics()
        assert not rm.trading_halted
        allowed, _ = rm.can_open_new_position()
        assert allowed


class TestPreTradeGate:
    def test_allows_by_default(self, rm):
        allowed, reason = rm.can_open_new_position(max_daily_trades=5)
        assert allowed and reason == ""

    def test_blocks_at_daily_trade_cap(self, rm):
        for _ in range(3):
            rm.update_daily_pnl(10.0)  # 3 trades, small profits
        allowed, reason = rm.can_open_new_position(max_daily_trades=3)
        assert not allowed
        assert 'trade cap' in reason

    def test_blocks_when_proposed_risk_exceeds_budget(self, rm):
        too_much = rm.max_daily_risk_amount + 1.0
        allowed, reason = rm.can_open_new_position(proposed_risk=too_much)
        assert not allowed
        assert 'budget' in reason

    def test_open_risk_counts_against_budget(self, rm):
        make_condor(rm, credit=1.50, wing_width=5.0, quantity=2)
        open_risk = rm.get_total_open_risk()
        assert open_risk == pytest.approx((5.0 - 1.50) * 2 * 100)

        remaining = rm.max_daily_risk_amount - open_risk
        allowed, _ = rm.can_open_new_position(proposed_risk=max(remaining + 1.0, 1.0))
        assert not allowed

    def test_explicit_halt_and_resume(self, rm):
        rm.halt_trading("test reason")
        allowed, reason = rm.can_open_new_position()
        assert not allowed and 'test reason' in reason
        rm.resume_trading()
        allowed, _ = rm.can_open_new_position()
        assert allowed


class TestShouldCloseForDay:
    def test_naive_datetime_treated_as_market_time(self, rm):
        # 15:45 with a 0.5h offset -> past the 15:30 cutoff
        assert rm.should_close_for_day(datetime(2026, 7, 8, 15, 45)) is True
        # 10:00 is comfortably before the cutoff
        assert rm.should_close_for_day(datetime(2026, 7, 8, 10, 0)) is False

    def test_aware_datetime_converted_to_market_tz(self, rm):
        # 19:45 UTC == 15:45 ET during daylight saving -> past cutoff
        utc_dt = datetime(2026, 7, 8, 19, 45, tzinfo=ZoneInfo("UTC"))
        assert rm.should_close_for_day(utc_dt) is True

        # 14:00 UTC == 10:00 ET -> before cutoff
        utc_dt = datetime(2026, 7, 8, 14, 0, tzinfo=ZoneInfo("UTC"))
        assert rm.should_close_for_day(utc_dt) is False
