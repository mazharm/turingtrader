"""
Tests for the live-trading safety behavior of TuringTrader and the
order-execution helpers of IBConnector (with the IB API fully mocked).

Covered:
- Pre-trade risk gate blocks entries (halt, daily loss, trade cap)
- Positions are registered with FILLED quantity, not requested quantity
- Stop-loss / profit-target monitoring closes positions
- Kill switch flattens and halts
- Trading-cycle gate ordering (kill switch first, EOD flatten before entries)
- Price-walk executor: ladder direction, partial-fill accounting
"""

from unittest.mock import MagicMock, patch

import pytest

from ibkr_trader.config import Config
from ibkr_trader.ib_connector import IBConnector
from ibkr_trader.trader import TuringTrader


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_trader(tmp_path) -> TuringTrader:
    """Build a TuringTrader with a mocked connector and isolated kill switch."""
    trader = TuringTrader(str(tmp_path / "no_such_config.ini"))
    trader.ib_connector = MagicMock()
    trader.ib_connector.check_connection.return_value = True
    trader.ib_connector.is_market_open.return_value = True
    trader.config.trading.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    trader.risk_manager.update_account_value(100000.0)
    trader.account_value = 100000.0
    return trader


def condor_decision(quantity=2) -> dict:
    return {
        'action': 'iron_condor',
        'symbol': 'SPY',
        'expiry': '20260814',
        'short_call_strike': 415.0,
        'short_put_strike': 395.0,
        'long_call_strike': 420.0,
        'long_put_strike': 390.0,
        'quantity': quantity,
        'net_credit': 1.50,
        'max_risk_per_spread': 350.0,
        'max_loss': 350.0 * quantity,
    }


# ----------------------------------------------------------------------
# execute_trade
# ----------------------------------------------------------------------

class TestExecuteTradeGates:
    def test_blocked_when_halted(self, tmp_path):
        trader = make_trader(tmp_path)
        trader.risk_manager.halt_trading("test halt")

        result = trader.execute_trade(condor_decision())
        assert result['executed'] is False
        assert 'risk_gate' in result['reason']
        trader.ib_connector.execute_limit_order_with_price_walk.assert_not_called()

    def test_blocked_when_daily_loss_limit_hit(self, tmp_path):
        trader = make_trader(tmp_path)
        trader.risk_manager.update_daily_pnl(-(trader.risk_manager.max_daily_risk_amount + 1))

        result = trader.execute_trade(condor_decision())
        assert result['executed'] is False
        assert 'risk_gate' in result['reason']

    def test_blocked_when_trade_cap_reached(self, tmp_path):
        trader = make_trader(tmp_path)
        trader.config.trading.max_daily_trades = 1
        trader.risk_manager.daily_trades = 1

        result = trader.execute_trade(condor_decision())
        assert result['executed'] is False
        assert 'risk_gate' in result['reason']

    def test_invalid_quantity_rejected(self, tmp_path):
        trader = make_trader(tmp_path)
        decision = condor_decision(quantity=0)
        decision['max_loss'] = 0
        result = trader.execute_trade(decision)
        assert result['executed'] is False


class TestExecuteTradeFills:
    def test_registers_filled_quantity_not_requested(self, tmp_path):
        """Partial fill: risk manager must track 1 contract, not 2."""
        trader = make_trader(tmp_path)
        trader.ib_connector.create_iron_condor_contract.return_value = MagicMock()
        trader.ib_connector.execute_limit_order_with_price_walk.return_value = {
            'filled_quantity': 1, 'avg_price': 1.48, 'complete': False, 'trades': [],
        }

        result = trader.execute_trade(condor_decision(quantity=2))

        assert result['executed'] is True
        assert result['quantity'] == 1
        assert result['requested_quantity'] == 2

        positions = trader.risk_manager.get_open_positions()
        assert len(positions) == 1
        position = next(iter(positions.values()))
        assert position['quantity'] == 1
        assert position['entry_price'] == pytest.approx(1.48)

    def test_no_fill_means_no_position(self, tmp_path):
        trader = make_trader(tmp_path)
        trader.ib_connector.create_iron_condor_contract.return_value = MagicMock()
        trader.ib_connector.execute_limit_order_with_price_walk.return_value = {
            'filled_quantity': 0, 'avg_price': 0.0, 'complete': False, 'trades': [],
        }

        result = trader.execute_trade(condor_decision())
        assert result['executed'] is False
        assert not trader.risk_manager.get_open_positions()
        assert trader.day_trade_count == 0

    def test_option_buy_requires_price_reference(self, tmp_path):
        """Never send an unprotected market order for a single option."""
        trader = make_trader(tmp_path)
        result = trader.execute_trade({
            'action': 'buy', 'symbol': 'SPY', 'expiry': '20260814',
            'strike': 400.0, 'option_type': 'call', 'quantity': 1,
            # no 'price'
        })
        assert result['executed'] is False
        assert result['reason'] == 'no_price_reference'


# ----------------------------------------------------------------------
# Position monitoring
# ----------------------------------------------------------------------

def open_test_condor(trader, credit=1.50):
    trader.risk_manager.add_position(
        symbol="SPY_IC_TEST",
        quantity=1,
        entry_price=credit,
        position_type="iron_condor",
        option_data={
            'short_put_strike': 395.0, 'long_put_strike': 390.0,
            'short_call_strike': 415.0, 'long_call_strike': 420.0,
            'max_risk_per_spread': 350.0,
            'contract': MagicMock(),
        },
    )
    return trader.risk_manager.current_positions["SPY_IC_TEST"]


class TestPositionMonitoring:
    def test_stop_loss_closes_position(self, tmp_path):
        trader = make_trader(tmp_path)
        position = open_test_condor(trader)
        breach_cost = position['stop_loss_value'] + 0.05

        trader.ib_connector.get_combo_close_price.return_value = breach_cost
        trader.ib_connector.execute_limit_order_with_price_walk.return_value = {
            'filled_quantity': 1, 'avg_price': breach_cost, 'complete': True, 'trades': [],
        }

        exits = trader.monitor_open_positions()

        assert len(exits) == 1
        assert exits[0]['reason'] == 'stop_loss'
        assert exits[0]['closed'] is True
        assert not trader.risk_manager.get_open_positions()
        # Loss must land in daily P&L: (1.50 - breach) * 100
        assert trader.risk_manager.daily_pnl == pytest.approx((1.50 - breach_cost) * 100)

    def test_take_profit_closes_position(self, tmp_path):
        trader = make_trader(tmp_path)
        position = open_test_condor(trader)
        target_cost = position['take_profit_value'] - 0.05

        trader.ib_connector.get_combo_close_price.return_value = target_cost
        trader.ib_connector.execute_limit_order_with_price_walk.return_value = {
            'filled_quantity': 1, 'avg_price': target_cost, 'complete': True, 'trades': [],
        }

        exits = trader.monitor_open_positions()
        assert len(exits) == 1
        assert exits[0]['reason'] == 'take_profit'
        assert trader.risk_manager.daily_pnl > 0

    def test_open_position_not_closed_without_trigger(self, tmp_path):
        trader = make_trader(tmp_path)
        open_test_condor(trader, credit=1.50)

        trader.ib_connector.get_combo_close_price.return_value = 1.50  # unchanged

        exits = trader.monitor_open_positions()
        assert exits == []
        assert len(trader.risk_manager.get_open_positions()) == 1

    def test_no_quote_skips_position_safely(self, tmp_path):
        trader = make_trader(tmp_path)
        open_test_condor(trader)
        trader.ib_connector.get_combo_close_price.return_value = None

        exits = trader.monitor_open_positions()
        assert exits == []
        assert len(trader.risk_manager.get_open_positions()) == 1

    def test_failed_limit_close_escalates_to_market(self, tmp_path):
        trader = make_trader(tmp_path)
        position = open_test_condor(trader)
        breach_cost = position['stop_loss_value'] + 0.05

        trader.ib_connector.get_combo_close_price.return_value = breach_cost
        # Limit walk fills nothing...
        trader.ib_connector.execute_limit_order_with_price_walk.return_value = {
            'filled_quantity': 0, 'avg_price': 0.0, 'complete': False, 'trades': [],
        }
        # ...market order picks up the remainder
        market_trade = MagicMock()
        market_trade.orderStatus.avgFillPrice = breach_cost + 0.10
        trader.ib_connector.market_order.return_value = market_trade
        trader.ib_connector.wait_for_fill.return_value = True

        exits = trader.monitor_open_positions()

        trader.ib_connector.market_order.assert_called_once()
        assert exits[0]['closed'] is True
        assert not trader.risk_manager.get_open_positions()


# ----------------------------------------------------------------------
# Kill switch and cycle gate ordering
# ----------------------------------------------------------------------

class TestKillSwitch:
    def test_kill_switch_flattens_and_halts(self, tmp_path):
        trader = make_trader(tmp_path)
        open_test_condor(trader)
        trader.ib_connector.flatten_all_positions.return_value = {
            'all_closed': True, 'closed': [], 'failed': [],
        }
        trader.ib_connector.get_combo_close_price.return_value = 1.50

        (tmp_path / "KILL_SWITCH").write_text("stop")

        assert trader.check_kill_switch() is True
        assert trader.risk_manager.trading_halted
        trader.ib_connector.flatten_all_positions.assert_called_once()
        assert not trader.risk_manager.get_open_positions()

    def test_cycle_halts_before_anything_else(self, tmp_path):
        trader = make_trader(tmp_path)
        (tmp_path / "KILL_SWITCH").write_text("stop")
        trader.ib_connector.flatten_all_positions.return_value = {
            'all_closed': True, 'closed': [], 'failed': [],
        }

        result = trader.run_trading_cycle()
        assert result['action'] == 'halt'
        # Market status must not even be consulted once the switch is thrown
        trader.ib_connector.is_market_open.assert_not_called()

    def test_no_kill_switch_no_halt(self, tmp_path):
        trader = make_trader(tmp_path)
        assert trader.check_kill_switch() is False
        assert not trader.risk_manager.trading_halted


class TestCycleGateOrdering:
    def test_eod_flatten_before_new_entries(self, tmp_path):
        trader = make_trader(tmp_path)
        open_test_condor(trader)
        trader.ib_connector.get_account_value.return_value = 100000.0
        trader.ib_connector.get_cash_balance.return_value = 100000.0
        trader.ib_connector.get_combo_close_price.return_value = 1.50
        trader.ib_connector.flatten_all_positions.return_value = {
            'all_closed': True, 'closed': [], 'failed': [],
        }

        with patch.object(trader, 'check_market_status', return_value=True), \
             patch.object(trader.options_strategy, 'should_close_positions', return_value=True), \
             patch.object(trader, 'fetch_option_chain') as mock_chain:
            result = trader.run_trading_cycle()

        assert result['action'] == 'close_all'
        mock_chain.assert_not_called()
        assert not trader.risk_manager.get_open_positions()

    def test_holds_open_position_without_new_entry(self, tmp_path):
        trader = make_trader(tmp_path)
        open_test_condor(trader)
        trader.ib_connector.get_account_value.return_value = 100000.0
        trader.ib_connector.get_cash_balance.return_value = 100000.0
        trader.ib_connector.get_combo_close_price.return_value = 1.50  # no trigger

        with patch.object(trader, 'check_market_status', return_value=True), \
             patch.object(trader.options_strategy, 'should_close_positions', return_value=False), \
             patch.object(trader, 'fetch_option_chain') as mock_chain:
            result = trader.run_trading_cycle()

        assert result['action'] == 'hold'
        mock_chain.assert_not_called()

    def test_halted_risk_manager_blocks_entries_in_cycle(self, tmp_path):
        trader = make_trader(tmp_path)
        trader.ib_connector.get_account_value.return_value = 100000.0
        trader.ib_connector.get_cash_balance.return_value = 100000.0
        trader.risk_manager.halt_trading("test")

        with patch.object(trader, 'check_market_status', return_value=True), \
             patch.object(trader.options_strategy, 'should_close_positions', return_value=False), \
             patch.object(trader, 'fetch_market_data') as mock_data:
            result = trader.run_trading_cycle()

        assert result['action'] == 'none'
        assert 'risk_gate' in result['reason']
        mock_data.assert_not_called()


# ----------------------------------------------------------------------
# Price-walk executor (IBConnector)
# ----------------------------------------------------------------------

def make_connector() -> IBConnector:
    config = Config('no_such_config_file.ini')
    connector = IBConnector(config)
    connector.ib = MagicMock()
    connector.connected = True
    connector.ib.isConnected.return_value = True
    return connector


def fake_trade(filled=0, avg_price=0.0, status='Submitted'):
    trade = MagicMock()
    trade.orderStatus.filled = filled
    trade.orderStatus.avgFillPrice = avg_price
    trade.orderStatus.status = status
    return trade


class TestPriceWalkExecutor:
    def test_fill_on_first_attempt(self):
        connector = make_connector()
        trade = fake_trade(filled=2, avg_price=1.52, status='Filled')
        connector.ib.placeOrder.return_value = trade

        with patch.object(connector, 'wait_for_fill', return_value=True):
            result = connector.execute_limit_order_with_price_walk(
                contract=MagicMock(), action='SELL', quantity=2,
                initial_price=1.53, worst_price=1.42, max_attempts=3,
                fill_timeout_seconds=0.1,
            )

        assert result['complete'] is True
        assert result['filled_quantity'] == 2
        assert result['avg_price'] == pytest.approx(1.52)
        assert connector.ib.placeOrder.call_count == 1

    def test_sell_prices_walk_down_and_partials_accumulate(self):
        connector = make_connector()
        trades = [
            fake_trade(filled=0),                                  # attempt 1: nothing
            fake_trade(filled=1, avg_price=1.50),                  # attempt 2: 1 of 3
            fake_trade(filled=2, avg_price=1.45, status='Filled'), # attempt 3: rest
        ]
        connector.ib.placeOrder.side_effect = trades

        with patch.object(connector, 'wait_for_fill', side_effect=[False, False, True]), \
             patch.object(connector, 'cancel_and_wait', side_effect=[0, 1]):
            result = connector.execute_limit_order_with_price_walk(
                contract=MagicMock(), action='SELL', quantity=3,
                initial_price=1.55, worst_price=1.45, max_attempts=3,
                fill_timeout_seconds=0.1,
            )

        assert result['complete'] is True
        assert result['filled_quantity'] == 3
        # Weighted: (1.50*1 + 1.45*2) / 3
        assert result['avg_price'] == pytest.approx((1.50 + 2 * 1.45) / 3)

        # Ladder walks DOWN for SELL; remaining quantity shrinks after partials
        placed_orders = [call.args[1] for call in connector.ib.placeOrder.call_args_list]
        prices = [order.lmtPrice for order in placed_orders]
        quantities = [order.totalQuantity for order in placed_orders]
        assert prices == sorted(prices, reverse=True)
        assert quantities == [3, 3, 2]

    def test_buy_prices_walk_up(self):
        connector = make_connector()
        connector.ib.placeOrder.side_effect = [fake_trade(), fake_trade(), fake_trade()]

        with patch.object(connector, 'wait_for_fill', return_value=False), \
             patch.object(connector, 'cancel_and_wait', return_value=0):
            result = connector.execute_limit_order_with_price_walk(
                contract=MagicMock(), action='BUY', quantity=1,
                initial_price=1.00, worst_price=1.15, max_attempts=3,
                fill_timeout_seconds=0.1,
            )

        assert result['complete'] is False
        assert result['filled_quantity'] == 0
        prices = [call.args[1].lmtPrice for call in connector.ib.placeOrder.call_args_list]
        assert prices == sorted(prices)
        assert prices[0] == pytest.approx(1.00)
        assert prices[-1] == pytest.approx(1.15)

    def test_orders_are_day_orders(self):
        """Intraday strategy must never leave GTC orders at the broker."""
        connector = make_connector()
        connector.ib.placeOrder.return_value = fake_trade(filled=1, status='Filled')

        with patch.object(connector, 'wait_for_fill', return_value=True):
            connector.execute_limit_order_with_price_walk(
                contract=MagicMock(), action='SELL', quantity=1,
                initial_price=1.50, worst_price=1.40, max_attempts=1,
                fill_timeout_seconds=0.1,
            )

        order = connector.ib.placeOrder.call_args.args[1]
        assert order.tif == 'DAY'

    def test_rejects_invalid_action(self):
        connector = make_connector()
        result = connector.execute_limit_order_with_price_walk(
            contract=MagicMock(), action='HOLD', quantity=1,
            initial_price=1.0, worst_price=0.9,
        )
        assert result['filled_quantity'] == 0
        connector.ib.placeOrder.assert_not_called()


class TestIronCondorOrderSignature:
    def test_accepts_action_keyword(self):
        """trader.py calls submit_iron_condor_order(action='SELL'); this used
        to raise TypeError on every single iron condor trade."""
        connector = make_connector()
        with patch.object(connector, 'create_iron_condor_contract') as mock_create, \
             patch.object(connector, 'check_connection', return_value=True):
            mock_create.return_value = MagicMock()
            connector.ib.placeOrder.return_value = fake_trade(status='Filled')
            trade = connector.submit_iron_condor_order(
                symbol='SPY', expiry='20260814',
                short_put_strike=395.0, short_call_strike=415.0,
                long_put_strike=390.0, long_call_strike=420.0,
                quantity=1, limit_price=1.50, action='SELL',
            )
        assert trade is not None

    def test_rejects_bad_action(self):
        connector = make_connector()
        with patch.object(connector, 'create_iron_condor_contract') as mock_create, \
             patch.object(connector, 'check_connection', return_value=True):
            mock_create.return_value = MagicMock()
            trade = connector.submit_iron_condor_order(
                symbol='SPY', expiry='20260814',
                short_put_strike=395.0, short_call_strike=415.0,
                long_put_strike=390.0, long_call_strike=420.0,
                quantity=1, limit_price=1.50, action='NONSENSE',
            )
        assert trade is None
