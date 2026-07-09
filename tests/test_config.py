"""
Tests for configuration loading (ibkr_trader.config).

These cover the bugs that previously made live trading impossible:
- Config(config_path) crashed (path was swallowed as the first dataclass field)
- config.ini was silently ignored (load_config was never called)
- boolean values crashed the parser (parser.getboolean misused as a converter)
"""

import os
import textwrap

import pytest

from ibkr_trader.config import Config, AppConfig, RiskParameters


@pytest.fixture
def ini_file(tmp_path):
    """Write a config.ini exercising every section and value type."""
    path = tmp_path / "test_config.ini"
    path.write_text(textwrap.dedent("""
        [IBKR]
        host = 10.0.0.5
        port = 7496
        client_id = 42
        read_only = true

        [Risk]
        risk_level = 3
        max_daily_risk_pct = 9.9

        [Trading]
        index_symbol = QQQ
        options_only = false
        max_daily_trades = 2
        kill_switch_file = STOP_NOW
    """))
    return str(path)


class TestConfigConstruction:
    def test_default_construction(self):
        config = Config()
        assert config.ibkr.host == "127.0.0.1"
        assert config.risk.risk_level == 5
        assert config.trading.index_symbol == "SPY"

    def test_construction_with_path_loads_file(self, ini_file):
        """Config('path.ini') must load the file, not crash."""
        config = Config(ini_file)
        assert config.ibkr.host == "10.0.0.5"
        assert config.ibkr.port == 7496
        assert config.ibkr.client_id == 42
        assert config.trading.index_symbol == "QQQ"

    def test_config_is_appconfig_alias(self):
        assert Config is AppConfig

    def test_missing_file_uses_defaults(self, tmp_path):
        config = Config(str(tmp_path / "does_not_exist.ini"))
        assert config.ibkr.host == "127.0.0.1"


class TestBooleanParsing:
    def test_read_only_true(self, ini_file):
        config = Config(ini_file)
        assert config.ibkr.read_only is True

    def test_options_only_false(self, ini_file):
        config = Config(ini_file)
        assert config.trading.options_only is False


class TestRiskLevelInteraction:
    def test_risk_level_scaling_applied(self, ini_file):
        """risk_level=3 must scale the derived parameters."""
        config = Config(ini_file)
        assert config.risk.risk_level == 3

        reference = RiskParameters()
        reference.adjust_for_risk_level(3)
        # Explicit override wins over scaling...
        assert config.risk.max_daily_risk_pct == 9.9
        # ...but non-overridden parameters follow the scaled values.
        assert config.risk.max_position_size_pct == reference.max_position_size_pct
        assert config.risk.stop_loss_pct == reference.stop_loss_pct

    def test_explicit_value_overrides_scaling(self, ini_file):
        """A per-parameter value in the file must survive risk-level scaling."""
        config = Config(ini_file)
        assert config.risk.max_daily_risk_pct == 9.9


class TestSafetySettings:
    def test_new_trading_safety_fields_loaded(self, ini_file):
        config = Config(ini_file)
        assert config.trading.max_daily_trades == 2
        assert config.trading.kill_switch_file == "STOP_NOW"

    def test_safety_defaults(self):
        config = Config()
        assert config.trading.max_daily_trades > 0
        assert config.trading.max_consecutive_errors > 0
        assert config.trading.kill_switch_file
        assert config.trading.order_fill_timeout_seconds > 0
        assert config.trading.market_timezone == "America/New_York"
