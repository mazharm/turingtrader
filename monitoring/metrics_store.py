"""
Metrics storage module for the TuringTrader monitoring system.
Provides persistence layer for performance and risk metrics to PostgreSQL/TimescaleDB.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr_trader.config import PostgresConfig


class MetricsStore:
    """
    Persistent storage for performance and risk metrics.
    Uses PostgreSQL/TimescaleDB for time-series data storage.
    """

    def __init__(
        self,
        postgres_config: Optional[PostgresConfig] = None,
        auto_connect: bool = True
    ):
        """
        Initialize the metrics store.

        Args:
            postgres_config: PostgreSQL configuration (None for defaults)
            auto_connect: Whether to connect automatically on init
        """
        self.logger = logging.getLogger(__name__)
        self.config = postgres_config or PostgresConfig()
        self._conn = None
        self._connected = False

        if auto_connect and HAS_PSYCOPG2:
            self._connect()

    def _connect(self) -> bool:
        """
        Establish database connection.

        Returns:
            True if connection successful, False otherwise
        """
        if not HAS_PSYCOPG2:
            self.logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
            return False

        try:
            self._conn = psycopg2.connect(self.config.get_connection_string())
            self._connected = True
            self.logger.info("Connected to PostgreSQL for metrics storage")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._connected = False
            return False

    def _ensure_connected(self) -> bool:
        """Ensure database connection is active."""
        if not self._connected or self._conn is None:
            return self._connect()
        try:
            # Test connection
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            self._connected = False
            return self._connect()

    def store_metric(
        self,
        name: str,
        value: float,
        risk_level: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store a single metric value.

        Args:
            name: Metric name (e.g., 'sharpe_ratio', 'daily_pnl')
            value: Metric value
            risk_level: Associated risk level (1-10)
            timestamp: Timestamp (None for now)
            metadata: Optional metadata dict

        Returns:
            True if stored successfully
        """
        if not self._ensure_connected():
            return False

        if timestamp is None:
            timestamp = datetime.now()

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO performance_metrics
                    (time, metric_name, metric_value, risk_level, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    timestamp,
                    name,
                    value,
                    risk_level,
                    json.dumps(metadata) if metadata else None
                ))
            self._conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error storing metric: {e}")
            self._conn.rollback()
            return False

    def store_batch(
        self,
        metrics: Dict[str, float],
        risk_level: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Store multiple metrics at once.

        Args:
            metrics: Dict of metric_name -> value
            risk_level: Associated risk level
            timestamp: Timestamp for all metrics

        Returns:
            True if all stored successfully
        """
        if not self._ensure_connected():
            return False

        if timestamp is None:
            timestamp = datetime.now()

        try:
            with self._conn.cursor() as cur:
                values = [
                    (timestamp, name, value, risk_level, None)
                    for name, value in metrics.items()
                ]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO performance_metrics
                    (time, metric_name, metric_value, risk_level, metadata)
                    VALUES %s
                    """,
                    values
                )
            self._conn.commit()
            self.logger.debug(f"Stored {len(metrics)} metrics")
            return True
        except Exception as e:
            self.logger.error(f"Error storing metric batch: {e}")
            self._conn.rollback()
            return False

    def store_risk_metrics(
        self,
        var_95: Optional[float] = None,
        var_99: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        margin_utilization: Optional[float] = None,
        delta_exposure: Optional[float] = None,
        position_count: Optional[int] = None,
        risk_level: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Store risk metrics snapshot.

        Args:
            var_95: Value at Risk 95%
            var_99: Value at Risk 99%
            current_drawdown: Current drawdown percentage
            max_drawdown: Maximum drawdown percentage
            margin_utilization: Margin utilization percentage
            delta_exposure: Total delta exposure
            position_count: Number of open positions
            risk_level: Current risk level
            timestamp: Timestamp (None for now)

        Returns:
            True if stored successfully
        """
        if not self._ensure_connected():
            return False

        if timestamp is None:
            timestamp = datetime.now()

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO risk_metrics
                    (time, var_95, var_99, current_drawdown, max_drawdown,
                     margin_utilization, delta_exposure, position_count, risk_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    timestamp, var_95, var_99, current_drawdown, max_drawdown,
                    margin_utilization, delta_exposure, position_count, risk_level
                ))
            self._conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error storing risk metrics: {e}")
            self._conn.rollback()
            return False

    def store_daily_performance(
        self,
        date: datetime,
        starting_balance: float,
        ending_balance: float,
        trades_count: int,
        winning_trades: int,
        losing_trades: int,
        max_drawdown_pct: float,
        risk_level: Optional[int] = None
    ) -> bool:
        """
        Store daily performance summary.

        Args:
            date: Trading date
            starting_balance: Balance at start of day
            ending_balance: Balance at end of day
            trades_count: Total trades for the day
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            max_drawdown_pct: Maximum intraday drawdown
            risk_level: Risk level used

        Returns:
            True if stored successfully
        """
        if not self._ensure_connected():
            return False

        daily_pnl = ending_balance - starting_balance
        daily_return_pct = (daily_pnl / starting_balance) * 100 if starting_balance > 0 else 0

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO daily_performance
                    (date, starting_balance, ending_balance, daily_pnl,
                     daily_return_pct, trades_count, winning_trades,
                     losing_trades, max_drawdown_pct, risk_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        starting_balance = EXCLUDED.starting_balance,
                        ending_balance = EXCLUDED.ending_balance,
                        daily_pnl = EXCLUDED.daily_pnl,
                        daily_return_pct = EXCLUDED.daily_return_pct,
                        trades_count = EXCLUDED.trades_count,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                        risk_level = EXCLUDED.risk_level
                """, (
                    date.date() if isinstance(date, datetime) else date,
                    starting_balance, ending_balance, daily_pnl,
                    daily_return_pct, trades_count, winning_trades,
                    losing_trades, max_drawdown_pct, risk_level
                ))
            self._conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error storing daily performance: {e}")
            self._conn.rollback()
            return False

    def get_history(
        self,
        metric_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve metric history.

        Args:
            metric_name: Name of metric to retrieve
            start: Start datetime (None for no lower bound)
            end: End datetime (None for now)
            limit: Maximum records to return

        Returns:
            List of dicts with time, value, and metadata
        """
        if not self._ensure_connected():
            return []

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=30)

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT time, metric_value, risk_level, metadata
                    FROM performance_metrics
                    WHERE metric_name = %s
                      AND time >= %s
                      AND time <= %s
                    ORDER BY time DESC
                    LIMIT %s
                """, (metric_name, start, end, limit))

                results = []
                for row in cur.fetchall():
                    results.append({
                        'time': row[0],
                        'value': row[1],
                        'risk_level': row[2],
                        'metadata': row[3]
                    })

                return results

        except Exception as e:
            self.logger.error(f"Error retrieving metric history: {e}")
            return []

    def get_risk_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve risk metrics history.

        Args:
            start: Start datetime
            end: End datetime
            limit: Maximum records

        Returns:
            List of risk metric snapshots
        """
        if not self._ensure_connected():
            return []

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=30)

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT time, var_95, var_99, current_drawdown, max_drawdown,
                           margin_utilization, delta_exposure, position_count, risk_level
                    FROM risk_metrics
                    WHERE time >= %s AND time <= %s
                    ORDER BY time DESC
                    LIMIT %s
                """, (start, end, limit))

                results = []
                for row in cur.fetchall():
                    results.append({
                        'time': row[0],
                        'var_95': row[1],
                        'var_99': row[2],
                        'current_drawdown': row[3],
                        'max_drawdown': row[4],
                        'margin_utilization': row[5],
                        'delta_exposure': row[6],
                        'position_count': row[7],
                        'risk_level': row[8]
                    })

                return results

        except Exception as e:
            self.logger.error(f"Error retrieving risk history: {e}")
            return []

    def get_daily_performance(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve daily performance history.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of daily performance records
        """
        if not self._ensure_connected():
            return []

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT date, starting_balance, ending_balance, daily_pnl,
                           daily_return_pct, trades_count, winning_trades,
                           losing_trades, max_drawdown_pct, risk_level
                    FROM daily_performance
                    WHERE date >= %s AND date <= %s
                    ORDER BY date DESC
                """, (
                    start_date.date() if isinstance(start_date, datetime) else start_date,
                    end_date.date() if isinstance(end_date, datetime) else end_date
                ))

                results = []
                for row in cur.fetchall():
                    results.append({
                        'date': row[0],
                        'starting_balance': row[1],
                        'ending_balance': row[2],
                        'daily_pnl': row[3],
                        'daily_return_pct': row[4],
                        'trades_count': row[5],
                        'winning_trades': row[6],
                        'losing_trades': row[7],
                        'max_drawdown_pct': row[8],
                        'risk_level': row[9]
                    })

                return results

        except Exception as e:
            self.logger.error(f"Error retrieving daily performance: {e}")
            return []

    def get_latest_metrics(self, metric_names: List[str]) -> Dict[str, float]:
        """
        Get the latest value for each specified metric.

        Args:
            metric_names: List of metric names to retrieve

        Returns:
            Dict mapping metric name to latest value
        """
        if not self._ensure_connected():
            return {}

        results = {}
        try:
            with self._conn.cursor() as cur:
                for name in metric_names:
                    cur.execute("""
                        SELECT metric_value
                        FROM performance_metrics
                        WHERE metric_name = %s
                        ORDER BY time DESC
                        LIMIT 1
                    """, (name,))
                    row = cur.fetchone()
                    if row:
                        results[name] = row[0]

            return results

        except Exception as e:
            self.logger.error(f"Error retrieving latest metrics: {e}")
            return {}

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._connected = False
            self.logger.info("Closed PostgreSQL connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions
def store_metric(name: str, value: float, **kwargs) -> bool:
    """Quick function to store a single metric."""
    with MetricsStore() as store:
        return store.store_metric(name, value, **kwargs)


def get_metric_history(name: str, days: int = 30) -> List[Dict]:
    """Quick function to get metric history."""
    with MetricsStore() as store:
        start = datetime.now() - timedelta(days=days)
        return store.get_history(name, start=start)
