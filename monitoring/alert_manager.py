"""
Alert management module for the TuringTrader monitoring system.
Provides threshold-based alerting, alert persistence, and notification utilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr_trader.config import PostgresConfig


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    DRAWDOWN = "drawdown"
    DAILY_LOSS = "daily_loss"
    POSITION_LIMIT = "position_limit"
    MARGIN = "margin"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class AlertThreshold:
    """Defines a threshold for triggering an alert."""
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    alert_type: AlertType
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 15  # Minimum time between repeated alerts


@dataclass
class Alert:
    """Represents an alert instance."""
    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    alert_type: str = ""
    severity: str = ""
    message: str = ""
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


# Default alert thresholds
DEFAULT_THRESHOLDS = [
    AlertThreshold(
        metric_name="current_drawdown",
        threshold_value=10.0,
        comparison="gt",
        alert_type=AlertType.DRAWDOWN,
        severity=AlertSeverity.CRITICAL,
        message_template="Drawdown exceeded 10%: current drawdown is {value:.2f}%"
    ),
    AlertThreshold(
        metric_name="current_drawdown",
        threshold_value=5.0,
        comparison="gt",
        alert_type=AlertType.DRAWDOWN,
        severity=AlertSeverity.WARNING,
        message_template="Drawdown warning at {value:.2f}%"
    ),
    AlertThreshold(
        metric_name="daily_loss_pct",
        threshold_value=2.0,
        comparison="gt",
        alert_type=AlertType.DAILY_LOSS,
        severity=AlertSeverity.CRITICAL,
        message_template="Daily loss exceeded 2%: current loss is {value:.2f}%"
    ),
    AlertThreshold(
        metric_name="daily_loss_pct",
        threshold_value=1.0,
        comparison="gt",
        alert_type=AlertType.DAILY_LOSS,
        severity=AlertSeverity.WARNING,
        message_template="Daily loss warning at {value:.2f}%"
    ),
    AlertThreshold(
        metric_name="position_utilization",
        threshold_value=80.0,
        comparison="gt",
        alert_type=AlertType.POSITION_LIMIT,
        severity=AlertSeverity.WARNING,
        message_template="Position utilization high: {value:.1f}%"
    ),
    AlertThreshold(
        metric_name="margin_utilization",
        threshold_value=90.0,
        comparison="gt",
        alert_type=AlertType.MARGIN,
        severity=AlertSeverity.CRITICAL,
        message_template="Margin utilization critical: {value:.1f}%"
    ),
    AlertThreshold(
        metric_name="margin_utilization",
        threshold_value=70.0,
        comparison="gt",
        alert_type=AlertType.MARGIN,
        severity=AlertSeverity.WARNING,
        message_template="Margin utilization elevated: {value:.1f}%"
    ),
]


class AlertManager:
    """
    Manages alerts for the trading system.
    Handles threshold checking, alert creation, persistence, and retrieval.
    """

    def __init__(
        self,
        postgres_config: Optional[PostgresConfig] = None,
        thresholds: Optional[List[AlertThreshold]] = None,
        auto_connect: bool = True
    ):
        """
        Initialize the alert manager.

        Args:
            postgres_config: PostgreSQL configuration
            thresholds: Alert thresholds (None for defaults)
            auto_connect: Whether to connect to DB automatically
        """
        self.logger = logging.getLogger(__name__)
        self.config = postgres_config or PostgresConfig()
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

        self._conn = None
        self._connected = False
        self._last_alerts: Dict[str, datetime] = {}  # For cooldown tracking
        self._callbacks: List[Callable[[Alert], None]] = []

        # In-memory alert storage when DB not available
        self._memory_alerts: List[Alert] = []

        if auto_connect and HAS_PSYCOPG2:
            self._connect()

    def _connect(self) -> bool:
        """Establish database connection."""
        if not HAS_PSYCOPG2:
            self.logger.warning("psycopg2 not available - using in-memory alert storage")
            return False

        try:
            self._conn = psycopg2.connect(self.config.get_connection_string())
            self._connected = True
            self.logger.info("Connected to PostgreSQL for alert management")
            return True
        except Exception as e:
            self.logger.warning(f"DB connection failed, using in-memory storage: {e}")
            self._connected = False
            return False

    def _ensure_connected(self) -> bool:
        """Ensure database connection or use memory fallback."""
        if self._connected and self._conn:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return True
            except Exception:
                self._connected = False

        return self._connect()

    def add_threshold(self, threshold: AlertThreshold) -> None:
        """Add a new alert threshold."""
        self.thresholds.append(threshold)
        self.logger.info(f"Added threshold for {threshold.metric_name}")

    def remove_threshold(self, metric_name: str, alert_type: AlertType) -> bool:
        """Remove a threshold by metric and type."""
        initial_count = len(self.thresholds)
        self.thresholds = [
            t for t in self.thresholds
            if not (t.metric_name == metric_name and t.alert_type == alert_type)
        ]
        return len(self.thresholds) < initial_count

    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Register a callback to be called when an alert is created.

        Args:
            callback: Function that takes an Alert instance
        """
        self._callbacks.append(callback)

    def check_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check metrics against all thresholds and create alerts as needed.

        Args:
            metrics: Dict of metric_name -> value

        Returns:
            List of newly created alerts
        """
        new_alerts = []

        for threshold in self.thresholds:
            if threshold.metric_name not in metrics:
                continue

            value = metrics[threshold.metric_name]
            triggered = self._check_threshold(value, threshold)

            if triggered:
                # Check cooldown
                cooldown_key = f"{threshold.metric_name}_{threshold.alert_type.value}"
                last_alert_time = self._last_alerts.get(cooldown_key)

                if last_alert_time:
                    time_since = datetime.now() - last_alert_time
                    if time_since.total_seconds() < threshold.cooldown_minutes * 60:
                        continue  # Still in cooldown

                # Create alert
                alert = self.create_alert(
                    alert_type=threshold.alert_type,
                    severity=threshold.severity,
                    message=threshold.message_template.format(value=value),
                    metric_name=threshold.metric_name,
                    metric_value=value,
                    threshold_value=threshold.threshold_value
                )

                if alert:
                    new_alerts.append(alert)
                    self._last_alerts[cooldown_key] = datetime.now()

                    # Call registered callbacks
                    for callback in self._callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            self.logger.error(f"Alert callback error: {e}")

        return new_alerts

    def _check_threshold(self, value: float, threshold: AlertThreshold) -> bool:
        """Check if a value triggers a threshold."""
        comparisons = {
            'gt': lambda v, t: v > t,
            'lt': lambda v, t: v < t,
            'gte': lambda v, t: v >= t,
            'lte': lambda v, t: v <= t,
            'eq': lambda v, t: abs(v - t) < 0.0001,
        }

        compare_fn = comparisons.get(threshold.comparison, lambda v, t: False)
        return compare_fn(value, threshold.threshold_value)

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None
    ) -> Optional[Alert]:
        """
        Create and persist a new alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            metric_name: Related metric name
            metric_value: Current metric value
            threshold_value: Threshold that was breached

        Returns:
            Created Alert instance or None on failure
        """
        alert = Alert(
            created_at=datetime.now(),
            alert_type=alert_type.value,
            severity=severity.value,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=threshold_value
        )

        if self._ensure_connected():
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO alerts
                        (created_at, alert_type, severity, message,
                         metric_name, metric_value, threshold_value)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        alert.created_at, alert.alert_type, alert.severity,
                        alert.message, alert.metric_name, alert.metric_value,
                        alert.threshold_value
                    ))
                    alert.id = cur.fetchone()[0]
                self._conn.commit()
                self.logger.info(f"Created alert: {alert.alert_type} - {alert.severity}")
                return alert
            except Exception as e:
                self.logger.error(f"Error creating alert in DB: {e}")
                self._conn.rollback()

        # Fallback to memory storage
        alert.id = len(self._memory_alerts) + 1
        self._memory_alerts.append(alert)
        self.logger.info(f"Created alert (in-memory): {alert.alert_type} - {alert.severity}")
        return alert

    def get_active_alerts(self) -> List[Alert]:
        """
        Get all unacknowledged alerts.

        Returns:
            List of active (unacknowledged) alerts
        """
        if self._ensure_connected():
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, created_at, alert_type, severity, message,
                               metric_name, metric_value, threshold_value,
                               acknowledged, acknowledged_at, acknowledged_by,
                               resolved, resolved_at
                        FROM alerts
                        WHERE acknowledged = FALSE
                        ORDER BY created_at DESC
                    """)

                    alerts = []
                    for row in cur.fetchall():
                        alerts.append(Alert(
                            id=row[0],
                            created_at=row[1],
                            alert_type=row[2],
                            severity=row[3],
                            message=row[4],
                            metric_name=row[5],
                            metric_value=row[6],
                            threshold_value=row[7],
                            acknowledged=row[8],
                            acknowledged_at=row[9],
                            acknowledged_by=row[10],
                            resolved=row[11],
                            resolved_at=row[12]
                        ))
                    return alerts
            except Exception as e:
                self.logger.error(f"Error fetching active alerts: {e}")

        # Fallback to memory
        return [a for a in self._memory_alerts if not a.acknowledged]

    def get_recent_alerts(
        self,
        hours: int = 24,
        include_acknowledged: bool = True
    ) -> List[Alert]:
        """
        Get alerts from the last N hours.

        Args:
            hours: Number of hours to look back
            include_acknowledged: Include acknowledged alerts

        Returns:
            List of recent alerts
        """
        since = datetime.now() - timedelta(hours=hours)

        if self._ensure_connected():
            try:
                with self._conn.cursor() as cur:
                    if include_acknowledged:
                        cur.execute("""
                            SELECT id, created_at, alert_type, severity, message,
                                   metric_name, metric_value, threshold_value,
                                   acknowledged, acknowledged_at, acknowledged_by,
                                   resolved, resolved_at
                            FROM alerts
                            WHERE created_at >= %s
                            ORDER BY created_at DESC
                        """, (since,))
                    else:
                        cur.execute("""
                            SELECT id, created_at, alert_type, severity, message,
                                   metric_name, metric_value, threshold_value,
                                   acknowledged, acknowledged_at, acknowledged_by,
                                   resolved, resolved_at
                            FROM alerts
                            WHERE created_at >= %s AND acknowledged = FALSE
                            ORDER BY created_at DESC
                        """, (since,))

                    alerts = []
                    for row in cur.fetchall():
                        alerts.append(Alert(
                            id=row[0],
                            created_at=row[1],
                            alert_type=row[2],
                            severity=row[3],
                            message=row[4],
                            metric_name=row[5],
                            metric_value=row[6],
                            threshold_value=row[7],
                            acknowledged=row[8],
                            acknowledged_at=row[9],
                            acknowledged_by=row[10],
                            resolved=row[11],
                            resolved_at=row[12]
                        ))
                    return alerts
            except Exception as e:
                self.logger.error(f"Error fetching recent alerts: {e}")

        # Fallback to memory
        alerts = [a for a in self._memory_alerts if a.created_at >= since]
        if not include_acknowledged:
            alerts = [a for a in alerts if not a.acknowledged]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def acknowledge_alert(
        self,
        alert_id: int,
        acknowledged_by: str = "system"
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: User/system acknowledging

        Returns:
            True if successful
        """
        if self._ensure_connected():
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        UPDATE alerts
                        SET acknowledged = TRUE,
                            acknowledged_at = %s,
                            acknowledged_by = %s
                        WHERE id = %s
                    """, (datetime.now(), acknowledged_by, alert_id))
                self._conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Error acknowledging alert: {e}")
                self._conn.rollback()
                return False

        # Memory fallback
        for alert in self._memory_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def resolve_alert(self, alert_id: int) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert ID

        Returns:
            True if successful
        """
        if self._ensure_connected():
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        UPDATE alerts
                        SET resolved = TRUE, resolved_at = %s
                        WHERE id = %s
                    """, (datetime.now(), alert_id))
                self._conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Error resolving alert: {e}")
                self._conn.rollback()
                return False

        # Memory fallback
        for alert in self._memory_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                return True
        return False

    def get_alert_counts_by_severity(self, hours: int = 24) -> Dict[str, int]:
        """
        Get count of alerts by severity level.

        Args:
            hours: Hours to look back

        Returns:
            Dict mapping severity to count
        """
        alerts = self.get_recent_alerts(hours=hours, include_acknowledged=True)

        counts = {
            AlertSeverity.INFO.value: 0,
            AlertSeverity.WARNING.value: 0,
            AlertSeverity.CRITICAL.value: 0,
        }

        for alert in alerts:
            if alert.severity in counts:
                counts[alert.severity] += 1

        return counts

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions
def check_and_alert(metrics: Dict[str, float]) -> List[Alert]:
    """Quick function to check metrics and create alerts."""
    with AlertManager() as manager:
        return manager.check_thresholds(metrics)


def get_active_alerts() -> List[Alert]:
    """Quick function to get active alerts."""
    with AlertManager() as manager:
        return manager.get_active_alerts()
