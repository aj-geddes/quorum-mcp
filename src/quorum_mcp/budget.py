"""
Budget Control System for Quorum-MCP

Tracks costs across all providers and enforces budget limits to prevent
unexpected API spending. Provides real-time cost monitoring, alerts, and
automatic budget enforcement.

Features:
- Per-provider and global budget tracking
- Real-time cost accumulation
- Budget alerts and warnings
- Automatic enforcement (optional)
- Cost history and analytics
- Budget period management (daily/weekly/monthly)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


class BudgetPeriod(str, Enum):
    """Budget tracking period."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TOTAL = "total"  # No reset, lifetime budget


class BudgetAlert(BaseModel):
    """Budget alert notification."""

    timestamp: datetime
    provider: str | None
    alert_type: str  # "warning", "limit_reached", "limit_exceeded"
    threshold: float
    current_cost: float
    limit: float
    message: str


@dataclass
class BudgetConfig:
    """
    Budget configuration for a provider or globally.

    Defines spending limits and alert thresholds.
    """

    limit: float  # Maximum allowed cost in USD
    period: BudgetPeriod = BudgetPeriod.DAILY
    warning_threshold: float = 0.80  # Alert at 80% of limit
    enforce: bool = True  # Reject requests if budget exceeded
    provider: str | None = None  # None for global budget


@dataclass
class CostEntry:
    """Single cost entry for tracking."""

    timestamp: datetime
    provider: str
    cost: float
    tokens_input: int
    tokens_output: int
    model: str
    session_id: str | None = None


class BudgetTracker:
    """
    Tracks costs for a single budget (provider or global).

    Monitors spending against configured limits and raises alerts when
    thresholds are crossed.
    """

    def __init__(self, config: BudgetConfig):
        """
        Initialize budget tracker.

        Args:
            config: Budget configuration
        """
        self.config = config
        self.cost_history: List[CostEntry] = []
        self.current_period_start = datetime.utcnow()
        self.alerts: List[BudgetAlert] = []
        self._lock = asyncio.Lock()

    def _get_period_start(self) -> datetime:
        """Get the start of the current budget period."""
        now = datetime.utcnow()

        if self.config.period == BudgetPeriod.HOURLY:
            return now.replace(minute=0, second=0, microsecond=0)
        elif self.config.period == BudgetPeriod.DAILY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif self.config.period == BudgetPeriod.WEEKLY:
            # Monday start
            days_since_monday = now.weekday()
            monday = now - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif self.config.period == BudgetPeriod.MONTHLY:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # TOTAL
            return datetime.min

    async def _reset_if_needed(self):
        """Reset budget if period has elapsed."""
        if self.config.period == BudgetPeriod.TOTAL:
            return  # Never reset

        period_start = self._get_period_start()
        if period_start > self.current_period_start:
            # New period started, reset
            self.current_period_start = period_start
            self.cost_history = [
                entry for entry in self.cost_history
                if entry.timestamp >= period_start
            ]

    async def get_current_cost(self) -> float:
        """
        Get total cost for current period.

        Returns:
            Total cost in USD
        """
        async with self._lock:
            await self._reset_if_needed()
            return sum(entry.cost for entry in self.cost_history)

    async def get_remaining_budget(self) -> float:
        """
        Get remaining budget for current period.

        Returns:
            Remaining budget in USD
        """
        current = await self.get_current_cost()
        return max(0.0, self.config.limit - current)

    async def check_budget(self, estimated_cost: float) -> tuple[bool, str | None]:
        """
        Check if a request would exceed budget.

        Args:
            estimated_cost: Estimated cost of the request

        Returns:
            Tuple of (allowed, reason) where allowed is True if within budget
        """
        async with self._lock:
            await self._reset_if_needed()
            current_cost = sum(entry.cost for entry in self.cost_history)
            projected_cost = current_cost + estimated_cost

            if projected_cost > self.config.limit:
                if self.config.enforce:
                    remaining = max(0.0, self.config.limit - current_cost)
                    return False, (
                        f"Budget limit exceeded. Current: ${current_cost:.4f}, "
                        f"Limit: ${self.config.limit:.4f}, Remaining: ${remaining:.4f}, "
                        f"Estimated cost: ${estimated_cost:.4f}"
                    )

            return True, None

    async def record_cost(self, entry: CostEntry):
        """
        Record a cost entry.

        Args:
            entry: Cost entry to record
        """
        async with self._lock:
            await self._reset_if_needed()
            self.cost_history.append(entry)

            current_cost = sum(e.cost for e in self.cost_history)
            utilization = current_cost / self.config.limit if self.config.limit > 0 else 0

            # Check for alerts
            if utilization >= 1.0 and self.config.enforce:
                alert = BudgetAlert(
                    timestamp=datetime.utcnow(),
                    provider=self.config.provider,
                    alert_type="limit_exceeded",
                    threshold=1.0,
                    current_cost=current_cost,
                    limit=self.config.limit,
                    message=f"Budget limit exceeded: ${current_cost:.4f} / ${self.config.limit:.4f}",
                )
                self.alerts.append(alert)
            elif utilization >= self.config.warning_threshold:
                # Check if we already sent a warning for this period
                recent_warnings = [
                    a for a in self.alerts
                    if a.alert_type == "warning"
                    and a.timestamp >= self.current_period_start
                ]
                if not recent_warnings:
                    alert = BudgetAlert(
                        timestamp=datetime.utcnow(),
                        provider=self.config.provider,
                        alert_type="warning",
                        threshold=self.config.warning_threshold,
                        current_cost=current_cost,
                        limit=self.config.limit,
                        message=f"Budget warning: ${current_cost:.4f} / ${self.config.limit:.4f} ({utilization:.1%})",
                    )
                    self.alerts.append(alert)

    async def get_status(self) -> Dict:
        """
        Get budget status.

        Returns:
            Dictionary with budget information
        """
        async with self._lock:
            await self._reset_if_needed()
            current_cost = sum(entry.cost for entry in self.cost_history)
            remaining = max(0.0, self.config.limit - current_cost)
            utilization = current_cost / self.config.limit if self.config.limit > 0 else 0

            return {
                "provider": self.config.provider or "global",
                "period": self.config.period.value,
                "limit": self.config.limit,
                "current_cost": current_cost,
                "remaining": remaining,
                "utilization": utilization,
                "enforce": self.config.enforce,
                "period_start": self.current_period_start.isoformat(),
                "entries_count": len(self.cost_history),
                "recent_alerts": len([
                    a for a in self.alerts
                    if a.timestamp >= self.current_period_start
                ]),
            }

    async def get_recent_alerts(self, limit: int = 10) -> List[BudgetAlert]:
        """
        Get recent budget alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        async with self._lock:
            return sorted(
                self.alerts,
                key=lambda a: a.timestamp,
                reverse=True
            )[:limit]


class BudgetManager:
    """
    Manages budgets for all providers and global spending.

    Coordinates budget tracking across the system and enforces limits.
    """

    def __init__(self):
        """Initialize the budget manager."""
        self._trackers: Dict[str, BudgetTracker] = {}
        self._global_tracker: BudgetTracker | None = None
        self._lock = asyncio.Lock()

    async def set_budget(self, config: BudgetConfig):
        """
        Set budget for a provider or globally.

        Args:
            config: Budget configuration
        """
        async with self._lock:
            tracker = BudgetTracker(config)

            if config.provider is None:
                self._global_tracker = tracker
            else:
                self._trackers[config.provider] = tracker

    async def record_cost(
        self,
        provider: str,
        cost: float,
        tokens_input: int,
        tokens_output: int,
        model: str,
        session_id: str | None = None,
    ):
        """
        Record a cost entry.

        Args:
            provider: Provider name
            cost: Cost in USD
            tokens_input: Input tokens used
            tokens_output: Output tokens used
            model: Model used
            session_id: Optional session ID
        """
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            provider=provider,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model,
            session_id=session_id,
        )

        # Record in provider-specific budget
        if provider in self._trackers:
            await self._trackers[provider].record_cost(entry)

        # Record in global budget
        if self._global_tracker:
            await self._global_tracker.record_cost(entry)

    async def check_budget(
        self,
        provider: str,
        estimated_cost: float
    ) -> tuple[bool, str | None]:
        """
        Check if a request is within budget.

        Args:
            provider: Provider name
            estimated_cost: Estimated cost of the request

        Returns:
            Tuple of (allowed, reason)
        """
        # Check provider-specific budget
        if provider in self._trackers:
            allowed, reason = await self._trackers[provider].check_budget(estimated_cost)
            if not allowed:
                return False, reason

        # Check global budget
        if self._global_tracker:
            allowed, reason = await self._global_tracker.check_budget(estimated_cost)
            if not allowed:
                return False, reason

        return True, None

    async def get_all_status(self) -> Dict[str, Dict]:
        """
        Get status of all budgets.

        Returns:
            Dictionary mapping budget names to their status
        """
        status = {}

        if self._global_tracker:
            status["global"] = await self._global_tracker.get_status()

        for provider, tracker in self._trackers.items():
            status[provider] = await tracker.get_status()

        return status

    async def get_all_alerts(self) -> List[BudgetAlert]:
        """
        Get all recent budget alerts.

        Returns:
            List of all recent alerts across all budgets
        """
        all_alerts = []

        if self._global_tracker:
            alerts = await self._global_tracker.get_recent_alerts(limit=100)
            all_alerts.extend(alerts)

        for tracker in self._trackers.values():
            alerts = await tracker.get_recent_alerts(limit=100)
            all_alerts.extend(alerts)

        return sorted(all_alerts, key=lambda a: a.timestamp, reverse=True)


# Global budget manager instance
_budget_manager: BudgetManager | None = None


def get_budget_manager() -> BudgetManager:
    """Get the global budget manager instance."""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = BudgetManager()
    return _budget_manager
