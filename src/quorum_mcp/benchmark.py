"""
Performance Benchmarking System for Quorum-MCP

Measures and compares provider performance across multiple dimensions:
- Latency (response time)
- Throughput (tokens per second)
- Cost efficiency (cost per token)
- Reliability (success rate, error types)
- Quality metrics (consensus agreement, confidence scores)

Provides comprehensive benchmarking tools for provider selection and optimization.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel


class BenchmarkMetric(str, Enum):
    """Types of benchmark metrics."""

    LATENCY = "latency"  # Response time in seconds
    THROUGHPUT = "throughput"  # Tokens per second
    COST_EFFICIENCY = "cost_efficiency"  # USD per 1000 tokens
    SUCCESS_RATE = "success_rate"  # Percentage of successful requests
    TOKENS_PER_REQUEST = "tokens_per_request"  # Average tokens per request
    QUALITY_SCORE = "quality_score"  # Subjective quality rating


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    timestamp: datetime
    provider: str
    model: str
    metric: BenchmarkMetric
    value: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProviderPerformance:
    """Performance statistics for a provider."""

    provider: str
    model: str

    # Latency metrics
    avg_latency: float  # Average response time (seconds)
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float

    # Throughput metrics
    avg_throughput: float  # Tokens per second
    max_throughput: float

    # Cost metrics
    avg_cost_per_1k_tokens: float  # USD per 1000 tokens
    total_cost: float

    # Reliability metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float

    # Token usage
    total_tokens: int
    avg_tokens_per_request: float

    # Time period
    period_start: datetime
    period_end: datetime

    # Error breakdown
    error_types: Dict[str, int] = field(default_factory=dict)


class BenchmarkTracker:
    """
    Tracks performance metrics for providers.

    Collects and analyzes performance data to generate comparative benchmarks.
    """

    def __init__(self):
        """Initialize the benchmark tracker."""
        self._results: List[BenchmarkResult] = []
        self._lock = asyncio.Lock()

    async def record_request(
        self,
        provider: str,
        model: str,
        latency: float,
        tokens_input: int,
        tokens_output: int,
        cost: float,
        success: bool,
        error_type: str | None = None,
    ):
        """
        Record a provider request for benchmarking.

        Args:
            provider: Provider name
            model: Model used
            latency: Response time in seconds
            tokens_input: Input tokens
            tokens_output: Output tokens
            cost: Request cost in USD
            success: Whether request succeeded
            error_type: Type of error if failed
        """
        now = datetime.utcnow()
        total_tokens = tokens_input + tokens_output

        async with self._lock:
            # Latency
            self._results.append(BenchmarkResult(
                timestamp=now,
                provider=provider,
                model=model,
                metric=BenchmarkMetric.LATENCY,
                value=latency,
                metadata={"success": success}
            ))

            # Throughput (tokens per second)
            if latency > 0 and total_tokens > 0:
                throughput = total_tokens / latency
                self._results.append(BenchmarkResult(
                    timestamp=now,
                    provider=provider,
                    model=model,
                    metric=BenchmarkMetric.THROUGHPUT,
                    value=throughput,
                    metadata={"tokens": total_tokens}
                ))

            # Cost efficiency (cost per 1000 tokens)
            if total_tokens > 0:
                cost_per_1k = (cost / total_tokens) * 1000
                self._results.append(BenchmarkResult(
                    timestamp=now,
                    provider=provider,
                    model=model,
                    metric=BenchmarkMetric.COST_EFFICIENCY,
                    value=cost_per_1k,
                    metadata={"total_cost": cost, "total_tokens": total_tokens}
                ))

            # Success rate
            self._results.append(BenchmarkResult(
                timestamp=now,
                provider=provider,
                model=model,
                metric=BenchmarkMetric.SUCCESS_RATE,
                value=1.0 if success else 0.0,
                metadata={"error_type": error_type} if error_type else {}
            ))

            # Tokens per request
            self._results.append(BenchmarkResult(
                timestamp=now,
                provider=provider,
                model=model,
                metric=BenchmarkMetric.TOKENS_PER_REQUEST,
                value=float(total_tokens),
                metadata={"input": tokens_input, "output": tokens_output}
            ))

    async def get_provider_performance(
        self,
        provider: str,
        model: str | None = None,
        time_window: timedelta | None = None,
    ) -> ProviderPerformance | None:
        """
        Get performance statistics for a provider.

        Args:
            provider: Provider name
            model: Specific model (None for all models)
            time_window: Time window for analysis (None for all time)

        Returns:
            ProviderPerformance statistics or None if no data
        """
        async with self._lock:
            # Filter results
            cutoff = datetime.utcnow() - time_window if time_window else datetime.min

            results = [
                r for r in self._results
                if r.provider == provider
                and (model is None or r.model == model)
                and r.timestamp >= cutoff
            ]

            if not results:
                return None

            # Calculate latency statistics
            latencies = [r.value for r in results if r.metric == BenchmarkMetric.LATENCY and r.metadata.get("success")]
            if not latencies:
                return None

            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)

            # Throughput statistics
            throughputs = [r.value for r in results if r.metric == BenchmarkMetric.THROUGHPUT]

            # Cost statistics
            cost_results = [r for r in results if r.metric == BenchmarkMetric.COST_EFFICIENCY]
            costs_per_1k = [r.value for r in cost_results]
            total_cost = sum(r.metadata.get("total_cost", 0) for r in cost_results)

            # Success rate
            success_results = [r for r in results if r.metric == BenchmarkMetric.SUCCESS_RATE]
            successful = sum(1 for r in success_results if r.value == 1.0)
            total = len(success_results)

            # Token usage
            token_results = [r for r in results if r.metric == BenchmarkMetric.TOKENS_PER_REQUEST]
            total_tokens = sum(r.value for r in token_results)

            # Error breakdown
            error_types: Dict[str, int] = {}
            for r in success_results:
                if r.value == 0.0 and "error_type" in r.metadata:
                    error_type = r.metadata["error_type"]
                    error_types[error_type] = error_types.get(error_type, 0) + 1

            return ProviderPerformance(
                provider=provider,
                model=model or "all",
                avg_latency=sum(latencies) / len(latencies),
                min_latency=min(latencies),
                max_latency=max(latencies),
                p50_latency=latencies_sorted[n // 2],
                p95_latency=latencies_sorted[int(n * 0.95)] if n > 20 else latencies_sorted[-1],
                p99_latency=latencies_sorted[int(n * 0.99)] if n > 100 else latencies_sorted[-1],
                avg_throughput=sum(throughputs) / len(throughputs) if throughputs else 0.0,
                max_throughput=max(throughputs) if throughputs else 0.0,
                avg_cost_per_1k_tokens=sum(costs_per_1k) / len(costs_per_1k) if costs_per_1k else 0.0,
                total_cost=total_cost,
                total_requests=total,
                successful_requests=successful,
                failed_requests=total - successful,
                success_rate=successful / total if total > 0 else 0.0,
                total_tokens=int(total_tokens),
                avg_tokens_per_request=total_tokens / len(token_results) if token_results else 0.0,
                period_start=min(r.timestamp for r in results),
                period_end=max(r.timestamp for r in results),
                error_types=error_types,
            )

    async def compare_providers(
        self,
        providers: List[str],
        time_window: timedelta | None = None,
    ) -> Dict[str, ProviderPerformance]:
        """
        Compare performance across multiple providers.

        Args:
            providers: List of provider names to compare
            time_window: Time window for analysis

        Returns:
            Dictionary mapping provider names to their performance stats
        """
        results = {}
        for provider in providers:
            perf = await self.get_provider_performance(provider, time_window=time_window)
            if perf:
                results[provider] = perf
        return results

    async def get_leaderboard(
        self,
        metric: BenchmarkMetric,
        time_window: timedelta | None = None,
        limit: int = 10,
    ) -> List[tuple[str, float]]:
        """
        Get provider leaderboard for a specific metric.

        Args:
            metric: Metric to rank by
            time_window: Time window for analysis
            limit: Maximum providers to return

        Returns:
            List of (provider, score) tuples sorted by performance
        """
        async with self._lock:
            cutoff = datetime.utcnow() - time_window if time_window else datetime.min

            results = [
                r for r in self._results
                if r.metric == metric and r.timestamp >= cutoff
            ]

            # Group by provider and calculate averages
            provider_scores: Dict[str, List[float]] = {}
            for r in results:
                key = f"{r.provider}/{r.model}"
                if key not in provider_scores:
                    provider_scores[key] = []
                provider_scores[key].append(r.value)

            # Calculate averages
            leaderboard = [
                (provider, sum(scores) / len(scores))
                for provider, scores in provider_scores.items()
            ]

            # Sort by metric (lower is better for latency/cost, higher for others)
            reverse = metric not in (BenchmarkMetric.LATENCY, BenchmarkMetric.COST_EFFICIENCY)
            leaderboard.sort(key=lambda x: x[1], reverse=reverse)

            return leaderboard[:limit]

    async def get_performance_summary(
        self,
        time_window: timedelta | None = None,
    ) -> Dict:
        """
        Get overall performance summary across all providers.

        Args:
            time_window: Time window for analysis

        Returns:
            Dictionary with summary statistics
        """
        async with self._lock:
            cutoff = datetime.utcnow() - time_window if time_window else datetime.min

            results = [r for r in self._results if r.timestamp >= cutoff]

            if not results:
                return {
                    "total_requests": 0,
                    "providers": [],
                    "time_period": None,
                }

            providers = set(r.provider for r in results)

            # Calculate overall metrics
            latencies = [r.value for r in results if r.metric == BenchmarkMetric.LATENCY]
            success_results = [r for r in results if r.metric == BenchmarkMetric.SUCCESS_RATE]
            cost_results = [r for r in results if r.metric == BenchmarkMetric.COST_EFFICIENCY]

            successful = sum(1 for r in success_results if r.value == 1.0)
            total_requests = len(success_results)
            total_cost = sum(r.metadata.get("total_cost", 0) for r in cost_results)

            return {
                "total_requests": total_requests,
                "successful_requests": successful,
                "failed_requests": total_requests - successful,
                "success_rate": successful / total_requests if total_requests > 0 else 0.0,
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
                "total_cost": total_cost,
                "providers": list(providers),
                "time_period": {
                    "start": min(r.timestamp for r in results).isoformat(),
                    "end": max(r.timestamp for r in results).isoformat(),
                },
            }


# Global benchmark tracker instance
_benchmark_tracker: BenchmarkTracker | None = None


def get_benchmark_tracker() -> BenchmarkTracker:
    """Get the global benchmark tracker instance."""
    global _benchmark_tracker
    if _benchmark_tracker is None:
        _benchmark_tracker = BenchmarkTracker()
    return _benchmark_tracker
