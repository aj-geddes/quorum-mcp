"""
Orchestration Engine for Quorum-MCP

This module implements the core orchestration engine that coordinates multiple AI
providers to build consensus through multi-round deliberation. It manages parallel
provider execution, handles provider failures gracefully, tracks session state,
and aggregates responses into coherent consensus results.

Key Features:
- Async parallel provider execution
- Multiple operational modes (full_deliberation, quick_consensus, devils_advocate)
- Graceful degradation (continues with available providers)
- Comprehensive error handling
- Cost and timing aggregation
- Session state tracking
- Consensus building algorithms

Operational Modes:
- quick_consensus: Single round, all providers respond independently
- full_deliberation: Multi-round with cross-review and final synthesis
- devils_advocate: One provider takes critical/opposing stance
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from quorum_mcp.providers.base import (
    HealthCheckResult,
    HealthStatus,
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
)
from quorum_mcp.session import Session, SessionManager, SessionStatus

# Configure logging
logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    pass


class InsufficientProvidersError(OrchestratorError):
    """Raised when too few providers are available for consensus."""

    pass


class Orchestrator:
    """
    Orchestration engine for multi-provider consensus building.

    The Orchestrator coordinates multiple AI providers through multi-round
    deliberation to build consensus responses. It handles parallel execution,
    provider failures, session management, and consensus synthesis.

    Attributes:
        providers: List of AI providers to use for consensus
        session_manager: Session manager for state tracking
        min_providers: Minimum providers required (default: 2)
        provider_timeout: Timeout per provider in seconds (default: 60.0)
        max_retries: Maximum retry attempts per provider (default: 1)
        check_health: Whether to check provider health before execution (default: True)
    """

    def __init__(
        self,
        providers: list[Provider],
        session_manager: SessionManager | None = None,
        min_providers: int = 1,
        provider_timeout: float = 60.0,
        max_retries: int = 1,
        check_health: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            providers: List of Provider instances to orchestrate
            session_manager: SessionManager instance for state tracking (optional, created if None)
            min_providers: Minimum providers required for consensus (default: 1)
            provider_timeout: Timeout per provider request in seconds (default: 60.0)
            max_retries: Maximum retry attempts per provider (default: 1)
            check_health: Whether to check provider health before execution (default: True)

        Raises:
            ValueError: If no providers provided
        """
        if len(providers) == 0:
            raise ValueError("At least one provider is required")

        if len(providers) < min_providers:
            raise InsufficientProvidersError(
                f"At least {min_providers} providers required, got {len(providers)}"
            )

        self.providers = providers
        self.session_manager = session_manager or SessionManager()
        self.min_providers = min_providers
        self.provider_timeout = provider_timeout
        self.max_retries = max_retries
        self.check_health = check_health

        logger.info(
            f"Orchestrator initialized with {len(providers)} providers "
            f"(min: {min_providers}, timeout: {provider_timeout}s, health_checks: {check_health})"
        )

    async def execute_quorum(
        self,
        query: str,
        context: str | None = None,
        mode: str = "full_deliberation",
        session_id: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Session:
        """
        Execute a quorum consensus process.

        This is the main entry point for the orchestration engine. It creates
        or retrieves a session, executes the appropriate operational mode, and
        builds consensus from provider responses.

        Args:
            query: User query to submit to providers
            context: Additional context for the query (optional)
            mode: Operational mode (default: "full_deliberation")
                - "quick_consensus": Single round, independent responses
                - "full_deliberation": Multi-round with cross-review
                - "devils_advocate": One provider takes critical stance
            session_id: Existing session ID to resume (optional)
            system_prompt: System-level instructions for providers (optional)
            temperature: Sampling temperature for providers (default: 0.7)
            max_tokens: Maximum tokens per response (default: 4096)

        Returns:
            Session object with complete quorum results

        Raises:
            OrchestratorError: If orchestration fails
            InsufficientProvidersError: If too few providers succeed
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Validate mode
        valid_modes = ["quick_consensus", "full_deliberation", "devils_advocate"]
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode: {mode}. Valid modes: {valid_modes}")

        start_time = time.time()

        # Create or retrieve session
        if session_id:
            try:
                session = await self.session_manager.get_session(session_id)
                logger.info(f"Resuming session {session_id}")
            except KeyError as e:
                raise OrchestratorError(f"Session not found: {session_id}") from e
        else:
            session = await self.session_manager.create_session(query=query, mode=mode)
            logger.info(f"Created new session {session.session_id} (mode: {mode})")

        # Update session status
        await self.session_manager.update_session(
            session.session_id, {"status": SessionStatus.IN_PROGRESS}
        )

        # Filter providers by health if enabled
        providers_to_use = self.providers
        health_results = None

        if self.check_health:
            try:
                providers_to_use, health_results = await self._filter_healthy_providers(self.providers)

                # Store health check results in session metadata
                session.metadata["health_checks"] = {
                    name: {
                        "status": result.status.value,
                        "response_time": result.response_time,
                        "error": result.error,
                        "is_usable": result.is_usable(),
                    }
                    for name, result in health_results.items()
                }

                logger.info(
                    f"Using {len(providers_to_use)}/{len(self.providers)} healthy providers"
                )
            except InsufficientProvidersError:
                # Store failure details and re-raise
                if health_results:
                    session.metadata["health_checks"] = {
                        name: {
                            "status": result.status.value,
                            "response_time": result.response_time,
                            "error": result.error,
                            "is_usable": result.is_usable(),
                        }
                        for name, result in health_results.items()
                    }
                raise

        # Track providers used
        providers_used = [p.get_provider_name() for p in providers_to_use]
        session.metadata["providers_used"] = providers_used

        try:
            # Execute based on mode
            if mode == "quick_consensus":
                await self._execute_quick_consensus(
                    session=session,
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    providers=providers_to_use,
                )
            elif mode == "full_deliberation":
                await self._execute_full_deliberation(
                    session=session,
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    providers=providers_to_use,
                )
            elif mode == "devils_advocate":
                await self._execute_devils_advocate(
                    session=session,
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    providers=providers_to_use,
                )

            # Build consensus from all rounds
            consensus = await self._build_consensus(session.session_id)

            # Calculate total time
            total_time = time.time() - start_time

            # Update session metadata
            session = await self.session_manager.get_session(session.session_id)
            session.metadata["total_time"] = total_time
            session.metadata["end_time"] = datetime.now(timezone.utc).isoformat()

            # Set consensus and mark complete
            session.set_consensus(consensus)
            await self.session_manager.update_session(
                session.session_id,
                {
                    "status": SessionStatus.COMPLETED,
                    "consensus": consensus,
                    "metadata": session.metadata,
                },
            )

            logger.info(
                f"Session {session.session_id} completed in {total_time:.2f}s "
                f"(confidence: {consensus.get('confidence', 0):.2f})"
            )

            return session

        except Exception as e:
            # Mark session as failed
            error_msg = f"Orchestration failed: {str(e)}"
            logger.error(f"Session {session.session_id} failed: {error_msg}", exc_info=True)

            await self.session_manager.update_session(
                session.session_id, {"status": SessionStatus.FAILED, "error": error_msg}
            )

            # For certain errors, return the failed session instead of raising
            # This allows callers to examine the failure details
            if isinstance(e, (InsufficientProvidersError, OrchestratorError)):
                session = await self.session_manager.get_session(session.session_id)
                return session

            # For other errors, re-raise
            raise OrchestratorError(error_msg) from e

    async def _execute_quick_consensus(
        self,
        session: Session,
        query: str,
        context: str | None,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        providers: list[Provider] | None = None,
    ) -> None:
        """
        Execute quick consensus mode (single round, independent responses).

        All providers respond independently to the query in parallel.
        No cross-review or deliberation. Fast but may have lower agreement.

        Args:
            session: Session object to update
            query: User query
            context: Additional context (optional)
            system_prompt: System instructions (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            providers: List of providers to use (default: self.providers)
        """
        if providers is None:
            providers = self.providers
        logger.info(f"Executing quick_consensus for session {session.session_id}")

        # Format prompt for single round
        prompt = self._format_prompt_for_round(
            query=query, context=context, round_num=1, previous_responses=None
        )

        # Run single round with all providers
        round_results = await self._run_round(
            session_id=session.session_id,
            round_num=1,
            providers=providers,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Quick consensus complete: {round_results['successful_count']}/{round_results['total_count']} providers succeeded"
        )

    async def _execute_full_deliberation(
        self,
        session: Session,
        query: str,
        context: str | None,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        providers: list[Provider] | None = None,
    ) -> None:
        """
        Execute full deliberation mode (3 rounds with cross-review).

        Round 1: Independent analysis by all providers
        Round 2: Cross-review (each sees others' responses)
        Round 3: Final synthesis and consensus

        Args:
            session: Session object to update
            query: User query
            context: Additional context (optional)
            system_prompt: System instructions (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            providers: List of providers to use (default: self.providers)
        """
        if providers is None:
            providers = self.providers
        logger.info(f"Executing full_deliberation for session {session.session_id}")

        # Round 1: Independent analysis
        logger.info(f"Session {session.session_id} - Round 1: Independent analysis")
        prompt_round1 = self._format_prompt_for_round(
            query=query, context=context, round_num=1, previous_responses=None
        )

        round1_results = await self._run_round(
            session_id=session.session_id,
            round_num=1,
            providers=providers,
            prompt=prompt_round1,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Round 2: Cross-review
        logger.info(f"Session {session.session_id} - Round 2: Cross-review")
        session = await self.session_manager.get_session(session.session_id)
        prompt_round2 = self._format_prompt_for_round(
            query=query, context=context, round_num=2, previous_responses=session.provider_responses
        )

        round2_results = await self._run_round(
            session_id=session.session_id,
            round_num=2,
            providers=providers,
            prompt=prompt_round2,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Round 3: Final synthesis
        logger.info(f"Session {session.session_id} - Round 3: Final synthesis")
        session = await self.session_manager.get_session(session.session_id)
        prompt_round3 = self._format_prompt_for_round(
            query=query, context=context, round_num=3, previous_responses=session.provider_responses
        )

        round3_results = await self._run_round(
            session_id=session.session_id,
            round_num=3,
            providers=providers,
            prompt=prompt_round3,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Full deliberation complete: Round 1: {round1_results['successful_count']}, "
            f"Round 2: {round2_results['successful_count']}, "
            f"Round 3: {round3_results['successful_count']}"
        )

    async def _execute_devils_advocate(
        self,
        session: Session,
        query: str,
        context: str | None,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        providers: list[Provider] | None = None,
    ) -> None:
        """
        Execute devil's advocate mode (one provider takes critical stance).

        First provider takes critical/opposing stance, others respond normally.
        Useful for challenging assumptions and exploring alternative viewpoints.

        Args:
            session: Session object to update
            query: User query
            context: Additional context (optional)
            system_prompt: System instructions (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            providers: List of providers to use (default: self.providers)
        """
        if providers is None:
            providers = self.providers
        logger.info(f"Executing devils_advocate for session {session.session_id}")

        if len(providers) < 2:
            raise InsufficientProvidersError("Devil's advocate mode requires at least 2 providers")

        # Round 1: Devil's advocate responds first
        devils_advocate = providers[0]
        other_providers = providers[1:]

        logger.info(
            f"Session {session.session_id} - Round 1: Devil's advocate ({devils_advocate.get_provider_name()})"
        )

        devils_prompt = self._format_devils_advocate_prompt(query=query, context=context)

        round1_results = await self._run_round(
            session_id=session.session_id,
            round_num=1,
            providers=[devils_advocate],
            prompt=devils_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Round 2: Other providers respond to devil's advocate
        logger.info(f"Session {session.session_id} - Round 2: Response to critique")
        session = await self.session_manager.get_session(session.session_id)

        response_prompt = self._format_prompt_for_round(
            query=query, context=context, round_num=2, previous_responses=session.provider_responses
        )

        round2_results = await self._run_round(
            session_id=session.session_id,
            round_num=2,
            providers=other_providers,
            prompt=response_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Devil's advocate complete: Round 1: {round1_results['successful_count']}, "
            f"Round 2: {round2_results['successful_count']}"
        )

    async def _run_round(
        self,
        session_id: str,
        round_num: int,
        providers: list[Provider],
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """
        Execute a single round of provider queries in parallel.

        Sends the same prompt to all providers concurrently, handles errors
        gracefully, and aggregates results. Stores responses in session.

        Args:
            session_id: Session identifier
            round_num: Round number (1-based)
            providers: List of providers to query
            prompt: Query prompt for this round
            system_prompt: System instructions (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response

        Returns:
            Dictionary with round results:
                - total_count: Number of providers attempted
                - successful_count: Number of successful responses
                - failed_count: Number of failed responses
                - total_cost: Aggregate cost
                - total_time: Total time for round
                - errors: List of error messages

        Raises:
            InsufficientProvidersError: If too few providers succeed
        """
        start_time = time.time()
        results = []
        errors = []

        logger.info(f"Running round {round_num} with {len(providers)} providers")

        # Create tasks for all providers
        tasks = []
        for provider in providers:
            task = self._query_provider_with_retry(
                provider=provider,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            tasks.append((provider, task))

        # Execute all provider queries in parallel
        responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Process results
        for (provider, _), response in zip(tasks, responses, strict=False):
            provider_name = provider.get_provider_name()

            if isinstance(response, Exception):
                # Provider failed
                error_msg = f"{provider_name}: {str(response)}"
                errors.append(error_msg)
                logger.warning(f"Provider {provider_name} failed in round {round_num}: {response}")

                # Store error in session
                session = await self.session_manager.get_session(session_id)
                session.add_provider_response(
                    provider=provider_name,
                    round_num=round_num,
                    response={"error": str(response), "timestamp": datetime.now(timezone.utc).isoformat()},
                )
                await self.session_manager.update_session(
                    session_id, {"provider_responses": session.provider_responses}
                )
            else:
                # Provider succeeded
                results.append(response)
                logger.info(
                    f"Provider {provider_name} succeeded in round {round_num} "
                    f"(tokens: {response.tokens_input}+{response.tokens_output}, "
                    f"cost: ${response.cost:.4f})"
                )

                # Store response in session
                session = await self.session_manager.get_session(session_id)
                session.add_provider_response(
                    provider=provider_name, round_num=round_num, response=response.to_dict()
                )
                await self.session_manager.update_session(
                    session_id, {"provider_responses": session.provider_responses}
                )

        # Check if we have enough successful responses
        successful_count = len(results)
        if successful_count < self.min_providers:
            raise InsufficientProvidersError(
                f"Only {successful_count}/{len(providers)} providers succeeded, "
                f"minimum {self.min_providers} required"
            )

        # Calculate aggregates
        total_cost = sum(r.cost or 0.0 for r in results)
        total_time = time.time() - start_time

        # Update session metadata
        session = await self.session_manager.get_session(session_id)
        if "rounds" not in session.metadata:
            session.metadata["rounds"] = {}

        session.metadata["rounds"][str(round_num)] = {
            "total_count": len(providers),
            "successful_count": successful_count,
            "failed_count": len(errors),
            "total_cost": total_cost,
            "total_time": total_time,
            "errors": errors,
        }

        await self.session_manager.update_session(session_id, {"metadata": session.metadata})

        logger.info(
            f"Round {round_num} complete: {successful_count}/{len(providers)} providers, "
            f"${total_cost:.4f}, {total_time:.2f}s"
        )

        return {
            "total_count": len(providers),
            "successful_count": successful_count,
            "failed_count": len(errors),
            "total_cost": total_cost,
            "total_time": total_time,
            "errors": errors,
        }

    async def _query_provider_with_retry(
        self,
        provider: Provider,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        """
        Query a provider with retry logic.

        Args:
            provider: Provider to query
            prompt: Query prompt
            system_prompt: System instructions (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response

        Returns:
            ProviderResponse from the provider

        Raises:
            ProviderError: If all retry attempts fail
        """
        request = ProviderRequest(
            query=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.provider_timeout,
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await provider.send_request(request)
                return response
            except ProviderError as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Provider {provider.get_provider_name()} failed "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(1.0 * (2**attempt))  # Exponential backoff
                else:
                    logger.error(
                        f"Provider {provider.get_provider_name()} failed after "
                        f"{self.max_retries + 1} attempts"
                    )

        if last_error:
            raise last_error
        raise ProviderError("Unknown error during provider query")

    async def _filter_healthy_providers(
        self, providers: list[Provider]
    ) -> tuple[list[Provider], dict[str, HealthCheckResult]]:
        """
        Filter providers based on health status.

        Performs health checks on all providers and returns only those that are
        usable (HEALTHY or DEGRADED status). Providers with UNHEALTHY status
        are filtered out.

        Args:
            providers: List of providers to check

        Returns:
            Tuple of (usable_providers, health_results_by_provider)

        Raises:
            InsufficientProvidersError: If no providers are usable after health checks
        """
        logger.info(f"Checking health of {len(providers)} providers...")
        start_time = time.time()

        # Check health of all providers concurrently
        health_checks = [provider.check_health() for provider in providers]
        health_results = await asyncio.gather(*health_checks, return_exceptions=True)

        # Build results map and filter usable providers
        health_map: dict[str, HealthCheckResult] = {}
        usable_providers: list[Provider] = []

        for provider, result in zip(providers, health_results):
            provider_name = provider.get_provider_name()

            # Handle exceptions during health check
            if isinstance(result, Exception):
                logger.error(
                    f"Health check failed for {provider_name}: {result}",
                    exc_info=result,
                )
                # Create unhealthy result for exception
                health_map[provider_name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    response_time=None,
                    error=f"Health check exception: {str(result)}",
                    details={"error_type": "health_check_exception"},
                )
                continue

            # Store health result
            health_map[provider_name] = result

            # Include only usable providers (HEALTHY or DEGRADED)
            if result.is_usable():
                usable_providers.append(provider)
                logger.info(
                    f"Provider {provider_name}: {result.status.value} "
                    f"(response_time: {result.response_time:.2f}s)"
                )
            else:
                logger.warning(
                    f"Provider {provider_name}: {result.status.value} - "
                    f"excluded from execution. Error: {result.error}"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"Health checks complete in {elapsed:.2f}s: "
            f"{len(usable_providers)}/{len(providers)} providers usable"
        )

        # Check if we have enough providers
        if len(usable_providers) < self.min_providers:
            raise InsufficientProvidersError(
                f"Only {len(usable_providers)} providers are healthy/usable, "
                f"but {self.min_providers} required. "
                f"Health status: {[(p, health_map[p].status.value) for p in health_map]}"
            )

        return usable_providers, health_map

    def _format_prompt_for_round(
        self,
        query: str,
        context: str | None,
        round_num: int,
        previous_responses: dict[str, dict[int, Any]] | None,
    ) -> str:
        """
        Format prompt for a specific round based on mode and previous responses.

        Args:
            query: Original user query
            context: Additional context (optional)
            round_num: Current round number (1-based)
            previous_responses: Previous round responses by provider (optional)

        Returns:
            Formatted prompt string for this round
        """
        if round_num == 1:
            # Round 1: Independent analysis
            parts = []
            if context:
                parts.append(f"Context:\n{context}\n")
            parts.append(f"Query:\n{query}\n")
            parts.append(
                "\nPlease provide a thorough, independent analysis of this query. "
                "Consider multiple perspectives and be specific in your reasoning."
            )
            return "\n".join(parts)

        elif round_num == 2:
            # Round 2: Cross-review
            parts = []
            if context:
                parts.append(f"Context:\n{context}\n")
            parts.append(f"Query:\n{query}\n")

            if previous_responses:
                parts.append("\nPrevious responses from other AI providers:\n")
                for provider_name, rounds in previous_responses.items():
                    if 1 in rounds and "content" in rounds[1]:
                        parts.append(f"\n--- {provider_name} ---")
                        parts.append(rounds[1]["content"][:500])  # Limit length
                        parts.append("")

            parts.append(
                "\nNow, having seen other perspectives, please provide your analysis. "
                "Note areas of agreement and disagreement. Refine your position based on "
                "the insights from other providers."
            )
            return "\n".join(parts)

        elif round_num == 3:
            # Round 3: Final synthesis
            parts = []
            if context:
                parts.append(f"Context:\n{context}\n")
            parts.append(f"Query:\n{query}\n")

            if previous_responses:
                parts.append("\nSummary of previous deliberation:\n")
                for provider_name, rounds in previous_responses.items():
                    if 2 in rounds and "content" in rounds[2]:
                        parts.append(f"\n{provider_name} (Round 2): {rounds[2]['content'][:300]}")

            parts.append(
                "\nProvide your final, synthesized response. Focus on areas of consensus "
                "while acknowledging any important minority viewpoints. Be concise and "
                "actionable."
            )
            return "\n".join(parts)

        else:
            # Default prompt
            return query

    def _format_devils_advocate_prompt(self, query: str, context: str | None) -> str:
        """
        Format prompt for devil's advocate mode.

        Args:
            query: Original user query
            context: Additional context (optional)

        Returns:
            Formatted prompt for devil's advocate
        """
        parts = []
        if context:
            parts.append(f"Context:\n{context}\n")
        parts.append(f"Query:\n{query}\n")
        parts.append(
            "\n**You are playing devil's advocate.** Challenge the assumptions in this query. "
            "Present counterarguments, identify potential flaws, and explore alternative "
            "viewpoints. Be critical but constructive."
        )
        return "\n".join(parts)

    async def _build_consensus(self, session_id: str) -> dict[str, Any]:
        """
        Build consensus from all provider responses across all rounds.

        Analyzes responses to identify areas of agreement, disagreements,
        key points, and calculates overall confidence. Produces a synthesized
        consensus response.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary containing consensus result:
                - summary: Synthesized consensus response
                - confidence: Confidence score (0.0-1.0)
                - agreement_areas: List of points where all providers agree
                - disagreement_areas: List of points where providers disagree
                - key_points: List of important points across all responses
                - provider_count: Number of providers that contributed
                - minority_opinions: List of minority viewpoints to preserve
                - recommendations: Weighted recommendations based on consensus
        """
        session = await self.session_manager.get_session(session_id)

        # Extract all successful responses
        all_responses = []
        provider_names = []

        for provider_name, rounds in session.provider_responses.items():
            # Get the latest round response
            latest_round = max(rounds.keys())
            response = rounds[latest_round]

            # Check if response is successful (has content and no error)
            if isinstance(response, dict) and response.get("content") and not response.get("error"):
                all_responses.append(response["content"])
                provider_names.append(provider_name)

        if not all_responses:
            raise OrchestratorError("No successful responses to build consensus from")

        # Calculate basic metrics
        provider_count = len(all_responses)

        # Extract key points (simple keyword extraction)
        key_points = self._extract_key_points(all_responses)

        # Identify agreement areas (points mentioned by multiple providers)
        agreement_areas = self._identify_agreements(all_responses, threshold=0.5)

        # Identify disagreement areas (conflicting points)
        disagreement_areas = self._identify_disagreements(all_responses)

        # Calculate confidence based on agreement level
        confidence = self._calculate_confidence(agreement_areas, disagreement_areas, provider_count)

        # Generate synthesized summary
        summary = self._synthesize_summary(
            responses=all_responses,
            provider_names=provider_names,
            agreement_areas=agreement_areas,
            disagreement_areas=disagreement_areas,
        )

        # Extract minority opinions (points mentioned by single provider)
        minority_opinions = self._extract_minority_opinions(all_responses, provider_names)

        # Generate weighted recommendations
        recommendations = self._generate_recommendations(
            agreement_areas=agreement_areas, confidence=confidence
        )

        # Calculate aggregate costs
        total_cost = 0.0
        total_tokens_input = 0
        total_tokens_output = 0
        provider_costs = {}

        for provider_name, rounds in session.provider_responses.items():
            provider_cost = 0.0
            for _round_num, response in rounds.items():
                if "cost" in response:
                    cost = response.get("cost", 0.0)
                    total_cost += cost
                    provider_cost += cost
                if "tokens_input" in response:
                    total_tokens_input += response.get("tokens_input", 0)
                if "tokens_output" in response:
                    total_tokens_output += response.get("tokens_output", 0)
            provider_costs[provider_name] = provider_cost

        consensus = {
            "summary": summary,
            "confidence": confidence,
            "agreement_areas": agreement_areas,
            "disagreement_areas": disagreement_areas,
            "key_points": key_points,
            "provider_count": provider_count,
            "minority_opinions": minority_opinions,
            "recommendations": recommendations,
            "cost": {
                "total_cost": total_cost,
                "total_tokens_input": total_tokens_input,
                "total_tokens_output": total_tokens_output,
                "avg_cost_per_provider": total_cost / provider_count if provider_count > 0 else 0.0,
                "providers": provider_costs,
            },
        }

        logger.info(
            f"Consensus built for session {session_id}: "
            f"{provider_count} providers, confidence {confidence:.2f}"
        )

        return consensus

    def _extract_key_points(self, responses: list[str]) -> list[str]:
        """
        Extract key points from responses.

        Simple extraction based on sentence analysis. In production, this could
        use more sophisticated NLP techniques.

        Args:
            responses: List of response strings

        Returns:
            List of key points
        """
        key_points = []

        for response in responses:
            # Split into sentences (simple approach)
            sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 20]

            # Take first few sentences as key points (simple heuristic)
            key_points.extend(sentences[:3])

        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for point in key_points:
            if point.lower() not in seen:
                seen.add(point.lower())
                unique_points.append(point)

        return unique_points[:10]  # Limit to top 10

    def _identify_agreements(self, responses: list[str], threshold: float = 0.5) -> list[str]:
        """
        Identify areas of agreement across responses.

        Args:
            responses: List of response strings
            threshold: Fraction of responses that must mention a point (default: 0.5)

        Returns:
            List of agreement areas
        """
        # Simple word-based agreement detection
        # In production, use semantic similarity

        agreements = []
        word_counts = {}

        for response in responses:
            words = set(response.lower().split())
            for word in words:
                if len(word) > 5:  # Only meaningful words
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Find words mentioned by threshold fraction of providers
        min_count = int(len(responses) * threshold)
        common_words = [word for word, count in word_counts.items() if count >= min_count]

        # Create agreement statements
        if common_words:
            agreements.append(
                f"Common themes: {', '.join(sorted(common_words[:10]))}"  # Top 10 common words
            )

        return agreements

    def _identify_disagreements(self, responses: list[str]) -> list[str]:
        """
        Identify areas of disagreement across responses.

        Args:
            responses: List of response strings

        Returns:
            List of disagreement areas
        """
        disagreements = []

        # Simple heuristic: if responses have very different lengths or structures
        lengths = [len(r) for r in responses]
        if max(lengths) > 2 * min(lengths):
            disagreements.append("Response complexity varies significantly across providers")

        # Look for negation patterns
        negation_counts = sum(
            1 for r in responses if any(neg in r.lower() for neg in ["no", "not", "however", "but"])
        )

        if negation_counts > len(responses) / 2:
            disagreements.append("Providers express caveats or contradictions")

        return disagreements

    def _calculate_confidence(
        self, agreement_areas: list[str], disagreement_areas: list[str], provider_count: int
    ) -> float:
        """
        Calculate confidence score based on agreement level.

        Args:
            agreement_areas: List of agreement areas
            disagreement_areas: List of disagreement areas
            provider_count: Number of providers

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence on number of providers
        base_confidence = min(provider_count / 5.0, 0.8)  # Max 0.8 from provider count

        # Adjust based on agreement/disagreement
        agreement_boost = min(len(agreement_areas) * 0.05, 0.15)
        disagreement_penalty = min(len(disagreement_areas) * 0.05, 0.15)

        confidence = base_confidence + agreement_boost - disagreement_penalty

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _synthesize_summary(
        self,
        responses: list[str],
        provider_names: list[str],
        agreement_areas: list[str],
        disagreement_areas: list[str],
    ) -> str:
        """
        Generate synthesized summary from all responses.

        Args:
            responses: List of response strings
            provider_names: List of provider names
            agreement_areas: List of agreement areas
            disagreement_areas: List of disagreement areas

        Returns:
            Synthesized summary string
        """
        parts = []

        parts.append(
            f"Consensus from {len(responses)} AI providers ({', '.join(provider_names)}):\n"
        )

        # Add agreement summary
        if agreement_areas:
            parts.append("\nAreas of Agreement:")
            for area in agreement_areas[:5]:  # Top 5
                parts.append(f"- {area}")

        # Add disagreement summary if any
        if disagreement_areas:
            parts.append("\nAreas of Divergence:")
            for area in disagreement_areas[:3]:  # Top 3
                parts.append(f"- {area}")

        # Add synthesized response (first provider's response as base)
        parts.append("\nSynthesized Response:")
        parts.append(responses[0][:1000])  # Limit length

        return "\n".join(parts)

    def _extract_minority_opinions(
        self, responses: list[str], provider_names: list[str]
    ) -> list[dict[str, str]]:
        """
        Extract minority opinions (unique viewpoints).

        Args:
            responses: List of response strings
            provider_names: List of provider names

        Returns:
            List of minority opinion dictionaries
        """
        minority_opinions = []

        # Simple heuristic: if a response is significantly different in length
        # or content, consider it a minority opinion

        avg_length = sum(len(r) for r in responses) / len(responses)

        for response, provider in zip(responses, provider_names, strict=False):
            if len(response) > avg_length * 1.5 or len(response) < avg_length * 0.5:
                minority_opinions.append(
                    {
                        "provider": provider,
                        "opinion": response[:200] + "...",  # Snippet
                        "note": "Significantly different in detail level",
                    }
                )

        return minority_opinions

    def _generate_recommendations(
        self, agreement_areas: list[str], confidence: float
    ) -> list[dict[str, Any]]:
        """
        Generate weighted recommendations based on consensus.

        Args:
            agreement_areas: List of agreement areas
            confidence: Overall confidence score

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        for i, area in enumerate(agreement_areas[:5]):  # Top 5 agreements
            recommendations.append(
                {
                    "recommendation": area,
                    "strength": confidence * (1.0 - i * 0.1),  # Decay for lower priority
                    "consensus_level": "high" if i < 2 else "medium",
                }
            )

        return recommendations
