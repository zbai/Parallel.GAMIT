"""
Project: Geodesy Database Engine (GeoDE)
Date: May 2026
Author: Demian D. Gomez

StationMetadataComparator - compares database metadata against external files
using Claude as the reasoning engine.
"""

import logging
import os
import time
from typing import Optional

import anthropic
from anthropic.types import TextBlockParam

from .serializers import (
    StationMetadataBundle,
    _bundles_equal,
    serialize_for_claude,
)
from .prompts import SYSTEM_PROMPT
from .report import ComparisonReport, parse_claude_response, ReportParseError

logger = logging.getLogger(__name__)


class ComparatorError(Exception):
    """Raised when the comparator encounters an unrecoverable error."""
    pass


class StationMetadataComparator:
    """
    Compares station metadata from the database against external files.

    Uses Claude as the reasoning engine to identify discrepancies between
    database sessions and sessions parsed from IGS logs or station info files.
    Implements a three-layer fast-path strategy to minimize API calls.

    API key is read from ANTHROPIC_API_KEY environment variable by default.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-6"):
        """
        Initialize the comparator.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: Claude model to use for comparison.

        Raises:
            ComparatorError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ComparatorError(
                "No Anthropic API key provided. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

        # Pricing per million tokens (as of 2025)
        # https://www.anthropic.com/pricing
        self._pricing = {
            "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
            "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
            "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of an API call based on token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = self._pricing.get(self.model, {"input": 3.00, "output": 15.00})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def compare(self,
                db_bundle: StationMetadataBundle,
                file_bundle: StationMetadataBundle,
                file_source: str = "IGS log") -> ComparisonReport:
        """
        Compare database sessions against external file sessions.

        Fast-path: skip API call if bundles are identical after normalization.
        Note: Audit table filtering is handled upstream in SyncMetadata before
        this method is called.

        Args:
            db_bundle: StationMetadataBundle from database
            file_bundle: StationMetadataBundle from external file
            file_source: Label for the external source ("IGS log" | "stninfo")

        Returns:
            ComparisonReport with findings from Claude, or no_action if fast-path applies

        Raises:
            ComparatorError: If API call fails
            ReportParseError: If Claude's response cannot be parsed
        """
        station_id = f"{db_bundle.network_code}.{db_bundle.station_code}"

        # Fast path: skip API call if bundles are identical
        if _bundles_equal(db_bundle, file_bundle):
            logger.debug(f"{station_id}: Bundles equal, skipping API call")
            return ComparisonReport.no_action(
                db_bundle.network_code,
                db_bundle.station_code
            )

        # Prepare payload and call Claude
        payload = serialize_for_claude(db_bundle, file_bundle)
        logger.info(f"{station_id}: Calling Claude API for comparison")

        # Debug: show payload in readable format
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{station_id}: Payload being sent to Claude:\n{payload}")

        # Use prompt caching for the system prompt to reduce costs
        # The system prompt is static and can be cached across calls
        cached_system = [
            TextBlockParam(
                type="text",
                text=SYSTEM_PROMPT,
                cache_control={"type": "ephemeral"}
            )
        ]

        # Retry configuration
        max_retries = 5           # Max retries for rate limits/server errors
        max_parse_retries = 2     # Max retries for JSON parse failures
        base_delay = 2.0          # Initial delay in seconds
        max_delay = 60.0          # Maximum delay between retries

        user_msg = (
            f"Compare the database sessions and the downloaded {file_source} "
            f"for station {station_id}:\n\n{payload}\n\n"
            f"IMPORTANT: Output ONLY valid JSON. No explanations or thinking."
        )

        report = None
        last_error = None
        parse_failures = 0

        for attempt in range(max_retries):
            try:
                # Use stronger prompt after JSON parse failures
                if attempt > 0 and isinstance(last_error, ReportParseError):
                    user_msg = (
                        f"Compare the database sessions and the downloaded {file_source} "
                        f"for station {station_id}:\n\n{payload}\n\n"
                        f"YOUR RESPONSE MUST BE ONLY A JSON OBJECT. "
                        f"DO NOT include any text, analysis, or explanation. "
                        f"Start with {{ and end with }}. Nothing else."
                    )

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    temperature=0,  # Deterministic output for consistent JSON responses
                    system=cached_system,
                    messages=[{"role": "user", "content": user_msg}]
                )

                # Log usage and cost
                usage = message.usage
                cost = self._calculate_cost(usage.input_tokens, usage.output_tokens)

                # Check for cache metrics
                cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
                cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0

                if cache_read > 0:
                    logger.info(
                        f"{station_id}: API usage - input: {usage.input_tokens} "
                        f"(cached: {cache_read}), output: {usage.output_tokens}, cost: ${cost:.4f}"
                    )
                elif cache_creation > 0:
                    logger.info(
                        f"{station_id}: API usage - input: {usage.input_tokens} "
                        f"(cache created: {cache_creation}), output: {usage.output_tokens}, cost: ${cost:.4f}"
                    )
                else:
                    logger.info(
                        f"{station_id}: API usage - input: {usage.input_tokens}, "
                        f"output: {usage.output_tokens}, cost: ${cost:.4f}"
                    )

                response_text = message.content[0].text
                logger.debug(f"{station_id}: Received response, parsing...")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{station_id}: Claude response:\n{response_text}")

                report = parse_claude_response(response_text)
                break  # Success, exit retry loop

            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"{station_id}: Rate limited, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise ComparatorError(
                        f"Rate limit exceeded for {station_id} after {max_retries} retries: {e}"
                    )
            except anthropic.APIStatusError as e:
                # Handle 5xx errors with retry
                if e.status_code >= 500:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"{station_id}: Server error ({e.status_code}), retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise ComparatorError(
                            f"Server error for {station_id} after {max_retries} retries: {e}"
                        )
                else:
                    raise ComparatorError(f"Anthropic API error for {station_id}: {e}")
            except anthropic.APIError as e:
                raise ComparatorError(f"Anthropic API error for {station_id}: {e}")
            except anthropic.AuthenticationError as e:
                raise ComparatorError(f"Anthropic authentication error: {e}")
            except ReportParseError as e:
                last_error = e
                parse_failures += 1
                if parse_failures < max_parse_retries:
                    logger.warning(
                        f"{station_id}: JSON parse failed, retrying with stronger prompt "
                        f"({parse_failures}/{max_parse_retries})"
                    )
                else:
                    raise  # Re-raise after max parse retries

        if report.needs_attention:
            # Truncate summary for cleaner log output
            summary = (report.summary[:80] + '...') if len(report.summary) > 80 else report.summary
            logger.info(f"{station_id}: {summary}")
        else:
            logger.debug(f"{station_id}: No action required")

        return report
