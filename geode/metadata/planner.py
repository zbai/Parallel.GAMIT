"""
Project: Geodesy Database Engine (GeoDE)
Date: May 2026
Author: Demian D. Gomez

StationInfoPlanner - Uses Claude to plan database operations for applying
findings that cannot be directly applied due to timeline conflicts.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

import anthropic
from anthropic.types import TextBlockParam

from .planner_prompt import PLANNER_SYSTEM_PROMPT
from .report import Finding
from .station_info import (
    StationInfo,
    StationInfoRecord,
    StationInfoException,
    StationInfoOverlapException,
)
from ..pyDate import Date
from ..dbConnection import Cnn

logger = logging.getLogger(__name__)


class PlannerError(Exception):
    """Raised when the planner encounters an unrecoverable error."""

    def __init__(self, message: str, raw_response: Optional[str] = None):
        self.message = message
        self.raw_response = raw_response
        super().__init__(message)


@dataclass
class PlannerOperation:
    """
    A single DB operation in an ordered execution plan.
    Operations must be executed in list order within a single transaction.
    """
    operation: str              # "INSERT" | "UPDATE" | "DELETE"
    target: Optional[Dict]      # {"NetworkCode": ..., "StationCode": ...,
                                #  "DateStart": "YYYY-MM-DD HH:MM:SS"}
                                # None for INSERT (no existing record to target)
    fields: Dict[str, Any]      # field name -> new value
    reason: str                 # human-readable explanation
    finding_hash: int           # hash of the finding this operation belongs to


@dataclass
class PlannerResult:
    """Result from StationInfoPlanner.plan()."""
    operations: List[PlannerOperation]
    conflicts: List[str]        # empty if no conflicts
    summary: str                # one-line human-readable summary


class StationInfoPlanner:
    """
    Given a finding to apply and the current state of the station's DB sessions,
    produces an ordered sequence of DB operations that correctly applies the finding
    without creating gaps or overlaps in the station info timeline.

    Claude is invoked when:
    - Direct SQL apply failed due to a conflict (StationInfoOverlapException)
    - The user has provided plain-language instructions (TUI interactive mode)
    - The finding is REVIEW type (always requires planner — no direct apply)

    Uses temperature=0 for deterministic output.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-6"):
        """
        Initialize the planner.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: Claude model to use.

        Raises:
            PlannerError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise PlannerError(
                "No Anthropic API key provided. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

        # Pricing per million tokens (as of 2025)
        self._pricing = {
            "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
            "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
            "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of an API call based on token usage."""
        pricing = self._pricing.get(self.model, {"input": 3.00, "output": 15.00})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def plan(self,
             finding: Finding,
             all_findings: List[tuple],
             db_sessions: List[StationInfoRecord],
             network_code: str,
             station_code: str,
             user_instructions: Optional[str] = None) -> PlannerResult:
        """
        Produce an ordered sequence of DB operations to apply a single finding.

        Args:
            finding:           The specific finding to apply.
            all_findings:      All findings for this station as (Finding, disposition) tuples.
                               Disposition is None for pending, or "APPLIED", "DISMISSED",
                               "DEFERRED", "NO_ACTION". Provides full context for planning.
            db_sessions:       Current DB sessions for the station.
            network_code:      Network code.
            station_code:      Station code.
            user_instructions: Optional plain-language instructions from user.

        Returns:
            PlannerResult with operations list, conflicts list, and summary.

        Raises:
            PlannerError: If API call fails or response cannot be parsed.
        """
        station_id = f"{network_code}.{station_code}"

        # Build payload for Claude
        payload = {
            "network_code": network_code,
            "station_code": station_code,
            "finding_to_apply": self._finding_to_dict(finding),
            "all_findings": [
                self._finding_to_dict(f, disposition)
                for f, disposition in all_findings
            ],
            "current_db_sessions": [self._session_to_dict(s) for s in db_sessions],
            "user_instructions": user_instructions,
        }

        logger.info(f"{station_id}: Calling planner for {finding.finding_type}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{station_id}: Planner payload:\n{json.dumps(payload, indent=2, default=str)}")

        # Use prompt caching for the system prompt
        cached_system = [
            TextBlockParam(
                type="text",
                text=PLANNER_SYSTEM_PROMPT,
                cache_control={"type": "ephemeral"}
            )
        ]

        # Retry configuration (same as comparator.py)
        max_retries = 5
        max_parse_retries = 2
        base_delay = 2.0
        max_delay = 60.0

        user_msg = json.dumps(payload, default=str)

        result = None
        last_error = None
        parse_failures = 0

        for attempt in range(max_retries):
            try:
                # Use stronger prompt after JSON parse failures
                if attempt > 0 and parse_failures > 0:
                    user_msg = (
                        f"{json.dumps(payload, default=str)}\n\n"
                        f"YOUR RESPONSE MUST BE ONLY A JSON OBJECT. "
                        f"DO NOT include any text, analysis, or explanation. "
                        f"Start with {{ and end with }}. Nothing else."
                    )

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0,
                    system=cached_system,
                    messages=[{"role": "user", "content": user_msg}]
                )

                # Log usage and cost
                usage = message.usage
                cost = self._calculate_cost(usage.input_tokens, usage.output_tokens)

                cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
                cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0

                if cache_read > 0:
                    logger.info(
                        f"{station_id}: Planner API usage - input: {usage.input_tokens} "
                        f"(cached: {cache_read}), output: {usage.output_tokens}, cost: ${cost:.4f}"
                    )
                elif cache_creation > 0:
                    logger.info(
                        f"{station_id}: Planner API usage - input: {usage.input_tokens} "
                        f"(cache created: {cache_creation}), output: {usage.output_tokens}, cost: ${cost:.4f}"
                    )
                else:
                    logger.info(
                        f"{station_id}: Planner API usage - input: {usage.input_tokens}, "
                        f"output: {usage.output_tokens}, cost: ${cost:.4f}"
                    )

                response_text = message.content[0].text
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{station_id}: Planner response:\n{response_text}")

                result = self._parse_response(response_text, finding.finding_type)
                break  # Success

            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"{station_id}: Rate limited, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise PlannerError(
                        f"Rate limit exceeded for {station_id} after {max_retries} retries: {e}"
                    )
            except anthropic.APIStatusError as e:
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
                        raise PlannerError(
                            f"Server error for {station_id} after {max_retries} retries: {e}"
                        )
                else:
                    raise PlannerError(f"Anthropic API error for {station_id}: {e}")
            except anthropic.APIError as e:
                raise PlannerError(f"Anthropic API error for {station_id}: {e}")
            except anthropic.AuthenticationError as e:
                raise PlannerError(f"Anthropic authentication error: {e}")
            except PlannerError as e:
                last_error = e
                parse_failures += 1
                if parse_failures < max_parse_retries:
                    logger.warning(
                        f"{station_id}: JSON parse failed, retrying with stronger prompt "
                        f"({parse_failures}/{max_parse_retries})"
                    )
                else:
                    raise

        if result.conflicts:
            logger.warning(f"{station_id}: Planner found conflicts: {result.conflicts}")
        else:
            logger.info(f"{station_id}: Planner produced {len(result.operations)} operation(s): {result.summary}")

        return result

    def _finding_to_dict(self, finding: Finding,
                          disposition: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert Finding to JSON-serializable dict.

        Args:
            finding: The Finding object to convert.
            disposition: The audit disposition status:
                - None: pending (not yet processed)
                - "APPLIED": already applied to DB
                - "DISMISSED": deliberately kept as-is
                - "DEFERRED": pending, deferred for later
                - "NO_ACTION": confirmed match, no action needed
        """
        return {
            "finding_type": finding.finding_type,
            "action": finding.action,
            "description": finding.description,
            "affected_fields": finding.affected_fields,
            "db_record": finding.db_record,
            "db_field_values": finding.db_field_values,
            "file_field_values": finding.file_field_values,
            "hash": finding.hash,
            "disposition": disposition,
        }

    def _session_to_dict(self, session: StationInfoRecord) -> Dict[str, Any]:
        """Convert StationInfoRecord to JSON-serializable dict."""
        return session.to_claude_dict()

    def _parse_response(self, response_text: str, finding_type: str) -> PlannerResult:
        """
        Parse Claude's JSON response into PlannerResult.

        Args:
            response_text: Raw response from Claude.
            finding_type: The finding type being processed (for DELETE validation).

        Returns:
            PlannerResult instance.

        Raises:
            PlannerError: If response is not valid JSON or has invalid structure.
        """
        # Strip any markdown fences if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they're fence markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise PlannerError(
                f"Invalid JSON response from planner: {e}",
                raw_response=response_text
            )

        # Validate structure
        if not isinstance(data, dict):
            raise PlannerError(
                "Planner response is not a JSON object",
                raw_response=response_text
            )

        if "operations" not in data:
            raise PlannerError(
                "Planner response missing 'operations' field",
                raw_response=response_text
            )

        operations = []
        for i, op_data in enumerate(data.get("operations", [])):
            # Validate DELETE is only for ORPHAN_SESSION
            if op_data.get("operation") == "DELETE" and finding_type != "ORPHAN_SESSION":
                raise PlannerError(
                    f"DELETE operation not allowed for finding type {finding_type}. "
                    f"DELETE is only permitted for ORPHAN_SESSION findings.",
                    raw_response=response_text
                )

            try:
                op = PlannerOperation(
                    operation=op_data.get("operation", ""),
                    target=op_data.get("target"),
                    fields=op_data.get("fields", {}),
                    reason=op_data.get("reason", ""),
                    finding_hash=op_data.get("finding_hash", 0),
                )
                operations.append(op)
            except Exception as e:
                raise PlannerError(
                    f"Invalid operation at index {i}: {e}",
                    raw_response=response_text
                )

        return PlannerResult(
            operations=operations,
            conflicts=data.get("conflicts", []),
            summary=data.get("summary", ""),
        )


def _find_record_by_date(stn_info: StationInfo,
                         target_date_start: str) -> StationInfoRecord:
    """
    Find a record in stn_info.records matching the given DateStart.

    Args:
        stn_info: StationInfo instance with loaded records.
        target_date_start: ISO format date string "YYYY-MM-DD HH:MM:SS".

    Returns:
        Matching StationInfoRecord.

    Raises:
        PlannerError: If no matching record found.
    """
    target_dt = datetime.strptime(target_date_start, '%Y-%m-%d %H:%M:%S')

    for rec in stn_info.records:
        if rec.DateStart and rec.DateStart.datetime() == target_dt:
            return rec

    raise PlannerError(
        f"Could not find record with DateStart={target_date_start} "
        f"in {stn_info.NetworkCode}.{stn_info.StationCode}"
    )


def _build_record_from_fields(fields: Dict[str, Any],
                              network_code: str,
                              station_code: str) -> StationInfoRecord:
    """
    Build a StationInfoRecord from a fields dict.

    Args:
        fields: Dict of field name -> value (dates as ISO strings).
        network_code: Network code.
        station_code: Station code.

    Returns:
        StationInfoRecord instance.
    """
    fields = fields.copy()
    fields['NetworkCode'] = network_code
    fields['StationCode'] = station_code

    # Convert date strings to Date objects
    if 'DateStart' in fields and fields['DateStart']:
        if isinstance(fields['DateStart'], str):
            dt = datetime.strptime(fields['DateStart'], '%Y-%m-%d %H:%M:%S')
            fields['DateStart'] = Date(datetime=dt)
    if 'DateEnd' in fields and fields['DateEnd']:
        if isinstance(fields['DateEnd'], str):
            dt = datetime.strptime(fields['DateEnd'], '%Y-%m-%d %H:%M:%S')
            fields['DateEnd'] = Date(datetime=dt)

    return StationInfoRecord.from_dict(fields, network_code, station_code)


def execute_plan(cnn: Cnn,
                 operations: List[PlannerOperation],
                 network_code: str,
                 station_code: str,
                 finding_type: str,
                 dry_run: bool = False) -> List[str]:
    """
    Execute an ordered list of PlannerOperations in a single transaction.

    Uses StationInfo methods for proper event logging and record management.

    Args:
        cnn:          GeoDE database connection.
        operations:   Ordered list from StationInfoPlanner.plan().
        network_code: Network code.
        station_code: Station code.
        finding_type: The finding type (for DELETE validation).
        dry_run:      If True, log operations but do not commit.

    Returns:
        List of human-readable strings describing what was done.

    Raises:
        PlannerError: On invalid operation or finding_type mismatch.
        StationInfoException: On database errors (rolls back transaction).
    """
    station_id = f"{network_code}.{station_code}"
    results = []

    if not operations:
        return ["No operations to execute"]

    # Start transaction
    cnn.begin_transac()

    try:
        stn_info = StationInfo(
            cnn, network_code, station_code, allow_empty=True
        )

        for op in operations:
            # Validate DELETE is only for ORPHAN_SESSION
            if op.operation == "DELETE" and finding_type != "ORPHAN_SESSION":
                raise PlannerError(
                    f"DELETE operation not allowed for finding type {finding_type}. "
                    f"DELETE is only permitted for ORPHAN_SESSION findings."
                )

            if op.operation == "INSERT":
                msg = _execute_insert(stn_info, op, network_code, station_code, dry_run)
            elif op.operation == "UPDATE":
                msg = _execute_update(stn_info, op, network_code, station_code, dry_run)
            elif op.operation == "DELETE":
                msg = _execute_delete(stn_info, op, network_code, station_code, dry_run)
            else:
                raise PlannerError(f"Unknown operation type: {op.operation}")

            results.append(msg)
            logger.info(f"{station_id}: {msg}")

        if not dry_run:
            cnn.commit_transac()
            logger.info(f"{station_id}: Plan executed successfully, committed")
        else:
            cnn.rollback_transac()
            logger.info(f"{station_id}: [DRY RUN] Plan executed, rolled back")

    except Exception as e:
        cnn.rollback_transac()
        logger.error(f"{station_id}: Plan execution failed, rolled back: {e}")
        raise

    return results


def _execute_insert(stn_info: StationInfo,
                    op: PlannerOperation,
                    network_code: str,
                    station_code: str,
                    dry_run: bool) -> str:
    """Execute an INSERT operation using StationInfo.insert_station_info()."""
    record = _build_record_from_fields(op.fields, network_code, station_code)

    date_start = op.fields.get('DateStart', '')

    if not dry_run:
        stn_info.insert_station_info(record)

    return f"INSERT {station_code} @ {date_start}"


def _execute_update(stn_info: StationInfo,
                    op: PlannerOperation,
                    network_code: str,
                    station_code: str,
                    dry_run: bool) -> str:
    """Execute an UPDATE operation using StationInfo.update_station_info()."""
    if not op.target:
        raise PlannerError("UPDATE operation missing target")

    target_date_start = op.target.get('DateStart')
    if not target_date_start:
        raise PlannerError("UPDATE target missing DateStart")

    # Find the existing record
    old_record = _find_record_by_date(stn_info, target_date_start)

    # Build new record by merging fields into old record
    new_fields = old_record.to_database_dict()
    for key, value in op.fields.items():
        if key in ('DateStart', 'DateEnd') and value:
            if isinstance(value, str):
                dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                new_fields[key] = dt
            else:
                new_fields[key] = value
        elif key == 'DateEnd' and value is None:
            # Handle DateEnd = null for open-ended sessions
            new_fields[key] = None
        else:
            new_fields[key] = value

    new_record = StationInfoRecord.from_dict(new_fields, network_code, station_code)

    field_summary = ", ".join(f"{k}={v}" for k, v in op.fields.items())

    if not dry_run:
        stn_info.update_station_info(old_record, new_record)

    return f"UPDATE {station_code} @ {target_date_start}: {field_summary}"


def _execute_delete(stn_info: StationInfo,
                    op: PlannerOperation,
                    network_code: str,
                    station_code: str,
                    dry_run: bool) -> str:
    """Execute a DELETE operation using StationInfo.delete_station_info()."""
    if not op.target:
        raise PlannerError("DELETE operation missing target")

    target_date_start = op.target.get('DateStart')
    if not target_date_start:
        raise PlannerError("DELETE target missing DateStart")

    # Find the record to delete
    record = _find_record_by_date(stn_info, target_date_start)

    if not dry_run:
        stn_info.delete_station_info(record)

    return f"DELETE {station_code} @ {target_date_start}"


def try_direct_apply(cnn: Cnn,
                     finding: Finding,
                     network_code: str,
                     station_code: str) -> bool:
    """
    Attempt direct SQL application of a finding.

    Returns True if successful, False if a conflict was detected
    (StationInfoOverlapException). Other exceptions propagate normally.

    Args:
        cnn: Database connection.
        finding: The finding to apply.
        network_code: Network code.
        station_code: Station code.

    Returns:
        True if successful, False if overlap conflict detected.

    Raises:
        StationInfoException: For non-overlap errors (propagates to caller).
    """
    station_id = f"{network_code}.{station_code}"

    # Start transaction
    cnn.begin_transac()

    try:
        stn_info = StationInfo(cnn, network_code, station_code, allow_empty=True)

        if finding.action == "INSERT":
            # Build record from file_field_values
            fields = finding.file_field_values.copy() if finding.file_field_values else {}
            record = _build_record_from_fields(fields, network_code, station_code)
            stn_info.insert_station_info(record)

        elif finding.action == "UPDATE":
            if not finding.db_record:
                raise StationInfoException("UPDATE finding missing db_record")

            target_date_start = finding.db_record.get('DateStart')
            if not target_date_start:
                raise StationInfoException("UPDATE finding db_record missing DateStart")

            # Find the existing record
            old_record = _find_record_by_date(stn_info, target_date_start)

            # Create new record by merging file_field_values into old_record
            new_fields = old_record.to_database_dict()
            if finding.file_field_values:
                for key, value in finding.file_field_values.items():
                    if value is not None:
                        if key in ('DateStart', 'DateEnd') and isinstance(value, str):
                            dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                            new_fields[key] = dt
                        else:
                            new_fields[key] = value

            new_record = StationInfoRecord.from_dict(new_fields, network_code, station_code)
            stn_info.update_station_info(old_record, new_record)

        else:
            # REVIEW findings should never go through try_direct_apply
            raise StationInfoException(
                f"try_direct_apply does not support action type: {finding.action}"
            )

        cnn.commit_transac()
        logger.info(f"{station_id}: Direct apply succeeded for {finding.finding_type}")
        return True

    except StationInfoOverlapException as e:
        cnn.rollback_transac()
        logger.debug(f"{station_id}: Direct apply failed (overlap): {e}")
        return False
    except Exception as e:
        # Rollback on any other error and re-raise
        cnn.rollback_transac()
        logger.error(f"{station_id}: Direct apply failed: {e}")
        raise
