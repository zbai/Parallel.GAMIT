"""
Project: Geodesy Database Engine (GeoDE)
Date: May 2026
Author: Demian D. Gomez

Dataclasses for structured comparison reports and Claude response parsing.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


class ReportParseError(Exception):
    """Raised when Claude's response cannot be parsed into a valid report."""
    pass


@dataclass
class Finding:
    """
    A single discrepancy finding from the metadata comparison.

    Represents one issue identified when comparing database sessions against
    external file sessions.

    The db_record field contains {"DateStart": "YYYY-MM-DD HH:MM:SS"} to identify
    which database session this finding refers to. For NEW_SESSION findings,
    db_record is None since there is no matching DB session.

    The db_field_values and file_field_values dicts contain structured field
    data for programmatic updates, with only the differing fields included.
    Field names match StationInfoRecord attributes exactly.
    """
    finding_type: str                           # NEW_SESSION, ORPHAN_SESSION, FIRMWARE_UPDATE, etc.
    action: str                                 # INSERT | UPDATE | REVIEW | NO_ACTION
    description: str                            # Human-readable explanation
    affected_fields: List[str]                  # ["receiver"], ["antenna"], ["receiver", "antenna"], ["dates"], ["eccentricity"]
    db_record: Optional[dict]                   # {"DateStart": "YYYY-MM-DD HH:MM:SS"} or None
    db_field_values: Optional[dict] = None      # {"FieldName": "value"} for differing DB fields
    file_field_values: Optional[dict] = None    # {"FieldName": "value"} for recommended values
    hash: int = 0                               # hash value passed and returned by claude


@dataclass
class ComparisonReport:
    """
    Complete comparison report for a single station.

    Contains all findings from comparing database sessions against external
    file sessions, plus metadata about the station and a human-readable summary.
    """
    network_code: str
    station_code: str
    summary: str
    findings: List[Finding] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        """Return True if any finding requires action (not NO_ACTION)."""
        return any(f.action != "NO_ACTION" for f in self.findings)

    def findings_by_action(self, action: str) -> List[Finding]:
        """Return all findings with the specified action type."""
        return [f for f in self.findings if f.action == action]

    def findings_by_type(self, finding_type: str) -> List[Finding]:
        """Return all findings with the specified finding type."""
        return [f for f in self.findings if f.finding_type == finding_type]

    @classmethod
    def no_action(cls, network_code: str, station_code: str) -> "ComparisonReport":
        """
        Create a report indicating no discrepancies were found.

        Used as a fast-path return when bundles are equal or all sessions
        have already been settled in the audit table.
        """
        return cls(
            network_code=network_code,
            station_code=station_code,
            summary="Sessions match. No action required.",
            findings=[]
        )


def parse_claude_response(raw: str) -> ComparisonReport:
    """
    Parse Claude's JSON response into a ComparisonReport.

    Args:
        raw: Raw JSON string from Claude's response

    Returns:
        ComparisonReport with parsed findings

    Raises:
        ReportParseError: If response is not valid JSON or missing required fields
    """
    import re

    # Strip any whitespace and potential markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        # Remove markdown code fences if present
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    # Try to extract JSON object if there's surrounding text
    # Look for the outermost { ... } pattern
    if not raw.startswith("{"):
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            raw = match.group(0)

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ReportParseError(
            f"Invalid JSON in Claude response: {e}\n"
            f"Response was: {raw[:500]}{'...' if len(raw) > 500 else ''}"
        )

    # Validate required top-level fields
    required_fields = ["network_code", "station_code", "summary"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ReportParseError(
            f"Missing required fields in Claude response: {missing}\n"
            f"Response was: {raw[:500]}{'...' if len(raw) > 500 else ''}"
        )

    # Validate field types
    if not isinstance(data.get("network_code"), str):
        raise ReportParseError(
            f"network_code must be a string, got {type(data.get('network_code'))}"
        )
    if not isinstance(data.get("station_code"), str):
        raise ReportParseError(
            f"station_code must be a string, got {type(data.get('station_code'))}"
        )
    if not isinstance(data.get("summary"), str):
        raise ReportParseError(
            f"summary must be a string, got {type(data.get('summary'))}"
        )

    # Parse findings
    findings_data = data.get("findings", [])
    if not isinstance(findings_data, list):
        raise ReportParseError(
            f"findings must be a list, got {type(findings_data)}"
        )

    findings = []
    for i, f in enumerate(findings_data):
        if not isinstance(f, dict):
            raise ReportParseError(
                f"Finding {i} must be a dict, got {type(f)}"
            )

        # Validate required finding fields
        finding_required = ["type", "action", "description"]
        finding_missing = [field for field in finding_required if field not in f]
        if finding_missing:
            raise ReportParseError(
                f"Finding {i} missing required fields: {finding_missing}\n"
                f"Finding was: {f}"
            )

        # Validate action value
        valid_actions = {"INSERT", "UPDATE", "REVIEW", "NO_ACTION"}
        if f["action"] not in valid_actions:
            raise ReportParseError(
                f"Finding {i} has invalid action '{f['action']}'. "
                f"Must be one of: {valid_actions}"
            )

        # Extract structured field values (may be None or dict)
        db_field_values = f.get("db_field_values")
        file_field_values = f.get("file_field_values")

        # Validate field values are dicts or None
        if db_field_values is not None and not isinstance(db_field_values, dict):
            raise ReportParseError(
                f"Finding {i} db_field_values must be a dict or null, "
                f"got {type(db_field_values)}"
            )
        if file_field_values is not None and not isinstance(file_field_values, dict):
            raise ReportParseError(
                f"Finding {i} file_field_values must be a dict or null, "
                f"got {type(file_field_values)}"
            )

        findings.append(Finding(
            finding_type=f["type"],
            action=f["action"],
            description=f["description"],
            affected_fields=f.get("affected_fields", []),
            db_record=f.get("db_record"),  # {"DateStart": "..."} or None
            db_field_values=db_field_values,
            file_field_values=file_field_values,
            hash=int(f.get("hash", '0'))
        ))

    return ComparisonReport(
        network_code=data["network_code"],
        station_code=data["station_code"],
        summary=data["summary"],
        findings=findings,
    )


# =============================================================================
# Timeline Visualization
# =============================================================================

def generate_session_timeline(findings: List[Finding],
                               station_id: str = "") -> str:
    """
    Generate a Rich-formatted table visualization of session date comparisons.

    Shows DB vs File sessions side by side with color coding:
    - Green: dates match
    - Yellow: dates differ (with delta shown)
    - Red: session missing from one source

    Args:
        findings: List of Finding objects for a station
        station_id: Optional station identifier for header

    Returns:
        Rich markup string for display in Textual
    """
    # Filter to date-related findings only
    DATE_TYPES = {
        'DATE_MISMATCH', 'MISSING_SESSION', 'NEW_SESSION', 'ORPHAN_SESSION'
    }
    date_findings = [f for f in findings if f.finding_type in DATE_TYPES]

    if not date_findings:
        return "No date-related findings to display."

    # Build session comparison rows
    # Each row: (db_start, db_end, file_start, file_end, finding_type, notes)
    rows = []
    chain_warnings = []

    for finding in date_findings:
        db_vals = finding.db_field_values or {}
        file_vals = finding.file_field_values or {}

        db_start = db_vals.get('DateStart', '')
        db_end = db_vals.get('DateEnd', '') or '(open)'
        file_start = file_vals.get('DateStart', '')
        file_end = file_vals.get('DateEnd', '') or '(open)'

        # If no db_start from db_field_values, try getting from db_record dict
        if not db_start and finding.db_record:
            if isinstance(finding.db_record, dict):
                # New format: {"DateStart": "YYYY-MM-DD HH:MM:SS"}
                db_start = finding.db_record.get('DateStart', '')
            elif isinstance(finding.db_record, str):
                # Legacy: parse from stninfo-format string
                dates = _extract_dates_from_record(finding.db_record)
                if len(dates) >= 1:
                    db_start = dates[0][2]
                if len(dates) >= 2:
                    db_end = dates[1][2]
                elif len(dates) == 1:
                    db_end = '(open)'

        rows.append({
            'db_start': db_start or '—',
            'db_end': db_end or '(open)',
            'file_start': file_start or '—',
            'file_end': file_end or '(open)',
            'type': finding.finding_type,
            'description': finding.description[:40] if finding.description else '',
        })

        # Check for chain implications
        if finding.finding_type == 'DATE_MISMATCH' and 'DateStart' in (db_vals.keys() | file_vals.keys()):
            if db_start and file_start and db_start != file_start:
                chain_warnings.append(
                    f"Start date change {db_start} → {file_start} may require "
                    f"updating end date of preceding session"
                )

    # Sort rows chronologically by start date (use whichever is available)
    def get_sort_key(row):
        # Try both DB and File start dates
        for date_str in (row['db_start'], row['file_start']):
            if date_str and date_str not in ('—', ''):
                dt = _parse_date_string(date_str)
                if dt:
                    return dt
        # Use a far-future date for unparseable entries (avoid datetime.max overflow)
        return datetime(2999, 12, 31)

    rows.sort(key=get_sort_key)

    # Build Rich markup output
    lines = []
    header = f"[bold]Session Timeline for {station_id}[/bold]" if station_id else "[bold]Session Timeline[/bold]"
    lines.append(header)
    lines.append("")

    # Legend
    lines.append("[dim]Legend:[/dim] [green]●[/green] match  [yellow]●[/yellow] mismatch  [red]●[/red] missing")
    lines.append("")

    # Table header
    lines.append("[bold]     DB Start           DB End            File Start         File End           Status[/bold]")
    lines.append("─" * 95)

    for i, row in enumerate(rows, 1):
        # Compare dates and colorize
        start_match = _dates_equal(row['db_start'], row['file_start'])
        end_match = _dates_equal(row['db_end'], row['file_end'])

        # Format DB dates (17 char width for "YYYY DDD HH MM SS")
        db_start_val = row['db_start'] if row['db_start'] != '—' else '—'
        if row['db_start'] == '—':
            db_start_fmt = f"[red]{db_start_val:17}[/red]"
        elif start_match:
            db_start_fmt = f"[green]{db_start_val:17}[/green]"
        else:
            db_start_fmt = f"[yellow]{db_start_val:17}[/yellow]"

        db_end_val = row['db_end'] if row['db_end'] != '—' else '—'
        if row['db_end'] == '—':
            db_end_fmt = f"[red]{db_end_val:17}[/red]"
        elif end_match:
            db_end_fmt = f"[green]{db_end_val:17}[/green]"
        else:
            db_end_fmt = f"[yellow]{db_end_val:17}[/yellow]"

        # Format File dates
        file_start_val = row['file_start'] if row['file_start'] != '—' else '—'
        if row['file_start'] == '—':
            file_start_fmt = f"[red]{file_start_val:17}[/red]"
        elif start_match:
            file_start_fmt = f"[green]{file_start_val:17}[/green]"
        else:
            file_start_fmt = f"[yellow]{file_start_val:17}[/yellow]"

        file_end_val = row['file_end'] if row['file_end'] != '—' else '—'
        if row['file_end'] == '—':
            file_end_fmt = f"[red]{file_end_val:17}[/red]"
        elif end_match:
            file_end_fmt = f"[green]{file_end_val:17}[/green]"
        else:
            file_end_fmt = f"[yellow]{file_end_val:17}[/yellow]"

        # Status indicator
        if row['type'] == 'NEW_SESSION':
            status = "[red]+ INSERT[/red]"
        elif row['type'] == 'ORPHAN_SESSION':
            status = "[magenta]? ORPHAN[/magenta]"
        elif row['type'] == 'DATE_MISMATCH':
            delta = _compute_date_delta(row['db_start'], row['file_start'])
            if delta:
                status = f"[yellow]≠ {delta}[/yellow]"
            else:
                # Check end dates if start dates match
                delta = _compute_date_delta(row['db_end'], row['file_end'])
                if delta:
                    status = f"[yellow]≠ end {delta}[/yellow]"
                else:
                    status = "[yellow]≠ <1d[/yellow]"
        else:
            status = "[dim]?[/dim]"

        lines.append(f" {i:2}  {db_start_fmt} {db_end_fmt} {file_start_fmt} {file_end_fmt} {status}")

    # Chain warnings
    if chain_warnings:
        lines.append("")
        lines.append("─" * 95)
        lines.append("[bold yellow]⚠ Chain Dependencies:[/bold yellow]")
        for warning in chain_warnings[:3]:
            lines.append(f"  [yellow]•[/yellow] {warning}")

    return "\n".join(lines)


def _dates_equal(d1: str, d2: str) -> bool:
    """Check if two date strings represent the same date."""
    if d1 == d2:
        return True
    if d1 in ('—', '') or d2 in ('—', ''):
        return False
    # Parse and compare
    dt1 = _parse_date_string(d1)
    dt2 = _parse_date_string(d2)
    if dt1 and dt2:
        # Compare to day precision
        return dt1.date() == dt2.date()
    return False


def _compute_date_delta(db_date: str, file_date: str) -> Optional[str]:
    """Compute human-readable delta between two dates."""
    if db_date in ('—', '', '(open)') or file_date in ('—', '', '(open)'):
        return None
    dt1 = _parse_date_string(db_date)
    dt2 = _parse_date_string(file_date)
    if dt1 and dt2:
        delta = dt2 - dt1
        days = delta.days
        # For differences less than a day, show hours
        total_seconds = delta.total_seconds()
        if abs(total_seconds) < 86400:  # Less than 1 day
            hours = int(total_seconds // 3600)
            if hours > 0:
                return f"+{hours}h"
            elif hours < 0:
                return f"{hours}h"
            else:
                return None  # Same time
        elif days > 0:
            return f"+{days}d"
        elif days < 0:
            return f"{days}d"
    return None


def _parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse various date string formats to datetime."""
    if not date_str or date_str == 'open' or date_str == '(open)':
        return None

    # Check for placeholder dates (9999 999 or year >= 2100)
    if date_str.startswith('9999') or date_str.startswith('2100'):
        return None

    # Try ISO format first (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD)
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            pass

    # Try DOY format (YYYY DDD HH MM SS)
    try:
        parts = date_str.strip().split()
        if len(parts) >= 2:
            year = int(parts[0])
            # Skip placeholder years
            if year >= 2100:
                return None
            doy = int(parts[1])
            hour = int(parts[2]) if len(parts) > 2 else 0
            minute = int(parts[3]) if len(parts) > 3 else 0
            second = int(parts[4]) if len(parts) > 4 else 0
            return datetime(year, 1, 1) + timedelta(
                days=doy - 1, hours=hour, minutes=minute, seconds=second
            )
    except (ValueError, IndexError, OverflowError):
        pass

    return None


def _extract_dates_from_record(record_str: str) -> List[tuple]:
    """
    Extract dates from a stninfo-format record string.

    Returns list of (datetime, field_name, raw_string) tuples.
    """
    results = []

    # Pattern for DOY dates: YYYY DDD HH MM SS
    doy_pattern = r'(\d{4})\s+(\d{1,3})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})'
    matches = list(re.finditer(doy_pattern, record_str))

    for i, match in enumerate(matches):
        try:
            year = int(match.group(1))
            doy = int(match.group(2))
            hour = int(match.group(3))
            minute = int(match.group(4))
            second = int(match.group(5))

            # Skip placeholder dates (9999)
            if year >= 2100:
                continue

            dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second)
            raw = match.group(0)
            field = 'DateStart' if i == 0 else 'DateEnd'
            results.append((dt, field, raw))
        except (ValueError, IndexError):
            pass

    return results
