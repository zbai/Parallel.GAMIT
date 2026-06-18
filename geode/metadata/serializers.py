"""
Project: Geodetic Database Engine (GeoDE)
Date: May 2026
Author: Demian D. Gomez

Serializers for normalizing GNSS station metadata from different sources
(database, IGS log files, station info files) into a common schema.

This module supports the StationMetadataComparator workflow described in DESIGN.md.
Uses StationInfoRecord as the common record type across all sources.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

from ..dbConnection import Cnn
from ..pyDate import Date
from .station_info import StationInfoRecord, StationInfo


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StationMetadataBundle:
    """Complete station metadata from a single source."""
    network_code: str                 # 3-char network code
    station_code: str                 # 4-char station code
    domes_number: Optional[str]
    sessions: List[StationInfoRecord]


# =============================================================================
# Bundle Construction Functions
# =============================================================================

def bundle_from_db(cnn: Cnn, network_code: str, station_code: str) -> StationMetadataBundle:
    """
    Query GeoDE database and return a StationMetadataBundle.

    Args:
        cnn: Database connection object
        network_code: 3-character network code
        station_code: 4-character station code

    Returns:
        StationMetadataBundle with session records from the database
    """

    # Query for DOMES number from stations table
    domes = None
    try:
        station_result = cnn.query_float(
            f'''SELECT "DomesNumber"
                FROM stations
                WHERE "NetworkCode" = '{network_code}'
                  AND "StationCode" = '{station_code}'
                LIMIT 1''',
            as_dict=True
        )
        if station_result and station_result[0].get('DomesNumber'):
            domes = station_result[0]['DomesNumber']
    except Exception:
        pass  # DOMES number is optional

    stninfo = StationInfo(cnn, network_code, station_code, allow_empty=True)

    return StationMetadataBundle(
        network_code=network_code,
        station_code=station_code,
        domes_number=domes,
        sessions=stninfo.records
    )


def bundle_from_file(path: Union[str, Path],
                     network_code: str,
                     station_code: str) -> StationMetadataBundle:
    """
    Parse a metadata file and normalize to StationMetadataBundle.

    Uses StationInfo.parse_station_info() which auto-detects the file format
    (IGS log, station info, or NGL) based on content, not extension.

    Args:
        path: Path to the metadata file (any supported format)
        network_code: 3-character network code
        station_code: 4-character station code

    Returns:
        StationMetadataBundle with session records from the file
    """
    from .station_info import StationInfo

    # Create a temporary StationInfo instance for parsing
    # We pass cnn=None since we only need the parsing functionality
    stn_info = StationInfo(None, network_code, station_code, allow_empty=True)

    # Parse the file (format is auto-detected from content)
    records = stn_info.parse_station_info(str(path))

    # Filter to only the target station (for multi-station files like stninfo)
    station_code_lower = station_code.lower()
    sessions = []
    for record in records:
        # Filter by station code (case-insensitive)
        if record.StationCode and record.StationCode.lower() != station_code_lower:
            continue

        # Handle open-ended sessions
        if record.DateEnd and record.DateEnd.year and record.DateEnd.year >= 2100:
            record.DateEnd = None

        # Set provenance
        record.source = 'file'
        sessions.append(record)

    return StationMetadataBundle(
        network_code=network_code,
        station_code=station_code,
        domes_number=None,
        sessions=sessions
    )


# =============================================================================
# Bundle Comparison Utilities
# =============================================================================

def _bundles_equal(a: StationMetadataBundle, b: StationMetadataBundle) -> bool:
    """
    Return True if both bundles are semantically identical after normalization.

    Dates are compared as ISO strings, floats rounded to 4 decimal places.
    Used as a fast-path guard inside compare() - not a substitute for the
    hash check which runs earlier in the workflow.

    Args:
        a: First bundle to compare
        b: Second bundle to compare

    Returns:
        True if bundles are semantically equal
    """
    def _date_to_str(d: Optional[Date]) -> str:
        """Convert Date to comparable string."""
        if d is None or d.year is None or d.year >= 2099:
            return 'open'
        return d.strftime()

    def _norm_session(s: StationInfoRecord) -> tuple:
        """Normalize session record for comparison."""
        return (
            _date_to_str(s.DateStart),
            _date_to_str(s.DateEnd),
            (s.ReceiverCode or '').strip().upper(),
            (s.ReceiverSerial or '').strip().upper(),
            (s.ReceiverVers or '').strip(),
            (s.ReceiverFirmware or '').strip(),
            (s.AntennaCode or '').strip().upper(),
            (s.RadomeCode or '').strip().upper(),
            (s.AntennaSerial or '').strip().upper(),
            round(s.AntennaHeight or 0.0, 4),
            (s.HeightCode or '').strip().upper(),
            round(s.AntennaNorth or 0.0, 4),
            round(s.AntennaEast or 0.0, 4),
            round(s.AntennaDAZ or 0.0, 4),
        )

    norm_a = sorted(_norm_session(s) for s in a.sessions)
    norm_b = sorted(_norm_session(s) for s in b.sessions)

    return norm_a == norm_b


def serialize_for_claude(db: StationMetadataBundle,
                         external: StationMetadataBundle) -> str:
    """
    Return JSON string with both bundles, ready for the API payload.

    Args:
        db: Bundle from database
        external: Bundle from external file (log or stninfo)

    Returns:
        JSON string suitable for Claude API
    """
    import json

    return json.dumps({
        'network_code': db.network_code,
        'station_code': db.station_code,
        'database_sessions': [s.to_claude_dict() for s in db.sessions],
        'external_sessions': [s.to_claude_dict() for s in external.sessions],
    }, indent=2)

