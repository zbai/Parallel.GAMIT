"""IGS Site Log File Parser

Parses IGS site log files (v1.0 and v2.0 formats) to extract station information
including receiver, antenna, and NEU eccentricity data.

Adapted from Geoscience Australia GNSS Analysis Toolbox under Apache License 2.0
https://github.com/GeoscienceAustralia/gnssanalysis/blob/main/gnssanalysis/gn_io/igslog.py

Revisions:
- Patrick D Smith, Jan 2025: Initial adaptation
- Demian Gomez, May 2026: Refactored for clarity, proper NEU handling
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, TYPE_CHECKING

# geode dependecies
from ..pyDate import Date

if TYPE_CHECKING:
    from .station_info import StationInfoRecord


logger = logging.getLogger(__name__)


class LogVersionError(Exception):
    """Log file does not conform to known IGS version standard"""
    pass


class LogParseError(Exception):
    """Error parsing log file content"""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReceiverEntry:
    """Receiver information from an IGS log file"""
    receiver_type: str
    serial_number: str
    firmware_version: str
    date_installed: Date
    date_removed: Date


@dataclass
class AntennaEntry:
    """Antenna information from an IGS log file"""
    antenna_type: str
    radome_type: str
    serial_number: str
    ecc_up: float      # Marker->ARP Up eccentricity in meters
    ecc_north: float   # Marker->ARP North eccentricity in meters
    ecc_east: float    # Marker->ARP East eccentricity in meters
    alignment: float   # Alignment from True N in degrees (antenna DAZ)
    date_installed: Date
    date_removed: Date


# =============================================================================
# Regex Patterns
# =============================================================================

# Version detection patterns
_RE_VERSION_1 = re.compile(rb'site log\)', re.IGNORECASE)
_RE_VERSION_2 = re.compile(rb'site log v2\.0', re.IGNORECASE)

# Site identification pattern (works for both v1 and v2)
_RE_SITE_ID = re.compile(
    rb'(?:Nine\sCharacter\sID|Four\sCharacter\sID|Site\sID)\s+:\s*(\w{4,9})'
    rb'.*?IERS[^\n:]+:\s*(\w{9})?',
    re.IGNORECASE | re.DOTALL
)

# Location pattern - extract city/town
_RE_LOCATION = re.compile(
    rb'City\s+or\s+Town\s+:\s*([^\n(,/?]+)',
    re.IGNORECASE
)

# Receiver block pattern
# Note: Firmware version can contain various characters (e.g., "1.3-1", "CQ00", "AA-004.43")
# so we match any non-newline characters
_RE_RECEIVER = re.compile(
    rb'3\.(\d+)\s+Receiver\s+Type\s+:\s*([\w\s\-\+]+?)\s*[\r\n]'
    rb'.*?Serial\s+Number\s+:\s*(\w*)'
    rb'.*?Firmware\s+Version\s+:\s*([^\r\n]*?)\s*[\r\n]'
    rb'.*?Date\s+Installed\s+:\s*(\d{4}[^\r\n]+)'
    rb'.*?Date\s+Removed\s+:\s*(\d{4}[^\r\n]+|\(CCYY[^\)]+\))?',
    re.IGNORECASE | re.DOTALL
)

# Antenna block pattern
_RE_ANTENNA = re.compile(
    rb'4\.(\d+)\s+Antenna\s+Type\s+:\s*(\S+)\s*(\w*)\s*[\r\n]'
    rb'.*?Serial\s+Number\s+:\s*(\S*)'
    rb'.*?Marker->ARP\s+Up\s+Ecc[^:]*:\s*([\d.\-]+)?'
    rb'.*?Marker->ARP\s+North\s+Ecc[^:]*:\s*([\d.\-]+)?'
    rb'.*?Marker->ARP\s+East\s+Ecc[^:]*:\s*([\d.\-]+)?'
    rb'(?:[^\n]*\n\s*Alignment\s+from\s+True\s+N[^:]*:\s*([\d.\-]+)?)?'  # Optional alignment field
    rb'.*?Antenna\s+Radome\s+Type\s+:\s*(\w*)'
    rb'.*?Date\s+Installed\s+:\s*(\d{4}[^\r\n]+)'
    rb'.*?Date\s+Removed\s+:\s*(\d{4}[^\r\n]+|\(CCYY[^\)]+\))?',
    re.IGNORECASE | re.DOTALL
)

# Far future date for open-ended sessions
_FAR_FUTURE = Date(stninfo='9999 999 00 00 00')


# =============================================================================
# Parsing Functions
# =============================================================================

def determine_log_version(data: bytes) -> str:
    """Determine the IGS log file version from file content.

    Args:
        data: Raw bytes from reading the log file

    Returns:
        Version string: "v1.0" or "v2.0"

    Raises:
        LogVersionError: If version cannot be determined
    """
    first_line = data.lstrip(b'\n').split(b'\n')[0]

    if _RE_VERSION_2.search(first_line):
        return 'v2.0'
    if _RE_VERSION_1.search(first_line):
        return 'v1.0'

    raise LogVersionError(
        f'File does not conform to any known IGS Site Log version. '
        f'First line: {first_line.decode(errors="ignore")}'
    )


def _parse_date(date_str: bytes) -> Date:
    """Parse date string from IGS log format.

    Args:
        date_str: Date bytes in format YYYY-MM-DDThh:mmZ or similar

    Returns:
        Parsed datetime object, or far future date if parsing fails
    """
    if not date_str or date_str.startswith(b'('):
        return _FAR_FUTURE

    date_text = date_str.decode('utf-8', errors='ignore').strip()

    # Try common formats
    formats = [
        '%Y-%m-%dT%H:%MZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d',
    ]

    for fmt in formats:
        try:
            return Date(datetime=datetime.strptime(date_text, fmt))
        except ValueError:
            continue

    logger.warning(f'Could not parse date: {date_text}')
    return _FAR_FUTURE


def _parse_float(value: Optional[bytes], default: float = 0.0) -> float:
    """Safely parse a float value from bytes.

    Args:
        value: Bytes containing a float value
        default: Default value if parsing fails

    Returns:
        Parsed float or default value
    """
    if not value:
        return default
    try:
        return float(value.decode('utf-8', errors='ignore').strip())
    except (ValueError, AttributeError):
        return default


def _extract_site_info(data: bytes, file_path: str) -> Tuple[str, str]:
    """Extract site code and name from log file.

    Args:
        data: Raw log file bytes
        file_path: Path to file (for error messages)

    Returns:
        Tuple of (site_code, site_name)
    """
    # Extract site code
    id_match = _RE_SITE_ID.search(data)
    if id_match:
        site_code = id_match.group(1).decode('utf-8', errors='ignore').upper()[:4]
    else:
        logger.warning(f'Could not extract site ID from {file_path}')
        site_code = 'UNKN'

    # Extract site name (city/town)
    loc_match = _RE_LOCATION.search(data)
    if loc_match:
        site_name = loc_match.group(1).decode('utf-8', errors='ignore').strip()
    else:
        site_name = ''

    return site_code, site_name


def _extract_receivers(data: bytes, file_path: str) -> List[ReceiverEntry]:
    """Extract all receiver entries from log file.

    Args:
        data: Raw log file bytes
        file_path: Path to file (for error messages)

    Returns:
        List of ReceiverEntry objects
    """
    receivers = []

    for match in _RE_RECEIVER.finditer(data):
        try:
            entry = ReceiverEntry(
                receiver_type=match.group(2).decode('utf-8', errors='ignore').strip(),
                serial_number=match.group(3).decode('utf-8', errors='ignore').strip() if match.group(3) else '',
                firmware_version=match.group(4).decode('utf-8', errors='ignore').strip() if match.group(4) else '',
                date_installed=_parse_date(match.group(5)),
                date_removed=_parse_date(match.group(6)) if match.group(6) else _FAR_FUTURE
            )
            receivers.append(entry)
        except Exception as e:
            logger.warning(f'Error parsing receiver block in {file_path}: {e}')

    if not receivers:
        logger.warning(f'No receiver blocks found in {file_path}')

    return receivers


def _extract_antennas(data: bytes, file_path: str) -> List[AntennaEntry]:
    """Extract all antenna entries from log file.

    Args:
        data: Raw log file bytes
        file_path: Path to file (for error messages)

    Returns:
        List of AntennaEntry objects
    """
    antennas = []

    for match in _RE_ANTENNA.finditer(data):
        try:
            # Get radome - could be on antenna type line or separate radome line
            radome_inline = match.group(3).decode('utf-8', errors='ignore').strip() if match.group(3) else ''
            radome_line = match.group(9).decode('utf-8', errors='ignore').strip() if match.group(9) else ''
            radome = radome_line if radome_line and radome_line != 'NONE' else radome_inline
            if not radome:
                radome = 'NONE'

            entry = AntennaEntry(
                antenna_type=match.group(2).decode('utf-8', errors='ignore').strip(),
                radome_type=radome,
                serial_number=match.group(4).decode('utf-8', errors='ignore').strip() if match.group(4) else '',
                ecc_up=_parse_float(match.group(5)),
                ecc_north=_parse_float(match.group(6)),
                ecc_east=_parse_float(match.group(7)),
                alignment=_parse_float(match.group(8)),  # Alignment from True N
                date_installed=_parse_date(match.group(10)),
                date_removed=_parse_date(match.group(11)) if match.group(11) else _FAR_FUTURE
            )
            antennas.append(entry)
        except Exception as e:
            logger.warning(f'Error parsing antenna block in {file_path}: {e}')

    if not antennas:
        logger.warning(f'No antenna blocks found in {file_path}')

    return antennas


def _merge_sessions(site_code: str, site_name: str,
                    receivers: List[ReceiverEntry],
                    antennas: List[AntennaEntry],
                    file_path: str) -> List['StationInfoRecord']:
    """Merge receiver and antenna entries into unified sessions.

    Creates session breaks at every receiver or antenna change.
    When entries have overlapping periods, prefers the most recently installed.

    Args:
        site_code: Station code
        site_name: Station name
        receivers: List of receiver entries
        antennas: List of antenna entries
        file_path: Path to file (for comments)

    Returns:
        List of StationInfoRecord objects covering all time periods
    """
    # Import here to avoid circular imports
    from .station_info import StationInfoRecord

    if not receivers or not antennas:
        return []

    # Collect all break times
    break_times = set()
    for r in receivers:
        break_times.add(r.date_installed)
        break_times.add(r.date_removed)
    for a in antennas:
        break_times.add(a.date_installed)
        break_times.add(a.date_removed)

    break_times = sorted(break_times)
    sessions = []

    for i, start_time in enumerate(break_times[:-1]):
        end_time = break_times[i + 1]

        # Find active receiver at this time (prefer most recently installed if overlapping)
        active_receiver = None
        for r in receivers:
            if r.date_installed <= start_time < r.date_removed:
                if active_receiver is None or r.date_installed > active_receiver.date_installed:
                    active_receiver = r

        # Find active antenna at this time (prefer most recently installed if overlapping)
        active_antenna = None
        for a in antennas:
            if a.date_installed <= start_time < a.date_removed:
                if active_antenna is None or a.date_installed > active_antenna.date_installed:
                    active_antenna = a

        if not active_receiver or not active_antenna:
            continue

        # Determine actual session end (minimum of receiver/antenna end times)
        session_end = min(active_receiver.date_removed, active_antenna.date_removed, end_time)

        session = StationInfoRecord(
            StationCode=site_code.lower(),
            StationName=site_name,
            ReceiverCode=active_receiver.receiver_type,
            ReceiverSerial=active_receiver.serial_number,
            ReceiverVers='',  # Hardware version not available in IGS logs
            ReceiverFirmware=active_receiver.firmware_version,
            AntennaCode=active_antenna.antenna_type,
            RadomeCode=active_antenna.radome_type or 'NONE',
            AntennaSerial=active_antenna.serial_number,
            AntennaHeight=active_antenna.ecc_up,
            AntennaNorth=active_antenna.ecc_north,
            AntennaEast=active_antenna.ecc_east,
            HeightCode='DHARP',
            DateStart=start_time,
            DateEnd=session_end,
            AntennaDAZ=active_antenna.alignment,
            Comments=f'from IGS logfile: {file_path}',
            source='logfile'
        )
        sessions.append(session)

    return sessions


# =============================================================================
# Public API
# =============================================================================

def parse_igs_log_data(data: bytes, file_path: str) -> Optional[List['StationInfoRecord']]:
    """Parse IGS log file data into station sessions.

    Args:
        data: Raw bytes from reading the log file
        file_path: Path to the file (for error messages and comments)

    Returns:
        List of StationInfoRecord objects, or None if parsing fails
    """
    try:
        version = determine_log_version(data)
        logger.debug(f'Parsing {file_path} as IGS log {version}')
    except LogVersionError as e:
        logger.warning(f'{e}, skipping file')
        return None

    site_code, site_name = _extract_site_info(data, file_path)
    receivers = _extract_receivers(data, file_path)
    antennas = _extract_antennas(data, file_path)

    sessions = _merge_sessions(site_code, site_name, receivers, antennas, file_path)

    if not sessions:
        logger.warning(f'No valid sessions extracted from {file_path}')
        return None

    return sessions


def parse_igs_log_file(file_path: str) -> Optional[List['StationInfoRecord']]:
    """Read and parse an IGS log file.

    Args:
        file_path: Path to the IGS log file

    Returns:
        List of StationInfoRecord objects, or None if parsing fails
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
    except IOError as e:
        logger.error(f'Could not read file {file_path}: {e}')
        return None

    return parse_igs_log_data(data, file_path)


# =============================================================================
# IGS Log File Writing
# =============================================================================

@dataclass
class SiteMetadata:
    """Optional site metadata for IGS log file generation.

    Contains information not available in station info records.
    All fields are optional - defaults will be used if not provided.
    """
    # Site identification
    nine_char_id: str = ''          # e.g., 'SRLP00ARG'
    domes_number: str = ''          # e.g., '41532M001'
    monument_description: str = ''
    date_installed: Optional[datetime] = None

    # Location
    city: str = ''
    state_province: str = ''
    country: str = ''
    tectonic_plate: str = ''
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    z_coord: Optional[float] = None
    latitude: Optional[float] = None   # decimal degrees
    longitude: Optional[float] = None  # decimal degrees
    elevation: Optional[float] = None  # meters

    # Form metadata
    prepared_by: str = ''
    agency: str = ''

    # Equipment defaults
    satellite_system: str = 'GPS'
    elevation_cutoff: float = 0.0
    antenna_ref_point: str = 'BAM'


def _format_date_igs(dt: datetime) -> str:
    """Format datetime for IGS log file.

    Args:
        dt: Datetime object

    Returns:
        Formatted string like '2024-07-24T00:00Z' or '(CCYY-MM-DDThh:mmZ)' for open dates
    """
    if dt.year >= 2100:
        return '(CCYY-MM-DDThh:mmZ)'
    return dt.strftime('%Y-%m-%dT%H:%MZ')


def _format_lat_dms(lat: float) -> str:
    """Format latitude in degrees-minutes-seconds for IGS log."""
    sign = '-' if lat < 0 else '+'
    lat = abs(lat)
    deg = int(lat)
    minutes = int((lat - deg) * 60)
    seconds = ((lat - deg) * 60 - minutes) * 60
    return f'{sign}{deg:02d}{minutes:02d}{seconds:05.2f}'


def _format_lon_dms(lon: float) -> str:
    """Format longitude in degrees-minutes-seconds for IGS log."""
    sign = '-' if lon < 0 else '+'
    lon = abs(lon)
    deg = int(lon)
    minutes = int((lon - deg) * 60)
    seconds = ((lon - deg) * 60 - minutes) * 60
    return f'{sign}{deg:03d}{minutes:02d}{seconds:05.2f}'


def _generate_header(station_code: str, metadata: SiteMetadata) -> str:
    """Generate IGS log file header."""
    nine_char = metadata.nine_char_id or f'{station_code.upper()}00XXX'
    return f"""     {nine_char} Site Information Form (site log)
     International GNSS Service
     See Instructions at:
       https://ftp.igs.org/pub/station/general/sitelog_instr.txt
"""


def _generate_form_section(metadata: SiteMetadata) -> str:
    """Generate section 0: Form."""
    prepared_by = metadata.prepared_by or '(full name)'
    date_prepared = datetime.now().strftime('%Y-%m-%d')

    return f"""0.   Form

     Prepared by (full name)  : {prepared_by}
     Date Prepared            : {date_prepared}
     Report Type              : NEW
     If Update:
      Previous Site Log       :
      Modified/Added Sections :
"""


def _generate_site_id_section(station_code: str, station_name: str,
                               metadata: SiteMetadata) -> str:
    """Generate section 1: Site Identification."""
    nine_char = metadata.nine_char_id or f'{station_code.upper()}00XXX'
    domes = metadata.domes_number or '(A9)'
    monument = metadata.monument_description or '(PILLAR/BRASS PLATE/STEEL MAST/etc)'
    date_installed = _format_date_igs(metadata.date_installed) if metadata.date_installed else '(CCYY-MM-DDThh:mmZ)'

    return f"""1.   Site Identification of the GNSS Monument

     Site Name                : {station_name or station_code}
     Nine Character ID        : {nine_char}
     Monument Inscription     :
     IERS DOMES Number        : {domes}
     CDP Number               : (A4)
     Monument Description     : {monument}
       Height of the Monument : (m)
       Monument Foundation    : (STEEL RODS, CONCRETE BLOCK, ROOF, etc)
       Foundation Depth       : (m)
     Marker Description       : (CHISELLED CROSS/DIVOT/BRASS NAIL/etc)
     Date Installed           : {date_installed}
     Geologic Characteristic  : (BEDROCK/CLAY/CONGLOMERATE/GRAVEL/SAND/etc)
       Bedrock Type           : (IGNEOUS/METAMORPHIC/SEDIMENTARY)
       Bedrock Condition      : (FRESH/JOINTED/WEATHERED)
       Fracture Spacing       : (1-10 cm/11-50 cm/51-200 cm/over 200 cm)
       Fault zones nearby     : (YES/NO/Name of the zone)
         Distance/activity    : (multiple lines)
     Additional Information   :
"""


def _generate_location_section(metadata: SiteMetadata) -> str:
    """Generate section 2: Site Location Information."""
    city = metadata.city or '(A30)'
    state = metadata.state_province or '(A30)'
    country = metadata.country or '(A30)'
    plate = metadata.tectonic_plate or '(AFRICAN/ANTARCTIC/AUSTRALIAN/etc)'

    x_coord = f'{metadata.x_coord:.1f} m' if metadata.x_coord is not None else '(m)'
    y_coord = f'{metadata.y_coord:.1f} m' if metadata.y_coord is not None else '(m)'
    z_coord = f'{metadata.z_coord:.1f} m' if metadata.z_coord is not None else '(m)'

    if metadata.latitude is not None:
        lat = _format_lat_dms(metadata.latitude)
    else:
        lat = '(N is +)'

    if metadata.longitude is not None:
        lon = _format_lon_dms(metadata.longitude)
    else:
        lon = '(E is +)'

    elev = f'{metadata.elevation:.1f} m' if metadata.elevation is not None else '(m)'

    return f"""2.   Site Location Information

     City or Town             : {city}
     State or Province        : {state}
     Country                  : {country}
     Tectonic Plate           : {plate}
     Approximate Position (ITRF)
       X coordinate (m)       : {x_coord}
       Y coordinate (m)       : {y_coord}
       Z coordinate (m)       : {z_coord}
       Latitude (N is +)      : {lat}
       Longitude (E is +)     : {lon}
       Elevation (m,ellips.)  : {elev}
     Additional Information   : (multiple lines)
"""


def _generate_receiver_block(index: int, receiver: ReceiverEntry,
                              metadata: SiteMetadata) -> str:
    """Generate a single receiver block (section 3.x)."""
    sat_sys = metadata.satellite_system
    cutoff = f'{metadata.elevation_cutoff:.0f} deg' if metadata.elevation_cutoff else '(deg)'

    return f"""3.{index}  Receiver Type            : {receiver.receiver_type}
     Satellite System         : {sat_sys}
     Serial Number            : {receiver.serial_number}
     Firmware Version         : {receiver.firmware_version}
     Elevation Cutoff Setting : {cutoff}
     Date Installed           : {_format_date_igs(receiver.date_installed)}
     Date Removed             : {_format_date_igs(receiver.date_removed)}
     Temperature Stabiliz.    :
     Additional Information   :
"""


def _generate_receiver_section(receivers: List[ReceiverEntry],
                                metadata: SiteMetadata) -> str:
    """Generate section 3: GNSS Receiver Information."""
    lines = ["3.   GNSS Receiver Information\n"]

    for i, receiver in enumerate(receivers, 1):
        lines.append(_generate_receiver_block(i, receiver, metadata))

    # Add template block
    lines.append("""3.x  Receiver Type            : (A20, from rcvr_ant.tab; see instructions)
     Satellite System         : (GPS+GLO+GAL+BDS+QZSS+SBAS)
     Serial Number            : (A20, but note the first A5 is used in SINEX)
     Firmware Version         : (A11)
     Elevation Cutoff Setting : (deg)
     Date Installed           : (CCYY-MM-DDThh:mmZ)
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Temperature Stabiliz.    : (none or tolerance in degrees C)
     Additional Information   : (multiple lines)
""")

    return '\n'.join(lines)


def _generate_antenna_block(index: int, antenna: AntennaEntry,
                             metadata: SiteMetadata) -> str:
    """Generate a single antenna block (section 4.x)."""
    # Format antenna type with radome (padded to 16 chars + radome)
    ant_type = f'{antenna.antenna_type:<16}{antenna.radome_type}'
    ref_point = metadata.antenna_ref_point

    return f"""4.{index}  Antenna Type             : {ant_type}
     Serial Number            : {antenna.serial_number}
     Antenna Reference Point  : {ref_point}
     Marker->ARP Up Ecc. (m)  : {antenna.ecc_up:.4f} m
     Marker->ARP North Ecc(m) : {antenna.ecc_north:.4f} m
     Marker->ARP East Ecc(m)  : {antenna.ecc_east:.4f} m
     Alignment from True N    : {antenna.alignment:.2f} deg
     Antenna Radome Type      : {antenna.radome_type}
     Radome Serial Number     :
     Antenna Cable Type       :
     Antenna Cable Length     :
     Date Installed           : {_format_date_igs(antenna.date_installed)}
     Date Removed             : {_format_date_igs(antenna.date_removed)}
     Additional Information   :
"""


def _generate_antenna_section(antennas: List[AntennaEntry],
                               metadata: SiteMetadata) -> str:
    """Generate section 4: GNSS Antenna Information."""
    lines = ["4.   GNSS Antenna Information\n"]

    for i, antenna in enumerate(antennas, 1):
        lines.append(_generate_antenna_block(i, antenna, metadata))

    # Add template block
    lines.append("""4.x  Antenna Type             : (A20, from rcvr_ant.tab; see instructions)
     Serial Number            : (A*, but note the first A5 is used in SINEX)
     Antenna Reference Point  : (BPA/BCR/XXX from "antenna.gra"; see instr.)
     Marker->ARP Up Ecc. (m)  : (F8.4)
     Marker->ARP North Ecc(m) : (F8.4)
     Marker->ARP East Ecc(m)  : (F8.4)
     Alignment from True N    : (deg; + is clockwise/east)
     Antenna Radome Type      : (A4 from rcvr_ant.tab; see instructions)
     Radome Serial Number     :
     Antenna Cable Type       : (vendor & type number)
     Antenna Cable Length     : (m)
     Date Installed           : (CCYY-MM-DDThh:mmZ)
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Additional Information   : (multiple lines)
""")

    return '\n'.join(lines)


def _generate_remaining_sections() -> str:
    """Generate sections 5-13 with template placeholders."""
    return """5.   Surveyed Local Ties

5.x  Tied Marker Name         :
     Tied Marker Usage        : (SLR/VLBI/LOCAL CONTROL/FOOTPRINT/etc)
     Tied Marker CDP Number   : (A4)
     Tied Marker DOMES Number : (A9)
     Differential Components from GNSS Marker to the tied monument (ITRS)
       dx (m)                 : (m)
       dy (m)                 : (m)
       dz (m)                 : (m)
     Accuracy (mm)            : (mm)
     Survey method            : (GPS CAMPAIGN/TRILATERATION/TRIANGULATION/etc)
     Date Measured            : (CCYY-MM-DDThh:mmZ)
     Additional Information   : (multiple lines)

6.   Frequency Standard

6.x  Standard Type            : (INTERNAL or EXTERNAL H-MASER/CESIUM/etc)
       Input Frequency        : (if external)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Notes                  : (multiple lines)

7.   Collocation Information

7.x  Instrumentation Type     : (GPS/GLONASS/DORIS/PRARE/SLR/VLBI/TIME/etc)
       Status                 : (PERMANENT/MOBILE)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Notes                  : (multiple lines)

8.   Meteorological Instrumentation

8.1.x  Humidity Sensor Model  :
       Manufacturer           :
       Serial Number          :
       Data Sampling Interval : (sec)
       Accuracy (% rel h)     : (% rel h)
       Aspiration             : (UNASPIRATED/NATURAL/FAN/etc)
       Height Diff to Ant     : (m)
       Calibration date       : (CCYY-MM-DD)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Notes                  : (multiple lines)

8.2.x  Pressure Sensor Model  :
       Manufacturer           :
       Serial Number          :
       Data Sampling Interval : (sec)
       Accuracy               : (hPa)
       Height Diff to Ant     : (m)
       Calibration date       : (CCYY-MM-DD)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Notes                  : (multiple lines)

8.3.x  Temp. Sensor Model     :
       Manufacturer           :
       Serial Number          :
       Data Sampling Interval : (sec)
       Accuracy               : (deg C)
       Aspiration             : (UNASPIRATED/NATURAL/FAN/etc)
       Height Diff to Ant     : (m)
       Calibration date       : (CCYY-MM-DD)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Notes                  : (multiple lines)

8.4.x  Water Vapor Radiometer :
       Manufacturer           :
       Serial Number          :
       Distance to Antenna    : (m)
       Height Diff to Ant     : (m)
       Calibration date       : (CCYY-MM-DD)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Notes                  : (multiple lines)

8.5.x  Other Instrumentation  : (multiple lines)

9.   Local Ongoing Conditions Possibly Affecting Computed Position

9.1.x  Radio Interferences    : (TV/CELL PHONE ANTENNA/RADAR/etc)
       Observed Degradations  : (SN RATIO/DATA GAPS/etc)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Additional Information : (multiple lines)

9.2.x  Multipath Sources      : (METAL ROOF/DOME/VLBI ANTENNA/etc)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Additional Information : (multiple lines)

9.3.x  Signal Obstructions    : (TREES/BUILDINGS/etc)
       Effective Dates        : (CCYY-MM-DD/CCYY-MM-DD)
       Additional Information : (multiple lines)

10.  Local Episodic Effects Possibly Affecting Data Quality

10.x Date                     : (CCYY-MM-DD/CCYY-MM-DD)
     Event                    : (TREE CLEARING/CONSTRUCTION/etc)

11.  On-Site, Point of Contact Agency Information

     Agency                   :
     Preferred Abbreviation   :
     Mailing Address          :
     Primary Contact
       Contact Name           :
       Telephone (primary)    :
       Telephone (secondary)  :
       Fax                    :
       E-mail                 :
     Secondary Contact
       Contact Name           :
       Telephone (primary)    :
       Telephone (secondary)  :
       Fax                    :
       E-mail                 :
     Additional Information   :

12.  Responsible Agency (if different from 11.)

     Agency                   :
     Preferred Abbreviation   :
     Mailing Address          :
     Primary Contact
       Contact Name           :
       Telephone (primary)    :
       Telephone (secondary)  :
       Fax                    :
       E-mail                 :
     Secondary Contact
       Contact Name           :
       Telephone (primary)    :
       Telephone (secondary)  :
       Fax                    :
       E-mail                 :
     Additional Information   :

13.  More Information

     Primary Data Center      :
     Secondary Data Center    :
     URL for More Information :
     Hardcopy on File
       Site Map               : (Y or URL)
       Site Diagram           : (Y or URL)
       Horizon Mask           : (Y or URL)
       Monument Description   : (Y or URL)
       Site Pictures          : (Y or URL)
     Additional Information   : (multiple lines)
     Antenna Graphics with Dimensions

"""


def sessions_to_equipment_lists(sessions: List['StationInfoRecord']
                                 ) -> Tuple[List[ReceiverEntry], List[AntennaEntry]]:
    """Convert station sessions to separate receiver and antenna lists.

    Merges consecutive sessions with the same equipment into single entries.

    Args:
        sessions: List of StationInfoRecord objects

    Returns:
        Tuple of (receivers, antennas) lists
    """
    if not sessions:
        return [], []

    # Sort sessions by start time
    sorted_sessions = sorted(sessions, key=lambda s: s.DateStart.datetime() if s.DateStart else _FAR_FUTURE)

    receivers = []
    antennas = []

    # Track current equipment
    current_receiver = None
    current_antenna = None

    for session in sorted_sessions:
        # Get datetime versions of dates
        session_start = session.DateStart.datetime() if session.DateStart else _FAR_FUTURE
        session_end = session.DateEnd.datetime() if session.DateEnd else _FAR_FUTURE

        # Check if receiver changed
        receiver_key = (session.ReceiverCode, session.ReceiverSerial, session.ReceiverFirmware)
        if current_receiver is None or receiver_key != (
            current_receiver.receiver_type,
            current_receiver.serial_number,
            current_receiver.firmware_version
        ):
            # Close previous receiver
            if current_receiver is not None:
                current_receiver.date_removed = session_start
                receivers.append(current_receiver)

            # Start new receiver
            current_receiver = ReceiverEntry(
                receiver_type=session.ReceiverCode or '',
                serial_number=session.ReceiverSerial or '',
                firmware_version=session.ReceiverFirmware or '',
                date_installed=session_start,
                date_removed=session_end
            )
        else:
            # Extend current receiver
            current_receiver.date_removed = session_end

        # Check if antenna changed
        antenna_key = (session.AntennaCode, session.AntennaSerial, session.RadomeCode,
                       session.AntennaHeight, session.AntennaNorth, session.AntennaEast)
        if current_antenna is None or antenna_key != (
            current_antenna.antenna_type,
            current_antenna.serial_number,
            current_antenna.radome_type,
            current_antenna.ecc_up,
            current_antenna.ecc_north,
            current_antenna.ecc_east
        ):
            # Close previous antenna
            if current_antenna is not None:
                current_antenna.date_removed = session_start
                antennas.append(current_antenna)

            # Start new antenna
            current_antenna = AntennaEntry(
                antenna_type=session.AntennaCode or '',
                radome_type=session.RadomeCode or 'NONE',
                serial_number=session.AntennaSerial or '',
                ecc_up=session.AntennaHeight or 0.0,
                ecc_north=session.AntennaNorth or 0.0,
                ecc_east=session.AntennaEast or 0.0,
                alignment=session.AntennaDAZ or 0.0,
                date_installed=session_start,
                date_removed=session_end
            )
        else:
            # Extend current antenna
            current_antenna.date_removed = session_end

    # Add final entries
    if current_receiver is not None:
        receivers.append(current_receiver)
    if current_antenna is not None:
        antennas.append(current_antenna)

    return receivers, antennas


def generate_igs_log(sessions: List['StationInfoRecord'],
                     metadata: Optional[SiteMetadata] = None) -> str:
    """Generate IGS site log file content from station sessions.

    Args:
        sessions: List of StationInfoRecord objects (typically from one station)
        metadata: Optional site metadata for additional information

    Returns:
        Complete IGS log file content as a string
    """
    if not sessions:
        raise ValueError("No sessions provided")

    # Use default metadata if not provided
    if metadata is None:
        metadata = SiteMetadata()

    # Get station info from first session
    station_code = sessions[0].StationCode or 'UNKN'
    station_name = ''  # StationInfoRecord doesn't store station name

    # Convert sessions to equipment lists
    receivers, antennas = sessions_to_equipment_lists(sessions)

    # Generate all sections
    sections = [
        _generate_header(station_code, metadata),
        _generate_form_section(metadata),
        _generate_site_id_section(station_code, station_name, metadata),
        _generate_location_section(metadata),
        _generate_receiver_section(receivers, metadata),
        _generate_antenna_section(antennas, metadata),
        _generate_remaining_sections()
    ]

    return '\n'.join(sections)


def write_igs_log_file(sessions: List['StationInfoRecord'],
                       file_path: str,
                       metadata: Optional[SiteMetadata] = None) -> None:
    """Write station sessions to an IGS site log file.

    Args:
        sessions: List of StationInfoRecord objects
        file_path: Output file path
        metadata: Optional site metadata
    """
    content = generate_igs_log(sessions, metadata)

    with open(file_path, 'w') as f:
        f.write(content)

    logger.info(f'Wrote IGS log file: {file_path}')


# =============================================================================
# Command-line testing
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python igslog.py <logfile.log>')
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG)

    result = parse_igs_log_file(sys.argv[1])

    if result:
        print(f'\nFound {len(result)} sessions:\n')
        print(f'{"Start":<20} {"End":<20} {"Receiver":<20} {"Antenna":<16} '
              f'{"Up":>8} {"North":>8} {"East":>8} {"DAZ":>6}')
        print('-' * 120)

        for session in result:
            start_str = str(session.DateStart)[:16] if session.DateStart else 'N/A'
            end_str = str(session.DateEnd)[:16] if session.DateEnd else 'open'
            print(f'{start_str:<20} '
                  f'{end_str:<20} '
                  f'{(session.ReceiverCode or "")[:20]:<20} '
                  f'{(session.AntennaCode or "")[:16]:<16} '
                  f'{session.AntennaHeight or 0:>8.4f} {session.AntennaNorth or 0:>8.4f} {session.AntennaEast or 0:>8.4f} '
                  f'{session.AntennaDAZ or 0:>6.1f}')
    else:
        print('Failed to parse log file')
