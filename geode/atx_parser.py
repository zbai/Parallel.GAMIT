"""
atx_parser.py
=============
Parse an ANTEX 1.4 (.atx) file and insert calibration data into the
PostgreSQL schema defined in atx_calibrations_schema.sql.

Tables written:
    atx_files                  – one row per ATX file (upserted)
    antenna_calibrations       – one row per antenna/radome/serial/atx_file
    antenna_calibration_freq   – one row per calibration × GNSS frequency (PCO)
    antenna_calibration_pcv    – one row per freq × azimuth bin (PCV values)

Usage (standalone):
    from atx_parser import parse_atx_file
    from dbConnection import Cnn

    cnn = Cnn("gnss_data")
    parse_atx_file("/path/to/igs20_2361.atx", cnn)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Data classes that mirror the DB tables
# ---------------------------------------------------------------------------

@dataclass
class AtxHeader:
    """Content of the ANTEX file-level header."""
    pcv_type: str          = 'A'   # 'A' | 'R'
    ref_antenna: str       = ''
    ref_antenna_serial: str = ''


@dataclass
class FreqRms:
    north: Optional[float] = None
    east:  Optional[float] = None
    up:    Optional[float] = None
    pcv:   Optional[list[float]] = None


@dataclass
class FreqData:
    """Data for one START OF FREQUENCY … END OF FREQUENCY block."""
    frequency: str                          # e.g. 'G01'
    north_offset: float = 0.0
    east_offset:  float = 0.0
    up_offset:    float = 0.0
    noazi_pcv:    Optional[list[float]] = None   # PCV for NOAZI row
    azi_pcv:      dict[float, list[float]] = field(default_factory=dict)
    rms:          FreqRms = field(default_factory=FreqRms)
    # RMS azimuth rows
    noazi_rms_pcv: Optional[list[float]] = None
    azi_rms_pcv:   dict[float, list[float]] = field(default_factory=dict)


@dataclass
class AntennaBlock:
    """Everything between START OF ANTENNA … END OF ANTENNA."""
    antenna_code:    str = ''
    radome_code:     str = ''
    serial_no:       str = ''
    method:          str = ''
    calibrated_by:   str = ''
    num_calibrations: int = 0
    cal_date:        str = ''
    dazi:            float = 0.0
    zen1:            float = 0.0
    zen2:            float = 90.0
    dzen:            float = 5.0
    num_frequencies: int = 0
    valid_from:      Optional[datetime] = None
    valid_until:     Optional[datetime] = None
    sinex_code:      str = ''
    comments:        list[str] = field(default_factory=list)
    frequencies:     list[FreqData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

def _record_label(line: str) -> str:
    """Return the record label from columns 61-80 (0-indexed 60:80), stripped."""
    return line[60:80].strip() if len(line) >= 60 else ''


def _parse_datetime(line: str) -> Optional[datetime]:
    """Parse a VALID FROM / VALID UNTIL record (5I6, F13.7)."""
    try:
        parts = line[:60].split()
        year, month, day, hour, minute = (int(p) for p in parts[:5])
        sec = float(parts[5]) if len(parts) > 5 else 0.0
        return datetime(year, month, day, hour, minute, int(sec))
    except (ValueError, IndexError):
        return None


def _parse_floats(text: str) -> list[float]:
    """Return all whitespace-separated floats found in *text*."""
    return [float(v) for v in text.split()]


# ---------------------------------------------------------------------------
# Main ATX parser
# ---------------------------------------------------------------------------

def _parse_atx_file(path: str) -> tuple[AtxHeader, list[AntennaBlock]]:
    """
    Parse an ANTEX 1.4 file.

    Returns
    -------
    header : AtxHeader
    antennas : list[AntennaBlock]
    """
    header   = AtxHeader()
    antennas: list[AntennaBlock] = []

    in_antenna     = False
    in_frequency   = False
    in_freq_rms    = False
    current_ant:   Optional[AntennaBlock] = None
    current_freq:  Optional[FreqData]     = None

    with open(path, 'r', encoding='latin-1') as fh:
        for raw in fh:
            line  = raw.rstrip('\n')
            label = _record_label(line)
            # For labelled records (≤80 cols), data lives in cols 1-60.
            # PCV data lines (NOAZI / azimuth rows) extend beyond 80 cols –
            # use the full stripped line for those sections.
            data  = line[:60]            # data portion (header records)

            # ---- File header ------------------------------------------------
            if label == 'ANTEX VERSION / SYST':
                # pcv_type is at column 20 (0-indexed)
                header.pcv_type = line[20].strip() if len(line) > 20 else 'A'
                continue

            if label == 'PCV TYPE / REFANT':
                header.pcv_type          = data[0].strip() or 'A'
                header.ref_antenna       = data[1:21].strip()
                header.ref_antenna_serial = data[21:41].strip()
                continue

            if label == 'END OF HEADER':
                continue

            # ---- Antenna block boundaries -----------------------------------
            if label == 'START OF ANTENNA':
                current_ant  = AntennaBlock()
                in_antenna   = True
                is_satellite = False   # set True when TYPE / SERIAL NO is parsed
                continue

            if label == 'END OF ANTENNA':
                # Only keep receiver antennas; silently discard satellite blocks.
                if current_ant is not None and not is_satellite:
                    antennas.append(current_ant)
                current_ant   = None
                in_antenna    = False
                is_satellite  = False
                in_frequency  = False
                in_freq_rms   = False
                current_freq  = None
                continue

            if not in_antenna or current_ant is None:
                continue

            # Skip every record inside a satellite antenna block.
            if is_satellite:
                continue

            # ---- Antenna-level records (outside frequency blocks) -----------
            if not in_frequency and not in_freq_rms:

                if label == 'TYPE / SERIAL NO':
                    # Receiver antenna layout: cols 1-16 = antenna code,
                    #                          cols 17-20 = radome code,
                    #                          cols 21-40 = serial number.
                    # Satellite antenna layout: cols 1-20 = full type name
                    #                           (e.g. "BLOCK IIA", "GLONASS-M").
                    # The definitive signal is cols 17-20: receiver antennas
                    # always carry a radome there (minimum "NONE"); satellite
                    # antennas leave those columns blank because their type
                    # name never fills all 20 characters with a trailing
                    # 4-letter radome code.
                    radome_field = data[16:20].strip()
                    if not radome_field:
                        is_satellite = True   # flag the whole block for skipping
                        continue
                    current_ant.antenna_code = data[:16].strip()
                    current_ant.radome_code  = radome_field
                    current_ant.serial_no    = data[20:40].strip()
                    continue

                if label == 'METH / BY / # / DATE':
                    current_ant.method          = data[:20].strip()
                    current_ant.calibrated_by   = data[20:40].strip()
                    num_date                    = data[40:60].strip()
                    # I6,4X,A10 → first token is the count, rest is date
                    tokens = num_date.split()
                    if tokens:
                        try:
                            current_ant.num_calibrations = int(tokens[0])
                        except ValueError:
                            pass
                        if len(tokens) > 1:
                            current_ant.cal_date = tokens[1]
                    continue

                if label == 'DAZI':
                    current_ant.dazi = float(data[2:8].strip() or '0')
                    continue

                if label == 'ZEN1 / ZEN2 / DZEN':
                    parts = data[2:20].split()
                    if len(parts) >= 3:
                        current_ant.zen1 = float(parts[0])
                        current_ant.zen2 = float(parts[1])
                        current_ant.dzen = float(parts[2])
                    continue

                if label == '# OF FREQUENCIES':
                    current_ant.num_frequencies = int(data[:6].strip() or '0')
                    continue

                if label == 'VALID FROM':
                    current_ant.valid_from = _parse_datetime(data)
                    continue

                if label == 'VALID UNTIL':
                    current_ant.valid_until = _parse_datetime(data)
                    continue

                if label == 'SINEX CODE':
                    current_ant.sinex_code = data[:10].strip()
                    continue

                if label == 'COMMENT':
                    current_ant.comments.append(data.strip())
                    continue

            # ---- Frequency block boundaries ---------------------------------
            if label == 'START OF FREQUENCY':
                freq_code    = data[3:6].strip()   # e.g. 'G01'
                current_freq = FreqData(frequency=freq_code)
                in_frequency = True
                continue

            if label == 'END OF FREQUENCY':
                if current_freq is not None:
                    current_ant.frequencies.append(current_freq)
                current_freq  = None
                in_frequency  = False
                continue

            if label == 'START OF FREQ RMS':
                in_freq_rms = True
                continue

            if label == 'END OF FREQ RMS':
                in_freq_rms = False
                continue

            # ---- Inside a frequency block -----------------------------------
            if (in_frequency or in_freq_rms) and current_freq is not None:

                if label == 'NORTH / EAST / UP':
                    vals = _parse_floats(data)
                    if len(vals) >= 3:
                        if in_freq_rms:
                            current_freq.rms.north = vals[0]
                            current_freq.rms.east  = vals[1]
                            current_freq.rms.up    = vals[2]
                        else:
                            current_freq.north_offset = vals[0]
                            current_freq.east_offset  = vals[1]
                            current_freq.up_offset    = vals[2]
                    continue

                # PCV data lines: either "   NOAZI  v1 v2 …" or "  azimuth  v1 v2 …"
                # These lines exceed 80 chars – use the full raw line, not data[:60]
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.upper().startswith('NOAZI'):
                    pcv = _parse_floats(stripped[5:])
                    if in_freq_rms:
                        current_freq.noazi_rms_pcv = pcv
                    else:
                        current_freq.noazi_pcv = pcv
                    continue

                # Try to read as "azimuth_value  v1 v2 …"
                tokens = stripped.split()
                try:
                    azi = float(tokens[0])
                    pcv = [float(v) for v in tokens[1:]]
                    if in_freq_rms:
                        current_freq.azi_rms_pcv[azi] = pcv
                    else:
                        current_freq.azi_pcv[azi] = pcv
                except (ValueError, IndexError):
                    pass  # skip unrecognised lines

    return header, antennas


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def _upsert_atx_file(cnn, path: str, header: AtxHeader) -> int:
    """
    Insert (or fetch existing) row in atx_files.
    Returns atx_file_id.
    """
    filename = os.path.basename(path)

    cnn.cursor.execute("""
        INSERT INTO atx_files (filename, pcv_type, ref_antenna, ref_antenna_serial)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE
            SET pcv_type            = EXCLUDED.pcv_type,
                ref_antenna         = EXCLUDED.ref_antenna,
                ref_antenna_serial  = EXCLUDED.ref_antenna_serial,
                loaded_at           = NOW()
        RETURNING atx_file_id
    """, (filename,
          header.pcv_type or 'A',
          header.ref_antenna or None,
          header.ref_antenna_serial or None))

    row = cnn.cursor.fetchone()
    if row:
        return row['atx_file_id']

    # Fallback: fetch existing id (conflict path returns nothing from RETURNING)
    cnn.cursor.execute(
        "SELECT atx_file_id FROM atx_files WHERE filename = %s", (filename,))
    return cnn.cursor.fetchone()['atx_file_id']


def _ensure_antenna_exists(cnn, antenna_code: str, radome_code: str) -> None:
    """Insert an (AntennaCode, RadomeCode) row into antennas if it does not exist.

    Uses WHERE NOT EXISTS rather than ON CONFLICT so that the insert works
    regardless of whether the antennas table has a single-column PK (pre-
    migration) or the new composite PK on (AntennaCode, RadomeCode).
    AntennaDescription is left NULL — it can be enriched separately from
    a rcvr_ant.tab file; the ATX format does not carry descriptions.
    """
    cnn.cursor.execute("""
        INSERT INTO antennas ("AntennaCode", "RadomeCode")
        SELECT %s, %s
        WHERE NOT EXISTS (
            SELECT 1 FROM antennas
            WHERE  "AntennaCode" = %s
            AND    "RadomeCode"  = %s
        )
    """, (antenna_code, radome_code, antenna_code, radome_code))


def _insert_calibration(cnn, ant: AntennaBlock, atx_file_id: int) -> int:
    """
    Upsert into antenna_calibrations.
    Returns calibration_id.
    """
    cnn.cursor.execute("""
        INSERT INTO antenna_calibrations (
            "AntennaCode", "RadomeCode", serial_no, atx_file_id,
            method, calibrated_by, num_calibrations, cal_date,
            dazi, zen1, zen2, dzen,
            num_frequencies, valid_from, valid_until, sinex_code, comments
        )
        VALUES (%s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s)
        ON CONFLICT ("AntennaCode", "RadomeCode", serial_no, atx_file_id)
        DO UPDATE SET
            method           = EXCLUDED.method,
            calibrated_by    = EXCLUDED.calibrated_by,
            num_calibrations = EXCLUDED.num_calibrations,
            cal_date         = EXCLUDED.cal_date,
            dazi             = EXCLUDED.dazi,
            zen1             = EXCLUDED.zen1,
            zen2             = EXCLUDED.zen2,
            dzen             = EXCLUDED.dzen,
            num_frequencies  = EXCLUDED.num_frequencies,
            valid_from       = EXCLUDED.valid_from,
            valid_until      = EXCLUDED.valid_until,
            sinex_code       = EXCLUDED.sinex_code,
            comments         = EXCLUDED.comments
        RETURNING calibration_id
    """, (
        ant.antenna_code,
        ant.radome_code,
        ant.serial_no,
        atx_file_id,
        ant.method          or None,
        ant.calibrated_by   or None,
        ant.num_calibrations or None,
        ant.cal_date        or None,
        ant.dazi,
        ant.zen1,
        ant.zen2,
        ant.dzen,
        ant.num_frequencies,
        ant.valid_from,
        ant.valid_until,
        ant.sinex_code      or None,
        ant.comments        or None,
    ))

    row = cnn.cursor.fetchone()
    if row:
        return row['calibration_id']

    # Conflict path: RETURNING yields nothing on DO UPDATE when the row
    # was already identical; fetch the id directly.
    cnn.cursor.execute("""
        SELECT calibration_id FROM antenna_calibrations
        WHERE "AntennaCode" = %s AND "RadomeCode" = %s
          AND serial_no = %s AND atx_file_id = %s
    """, (ant.antenna_code, ant.radome_code, ant.serial_no, atx_file_id))
    return cnn.cursor.fetchone()['calibration_id']


def _insert_freq(cnn, calibration_id: int, freq: FreqData) -> int:
    """
    Upsert into antenna_calibration_freq.
    Returns freq_id.
    """
    cnn.cursor.execute("""
        INSERT INTO antenna_calibration_freq (
            calibration_id, frequency,
            north_offset, east_offset, up_offset,
            north_offset_rms, east_offset_rms, up_offset_rms
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (calibration_id, frequency) DO UPDATE SET
            north_offset     = EXCLUDED.north_offset,
            east_offset      = EXCLUDED.east_offset,
            up_offset        = EXCLUDED.up_offset,
            north_offset_rms = EXCLUDED.north_offset_rms,
            east_offset_rms  = EXCLUDED.east_offset_rms,
            up_offset_rms    = EXCLUDED.up_offset_rms
        RETURNING freq_id
    """, (
        calibration_id,
        freq.frequency,
        freq.north_offset,
        freq.east_offset,
        freq.up_offset,
        freq.rms.north,
        freq.rms.east,
        freq.rms.up,
    ))

    row = cnn.cursor.fetchone()
    if row:
        return row['freq_id']

    cnn.cursor.execute("""
        SELECT freq_id FROM antenna_calibration_freq
        WHERE calibration_id = %s AND frequency = %s
    """, (calibration_id, freq.frequency))
    return cnn.cursor.fetchone()['freq_id']


def _insert_pcv_row(cnn,
                    freq_id: int,
                    azimuth: Optional[float],
                    pcv: list[float],
                    pcv_rms: Optional[list[float]]) -> None:
    """Upsert a single PCV row (NOAZI or azimuth-dependent).

    The schema uses two partial unique indexes rather than one full constraint,
    so the ON CONFLICT clause must carry the matching WHERE predicate:
      - NOAZI row  → ON CONFLICT (freq_id) WHERE azimuth IS NULL
      - azimuth row → ON CONFLICT (freq_id, azimuth) WHERE azimuth IS NOT NULL
    """
    if azimuth is None:
        cnn.cursor.execute("""
            INSERT INTO antenna_calibration_pcv
                (freq_id, azimuth, pcv_values, pcv_rms_values)
            VALUES (%s, NULL, %s, %s)
            ON CONFLICT (freq_id) WHERE azimuth IS NULL DO UPDATE SET
                pcv_values     = EXCLUDED.pcv_values,
                pcv_rms_values = EXCLUDED.pcv_rms_values
        """, (freq_id, pcv, pcv_rms))
    else:
        cnn.cursor.execute("""
            INSERT INTO antenna_calibration_pcv
                (freq_id, azimuth, pcv_values, pcv_rms_values)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (freq_id, azimuth) WHERE azimuth IS NOT NULL DO UPDATE SET
                pcv_values     = EXCLUDED.pcv_values,
                pcv_rms_values = EXCLUDED.pcv_rms_values
        """, (freq_id, azimuth, pcv, pcv_rms))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_atx_file(atx_path: str, cnn) -> dict:
    """
    Parse *atx_path* and insert calibration data into the database.

    Parameters
    ----------
    atx_path : str
        Full path to the ANTEX 1.4 file.
    cnn : dbConnection.Cnn
        Open database connection (psycopg2-based).

    Returns
    -------
    dict with keys:
        'atx_file_id'     : int  – ID of the row in atx_files
        'antennas_parsed' : int  – total antenna blocks found
        'antennas_loaded' : int  – antenna blocks successfully inserted
        'errors'          : list[str]  – per-antenna error messages (if any)
    """
    if not os.path.isfile(atx_path):
        raise FileNotFoundError(f"ATX file not found: {atx_path}")

    print(f"[ATX] Parsing {atx_path} …")
    header, antenna_blocks = _parse_atx_file(atx_path)
    print(f"[ATX] Found {len(antenna_blocks)} antenna block(s).")

    errors: list[str] = []
    loaded = 0

    cnn.begin_transac()
    try:
        atx_file_id = _upsert_atx_file(cnn, atx_path, header)
        print(f"[ATX] atx_file_id = {atx_file_id}")

        for ant in antenna_blocks:
            # Each antenna is wrapped in a savepoint so that a failure on one
            # antenna rolls back only its own writes without aborting the
            # surrounding transaction.  Without this, any exception would put
            # the connection into "transaction aborted" state and every
            # subsequent cursor.execute() would fail with the same error.
            cnn.cursor.execute("SAVEPOINT atx_antenna")
            try:
                _ensure_antenna_exists(cnn, ant.antenna_code, ant.radome_code)

                cal_id = _insert_calibration(cnn, ant, atx_file_id)

                for freq in ant.frequencies:
                    freq_id = _insert_freq(cnn, cal_id, freq)

                    # NOAZI row (azimuth = NULL)
                    if freq.noazi_pcv is not None:
                        _insert_pcv_row(
                            cnn, freq_id,
                            azimuth  = None,
                            pcv      = freq.noazi_pcv,
                            pcv_rms  = freq.noazi_rms_pcv,
                        )

                    # Azimuth-dependent rows
                    for azi, pcv in sorted(freq.azi_pcv.items()):
                        _insert_pcv_row(
                            cnn, freq_id,
                            azimuth = azi,
                            pcv     = pcv,
                            pcv_rms = freq.azi_rms_pcv.get(azi),
                        )

                cnn.cursor.execute("RELEASE SAVEPOINT atx_antenna")
                loaded += 1
                print(f"[ATX]   Loaded {ant.antenna_code}/{ant.radome_code}"
                      f" ({len(ant.frequencies)} freq.)")

            except Exception as exc:
                # Roll back to the savepoint to clear the aborted-transaction
                # state, then continue processing the remaining antennas.
                cnn.cursor.execute("ROLLBACK TO SAVEPOINT atx_antenna")
                msg = (f"{ant.antenna_code}/{ant.radome_code} "
                       f"serial='{ant.serial_no}': {exc}")
                errors.append(msg)
                print(f"[ATX] WARNING – skipped {msg}")

        cnn.commit_transac()
        print(f"[ATX] Done. {loaded}/{len(antenna_blocks)} antenna(s) loaded.")

    except Exception:
        cnn.rollback_transac()
        raise

    return {
        'atx_file_id':     atx_file_id,
        'antennas_parsed': len(antenna_blocks),
        'antennas_loaded': loaded,
        'errors':          errors,
    }


# ---------------------------------------------------------------------------
# ATX section reconstructor
# ---------------------------------------------------------------------------

def reconstruct_atx_antenna(cnn, antenna_code: str, radome_code: str,
                             atx_file_id: int,
                             serial_no: str = '') -> str:
    """
    Reconstruct the ATX antenna section for a given calibration
    exactly as it would appear in the original ATX file.

    Returns the reconstructed text block as a string.
    """
    # Fetch calibration header
    cnn.cursor.execute("""
        SELECT * FROM antenna_calibrations
        WHERE "AntennaCode" = %s AND "RadomeCode" = %s
          AND serial_no = %s AND atx_file_id = %s
    """, (antenna_code, radome_code, serial_no, atx_file_id))
    cal = cnn.cursor.fetchone()
    if not cal:
        raise ValueError(f"No calibration found for {antenna_code}/{radome_code}")

    lines: list[str] = []

    def rec(data: str, label: str) -> str:
        return f"{data:<60}{label:<20}\n"

    lines.append(rec('', 'START OF ANTENNA'))

    # TYPE / SERIAL NO
    type_radome = f"{antenna_code:<16}{radome_code:<4}"
    lines.append(rec(f"{type_radome:<20}{serial_no:<20}", 'TYPE / SERIAL NO'))

    # METH / BY / # / DATE
    meth_line = (f"{(cal['method'] or ''):<20}"
                 f"{(cal['calibrated_by'] or ''):<20}"
                 f"{(cal['num_calibrations'] or 0):6}    "
                 f"{(cal['cal_date'] or ''):<10}")
    lines.append(rec(meth_line, 'METH / BY / # / DATE'))

    # DAZI
    lines.append(rec(f"  {cal['dazi']:6.1f}", 'DAZI'))

    # ZEN1 / ZEN2 / DZEN
    lines.append(rec(f"  {cal['zen1']:6.1f}{cal['zen2']:6.1f}{cal['dzen']:6.1f}", 'ZEN1 / ZEN2 / DZEN'))

    # # OF FREQUENCIES
    lines.append(rec(f"{cal['num_frequencies']:6}", '# OF FREQUENCIES'))

    if cal.get('valid_from'):
        dt = cal['valid_from']
        lines.append(rec(f"{dt.year:6}{dt.month:6}{dt.day:6}"
                         f"{dt.hour:6}{dt.minute:6}{dt.second:13.7f}", 'VALID FROM'))

    if cal.get('valid_until'):
        dt = cal['valid_until']
        lines.append(rec(f"{dt.year:6}{dt.month:6}{dt.day:6}"
                         f"{dt.hour:6}{dt.minute:6}{dt.second:13.7f}", 'VALID UNTIL'))

    if cal.get('sinex_code'):
        lines.append(rec(f"{cal['sinex_code']:<10}", 'SINEX CODE'))

    for comment in (cal['comments'] or []):
        lines.append(rec(f"{comment:<60}", 'COMMENT'))

    # Frequencies
    cnn.cursor.execute("""
        SELECT * FROM antenna_calibration_freq
        WHERE calibration_id = %s ORDER BY frequency
    """, (cal['calibration_id'],))
    freqs = cnn.cursor.fetchall()

    for freq in freqs:
        freq_code = freq['frequency']
        lines.append(rec(f"   {freq_code}", 'START OF FREQUENCY'))

        # NORTH / EAST / UP
        lines.append(rec(f"{freq['north_offset']:10.2f}"
                         f"{freq['east_offset']:10.2f}"
                         f"{freq['up_offset']:10.2f}", 'NORTH / EAST / UP'))

        # PCV rows
        cnn.cursor.execute("""
            SELECT azimuth, pcv_values, pcv_rms_values
            FROM antenna_calibration_pcv
            WHERE freq_id = %s
            ORDER BY azimuth NULLS FIRST
        """, (freq['freq_id'],))
        pcv_rows = cnn.cursor.fetchall()

        for row in pcv_rows:
            vals_str = ''.join(f"{v:8.2f}" for v in row['pcv_values'])
            if row['azimuth'] is None:
                lines.append(f"   NOAZI{vals_str}\n")
            else:
                lines.append(f"{row['azimuth']:8.1f}{vals_str}\n")

        # Optional RMS section
        has_rms = (freq['north_offset_rms'] is not None or
                   any(r['pcv_rms_values'] for r in pcv_rows))
        if has_rms:
            lines.append(rec(f"   {freq_code}", 'START OF FREQ RMS'))
            if freq['north_offset_rms'] is not None:
                lines.append(rec(f"{freq['north_offset_rms']:10.2f}"
                                 f"{freq['east_offset_rms']:10.2f}"
                                 f"{freq['up_offset_rms']:10.2f}", 'NORTH / EAST / UP'))
            for row in pcv_rows:
                if row['pcv_rms_values']:
                    rms_str = ''.join(f"{v:8.2f}" for v in row['pcv_rms_values'])
                    if row['azimuth'] is None:
                        lines.append(f"   NOAZI{rms_str}\n")
                    else:
                        lines.append(f"{row['azimuth']:8.1f}{rms_str}\n")
            lines.append(rec(f"   {freq_code}", 'END OF FREQ RMS'))

        lines.append(rec(f"   {freq_code}", 'END OF FREQUENCY'))

    lines.append(rec('', 'END OF ANTENNA'))
    return ''.join(lines)
