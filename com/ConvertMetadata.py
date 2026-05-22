#!/usr/bin/env python
"""
Project: Geodesy Database Engine (GeoDE)
Date: May 2026
Author: Demian D. Gomez

Convert between different GNSS metadata formats.

Supported conversions:
  - IGS site log -> Station info format
  - Station info -> IGS site log format

Usage examples:
    ConvertMetadata.py igslog station.log
    ConvertMetadata.py igslog *.log -o station.info
    ConvertMetadata.py stninfo station.info -o XXXX.log
"""

import argparse
import sys
import os
import glob
from datetime import datetime
from typing import List, Dict

from geode.metadata import igslog
from geode.metadata.station_info import StationInfoRecord, StationInfo
from geode.Utils import add_version_argument


# =============================================================================
# Station Info Format Output
# =============================================================================

def format_station_info_header() -> str:
    """Return the standard station info file header."""
    return StationInfo.HEADER


# =============================================================================
# IGS Log Conversion
# =============================================================================

def process_igslog_file(log_path: str, verbose: bool = False) -> list:
    """Process a single IGS log file.

    Args:
        log_path: Path to the IGS log file
        verbose: Print progress information

    Returns:
        List of formatted station info records (strings)
    """
    if verbose:
        print(f'Processing: {log_path}')

    sessions = igslog.parse_igs_log_file(log_path)

    if not sessions:
        print(f'Warning: No sessions found in {log_path}', file=sys.stderr)
        return []

    # Use StationInfoRecord.__str__() to format as station info line
    records = [str(session) for session in sessions]

    if verbose:
        print(f'  Found {len(records)} session(s)')

    return records


def cmd_igslog(args):
    """Handle the 'igslog' subcommand - convert IGS log to station info."""

    # Expand glob patterns (needed on Windows)
    log_files = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            log_files.extend(matches)
        else:
            log_files.append(pattern)

    if not log_files:
        print('Error: No log files found', file=sys.stderr)
        sys.exit(1)

    # Process all log files
    all_records = []
    for log_file in log_files:
        if not os.path.isfile(log_file):
            print(f'Error: File not found: {log_file}', file=sys.stderr)
            continue

        records = process_igslog_file(log_file, verbose=args.verbose)
        all_records.extend(records)

    if not all_records:
        print('Error: No station info records generated', file=sys.stderr)
        sys.exit(1)

    # Determine output destination
    if args.append:
        output_file = args.append
        write_header = False
        mode = 'a'

        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            write_header = not args.no_header
            mode = 'w'
    elif args.output:
        output_file = args.output
        write_header = not args.no_header
        mode = 'w'
    else:
        output_file = None
        write_header = not args.no_header

    # Write output
    if output_file:
        with open(output_file, mode) as f:
            if write_header:
                f.write(format_station_info_header() + '\n')
            for record in all_records:
                f.write(record + '\n')

        if args.verbose:
            action = 'Appended to' if args.append else 'Wrote'
            print(f'{action} {output_file}: {len(all_records)} record(s)')
    else:
        if write_header:
            print(format_station_info_header())
        for record in all_records:
            print(record)

    if args.verbose:
        print(f'Total: {len(all_records)} station info record(s) from {len(log_files)} file(s)')


# =============================================================================
# Station Info to IGS Log Conversion
# =============================================================================

def parse_station_info_file(file_path: str, verbose: bool = False) -> List[StationInfoRecord]:
    """Parse a station info file into records.

    Args:
        file_path: Path to station info file
        verbose: Print progress information

    Returns:
        List of StationInfoRecord objects
    """
    if verbose:
        print(f'Reading: {file_path}')

    records = []
    with open(file_path, 'r') as f:
        for line in f:
            # Skip header and comment lines
            if line.startswith('*') or line.startswith('#') or not line.strip():
                continue

            record = StationInfoRecord.from_string(line)
            if record and record.DateStart:
                record.source = 'stninfo'
                records.append(record)

    if verbose:
        print(f'  Found {len(records)} record(s)')

    return records


def group_records_by_station(
    records: List[StationInfoRecord]
) -> Dict[str, List[StationInfoRecord]]:
    """Group station info records by station code.

    Args:
        records: List of StationInfoRecord objects

    Returns:
        Dictionary mapping station codes to their records
    """
    grouped = {}
    for record in records:
        code = (record.StationCode or 'XXXX').upper()
        if code not in grouped:
            grouped[code] = []
        grouped[code].append(record)
    return grouped


def process_stninfo_file(
    file_path: str,
    output_dir: str,
    verbose: bool = False
) -> List[str]:
    """Process a station info file and generate IGS log files.

    Args:
        file_path: Path to station info file
        output_dir: Directory for output IGS log files
        verbose: Print progress information

    Returns:
        List of generated IGS log file paths
    """
    records = parse_station_info_file(file_path, verbose)
    if not records:
        return []

    # Group records by station
    grouped = group_records_by_station(records)

    generated_files = []
    for station_code, station_records in grouped.items():
        # Sort records by start time
        station_records.sort(key=lambda r: r.DateStart.datetime() if r.DateStart else datetime.min)

        # Generate output filename
        output_file = os.path.join(output_dir, f'{station_code}.log')

        # Write IGS log file (now accepts StationInfoRecord directly)
        igslog.write_igs_log_file(station_records, output_file)
        generated_files.append(output_file)

        if verbose:
            print(f'  Generated: {output_file} ({len(station_records)} session(s))')

    return generated_files


def cmd_stninfo(args):
    """Handle the 'stninfo' subcommand - convert station info to IGS log."""

    # Expand glob patterns (needed on Windows)
    stninfo_files = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            stninfo_files.extend(matches)
        else:
            stninfo_files.append(pattern)

    if not stninfo_files:
        print('Error: No station info files found', file=sys.stderr)
        sys.exit(1)

    # Collect all records from all files
    all_records = []
    for stninfo_file in stninfo_files:
        if not os.path.isfile(stninfo_file):
            print(f'Error: File not found: {stninfo_file}', file=sys.stderr)
            continue
        records = parse_station_info_file(stninfo_file, verbose=args.verbose)
        all_records.extend(records)

    if not all_records:
        print('Error: No station info records found', file=sys.stderr)
        sys.exit(1)

    # Group records by station
    grouped = group_records_by_station(all_records)

    # Handle single station output to file or stdout
    if args.output:
        if len(grouped) > 1 and not args.output.endswith('/'):
            print('Warning: Multiple stations found, using output as directory',
                  file=sys.stderr)
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            single_output = False
        else:
            single_output = True
    else:
        single_output = False
        output_dir = args.output_dir or '.'

    generated_count = 0
    for station_code, station_records in grouped.items():
        # Sort records by start time
        station_records.sort(key=lambda r: r.DateStart.datetime() if r.DateStart else datetime.min)

        if single_output:
            # Write to specific file
            igslog.write_igs_log_file(station_records, args.output)
            if args.verbose:
                print(f'Wrote: {args.output} ({len(station_records)} session(s))')
            generated_count += 1
            break  # Only output first station when writing to single file
        elif args.output_dir:
            # Write to output directory
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{station_code}.log')
            igslog.write_igs_log_file(station_records, output_file)
            if args.verbose:
                print(f'Wrote: {output_file} ({len(station_records)} session(s))')
            generated_count += 1
        else:
            # Output to stdout
            content = igslog.generate_igs_log(station_records)
            print(content)
            print()  # Blank line between stations
            generated_count += 1

    if args.verbose and generated_count > 0:
        print(f'Total: {generated_count} IGS log file(s) from '
              f'{len(all_records)} record(s)')


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert between different GNSS metadata formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Supported conversions:
  igslog    Convert IGS site log files to station info format
  stninfo   Convert station info files to IGS site log format

Examples:
  %(prog)s igslog station.log                    # IGS log to station info (stdout)
  %(prog)s igslog *.log -o station.info          # Multiple IGS logs to single file
  %(prog)s stninfo station.info                  # Station info to IGS log (stdout)
  %(prog)s stninfo station.info -o SRLP.log      # Single station to IGS log file
  %(prog)s stninfo station.info -d logs/         # Multiple stations to directory
        '''
    )

    add_version_argument(parser)

    subparsers = parser.add_subparsers(
        dest='command',
        title='commands',
        description='Available conversion commands'
    )

    # IGS log to station info subcommand
    igslog_parser = subparsers.add_parser(
        'igslog',
        help='Convert IGS site log files to station info format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s station.log                    # Parse single file, output to stdout
  %(prog)s *.log -o station.info          # Parse multiple files, write to file
  %(prog)s logs/*.log --append stn.info   # Append to existing station info file
  %(prog)s station.log --no-header        # Output without header line
        '''
    )

    igslog_parser.add_argument(
        'files',
        nargs='+',
        metavar='LOGFILE',
        help='IGS site log file(s) to convert'
    )

    igslog_parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Output file (default: stdout)'
    )

    igslog_parser.add_argument(
        '--append',
        metavar='FILE',
        help='Append to existing station info file'
    )

    igslog_parser.add_argument(
        '--no-header',
        action='store_true',
        help='Do not print the header line'
    )

    igslog_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print progress information'
    )

    igslog_parser.set_defaults(func=cmd_igslog)

    # Station info to IGS log subcommand
    stninfo_parser = subparsers.add_parser(
        'stninfo',
        help='Convert station info files to IGS site log format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s station.info                     # Output IGS log to stdout
  %(prog)s station.info -o SRLP.log         # Write single station to file
  %(prog)s station.info -d logs/            # Write multiple stations to directory
  %(prog)s *.info -d logs/                  # Process multiple station info files
        '''
    )

    stninfo_parser.add_argument(
        'files',
        nargs='+',
        metavar='STNINFO',
        help='Station info file(s) to convert'
    )

    stninfo_parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Output IGS log file (for single station)'
    )

    stninfo_parser.add_argument(
        '-d', '--output-dir',
        metavar='DIR',
        help='Output directory for IGS log files (one per station)'
    )

    stninfo_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print progress information'
    )

    stninfo_parser.set_defaults(func=cmd_stninfo)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Validate arguments
    if args.command == 'igslog':
        if args.output and args.append:
            igslog_parser.error('Cannot use both --output and --append')
    elif args.command == 'stninfo':
        if args.output and args.output_dir:
            stninfo_parser.error('Cannot use both --output and --output-dir')

    # Execute the command
    args.func(args)


if __name__ == '__main__':
    main()
