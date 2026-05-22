#!/usr/bin/env python
"""
Test script for StationMetadataComparator.

Tests the comparator with a real station from the database against an IGS log file.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from geode.dbConnection import Cnn
from geode.metadata.serializers import bundle_from_db, bundle_from_file
from geode.metadata.comparator import StationMetadataComparator, ComparatorError
from geode.metadata.report import ReportParseError


def run_comparator_test(cfg_path: str,
                        log_path: str,
                        network_code: str,
                        station_code: str):
    """
    Manual integration test for the comparator with a real station.

    Args:
        cfg_path: Path to gnss_data.cfg
        log_path: Path to IGS log file
        network_code: Network code (e.g., 'rms')
        station_code: Station code (e.g., 'unro')
    """
    print(f"\n{'='*60}")
    print(f"Testing StationMetadataComparator")
    print(f"Station: {network_code}.{station_code}")
    print(f"Log file: {log_path}")
    print(f"{'='*60}\n")

    # Connect to database
    print("1. Connecting to database...")
    try:
        cnn = Cnn(cfg_path)
        print("   Database connection successful")
    except Exception as e:
        print(f"   ERROR: Failed to connect to database: {e}")
        return False

    # Fetch database bundle
    print("\n2. Fetching database sessions...")
    try:
        db_bundle = bundle_from_db(cnn, network_code, station_code)
        print(f"   Found {len(db_bundle.sessions)} session(s) in database")
        for i, s in enumerate(db_bundle.sessions):
            end_str = s.DateEnd.strftime() if s.DateEnd else "open"
            print(f"   [{i+1}] {s.DateStart.strftime()} - {end_str}")
            print(f"       Rcv: {s.ReceiverCode} ({s.ReceiverFirmware})")
            print(f"       Ant: {s.AntennaCode} {s.RadomeCode}")
    except Exception as e:
        print(f"   ERROR: Failed to fetch database bundle: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Parse IGS log file
    print("\n3. Parsing IGS log file...")
    try:
        file_bundle = bundle_from_file(log_path, network_code, station_code)
        print(f"   Found {len(file_bundle.sessions)} session(s) in log file")
        for i, s in enumerate(file_bundle.sessions):
            end_str = s.DateEnd.strftime() if s.DateEnd else "open"
            print(f"   [{i+1}] {s.DateStart.strftime()} - {end_str}")
            print(f"       Rcv: {s.ReceiverCode} ({s.ReceiverFirmware})")
            print(f"       Ant: {s.AntennaCode} {s.RadomeCode}")
    except Exception as e:
        print(f"   ERROR: Failed to parse log file: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n4. ANTHROPIC_API_KEY not set - skipping API call")
        print("   Set the environment variable to test the full comparator")
        return True

    # Initialize comparator
    print("\n4. Initializing comparator...")
    try:
        comparator = StationMetadataComparator()
        print(f"   Using model: {comparator.model}")
    except ComparatorError as e:
        print(f"   ERROR: {e}")
        return False

    # Run comparison
    print("\n5. Running comparison...")
    try:
        report = comparator.compare(db_bundle, file_bundle, file_source="IGS log")
        print(f"   Comparison complete")
    except ComparatorError as e:
        print(f"   ERROR: API error: {e}")
        return False
    except ReportParseError as e:
        print(f"   ERROR: Failed to parse response: {e}")
        return False

    # Display results
    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")
    print(f"Station: {report.network_code}.{report.station_code}")
    print(f"Summary: {report.summary}")
    print(f"Needs attention: {report.needs_attention}")
    print(f"Total findings: {len(report.findings)}")

    if report.findings:
        print(f"\nFindings:")
        for i, f in enumerate(report.findings):
            print(f"\n  [{i+1}] {f.finding_type} - Action: {f.action}")
            print(f"      Description: {f.description}")
            print(f"      Affected fields: {f.affected_fields}")
            if f.db_record:
                print(f"      DB record: {f.db_record}")

    print(f"\n{'='*60}")
    print("TEST COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")

    return True


if __name__ == '__main__':
    # Default paths
    cfg_path = os.path.expanduser('~/pg_osu/gnss_data.cfg')
    log_path = os.path.expanduser('~/Downloads/UNRO.log')

    # Network and station codes for UNRO
    # UNRO is in Argentina
    network_code = 'arg'
    station_code = 'unro'

    # Allow overrides from command line
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    if len(sys.argv) > 2:
        station_code = sys.argv[2].lower()
    if len(sys.argv) > 3:
        network_code = sys.argv[3].lower()

    success = run_comparator_test(cfg_path, log_path, network_code, station_code)
    sys.exit(0 if success else 1)
