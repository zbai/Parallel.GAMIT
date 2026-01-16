"""
Unit tests for station_info module.

Tests parsing of station information records with and without AntennaDAZ field.
"""

import pytest
from datetime import datetime

from geode.metadata.station_info import (
    StationInfoRecord,
    StationInfo,
    StationInfoException,
    StationInfoHeightCodeNotFound,
    create_record
)


class TestStationInfoRecordParsing:
    """Test parsing of station info records from strings."""

    # Test data with AntennaDAZ
    STN_LIST_WITH_DAZ = [
        ' IGM1  Ciudad Autonoma   2003 200  0  0  0  2006 276 11  0  0   0.0000  DHARP   0.0000   0.0000  '
        'ASHTECH UZ-12         CJ00                   9.93  ZR20012102            ASH700936D_M     SNOW   '
        '762                      0.  mstinf: igm1_20140107.log SIRGAS',
        ' IGM1  Ciudad Autonoma   2006 279 13  0  0  2013  86  0  0  0   0.0000  DHARP   0.0000   0.0000  '
        'TRIMBLE NETRS         1.3-1                  0.00  4612261955            ASH700936D_M     SNOW   '
        '762                      0.  mstinf: igm1_20140107.log SIRGAS',
        ' IGM1  Ciudad Autonoma   2013  86  0  0  0  2014   7  0  0  0   0.0000  DHARP   0.0000   0.0000  '
        'TRIMBLE NETRS         1.3-1                  0.00  4912167682            ASH700936D_M     SNOW   '
        '762                      0.  mstinf: igm1_20140107.log SIRGAS',
        ' IGM1  Ciudad Autonoma   2014   7  0  0  0  9999 999  0  0  0   0.0000  DHARP   0.0000   0.0000  '
        'TRIMBLE NETR9         4.17                   4.17  5035K70010            ASH700936D_M     SNOW   '
        '762                      0.  mstinf: igm1_20140107.log SIRGAS'
    ]



    # Test data with AntennaDAZ
    STN_LIST_NO_DAZ = [
        ' IGM1  Ciudad Autonoma   2003 200  0  0  0  2006 276 11  0  0   0.0000  DHARP   0.0000   0.0000  '
        'ASHTECH UZ-12         CJ00                   9.93  ZR20012102            ASH700936D_M     SNOW   '
        '762                 mstinf: igm1_20140107.log SIRGAS',
        ' IGM1  Ciudad Autonoma   2006 279 13  0  0  2013  86  0  0  0   0.0000  DHARP   0.0000   0.0000  '
        'TRIMBLE NETRS         1.3-1                  0.00  4612261955            ASH700936D_M     SNOW   '
        '762                 mstinf: igm1_20140107.log SIRGAS',
        ' IGM1  Ciudad Autonoma   2013  86  0  0  0  2014   7  0  0  0   0.0000  DHARP   0.0000   0.0000  '
        'TRIMBLE NETRS         1.3-1                  0.00  4912167682            ASH700936D_M     SNOW   '
        '762                 10.0',
        ' IGM1  Ciudad Autonoma   2014   7  0  0  0  9999 999  0  0  0   0.0000  DHARP   0.0000   0.0000  '
        'TRIMBLE NETR9         4.17                   4.17  5035K70010            ASH700936D_M     SNOW   '
        '762                 '
    ]

    def test_parse_record_without_daz(self):
        """Test parsing records without AntennaDAZ field."""
        record = StationInfoRecord.from_string(self.STN_LIST_NO_DAZ[0])
        
        assert record is not None
        assert record.StationCode == 'igm1'
        assert record.AntennaHeight == 0.0
        assert record.HeightCode == 'DHARP'
        assert record.AntennaNorth == 0.0
        assert record.AntennaEast == 0.0
        assert record.ReceiverCode == 'ASHTECH UZ-12'
        assert record.ReceiverVers == 'CJ00'
        assert record.ReceiverFirmware == '9.93'
        assert record.ReceiverSerial == 'ZR20012102'
        assert record.AntennaCode == 'ASH700936D_M'
        assert record.RadomeCode == 'SNOW'
        assert record.AntennaSerial == '762'
        assert record.AntennaDAZ == 0.0  # Should default to 0.0
        assert 'mstinf: igm1_20140107.log SIRGAS' in record.Comments

    def test_parse_record_with_daz(self):
        """Test parsing records with AntennaDAZ field."""
        record = StationInfoRecord.from_string(self.STN_LIST_WITH_DAZ[0])
        
        assert record is not None
        assert record.StationCode == 'igm1'
        assert record.AntennaDAZ == 0.0
        assert record.ReceiverCode == 'ASHTECH UZ-12'
        assert record.AntennaCode == 'ASH700936D_M'
        assert 'mstinf: igm1_20140107.log SIRGAS' in record.Comments

    def test_parse_all_records_without_daz(self):
        """Test parsing all records without AntennaDAZ."""
        records = [StationInfoRecord.from_string(line) for line in self.STN_LIST_NO_DAZ]
        
        assert len(records) == 4
        assert all(r is not None for r in records)
        assert all(r.StationCode == 'igm1' for r in records)
        
        # Check date progression
        assert records[0].DateStart.year == 2003
        assert records[0].DateStart.doy == 200
        assert records[0].DateEnd.year == 2006
        assert records[0].DateEnd.doy == 276
        
        # Check last record has open end date
        assert records[-1].DateEnd.year is None

    def test_parse_all_records_with_daz(self):
        """Test parsing all records with AntennaDAZ."""
        records = [StationInfoRecord.from_string(line) for line in self.STN_LIST_WITH_DAZ]
        
        assert len(records) == 4
        assert all(r is not None for r in records)
        assert all(r.AntennaDAZ == 0.0 for r in records)

    def test_parse_invalid_record(self):
        """Test parsing invalid records returns None."""
        assert StationInfoRecord.from_string("") is None
        assert StationInfoRecord.from_string("short") is None
        assert StationInfoRecord.from_string("X" * 100) is None  # Doesn't start with space

    def test_equipment_changes(self):
        """Test that equipment changes are detected correctly."""
        records = [StationInfoRecord.from_string(line) for line in self.STN_LIST_NO_DAZ]
        
        # First record: ASHTECH receiver
        assert records[0].ReceiverCode == 'ASHTECH UZ-12'
        assert records[0].ReceiverSerial == 'ZR20012102'
        
        # Second record: TRIMBLE NETRS
        assert records[1].ReceiverCode == 'TRIMBLE NETRS'
        assert records[1].ReceiverSerial == '4612261955'
        
        # Third record: Different serial
        assert records[2].ReceiverCode == 'TRIMBLE NETRS'
        assert records[2].ReceiverSerial == '4912167682'
        
        # Fourth record: NETR9
        assert records[3].ReceiverCode == 'TRIMBLE NETR9'
        assert records[3].ReceiverSerial == '5035K70010'


class TestStationInfoRecordCreation:
    """Test StationInfoRecord creation and manipulation."""

    def test_create_empty_record(self):
        """Test creating an empty record."""
        record = StationInfoRecord()
        
        assert record.NetworkCode is None
        assert record.StationCode is None
        assert record.AntennaHeight == 0.0
        assert record.hash is not None  # Hash should be calculated

    def test_create_record_with_params(self):
        """Test creating record with parameters."""
        record = StationInfoRecord(
            NetworkCode='TEST',
            StationCode='TST1',
            AntennaHeight=1.5,
            AntennaCode='TEST_ANT',
            ReceiverCode='TEST_REC'
        )
        
        assert record.NetworkCode == 'TEST'
        assert record.StationCode == 'TST1'
        assert record.AntennaHeight == 1.5
        assert record.AntennaCode == 'TEST_ANT'
        assert record.ReceiverCode == 'TEST_REC'

    def test_record_with_string_init(self):
        """Test initializing record with string using _record parameter."""
        stn_str = (' IGM1  Ciudad Autonoma   2003 200  0  0  0  2006 276 11  0  0   0.0000  DHARP   '
                   '0.0000   0.0000  ASHTECH UZ-12         CJ00                   9.93  ZR20012102            '
                   'ASH700936D_M     SNOW   762                      0.  Test comment')
        
        record = StationInfoRecord(_record=stn_str)
        
        assert record.StationCode == 'igm1'
        assert record.ReceiverCode == 'ASHTECH UZ-12'
        assert record.AntennaCode == 'ASH700936D_M'

    def test_record_to_string(self):
        """Test converting record back to string format."""
        stn_str = TestStationInfoRecordParsing.STN_LIST_WITH_DAZ[0]
        record = StationInfoRecord.from_string(stn_str)
        
        output = str(record)
        
        # Should start with space and station code
        assert output.startswith(' IGM1')
        # Should contain key fields
        assert 'ASHTECH UZ-12' in output
        assert 'ASH700936D_M' in output
        assert 'DHARP' in output

    def test_dictionary_access(self):
        """Test dictionary-style access for backward compatibility."""
        record = StationInfoRecord(StationCode='TEST', AntennaHeight=1.5)
        
        # Get
        assert record['StationCode'] == 'TEST'
        assert record['AntennaHeight'] == 1.5
        
        # Set
        record['AntennaHeight'] = 2.0
        assert record.AntennaHeight == 2.0

    def test_hash_calculation(self):
        """Test that hash is calculated correctly."""
        record1 = StationInfoRecord(
            AntennaHeight=1.5,
            AntennaNorth=0.0,
            AntennaEast=0.0,
            HeightCode='DHARP',
            AntennaCode='TEST_ANT',
            RadomeCode='NONE',
            ReceiverCode='TEST_REC'
        )
        
        record2 = StationInfoRecord(
            AntennaHeight=1.5,
            AntennaNorth=0.0,
            AntennaEast=0.0,
            HeightCode='DHARP',
            AntennaCode='TEST_ANT',
            RadomeCode='NONE',
            ReceiverCode='TEST_REC'
        )
        
        # Same configuration should have same hash
        assert record1.hash == record2.hash
        
        # Different configuration should have different hash
        record3 = StationInfoRecord(
            AntennaHeight=2.0,  # Different
            AntennaNorth=0.0,
            AntennaEast=0.0,
            HeightCode='DHARP',
            AntennaCode='TEST_ANT',
            RadomeCode='NONE',
            ReceiverCode='TEST_REC'
        )
        assert record1.hash != record3.hash

    def test_to_json(self):
        """Test JSON serialization."""
        record = StationInfoRecord(
            NetworkCode='TEST',
            StationCode='TST1',
            AntennaHeight=1.5
        )
        
        json_data = record.to_json()
        
        assert isinstance(json_data, dict)
        assert json_data['NetworkCode'] == 'TEST'
        assert json_data['StationCode'] == 'TST1'
        assert json_data['AntennaHeight'] == 1.5


class TestStationInfoClass:
    """Test StationInfo class (without database connection)."""

    def test_create_empty_station_info(self):
        """Test creating StationInfo without database."""
        stninfo = StationInfo()
        
        assert stninfo.NetworkCode is None
        assert stninfo.StationCode is None
        assert len(stninfo.records) == 0
        assert stninfo.record_count == 0

    def test_parse_station_info_list_without_daz(self):
        """Test parsing station info from list without DAZ."""
        stninfo = StationInfo()
        records = stninfo.parse_station_info(
            TestStationInfoRecordParsing.STN_LIST_NO_DAZ
        )
        
        assert len(records) == 4
        assert all(isinstance(r, StationInfoRecord) for r in records)
        assert records[0].ReceiverCode == 'ASHTECH UZ-12'
        assert records[-1].ReceiverCode == 'TRIMBLE NETR9'

    def test_parse_station_info_list_with_daz(self):
        """Test parsing station info from list with DAZ."""
        stninfo = StationInfo()
        records = stninfo.parse_station_info(
            TestStationInfoRecordParsing.STN_LIST_WITH_DAZ
        )
        
        assert len(records) == 4
        assert all(r.AntennaDAZ == 0.0 for r in records)

    def test_records_are_equal(self):
        """Test static method for comparing records."""
        record1 = StationInfoRecord(
            ReceiverCode='TEST_REC',
            ReceiverSerial='12345',
            AntennaCode='TEST_ANT',
            AntennaSerial='67890',
            AntennaHeight=1.5,
            AntennaNorth=0.0,
            AntennaEast=0.0,
            HeightCode='DHARP',
            RadomeCode='NONE'
        )
        
        record2 = StationInfoRecord(
            ReceiverCode='TEST_REC',
            ReceiverSerial='12345',
            AntennaCode='TEST_ANT',
            AntennaSerial='67890',
            AntennaHeight=1.5,
            AntennaNorth=0.0,
            AntennaEast=0.0,
            HeightCode='DHARP',
            RadomeCode='NONE'
        )
        
        assert StationInfo.records_are_equal(record1, record2)
        
        # Change one field
        record2.ReceiverSerial = '99999'
        assert not StationInfo.records_are_equal(record1, record2)


class TestCreateRecordFunction:
    """Test the create_record helper function."""

    def test_create_record(self):
        """Test creating a formatted record string."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)
        
        record_str = create_record(
            station_code='TEST',
            start_date=start,
            end_date=end,
            antenna_code='TEST_ANT',
            comment='Test comment'
        )
        
        assert record_str.startswith(' TEST')
        assert 'DHARP' in record_str
        assert 'TEST_ANT' in record_str
        assert 'Test comment' in record_str


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_record_with_missing_fields(self):
        """Test parsing record with some empty fields."""
        # Record with empty receiver serial
        stn_str = (
            ' IGM1  Ciudad Autonoma   2014   7  0  0  0  9999 999  0  0  0   0.0000  DHARP   0.0000   0.0000  '
            'TRIMBLE NETR9         4.17                   4.17                        ASH700936D_M     SNOW   '
            '762                      0.  mstinf: igm1_20140107.log SIRGAS'
        )


        record = StationInfoRecord.from_string(stn_str)
        
        assert record is not None
        assert record.StationCode == 'igm1'
        assert record.ReceiverCode == 'TRIMBLE NETR9'
        assert record.ReceiverSerial == ''

    def test_parse_record_with_whitespace(self):
        """Test that whitespace is properly stripped."""
        stn_str = (
            ' IGM1  Ciudad Autonoma   2003 200  0  0  0  2006 276 11  0  0   0.0000  DHARP   0.0000   0.0000  '
            'ASHTECH UZ-12         CJ00                   9.93  ZR20012102            ASH700936D_M     SNOW   '
            '762                 mstinf: igm1_20140107.log SIRGAS                                  '
        )
        
        record = StationInfoRecord.from_string(stn_str)
        
        assert record is not None
        # Fields should not have trailing/leading whitespace
        assert record.ReceiverCode == 'ASHTECH UZ-12'
        assert not record.ReceiverCode.startswith(' ')
        assert not record.ReceiverCode.endswith(' ')

    def test_round_trip_parsing(self):
        """Test that parsing and stringifying produces consistent results."""
        original = TestStationInfoRecordParsing.STN_LIST_WITH_DAZ[0]
        record = StationInfoRecord.from_string(original)
        reparsed = StationInfoRecord.from_string(str(record))
        
        # Key fields should match
        assert record.StationCode == reparsed.StationCode
        assert record.ReceiverCode == reparsed.ReceiverCode
        assert record.AntennaCode == reparsed.AntennaCode
        assert record.AntennaHeight == reparsed.AntennaHeight
        assert record.AntennaDAZ == reparsed.AntennaDAZ


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager(self):
        """Test using StationInfo as context manager."""
        with StationInfo() as stninfo:
            stninfo.records = [StationInfoRecord()]
            assert len(stninfo.records) == 1
        
        # Records should be cleared after context
        assert stninfo.records is None


class TestPickling:
    """Test pickle support."""

    def test_getstate_removes_connection(self):
        """Test that __getstate__ removes database connection."""
        stninfo = StationInfo()
        stninfo.cnn = "mock_connection"  # Simulate connection
        
        state = stninfo.__getstate__()
        
        assert 'cnn' not in state

    def test_setstate_restores_without_connection(self):
        """Test that __setstate__ restores object without connection."""
        stninfo = StationInfo()
        state = {'NetworkCode': 'TEST', 'StationCode': 'TST1', 'records': []}
        
        stninfo.__setstate__(state)
        
        assert stninfo.NetworkCode == 'TEST'
        assert stninfo.StationCode == 'TST1'
        assert stninfo.cnn is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
