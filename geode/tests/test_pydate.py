"""
Comprehensive unit tests for geode/pyDate.py

Tests cover:
- Date creation from various input formats (year/doy, datetime, gpsWeek, fyear, mjd, stninfo)
- Year 9999 handling (open-ended/far-future dates)
- None and empty string handling for stninfo
- Invalid input handling and error messages
- Comparison operators
- Arithmetic operations (add/subtract days)
- datetime() conversion
- first_epoch/last_epoch methods
- String representations and formatting
- Helper functions
"""

import pytest
from datetime import datetime

from geode.pyDate import (
    Date,
    pyDateException,
    parse_stninfo,
    yeardoy2fyear,
    fyear2yeardoy,
    date2doy,
    doy2date,
    date2gpsDate,
    gpsDate2mjd,
    mjd2date,
)


class TestDateCreation:
    """Tests for Date object creation from various input formats."""

    def test_year_doy(self):
        """Create Date from year and day of year."""
        d = Date(year=2024, doy=100)
        assert d.year == 2024
        assert d.doy == 100
        assert d.month == 4
        assert d.day == 9  # April 9, 2024 (leap year)

    def test_year_doy_non_leap_year(self):
        """Create Date from year and doy in non-leap year."""
        d = Date(year=2023, doy=100)
        assert d.year == 2023
        assert d.doy == 100
        assert d.month == 4
        assert d.day == 10  # April 10, 2023 (non-leap year)

    def test_year_month_day(self):
        """Create Date from year, month, and day."""
        d = Date(year=2024, month=4, day=9)
        assert d.year == 2024
        assert d.month == 4
        assert d.day == 9
        assert d.doy == 100  # leap year

    def test_datetime_input(self):
        """Create Date from datetime object."""
        dt = datetime(2024, 7, 15, 10, 30, 45)
        d = Date(datetime=dt)
        assert d.year == 2024
        assert d.month == 7
        assert d.day == 15
        assert d.hour == 10
        assert d.minute == 30
        assert d.second == 45
        assert d.doy == 197  # July 15, 2024 in leap year

    def test_gpsweek_gpsweekday(self):
        """Create Date from GPS week and week day."""
        # GPS week 2350, day 0 = Sunday of that week
        d = Date(gpsWeek=2350, gpsWeekDay=0)
        assert d.gpsWeek == 2350
        assert d.gpsWeekDay == 0
        assert d.year is not None
        assert d.doy is not None

    def test_fyear(self):
        """Create Date from fractional year."""
        d = Date(fyear=2024.5)  # Middle of 2024
        assert d.year == 2024
        # Should be around day 183-184
        assert 180 < d.doy < 190

    def test_mjd(self):
        """Create Date from Modified Julian Day."""
        # MJD 60000 corresponds to 2023-02-25
        d = Date(mjd=60000)
        assert d.year == 2023
        assert d.month == 2
        assert d.day == 25

    def test_stninfo_format(self):
        """Create Date from station info format string."""
        d = Date(stninfo='2024 100 12 30 45')
        assert d.year == 2024
        assert d.doy == 100
        assert d.hour == 12
        assert d.minute == 30
        assert d.second == 45
        assert d.from_stninfo is True

    def test_two_digit_year_after_80(self):
        """Two-digit year after 80 should be 19xx."""
        d = Date(year=85, doy=100)
        assert d.year == 1985

    def test_two_digit_year_at_or_below_80(self):
        """Two-digit year at or below 80 should be 20xx."""
        d = Date(year=24, doy=100)
        assert d.year == 2024
        d2 = Date(year=80, doy=100)
        assert d2.year == 2080  # 80 is not > 80, so becomes 2080

    def test_with_hour_minute_second(self):
        """Create Date with explicit time components."""
        d = Date(year=2024, doy=100, hour=15, minute=30, second=45)
        assert d.hour == 15
        assert d.minute == 30
        assert d.second == 45


class TestYear9999Handling:
    """Tests for year 9999 (open-ended/far-future date) handling."""

    def test_year_9999_doy_creation(self):
        """Year 9999 with doy should normalize to standard far-future date."""
        d = Date(year=9999, doy=1)
        assert d.year == 9999
        assert d.doy == 1
        assert d.month == 1
        assert d.day == 1
        assert d.fyear == 9999.0
        assert d.gpsWeek == 99999
        assert d.gpsWeekDay == 0
        assert d.mjd == 99999999

    def test_year_9999_arbitrary_doy(self):
        """Year 9999 with arbitrary doy should normalize."""
        d = Date(year=9999, doy=365)
        assert d.year == 9999
        assert d.doy == 1  # Normalized
        assert d.fyear == 9999.0

    def test_stninfo_9999_format(self):
        """Station info format with year 9999."""
        d = Date(stninfo='9999 999 00 00 00')
        assert d.year == 9999
        assert d.fyear == 9999.0
        assert d.from_stninfo is True

    def test_str_year_9999(self):
        """String representation for year 9999."""
        d = Date(year=9999, doy=1)
        assert str(d) == '9999 999 00 00 00'

    def test_datetime_year_9999(self):
        """datetime() method for year 9999."""
        d = Date(year=9999, doy=1)
        dt = d.datetime()
        assert dt.year == 9999
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0

    def test_comparison_with_year_9999(self):
        """Year 9999 dates should compare as greater than normal dates."""
        d_far_future = Date(year=9999, doy=1)
        d_normal = Date(year=2024, doy=100)

        assert d_far_future > d_normal
        assert d_far_future >= d_normal
        assert d_normal < d_far_future
        assert d_normal <= d_far_future
        assert d_far_future != d_normal

    def test_two_year_9999_dates_equal(self):
        """Two year 9999 dates should be equal."""
        d1 = Date(year=9999, doy=1)
        d2 = Date(stninfo='9999 999 00 00 00')
        # fyear comparison for equality
        assert d1.fyear == d2.fyear


class TestStnInfoNoneAndEmpty:
    """Tests for None and empty string handling in stninfo."""

    def test_stninfo_none(self):
        """stninfo=None should create far-future date."""
        d = Date(stninfo=None)
        assert d.year == 9999
        assert d.doy == 1
        assert d.fyear == 9999.0
        assert d.gpsWeek == 99999
        assert d.mjd == 99999999
        assert d.from_stninfo is True

    def test_stninfo_empty_string(self):
        """stninfo='' should create far-future date."""
        d = Date(stninfo='')
        assert d.year == 9999
        assert d.fyear == 9999.0
        assert d.from_stninfo is True

    def test_stninfo_whitespace_only(self):
        """stninfo='   ' should create far-future date."""
        d = Date(stninfo='   ')
        assert d.year == 9999
        assert d.fyear == 9999.0

    def test_str_of_stninfo_none(self):
        """String representation of stninfo=None should be '9999 999 00 00 00'."""
        d = Date(stninfo=None)
        assert str(d) == '9999 999 00 00 00'


class TestParseStninfo:
    """Tests for the parse_stninfo function."""

    def test_valid_format(self):
        """Parse valid station info format."""
        year, doy, hour, minute, second = parse_stninfo('2024 100 12 30 45')
        assert year == 2024
        assert doy == 100
        assert hour == 12
        assert minute == 30
        assert second == 45

    def test_year_9999(self):
        """Parse year 9999 returns normalized values."""
        year, doy, hour, minute, second = parse_stninfo('9999 999 00 00 00')
        assert year == 9999
        assert doy == 1  # Normalized
        assert hour == 0
        assert minute == 0
        assert second == 0

    def test_none_input(self):
        """None input returns far-future date components."""
        year, doy, hour, minute, second = parse_stninfo(None)
        assert year == 9999
        assert doy == 1
        assert hour == 0

    def test_invalid_format_too_few_parts(self):
        """Invalid format with too few parts raises exception."""
        with pytest.raises(pyDateException) as exc_info:
            parse_stninfo('2024 100')
        assert "Invalid stninfo date format" in str(exc_info.value)
        assert "got 2 parts" in str(exc_info.value)

    def test_hour_clamping(self):
        """Hours > 23 should be clamped to 23:59:59."""
        year, doy, hour, minute, second = parse_stninfo('2024 100 24 00 00')
        assert hour == 23
        assert minute == 59
        assert second == 59


class TestInvalidInputs:
    """Tests for invalid input handling."""

    def test_invalid_datetime_type(self):
        """Non-datetime type for datetime arg raises exception."""
        with pytest.raises(pyDateException):
            Date(datetime="not a datetime")

    def test_invalid_stninfo_type(self):
        """Invalid type for stninfo raises exception."""
        with pytest.raises(pyDateException):
            Date(stninfo=12345)

    def test_unrecognized_argument(self):
        """Unrecognized argument raises exception."""
        with pytest.raises(pyDateException) as exc_info:
            Date(invalid_arg=100)
        assert "unrecognized input arg" in str(exc_info.value)

    def test_insufficient_arguments(self):
        """Insufficient arguments to compute full date raises exception."""
        with pytest.raises(pyDateException) as exc_info:
            Date(year=2024)  # Missing doy or month/day
        assert "not enough independent input args" in str(exc_info.value)

    def test_invalid_doy(self):
        """Invalid day of year raises exception."""
        with pytest.raises(pyDateException):
            Date(year=2023, doy=366)  # 2023 is not a leap year

    def test_invalid_doy_zero(self):
        """Day of year 0 raises exception."""
        with pytest.raises(pyDateException):
            Date(year=2024, doy=0)


class TestComparisonOperators:
    """Tests for comparison operators."""

    def test_less_than(self):
        """Test < operator."""
        d1 = Date(year=2024, doy=100)
        d2 = Date(year=2024, doy=200)
        assert d1 < d2
        assert not d2 < d1

    def test_less_than_or_equal(self):
        """Test <= operator."""
        d1 = Date(year=2024, doy=100)
        d2 = Date(year=2024, doy=100)
        d3 = Date(year=2024, doy=200)
        assert d1 <= d2
        assert d1 <= d3
        assert not d3 <= d1

    def test_greater_than(self):
        """Test > operator."""
        d1 = Date(year=2024, doy=200)
        d2 = Date(year=2024, doy=100)
        assert d1 > d2
        assert not d2 > d1

    def test_greater_than_or_equal(self):
        """Test >= operator."""
        d1 = Date(year=2024, doy=100)
        d2 = Date(year=2024, doy=100)
        d3 = Date(year=2024, doy=50)
        assert d1 >= d2
        assert d1 >= d3
        assert not d3 >= d1

    def test_equal(self):
        """Test == operator."""
        d1 = Date(year=2024, doy=100)
        d2 = Date(year=2024, doy=100)
        assert d1 == d2

    def test_not_equal(self):
        """Test != operator."""
        d1 = Date(year=2024, doy=100)
        d2 = Date(year=2024, doy=101)
        assert d1 != d2

    def test_compare_with_datetime(self):
        """Compare Date with datetime object."""
        d = Date(year=2024, doy=100)  # April 9, 2024
        dt_before = datetime(2024, 4, 8)
        dt_after = datetime(2024, 4, 10)

        assert d > dt_before
        assert d < dt_after

    def test_compare_invalid_type(self):
        """Comparing with invalid type raises exception."""
        d = Date(year=2024, doy=100)
        with pytest.raises(pyDateException):
            d < "not a date"


class TestArithmeticOperations:
    """Tests for arithmetic operations."""

    def test_add_days(self):
        """Add days to a Date."""
        d = Date(year=2024, doy=100)
        d_plus_10 = d + 10
        assert d_plus_10.doy == 110 or (d_plus_10.doy == 110 - 366 + 1 and d_plus_10.year == 2025)
        # For 2024 (leap year), doy 100 + 10 = doy 110

    def test_add_days_crossing_year(self):
        """Add days crossing year boundary."""
        d = Date(year=2024, doy=360)
        d_plus_10 = d + 10
        # 2024 is leap year with 366 days, 360 + 10 = 370 = day 4 of 2025
        assert d_plus_10.year == 2025
        assert d_plus_10.doy == 4

    def test_subtract_days(self):
        """Subtract days from a Date."""
        d = Date(year=2024, doy=100)
        d_minus_10 = d - 10
        assert d_minus_10.doy == 90

    def test_subtract_date_from_date(self):
        """Subtract two Dates to get difference in days."""
        d1 = Date(year=2024, doy=100)
        d2 = Date(year=2024, doy=90)
        diff = d1 - d2
        assert diff == 10

    def test_add_invalid_type(self):
        """Adding non-integer raises exception."""
        d = Date(year=2024, doy=100)
        with pytest.raises(pyDateException):
            d + 1.5

    def test_subtract_invalid_type(self):
        """Subtracting invalid type raises exception."""
        d = Date(year=2024, doy=100)
        with pytest.raises(pyDateException):
            d - "invalid"


class TestDatetimeConversion:
    """Tests for datetime() method."""

    def test_normal_date(self):
        """Convert normal Date to datetime."""
        d = Date(year=2024, doy=100, hour=10, minute=30, second=45)
        dt = d.datetime()
        assert dt.year == 2024
        assert dt.month == 4
        assert dt.day == 9  # doy 100 in 2024
        assert dt.hour == 10
        assert dt.minute == 30
        assert dt.second == 45

    def test_year_9999(self):
        """datetime() for year 9999 returns datetime(9999, 1, 1)."""
        d = Date(year=9999, doy=1)
        dt = d.datetime()
        assert dt == datetime(9999, 1, 1, 0, 0, 0)

    def test_stninfo_none(self):
        """datetime() for stninfo=None returns far-future datetime."""
        d = Date(stninfo=None)
        dt = d.datetime()
        assert dt.year == 9999


class TestFirstLastEpoch:
    """Tests for first_epoch and last_epoch methods."""

    def test_first_epoch_datetime_str(self):
        """first_epoch with datetime_str format."""
        d = Date(year=2024, doy=100, hour=12)
        result = d.first_epoch(out_format='datetime_str')
        assert result == '2024-04-09 00:00:00'

    def test_first_epoch_fyear(self):
        """first_epoch with fyear format."""
        d = Date(year=2024, doy=100)
        result = d.first_epoch(out_format='fyear')
        assert isinstance(result, float)
        # Should be fractional year at start of day (hour=0)

    def test_first_epoch_datetime(self):
        """first_epoch with datetime format."""
        d = Date(year=2024, doy=100, hour=12)
        result = d.first_epoch(out_format='datetime')
        assert isinstance(result, datetime)
        assert result.hour == 0
        assert result.minute == 0

    def test_last_epoch_datetime_str(self):
        """last_epoch with datetime_str format."""
        d = Date(year=2024, doy=100, hour=12)
        result = d.last_epoch(out_format='datetime_str')
        assert result == '2024-04-09 23:59:59'

    def test_last_epoch_datetime(self):
        """last_epoch with datetime format."""
        d = Date(year=2024, doy=100)
        result = d.last_epoch(out_format='datetime')
        assert result.hour == 23
        assert result.minute == 59
        assert result.second == 59

    def test_first_epoch_year_9999(self):
        """first_epoch for year 9999."""
        d = Date(year=9999, doy=1)

        assert d.first_epoch(out_format='datetime_str') == '9999-01-01 00:00:00'
        assert d.first_epoch(out_format='fyear') == 9999.0
        assert d.first_epoch(out_format='datetime') == datetime(9999, 1, 1, 0, 0, 0)

    def test_last_epoch_year_9999(self):
        """last_epoch for year 9999."""
        d = Date(year=9999, doy=1)

        assert d.last_epoch(out_format='datetime_str') == '9999-12-31 23:59:59'
        assert d.last_epoch(out_format='fyear') == 9999.999
        assert d.last_epoch(out_format='datetime') == datetime(9999, 12, 31, 23, 59, 59)


class TestStringRepresentation:
    """Tests for string representations and formatting methods."""

    def test_str_normal(self):
        """__str__ for normal date."""
        d = Date(year=2024, doy=100, hour=12, minute=30, second=45)
        assert str(d) == '2024 100 12 30 45'

    def test_str_padding(self):
        """__str__ pads values correctly."""
        d = Date(year=2024, doy=5, hour=1, minute=2, second=3)
        assert str(d) == '2024 005 01 02 03'

    def test_repr(self):
        """__repr__ format."""
        d = Date(year=2024, doy=100)
        assert repr(d) == 'pyDate.Date(2024,100)'

    def test_ddd(self):
        """ddd() returns zero-padded day of year."""
        d = Date(year=2024, doy=5)
        assert d.ddd() == '005'
        d2 = Date(year=2024, doy=100)
        assert d2.ddd() == '100'

    def test_yyyy(self):
        """yyyy() returns 4-digit year."""
        d = Date(year=2024, doy=100)
        assert d.yyyy() == '2024'

    def test_wwww(self):
        """wwww() returns GPS week."""
        d = Date(year=2024, doy=100)
        assert len(d.wwww()) == 4
        assert d.wwww().isdigit()

    def test_wwwwd(self):
        """wwwwd() returns GPS week + day."""
        d = Date(year=2024, doy=100)
        result = d.wwwwd()
        assert len(result) == 5

    def test_yyyymmdd(self):
        """yyyymmdd() format."""
        d = Date(year=2024, month=4, day=9)
        assert d.yyyymmdd() == '2024/4/9'

    def test_yyyyddd_with_space(self):
        """yyyyddd() with space."""
        d = Date(year=2024, doy=100)
        assert d.yyyyddd(space=True) == '2024 100'

    def test_yyyyddd_without_space(self):
        """yyyyddd() without space."""
        d = Date(year=2024, doy=100)
        assert d.yyyyddd(space=False) == '2024100'

    def test_iso_date(self):
        """iso_date() format."""
        d = Date(year=2024, month=4, day=9)
        assert d.iso_date() == '2024-04-09'

    def test_strftime(self):
        """strftime() method."""
        d = Date(year=2024, doy=100, hour=12, minute=30, second=45)
        assert d.strftime() == '2024-04-09 12:30:45'


class TestJSONSerialization:
    """Tests for JSON serialization."""

    def test_to_json_normal(self):
        """to_json for normal date."""
        d = Date(year=2024, doy=100, hour=12, minute=30, second=45)
        json_data = d.to_json()
        assert json_data['year'] == 2024
        assert json_data['doy'] == 100
        assert json_data['hour'] == 12
        assert json_data['minute'] == 30
        assert json_data['second'] == 45

    def test_to_json_stninfo(self):
        """to_json for stninfo date."""
        d = Date(stninfo='2024 100 12 30 45')
        json_data = d.to_json()
        assert 'stninfo' in json_data
        assert json_data['stninfo'] == str(d)


class TestHashability:
    """Tests for hashability."""

    def test_hash(self):
        """Date objects are hashable."""
        d1 = Date(year=2024, doy=100)
        d2 = Date(year=2024, doy=100)

        # Should be able to use in sets and dicts
        date_set = {d1, d2}
        # Two equal dates should hash to same value
        assert hash(d1) == hash(d2)

    def test_in_dict(self):
        """Date can be used as dictionary key."""
        d = Date(year=2024, doy=100)
        data = {d: "test value"}
        assert data[d] == "test value"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_yeardoy2fyear(self):
        """yeardoy2fyear computes fractional year."""
        fyear = yeardoy2fyear(2024, 1)  # First day
        assert 2024.0 <= fyear < 2024.01

        fyear_mid = yeardoy2fyear(2024, 183)  # Middle of year
        assert 2024.49 < fyear_mid < 2024.51

    def test_yeardoy2fyear_invalid_doy(self):
        """yeardoy2fyear raises on invalid doy."""
        with pytest.raises(pyDateException):
            yeardoy2fyear(2024, 0)
        with pytest.raises(pyDateException):
            yeardoy2fyear(2024, 367)  # Leap year has max 366

    def test_fyear2yeardoy(self):
        """fyear2yeardoy converts fractional year back."""
        year, doy, hour, minute, second = fyear2yeardoy(2024.0)
        assert year == 2024
        assert doy == 1

    def test_date2doy(self):
        """date2doy computes day of year."""
        doy, fyear = date2doy(2024, 4, 9)  # April 9 in leap year
        assert doy == 100

    def test_doy2date(self):
        """doy2date computes month and day."""
        month, day = doy2date(2024, 100)  # doy 100 in leap year
        assert month == 4
        assert day == 9

    def test_doy2date_invalid(self):
        """doy2date raises on invalid doy."""
        with pytest.raises(pyDateException):
            doy2date(2023, 366)  # 2023 is not leap year

    def test_date2gpsDate(self):
        """date2gpsDate computes GPS week and day."""
        gps_week, gps_day = date2gpsDate(2024, 4, 9)
        assert isinstance(gps_week, int)
        assert 0 <= gps_day <= 6

    def test_gpsDate2mjd(self):
        """gpsDate2mjd computes MJD from GPS date."""
        mjd = gpsDate2mjd(2350, 0)
        assert isinstance(mjd, int)

    def test_mjd2date(self):
        """mjd2date converts MJD back to date."""
        year, month, day = mjd2date(60000)
        assert year == 2023
        assert month == 2
        assert day == 25


class TestEdgeCases:
    """Tests for edge cases."""

    def test_leap_year_feb_29(self):
        """February 29 in leap year."""
        d = Date(year=2024, month=2, day=29)
        assert d.doy == 60  # Feb 29 is day 60 in leap year

    def test_last_day_of_year_leap(self):
        """Day 366 in leap year."""
        d = Date(year=2024, doy=366)
        assert d.month == 12
        assert d.day == 31

    def test_last_day_of_year_non_leap(self):
        """Day 365 in non-leap year."""
        d = Date(year=2023, doy=365)
        assert d.month == 12
        assert d.day == 31

    def test_first_day_of_year(self):
        """Day 1 of year."""
        d = Date(year=2024, doy=1)
        assert d.month == 1
        assert d.day == 1

    def test_stninfo_datetime_input(self):
        """stninfo parameter with datetime object."""
        dt = datetime(2024, 4, 9, 12, 30, 45)
        d = Date(stninfo=dt)
        assert d.year == 2024
        assert d.month == 4
        assert d.day == 9
        assert d.from_stninfo is True

    def test_stninfo_date_input(self):
        """stninfo parameter with Date object."""
        d1 = Date(year=2024, doy=100, hour=12, minute=30, second=45)
        d2 = Date(stninfo=d1)
        assert d2.year == 2024
        assert d2.doy == 100
        assert d2.hour == 12


class TestRoundTrips:
    """Tests for round-trip conversions."""

    def test_year_doy_to_datetime_and_back(self):
        """Year/doy to datetime and back."""
        d1 = Date(year=2024, doy=100, hour=12, minute=30, second=45)
        dt = d1.datetime()
        d2 = Date(datetime=dt)
        assert d2.year == d1.year
        assert d2.month == d1.month
        assert d2.day == d1.day
        assert d2.hour == d1.hour
        assert d2.minute == d1.minute
        assert d2.second == d1.second

    def test_mjd_round_trip(self):
        """MJD to date and back."""
        d1 = Date(year=2024, doy=100)
        mjd = d1.mjd
        d2 = Date(mjd=mjd)
        assert d2.year == d1.year
        assert d2.month == d1.month
        assert d2.day == d1.day

    def test_gps_week_round_trip(self):
        """GPS week to date and back."""
        d1 = Date(year=2024, doy=100)
        gps_week = d1.gpsWeek
        gps_day = d1.gpsWeekDay
        d2 = Date(gpsWeek=gps_week, gpsWeekDay=gps_day)
        assert d2.year == d1.year
        assert d2.month == d1.month
        assert d2.day == d1.day

    def test_stninfo_string_round_trip(self):
        """stninfo string to Date and back to string."""
        original = '2024 100 12 30 45'
        d = Date(stninfo=original)
        result = str(d)
        assert result == original
