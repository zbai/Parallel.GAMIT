"""
The testing script to help debug functions and classes.  The classes are split up into the script that they call.
"""
import unittest
import GAMITArchive
from bin.ArchiveService import *


class ArchiveServiceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config_file = 'debug.cfg'
        self.cnn = GAMITArchive.Connection(self.config_file)
        self.stationcode = 'test'

    def test_insert_station_w_lock(self):
        # insert_station_w_lock(self.cnn, self.stationcode, filename, lat, lon, h, x, y, z, otl)
        self.fail()

    def test_test_callback_handle(self):
        self.fail()

    def test_check_rinex_timespan_int(self):
        self.fail()

    def test_write_error(self):
        self.fail()

    def test_error_handle(self):
        self.fail()

    def test_insert_data(self):
        self.fail()

    def test_verify_rinex_multiday(self):
        self.fail()

    def test_process_crinex_file(self):
        self.fail()

    def test_duplicate_archive(self):
        self.fail()

    def test_zmain(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()
