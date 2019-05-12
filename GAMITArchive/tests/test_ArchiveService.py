"""
The testing script to help debug functions and classes.  The classes are split up into the script that they call.
GAMITArchive is implicitly tested via the ArchiveService routines.
"""
import unittest
from bin.ArchiveService import *
from psycopg2 import sql
import shutil
import os
import argparse
from collections import defaultdict
import datetime as dt


def dict2sqlresult(record: dict) -> defaultdict:
    result = defaultdict(list)
    for k, v in record.items():
        result[k].append(v)
    return result


class ArchiveServiceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config_file = 'debug.cfg'
        self.cnn = GAMITArchive.Connection(self.config_file)
        self.assertIn('gnss_data_debug', self.cnn.conn.dsn)
        # Clear the debug database.
        delete_tables = [sql.Identifier('rinex'),
                         sql.Identifier('stations'),
                         sql.Identifier('locks'),
                         sql.Identifier('ppp_soln'),
                         sql.Identifier('executions')]
        with self.cnn.conn.cursor() as curs:
            for delete_stmt in delete_tables:
                delete_all = sql.SQL('DELETE FROM {}').format(delete_stmt)
                curs.execute(delete_all)
        self.cnn.conn.commit()
        self.stationcode = 'test'
        self.working_dir = os.getcwd()
        self.test_files = os.path.join(os.getcwd(), 'test_files')
        clearout_folders = ['repository', 'archive']
        for fldr in clearout_folders:
            shutil.rmtree(os.path.join(self.test_files, fldr))
        os.makedirs(os.path.join(self.test_files, 'repository'))
        os.makedirs(os.path.join(self.test_files, 'repository', 'data_in'))
        os.makedirs(os.path.join(self.test_files, 'archive'))
        self.parser = argparse.ArgumentParser(description='Archive operations Main Program')
        self.parser.add_argument('-purge', '--purge_locks', action='store_true',
                                 help="Delete any network starting with '?' from the stations "
                                      "table and purge the contents of the locks table, deleting "
                                      "the associated files from data_in.")

        self.parser.add_argument('-dup', '--duplicate', type=str,
                                 help='Duplicate the archive as it is seen by the database')
        self.parser.add_argument('-config', '--config_file', type=str, default='gnss_data.cfg',
                                 help='Specify the config file, defaults to gnss_data.cfg in the current directory')

    def tearDown(self) -> None:
        """
        Placeholder so I can write notes on why I'm not doing some stuff.
        :return:
        """
        # Don't delete everything in the database in case we want to check the output.
        print('Test complete')

    def valid_executions(self):
        executions_entry = defaultdict(list)
        executions_ = {'script': 'ArchiveService.py'}
        for k, v in executions_.items():
            executions_entry[k].append(v)

        executions_check = sql.SQL('SELECT {} FROM {}').format(sql.Identifier('script'),
                                                               sql.Identifier('executions'))
        executions_tested = self.cnn._execute_wrapper(executions_check, retval=True, return_dict=True)
        self.assertEqual(executions_entry, executions_tested)

    def valid_stations(self, stations_entry):
        """
        :param stations_entry:
        :return:
        """
        stations_cols = [sql.Identifier(s) for s in stations_entry.keys()]
        stations_check = sql.SQL('SELECT {} FROM {}').format(sql.SQL(', ').join(stations_cols),
                                                             sql.Identifier('stations'))
        stations_tested = self.cnn._execute_wrapper(stations_check, retval=True, return_dict=True)
        self.assertEqual(stations_entry, stations_tested)

    def valid_rinex(self, rinex_entry):
        rinex_cols = [sql.Identifier(s) for s in rinex_entry.keys()]
        rinex_check = sql.SQL('SELECT {} FROM {}').format(sql.SQL(', ').join(rinex_cols),
                                                          sql.Identifier('rinex'))
        rinex_tested = self.cnn._execute_wrapper(rinex_check, retval=True, return_dict=True)
        self.assertEqual(rinex_entry, rinex_tested)

    def valid_locks(self, locks_entry):
        locks_check = sql.SQL('SELECT * FROM {}').format(sql.Identifier('locks'))
        locks_tested = self.cnn._execute_wrapper(locks_check, retval=True, return_dict=True)
        self.assertEqual(locks_entry, locks_tested)

    def test_valid_rinex(self):
        """
        Make sure that good RINEX makes its way into the database.
        :return:
        """
        # Build the query results (either to insert into the database or to check against the result).
        stations_ = {'NetworkCode': 'bug', 'StationCode': 'test', 'StationName': None, 'DateStart': None,
                     'DateEnd': None, 'auto_x': -358696.4095, 'auto_y': -1918978.7944, 'auto_z': -6051724.6903,
                     'lat': -72.23305179, 'lon': -100.58757474, 'height': 55.174, 'max_dist': None,
                     'dome': None}

        rinex_a = {'NetworkCode': 'bug', 'StationCode': 'test', 'ObservationYear': 2019, 'ObservationMonth': 1,
                   'ObservationDay': 3, 'ObservationDOY': 3, 'ObservationFYear': 2019.0068,
                   'ObservationSTime': dt.datetime(2019, 1, 3, 0, 0, 0, 0),
                   'ObservationETime': dt.datetime(2019, 1, 3, 23, 59, 30, 0), 'ReceiverType': 'ALERTGEO RESOLUTE',
                   'ReceiverSerial': '0206', 'ReceiverFw': '4878', 'AntennaType': 'SEPPOLANT_X_MF',
                   'AntennaSerial': '13998', 'AntennaDome': 'NONE', 'Filename': 'test0030.19o', 'Interval': 30,
                   'AntennaOffset': 0.3048, 'Completion': 1.0}
        rinex_entry = dict2sqlresult(rinex_a)

        # Move the test file into data_in
        shutil.copyfile(os.path.join(self.test_files, 'test0030.19d.Z'),
                        os.path.join(self.test_files, 'repository/data_in', 'test0030.19d.Z'))

        # Add the test station to the database and assign it to the 'bug' network.
        self.cnn.insert('stations', stations_)

        # Set up the arguments and run the main script.
        args = self.parser.parse_args(['-config', 'debug.cfg'])
        main(args)

        # Start testing the output that is saved in the tables. (executions, rinex)
        self.valid_executions()
        self.valid_rinex(rinex_entry)

        # TODO: Make sure the file is moved into the archive correctly.

    def test_locking(self):
        """
        Check that the locks work correctly.  There should be an entry in the locks table and an entry in the stations
        table.
        :return:
        """
        # TODO: Figure out a way to check the HARPOS output since it has a timestamp in it.

        locks_ = {'StationCode': 'test', 'NetworkCode': '???', 'filename': 'test0030.19d.Z'}

        stations_ = {'NetworkCode': '???', 'StationCode': 'test', 'StationName': None, 'DateStart': None,
                     'DateEnd': None, 'auto_x': -358696.4095, 'auto_y': -1918978.7944, 'auto_z': -6051724.6903,
                     'lat': -72.23305179, 'lon': -100.58757474, 'height': 55.174, 'max_dist': None,
                     'dome': None}

        locks_entry = dict2sqlresult(locks_)
        stations_entry = dict2sqlresult(stations_)

        shutil.copyfile(os.path.join(self.test_files, 'test0030.19d.Z'),
                        os.path.join(self.test_files, 'repository/data_in', 'test0030.19d.Z'))
        args = self.parser.parse_args(['-config', 'debug.cfg'])
        main(args)

        # As long as the program finished correctly we can now check out the table entries for consistancy.

        self.valid_executions()
        self.valid_locks(locks_entry)
        self.valid_stations(stations_entry)


if __name__ == '__main__':
    unittest.main()
