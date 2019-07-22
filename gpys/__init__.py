"""
Contains a few classes that make organizing the executable scripts easier.  Most repetative tasks will get migrated into
this module.
TODO: Add an executable version checker. I suspect that different compilers will give different PPP results.
TODO: Add docstrings around try blocks.
"""
import configparser
import logging
import os
import re
import socket
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import dispy
import psycopg2
from psycopg2 import sql

logger = logging.getLogger('gpys')


class Connection:
    """
    Ideally this class is to be used as a direct connection to the gnss_data database.  As such it should initiate to
    a pgdb.Connection object through the uses of pgdb.connect().  Once the object has established a connection to the
    database, class methods should be used to interact with the database.  The goal is to abstract all the SQL commands
    so we can deal with the errors here instead of out in the wild.
    """

    __slots__ = ['conn',
                 'logger']

    def __init__(self, config, parent_logger: str = 'archive'):

        # Open connection to server
        connect_dsn = None
        try:
            self.logger = logging.getLogger('.'.join([parent_logger, 'gpys.Connection']))
            self.logger.debug('Initializing Connection object.')
            self.conn = None
            connect_dsn = 'dbname={} host={} user={} password={}'.format(config['database'],
                                                                         config['hostname'],
                                                                         config['username'],
                                                                         config['password'])
            self.conn = psycopg2.connect(connect_dsn)
            self.logger.debug(f'Connection established: {self.conn.dsn}')
            self.insert('executions', {'script': parent_logger})
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')
        finally:
            del connect_dsn, config, parent_logger

    def __del__(self):
        try:
            if self.conn:
                self.conn.close()
                self.logger.debug('Connection closed: {}'.format(self.conn.dsn))
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')

    def _execute_wrapper(self, sql_statement: sql.Composed = None,
                         values: list = None,
                         retval: bool = False,
                         return_dict: bool = True):
        """
        Deal with all the actual database interactions here and deal with the related error possibilities.
        TODO: Add how many rows were deleted/updated/inserted to logger.
        :param sql_statement: A composable object.
        :param values: List or tuple of values for the sql statement.
        :param retval: Whether to return a value or not.
        :param return_dict: Return it as a dictionary or not.
        :return: Returns either a list of tuples containing the results or a dictionary or None.
        """
        try:
            with self.conn.cursor() as curs:
                if values is not None:
                    curs.execute(sql_statement, values)
                else:
                    curs.execute(sql_statement)
                self.logger.debug(f'{curs.query}')
                self.logger.debug(f'{curs.statusmessage}')
                if retval:
                    if return_dict:
                        qresults = curs.fetchall()
                        if not qresults:
                            return None
                        keys = [name[0] for name in curs.description]
                        d = defaultdict(list)
                        for n, key in enumerate(keys):
                            for rec in qresults:
                                if type(rec[n]) != Decimal:
                                    d[key].append(rec[n])
                                else:
                                    d[key].append(float(rec[n]))
                        return d
                    else:
                        return curs.fetchall()
            self.conn.commit()
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')

    def insert(self, table: str, record: dict):
        try:
            x = sql.SQL('{}').format(sql.Identifier(table))
            y = sql.SQL(', ').join([sql.Identifier(key) for key in record.keys()])
            z = sql.SQL(', ').join(sql.Placeholder() * len(record))
            insert_statement = sql.SQL("INSERT INTO {0} ({1}) VALUES ({2});").format(x, y, z)
            self._execute_wrapper(insert_statement, [v for v in record.values()])
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')

    def update_locks(self):
        try:
            clear_statement = sql.SQL("DELETE FROM locks WHERE {} NOT LIKE {};").format(sql.Identifier('NetworkCode'),
                                                                                        sql.Placeholder())
            self._execute_wrapper(clear_statement, ['?%'])
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')

    def load_table(self, table: str = None, columns: list = None):
        try:
            if columns in [None]:
                select_statement = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
            else:
                sqlcols = sql.SQL(', ').join([sql.Identifier(x) for x in columns])
                select_statement = sql.SQL("SELECT {} FROM {}").format(sqlcols, sql.Identifier(table))
            return self._execute_wrapper(select_statement, retval=True)
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')
            return None

    def load_tankstruct(self):
        """
        Determines the archive structure based on the two tables, rinex_tank_struct and keys.  Returns a dictionary with
        the following keys:

        **KeyCode**: (*list(str)*) Property of a file e.g. 'network' or 'doy'.

        **Level**: (*list(int)*) The heirachy for the properties, the first entry will be the highest level in the
        archive.

        **TotalChars**: (*list(int)*) The number of characters in the keycode.

        .. todo:: Finish adding all the different columns returned by this function, rinex_col_out, rinex_col_in and
        isnumeric.

        :return:
        """
        try:
            sql_statement = sql.SQL('SELECT * FROM {0} INNER JOIN {1} '
                                    'USING ({2}) ORDER BY {3}').format(sql.Identifier('rinex_tank_struct'),
                                                                       sql.Identifier('keys'),
                                                                       sql.Identifier('KeyCode'),
                                                                       sql.Identifier('Level'))
            return self._execute_wrapper(sql_statement, retval=True)
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')
            return None

    def insert_event(self, event):
        try:
            event_dict = event.db_dict()
            y = sql.SQL(', ').join([sql.Identifier(key) for key in event_dict.keys()])
            z = sql.SQL(', ').join(sql.Placeholder() * len(event_dict))
            insert_statement = sql.SQL("insert into {0} ({1}) VALUES ({2});").format(sql.Identifier('events'), y, z)
            self._execute_wrapper(insert_statement, [v for v in event_dict.values()])
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')

    def print_summary(self, script):
        try:
            script_start = sql.SQL('SELECT MAX({}) AS mx FROM {} WHERE {} = {}').format(sql.Identifier('exec_date'),
                                                                                        sql.Identifier('executions'),
                                                                                        sql.Identifier('script'),
                                                                                        sql.Placeholder())
            st = self._execute_wrapper(script_start, [script], retval=True)
            counter = sql.SQL(
                'SELECT COUNT(*) AS cc FROM {0} WHERE {1} >= {2} AND {3} = {2}').format(sql.Identifier('events'),
                                                                                        sql.Identifier('EventDate'),
                                                                                        sql.Placeholder(),
                                                                                        sql.Identifier('EventType'))
            info = self._execute_wrapper(counter, [st[0][0], 'info'], retval=True)
            erro = self._execute_wrapper(counter, [st[0][0], 'error'], retval=True)
            warn = self._execute_wrapper(counter, [st[0][0], 'warning'], retval=True)
            self.logger.info(' >> Summary of events for this run:')
            self.logger.info(f' -- info    : {info[0][0]}')
            self.logger.info(f' -- errors  : {erro[0][0]}')
            self.logger.info(f' -- warnings: {warn[0][0]}')
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')

    def spatial_check(self, vals, search_in_new: bool = False):
        """
        Used to find the nearest station to a given RINEX file.  It only goes out to a range of 20 meters or the value
        listed in the max_dist column of the stations table.  It will return None if there are no matching stations
        and if there is a match a defaultdict(list) object containing every column from the stations table in addition
        to the distance between the RINEX and the given stations entry.  The distance calculation is performed using the
        haversine formula.
        :param vals: List with the lattitude as the first element and the longitude as the second.
        :param search_in_new: Whether to search in networks matching ??? or not.
        :return: defaultdict(list) with the following keys, NetworkCode, StationCode, StationName, DateStart, DateEnd,
        auto_x, auto_y, auto_z, Harpos_coeff_otl, lat, lon, height, max_dist, dome, distance.
        """
        try:
            if len(vals) != 2:
                raise Exception('Incorrect length of values, should be of the format [lat, lon]')
            if not search_in_new:
                where_clause = sql.SQL('WHERE {} NOT LIKE {}').format(sql.Identifier('NetworkCode'),
                                                                      sql.Literal('?%%'))
            else:
                where_clause = sql.SQL('')
            sql_select = sql.SQL(
                'SELECT {0} FROM (SELECT *, 2*ASIN(SQRT(SIN((RADIANS({1})-RADIANS({2}))/2)^2 + COS(RADIANS({2}))'
                '* COS(RADIANS({1})) * SIN((RADIANS({1}) - RADIANS({3}))/2)^2))*6371000 AS distance FROM '
                '{4} {5}) AS st1 LEFT JOIN {4} AS st2 ON st1.{6} = st2.{6} AND st1.{7} = st2.{7} AND '
                'st1.distance < COALESCE(st2.{8}, 20) WHERE st2.{7} IS NOT NULL').format(
                sql.SQL(', ').join([sql.SQL('st1.{}').format(sql.Identifier('NetworkCode')),
                                    sql.SQL('st1.{}').format(sql.Identifier('StationCode')),
                                    sql.SQL('st1.{}').format(sql.Identifier('StationName')),
                                    sql.SQL('st1.{}').format(sql.Identifier('DateStart')),
                                    sql.SQL('st1.{}').format(sql.Identifier('DateEnd')),
                                    sql.SQL('st1.{}').format(sql.Identifier('auto_x')),
                                    sql.SQL('st1.{}').format(sql.Identifier('auto_y')),
                                    sql.SQL('st1.{}').format(sql.Identifier('auto_z')),
                                    sql.SQL('st1.{}').format(sql.Identifier('Harpos_coeff_otl')),
                                    sql.SQL('st1.{}').format(sql.Identifier('lat')),
                                    sql.SQL('st1.{}').format(sql.Identifier('lon')),
                                    sql.SQL('st1.{}').format(sql.Identifier('height')),
                                    sql.SQL('st1.{}').format(sql.Identifier('max_dist')),
                                    sql.SQL('st1.{}').format(sql.Identifier('dome')),
                                    sql.SQL('st1.distance')]),
                sql.Placeholder(),
                sql.Identifier('lat'),
                sql.Identifier('lon'),
                sql.Identifier('stations'),
                where_clause,
                sql.Identifier('StationCode'),
                sql.Identifier('NetworkCode'),
                sql.Identifier('max_dist'))
            return self._execute_wrapper(sql_select, [vals[0], vals[0], vals[1]], retval=True)
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')
            return None

    def nearest_station(self, vals, search_in_new: bool = False):
        """
        Sorts all stations by  distance from the given [lat, lon] in the vals variable, always returns a station unless
        the database is empty.
        :param vals: list with format [lattitude, longitude] in decimal degrees
        :param search_in_new: Whether to also search stations that are not yet assigned to a network.
        :return: A single entry from the stations table of the database.
        TODO: Make sure that it returns just one entry.
        """
        try:
            if len(vals) != 2:
                raise Exception('Incorrect length of values, should be of the format [lat, lon]')
            if not search_in_new:
                where_clause = sql.SQL('WHERE {} NOT LIKE {}').format(sql.Identifier('NetworkCode'),
                                                                      sql.Literal('?%%'))
            else:
                where_clause = sql.SQL('')
            sql_select = sql.SQL(
                'SELECT * FROM (SELECT *, 2*ASIN(SQRT(SIN((RADIANS({0})-RADIANS({1}))/2)^2 + COS(RADIANS({1}))'
                '* COS(RADIANS({0})) * SIN((RADIANS({0}) - RADIANS({2}))/2)^2))*6371000 AS distance FROM '
                '{3} {4}) AS dd ORDER BY distance').format(
                sql.Placeholder(),
                sql.Identifier('lat'),
                sql.Identifier('lon'),
                sql.Identifier('stations'),
                where_clause)
            return self._execute_wrapper(sql_select, [vals[0], vals[0], vals[1]], retval=True)
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')
            return None

    def similar_locked(self, vals):
        try:
            sql_select = sql.SQL(
                'SELECT * FROM (SELECT *, 2*ASIN(SQRT(SIN((RADIANS({0})-RADIANS({1}))/2)^2 + COS(RADIANS({1}))'
                '* COS(RADIANS({0})) * SIN((RADIANS({0}) - RADIANS({2}))/2)^2))*6371000 AS distance FROM '
                '{3} WHERE {4} LIKE {5} AND {6} LIKE {0}) AS dd WHERE distance <= 100').format(
                sql.Placeholder(),
                sql.Identifier('lat'),
                sql.Identifier('lon'),
                sql.Identifier('stations'),
                sql.Identifier('NetworkCode'),
                sql.Literal('?%%'),
                sql.Identifier('StationCode'))
            return self._execute_wrapper(sql_select, vals, retval=True)
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')
            return None

    def update(self, table: str = None, row: dict = None, record: dict = None):
        try:
            a = sql.SQL('{}').format(sql.Identifier(table))
            b = sql.SQL(', ').join([sql.Identifier(key) for key in record.keys()])
            c = sql.SQL(', ').join(sql.Placeholder() * len(record))
            d = sql.SQL(', ').join([sql.Identifier(key) for key in row.keys()])
            e = sql.SQL(', ').join(sql.Placeholder() * len(row))
            insert_statement = sql.SQL("UPDATE {0} SET ({1}) = ({2}) WHERE ({3}) LIKE ({4})").format(a, b, c, d, e)
            vals = []
            for v in record.values():
                vals.append(v)
            for v in row.values():
                vals.append(v)
            self._execute_wrapper(insert_statement, vals)
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')

    def load_table_matching(self, table: str = None, where_dict: dict = None):
        try:
            select_statement = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
            where_statement = sql.SQL("WHERE")
            if len(where_dict) > 1:
                like_statement = sql.SQL(' AND ').join(
                    [sql.SQL(f'{sql.Identifier(k)} = {sql.Placeholder()}') for k in where_dict.keys()])
            else:
                like_statement = [sql.SQL(f'{sql.Identifier(k)} LIKE {sql.Placeholder()}') for k in where_dict.keys()]
                like_statement = like_statement[0]

            full_statement = sql.SQL(' ').join([select_statement, where_statement, like_statement])
            return self._execute_wrapper(full_statement, [v for v in where_dict.values()], retval=True)
        except Exception as e:
            self.logger.error(f'Uncaught Exception {type(e)} {e}')
            return None


class ReadOptions:
    """
    Class that deals with reading in the default configuration file gnss_data.cfg
    TODO: Clean up some of the uneccesary parts.
    """

    def __init__(self, configfile: str = 'gnss_data.cfg', parent_logger: str = 'archive'):
        """-------------------------------------------------------------------------------------------------------------
        Initialize the logger.
        TODO: Add an option for which QC program to use (TEQC or RinSum)
        """
        rologger = logging.getLogger('.'.join([parent_logger, 'gpys.ReadOptions']))
        rologger.debug('Initializing ReadOptions object from {}'.format(configfile))
        conn = None
        self.options = {'path': None,
                        'repository': None,
                        'parallel': False,
                        'node_list': None,
                        'brdc': None,
                        'sp3_type_1': None,
                        'sp3_type_2': None,
                        'sp3_type_3': None,
                        'sp3_altr_1': None,
                        'sp3_altr_2': None,
                        'sp3_altr_3': None,
                        'grdtab': None,
                        'otlgrid': None,
                        'otlmodel': 'FES2014b',
                        'ppp_path': None,
                        'institution': None,
                        'info': None,
                        'sp3': None,
                        'frames': None,
                        'atx': None,
                        'height_codes': None,
                        'ppp_exe': None,
                        'ppp_remote_local': (),
                        'gg': None,
                        'uppper_bound': 10000,
                        'lower_bound': 9000}
        """
        Read in the config file.
        Parse the sections into the options dict.
        Check that paths exist.
        
        """
        fp, p, config, section = None, None, None, None
        try:
            rologger.debug(f'Checking config_file: {configfile}')
            config = configparser.ConfigParser()
            with open(configfile) as fp:
                config.read_file(fp)
            for section in config.sections():
                for key in config[section]:
                    self.options[key] = config[section][key]
            rologger.debug(f'{configfile} read into program.')
        except FileNotFoundError as e:
            rologger.error(f'FileNotFoundError: {e}')
            sys.exit(1)
        except Exception as e:
            rologger.error(f'Uncaught Exception {type(e)} {e}')
            sys.exit(1)
        finally:
            del fp, p, section
        rologger.debug('Parent directories are okay, some of the files might not exist though.')
        """-------------------------------------------------------------------------------------------------------------
        Frames and dates
        TODO: Implement the frames.
        """
        frame, atx = None, None
        try:
            rologger.debug('Building the reference frames.')
            rologger.debug('Reference frames built.')
        except KeyError:
            rologger.error('The frames were not correctly defined in the config file {}'.format(configfile),
                           exc_info=sys.exc_info())
            sys.exit(1)
        except Exception as e:
            rologger.error(f'Uncaught Exception {type(e)} {e}')
            sys.exit(1)
        finally:
            del frame, atx
        """-------------------------------------------------------------------------------------------------------------
        Assign variables based on options.
        Alternative sp3 types
        """
        try:
            rologger.debug('Creating some properties for ease of use.')
            self.data_in = Path(self.options['repository']) / Path('data_in')
            self.data_in_retry = Path(self.options['repository']) / Path('data_in_retry')
            self.data_reject = Path(self.options['repository']) / Path('data_rejected')
            os.makedirs(self.data_in, exist_ok=True)
            os.makedirs(self.data_in_retry, exist_ok=True)
            os.makedirs(self.data_reject, exist_ok=True)
            self.sp3types = [self.options['sp3_type_1'], self.options['sp3_type_2'], self.options['sp3_type_3']]
            self.sp3types = [sp3type for sp3type in self.sp3types if sp3type is not None]
            if self.options['parallel'] == 'True':
                self.options['parallel'] = True
            else:
                self.options['parallel'] = False
            self.options['node_list'] = self.options['node_list'].strip(' ').split(',')
        except KeyError as e:
            rologger.error(e)
            sys.exit(1)
        except OSError as e:
            rologger.error(e)
            sys.exit(1)
        except Exception as e:
            rologger.error(f'Uncaught Exception {type(e)} {e}')
            sys.exit(1)
        if self.options['parallel']:
            rologger.debug('Testing JobServer connection.')
            try:
                JobServer(self.options).cluster_test()
            except Exception as e:
                rologger.error(f'Uncaught Exception {type(e)} {e}')
                sys.exit(1)
            rologger.debug('JobServer connected.')
        else:
            rologger.debug('Running in serial.')
        """-------------------------------------------------------------------------------------------------------------
        PostgreSQL connection.
        """
        rologger.debug('Check out the database connection.')
        try:
            conn = Connection(self.options, parent_logger=parent_logger)
            rinex_struct = conn.load_tankstruct()
            self.rinex_struct = rinex_struct['KeyCode']
            rologger.debug(f'Loaded archive structure: {self.rinex_struct}')
        except Exception as e:
            rologger.error(f'Uncaught Exception {type(e)} {e}')
            sys.exit(1)
        finally:
            del conn
        rologger.debug('Database connection established.')
        rologger.debug('Config sucessfully read in.')

    def scan_archive_struct(self, rootdir = None) -> list:
        """
        Recursive member method of RinexArcvhive that searches through the given rootdir
        to find files matching a compressed rinex file e.g. ending with d.Z.  The method
        self.scan_archive_struct() is used to determine the file type.
        :param rootdir:
        :return:
        """
        try:
            if rootdir in [None]:
                rootdir = self.options['repository']
            file = []
            with os.scandir(rootdir) as it:
                for entry in it:
                    entry = Path(entry.path)
                    if entry.is_dir():
                        file.extend(self.scan_archive_struct(entry.path))
                    # DDG issue #15: match the name of the file to a valid rinex filename
                    elif self.parse_crinex_filename(entry.name):
                        # only add valid rinex compressed files
                        file.append(entry)
            return file
        except Exception as e:
            print(f'Uncaught Exception: {type(e)} {e}')

    @staticmethod
    def parse_crinex_filename(filename):
        # parse a crinex filename
        try:
            sfile = re.findall(r'(\w{4})(\d{3})(\w{1})\.(\d{2})([d])\.[Z]$', filename)
            if sfile:
                return sfile[0]
            else:
                return None
        except Exception as e:
            print('Uncaught exception: {} {}'.format(type(e), e))
            sys.exit(1)


class RinexArchive:
    """
    Loads what is in the archive already.
    TODO: Clean up some of the uneccesary parts.
    """
    def __init__(self, archiveoptions: ReadOptions):
        try:
            conn = Connection(archiveoptions.options)
            # Read the structure definition table
            self.levels = conn.load_tankstruct()
            self.keys = conn.load_table('keys')
            # Read the station and network tables
            self.networks = conn.load_table('networks')
            self.stations = conn.load_table('stations')
            del conn
        except Exception as e:
            print('Uncaught Exception: {} {}'.format(type(e), e))
            sys.exit(1)

    @staticmethod
    def parse_rinex_filename(filename):
        # parse a rinex filename
        sfile = re.findall(r'(\w{4})(\d{3})(\w{1})\.(\d{2})([o])$', filename)

        if sfile:
            return sfile[0]
        else:
            return []


class JobServer:

    def __init__(self, options, parent_logger='archive'):
        """
        Initialize the the dispy scheduler and test the connection to the expected nodes.
        :param options: gpys.ReadOptions instance
        :param parent_logger: Name of the function creating a new instance of JobServer
        """
        self.head_logger = logging.getLogger('.'.join([parent_logger, 'gpys.JobServer']))
        self.cluster_options = {'ip_addr': options['head_node'],
                                'ping_interval': int(options['ping_interval']),
                                'pulse_interval': 6,
                                'loglevel': 60}
        self.workers = options['node_list']
        self.tested = False

    def _connect(self, compute):
        try:
            self.head_logger.debug('Testing out the cluster.')
            cluster = dispy.JobCluster(compute, **self.cluster_options)
            for node in self.workers:
                cluster.discover_nodes(node)
                self.head_logger.debug('Waiting {} seconds.'.format(2*int(self.cluster_options['ping_interval'])))
                time.sleep(2*int(self.cluster_options['ping_interval']))
            return cluster
        except ConnectionError as e:
            self.head_logger.error(e, exc_info=sys.exc_info())
            sys.exit(1)
        except Exception as e:
            self.head_logger.error(f'Uncaught Exception: {type(e)} {e}')
            sys.exit(1)

    def cluster_test(self):
        j, tend, tstart = None, None, None
        try:
            self.head_logger.debug('Testing out the cluster.')

            def compute():
                time.sleep(0.1)
                return socket.gethostname()

            cluster = self._connect(compute)

            jobs = []
            for node in self.workers:
                self.head_logger.debug('Sending job to {}'.format(node))
                job = cluster.submit_job_id_node('InitialTest-{}'.format(node), node)
                jobs.append(job)
            if None in jobs:
                raise ConnectionError('Error while submitting job.  '
                                      'The server may not be '
                                      'started on {}.'.format(self.cluster_options['node_list'][jobs.index(None)]))
            tstart = time.time()
            while not cluster.wait():
                time.sleep(0.1)
            tend = time.time()
            self.head_logger.debug('Jobs took {:.4} seconds.'.format(tend-tstart))
            for j in jobs:
                self.head_logger.debug(f'Node worked! {j.result}.')
            for node in self.workers:
                self.head_logger.debug(f'Sending files to {node}.')
            self.head_logger.debug('Started a Dispy job and it worked :D')
            self.tested = True
        except ConnectionError as e:
            self.head_logger.error(e)
            sys.exit(1)
        except Exception as e:
            self.head_logger.error(f'Uncaught Exception: {type(e)} {e}')
            sys.exit(1)
        finally:
            self.head_logger.debug('Shutting down cluster.')
            if isinstance(cluster, dispy.JobCluster):
                cluster.shutdown()
            del j, tend, tstart


class Distribute:
    """

    """
    def __init__(self, config):
        import threading
        self.config = config
        self.status = {'Created': dispy.DispyJob.Created,
                       'Running': dispy.DispyJob.Running,
                       'Terminated': dispy.DispyJob.Terminated,
                       'Finished': dispy.DispyJob.Finished,
                       'Abandoned': dispy.DispyJob.Abandoned,
                       'Cancelled': dispy.DispyJob.Cancelled}  # Used to shorten some if statements.
        self.cluster_options = {'ip_addr': config.options['head_node'],
                                'ping_interval': int(config.options['ping_interval']),
                                'setup': self.node_setup,
                                'callback': self.callback,
                                'pulse_interval': 6,
                                'loglevel': 60}
        self.lower_bound, self.upper_bound = 1350, 2700  # Lower and upper limits to the number of pending jobs.
        self.jobs_cond = threading.Condition()  # The condition lock used to limit the number of jobs submitted.
        self.pending_jobs = dict()  # Contains running jobs.
        self.popped_jobs = 0

    def dist(self, inputdata, function):
        try:
            logger.info('Parallel PPP run.')
            jobs = list()
            runtimes = list()
            processed_rinex = list()
            parse_data_cluster = dispy.JobCluster(function, **self.cluster_options)
            for n, file in enumerate(inputdata):
                logger.debug(f'Submitting job#{n}: {file.as_posix()}')
                job = parse_data_cluster.submit(file.as_posix(), self.config, n)
                jobs.append(job)
                self.jobs_cond.acquire()
                if job.status in [self.status['Created'], self.status['Running']]:
                    self.pending_jobs[job.id] = job
                    if len(self.pending_jobs) >= self.upper_bound:
                        logger.debug(f'More pending_jobs than {self.upper_bound}, '
                                     f'waiting for the next jobs_cond.notify.')
                        while len(self.pending_jobs) > self.lower_bound:
                            self.jobs_cond.wait()
                        logger.debug(f'')
                self.jobs_cond.release()
            logger.debug('Waiting for jobs to finish up submission.')
            parse_data_cluster.wait()
            logger.info(f'{len(jobs)} jobs have been submitted.')
            for job in jobs:
                logger.debug(f'Reading Job {job.id}')
                if job.status in [self.status['Terminated']]:
                    logger.error('Job was terminated.')
                    logger.error(f'STDOUT:\n{job.stdout}')
                    logger.error(f'STDERR:\n{job.stderr}')
                    logger.error(f'Result:\n{job.result}')
                    logger.error(job.exception)
                elif job.status in [self.status['Finished']]:
                    logger.info(f'Job {job.id} has finished')
                    if 'ERROR' in job.stderr:
                        logger.error(f'Job failed:\n{job.stderr}')
                    else:
                        logger.debug(f'STDOUT:\n{job.stdout}')
                        logger.debug(f'STDERR:\n{job.stderr}')
                        logger.debug(f'Result:\n{job.result}')
                    if job.result['completed']:
                        processed_rinex.append(job.result)
                        runtimes.append(job.result['runtime'])
            return processed_rinex
        except Exception as e:
            logger.error(f'Uncaught Exception {type(e)} {e}')
        finally:
            parse_data_cluster.shutdown()

    @staticmethod
    def node_setup() -> int:
        """
        Placeholder if in the future using a setup function is deemed to be useful.
        :return:
        """
        import logging
        global nodeloglevel, nodeform
        nodeform = logging.Formatter('%(asctime)-15s %(name)-50s %(levelname)-5s:%(lineno)4s %(message)s',
                                     '%Y-%m-%d %H:%M:%S')
        nodeloglevel = logging.DEBUG
        return 0

    @staticmethod
    def parse_data_in(filepath: str, config, n: int) -> dict:
        """
        Runs PPP on a RINEX and either send it to the rejected folder, retry folder, lock it or add it to an existing
        station record.  Most parameters from the run are stored in the rinex_dict variable and returned at the end of the
        function.  The function should run through each try block unless: there is a problem importing modules, adding
        global variables or assigning local variables.  Otherwise everything happens within a try block with a general
        exception catch that will just log an ERROR with the exception type and message.
        Steps in program:
        1. Import packages.
        2. Custom functions
            i. fileopts
        3. Add globals
        4. Define local variables.
        5. Set up the logger & create the working directory structure.
        6. Create the working directory.
        7. Pull the metadata.
        8. Set up the PPP run
        9. Run PPP
        10. Parse the PPP .sum file.

        TODO: Add file existence checking before PPP run?
        TODO: Implement the OTL correction either using grdtab during processing or
              http://holt.oso.chalmers.se/loading/hfo.html after the stations are added to the database.
        TODO: Find a newer version of the .svb_gps_yrly PPP file.
        TODO: Add command (gpsppp.cmd) file customization?
        TODO: Write PPP in python, it's a nightmare to work with!!!!
        :param filepath: Path to the RINEX we're working on.
        :param config: gpys.ReadOptions object containing parameters from the .cfg file.
        :param n: The job number.
        :return: Dictionary containing the parameters used during the run as well as the results with the following keys:
                 ofile:             Path object of the original file in data_in
                 file:              Path object to the location of the file in the production folder
                 start_date:        Date of the RINEX as determined by TEQC +qc
                 name:              4-character name of the station as determined by TEQC +qc
                 orbit:             Path object for the broadcast orbit file.
                 completion:        The percentage completion as a string between 0-100 as reported by TEQC
                 teqc_xyz:          3-element list with strings representing the X, Y and Z  coordinates found by TEQC
                 sp3path:           Path object to the precise orbit file in the production folder for the current
                                    day of the RINEX.
                 nextsp3path:       Same as sp3path but for the next day's precise orbit file.
                 ppp_input_string:  The string read into PPP via STDIN.
                 pppcoords:         3-element list containing float values for the ITRF X, Y and Z.
                 latlonh:           3-element list containing float values for the ITRF latitude, longitude and height.
                 pppref:            String representing the ITRF datum as reported by the PPP summary file.
                 completed:         Bool that is False if any exception was raised during the operation of this function.
                 runtime:           Float representing how long it took to complete the program.
        """
        import logging
        import socket
        import os
        from pathlib import Path
        import datetime as dt
        import shutil
        import subprocess
        import sys
        from decimal import Decimal

        ntimeout = 60
        program_starttime = dt.datetime.now()
        month2int = {'Jan': 1,
                     'Feb': 2,
                     'Mar': 3,
                     'Apr': 4,
                     'May': 5,
                     'Jun': 6,
                     'Jul': 7,
                     'Aug': 8,
                     'Sep': 9,
                     'Oct': 10,
                     'Nov': 11,
                     'Dec': 12}

        def fileopts(orig_file: Path) -> Path:
            """
            First determine which compression was used by invoking UNIX `file`
            Remove the .Z suffix
            Change the internal path suffix to o from d
            Raise exception if the last letter in the extention doesn't match [sp]3, [cl]k, [##]n, [##]o
            :param orig_file:
            :return:
            """
            nodelogger.debug('Determining which compression was used:')
            compression_string = subprocess.run(f'file {orig_file.name} | cut -d\  -f2',
                                                shell=True,
                                                timeout=ntimeout,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT,
                                                encoding='utf-8').stdout.strip()
            if compression_string in ['gzip']:
                compression = 'gunzip'
            elif compression_string in ["compress'd"]:
                compression = 'uncompress'
            else:
                raise Exception(f"Uncrecognized compression: {compression_string}"
                                f" ['file', {orig_file.name}, '|', 'cut', '-d ', '-f2']")
            crx2rnx = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'crx2rnx']))
            if orig_file.suffix in '.Z':
                nodelogger.debug(f'Decompressing {orig_file.name}')
                subprocess.run([compression, '-f', orig_file.name], check=True, timeout=ntimeout)
                orig_file = orig_file.with_suffix('')
            if orig_file.match('*.??d'):
                subprocess.run([str(crx2rnx), '-f', orig_file], check=True, timeout=ntimeout)
                os.remove(orig_file.as_posix())
                orig_file = orig_file.with_suffix(orig_file.suffix.replace('d', 'o'))
            if orig_file.suffix[-1] not in ['3', 'k', 'n', 'o']:
                raise Exception(f'Unrecognized file extension: {orig_file.suffix}')
            return orig_file

        # Globals
        global nodeloglevel, nodeform
        # Internal variable set up and definition for later reference.
        nodelogger = None  # Internal logger for the function.
        ofile = None  # Original file converted to a pathlib.Path object.
        file = None  # pathlib.Path object for the RINEX we're working on, updated as it is operated on.
        prodpath = None  # pathlib.Path object defining where the production folder is.
        start_date = None  # datetime.date object for the RINEX as determined by TEQC
        orbit = None  # pathlib.Path object with the archive path to the BRDC orbit file.
        sp3path = None  # Path to the IGS sp3, also used for the clk files
        nextsp3path = None  # Path to next day's sp3 file.
        ppp_input_string = 'BadFilename'  # The STDIN input to the PPP command.
        rinex_dict = dict()  # Where all the info about the run is stored.
        tanktype = {'<year>': None,
                    '<month>': None,
                    '<day>': None,
                    '<gpsweek>': None}  # Helps define the path to the brdc or igs tanks.
        gpsweek = None  # Normally just a string
        gpsweekday = None  # Just a string.
        nextweek = None  # The week of the next day's orbits
        nextday = None  # Day of the next orbit
        igsdir = None  # Shorthand for the archive location for igs orbits
        brdcdir = None  # Shorthand for brdc location
        clkpath = None  # Same but for the clk files
        nextclkpath = None  # Ditto, next day though.
        pppref = None  # The reference frame that PPP uses, sometimes it swaps between IGS14 and IGb08
        complete = True
        """Set_Logger-------------------------------------------------------------------------------------------------------
        Set up the logger internal to parse_data_in in order to make debugging any problems much easier.  Also convert the 
        string filepath into a pathlib.Path.

        Previous errors:
        None (yay!)
        """
        try:
            ofile = Path(filepath)
            nodelogger = logging.getLogger(f'parse_data_in.{n}')
            nodelogger.setLevel(nodeloglevel)
            nodestream = logging.StreamHandler()
            nodestream.setFormatter(nodeform)
            nodestream.setLevel(nodeloglevel)
            nodelogger.addHandler(nodestream)
            nodelogger.debug(f'Logger setup on {socket.gethostname()}.')
            rinex_dict['ofile'] = ofile
        except Exception as e:
            print(f'Exception raised during logger setup {type(e)} {e}', file=sys.stderr)
            print(f'{globals()}', file=sys.stderr)
            complete = False
        """Create_Production_Folder-----------------------------------------------------------------------------------------
        First create a pathlib.Path object that sets the working directory for the remainder of the function.  The working
        directory is defined in the config read by the main program.  The production/job# folder is created in that working
        directory.  We also copy the file from data_in to the production folder and then uncompress it.

        Previous errors:
        None (yay!)
        """
        try:
            prodpath = Path(f'production/job{n}')
            prodpath = os.sep.join([config.options['working_dir'], str(prodpath)])
            prodpath = Path(prodpath)
            os.makedirs(str(prodpath), exist_ok=True)
            os.chdir(str(prodpath))
            nodelogger.debug(f'Created folder: {prodpath.name}')
            shutil.copy(str(ofile), str(prodpath))
            file = prodpath / ofile.name
            nodelogger.debug(f'Working with {file.name}')
            nodelogger.debug('Created paths to dependencies.')
            teqc = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'teqc']))
            ppp = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'ppp']))
            atxfile = Path(os.sep.join([config.options['working_dir'], 'dependencies', config.options['atx']]))
            trffile = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'gpsppp.trf']))
            svbfile = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'gpsppp.svb_gnss_yrly']))
            nodelogger.debug(f'Using {teqc}, {atxfile}, {ppp}')
            file = fileopts(file)
            for f in [atxfile, trffile, svbfile]:
                shutil.copy(str(f), str(prodpath))
            rinex_dict['file'] = file
        except PermissionError as e:
            nodelogger.error(f'Permission error: {type(e)} {e}')
            complete = False
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """Metadata---------------------------------------------------------------------------------------------------------
        First get the date information from the RINEX file using teqc +meta, then parse out the 'start date & time' field of
        the TEQC metadata record and create some useful date objects.  Since we're already dealing with some dates here we
        go ahead and update the tank structure dict so we can find the orbits based on how they're defined in the config
        file.  Also pull the station name.

        Previous errors:
        None (yay!)
        """
        try:
            nodelogger.debug(f'Loading metadata from {file.name}')
            metadata = subprocess.run([teqc.as_posix(),
                                       '+meta',
                                       file.name],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      encoding='utf-8',
                                      check=True,
                                      timeout=ntimeout)
            start_date = [x for x in metadata.stdout.splitlines() if 'start date & time:' in x][0]
            start_date = [int(x) for x in start_date.partition(':')[-1].strip().partition(' ')[0].split('-')]
            start_date = dt.date(start_date[0], start_date[1], start_date[2])
            gpsweek = abs(start_date - dt.date(1980, 1, 6)).days / 7
            gpsweekday = abs(start_date - dt.date(1980, 1, 6)).days % 7
            gpsweek = str(int(gpsweek))
            gpsweekday = str(int(gpsweekday))
            if gpsweekday != '6':
                nextday = str(int(gpsweekday) + 1)
                nextweek = gpsweek
            else:
                nextday = '0'
                nextweek = str(int(gpsweek) + 1)
            tanktype['<year>'] = str(start_date.year)
            tanktype['<month>'] = str(start_date.month)
            tanktype['<day>'] = str(start_date.day)
            tanktype['<gpsweek>'] = gpsweek
            brdcdir = Path(config.options['brdc']).with_name(tanktype[Path(config.options['brdc']).name])
            igsdir = Path(config.options['sp3']).with_name(tanktype[Path(config.options['sp3']).name])
            name = [x for x in metadata.stdout.splitlines() if 'station name:' in x][0]
            name = name.partition(':')[2].strip()
            rinex_dict['start_date'] = start_date
            rinex_dict['name'] = name
        except FileNotFoundError as e:
            nodelogger.error(f'Make sure that TEQC is on system path and executable: {type(e)} {e}')
            complete = False
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """BRDC-------------------------------------------------------------------------------------------------------------
        Move the BRDC orbit into the production/job# folder for the next step where we run teqc, without the broadcast
        orbits we only get qc lite instead of qc full.

        Previous errors:
        Once a brdc file was compressed with gzip instead of unix compress -> check the compression first?
        """
        try:
            orbit = brdcdir / Path('brdc{}0.{}n.Z'.format(start_date.strftime('%j'),
                                                          start_date.strftime('%y')))
            shutil.copy(str(orbit), str(prodpath))
            orbit = fileopts(orbit)
            rinex_dict['orbit'] = orbit
        except AttributeError as e:
            nodelogger.error(f'A variable was incorrectly assigned, did a previous block fail?: {type(e)} {e}')
            complete = False
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """QC---------------------------------------------------------------------------------------------------------------
        Finally, something other than file operations!
        Run teqc with the QC flags (+qc), no qc file (+q) with the specific orbit file (-nav), no GLONASS (-R), 
        no SBAS (-S), no GALILEO (-E), no BEIDOU (-C) and no QZSS (-J). Use -no_orbit G so that no error is raised about 
        missing nav files for GPS.  Check the result of the qc run.  Right now we're just going to check the percentage 
        completion and make sure it's above 90% and pulling the TEQC coordinates.

        Previous errors:
        None (yay!)
        """
        try:
            nodelogger.debug(f'Running TEQC on {file.name}')
            qcresults = subprocess.run([teqc,
                                        '+qcq',
                                        '-nav',
                                        orbit.name,
                                        '-R', '-S', '-E', '-C', '-J',
                                        file.name],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       check=True,
                                       encoding='utf-8',
                                       timeout=ntimeout).stdout.splitlines()
            completion = qcresults[-1].split(' ')
            completion = [x for x in completion if x != '']
            completion = completion[-4]
            teqc_xyz = [x for x in qcresults if 'antenna WGS 84 (xyz)' in x][0].partition(':')[-1].strip().split(' ')[
                       0:3]
            teqc_xyz = [Decimal(x) for x in teqc_xyz]
            observationstime = [x for x in qcresults if 'Time of start of window' in x][0].partition(':')[
                2].strip().split()
            observationstime = dt.datetime(int(observationstime[0]),
                                           month2int[observationstime[1]],
                                           int(observationstime[2]),
                                           int(observationstime[3].split(':')[0]),
                                           int(observationstime[3].split(':')[1]),
                                           int(observationstime[3].split(':')[2].split('.')[0]),
                                           int(float(observationstime[3].split(':')[2].split('.')[1]) * 1e6))
            observationftime = [x for x in qcresults if 'Time of  end  of window' in x][0].partition(':')[
                2].strip().split()
            observationftime = dt.datetime(int(observationftime[0]),
                                           month2int[observationftime[1]],
                                           int(observationftime[2]),
                                           int(observationftime[3].split(':')[0]),
                                           int(observationftime[3].split(':')[1]),
                                           int(observationftime[3].split(':')[2].split('.')[0]),
                                           int(float(observationftime[3].split(':')[2].split('.')[1]) * 1e6))
            interval = Decimal(
                [x for x in qcresults if 'Observation interval' in x][0].partition(':')[2].strip().split()[0])
            receivertype = [x for x in qcresults if 'Receiver type' in x][0].partition(':')[2].strip().partition('(')[
                0].strip()
            receiverserial = \
            [x for x in qcresults if 'Receiver type' in x][0].partition(':')[2].strip().partition('(')[2].partition(
                ')')[0].split()[2]
            receiverfw = \
            [x for x in qcresults if 'Receiver type' in x][0].partition(':')[2].strip().partition('(')[2].partition(
                ')')[2].strip().strip('(').strip(')').split()[2]
            antennatype = [x for x in qcresults if 'Antenna type' in x][0].partition(':')[2].strip().split()[0]
            antennaserial = \
            [x for x in qcresults if 'Antenna type' in x][0].partition(':')[2].strip().partition('(')[2].partition(')')[
                0].split()[2]
            observationfyear = float(start_date.year) + float(start_date.strftime('%j')) / \
                               float(dt.date(start_date.year, 12, 31).strftime('%j'))

            rinex_dict['ObservationFYear'] = observationfyear
            rinex_dict['ObservationSTime'] = observationstime
            rinex_dict['ObservationFTime'] = observationftime
            rinex_dict['Interval'] = interval
            rinex_dict['ReceiverType'] = receivertype
            rinex_dict['ReceiverSerial'] = receiverserial
            rinex_dict['ReceiverFw'] = receiverfw
            rinex_dict['AntennaType'] = antennatype
            rinex_dict['AntennaSerial'] = antennaserial
            rinex_dict['completion'] = Decimal(completion)
            rinex_dict['teqc_xyz'] = teqc_xyz
        except AttributeError as e:
            nodelogger.error(f'A variable was incorrectly assigned, did a previous block fail?: {type(e)} {e}')
            complete = False
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """IGS file operations----------------------------------------------------------------------------------------------
        Big for loop in a try statement... Cycle through the types of orbits, if there aren't any orbits available, raise
        an exception.

        Previous errors:
        None (yay!)
        """
        try:
            for sp3type in config.sp3types:
                sp3path = igsdir / Path(f'{sp3type}{gpsweek}{gpsweekday}.sp3.Z')
                clkpath = igsdir / Path(f'{sp3type}{gpsweek}{gpsweekday}.clk.Z')
                nextsp3path = igsdir.with_name(nextweek) / Path(f'{sp3type}{nextweek}{nextday}.sp3.Z')
                nextclkpath = igsdir.with_name(nextweek) / Path(f'{sp3type}{nextweek}{nextday}.clk.Z')
                if sp3path.exists() and clkpath.exists() and nextsp3path.exists() and nextclkpath.exists():
                    nodelogger.debug(f'Using {sp3type} orbits.')
                    shutil.copy(sp3path, prodpath)
                    shutil.copy(clkpath, prodpath)
                    shutil.copy(nextsp3path, prodpath)
                    shutil.copy(nextclkpath, prodpath)
                    sp3path = prodpath / sp3path.name
                    clkpath = prodpath / clkpath.name
                    nextsp3path = prodpath / nextsp3path.name
                    nextclkpath = prodpath / nextclkpath.name
                    sp3path = fileopts(sp3path)
                    clkpath = fileopts(clkpath)
                    nextsp3path = fileopts(nextsp3path)
                    nextclkpath = fileopts(nextclkpath)
                    break
                elif sp3type == config.sp3types[-1]:
                    raise Exception(f"Didn't find any valid orbits in {sp3path.parent}")
            rinex_dict['sp3path'] = sp3path
            rinex_dict['nextsp3path'] = nextsp3path
        except TypeError as e:
            nodelogger.error(f'A variable was incorrectly assigned, did a previous block fail?: {type(e)} {e}')
            complete = False
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """PPP file operations----------------------------------------------------------------------------------------------
        Generate the files: gpsppp.def, gpsppp.flt, gpsppp.olc, gpsppp.met, gpsppp.eop, and gpsppp.cmd.  Right now the only
        ones that actually have anything in them are the .def, .cmd, .

        Previous errors:
        The PPP metadata files have to be named gpsppp.[ext] :(
        """
        try:
            nodelogger.debug('Creating files for the PPP run.')
            defstring = "'LNG' 'ENGLISH'\n" \
                f"'TRF' '{trffile.name}'\n" \
                f"'SVB' '{svbfile.name}'\n" \
                f"'PCV' '{atxfile.name}'\n" \
                        "'FLT' 'gpsppp.flt'\n" \
                        "'OLC' 'gpsppp.olc'\n" \
                        "'MET' 'gpsppp.met'\n" \
                        "'ERP' 'gpsppp.eop'\n" \
                        "'GSD' 'The Ohio State University'\n" \
                        "'GSD' '--'\n"
            # TODO: Change to a formatted string.
            fltstring = 'FLT    8.1    130    150   5000    300  1     0        2        1        1     0       2'
            cmdfile = 'gpsppp.cmd'
            cmdstring = "' UT DAYS OBSERVED                      (1-45)'               1\n" \
                        "' USER DYNAMICS         (1=STATIC,2=KINEMATIC)'               1\n" \
                        "' OBSERVATION TO PROCESS         (1=COD,2=C&P)'               2\n" \
                        "' FREQUENCY TO PROCESS        (1=L1,2=L2,3=L3)'               3\n" \
                        "' SATELLITE EPHEMERIS INPUT     (1=BRD ,2=SP3)'               2\n" \
                        "' SATELLITE PRODUCT (1=NO,2=Prc,3=RTCA,4=RTCM)'               2\n" \
                        "' SATELLITE CLOCK INTERPOLATION   (1=NO,2=YES)'               1\n" \
                        "' IONOSPHERIC GRID INPUT          (1=NO,2=YES)'               1\n" \
                        "' SOLVE STATION COORDINATES       (1=NO,2=YES)'               2\n" \
                        "' SOLVE TROP. (1=NO,2-5=RW MM/HR) (+100=grad) '             105\n" \
                        "' BACKWARD SUBSTITUTION           (1=NO,2=YES)'               1\n" \
                        "' REFERENCE SYSTEM            (1=NAD83,2=ITRF)'               2\n" \
                        "' COORDINATE SYSTEM(1=ELLIPSOIDAL,2=CARTESIAN)'               2\n" \
                        "' A-PRIORI PSEUDORANGE SIGMA               (m)'           2.000\n" \
                        "' A-PRIORI CARRIER PHASE SIGMA             (m)'           0.015\n" \
                        "' LATITUDE  (ddmmss.sss,+N) or ECEF X      (m)'          0.0000\n" \
                        "' LONGITUDE (ddmmss.sss,+E) or ECEF Y      (m)'          0.0000\n" \
                        "' HEIGHT (m)                or ECEF Z      (m)'          0.0000\n" \
                        "' ANTENNA HEIGHT                           (m)'          0.0000\n" \
                        "' CUTOFF ELEVATION                       (deg)'          10.000\n" \
                        "' GDOP CUTOFF                                 '          20.000"
            ppp_input_string = f'{file.name}\n' \
                f'{cmdfile}\n' \
                '0 0\n' \
                '0 0\n' \
                f'{sp3path.name}\n' \
                f'{clkpath.name}\n' \
                f'{nextsp3path.name}\n' \
                f'{nextclkpath.name}'
            with open('gpsppp.flt', 'w') as f:
                f.write(fltstring)
            with open('gpsppp.olc', 'w') as f:
                f.write('')
            with open('gpsppp.met', 'w') as f:
                f.write('')
            with open('gpsppp.eop', 'w') as f:
                f.write('')
            with open(cmdfile, 'w') as f:
                f.write(cmdstring)
            with open('gpsppp.def', 'w') as f:
                f.write(defstring)
            nodelogger.debug('Done creating PPP files.')
            rinex_dict['ppp_input_string'] = ppp_input_string
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """PPP--------------------------------------------------------------------------------------------------------------
        Pretty basic step here, just run PPP and send the stderr and stdout to /dev/null (trash).  Raise an exception if it
        fails.  Added the encoding parameter as to properly input the ppp_input_string.

        Previous errors:
        None (yay!)
        """
        try:
            nodelogger.debug('Running PPP.')
            subprocess.run(ppp.as_posix(),
                           input=ppp_input_string,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=True,
                           encoding='utf-8',
                           timeout=ntimeout)
            nodelogger.debug('Done running PPP')
        except FileNotFoundError as e:
            nodelogger.error(f'Make sure that PPP is on system path and executable: {type(e)} {e}')
            complete = False
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """Read_SUM---------------------------------------------------------------------------------------------------------
        Load the .sum file that PPP creates into a variable that we can then parse to get all the results.  The .sum file
        isn't usually too big so it shouldn't end up being a memory hog.  The block containing the relevant coordinates is
        block 3.3 so we grab that block.  Due to previous error (1) we try with the IGS14 reference frame which is the most
        common case and raise an exception if it doesn't work and assume that it used IGb08 that is the only other frame 
        I've encountered during debugging.  Then we grab the block with the line starting with CARTESIAN.  Reducing to the 
        second block was implemented due to previous error number (2).  

        Previous errors:
        1. Sometimes it uses IGb08 instead of IGS14 that changes the section 3.4 header in the .sum file.
        2. Sometimes right after the 3.3 block header it says something like "updated the rinex header" or something.
        """
        try:
            nodelogger.debug('Parsing {}'.format(file.with_suffix('.sum').name))
            with open(file.with_suffix('.sum').name) as f:
                summary = f.read()
            split_sum = summary.splitlines()
            coordstr = [x for x in split_sum if ' 3.4 Coordinate differences' in x]
            pppref = coordstr[0].partition('(')[2].strip(')')
            coordindex = [split_sum.index(' 3.3 Coordinate estimates'), split_sum.index(coordstr[0])]
        except FileNotFoundError as e:
            nodelogger.error(f"Couldn't find the .sum file, PPP probably failed: {type(e)} {e}")
            complete = False
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """Parse .sum into variables----------------------------------------------------------------------------------------
        """
        try:
            resultblock = summary.splitlines()[coordindex[0]:coordindex[1] - 1]
            cartind = resultblock.index(f' CARTESIAN           NAD83(CSRS )     ITRF ({pppref})   Sigma(m) NAD-ITR(m)')
            pppcoords = [Decimal(x.split()[3]) for x in resultblock[cartind + 1:cartind + 4]]
            ellipend = resultblock.index(f' ELLIPSOIDAL    ')
            ll = [x.partition(')')[2].strip().split()[3:6] for x in resultblock[ellipend + 1: ellipend + 3]]
            nodelogger.debug(f'Read in location: {ll}')
            latlonh = [(abs(Decimal(x[0])) + Decimal(x[1]) / 60
                        + Decimal(x[2]) / 3600) * Decimal(x[0]) / abs(Decimal(x[0])) for x in ll]
            latlonh = [x.quantize(Decimal('.0001')) for x in latlonh]
            height = Decimal(resultblock[ellipend + 3].split()[3]).quantize(Decimal('.1'))
            latlonh.append(height)
            nodelogger.debug(f'Got coordinates: {pppcoords}')
            rinex_dict['pppcoords'] = pppcoords
            rinex_dict['latlonh'] = latlonh
            rinex_dict['pppref'] = pppref
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        """Delete the production folder-------------------------------------------------------------------------------------
        """
        try:
            if complete:
                nodelogger.debug(f'No errors found, deleting {prodpath}')
                shutil.rmtree(prodpath.as_posix())
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
            complete = False
        '------------------------------------------------------------------------------------------------------------------'
        rinex_dict['completed'] = complete
        program_endtime = dt.datetime.now()
        rinex_dict['runtime'] = (program_endtime - program_starttime).total_seconds()
        return rinex_dict

    @staticmethod
    def database_ops(rinex_dict: dict = None, config=None, n=0):
        """
        Compares the information in the rinex_dict object returned by parse_data_in with the information in the database
        collected by the head node.  Returns an string that indicates what should be done with the file.
        Possible outcomes:
        -File is not close to any stations and is added with the ??? network name.
        -File is close to another station but has a different name added with the ??? network name.
        -File matches the location of another station and has the same name and it has a network code not matching ???,
        it is moved into the archive.

        Locked files remain in data_in until they are unlocked by adding a network code to the station in the stations table
        of the database.
        :return:
        """
        import logging
        import socket
        import gpys
        import shutil
        from pathlib import Path
        import os
        # Globals
        global nodeloglevel, nodeform

        # Local variables           Notes:
        nodelogger = None  # Logger
        """Set_Logger-------------------------------------------------------------------------------------------------------
        Set up the logger internal to parse_data_in in order to make debugging any problems much easier.  Also convert the 
        string filepath into a pathlib.Path object.

        Previous errors:
        None (yay!)
        """
        try:
            nodelogger = logging.getLogger(f"database_ops.{rinex_dict['name']}-{rinex_dict['start_date']}")
            nodelogger.setLevel(nodeloglevel)
            nodestream = logging.StreamHandler()
            nodestream.setFormatter(nodeform)
            nodestream.setLevel(nodeloglevel)
            nodelogger.addHandler(nodestream)
            nodelogger.debug(f'Logger setup on {socket.gethostname()}.')
        except Exception as e:
            print(f'Exception raised during logger setup {type(e)} {e}', file=sys.stderr)
        """Check the locks table--------------------------------------------------------------------------------------------
        """
        try:
            range_checkcnn = gpys.Connection(config.options, parent_logger=nodelogger.name)
            nodelogger.debug(f"Looking for stations in the database nearby {rinex_dict['name']}.")
            nearest = range_checkcnn.spatial_check(rinex_dict['latlonh'][0:2], search_in_new=True)
            nodelogger.debug(f'Nearest station found: {nearest}')
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')
        try:
            if not nearest:
                nodelogger.debug(f"No nearby stations found, adding {rinex_dict['name']} to the stations table.")
                stations_entry = {'NetworkCode': '???',
                                  'StationCode': rinex_dict['name'],
                                  'auto_x': rinex_dict['teqc_xyz'][0],
                                  'auto_y': rinex_dict['teqc_xyz'][1],
                                  'auto_z': rinex_dict['teqc_xyz'][2],
                                  'lat': rinex_dict['latlonh'][0],
                                  'lon': rinex_dict['latlonh'][1],
                                  'height': rinex_dict['latlonh'][2]}
                range_checkcnn.insert('stations', stations_entry)
                locks_entry = {'filename': rinex_dict['ofile'].name,
                               'NetworkCode': '???',
                               'StationCode': rinex_dict['name']}
                range_checkcnn.insert('locks', locks_entry)
            else:
                nodelogger.debug(f'Found a nearby station: {nearest}')
                if nearest['NetworkCode'][0] in '???' and nearest['StationCode'][0] in rinex_dict['name']:
                    nodelogger.debug(f"Station is in temporary network already {nearest['NetworkCode'][0]}, "
                                     f"{rinex_dict['name']}")
                    locks_entry = {'filename': rinex_dict['ofile'].name,
                                   'NetworkCode': '???',
                                   'StationCode': rinex_dict['name']}
                    range_checkcnn.insert('locks', locks_entry)
                    rinex_entry = {'NetworkCode': nearest['NetworkCode'][0],
                                   'StationCode': rinex_dict['name'],
                                   'ObservationYear': rinex_dict['start_date'].year,
                                   'ObservationMonth': rinex_dict['start_date'].month,
                                   'ObservationDay': rinex_dict['start_date'].day,
                                   'ObservationDOY': int(rinex_dict['start_date'].strftime('%j')),
                                   'ObservationFYear': rinex_dict['ObservationFYear'],
                                   'ObservationSTime': rinex_dict['ObservationSTime'],
                                   'ObservationETime': rinex_dict['ObservationFTime'],
                                   'ReceiverType': rinex_dict['ReceiverType'],
                                   'ReceiverSerial': rinex_dict['ReceiverSerial'],
                                   'ReceiverFw': rinex_dict['ReceiverFw'],
                                   'AntennaType': rinex_dict['AntennaType'],
                                   'AntennaSerial': rinex_dict['AntennaSerial'],
                                   'AntennaDome': None,
                                   'Filename': rinex_dict['ofile'].name,
                                   'Interval': rinex_dict['Interval'],
                                   'AntennaOffset': None,
                                   'Completion': rinex_dict['completion']}
                    range_checkcnn.insert('rinex', rinex_entry)
                elif nearest['NetworkCode'][0] not in '???' and nearest['StationCode'][0] in rinex_dict['name']:
                    nodelogger.debug(f"Station already has a network code: {nearest['NetworkCode'][0]}, "
                                     f"{rinex_dict['name']}")
                    rinex_entry = {'NetworkCode': nearest['NetworkCode'][0],
                                   'StationCode': rinex_dict['name'],
                                   'ObservationYear': rinex_dict['start_date'].year,
                                   'ObservationMonth': rinex_dict['start_date'].month,
                                   'ObservationDay': rinex_dict['start_date'].day,
                                   'ObservationDOY': int(rinex_dict['start_date'].strftime('%j')),
                                   'ObservationFYear': rinex_dict['ObservationFYear'],
                                   'ObservationSTime': rinex_dict['ObservationSTime'],
                                   'ObservationETime': rinex_dict['ObservationFTime'],
                                   'ReceiverType': rinex_dict['ReceiverType'],
                                   'ReceiverSerial': rinex_dict['ReceiverSerial'],
                                   'ReceiverFw': rinex_dict['ReceiverFw'],
                                   'AntennaType': rinex_dict['AntennaType'],
                                   'AntennaSerial': rinex_dict['AntennaSerial'],
                                   'AntennaDome': None,
                                   'Filename': rinex_dict['ofile'].name,
                                   'Interval': rinex_dict['Interval'],
                                   'AntennaOffset': None,
                                   'Completion': rinex_dict['completion']}
                    range_checkcnn.insert('rinex', rinex_entry)
                    archive_structure = {'network': nearest['NetworkCode'][0],
                                         'station': rinex_dict['name'],
                                         'year': str(rinex_dict['start_date'].year),
                                         'doy': rinex_dict['start_date'].strftime('%j')}
                    targetdir = [config.options['path']]
                    for level in config.rinex_struct:
                        targetdir.append(archive_structure[level])
                    targetdir = Path(os.sep.join(targetdir))
                    os.makedirs(targetdir, exist_ok=True)
                    shutil.move(rinex_dict['ofile'].as_posix(), targetdir.as_posix())
        except Exception as e:
            nodelogger.error(f'Uncaught Exception {type(e)} {e}')

    def callback(self, job):
        """
        Simple callback function that helps reduce the amount of submissions.  Typing hints don't work here.
        :param job: An instance of dispy.DispyJob
        :return:
        """
        if job.status in [self.status['Finished'],
                          self.status['Terminated'],
                          self.status['Abandoned'],
                          self.status['Cancelled']]:
            self.jobs_cond.acquire()
            if job.id:
                self.pending_jobs.pop(job.id)
                self.popped_jobs += 1
                if len(self.pending_jobs) <= self.lower_bound:
                    self.jobs_cond.notify()
                self.jobs_cond.release()