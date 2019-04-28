"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez

This class is used to connect to the database and handles inserts, updates and selects
It also handles the error, info and warning messages
"""

from decimal import Decimal

import pg
import pgdb


class DberrInsert(Exception):
    """
    Template for an error.
    """
    pass


class DberrUpdate(Exception):
    """
    Template for an error.
    """
    pass


class DberrConnect(Exception):
    """
    Template for an error.
    """
    pass


class DberrDelete(Exception):
    """
    Template for an error.
    """
    pass


class Cnn(pg.DB):
    """
    Template
    """
    def __init__(self, configfile, use_float=False):

        # set casting of numeric to floats
        pg.set_typecast('Numeric', float)

        options = {'hostname': 'localhost',
                   'username': 'postgres',
                   'password': '',
                   'database': 'gnss_data'}

        self.active_transaction = False
        self.options = options
        # parse session config file
        config = configparser.ConfigParser()
        config.readfp(open(configfile))

        # get the database config
        for iconfig, val in dict(config.items('postgres')).items():
            options[iconfig] = val

        # open connection to server
        tries = 0
        while True:
            try:
                pg.DB.__init__(self, host=options['hostname'], user=options['username'], passwd=options['password'],
                               dbname=options['database'])
                # set casting of numeric to floats
                pg.set_typecast('Numeric', float)
                if use_float:
                    pg.set_decimal(float)
                else:
                    pg.set_decimal(Decimal)
                break
            except pg.InternalError as e:
                if 'Operation timed out' in str(e) or 'Connection refused' in str(e):
                    if tries < 4:
                        tries += 1
                        continue
                    else:
                        raise DberrConnect(e)
                else:
                    raise e
            except Exception as e:
                raise e

    def query_float(self, command, as_dict=False):
        """
        Template
        """

        pg.set_typecast('Numeric', float)
        pg.set_decimal(float)

        rs = self.query(command)

        if as_dict:
            recordset = rs.dictresult()
        else:
            recordset = rs.getresult()

        pg.set_typecast('Numeric', Decimal)
        pg.set_decimal(Decimal)

        return recordset

    def get_columns(self, table):
        """
        Template
        """
        tblinfo = self.query(
            'select column_name, data_type from information_schema.columns where table_name=\'%s\'' % table)

        field_dict = dict()

        for field in tblinfo.dictresult():
            field_dict[field['column_name']] = field['data_type']

        return field_dict

    def begin_transac(self):
        """
        Template
        """
        # do not begin a new transaction with another one active.
        if self.active_transaction:
            self.rollback_transac()

        self.active_transaction = True
        self.begin()

    def commit_transac(self):
        self.active_transaction = False
        self.commit()

    def rollback_transac(self):
        self.active_transaction = False
        self.rollback()

    def insert(self, table, row=None, **kw):

        try:
            pg.DB.insert(self, table, row, **kw)
        except Exception as e:
            raise Dberrinsert(e)

    def executemany(self, sql, parameters):

        con = pgdb.connect(host=self.options['hostname'],
                           user=self.options['username'],
                           password=self.options['password'],
                           database=self.options['database'])

        cur = con.cursor()
        cur.executemany(sql, parameters)
        con.commit()

    def update(self, table, row=None, **kw):

        try:
            pg.DB.update(self, table, row, **kw)
        except Exception as e:
            raise DberrUpdate(e)

    def delete(self, table, row=None, **kw):

        try:
            pg.DB.delete(self, table, row, **kw)
        except Exception as e:
            raise DberrDelete(e)

    def insert_event(self, event):

        self.insert('events', event.db_dict())

        return

    def insert_event_bak(self, type, module, desc):

        # do not insert if record exists
        desc = '%s%s' % (module, desc.replace('\'', ''))
        desc = re.sub(r'[^\x00-\x7f]+', '', desc)
        # remove commands from events
        # modification introduced by DDG (suggested by RS)
        desc = re.sub(r'BASH.*', '', desc)
        desc = re.sub(r'PSQL.*', '', desc)

        # warn = self.query('SELECT * FROM events WHERE "EventDescription" = \'%s\'' % (desc))

        # if warn.ntuples() == 0:
        self.insert('events', EventType=type, EventDescription=desc)

        return

    def insert_warning(self, desc):
        line = inspect.stack()[1][2]
        caller = inspect.stack()[1][3]

        mod = platform.node()

        module = '[%s:%s(%s)]\n' % (mod, caller, str(line))

        # get the module calling for insert_warning to make clear how is logging this message
        self.insert_event_bak('warn', module, desc)

    def insert_error(self, desc):
        line = inspect.stack()[1][2]
        caller = inspect.stack()[1][3]

        mod = platform.node()

        module = '[%s:%s(%s)]\n' % (mod, caller, str(line))

        # get the module calling for insert_warning to make clear how is logging this message
        self.insert_event_bak('error', module, desc)

    def insert_info(self, desc):
        line = inspect.stack()[1][2]
        caller = inspect.stack()[1][3]

        mod = platform.node()

        module = '[%s:%s(%s)]\n' % (mod, caller, str(line))

        self.insert_event_bak('info', module, desc)

    def __del__(self):
        if self.active_transaction:
            self.rollback()


"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez

This class handles the interface between the directory structure of the rinex archive and the databased records.
It can be used to retrieve a rinex path based on a rinex database record
It can also scan the dirs of a supplied path for d.Z and station.info files (the directories and files have to match the
declared directory structure and {stmn}{doy}{session}.{year}d.Z, respectively)
"""

import pyOptions
import scandir


class RinexStruct():

    def __init__(self, cnn):

        self.cnn = cnn

        # read the structure definition table
        levels = cnn.query('SELECT rinex_tank_struct.*, keys.* FROM rinex_tank_struct '
                           'LEFT JOIN keys ON keys."KeyCode" = rinex_tank_struct."KeyCode" ORDER BY "Level"')
        self.levels = levels.dictresult()

        keys = cnn.query('SELECT * FROM keys')
        self.keys = keys.dictresult()

        # read the station and network tables
        networks = cnn.query('SELECT * FROM networks')
        self.networks = networks.dictresult()

        stations = cnn.query('SELECT * FROM stations')
        self.stations = stations.dictresult()

        self.Config = pyOptions.ReadOptions('gnss_data.cfg')

    def insert_rinex(self, record=None, rinexobj=None):
        """
        Insert a RINEX record and file into the database and archive. If only record is provided, only insert into db
        If only rinexobj is provided, then RinexRecord of rinexobj is used for the insert. If both are given, then
        RinexRecord overrides the passed record.
        :param record: a RinexRecord dictionary to make the insert to the db
        :param rinexobj: the pyRinex object containing the file being processed
        :return: True if insertion was successful. False if no insertion was done.
        """

        if record is None and rinexobj is None:
            raise ValueError('insert_rinex exception: both record and rinexobj cannot be None.')

        if rinexobj is not None:
            record = rinexobj.record

        copy_succeeded = False
        archived_crinex = ''

        # check if record exists in the database
        if not self.get_rinex_record(NetworkCode=record['NetworkCode'],
                                     StationCode=record['StationCode'],
                                     ObservationYear=record['ObservationYear'],
                                     ObservationDOY=record['ObservationDOY'],
                                     Interval=record['Interval'],
                                     Completion=float('%.3f' % record['Completion'])):
            # no record, proceed

            # check if we need to perform any rinex operations. We might be inserting a new record, but it may just be
            # a ScanRinex op where we don't copy the file into the archive
            if rinexobj is not None:
                # is the rinex object correctly named?
                rinexobj.apply_file_naming_convention()
                # update the record to the (possible) new name
                record['Filename'] = rinexobj.rinex

            self.cnn.begin_transac()

            try:
                self.cnn.insert('rinex', record)

                if rinexobj is not None:
                    # a rinexobj was passed, copy it into the archive.

                    path2archive = os.path.join(self.Config.archive_path,
                                                self.build_rinex_path(record['NetworkCode'], record['StationCode'],
                                                                      record['ObservationYear'],
                                                                      record['ObservationDOY'],
                                                                      with_filename=False, rinexobj=rinexobj))

                    # copy fixed version into the archive
                    archived_crinex = rinexobj.compress_local_copyto(path2archive)
                    copy_succeeded = True
                    # get the rinex filename to update the database
                    rnx = rinexobj.to_format(os.path.basename(archived_crinex), pyRinex.TYPE_RINEX)

                    if rnx != rinexobj.rinex:
                        # update the table with the filename (always force with step)
                        self.cnn.query('UPDATE rinex SET "Filename" = \'%s\' '
                                       'WHERE "NetworkCode" = \'%s\' '
                                       'AND "StationCode" = \'%s\' '
                                       'AND "ObservationYear" = %i '
                                       'AND "ObservationDOY" = %i '
                                       'AND "Interval" = %i '
                                       'AND "Completion" = %.3f '
                                       'AND "Filename" = \'%s\'' %
                                       (rnx,
                                        record['NetworkCode'],
                                        record['StationCode'],
                                        record['ObservationYear'],
                                        record['ObservationDOY'],
                                        record['Interval'],
                                        record['Completion'],
                                        record['Filename']))

                    event = pyEvents.Event(Description='A new RINEX was added to the archive: %s' % record['Filename'],
                                           NetworkCode=record['NetworkCode'],
                                           StationCode=record['StationCode'],
                                           Year=record['ObservationYear'],
                                           DOY=record['ObservationDOY'])
                else:
                    event = pyEvents.Event(Description='Archived CRINEX file %s added to the database.' %
                                                       record['Filename'],
                                           NetworkCode=record['NetworkCode'],
                                           StationCode=record['StationCode'],
                                           Year=record['ObservationYear'],
                                           DOY=record['ObservationDOY'])

                self.cnn.insert_event(event)

            except Exception:
                self.cnn.rollback_transac()

                if rinexobj and copy_succeeded:
                    # transaction rolled back due to error. If file made into the archive, delete it.
                    os.remove(archived_crinex)

                raise

            self.cnn.commit_transac()

            return True
        else:
            # record already existed
            return False

    def remove_rinex(self, record, move_to_dir=None):
        # function to remove a file from the archive
        # should receive a rinex record
        # if move_to is None, file is deleted
        # otherwise, moves file to specified location
        try:
            self.cnn.begin_transac()
            # propagate the deletes
            # check if this rinex file is the file that was processed and used for solutions
            rs = self.cnn.query(
                'SELECT * FROM rinex_proc WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND '
                '"ObservationYear" = %i AND "ObservationDOY" = %i'
                % (record['NetworkCode'], record['StationCode'],
                   record['ObservationYear'], record['ObservationDOY']))

            if rs.ntuples() > 0:
                self.cnn.query(
                    'DELETE FROM gamit_soln WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND '
                    '"Year" = %i AND "DOY" = %i'
                    % (record['NetworkCode'], record['StationCode'],
                       record['ObservationYear'], record['ObservationDOY']))

                self.cnn.query(
                    'DELETE FROM ppp_soln WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND '
                    '"Year" = %i AND "DOY" = %i'
                    % (record['NetworkCode'], record['StationCode'],
                       record['ObservationYear'], record['ObservationDOY']))

            # get the filename
            rinex_path = self.build_rinex_path(record['NetworkCode'], record['StationCode'],
                                               record['ObservationYear'], record['ObservationDOY'],
                                               filename=record['Filename'])

            rinex_path = os.path.join(self.Config.archive_path, rinex_path)

            # delete the rinex record
            self.cnn.query(
                'DELETE FROM rinex WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND '
                '"ObservationYear" = %i AND "ObservationDOY" = %i AND "Filename" = \'%s\''
                % (record['NetworkCode'], record['StationCode'], record['ObservationYear'],
                   record['ObservationDOY'], record['Filename']))

            if os.path.isfile(rinex_path):
                if move_to_dir:

                    filename = Utils.move(rinex_path, os.path.join(move_to_dir, os.path.basename(rinex_path)))
                    description = 'RINEX %s was removed from the database and archive. ' \
                                  'File moved to %s. See next events for reason.' % (record['Filename'], filename)
                else:

                    os.remove(rinex_path)
                    description = 'RINEX %s was removed from the database and archive. ' \
                                  'File was deleted. See next events for reason.' % (record['Filename'])

            else:
                description = 'RINEX %s was removed from the database and archive. File was NOT found in the archive ' \
                              'so no deletion was performed. See next events for reason.' % (record['Filename'])

            # insert an event
            event = pyEvents.Event(
                Description=description,
                NetworkCode=record['NetworkCode'],
                StationCode=record['StationCode'],
                EventType='info',
                Year=record['ObservationYear'],
                DOY=record['ObservationDOY'])

            self.cnn.insert_event(event)

            self.cnn.commit_transac()
        except Exception:
            self.cnn.rollback_transac()
            raise

    def get_rinex_record(self, **kwargs):
        """
        Retrieve a single or multiple records from the rinex table given a set parameters. If parameters are left empty,
        it wil return all records matching the specified criteria. Each parameter acts like a filter, narrowing down the
        records returned by the function. The default behavior is to use tables rinex or rinex_proc depending on the
        provided parameters. E.g. if Interval, Completion and Filename are all left blank, the function will return the
        records using rinex_proc. Otherwise, the rinex table will be used.
        :return: a dictionary will the records matching the provided parameters
        """

        if any(param in ['Interval', 'Completion', 'Filename'] for param in list(kwargs.keys())):
            table = 'rinex'
        else:
            table = 'rinex_proc'

        # get table fields
        fields = self.cnn.get_columns(table)
        psql = []

        # parse args
        for key in kwargs:

            if key not in [field for field in list(fields.keys())]:
                raise ValueError('Parameter ' + key + ' is not a field in table ' + table)

            if key is not 'ObservationFYear':
                # avoid FYear due to round off problems
                arg = kwargs[key]

                if 'character' in fields[key]:
                    psql += ['"%s" = \'%s\'' % (key, arg)]

                elif 'numeric' in fields[key]:
                    psql += ['"%s" = %f' % (key, arg)]

        sql = 'SELECT * FROM %s ' % table
        sql += 'WHERE ' + ' AND '.join(psql) if psql else ''

        return self.cnn.query(sql).dictresult()

    def check_directory_struct(self, ArchivePath, NetworkCode, StationCode, date):

        path = self.build_rinex_path(NetworkCode, StationCode, date.year, date.doy, False)

        try:
            if not os.path.isdir(os.path.join(ArchivePath, path)):
                os.makedirs(os.path.join(ArchivePath, path))
        except OSError:
            # race condition: two prcesses trying to create the same folder
            pass

        return

    @staticmethod
    def parse_crinex_filename(filename):
        # parse a crinex filename
        sfile = re.findall(r'(\w{4})(\d{3})(\w{1})\.(\d{2})([d])\.[Z]$', filename)

        if sfile:
            return sfile[0]
        else:
            return []

    @staticmethod
    def parse_rinex_filename(filename):
        # parse a rinex filename
        sfile = re.findall(r'(\w{4})(\d{3})(\w{1})\.(\d{2})([o])$', filename)

        if sfile:
            return sfile[0]
        else:
            return []

    def scan_archive_struct(self, rootdir, progress_bar=None):

        self.archiveroot = rootdir

        rnx = []
        path2rnx = []
        fls = []
        for path, _, files in scandir.walk(rootdir):
            for file in files:
                if progress_bar is not None:
                    progress_bar.set_postfix(crinex=os.path.join(path, file).rsplit(rootdir + '/')[1])
                    progress_bar.update()

                # DDG issue #15: match the name of the file to a valid rinex filename
                if self.parse_crinex_filename(file):
                    # only add valid rinex compressed files
                    fls.append(file)
                    rnx.append(os.path.join(path, file).rsplit(rootdir + '/')[1])
                    path2rnx.append(os.path.join(path, file))

                else:
                    if file.endswith('DS_Store') or file[0:2] == '._':
                        # delete the stupid mac files
                        try:
                            os.remove(os.path.join(path, file))
                        except Exception:
                            sys.exc_clear()

        return rnx, path2rnx, fls

    def scan_archive_struct_stninfo(self, rootdir):

        # same as scan archive struct but looks for station info files
        self.archiveroot = rootdir

        stninfo = []
        path2stninfo = []
        for path, dirs, files in scandir.walk(rootdir):
            for file in files:
                if file.endswith(".info"):
                    # only add valid rinex compressed files
                    stninfo.append(os.path.join(path, file).rsplit(rootdir + '/')[1])
                    path2stninfo.append(os.path.join(path, file))
                else:
                    if file.endswith('DS_Store') or file[0:2] == '._':
                        # delete the stupid mac files
                        try:
                            os.remove(os.path.join(path, file))
                        except Exception:
                            sys.exc_clear()

        return stninfo, path2stninfo

    def build_rinex_path(self, NetworkCode, StationCode, ObservationYear, ObservationDOY,
                         with_filename=True, filename=None, rinexobj=None):
        """
        Function to get the location in the archive of a rinex file. It has two modes of operation:
        1) retrieve an existing rinex file, either specific or the rinex for processing
        (most complete, largest interval) or a specific rinex file (already existing in the rinex table).
        2) To get the location of a potential file (probably used for injecting a new file in the archive. No this mode,
        filename has no effect.
        :param NetworkCode: NetworkCode of the station being retrieved
        :param StationCode: StationCode of the station being retrieved
        :param ObservationYear: Year of the rinex file being retrieved
        :param ObservationDOY: DOY of the rinex file being retrieved
        :param with_filename: if set, returns a path including the filename. Otherwise, just returns the path
        :param filename: name of a specific file to search in the rinex table
        :param rinexobj: a pyRinex object to pull the information from (to fill the achive keys).
        :return: a path with or without filename
        """
        if not rinexobj:
            # not an insertion (user wants the rinex path of existing file)
            # build the levels struct
            sql_list = []
            for level in self.levels:
                sql_list.append('"' + level['rinex_col_in'] + '"')

            sql_list.append('"Filename"')

            sql_string = ", ".join(sql_list)

            if filename:
                if self.parse_crinex_filename(filename):
                    filename = filename.replace('d.Z', 'o')

                # if filename is set, user requesting a specific file: query rinex table
                rs = self.cnn.query('SELECT ' + sql_string + ' FROM rinex WHERE "NetworkCode" = \'' +
                                    NetworkCode + '\' AND "StationCode" = \'' + StationCode +
                                    '\' AND "ObservationYear" = ' + str(ObservationYear) + ' AND "ObservationDOY" = ' +
                                    str(ObservationDOY) + ' AND "Filename" = \'' + filename + '\'')
            else:
                # if filename is NOT set, user requesting a the processing file: query rinex_proc
                rs = self.cnn.query(
                    'SELECT ' + sql_string + ' FROM rinex_proc WHERE "NetworkCode" = \'' + NetworkCode +
                    '\' AND "StationCode" = \'' + StationCode + '\' AND "ObservationYear" = ' + str(
                        ObservationYear) + ' AND "ObservationDOY" = ' + str(ObservationDOY))

            if rs.ntuples() != 0:
                field = rs.dictresult()[0]
                keys = []
                for level in self.levels:
                    keys.append(str(field[level['rinex_col_in']]).zfill(level['TotalChars']))

                if with_filename:
                    # database stores rinex, we want crinez
                    retval = "/".join(keys) + "/" + \
                             field['Filename'].replace(field['Filename'].split('.')[-1],
                                                       field['Filename'].split('.')[-1].replace('o', 'd.Z'))
                    if retval[0] == os.path.sep:
                        return retval[1:]
                    else:
                        return retval

                else:
                    return "/".join(keys)
            else:
                return None
        else:
            # new file (get the path where it's supposed to go)
            keys = []
            for level in self.levels:
                if level['isnumeric'] == '1':
                    kk = str(rinexobj.record[level['rinex_col_in']]).zfill(level['TotalChars'])
                else:
                    kk = str(rinexobj.record[level['rinex_col_in']])

                if len(kk) != level['TotalChars']:
                    raise ValueError('Invalid record \'%s\' for key \'%s\'' % (kk, level['KeyCode']))

                keys += [kk]

            path = '/'.join(keys)
            valid, _ = self.parse_archive_keys(os.path.join(path, rinexobj.crinez),
                                               tuple([item['KeyCode'] for item in self.levels]))

            if valid:
                if with_filename:
                    return os.path.join(path, rinexobj.crinez)
                else:
                    return path
            else:
                raise ValueError('Invalid path result: %s' % path)

    def parse_archive_keys(self, path, key_filter=()):

        try:
            pathparts = path.split('/')
            filename = path.split('/')[-1]

            # check the number of levels in pathparts against the number of expected levels
            # subtract one for the filename
            if len(pathparts) - 1 != len(self.levels):
                return False, {}

            if not filename.endswith('.info'):
                fileparts = self.parse_crinex_filename(filename)
            else:
                # parsing a station info file, fill with dummy the doy and year
                fileparts = ('dddd', '1', '0', '80')

            if fileparts:
                keys = dict()

                # fill in all the possible keys using the crinex file info
                keys['station'] = fileparts[0]
                keys['doy'] = int(fileparts[1])
                keys['session'] = fileparts[2]
                keys['year'] = int(fileparts[3])
                keys['network'] = 'rnx'

                # now look in the different levels to match more data (or replace filename keys)
                for key in self.levels:

                    if len(pathparts[key['Level'] - 1]) != key['TotalChars']:
                        return False, {}

                    if key['isnumeric'] == '1':
                        keys[key['KeyCode']] = int(pathparts[key['Level'] - 1])
                    else:
                        keys[key['KeyCode']] = pathparts[key['Level'] - 1].lower()

                # check date is valid and also fill day and month keys
                date = pyDate.Date(year=keys['year'], doy=keys['doy'])
                keys['day'] = date.day
                keys['month'] = date.month

                return True, {key: keys[key] for key in list(keys.keys()) if key in key_filter}
            else:
                return False, {}

        except Exception as e:
            return False, {}


"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez

This class fetches broadcast orbits from the brdc folder (specified in the gnss_data.cfg file) passed as an argument (brdc_archive)
"""

import pyProducts


class pyBrdcException(pyProducts.pyProductsException):
    pass


class GetBrdcOrbits(pyProducts.OrbitalProduct):

    def __init__(self, brdc_archive, date, copyto, no_cleanup=False):

        self.brdc_archive = brdc_archive
        self.brdc_path = None
        self.no_cleanup = no_cleanup

        # try both zipped and unzipped n files
        self.brdc_filename = 'brdc' + str(date.doy).zfill(3) + '0.' + str(date.year)[2:4] + 'n'

        try:
            pyProducts.OrbitalProduct.__init__(self, self.brdc_archive, date, self.brdc_filename, copyto)
            self.brdc_path = self.file_path

        except pyProducts.pyProductsExceptionUnreasonableDate:
            raise
        except pyProducts.pyProductsException:
            raise pyBrdcException(
                'Could not find the broadcast ephemeris file for ' + str(date.year) + ' ' + str(date.doy))

        return

    def cleanup(self):
        if self.brdc_path and not self.no_cleanup:
            # delete files
            if os.path.isfile(self.brdc_path):
                os.remove(self.brdc_path)

        return

    def __del__(self):
        self.cleanup()
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self


# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" Bunch is a subclass of dict with attribute-style access.

    >>> b = Bunch()
    >>> b.hello = 'world'
    >>> b.hello
    'world'
    >>> b['hello'] += "!"
    >>> b.hello
    'world!'
    >>> b.foo = Bunch(lol=True)
    >>> b.foo.lol
    True
    >>> b.foo is b['foo']
    True

    It is safe to import * from this module:

        __all__ = ('Bunch', 'bunchify','unbunchify')

    un/bunchify provide dictionary conversion; Bunches can also be
    converted via Bunch.to/fromDict().
"""

__version__ = '1.0.1'
VERSION = tuple(map(int, __version__.split('.')))

__all__ = ('Bunch', 'bunchify', 'unbunchify',)


class Bunch(dict):
    """ A dictionary that provides attribute-style access.

        >>> b = Bunch()
        >>> b.hello = 'world'
        >>> b.hello
        'world'
        >>> b['hello'] += "!"
        >>> b.hello
        'world!'
        >>> b.foo = Bunch(lol=True)
        >>> b.foo.lol
        True
        >>> b.foo is b['foo']
        True

        A Bunch is a subclass of dict; it supports all the methods a dict does...

        >>> sorted(b.keys())
        ['foo', 'hello']

        Including update()...

        >>> b.update({ 'ponies': 'are pretty!' }, hello=42)
        >>> print (repr(b))
        Bunch(foo=Bunch(lol=True), hello=42, ponies='are pretty!')

        As well as iteration...

        >>> [ (k,b[k]) for k in b ]
        [('ponies', 'are pretty!'), ('foo', Bunch(lol=True)), ('hello', 42)]

        And "splats".

        >>> "The {knights} who say {ni}!".format(**Bunch(knights='lolcats', ni='can haz'))
        'The lolcats who say can haz!'

        See unbunchify/Bunch.toDict, bunchify/Bunch.fromDict for notes about conversion.
    """

    def __contains__(self, k):
        """ >>> b = Bunch(ponies='are pretty!')
            >>> 'ponies' in b
            True
            >>> 'foo' in b
            False
            >>> b['foo'] = 42
            >>> 'foo' in b
            True
            >>> b.hello = 'hai'
            >>> 'hello' in b
            True
            >>> b[None] = 123
            >>> None in b
            True
            >>> b[False] = 456
            >>> False in b
            True
        """
        try:
            return dict.__contains__(self, k) or hasattr(self, k)
        except Exception:
            return False

    # only called if k not found in normal places
    def __getattr__(self, k):
        """ Gets key if it exists, otherwise throws AttributeError.

            nb. __getattr__ is only called if key is not found in normal places.

            >>> b = Bunch(bar='baz', lol={})
            >>> b.foo
            Traceback (most recent call last):
                ...
            AttributeError: foo

            >>> b.bar
            'baz'
            >>> getattr(b, 'bar')
            'baz'
            >>> b['bar']
            'baz'

            >>> b.lol is b['lol']
            True
            >>> b.lol is getattr(b, 'lol')
            True
        """
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    def __setattr__(self, k, v):
        """ Sets attribute k if it exists, otherwise sets key k. A KeyError
            raised by set-item (only likely if you subclass Bunch) will
            propagate as an AttributeError instead.

            >>> b = Bunch(foo='bar', this_is='useful when subclassing')
            >>> b.values                            #doctest: +ELLIPSIS
            <built-in method values of Bunch object at 0x...>
            >>> b.values = 'uh oh'
            >>> b.values
            'uh oh'
            >>> b['values']
            Traceback (most recent call last):
                ...
            KeyError: 'values'
        """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def __delattr__(self, k):
        """ Deletes attribute k if it exists, otherwise deletes key k. A KeyError
            raised by deleting the key--such as when the key is missing--will
            propagate as an AttributeError instead.

            >>> b = Bunch(lol=42)
            >>> del b.values
            Traceback (most recent call last):
                ...
            AttributeError: 'Bunch' object attribute 'values' is read-only
            >>> del b.lol
            >>> b.lol
            Traceback (most recent call last):
                ...
            AttributeError: lol
        """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)

    def toDict(self):
        """ Recursively converts a bunch back into a dictionary.

            >>> b = Bunch(foo=Bunch(lol=True), hello=42, ponies='are pretty!')
            >>> b.toDict()
            {'ponies': 'are pretty!', 'foo': {'lol': True}, 'hello': 42}

            See unbunchify for more info.
        """
        return unbunchify(self)

    def __repr__(self):
        """ Invertible* string-form of a Bunch.

            >>> b = Bunch(foo=Bunch(lol=True), hello=42, ponies='are pretty!')
            >>> print (repr(b))
            Bunch(foo=Bunch(lol=True), hello=42, ponies='are pretty!')
            >>> eval(repr(b))
            Bunch(foo=Bunch(lol=True), hello=42, ponies='are pretty!')

            (*) Invertible so long as collection contents are each repr-invertible.
        """
        keys = list(self.keys())
        keys.sort()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys])
        return '%s(%s)' % (self.__class__.__name__, args)

    @staticmethod
    def fromDict(d):
        """ Recursively transforms a dictionary into a Bunch via copy.

            >>> b = Bunch.fromDict({'urmom': {'sez': {'what': 'what'}}})
            >>> b.urmom.sez.what
            'what'

            See bunchify for more info.
        """
        return bunchify(d)


# While we could convert abstract types like Mapping or Iterable, I think
# bunchify is more likely to "do what you mean" if it is conservative about
# casting (ex: isinstance(str,Iterable) == True ).
#
# Should you disagree, it is not difficult to duplicate this function with
# more aggressive coercion to suit your own purposes.

def bunchify(x):
    """ Recursively transforms a dictionary into a Bunch via copy.

        >>> b = bunchify({'urmom': {'sez': {'what': 'what'}}})
        >>> b.urmom.sez.what
        'what'

        bunchify can handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.

        >>> b = bunchify({ 'lol': ('cats', {'hah':'i win again'}),
        ...         'hello': [{'french':'salut', 'german':'hallo'}] })
        >>> b.hello[0].french
        'salut'
        >>> b.lol[1].hah
        'i win again'

        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    if isinstance(x, dict):
        return Bunch((k, bunchify(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(bunchify(v) for v in x)
    else:
        return x


def unbunchify(x):
    """ Recursively converts a Bunch into a dictionary.

        >>> b = Bunch(foo=Bunch(lol=True), hello=42, ponies='are pretty!')
        >>> unbunchify(b)
        {'ponies': 'are pretty!', 'foo': {'lol': True}, 'hello': 42}

        unbunchify will handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.

        >>> b = Bunch(foo=['bar', Bunch(lol=True)], hello=42,
        ...         ponies=('are pretty!', Bunch(lies='are trouble!')))
        >>> unbunchify(b) #doctest: +NORMALIZE_WHITESPACE
        {'ponies': ('are pretty!', {'lies': 'are trouble!'}),
         'foo': ['bar', {'lol': True}], 'hello': 42}

        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    if isinstance(x, dict):
        return dict((k, unbunchify(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(unbunchify(v) for v in x)
    else:
        return x


### Serialization

try:
    try:
        import json
    except ImportError:
        import simplejson as json


    def toJSON(self, **options):
        """ Serializes this Bunch to JSON. Accepts the same keyword options as `json.dumps()`.

            >>> b = Bunch(foo=Bunch(lol=True), hello=42, ponies='are pretty!')
            >>> json.dumps(b)
            '{"ponies": "are pretty!", "foo": {"lol": true}, "hello": 42}'
            >>> b.toJSON()
            '{"ponies": "are pretty!", "foo": {"lol": true}, "hello": 42}'
        """
        return json.dumps(self, **options)


    Bunch.toJSON = toJSON

except ImportError:
    pass

try:
    # Attempt to register ourself with PyYAML as a representer
    import yaml
    from yaml.representer import Representer, SafeRepresenter


    def from_yaml(loader, node):
        """ PyYAML support for Bunches using the tag `!bunch` and `!bunch.Bunch`.

            >>> import yaml
            >>> yaml.load('''
            ... Flow style: !bunch.Bunch { Clark: Evans, Brian: Ingerson, Oren: Ben-Kiki }
            ... Block style: !bunch
            ...   Clark : Evans
            ...   Brian : Ingerson
            ...   Oren  : Ben-Kiki
            ... ''') #doctest: +NORMALIZE_WHITESPACE
            {'Flow style': Bunch(Brian='Ingerson', Clark='Evans', Oren='Ben-Kiki'),
             'Block style': Bunch(Brian='Ingerson', Clark='Evans', Oren='Ben-Kiki')}

            This module registers itself automatically to cover both Bunch and any
            subclasses. Should you want to customize the representation of a subclass,
            simply register it with PyYAML yourself.
        """
        data = Bunch()
        yield data
        value = loader.construct_mapping(node)
        data.update(value)


    def to_yaml_safe(dumper, data):
        """ Converts Bunch to a normal mapping node, making it appear as a
            dict in the YAML output.

            >>> b = Bunch(foo=['bar', Bunch(lol=True)], hello=42)
            >>> import yaml
            >>> yaml.safe_dump(b, default_flow_style=True)
            '{foo: [bar, {lol: true}], hello: 42}\\n'
        """
        return dumper.represent_dict(data)


    def to_yaml(dumper, data):
        """
        Converts Bunch to a representation node.

            >>> b = Bunch(foo=['bar', Bunch(lol=True)], hello=42)
            >>> import yaml
            >>> yaml.dump(b, default_flow_style=True)
            '!bunch.Bunch {foo: [bar, !bunch.Bunch {lol: true}], hello: 42}\\n'
        """
        return dumper.represent_mapping(u('!bunch.Bunch'), data)


    yaml.add_constructor(u('!bunch'), from_yaml)
    yaml.add_constructor(u('!bunch.Bunch'), from_yaml)

    SafeRepresenter.add_representer(Bunch, to_yaml_safe)
    SafeRepresenter.add_multi_representer(Bunch, to_yaml_safe)

    Representer.add_representer(Bunch, to_yaml)
    Representer.add_multi_representer(Bunch, to_yaml)


    # Instance methods for YAML conversion
    def toYAML(self, **options):
        """ Serializes this Bunch to YAML, using `yaml.safe_dump()` if
            no `Dumper` is provided. See the PyYAML documentation for more info.

            >>> b = Bunch(foo=['bar', Bunch(lol=True)], hello=42)
            >>> import yaml
            >>> yaml.safe_dump(b, default_flow_style=True)
            '{foo: [bar, {lol: true}], hello: 42}\\n'
            >>> b.toYAML(default_flow_style=True)
            '{foo: [bar, {lol: true}], hello: 42}\\n'
            >>> yaml.dump(b, default_flow_style=True)
            '!bunch.Bunch {foo: [bar, !bunch.Bunch {lol: true}], hello: 42}\\n'
            >>> b.toYAML(Dumper=yaml.Dumper, default_flow_style=True)
            '!bunch.Bunch {foo: [bar, !bunch.Bunch {lol: true}], hello: 42}\\n'
        """
        opts = dict(indent=4, default_flow_style=False)
        opts.update(options)
        if 'Dumper' not in opts:
            return yaml.safe_dump(self, **opts)
        else:
            return yaml.dump(self, **opts)


    def fromYAML(*args, **kwargs):
        return bunchify(yaml.load(*args, **kwargs))


    Bunch.toYAML = toYAML
    Bunch.fromYAML = staticmethod(fromYAML)

except ImportError:
    pass

"""
Project: Parallel.Archive
Date: 2/22/17 3:27 PM
Author: Demian D. Gomez

This class fetches statellite clock files from the orbits folder (specified in the gnss_data.cfg file) passed as an argument (clk_archive)
"""

import pyProducts


class pyClkException(pyProducts.pyProductsException):
    pass


class GetClkFile(pyProducts.OrbitalProduct):

    def __init__(self, clk_archive, date, sp3types, copyto, no_cleanup=False):

        # try both compressed and non-compressed sp3 files
        # loop through the types of sp3 files to try
        self.clk_path = None
        self.no_cleanup = no_cleanup

        for sp3type in sp3types:
            self.clk_filename = sp3type + date.wwwwd() + '.clk'

            try:
                pyProducts.OrbitalProduct.__init__(self, clk_archive, date, self.clk_filename, copyto)
                self.clk_path = self.file_path
                break
            except pyProducts.pyProductsExceptionUnreasonableDate:
                raise
            except pyProducts.pyProductsException:
                # if the file was not found, go to next
                pass

        # if we get here and self.sp3_path is still none, then no type of sp3 file was found
        if self.clk_path is None:
            raise pyClkException(
                'Could not find a valid clocks file for ' + date.wwwwd() + ' using any of the provided sp3 types')

        return

    def cleanup(self):
        if self.clk_path and not self.no_cleanup:
            # delete files
            if os.path.isfile(self.clk_path):
                os.remove(self.clk_path)

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self


"""
Project: Parallel.Archive
Date: 2/23/17 9:28 AM
Author: Demian D. Gomez

Not really used for the moment...
"""

import gzip

import magic
import zip_file


class Compress():

    def __init__(self, file, dest):

        pymagic = magic.Magic(uncompress=False)

        filetype = pymagic.from_file(file)

        if 'zip archive' in filetype.lower():

            file = zip_file.ZipFile(file)
            file.extractall(dest)

        elif 'gzip' in filetype.lower():

            f = gzip.open(dest)
            sp3file = f.read()
            f.close()


"""
Project: Parallel.Archive
Date: 2/23/17 9:28 AM
Author: Abel Brown
Modified by: Demian D. Gomez

Class that handles all the date conversions betweem different systems and formats

"""

from json import JSONEncoder


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default


class pyDateException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


def yeardoy2fyear(year, doy, hour=12, minute=0, second=0):
    # parse to integers (defensive)
    year = int(year)
    doy = int(doy)
    hour = int(hour)

    # default number of days in a year
    diy = 365

    # check for leap years
    if year % 4 == 0:
        diy += 1.

    # make sure day of year is valid
    if doy < 1 or doy > diy:
        raise pyDateException('invalid day of year')

    # compute the fractional year
    fractionalYear = year + ((doy - 1) + hour / 24. + minute / 1440. + second / 86400.) / diy

    # that's all ...
    return fractionalYear


def fyear2yeardoy(fyear):
    year = math.floor(fyear)
    fractionOfyear = fyear - year

    if year % 4 == 0:
        days = 366
    else:
        days = 365

    doy = math.floor(days * fractionOfyear) + 1
    hh = (days * fractionOfyear - math.floor(days * fractionOfyear)) * 24.
    hour = math.floor(hh)
    mm = (hh - math.floor(hh)) * 60.
    minute = math.floor(mm)
    ss = (mm - math.floor(mm)) * 60.
    second = math.floor(ss)

    return int(year), int(doy), int(hour), int(minute), int(second)


def date2doy(year, month, day, hour=12, minute=0, second=0):
    # parse to integers (defensive)
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)

    # localized days of year
    if year % 4 == 0:
        lday = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    else:
        lday = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    # compute the day of year
    doy = lday[month - 1] + day

    # finally, compute fractional year
    fyear = yeardoy2fyear(year, doy, hour, minute, second)

    # that's a [w]rap
    return doy, fyear


def doy2date(year, doy):
    # parsem up to integers
    year = int(year)
    doy = int(doy)

    # make note of leap year or not
    isLeapYear = False
    if year % 4 == 0:
        isLeapYear = True

    # make note of valid doy for year
    mxd = 365
    if isLeapYear:
        mxd += 1

    # check doy based on year
    if doy < 1 or doy > mxd:
        raise pyDateException('day of year input is invalid')

    # localized days
    if not isLeapYear:
        fday = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        lday = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    if isLeapYear:
        fday = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
        lday = [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

    # compute the month
    for i in range(0, 12):
        if doy <= lday[i]:
            # remember: zero based indexing!
            month = i + 1
            break

    # compute the day (dont forget zero based indexing)
    day = doy - fday[month - 1] + 1

    return month, day


def date2gpsDate(year, month, day):
    year = int(year)
    month = int(month)
    day = int(day)

    if month <= 2:
        month += 12
        year -= 1

    ut = (day % 1) * 24.
    day = math.floor(day)

    julianDay = math.floor(365.25 * year) \
                + math.floor(30.6001 * (month + 1.)) \
                + day \
                + ut / 24. \
                + 1720981.5

    gpsWeek = math.floor((julianDay - 2444244.5) / 7.)
    gpsWeekDay = (julianDay - 2444244.5) % 7

    # that's a [w]rap
    return int(gpsWeek), int(gpsWeekDay)


def gpsDate2mjd(gpsWeek, gpsWeekDay):
    # parse to integers
    gpsWeek = int(gpsWeek)
    gpsWeekDay = int(gpsWeekDay)

    mjd = (gpsWeek * 7.) + 44244. + gpsWeekDay

    return int(mjd)


def mjd2date(mjd):
    mjd = float(mjd)

    jd = mjd + 2400000.5

    ijd = math.floor(jd + 0.5)

    a = ijd + 32044.
    b = math.floor((4. * a + 3.) / 146097.)
    c = a - math.floor((b * 146097.) / 4.)

    d = math.floor((4. * c + 3.) / 1461.)
    e = c - math.floor((1461. * d) / 4.)
    m = math.floor((5. * e + 2.) / 153.)

    day = e - math.floor((153. * m + 2.) / 5.) + 1.
    month = m + 3. - 12. * math.floor(m / 10.)
    year = b * 100. + d - 4800. + math.floor(m / 10.)

    return int(year), int(month), int(day)


def parse_stninfo(stninfo_datetime):
    sdate = stninfo_datetime.split()

    if int(sdate[2]) > 23:
        sdate[2] = '23'
        sdate[3] = '59'
        sdate[4] = '59'

    if int(sdate[0]) == 9999:
        return None, None, None, None, None
    else:
        return int(sdate[0]), int(sdate[1]), int(sdate[2]), int(sdate[3]), int(sdate[4])


class Date(object):

    def __init__(self, **kwargs):

        # init
        self.mjd = None
        self.fyear = None
        self.year = None
        self.doy = None
        self.day = None
        self.month = None
        self.gpsWeek = None
        self.gpsWeekDay = None
        self.hour = 12  # DDG 03-28-2017: include hour and minute to work with station info object
        self.minute = 0
        self.second = 0

        self.from_stninfo = False

        # parse args
        for key in kwargs:

            arg = kwargs[key]
            key = key.lower()

            if key == 'year':
                if int(arg) < 1900:
                    # the date is in 2 digit format
                    if int(arg) > 80:
                        self.year = int(arg) + 1900
                    else:
                        self.year = int(arg) + 2000
                else:
                    self.year = arg
            elif key == 'doy':
                self.doy = arg
            elif key == 'day':
                self.day = arg
            elif key == 'month':
                self.month = arg
            elif key == 'gpsweek':
                self.gpsWeek = arg
            elif key == 'gpsweekday':
                self.gpsWeekDay = arg
            elif key in ('fyear', 'fractionalyear', 'fracyear'):
                self.fyear = arg
            elif key == 'mjd':
                self.mjd = arg
            elif key == 'hour':  # DDG 03-28-2017: include hour to work with station info object
                self.hour = arg
            elif key == 'minute':  # DDG 03-28-2017: include minute to work with station info object
                self.minute = arg
            elif key == 'second':  # DDG 03-28-2017: include second to work with station info object
                self.second = arg
            elif key == 'datetime':  # DDG 03-28-2017: handle conversion from datetime to pyDate
                if isinstance(arg, datetime):
                    self.day = arg.day
                    self.month = arg.month
                    self.year = arg.year
                    self.hour = arg.hour
                    self.minute = arg.minute
                    self.second = arg.second
                else:
                    raise pyDateException('invalid type for ' + key + '\n')
            elif key == 'stninfo':  # DDG: handle station information records

                self.from_stninfo = True

                if isinstance(arg, str) or isinstance(arg, str):
                    self.year, self.doy, self.hour, self.minute, self.second = parse_stninfo(arg)
                elif isinstance(arg, datetime):
                    self.day = arg.day
                    self.month = arg.month
                    self.year = arg.year
                    self.hour = arg.hour
                    self.minute = arg.minute
                    self.second = arg.second
                elif arg is None:
                    # ok to receive a None argument from the database due to 9999 999 00 00 00 records
                    break
                else:
                    raise pyDateException('invalid type ' + str(type(arg)) + ' for ' + key + '\n')
            else:
                raise pyDateException('unrecognized input arg: ' + key + '\n')

        # make due with what we gots
        if self.year is not None and self.doy is not None:

            # compute the month and day of month
            self.month, self.day = doy2date(self.year, self.doy)

            # compute the fractional year
            self.fyear = yeardoy2fyear(self.year, self.doy, self.hour, self.minute, self.second)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsDate(self.year, self.month, self.day)

            self.mjd = gpsDate2mjd(self.gpsWeek, self.gpsWeekDay)

        elif self.gpsWeek is not None and self.gpsWeekDay is not None:

            # initialize modified julian day from gps date
            self.mjd = gpsDate2mjd(self.gpsWeek, self.gpsWeekDay)

            # compute year, month, and day of month from modified julian day
            self.year, self.month, self.day = mjd2date(self.mjd)

            # compute day of year from month and day of month
            self.doy, self.fyear = date2doy(self.year, self.month, self.day, self.hour, self.minute, self.second)

        elif self.year is not None and self.month is not None and self.day:

            # initialize day of year and fractional year from date
            self.doy, self.fyear = date2doy(self.year, self.month, self.day, self.hour, self.minute, self.second)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsDate(self.year, self.month, self.day)

            # init the modified julian date
            self.mjd = gpsDate2mjd(self.gpsWeek, self.gpsWeekDay)

        elif self.fyear is not None:

            # initialize year and day of year
            self.year, self.doy, self.hour, self.minute, self.second = fyear2yeardoy(self.fyear)

            # set the month and day of month
            # compute the month and day of month
            self.month, self.day = doy2date(self.year, self.doy)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsDate(self.year, self.month, self.day)

            # finally, compute modified jumlian day
            self.mjd = gpsDate2mjd(self.gpsWeek, self.gpsWeekDay)

        elif self.mjd is not None:

            # compute year, month, and day of month from modified julian day
            self.year, self.month, self.day = mjd2date(self.mjd)

            # compute day of year from month and day of month
            self.doy, self.fyear = date2doy(self.year, self.month, self.day, self.hour, self.minute, self.second)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsDate(self.year, self.month, self.day)

        else:
            if not self.from_stninfo:
                # if empty Date object from a station info, it means that it should be printed as 9999 999 00 00 00
                raise pyDateException('not enough independent input args to compute full date')

    def strftime(self):
        return self.datetime().strftime('%Y-%m-%d %H:%M:%S')

    def to_json(self):
        if self.from_stninfo:
            return {'stninfo': str(self)}
        else:
            return {'year': self.year, 'doy': self.doy, 'hour': self.hour, 'minute': self.minute, 'second': self.second}

    def __repr__(self):
        return 'pyDate.Date(' + str(self.year) + ', ' + str(self.doy) + ')'

    def __str__(self):
        if self.year is None:
            return '9999 999 00 00 00'
        else:
            return '%04i %03i %02i %02i %02i' % (self.year, self.doy, self.hour, self.minute, self.second)

    def __lt__(self, date):

        if not isinstance(date, Date):
            raise pyDateException('type: ' + str(type(date)) + ' invalid. Can only compare pyDate.Date objects')

        return self.fyear < date.fyear

    def __le__(self, date):

        if not isinstance(date, Date):
            raise pyDateException('type: ' + str(type(date)) + ' invalid. Can only compare pyDate.Date objects')

        return self.fyear <= date.fyear

    def __gt__(self, date):

        if not isinstance(date, Date):
            raise pyDateException('type: ' + str(type(date)) + ' invalid. Can only compare pyDate.Date objects')

        return self.fyear > date.fyear

    def __ge__(self, date):

        if not isinstance(date, Date):
            raise pyDateException('type: ' + str(type(date)) + ' invalid.  Can only compare pyDate.Date objects')

        return self.fyear >= date.fyear

    def __eq__(self, date):

        if not isinstance(date, Date):
            raise pyDateException('type: ' + str(type(date)) + ' invalid.  Can only compare pyDate.Date objects')

        return self.mjd == date.mjd

    def __ne__(self, date):

        if not isinstance(date, Date):
            raise pyDateException('type: ' + str(type(date)) + ' invalid.  Can only compare pyDate.Date objects')

        return self.mjd != date.mjd

    def __add__(self, ndays):

        if not isinstance(ndays, int):
            raise pyDateException('type: ' + str(type(ndays)) + ' invalid.  Can only add integer number of days')

        return Date(mjd=self.mjd + ndays)

    def __sub__(self, ndays):

        if not (isinstance(ndays, int) or isinstance(ndays, Date)):
            raise pyDateException('type: ' + str(type(ndays)) + ' invalid. Can only subtract integer number of days')

        if isinstance(ndays, int):
            return Date(mjd=self.mjd - ndays)
        else:
            return self.mjd - ndays.mjd

    def __hash__(self):
        # to make the object hashable
        return hash(self.fyear)

    def ddd(self):
        doystr = str(self.doy)
        return "0" * (3 - len(doystr)) + doystr

    def yyyy(self):
        return str(self.year)

    def wwww(self):
        weekstr = str(self.gpsWeek)
        return '0' * (4 - len(weekstr)) + weekstr

    def wwwwd(self):
        return self.wwww() + str(self.gpsWeekDay)

    def yyyymmdd(self):
        return str(self.year) + '/' + str(self.month) + '/' + str(self.day)

    def yyyyddd(self):
        doystr = str(self.doy)
        return str(self.year) + ' ' + doystr.rjust(3, '0')

    def datetime(self):
        if self.year is None:
            return datetime(year=9999, month=1, day=1,
                            hour=1, minute=1, second=1)
        else:
            return datetime(year=self.year, month=self.month, day=self.day,
                            hour=self.hour, minute=self.minute, second=self.second)

    def first_epoch(self, out_format='datetime'):
        if out_format == 'datetime':
            return datetime(year=self.year, month=self.month, day=self.day, hour=0, minute=0, second=0).strftime(
                '%Y-%m-%d %H:%M:%S')
        else:
            _, fyear = date2doy(self.year, self.month, self.day, 0, 0, 0)
            return fyear

    def last_epoch(self, out_format='datetime'):
        if out_format == 'datetime':
            return datetime(year=self.year, month=self.month, day=self.day, hour=23, minute=59, second=59).strftime(
                '%Y-%m-%d %H:%M:%S')
        else:
            _, fyear = date2doy(self.year, self.month, self.day, 23, 59, 59)
            return fyear


"""
Project: Parallel.Archive
Date: 2/23/17 2:52 PM
Author: Demian D. Gomez

This class fetches earth orientation parameters files from the orbits folder (specified in the gnss_data.cfg file) passed as an argument (sp3archive)

"""

import pyProducts


class pyEOPException(pyProducts.pyProductsException):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class GetEOP(pyProducts.OrbitalProduct):

    def __init__(self, sp3archive, date, sp3types, copyto):

        # try both compressed and non-compressed sp3 files
        # loop through the types of sp3 files to try
        self.eop_path = None

        for sp3type in sp3types:

            self.eop_filename = sp3type + date.wwww() + '7.erp'

            try:
                pyProducts.OrbitalProduct.__init__(self, sp3archive, date, self.eop_filename, copyto)
                self.eop_path = self.file_path
                self.type = sp3type
                break

            except pyProducts.pyProductsExceptionUnreasonableDate:
                raise

            # rapid EOP files do not work in NRCAN PPP
            # except pyProducts.pyProductsException:
            #    # rapid orbits do not have 7.erp, try wwwwd.erp

            #    self.eop_filename = sp3type + date.wwwwd() + '.erp'

            #    pyProducts.OrbitalProduct.__init__(self, sp3archive, date, self.eop_filename, copyto)
            #    self.eop_path = self.file_path

            except pyProducts.pyProductsException:
                # if the file was not found, go to next
                pass

        # if we get here and self.sp3_path is still none, then no type of sp3 file was found
        if self.eop_path is None:
            raise pyEOPException(
                'Could not find a valid earth orientation parameters file for gps week ' + date.wwww() +
                ' using any of the provided sp3 types')

        return


"""
Project: Parallel.Archive
Date: 3/3/17 11:27 AM
Author: Demian D. Gomez
"""

import os

import matplotlib
import numpy as np
import pg
from Utils import rotlg2ct
from matplotlib.widgets import Button
from pyBunch import Bunch

if 'DISPLAY' in list(os.environ.keys()):
    if not os.environ['DISPLAY']:
        matplotlib.use('Agg')
else:
    matplotlib.use('Agg')


def tic():
    global tt
    tt = time()


def toc(text):
    global tt
    print(text + ': ' + str(time() - tt))


LIMIT = 2.5

NO_EFFECT = None
UNDETERMINED = -1
GENERIC_JUMP = 0
CO_SEISMIC_DECAY = 1
CO_SEISMIC_JUMP_DECAY = 2

EQ_MIN_DAYS = 15
JP_MIN_DAYS = 5

DEFAULT_RELAXATION = np.array([0.5])
DEFAULT_POL_TERMS = 2
DEFAULT_FREQUENCIES = np.array(
    (1 / 365.25, 1 / (365.25 / 2)))  # (1 yr, 6 months) expressed in 1/days (one year = 365.25)

SIGMA_FLOOR_H = 0.10
SIGMA_FLOOR_V = 0.15

VERSION = '1.0.0'


class pyETMException(Exception):

    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class pyETMException_NoDesignMatrix(pyETMException):
    pass


def distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1 = lon1 * pi / 180
    lat1 = lat1 * pi / 180
    lon2 = lon2 * pi / 180
    lat2 = lat2 * pi / 180
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def to_postgres(dictionary):
    if isinstance(dictionary, dict):
        for key, val in list(dictionary.items()):
            if isinstance(val, np.ndarray):
                dictionary[key] = str(val.flatten().tolist()).replace('[', '{').replace(']', '}')
    else:
        dictionary = str(dictionary.flatten().tolist()).replace('[', '{').replace(']', '}')

    return dictionary


def to_list(dictionary):
    for key, val in list(dictionary.items()):
        if isinstance(val, np.ndarray):
            dictionary[key] = val.tolist()

        if isinstance(val, pyDate.datetime):
            dictionary[key] = val.strftime('%Y-%m-%d %H:%M:%S')

    return dictionary


class PppSoln:
    """"class to extract the PPP solutions from the database"""

    def __init__(self, cnn, NetworkCode, StationCode):

        self.NetworkCode = NetworkCode
        self.StationCode = StationCode
        self.hash = 0

        self.type = 'ppp'

        # get the station from the stations table
        stn = cnn.query('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                        % (NetworkCode, StationCode))

        stn = stn.dictresult()[0]

        if stn['lat'] is not None:
            self.lat = np.array([float(stn['lat'])])
            self.lon = np.array([float(stn['lon'])])
            self.height = np.array([float(stn['height'])])
            self.auto_x = np.array([float(stn['auto_x'])])
            self.auto_y = np.array([float(stn['auto_y'])])
            self.auto_z = np.array([float(stn['auto_z'])])

            x = np.array([float(stn['auto_x'])])
            y = np.array([float(stn['auto_y'])])
            z = np.array([float(stn['auto_z'])])

            if stn['max_dist'] is not None:
                self.max_dist = stn['max_dist']
            else:
                self.max_dist = 20

            # load all the PPP coordinates available for this station
            # exclude ppp solutions in the exclude table and any solution that is more than 20 meters from the simple
            # linear trend calculated above

            self.excluded = cnn.query_float('SELECT "Year", "DOY" FROM ppp_soln_excl '
                                            'WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                                            % (NetworkCode, StationCode))

            self.table = cnn.query_float(
                'SELECT "X", "Y", "Z", "Year", "DOY" FROM ppp_soln p1 '
                'WHERE p1."NetworkCode" = \'%s\' AND p1."StationCode" = \'%s\' ORDER BY "Year", "DOY"'
                % (NetworkCode, StationCode))

            self.table = [item for item in self.table
                          if np.sqrt(np.square(item[0] - x) + np.square(item[1] - y) + np.square(item[2] - z)) <=
                          self.max_dist and item[3:] not in self.excluded]

            self.blunders = [item for item in self.table
                             if np.sqrt(np.square(item[0] - x) + np.square(item[1] - y) + np.square(item[2] - z)) >
                             self.max_dist and item[3:] not in self.excluded]

            self.solutions = len(self.table)

            self.ts_blu = np.array([pyDate.Date(year=item[3], doy=item[4]).fyear for item in self.blunders])

            if self.solutions >= 1:
                a = np.array(self.table)

                self.x = a[:, 0]
                self.y = a[:, 1]
                self.z = a[:, 2]
                self.t = np.array([pyDate.Date(year=item[0], doy=item[1]).fyear for item in a[:, 3:5]])
                self.mjd = np.array([pyDate.Date(year=item[0], doy=item[1]).mjd for item in a[:, 3:5]])

                # continuous time vector for plots
                ts = np.arange(np.min(self.mjd), np.max(self.mjd) + 1, 1)
                self.mjds = ts
                self.ts = np.array([pyDate.Date(mjd=tts).fyear for tts in ts])
            else:
                if len(self.blunders) >= 1:
                    raise pyETMException('No viable PPP solutions available for %s.%s (all blunders!)\n'
                                         '  -> min distance to station coordinate is %.1f meters'
                                         % (NetworkCode, StationCode, np.array([item[5]
                                                                                for item in self.blunders]).min()))
                else:
                    raise pyETMException('No PPP solutions available for %s.%s' % (NetworkCode, StationCode))

            # get a list of the epochs with files but no solutions.
            # This will be shown in the outliers plot as a special marker

            rnx = cnn.query(
                'SELECT r."ObservationFYear" FROM rinex_proc as r '
                'LEFT JOIN ppp_soln as p ON '
                'r."NetworkCode" = p."NetworkCode" AND '
                'r."StationCode" = p."StationCode" AND '
                'r."ObservationYear" = p."Year"    AND '
                'r."ObservationDOY"  = p."DOY"'
                'WHERE r."NetworkCode" = \'%s\' AND r."StationCode" = \'%s\' AND '
                'p."NetworkCode" IS NULL' % (NetworkCode, StationCode))

            self.rnx_no_ppp = rnx.getresult()

            self.ts_ns = np.array([item for item in self.rnx_no_ppp])

            self.completion = 100. - float(len(self.ts_ns)) / float(len(self.ts_ns) + len(self.t)) * 100.

            ppp_hash = cnn.query_float('SELECT sum(hash) FROM ppp_soln p1 '
                                       'WHERE p1."NetworkCode" = \'%s\' AND p1."StationCode" = \'%s\''
                                       % (NetworkCode, StationCode))

            self.hash = crc32(bytearray((str(len(self.t)) + str(len(self.blunders))) + ' ' + str(self.auto_x) + \
                                        str(self.auto_y) + str(self.auto_z) + str(ts[0]) + ' ' + str(ts[-1]) + ' ' + \
                                        str(ppp_hash[0][0]), 'utf-8'))

        else:
            raise pyETMException('Station %s.%s has no valid metadata in the stations table.'
                                 % (NetworkCode, StationCode))


class GamitSoln:
    """"class to extract the GAMIT polyhedrons from the database"""

    def __init__(self, cnn, polyhedrons, NetworkCode, StationCode):

        self.NetworkCode = NetworkCode
        self.StationCode = StationCode
        self.hash = 0

        self.type = 'gamit'

        # get the station from the stations table
        stn = cnn.query_float('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                              % (NetworkCode, StationCode), as_dict=True)[0]

        if stn['lat'] is not None:
            self.lat = np.array([float(stn['lat'])])
            self.lon = np.array([float(stn['lon'])])
            self.height = np.array([stn['height']])
            self.auto_x = np.array([float(stn['auto_x'])])
            self.auto_y = np.array([float(stn['auto_y'])])
            self.auto_z = np.array([float(stn['auto_z'])])

            if stn['max_dist'] is not None:
                self.max_dist = stn['max_dist']
            else:
                self.max_dist = 20

            self.solutions = len(polyhedrons)

            # blunders
            self.blunders = []
            self.ts_blu = np.array([])

            if self.solutions >= 1:
                a = np.array(polyhedrons, dtype=float)

                if np.sqrt(np.square(np.sum(np.square(a[0, 0:3])))) > 6.3e3:
                    # coordinates given in XYZ
                    nb = np.sqrt(np.square(np.sum(
                        np.square(a[:, 0:3] - np.array([stn['auto_x'], stn['auto_y'], stn['auto_z']])), axis=1))) \
                         <= self.max_dist
                else:
                    # coordinates are differences
                    nb = np.sqrt(np.square(np.sum(np.square(a[:, 0:3]), axis=1))) <= self.max_dist

                if np.any(nb):
                    self.x = a[nb, 0]
                    self.y = a[nb, 1]
                    self.z = a[nb, 2]
                    self.t = np.array([pyDate.Date(year=item[0], doy=item[1]).fyear for item in a[nb, 3:5]])
                    self.mjd = np.array([pyDate.Date(year=item[0], doy=item[1]).mjd for item in a[nb, 3:5]])

                    self.date = [pyDate.Date(year=item[0], doy=item[1]) for item in a[nb, 3:5]]

                    # continuous time vector for plots
                    ts = np.arange(np.min(self.mjd), np.max(self.mjd) + 1, 1)
                    self.mjds = ts
                    self.ts = np.array([pyDate.Date(mjd=tts).fyear for tts in ts])
                else:
                    dd = np.sqrt(np.square(np.sum(
                        np.square(a[:, 0:3] - np.array([stn['auto_x'], stn['auto_y'], stn['auto_z']])), axis=1)))

                    raise pyETMException('No viable GAMIT solutions available for %s.%s (all blunders!)\n'
                                         '  -> min distance to station coordinate is %.1f meters'
                                         % (NetworkCode, StationCode, dd.min()))
            else:
                raise pyETMException('No GAMIT polyhedrons vertices available for %s.%s' % (NetworkCode, StationCode))

            # get a list of the epochs with files but no solutions.
            # This will be shown in the outliers plot as a special marker
            rnx = cnn.query(
                'SELECT r.* FROM rinex_proc as r '
                'LEFT JOIN gamit_soln as p ON '
                'r."NetworkCode" = p."NetworkCode" AND '
                'r."StationCode" = p."StationCode" AND '
                'r."ObservationYear" = p."Year"    AND '
                'r."ObservationDOY"  = p."DOY"'
                'WHERE r."NetworkCode" = \'%s\' AND r."StationCode" = \'%s\' AND '
                'p."NetworkCode" IS NULL' % (NetworkCode, StationCode))

            self.rnx_no_ppp = rnx.dictresult()
            self.ts_ns = np.array([float(item['ObservationFYear']) for item in self.rnx_no_ppp])

            self.completion = 100. - float(len(self.ts_ns)) / float(len(self.ts_ns) + len(self.t)) * 100.

            self.hash = crc32(str(len(self.t) + len(self.blunders)) + ' ' + str(ts[0]) + ' ' + str(ts[-1]))

        else:
            raise pyETMException('Station %s.%s has no valid metadata in the stations table.'
                                 % (NetworkCode, StationCode))


class JumpTable:

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, FitEarthquakes=True, FitGenericJumps=True):

        self.table = []

        # get earthquakes for this station
        self.earthquakes = Earthquakes(cnn, NetworkCode, StationCode, solution, t, FitEarthquakes)

        self.generic_jumps = GenericJumps(cnn, NetworkCode, StationCode, solution, t, FitGenericJumps)

        jumps = self.earthquakes.table + self.generic_jumps.table

        jumps.sort()

        # add the relevant jumps, make sure none are incompatible
        for jump in jumps:
            self.insert_jump(jump)

        # add the "NO_EFFECT" jumps and resort the table
        ne_jumps = [j for j in jumps if j.p.jump_type == NO_EFFECT
                    and j.date > pyDate.Date(fyear=t.min()) < j.date < pyDate.Date(fyear=t.max())]

        self.table += ne_jumps

        self.table.sort()

        self.param_count = sum([jump.param_count for jump in self.table])

        self.constrains = np.array([])

    def insert_jump(self, jump):

        if len(self.table) == 0:
            if jump.p.jump_type != NO_EFFECT:
                self.table.append(jump)
        else:
            # take last jump and compare to adding jump
            jj = self.table[-1]

            if jump.p.jump_type != NO_EFFECT:
                result, decision = jj == jump

                if not result:
                    # jumps are not equal, add it
                    self.table.append(jump)
                else:
                    # decision branches:
                    # 1) decision == jump, remove previous; add jump
                    # 2) decision == jj  , do not add jump (i.e. do nothing)
                    if decision is jump:
                        self.table.pop(-1)
                        self.table.append(jump)

    def get_design_ts(self, t):
        # if function call NOT for inversion, return the columns even if the design matrix is unstable

        A = np.array([])

        # get the design matrix for the jump table
        for jump in self.table:
            if jump.p.jump_type is not NO_EFFECT:
                a = jump.eval(t)

                if a.size:
                    if A.size:
                        # if A is not empty, verify that this jump will not make the matrix singular
                        tA = np.column_stack((A, a))
                        # getting the condition number might trigger divide_zero warning => turn off
                        np.seterr(divide='ignore', invalid='ignore')
                        if np.linalg.cond(tA) < 1e10:
                            # adding this jumps doesn't make the matrix singular
                            A = tA
                        else:
                            # flag this jump by setting its type = None
                            jump.remove()
                            warnings.warn('%s had to be removed due to high condition number' % str(jump))
                    else:
                        A = a

        return A

    def load_parameters(self, params, sigmas):

        for jump in self.table:
            jump.load_parameters(params=params, sigmas=sigmas)

    def print_parameters(self):

        output_n = ['Year     Relx    [mm] Mag']
        output_e = ['Year     Relx    [mm] Mag']
        output_u = ['Year     Relx    [mm] Mag']

        for jump in self.table:

            if jump.p.jump_type is not NO_EFFECT:

                # relaxation counter
                rx = 0

                for j, p in enumerate(np.arange(jump.param_count)):
                    psc = jump.p.params[:, p]

                    if j == 0 and jump.p.jump_type in (GENERIC_JUMP, CO_SEISMIC_JUMP_DECAY):
                        output_n.append('{}      {:>7.1f} {}'.format(jump.date.yyyyddd(), psc[0] * 1000.0,
                                                                     jump.magnitude))
                        output_e.append('{}      {:>7.1f} {}'.format(jump.date.yyyyddd(), psc[1] * 1000.0,
                                                                     jump.magnitude))
                        output_u.append('{}      {:>7.1f} {}'.format(jump.date.yyyyddd(), psc[2] * 1000.0,
                                                                     jump.magnitude))
                    else:

                        output_n.append('{} {:4.2f} {:>7.1f} {}'.format(jump.date.yyyyddd(), jump.p.relaxation[rx],
                                                                        psc[0] * 1000.0, jump.magnitude))
                        output_e.append('{} {:4.2f} {:>7.1f} {}'.format(jump.date.yyyyddd(), jump.p.relaxation[rx],
                                                                        psc[1] * 1000.0, jump.magnitude))
                        output_u.append('{} {:4.2f} {:>7.1f} {}'.format(jump.date.yyyyddd(), jump.p.relaxation[rx],
                                                                        psc[2] * 1000.0, jump.magnitude))
                        # relaxation counter
                        rx += 1

        if len(output_n) > 22:
            output_n = output_n[0:22] + ['Table too long to print!']
            output_e = output_e[0:22] + ['Table too long to print!']
            output_u = output_u[0:22] + ['Table too long to print!']

        return '\n'.join(output_n), '\n'.join(output_e), '\n'.join(output_u)


class EtmFunction(object):

    def __init__(self, **kwargs):

        self.p = Bunch()

        self.p.NetworkCode = kwargs['NetworkCode']
        self.p.StationCode = kwargs['StationCode']
        self.p.soln = kwargs['solution']

        self.p.params = np.array([])
        self.p.sigmas = np.array([])
        self.p.object = ''
        self.p.metadata = None
        self.p.hash = 0

        self.param_count = 0
        self.column_index = np.array([])
        self.format_str = ''

    def load_parameters(self, **kwargs):

        params = kwargs['params']
        sigmas = kwargs['sigmas']

        if params.ndim == 1:
            # parameters coming from the database, reshape
            params = params.reshape((3, params.shape[0] // 3))

        if sigmas.ndim == 1:
            # parameters coming from the database, reshape
            sigmas = sigmas.reshape((3, sigmas.shape[0] // 3))

        # determine if parameters are coming from the X vector (LSQ) or from the database (solution for self only)
        if params.shape[1] > self.param_count:
            # X vector
            self.p.params = params[:, self.column_index]
            self.p.sigmas = sigmas[:, self.column_index]
        else:
            # database (solution for self only; no need for column_index)
            self.p.params = params
            self.p.sigmas = sigmas


class Jump(EtmFunction):
    """
    generic jump (mechanic jump, frame change, etc) class
    """

    def __init__(self, NetworkCode, StationCode, solution, t, date, metadata, dtype=UNDETERMINED):

        super(Jump, self).__init__(NetworkCode=NetworkCode, StationCode=StationCode, solution=solution)

        # in the future, can load parameters from the db
        self.p.object = 'jump'

        # define initial state variables
        self.date = date

        self.p.jump_date = date.datetime()
        self.p.metadata = metadata
        self.p.jump_type = dtype

        # add the magnitude property to allow transformation from CO_SEISMIC_JUMP_DECAY to GENERIC_JUMP and still
        # print the magnitude of the event in the jump table
        self.magnitude = ''
        self.design = Jump.eval(self, t)

        if np.any(self.design) and not np.all(self.design):
            self.p.jump_type = GENERIC_JUMP
            self.param_count = 1
        else:
            self.p.jump_type = NO_EFFECT
            self.param_count = 0

        self.p.hash = crc32(bytearray(f'{self.date}', 'utf-8'))

    def remove(self):
        # this method will make this jump type = 0 and adjust its params
        self.p.jump_type = NO_EFFECT
        self.param_count = 0

    def eval(self, t):
        # given a time vector t, return the design matrix column vector(s)
        if self.p.jump_type == NO_EFFECT:
            return np.array([])

        ht = np.zeros((t.shape[0], 1))

        ht[t > self.date.fyear] = 1.

        return ht

    def __eq__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid. Can compare two Jump objects')

        # compare two jumps together and make sure they will not generate a singular (or near singular) system of eq
        c = np.sum(np.logical_xor(self.design[:, 0], jump.design[:, 0]))

        if self.p.jump_type in (CO_SEISMIC_JUMP_DECAY,
                                CO_SEISMIC_DECAY) and jump.p.jump_type in (CO_SEISMIC_JUMP_DECAY, CO_SEISMIC_DECAY):

            # if self is a co-seismic jump and next jump is also co-seismic
            # and there are less than two weeks of data to constrain params, return false
            if c <= EQ_MIN_DAYS:
                return True, jump
            else:
                return False, None

        elif self.p.jump_type in (CO_SEISMIC_JUMP_DECAY,
                                  CO_SEISMIC_DECAY, GENERIC_JUMP) and jump.p.jump_type == GENERIC_JUMP:

            if c <= JP_MIN_DAYS:
                # can't fit the co-seismic or generic jump AND the generic jump after, remove "jump" generic jump
                return True, self
            else:
                return False, None

        elif self.p.jump_type == GENERIC_JUMP and jump.p.jump_type == (CO_SEISMIC_JUMP_DECAY, CO_SEISMIC_DECAY):

            if c <= JP_MIN_DAYS:
                # if generic jump before an earthquake jump and less than 5 days, co-seismic prevails
                return True, jump
            else:
                return False, None

        elif self.p.jump_type == NO_EFFECT and jump.p.jump_type != NO_EFFECT:
            # if comparing to a self that has NO_EFFECT, remove and keep jump
            return True, jump

        elif self.p.jump_type != NO_EFFECT and jump.p.jump_type == NO_EFFECT:
            # if comparing against a jump that has NO_EFFECT, remove jump keep self
            return True, self

        elif self.p.jump_type == NO_EFFECT and jump.p.jump_type == NO_EFFECT:
            # no jump has an effect, return None. This will be interpreted as False (if not result)
            return None, None

    def __str__(self):
        return '(' + str(self.date) + '), ' + str(self.p.jump_type) + ', "' + str(self.p.jump_type) + '"'

    def __repr__(self):
        return 'pyPPPETM.Jump(' + str(self) + ')'

    def __lt__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear < jump.date.fyear

    def __le__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear <= jump.date.fyear

    def __gt__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear > jump.date.fyear

    def __ge__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear >= jump.date.fyear

    def __hash__(self):
        # to make the object hashable
        return hash(self.date.fyear)


class CoSeisJump(Jump):

    def __init__(self, NetworkCode, StationCode, solution, t, date, relaxation, metadata, dtype=UNDETERMINED):

        # super-class initialization
        Jump.__init__(self, NetworkCode, StationCode, solution, t, date, metadata, dtype)

        # new feature informs the magnitude of the event in the plot
        self.magnitude = float(metadata.split('=')[1].strip())

        if dtype is NO_EFFECT:
            # passing default_type == NO_EFFECT, add the jump but make it NO_EFFECT by default
            self.p.jump_type = NO_EFFECT
            self.params_count = 0
            self.p.relaxation = None

            self.design = np.array([])
            return

        if self.p.jump_type == NO_EFFECT:
            # came back from init as NO_EFFECT. May be a jump before t.min()
            # assign just the decay
            self.p.jump_type = CO_SEISMIC_DECAY
        else:
            self.p.jump_type = CO_SEISMIC_JUMP_DECAY

        # if T is an array, it contains the corresponding decays
        # otherwise, it is a single decay
        if not isinstance(relaxation, np.ndarray):
            relaxation = np.array([relaxation])

        self.param_count += relaxation.shape[0]
        self.nr = relaxation.shape[0]
        self.p.relaxation = relaxation

        self.design = self.eval(t)

        # test if earthquake generates at least 10 days of data to adjust
        if self.design.size:
            if np.count_nonzero(self.design[:, -1]) < 10:
                if self.p.jump_type == CO_SEISMIC_JUMP_DECAY:
                    # was a jump and decay, leave the jump
                    self.p.jump_type = GENERIC_JUMP
                    self.p.params = np.zeros((3, 1))
                    self.p.sigmas = np.zeros((3, 1))
                    self.param_count -= 1
                    # reevaluate the design matrix!
                    self.design = self.eval(t)
                else:
                    self.p.jump_type = NO_EFFECT
                    self.p.params = np.array([])
                    self.p.sigmas = np.array([])
                    self.param_count = 0
        else:
            self.p.jump_type = NO_EFFECT
            self.p.params = np.array([])
            self.p.sigmas = np.array([])
            self.param_count = 0

        self.p.hash += crc32(str(self.param_count) + ' ' + str(self.p.jump_type) + ' ' + str(self.p.relaxation))

    def eval(self, t):

        ht = Jump.eval(self, t)

        # if there is nothing in ht, then there is no expected output, return none
        if not np.any(ht):
            return np.array([])

        # if it was determined that this is just a generic jump, return ht
        if self.p.jump_type == GENERIC_JUMP:
            return ht

        # support more than one decay
        hl = np.zeros((t.shape[0], self.nr))

        for i, T in enumerate(self.p.relaxation):
            hl[t > self.date.fyear, i] = np.log10(1. + (t[t > self.date.fyear] - self.date.fyear) / T)

        # if it's both jump and decay, return ht + hl
        if np.any(hl) and self.p.jump_type == CO_SEISMIC_JUMP_DECAY:
            return np.column_stack((ht, hl))

        # if decay only, return hl
        elif np.any(hl) and self.p.jump_type == CO_SEISMIC_DECAY:
            return hl

    def __str__(self):
        return '(' + str(self.date) + '), ' + str(self.p.jump_type) + ', ' + str(self.p.relaxation) + ', "' + str(
            self.p.metadata) + '"'

    def __repr__(self):
        return 'pyPPPETM.CoSeisJump(' + str(self) + ')'


class Earthquakes:

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, FitEarthquakes=True):

        self.StationCode = StationCode
        self.NetworkCode = NetworkCode

        # station location
        stn = cnn.query('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                        % (NetworkCode, StationCode))

        stn = stn.dictresult()[0]

        # load metadata
        lat = float(stn['lat'])
        lon = float(stn['lon'])

        # establish the limit dates. Ignore jumps before 5 years from the earthquake
        sdate = pyDate.Date(fyear=t.min() - 5)
        edate = pyDate.Date(fyear=t.max())

        # get the earthquakes based on Mike's expression
        jumps = cnn.query('SELECT * FROM earthquakes WHERE date BETWEEN \'%s\' AND \'%s\' ORDER BY date'
                          % (sdate.yyyymmdd(), edate.yyyymmdd()))
        jumps = jumps.dictresult()

        # check if data range returned any jumps
        if jumps and FitEarthquakes:
            eq = [[float(jump['lat']), float(jump['lon']), float(jump['mag']),
                   int(jump['date'].year), int(jump['date'].month), int(jump['date'].day),
                   int(jump['date'].hour), int(jump['date'].minute), int(jump['date'].second)] for jump in jumps]

            eq = np.array(list(eq))

            dist = distance(lon, lat, eq[:, 1], eq[:, 0])

            m = -0.8717 * (np.log10(dist) - 2.25) + 0.4901 * (eq[:, 2] - 6.6928)
            # build the earthquake jump table
            # remove event events that happened the same day

            eq_jumps = list(set((float(eqs[2]), pyDate.Date(year=int(eqs[3]), month=int(eqs[4]), day=int(eqs[5]),
                                                            hour=int(eqs[6]), minute=int(eqs[7]), second=int(eqs[8])))
                                for eqs in eq[m > 0, :]))

            eq_jumps.sort(key=lambda x: (x[1], x[0]))

            # open the jumps table
            jp = cnn.query_float('SELECT * FROM etm_params WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' '
                                 'AND soln = \'%s\' AND jump_type <> 0 AND object = \'jump\''
                                 % (NetworkCode, StationCode, solution), as_dict=True)

            # start by collapsing all earthquakes for the same day.
            # Do not allow more than one earthquake on the same day
            f_jumps = []
            next_date = None

            for mag, date in eq_jumps:

                # jumps are analyzed in windows that are EQ_MIN_DAYS long
                # a date should not be analyzed is it's < next_date
                if next_date is not None:
                    if date < next_date:
                        continue

                # obtain jumps in a EQ_MIN_DAYS window
                jumps = [(m, d) for m, d in eq_jumps if date <= d < date + EQ_MIN_DAYS]

                if len(jumps) > 1:
                    # if more than one jump, get the max magnitude
                    mmag = max([m for m, _ in jumps])

                    # only keep the earthquake with the largest magnitude
                    for m, d in jumps:

                        table = [j['action'] for j in jp if j['Year'] == d.year and j['DOY'] == d.doy]

                        # get a different relaxation for this date
                        relax = [j['relaxation'] for j in jp if j['Year'] == d.year and j['DOY'] == d.doy]

                        if relax:
                            if relax[0] is not None:
                                relaxation = np.array(relax[0])
                            else:
                                relaxation = DEFAULT_RELAXATION
                        else:
                            relaxation = DEFAULT_RELAXATION

                        # if present in jump table, with either + of -, don't use default decay
                        if m == mmag and '-' not in table:
                            f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, d, relaxation,
                                                   'mag=%.1f' % m)]
                            # once the jump was added, exit for loop
                            break
                        else:
                            # add only if in jump list with a '+'
                            if '+' in table:
                                f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, d,
                                                       relaxation, 'mag=%.1f' % m)]
                                # once the jump was added, exit for loop
                                break
                            else:
                                f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, d,
                                                       relaxation, 'mag=%.1f' % m, NO_EFFECT)]
                else:
                    # add, unless marked in table with '-'
                    table = [j['action'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]
                    # get a different relaxation for this date
                    relax = [j['relaxation'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]

                    if relax:
                        if relax[0] is not None:
                            relaxation = np.array(relax[0])
                        else:
                            relaxation = DEFAULT_RELAXATION
                    else:
                        relaxation = DEFAULT_RELAXATION

                    if '-' not in table:
                        f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, date,
                                               relaxation, 'mag=%.1f' % mag)]
                    else:
                        # add it with NO_EFFECT for display purposes
                        f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, date,
                                               relaxation, 'mag=%.1f' % mag, NO_EFFECT)]

                next_date = date + EQ_MIN_DAYS

            # final jump table
            self.table = f_jumps
        else:
            self.table = []


class GenericJumps(object):

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, FitGenericJumps=True):

        self.solution = solution
        self.table = []

        if t.size >= 2:
            # analyze if it is possible to add the jumps (based on the available data)
            wt = np.sort(np.unique(t - np.fix(t)))
            # analyze the gaps in the data
            dt = np.diff(wt)
            # max dt (internal)
            dtmax = np.max(dt)
            # dt wrapped around
            dt_interyr = 1 - wt[-1] + wt[0]

            if dt_interyr > dtmax:
                dtmax = dt_interyr

            if dtmax <= 0.2465 and FitGenericJumps:
                # put jumps in
                self.add_metadata_jumps = True
            else:
                # no jumps
                self.add_metadata_jumps = False
        else:
            self.add_metadata_jumps = False

        # open the jumps table
        jp = cnn.query('SELECT * FROM etm_params WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' '
                       'AND soln = \'%s\' AND jump_type = 0 AND object = \'jump\''
                       % (NetworkCode, StationCode, solution))

        jp = jp.dictresult()

        # get station information
        self.stninfo = pyStationInfo.StationInfo(cnn, NetworkCode, StationCode)

        for stninfo in self.stninfo.records[1:]:

            date = stninfo['DateStart']

            table = [j['action'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]

            # add to list only if:
            # 1) add_meta = True AND there is no '-' OR
            # 2) add_meta = False AND there is a '+'

            if (not self.add_metadata_jumps and '+' in table) or (self.add_metadata_jumps and '-' not in table):
                self.table.append(Jump(NetworkCode, StationCode, solution, t, date,
                                       'Ant-Rec: %s-%s' % (stninfo['AntennaCode'], stninfo['ReceiverCode'])))

        # frame changes if ppp
        if solution == 'ppp':
            frames = cnn.query(
                'SELECT distinct on ("ReferenceFrame") "ReferenceFrame", "Year", "DOY" from ppp_soln WHERE '
                '"NetworkCode" = \'%s\' AND "StationCode" = \'%s\' order by "ReferenceFrame", "Year", "DOY"' %
                (NetworkCode, StationCode))

            frames = frames.dictresult()

            if len(frames) > 1:
                # more than one frame, add a jump
                frames.sort(key=lambda k: k['Year'])

                for frame in frames[1:]:
                    date = pyDate.Date(Year=frame['Year'], doy=frame['DOY'])

                    table = [j['action'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]

                    if '-' not in table:
                        self.table.append(Jump(NetworkCode, StationCode, solution, t, date,
                                               'Frame Change: %s' % frame['ReferenceFrame']))

        # now check the jump table to add specific jumps
        jp = cnn.query('SELECT * FROM etm_params WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' '
                       'AND soln = \'%s\' AND jump_type = 0 AND object = \'jump\' '
                       'AND action = \'+\'' % (NetworkCode, StationCode, solution))

        jp = jp.dictresult()

        table = [j.date for j in self.table]

        for j in jp:
            date = pyDate.Date(Year=j['Year'], doy=j['DOY'])

            if date not in table:
                self.table.append(Jump(NetworkCode, StationCode, solution, t, date, 'mechanic-jump'))


class Periodic(EtmFunction):
    """"class to determine the periodic terms to be included in the ETM"""

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, FitPeriodic=True):

        super(Periodic, self).__init__(NetworkCode=NetworkCode, StationCode=StationCode, solution=solution)

        try:
            # load the frequencies from the database
            etm_param = cnn.get('etm_params',
                                {'NetworkCode': NetworkCode, 'StationCode': StationCode, 'soln': solution,
                                 'object': 'periodic'},
                                ['NetworkCode', 'StationCode', 'soln', 'object'])

            self.p.frequencies = np.array([float(p) for p in etm_param['frequencies']])

        except pg.DatabaseError:
            self.p.frequencies = DEFAULT_FREQUENCIES

        self.p.object = 'periodic'

        if t.size > 1 and FitPeriodic:
            # wrap around the solutions
            wt = np.sort(np.unique(t - np.fix(t)))

            # analyze the gaps in the data
            dt = np.diff(wt)

            # max dt (internal)
            dtmax = np.max(dt)

            # dt wrapped around
            dt_interyr = 1 - wt[-1] + wt[0]

            if dt_interyr > dtmax:
                dtmax = dt_interyr

            # save the value of the max wrapped delta time
            self.dt_max = dtmax

            # get the 50 % of Nyquist for each component (and convert to average fyear)
            self.nyquist = ((1 / self.p.frequencies) / 2.) * 0.5 * 1 / 365.25

            # frequency count
            self.frequency_count = int(np.sum(self.dt_max <= self.nyquist))

            # redefine the frequencies vector to accommodate only the frequencies that can be fit
            self.p.frequencies = self.p.frequencies[self.dt_max <= self.nyquist]

        else:
            # no periodic terms
            self.frequency_count = 0
            self.p.frequencies = np.array([])
            self.dt_max = 1  # one year of delta t

        # build the metadata description for the json string
        self.p.metadata = '['
        for k in ['n', 'e', 'u']:
            self.p.metadata = self.p.metadata + '['
            meta = []
            for i in ['sin', 'cos']:
                for f in (1 / (self.p.frequencies * 365.25)).tolist():
                    meta.append('%s:%s(%.1f yr)' % (k, i, f))

            self.p.metadata = self.p.metadata + ','.join(meta) + '],'

        self.p.metadata = self.p.metadata + ']'

        self.design = self.get_design_ts(t)
        self.param_count = self.frequency_count * 2
        # declare the location of the answer (to be filled by Design object)
        self.column_index = np.array([])

        self.format_str = 'Periodic amp (' + \
                          ', '.join(['%.1f yr' % i for i in (1 / (self.p.frequencies * 365.25)).tolist()]) + \
                          ') N: %s E: %s U: %s [mm]'

        self.p.hash = crc32(bytearray(f'{self.p.frequencies}', 'utf-8'))

    def get_design_ts(self, ts):
        # if dtmax < 3 months (90 days = 0.1232), then we can fit the annual
        # if dtmax < 1.5 months (45 days = 0.24657), then we can fit the semi-annual too

        if self.frequency_count > 0:
            f = self.p.frequencies
            f = np.tile(f, (ts.shape[0], 1))

            As = np.array(sin(2 * pi * f * 365.25 * np.tile(ts[:, np.newaxis], (1, f.shape[1]))))
            Ac = np.array(cos(2 * pi * f * 365.25 * np.tile(ts[:, np.newaxis], (1, f.shape[1]))))

            A = np.column_stack((As, Ac))
        else:
            # no periodic terms
            A = np.array([])

        return A

    def print_parameters(self):

        n = np.array([])
        e = np.array([])
        u = np.array([])

        for p in np.arange(self.param_count):
            psc = self.p.params[:, p]

            sn = psc[0]
            se = psc[1]
            su = psc[2]

            n = np.append(n, sn)
            e = np.append(e, se)
            u = np.append(u, su)

        n = n.reshape((2, self.param_count // 2))
        e = e.reshape((2, self.param_count // 2))
        u = u.reshape((2, self.param_count // 2))

        # calculate the amplitude of the components
        an = np.sqrt(np.square(n[0, :]) + np.square(n[1, :]))
        ae = np.sqrt(np.square(e[0, :]) + np.square(e[1, :]))
        au = np.sqrt(np.square(u[0, :]) + np.square(u[1, :]))

        return self.format_str % (np.array_str(an * 1000.0, precision=1),
                                  np.array_str(ae * 1000.0, precision=1),
                                  np.array_str(au * 1000.0, precision=1))


class Polynomial(EtmFunction):
    """"class to build the linear portion of the design matrix"""

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, t_ref=0):

        super(Polynomial, self).__init__(NetworkCode=NetworkCode, StationCode=StationCode, solution=solution)

        # t ref (just the beginning of t vector)
        if t_ref == 0:
            t_ref = np.min(t)

        self.p.object = 'polynomial'
        self.p.t_ref = t_ref

        try:
            # load the number of terms from the database
            etm_param = cnn.get('etm_params',
                                {'NetworkCode': NetworkCode, 'StationCode': StationCode, 'soln': solution,
                                 'object': 'polynomial'},
                                ['NetworkCode', 'StationCode', 'soln', 'object'])

            self.terms = int(etm_param['terms'])

        except pg.DatabaseError:
            self.terms = DEFAULT_POL_TERMS

        if self.terms == 1:
            self.format_str = 'Ref Position (' + '%.3f' % t_ref + ') X: {:.3f} Y: {:.3f} Z: {:.3f} [m]'
            self.p.metadata = '[[n:pos],[e:pos],[u:pos]]'

        elif self.terms == 2:
            self.format_str = 'Ref Position (' + '%.3f' % t_ref + ') X: {:.3f} Y: {:.3f} Z: {:.3f} [m]\n' \
                                                                  'Velocity N: {:.2f} E: {:.2f} U: {:.2f} [mm/yr]'
            self.p.metadata = '[[n:pos, n:vel],[e:pos, e:vel],[u:pos, u:vel]]'

        elif self.terms == 3:
            self.format_str = 'Ref Position (' + '%.3f' % t_ref + ') X: {:.3f} Y: {:.3f} Z: {:.3f} [m]\n' \
                                                                  'Velocity N: {:.3f} E: {:.3f} U: {:.3f} [mm/yr]\n' \
                                                                  'Acceleration N: {:.2f} E: {:.2f} U: {:.2f} [mm/yr^2]'
            self.p.metadata = '[[n:pos, n:vel, n:acc],[e:pos, e:vel, e:acc],[u:pos, u:vel, u:acc]]'

        elif self.terms > 3:
            self.format_str = 'Ref Position (' + '%.3f' % t_ref + ') X: {:.3f} Y: {:.3f} Z: {:.3f} [m]\n' \
                                                                  'Velocity N: {:.3f} E: {:.3f} U: {:.3f} [mm/yr]\n' \
                                                                  'Acceleration N: {:.2f} E: {:.2f} U: {:.2f} [mm/yr^2] + ' + '%i' % (
                                          self.terms - 3) + \
                              ' other polynomial terms'
            self.p.metadata = '[[n:pos, n:vel, n:acc, n:tx...],' \
                              '[e:pos, e:vel, e:acc, e:tx...],' \
                              '[u:pos, u:vel, u:acc, u:tx...]]'

        self.design = self.get_design_ts(t)

        # always first in the list of A, index columns are fixed
        self.column_index = np.arange(self.terms)
        # param count is the same as terms
        self.param_count = self.terms
        # save the hash of the object
        self.p.hash = crc32(bytearray(f'{self.terms}', 'utf-8'))

    def load_parameters(self, params, sigmas, t_ref):

        super(Polynomial, self).load_parameters(params=params, sigmas=sigmas)

        self.p.t_ref = t_ref

    def print_parameters(self, ref_xyz, lat, lon):

        params = np.zeros((3, 1))

        for p in np.arange(self.terms):
            if p == 0:
                params[0], params[1], params[2] = lg2ct(self.p.params[0, 0],
                                                        self.p.params[1, 0],
                                                        self.p.params[2, 0], lat, lon)
                params += ref_xyz

            elif p > 0:
                n = self.p.params[0, p]
                e = self.p.params[1, p]
                u = self.p.params[2, p]

                params = np.append(params, (n * 1000, e * 1000, u * 1000))

        return self.format_str.format(*params.tolist())

    def get_design_ts(self, ts):

        A = np.zeros((ts.size, self.terms))

        for p in np.arange(self.terms):
            A[:, p] = np.power(ts - self.p.t_ref, p)

        return A


class Design(np.ndarray):

    def __new__(subtype, Linear, Jumps, Periodic, dtype=float, buffer=None, offset=0, strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__

        shape = (Linear.design.shape[0], Linear.param_count + Jumps.param_count + Periodic.param_count)
        A = super(Design, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)

        A[:, Linear.column_index] = Linear.design

        # determine the column_index for all objects
        col_index = Linear.param_count

        for jump in Jumps.table:
            # save the column index
            jump.column_index = np.arange(col_index, col_index + jump.param_count)
            # assign the portion of the design matrix
            A[:, jump.column_index] = jump.design
            # increment the col_index
            col_index += jump.param_count

        Periodic.column_index = np.arange(col_index, col_index + Periodic.param_count)

        A[:, Periodic.column_index] = Periodic.design

        # save the object list
        A.objects = (Linear, Jumps, Periodic)

        # save the number of total parameters
        A.linear_params = Linear.param_count
        A.jump_params = Jumps.param_count
        A.periodic_params = Periodic.param_count

        A.params = Linear.param_count + Jumps.param_count + Periodic.param_count

        # save the constrains matrix
        A.constrains = Jumps.constrains

        # Finally, we must return the newly created object:
        return A

    def __call__(self, ts=None, constrains=False):

        if ts is None:
            if constrains:
                if self.constrains.size:
                    A = self.copy()
                    # resize matrix (use A.resize so that it fills with zeros)
                    A.resize((self.shape[0] + self.constrains.shape[0], self.shape[1]), refcheck=False)
                    # apply constrains
                    A[-self.constrains.shape[0]:, self.jump_params] = self.constrains
                    return A

                else:
                    return self

            else:
                return self

        else:

            A = np.array([])

            for obj in self.objects:
                tA = obj.get_design_ts(ts)
                if A.size:
                    A = np.column_stack((A, tA)) if tA.size else A
                else:
                    A = tA

            return A

    def get_l(self, L, constrains=False):

        if constrains:
            if self.constrains.size:
                tL = L.copy()
                tL.resize((L.shape[0] + self.constrains.shape[0]), refcheck=False)
                return tL

            else:
                return L

        else:
            return L

    def get_p(self, constrains=False):
        # return a weight matrix full of ones with or without the extra elements for the constrains
        return np.ones((self.shape[0])) if not constrains else \
            np.ones((self.shape[0] + self.constrains.shape[0]))

    def remove_constrains(self, v):
        # remove the constrains to whatever vector is passed
        if self.constrains.size:
            return v[0:-self.constrains.shape[0]]
        else:
            return v


class ETM:

    def __init__(self, cnn, soln, no_model=False, FitEarthquakes=True, FitGenericJumps=True, FitPeriodic=True):

        # to display more verbose warnings
        # warnings.showwarning = self.warn_with_traceback

        self.C = np.array([])
        self.S = np.array([])
        self.F = np.array([])
        self.R = np.array([])
        self.P = np.array([])
        self.factor = np.array([])
        self.covar = np.zeros((3, 3))
        self.A = None
        self.soln = soln
        self.no_model = no_model
        self.FitEarthquakes = FitEarthquakes
        self.FitGenericJumps = FitGenericJumps
        self.FitPeriodic = FitPeriodic

        self.NetworkCode = soln.NetworkCode
        self.StationCode = soln.StationCode

        # save the function objects
        self.Linear = Polynomial(cnn, soln.NetworkCode, soln.StationCode, self.soln.type, soln.t)
        self.Periodic = Periodic(cnn, soln.NetworkCode, soln.StationCode, self.soln.type, soln.t, FitPeriodic)
        self.Jumps = JumpTable(cnn, soln.NetworkCode, soln.StationCode, soln.type, soln.t,
                               FitEarthquakes, FitGenericJumps)
        # calculate the hash value for this station
        # now hash also includes the timestamp of the last time pyETM was modified.
        self.hash = soln.hash + crc32(bytearray(f'{VERSION}', 'utf-8'))

        # anything less than four is not worth it
        if soln.solutions > 4 and not no_model:

            # to obtain the parameters
            self.A = Design(self.Linear, self.Jumps, self.Periodic)

            # check if problem can be solved!
            if self.A.shape[1] >= soln.solutions:
                self.A = None
                return

            self.As = self.A(soln.ts)

    def run_adjustment(self, cnn, l, plotit=False):

        c = []
        f = []
        s = []
        r = []
        p = []
        factor = []

        if self.A is not None:
            # try to load the last ETM solution from the database

            etm_objects = cnn.query_float('SELECT * FROM etmsv2 WHERE "NetworkCode" = \'%s\' '
                                          'AND "StationCode" = \'%s\' AND soln = \'%s\''
                                          % (self.NetworkCode, self.StationCode, self.soln.type), as_dict=True)

            db_hash_sum = sum([obj['hash'] for obj in etm_objects])
            ob_hash_sum = sum([o.p.hash for o in self.Jumps.table + [self.Periodic] + [self.Linear]]) + self.hash
            cn_object_sum = len(self.Jumps.table) + 2

            # -1 to account for the var_factor entry
            if len(etm_objects) - 1 == cn_object_sum and db_hash_sum == ob_hash_sum:
                # load the parameters from th db
                self.load_parameters(etm_objects, l)
            else:
                # purge table and recompute
                cnn.query('DELETE FROM etmsv2 WHERE "NetworkCode" = \'%s\' AND '
                          '"StationCode" = \'%s\' AND soln = \'%s\''
                          % (self.NetworkCode, self.StationCode, self.soln.type))

                # use the default parameters from the objects
                t_ref = self.Linear.p.t_ref

                for i in range(3):
                    x, sigma, index, residuals, fact, w = self.adjust_lsq(self.A, l[i])

                    c.append(x)
                    s.append(sigma)
                    f.append(index)
                    r.append(residuals)
                    factor.append(fact)
                    p.append(w)

                self.C = np.array(c)
                self.S = np.array(s)
                self.F = np.array(f)
                self.R = np.array(r)
                self.factor = np.array(factor)
                self.P = np.array(p)

                # load_parameters to the objects
                self.Linear.load_parameters(self.C, self.S, t_ref)
                self.Jumps.load_parameters(self.C, self.S)
                self.Periodic.load_parameters(params=self.C, sigmas=self.S)

                # save the parameters in each object to the db
                self.save_parameters(cnn)

            # load the covariances using the correlations
            self.process_covariance()

            if plotit:
                self.plot()

    def process_covariance(self):

        cov = np.zeros((3, 1))

        # save the covariance between N-E, E-U, N-U
        f = self.F[0] * self.F[1] * self.F[2]

        # load the covariances using the correlations
        cov[0] = np.corrcoef(self.R[0][f], self.R[1][f])[0, 1] * self.factor[0] * self.factor[1]
        cov[1] = np.corrcoef(self.R[1][f], self.R[2][f])[0, 1] * self.factor[1] * self.factor[2]
        cov[2] = np.corrcoef(self.R[0][f], self.R[2][f])[0, 1] * self.factor[0] * self.factor[2]

        # build a variance-covariance matrix
        self.covar = np.diag(np.square(self.factor))

        self.covar[0, 1] = cov[0]
        self.covar[1, 0] = cov[0]
        self.covar[2, 1] = cov[1]
        self.covar[1, 2] = cov[1]
        self.covar[0, 2] = cov[2]
        self.covar[2, 0] = cov[2]

        if not self.isPD(self.covar):
            self.covar = self.nearestPD(self.covar)

    def save_parameters(self, cnn):

        # insert linear parameters
        cnn.insert('etmsv2', row=to_postgres(self.Linear.p.toDict()))

        # insert jumps
        for jump in self.Jumps.table:
            cnn.insert('etmsv2', row=to_postgres(jump.p.toDict()))

        # insert periodic params
        cnn.insert('etmsv2', row=to_postgres(self.Periodic.p.toDict()))

        cnn.query('INSERT INTO etmsv2 ("NetworkCode", "StationCode", soln, object, params, hash) VALUES '
                  '(\'%s\', \'%s\', \'ppp\', \'var_factor\', \'%s\', %i)'
                  % (self.NetworkCode, self.StationCode, to_postgres(self.factor), self.hash))

    def plot(self, pngfile=None, t_win=None, residuals=False, plot_missing=True, ecef=False):

        import matplotlib.pyplot as plt

        L = self.l * 1000

        # definitions
        m = []
        if ecef:
            labels = ('X [mm]', 'Y [mm]', 'Z [mm]')
        else:
            labels = ('North [mm]', 'East [mm]', 'Up [mm]')

        # get filtered observations
        if self.A is not None:
            filt = self.F[0] * self.F[1] * self.F[2]

            for i in range(3):
                m.append((np.dot(self.As, self.C[i])) * 1000)

        else:
            filt = np.ones(self.soln.x.shape[0], dtype=bool)

        # rotate to NEU
        if ecef:
            lneu = self.rotate_2xyz(L)
        else:
            lneu = L

        # determine the window of the plot, if requested
        if t_win is not None:
            if type(t_win) is tuple:
                # data range, with possibly a final value
                if len(t_win) == 1:
                    t_win = (t_win[0], self.soln.t.max())
            else:
                # approximate a day in fyear
                t_win = (self.soln.t.max() - t_win / 365.25, self.soln.t.max())

        # new behaviour: plots the time series even if there is no ETM fit

        if self.A is not None:

            # create the axis
            f, axis = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(15, 10))  # type: plt.subplots

            # rotate modeled ts
            if not ecef:
                mneu = m
                rneu = self.R
                fneu = self.factor * 1000
            else:
                mneu = self.rotate_2xyz(m)
                # rotate residuals
                rneu = self.rotate_2xyz(self.R)
                fneu = np.sqrt(np.diag(self.rotate_sig_cov(covar=self.covar))) * 1000

            # ################# FILTERED PLOT #################

            f.suptitle('Station: %s.%s lat: %.5f lon: %.5f\n'
                       '%s completion: %.2f%%\n%s\n%s\n'
                       'NEU wrms [mm]: %5.2f %5.2f %5.2f' %
                       (self.NetworkCode, self.StationCode, self.soln.lat, self.soln.lon, self.soln.type.upper(),
                        self.soln.completion,
                        self.Linear.print_parameters(np.array([self.soln.auto_x, self.soln.auto_y, self.soln.auto_z]),
                                                     self.soln.lat, self.soln.lon),
                        self.Periodic.print_parameters(),
                        fneu[0], fneu[1], fneu[2]), fontsize=9, family='monospace')

            table_n, table_e, table_u = self.Jumps.print_parameters()
            tables = (table_n, table_e, table_u)

            for i, ax in enumerate((axis[0][0], axis[1][0], axis[2][0])):

                # plot filtered time series
                if not residuals:
                    ax.plot(self.soln.t[filt], lneu[i][filt], 'ob', markersize=2)
                    ax.plot(self.soln.ts, mneu[i], 'r')
                    # error bars
                    ax.plot(self.soln.ts, mneu[i] - fneu[i] * LIMIT, 'b', alpha=0.1)
                    ax.plot(self.soln.ts, mneu[i] + fneu[i] * LIMIT, 'b', alpha=0.1)
                    ax.fill_between(self.soln.ts, mneu[i] - fneu[i] * LIMIT, mneu[i] + fneu[i] * LIMIT,
                                    antialiased=True, alpha=0.2)
                else:
                    ax.plot(self.soln.t[filt], rneu[i][filt] * 1000, 'ob', markersize=2)
                    # error bars
                    ax.plot(self.soln.ts, - np.repeat(fneu[i], self.soln.ts.shape[0]) * LIMIT, 'b', alpha=0.1)
                    ax.plot(self.soln.ts, np.repeat(fneu[i], self.soln.ts.shape[0]) * LIMIT, 'b', alpha=0.1)
                    ax.fill_between(self.soln.ts, -fneu[i] * LIMIT, fneu[i] * LIMIT, antialiased=True, alpha=0.2)

                ax.grid(True)

                # labels
                ax.set_ylabel(labels[i])
                p = ax.get_position()
                f.text(0.005, p.y0, tables[i], fontsize=8, family='monospace')

                # window data
                self.set_lims(t_win, plt, ax)

                # plot jumps
                self.plot_jumps(ax)

            # ################# OUTLIERS PLOT #################

            for i, ax in enumerate((axis[0][1], axis[1][1], axis[2][1])):
                ax.plot(self.soln.t, lneu[i], 'oc', markersize=2)
                ax.plot(self.soln.t[filt], lneu[i][filt], 'ob', markersize=2)
                ax.plot(self.soln.ts, mneu[i], 'r')
                # error bars
                ax.plot(self.soln.ts, mneu[i] - fneu[i] * LIMIT, 'b', alpha=0.1)
                ax.plot(self.soln.ts, mneu[i] + fneu[i] * LIMIT, 'b', alpha=0.1)
                ax.fill_between(self.soln.ts, mneu[i] - fneu[i] * LIMIT, mneu[i] + fneu[i] * LIMIT,
                                antialiased=True, alpha=0.2)

                self.set_lims(t_win, plt, ax)

                ax.set_ylabel(labels[i])

                ax.grid(True)

                if plot_missing:
                    self.plot_missing_soln(ax)

            f.subplots_adjust(left=0.17)

        else:

            f, axis = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15, 10))  # type: plt.subplots

            f.suptitle('Station: %s.%s lat: %.5f lon: %.5f'
                       % (self.NetworkCode, self.StationCode, self.soln.lat, self.soln.lon) +
                       '\nNot enough solutions to fit an ETM.', fontsize=9, family='monospace')

            for i, ax in enumerate((axis[0], axis[1], axis[2])):
                ax.plot(self.soln.t, lneu[i], 'ob', markersize=2)

                ax.set_ylabel(labels[i])

                ax.grid(True)

                self.set_lims(t_win, plt, ax)

                self.plot_jumps(ax)

                if plot_missing:
                    self.plot_missing_soln(ax)

        if not pngfile:
            self.f = f
            self.picking = False
            self.plt = plt
            axprev = plt.axes([0.85, 0.01, 0.08, 0.055])
            bcut = Button(axprev, 'Add jump', color='red', hovercolor='green')
            bcut.on_clicked(self.enable_picking)
            plt.show()
            plt.close()
        else:
            plt.savefig(pngfile)
            plt.close()

    def onpick(self, event):

        import dbConnection

        self.f.canvas.mpl_disconnect(self.cid)
        self.picking = False
        print('Epoch: %s' % pyDate.Date(fyear=event.xdata).yyyyddd())
        jtype = int(eval(input(' -- Enter type of jump (0 = mechanic; 1 = geophysical): ')))
        if jtype == 1:
            relx = eval(input(' -- Enter relaxation (e.g. 0.5, 0.5,0.01): '))
        operation = str(input(' -- Enter operation (+, -): '))
        print(' >> Jump inserted')

        # now insert the jump into the db
        cnn = dbConnection.Cnn('gnss_data.cfg')

        self.plt.close()

        # reinitialize ETM

        # wait for 'keep' or 'undo' command

    def enable_picking(self, event):
        if not self.picking:
            print('Entering picking mode')
            self.picking = True
            self.cid = self.f.canvas.mpl_connect('button_press_event', self.onpick)
        else:
            print('Disabling picking mode')
            self.picking = False
            self.f.canvas.mpl_disconnect(self.cid)

    def plot_hist(self):

        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab
        from scipy.stats import norm

        L = self.l * 1000

        if self.A is not None:
            residuals = np.sqrt(np.square(L[0]) + np.square(L[1]) + np.square(L[2])) - \
                        np.sqrt(np.square(np.dot(self.A, self.C[0])) + np.square(np.dot(self.A, self.C[1])) +
                                np.square(np.dot(self.A, self.C[2])))

            (mu, sigma) = norm.fit(residuals)

            n, bins, patches = plt.hist(residuals, 200, normed=1, alpha=0.75, facecolor='blue')

            y = mlab.normpdf(bins, mu, sigma)
            plt.plot(bins, y, 'r--', linewidth=2)
            plt.title(r'$\mathrm{Histogram\ of\ residuals (mm):}\ \mu=%.3f,\ \sigma=%.3f$' % (mu * 1000, sigma * 1000))
            plt.grid(True)

            plt.show()

    @staticmethod
    def autoscale_y(ax, margin=0.1):
        """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
        ax -- a matplotlib axes object
        margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

        def get_bottom_top(line):
            xd = line.get_xdata()
            yd = line.get_ydata()
            lo, hi = ax.get_xlim()
            y_displayed = yd[((xd > lo) & (xd < hi))]
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed) - margin * h
            top = np.max(y_displayed) + margin * h
            return bot, top

        lines = ax.get_lines()
        bot, top = np.inf, -np.inf

        for line in lines:
            new_bot, new_top = get_bottom_top(line)
            if new_bot < bot:
                bot = new_bot
            if new_top > top:
                top = new_top
        if bot == top:
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.autoscale(enable=False, axis='y', tight=False)
        else:
            ax.set_ylim(bot, top)

    def set_lims(self, t_win, plt, ax):

        if t_win is None:
            # turn on to adjust the limits, then turn off to plot jumps
            ax.autoscale(enable=True, axis='x', tight=False)
            ax.autoscale(enable=False, axis='x', tight=False)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.autoscale(enable=False, axis='y', tight=False)
        else:
            if t_win[0] == t_win[1]:
                t_win[0] = t_win[0] - 1. / 365.25
                t_win[1] = t_win[1] + 1. / 365.25

            plt.xlim(t_win)
            self.autoscale_y(ax)

    def plot_missing_soln(self, ax):

        # plot missing solutions
        for missing in self.soln.ts_ns:
            ax.plot((missing, missing), ax.get_ylim(), color=(1, 0, 1, 0.2), linewidth=1)

        # plot the position of the outliers
        for blunder in self.soln.ts_blu:
            ax.quiver((blunder, blunder), ax.get_ylim(), (0, 0), (-0.01, 0.01), scale_units='height',
                      units='height', pivot='tip', width=0.008, edgecolors='r')

    def plot_jumps(self, ax):

        for jump in self.Jumps.table:
            if jump.p.jump_type == GENERIC_JUMP and 'Frame Change' not in jump.p.metadata:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), 'b:')

            elif jump.p.jump_type == GENERIC_JUMP and 'Frame Change' in jump.p.metadata:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), ':', color='tab:green')

            elif jump.p.jump_type == CO_SEISMIC_JUMP_DECAY:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), 'r:')

            elif jump.p.jump_type == NO_EFFECT:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), ':', color='tab:gray')

    def todictionary(self, time_series=False):
        # convert the ETM adjustment into a dictionary
        # optionally, output the whole time series as well

        L = self.l

        # start with the parameters
        etm = dict()
        etm['Network'] = self.NetworkCode
        etm['Station'] = self.StationCode
        etm['lat'] = self.soln.lat[0]
        etm['lon'] = self.soln.lon[0]
        etm['ref_x'] = self.soln.auto_x[0]
        etm['ref_y'] = self.soln.auto_y[0]
        etm['ref_z'] = self.soln.auto_z[0]
        etm['Jumps'] = [to_list(jump.p.toDict()) for jump in self.Jumps.table]

        if self.A is not None:
            etm['Polynomial'] = to_list(self.Linear.p.toDict())

            etm['Periodic'] = to_list(self.Periodic.p.toDict())

            etm['wrms'] = {'n': self.factor[0], 'e': self.factor[1], 'u': self.factor[2]}

            etm['xyz_covariance'] = self.rotate_sig_cov(covar=self.covar).tolist()

            etm['neu_covariance'] = self.covar.tolist()

        if time_series:
            ts = dict()
            ts['t'] = np.array([self.soln.t.tolist(), self.soln.mjd.tolist()]).transpose().tolist()
            ts['mjd'] = self.soln.mjd.tolist()
            ts['x'] = self.soln.x.tolist()
            ts['y'] = self.soln.y.tolist()
            ts['z'] = self.soln.z.tolist()
            ts['n'] = L[0].tolist()
            ts['e'] = L[1].tolist()
            ts['u'] = L[2].tolist()
            ts['residuals'] = self.R.tolist()
            ts['weights'] = self.P.transpose().tolist()

            if self.A is not None:
                ts['filter'] = np.logical_and(np.logical_and(self.F[0], self.F[1]), self.F[2]).tolist()
            else:
                ts['filter'] = []

            etm['time_series'] = ts

        return etm

    def get_xyz_s(self, year, doy, jmp=None, sigma_h=SIGMA_FLOOR_H, sigma_v=SIGMA_FLOOR_V):
        # this function find the requested epochs and returns an X Y Z and sigmas
        # jmp = 'pre' returns the coordinate immediately before a jump
        # jmp = 'post' returns the coordinate immediately after a jump
        # jmp = None returns either the coordinate before or after, depending on the time of the jump.

        # find this epoch in the t vector
        date = pyDate.Date(year=year, doy=doy)
        window = None

        for jump in self.Jumps.table:
            if jump.date == date and jump.p.jump_type in (GENERIC_JUMP, CO_SEISMIC_JUMP_DECAY):
                if np.sqrt(np.sum(np.square(jump.p.params[:, 0]))) > 0.02:
                    window = jump.date
                    # if no pre or post specified, then determine using the time of the jump
                    if jmp is None:
                        if (jump.date.datetime().hour + jump.date.datetime().minute / 60.0) < 12:
                            jmp = 'post'
                        else:
                            jmp = 'pre'
                    # use the previous or next date to get the APR
                    # if jmp == 'pre':
                    #    date -= 1
                    # else:
                    #    date += 1

        index = np.where(self.soln.mjd == date.mjd)
        index = index[0]

        neu = np.zeros((3, 1))

        L = self.L
        ref_pos = np.array([self.soln.auto_x, self.soln.auto_y, self.soln.auto_z])

        if index.size and self.A is not None:
            # found a valid epoch in the t vector
            # now see if this epoch was filtered
            if np.all(self.F[:, index]):
                # the coordinate is good
                xyz = L[:, index]
                sig = self.R[:, index]
                source = 'PPP with ETM solution: good'

            else:
                # the coordinate is marked as bad
                # get the requested epoch from the ETM
                idt = np.argmin(np.abs(self.soln.ts - date.fyear))

                for i in range(3):
                    neu[i] = np.dot(self.As[idt, :], self.C[i])

                xyz = self.rotate_2xyz(neu) + ref_pos
                # Use the deviation from the ETM multiplied by 2.5 to estimate the error
                sig = 2.5 * self.R[:, index]
                source = 'PPP with ETM solution: filtered'

        elif not index.size and self.A is not None:

            # the coordinate doesn't exist, get it from the ETM
            idt = np.argmin(np.abs(self.soln.ts - date.fyear))
            source = 'No PPP solution: ETM'

            for i in range(3):
                neu[i] = np.dot(self.As[idt, :], self.C[i])

            xyz = self.rotate_2xyz(neu) + ref_pos
            # since there is no way to estimate the error,
            # use the nominal sigma multiplied by 2.5
            sig = 2.5 * self.factor[:, np.newaxis]

        elif index.size and self.A is None:

            # no ETM (too few points), but we have a solution for the requested day
            xyz = L[:, index]
            # set the uncertainties in NEU by hand
            sig = np.array([[9.99], [9.99], [9.99]])
            source = 'PPP solution, no ETM'

        else:
            # no ETM (too few points) and no solution for this day, get average
            source = 'No PPP solution, no ETM: mean coordinate'
            xyz = np.mean(L, axis=1)[:, np.newaxis]
            # set the uncertainties in NEU by hand
            sig = np.array([[9.99], [9.99], [9.99]])

        if self.A is not None:
            # get the velocity of the site
            if np.sqrt(np.square(self.Linear.p.params[0, 1]) +
                       np.square(self.Linear.p.params[1, 1]) +
                       np.square(self.Linear.p.params[2, 1])) > 0.2:
                # fast moving station! bump up the sigma floor
                sigma_h = 99.9
                sigma_v = 99.9
                source += '. fast moving station, bumping up sigmas'

        # apply floor sigmas
        sig = np.sqrt(np.square(sig) + np.square(np.array([[sigma_h], [sigma_h], [sigma_v]])))

        return xyz, sig, window, source

    def rotate_2neu(self, ecef):

        return np.array(ct2lg(ecef[0], ecef[1], ecef[2], self.soln.lat, self.soln.lon))

    def rotate_2xyz(self, neu):

        return np.array(lg2ct(neu[0], neu[1], neu[2], self.soln.lat, self.soln.lon))

    def rotate_sig_cov(self, sigmas=None, covar=None):

        if sigmas is None and covar is None:
            raise pyETMException('Error in rotate_sig_cov: must provide either sigmas or covariance matrix')

        R = rotlg2ct(self.soln.lat, self.soln.lon)

        if sigmas is not None:
            # build a covariance matrix based on sigmas
            sd = np.diagflat(np.square(sigmas))

            sd[0, 1] = self.covar[0, 1]
            sd[1, 0] = self.covar[1, 0]
            sd[2, 1] = self.covar[2, 1]
            sd[1, 2] = self.covar[1, 2]
            sd[0, 2] = self.covar[0, 2]
            sd[2, 0] = self.covar[2, 0]

            # check that resulting matrix is PSD:
            if not self.isPD(sd):
                sd = self.nearestPD(sd)

            sneu = np.dot(np.dot(R[:, :, 0], sd), R[:, :, 0].transpose())

            dneu = np.sqrt(np.diag(sneu))

        else:
            # covariance matrix given, assume it is a covariance matrix
            dneu = np.dot(np.dot(R[:, :, 0], covar), R[:, :, 0].transpose())

        return dneu

    def nearestPD(self, A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1

        while not self.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    @staticmethod
    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def load_parameters(self, params, l):

        factor = 1
        index = []
        residuals = []
        p = []

        for param in params:
            par = np.array(param['params'])
            sig = np.array(param['sigmas'])

            if param['object'] == 'polynomial':
                self.Linear.load_parameters(par, sig, param['t_ref'])

            if param['object'] == 'periodic':
                self.Periodic.load_parameters(params=par, sigmas=sig)

            if param['object'] == 'jump':
                for jump in self.Jumps.table:
                    if jump.p.hash == param['hash']:
                        jump.load_parameters(params=par, sigmas=sig)

            if param['object'] == 'var_factor':
                # already a vector in the db
                factor = par

        x = self.Linear.p.params
        s = self.Linear.p.sigmas

        for jump in self.Jumps.table:
            x = np.append(x, jump.p.params, axis=1)
            s = np.append(s, jump.p.sigmas, axis=1)

        x = np.append(x, self.Periodic.p.params, axis=1)
        s = np.append(s, self.Periodic.p.sigmas, axis=1)

        for i in range(3):
            residuals.append(l[i] - np.dot(self.A(constrains=False), x[i, :]))

            ss = np.abs(np.divide(residuals[i], factor[i]))
            index.append(ss <= LIMIT)

            f = np.ones((l.shape[1],))

            sw = np.power(10, LIMIT - ss[ss > LIMIT])
            sw[sw < np.finfo(np.float).eps] = np.finfo(np.float).eps
            f[ss > LIMIT] = sw

            p.append(np.square(np.divide(f, factor[i])))

        self.C = x
        self.S = s
        self.F = np.array(index)
        self.R = np.array(residuals)
        self.factor = factor
        self.P = np.array(p)

    def adjust_lsq(self, Ai, Li):

        A = Ai(constrains=True)
        L = Ai.get_l(Li, constrains=True)

        cst_pass = False
        iteration = 0
        factor = 1
        So = 1
        dof = (Ai.shape[0] - Ai.shape[1])
        X1 = chi2.ppf(1 - 0.05 / 2, dof)
        X2 = chi2.ppf(0.05 / 2, dof)

        s = np.array([])
        v = np.array([])
        C = np.array([])

        P = Ai.get_p(constrains=True)

        while not cst_pass and iteration <= 10:

            W = np.sqrt(P)

            Aw = np.multiply(W[:, None], A)
            Lw = np.multiply(W, L)

            C = np.linalg.lstsq(Aw, Lw, rcond=-1)[0]

            v = L - np.dot(A, C)

            # unit variance
            So = np.sqrt(np.dot(v, np.multiply(P, v)) / dof)

            x = np.power(So, 2) * dof

            # obtain the overall uncertainty predicted by lsq
            factor = factor * So

            # calculate the normalized sigmas
            s = np.abs(np.divide(v, factor))

            if x < X2 or x > X1:
                # if it falls in here it's because it didn't pass the Chi2 test
                cst_pass = False

                # reweigh by Mike's method of equal weight until 2 sigma
                f = np.ones((v.shape[0],))
                # f[s > LIMIT] = 1. / (np.power(10, LIMIT - s[s > LIMIT]))
                # do not allow sigmas > 100 m, which is basically not putting
                # the observation in. Otherwise, due to a model problem
                # (missing jump, etc) you end up with very unstable inversions
                # f[f > 500] = 500
                sw = np.power(10, LIMIT - s[s > LIMIT])
                sw[sw < np.finfo(np.float).eps] = np.finfo(np.float).eps
                f[s > LIMIT] = sw

                P = np.square(np.divide(f, factor))
            else:
                cst_pass = True

            iteration += 1

        # make sure there are no values below eps. Otherwise matrix becomes singular
        P[P < np.finfo(np.float).eps] = 1e-6

        # some statistics
        SS = np.linalg.inv(np.dot(A.transpose(), np.multiply(P[:, None], A)))

        sigma = So * np.sqrt(np.diag(SS))

        # mark observations with sigma <= LIMIT
        index = Ai.remove_constrains(s <= LIMIT)

        v = Ai.remove_constrains(v)

        return C, sigma, index, v, factor, P

    @staticmethod
    def chi2inv(chi, df):
        """Return prob(chisq >= chi, with df degrees of
        freedom).

        df must be even.
        """
        assert df & 1 == 0
        # XXX If chi is very large, exp(-m) will underflow to 0.
        m = chi / 2.0
        sum = term = np.exp(-m)
        for i in range(1, df // 2):
            term *= m / i
            sum += term
        # With small chi and large df, accumulated
        # roundoff error, plus error in
        # the platform exp(), can cause this to spill
        # a few ULP above 1.0. For
        # example, chi2P(100, 300) on my box
        # has sum == 1.0 + 2.0**-52 at this
        # point.  Returning a value even a teensy
        # bit over 1.0 is no good.
        return np.min(sum)

    @staticmethod
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    def get_outliers_list(self):
        """
        Function to obtain the outliers based on the ETMs sigma
        :return: a list containing the network code, station code and dates of the outliers in the time series
        """

        filt = self.F[0] * self.F[1] * self.F[2]
        dates = [pyDate.Date(mjd=mjd) for mjd in self.soln.mjd[~filt]]

        return [(net, stn, date) for net, stn, date in zip(repeat(self.NetworkCode), repeat(self.StationCode), dates)]


class PPPETM(ETM):

    def __init__(self, cnn, NetworkCode, StationCode, plotit=False, no_model=False):
        # load all the PPP coordinates available for this station
        # exclude ppp solutions in the exclude table and any solution that is more than 100 meters from the auto coord

        self.ppp_soln = PppSoln(cnn, NetworkCode, StationCode)

        ETM.__init__(self, cnn, self.ppp_soln, no_model)

        # no offset applied
        self.L = np.array([self.soln.x,
                           self.soln.y,
                           self.soln.z])

        # reduced to x y z coordinate of the station
        self.l = self.rotate_2neu(np.array([self.ppp_soln.x - self.ppp_soln.auto_x,
                                            self.ppp_soln.y - self.ppp_soln.auto_y,
                                            self.ppp_soln.z - self.ppp_soln.auto_z]))

        self.run_adjustment(cnn, self.l, plotit)


class GamitETM(ETM):

    def __init__(self, cnn, NetworkCode, StationCode, plotit=False,
                 no_model=False, gamit_soln=None, project=None):

        if gamit_soln is None:
            self.polyhedrons = cnn.query_float('SELECT "X", "Y", "Z", "Year", "DOY" FROM gamit_soln '
                                               'WHERE "Project" = \'%s\' AND "NetworkCode" = \'%s\' AND '
                                               '"StationCode" = \'%s\' '
                                               'ORDER BY "Year", "DOY", "NetworkCode", "StationCode"'
                                               % (project, NetworkCode, StationCode))

            self.gamit_soln = GamitSoln(cnn, self.polyhedrons, NetworkCode, StationCode)

        else:
            # load the GAMIT polyhedrons
            self.gamit_soln = gamit_soln

        ETM.__init__(self, cnn, self.gamit_soln, no_model)

        # no offset applied
        self.L = np.array([self.gamit_soln.x,
                           self.gamit_soln.y,
                           self.gamit_soln.z])

        # reduced to x y z coordinate of the station
        self.l = self.rotate_2neu(np.array([self.gamit_soln.x - self.gamit_soln.auto_x,
                                            self.gamit_soln.y - self.gamit_soln.auto_y,
                                            self.gamit_soln.z - self.gamit_soln.auto_z]))

        self.run_adjustment(cnn, self.l, plotit)

    def get_etm_soln_list(self, use_ppp_model=False, cnn=None):
        # this function return the values of the ETM ONLY

        dict_o = []
        if self.A is not None:

            neu = []

            if not use_ppp_model:
                # get residuals from GAMIT solutions to GAMIT model
                for i in range(3):
                    neu.append(np.dot(self.A, self.C[i]))
            else:
                # get residuals from GAMIT solutions to PPP model
                etm = PPPETM(cnn, self.NetworkCode, self.StationCode)
                # DDG: 20-SEP-2018 compare using MJD not FYEAR to avoid round off errors
                index = np.isin(etm.soln.mjds, self.soln.mjd)
                for i in range(3):
                    # use the etm object to obtain the design matrix that matches the dimensions of self.soln.t
                    neu.append(np.dot(etm.As[index, :], etm.C[i]))

                del etm

            rxyz = self.rotate_2xyz(np.array(neu)) + np.array([self.soln.auto_x, self.soln.auto_y, self.soln.auto_z])

            dict_o += [(net_stn, x, y, z, year, doy, fyear)
                       for x, y, z, net_stn, year, doy, fyear in
                       zip(rxyz[0].tolist(), rxyz[1].tolist(), rxyz[2].tolist(),
                           repeat(self.NetworkCode + '.' + self.StationCode),
                           [date.year for date in self.gamit_soln.date],
                           [date.doy for date in self.gamit_soln.date],
                           [date.fyear for date in self.gamit_soln.date])]
        else:
            raise pyETMException_NoDesignMatrix('No design matrix available for %s.%s' %
                                                (self.NetworkCode, self.StationCode))

        return dict_o


class DailyRep(ETM):

    def __init__(self, cnn, NetworkCode, StationCode, plotit=False,
                 no_model=False, gamit_soln=None, project=None):

        if gamit_soln is None:
            self.polyhedrons = cnn.query_float('SELECT "X", "Y", "Z", "Year", "DOY" FROM gamit_soln '
                                               'WHERE "Project" = \'%s\' AND "NetworkCode" = \'%s\' AND '
                                               '"StationCode" = \'%s\' '
                                               'ORDER BY "Year", "DOY", "NetworkCode", "StationCode"'
                                               % (project, NetworkCode, StationCode))

            self.gamit_soln = GamitSoln(cnn, self.polyhedrons, NetworkCode, StationCode)

        else:
            # load the GAMIT polyhedrons
            self.gamit_soln = gamit_soln

        ETM.__init__(self, cnn, self.gamit_soln, no_model, False, False, False)

        # for repetitivities, vector with difference
        self.l = self.rotate_2neu(np.array([self.gamit_soln.x,
                                            self.gamit_soln.y,
                                            self.gamit_soln.z]))

        # for repetitivities, same vector for both
        self.L = self.l

        self.run_adjustment(cnn, self.l, plotit)

    def get_residuals_dict(self):
        # this function return the values of the ETM ONLY

        dict_o = []
        if self.A is not None:

            neu = []

            for i in range(3):
                neu.append(np.dot(self.A, self.C[i]))

            xyz = self.rotate_2xyz(np.array(neu)) + np.array([self.soln.auto_x, self.soln.auto_y, self.soln.auto_z])

            rxyz = xyz - self.L

            px = np.ones(self.P[0].shape)
            py = np.ones(self.P[1].shape)
            pz = np.ones(self.P[2].shape)

            dict_o += [(net, stn, x, y, z, sigx, sigy, sigz, year, doy)
                       for x, y, z, sigx, sigy, sigz, net, stn, year, doy in
                       zip(rxyz[0].tolist(), rxyz[1].tolist(), rxyz[2].tolist(),
                           px.tolist(), py.tolist(), pz.tolist(),
                           repeat(self.NetworkCode), repeat(self.StationCode),
                           [date.year for date in self.gamit_soln.date],
                           [date.doy for date in self.gamit_soln.date])]
        else:
            raise pyETMException_NoDesignMatrix('No design matrix available for %s.%s' %
                                                (self.NetworkCode, self.StationCode))

        return dict_o


"""
Project: Parallel.Archive
Date: 3/3/17 11:27 AM
Author: Demian D. Gomez
"""

import warnings
from itertools import repeat
from os.path import getmtime
from time import time
from zlib import crc32

import numpy as np
from Utils import rotct2lg
from numpy import cos
from numpy import pi
from numpy import sin
from pyBunch import Bunch


def tic():
    global tt
    tt = time()


def toc(text):
    global tt
    print(text + ': ' + str(time() - tt))


LIMIT = 2.5

NO_EFFECT = None
UNDETERMINED = -1
GENERIC_JUMP = 0
CO_SEISMIC_DECAY = 1
CO_SEISMIC_JUMP_DECAY = 2

EQ_MIN_DAYS = 15
JP_MIN_DAYS = 5

DEFAULT_RELAXATION = np.array([0.5])
DEFAULT_POL_TERMS = 2
DEFAULT_FREQUENCIES = np.array(
    (1 / 365.25, 1 / (365.25 / 2)))  # (1 yr, 6 months) expressed in 1/days (one year = 365.25)


class pyETMException(Exception):

    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class pyETMException_NoDesignMatrix(pyETMException):
    pass


def distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1 = lon1 * pi / 180
    lat1 = lat1 * pi / 180
    lon2 = lon2 * pi / 180
    lat2 = lat2 * pi / 180
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def to_postgres(dictionary):
    if isinstance(dictionary, dict):
        for key, val in list(dictionary.items()):
            if isinstance(val, np.ndarray):
                dictionary[key] = str(val.flatten().tolist()).replace('[', '{').replace(']', '}')
    else:
        dictionary = str(dictionary.flatten().tolist()).replace('[', '{').replace(']', '}')

    return dictionary


def to_list(dictionary):
    for key, val in list(dictionary.items()):
        if isinstance(val, np.ndarray):
            dictionary[key] = val.tolist()

        if isinstance(val, pyDate.datetime):
            dictionary[key] = val.strftime('%Y-%m-%d %H:%M:%S')

    return dictionary


class PppSoln:
    """"class to extract the PPP solutions from the database"""

    def __init__(self, cnn, NetworkCode, StationCode):

        self.NetworkCode = NetworkCode
        self.StationCode = StationCode
        self.hash = 0

        self.type = 'ppp'

        # get the station from the stations table
        stn = cnn.query('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                        % (NetworkCode, StationCode))

        stn = stn.dictresult()[0]

        if stn['lat'] is not None:
            self.lat = np.array([float(stn['lat'])])
            self.lon = np.array([float(stn['lon'])])
            self.height = np.array([float(stn['height'])])

            x = np.array([float(stn['auto_x'])])
            y = np.array([float(stn['auto_y'])])
            z = np.array([float(stn['auto_z'])])

            if stn['max_dist'] is not None:
                self.max_dist = stn['max_dist']
            else:
                self.max_dist = 20

            # load all the PPP coordinates available for this station
            # exclude ppp solutions in the exclude table and any solution that is more than 20 meters from the simple
            # linear trend calculated above

            self.excluded = cnn.query_float('SELECT "Year", "DOY" FROM ppp_soln_excl '
                                            'WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                                            % (NetworkCode, StationCode))

            self.table = cnn.query_float(
                'SELECT "X", "Y", "Z", "Year", "DOY" FROM ppp_soln p1 '
                'WHERE p1."NetworkCode" = \'%s\' AND p1."StationCode" = \'%s\' ORDER BY "Year", "DOY"'
                % (NetworkCode, StationCode))

            self.table = [item for item in self.table
                          if np.sqrt(np.square(item[0] - x) + np.square(item[1] - y) + np.square(item[2] - z)) <=
                          self.max_dist and item[3:] not in self.excluded]

            self.blunders = [item for item in self.table
                             if np.sqrt(np.square(item[0] - x) + np.square(item[1] - y) + np.square(item[2] - z)) >
                             self.max_dist and item[3:] not in self.excluded]

            self.solutions = len(self.table)

            self.ts_blu = np.array([pyDate.Date(year=item[3], doy=item[4]).fyear for item in self.blunders])

            if self.solutions >= 1:
                a = np.array(self.table)

                self.x = a[:, 0]
                self.y = a[:, 1]
                self.z = a[:, 2]
                self.t = np.array([pyDate.Date(year=item[0], doy=item[1]).fyear for item in a[:, 3:5]])
                self.mjd = np.array([pyDate.Date(year=item[0], doy=item[1]).mjd for item in a[:, 3:5]])

                # continuous time vector for plots
                ts = np.arange(np.min(self.mjd), np.max(self.mjd) + 1, 1)
                self.ts = np.array([pyDate.Date(mjd=tts).fyear for tts in ts])
            else:
                if len(self.blunders) >= 1:
                    raise pyETMException('No viable PPP solutions available for %s.%s (all blunders!)\n'
                                         '  -> min distance to station coordinate is %.1f meters'
                                         % (
                                         NetworkCode, StationCode, np.array([item[5] for item in self.blunders]).min()))
                else:
                    raise pyETMException('No PPP solutions available for %s.%s' % (NetworkCode, StationCode))

            # get a list of the epochs with files but no solutions.
            # This will be shown in the outliers plot as a special marker

            rnx = cnn.query(
                'SELECT r."ObservationFYear" FROM rinex_proc as r '
                'LEFT JOIN ppp_soln as p ON '
                'r."NetworkCode" = p."NetworkCode" AND '
                'r."StationCode" = p."StationCode" AND '
                'r."ObservationYear" = p."Year"    AND '
                'r."ObservationDOY"  = p."DOY"'
                'WHERE r."NetworkCode" = \'%s\' AND r."StationCode" = \'%s\' AND '
                'p."NetworkCode" IS NULL' % (NetworkCode, StationCode))

            self.rnx_no_ppp = rnx.getresult()

            self.ts_ns = np.array([item for item in self.rnx_no_ppp])

            self.completion = 100. - float(len(self.ts_ns)) / float(len(self.ts_ns) + len(self.t)) * 100.

            self.hash = crc32(str(len(self.t) + len(self.blunders)) + ' ' + str(ts[0]) + ' ' + str(ts[-1]))

        else:
            raise pyETMException('Station %s.%s has no valid metadata in the stations table.'
                                 % (NetworkCode, StationCode))


class GamitSoln:
    """"class to extract the GAMIT polyhedrons from the database"""

    def __init__(self, cnn, polyhedrons, NetworkCode, StationCode):

        self.NetworkCode = NetworkCode
        self.StationCode = StationCode
        self.hash = 0

        self.type = 'gamit'

        # get the station from the stations table
        stn = cnn.query_float('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                              % (NetworkCode, StationCode), as_dict=True)[0]

        if stn['lat'] is not None:
            self.lat = np.array([float(stn['lat'])])
            self.lon = np.array([float(stn['lon'])])
            self.height = np.array([stn['height']])

            if stn['max_dist'] is not None:
                self.max_dist = stn['max_dist']
            else:
                self.max_dist = 20

            self.solutions = len(polyhedrons)

            # blunders
            self.blunders = []
            self.ts_blu = np.array([])

            if self.solutions >= 1:
                a = np.array(polyhedrons)

                if np.sqrt(np.square(np.sum(np.square(a[0, 0:3])))) > 6.3e3:
                    # coordinates given in XYZ
                    nb = np.sqrt(np.square(np.sum(
                        np.square(a[:, 0:3] - np.array([stn['auto_x'], stn['auto_y'], stn['auto_z']])), axis=1))) \
                         <= self.max_dist
                else:
                    # coordinates are differences
                    nb = np.sqrt(np.square(np.sum(np.square(a[:, 0:3]), axis=1))) <= self.max_dist

                if np.any(nb):
                    self.x = a[nb, 0]
                    self.y = a[nb, 1]
                    self.z = a[nb, 2]
                    self.t = np.array([pyDate.Date(year=item[0], doy=item[1]).fyear for item in a[nb, 3:5]])
                    self.mjd = np.array([pyDate.Date(year=item[0], doy=item[1]).mjd for item in a[nb, 3:5]])

                    self.date = [pyDate.Date(year=item[0], doy=item[1]) for item in a[nb, 3:5]]

                    # continuous time vector for plots
                    ts = np.arange(np.min(self.mjd), np.max(self.mjd) + 1, 1)
                    self.ts = np.array([pyDate.Date(mjd=tts).fyear for tts in ts])
                else:
                    dd = np.sqrt(np.square(np.sum(
                        np.square(a[:, 0:3] - np.array([stn['auto_x'], stn['auto_y'], stn['auto_z']])), axis=1)))

                    raise pyETMException('No viable PPP solutions available for %s.%s (all blunders!)\n'
                                         '  -> min distance to station coordinate is %.1f meters'
                                         % (NetworkCode, StationCode, dd.min()))
            else:
                raise pyETMException('No GAMIT polyhedrons vertices available for %s.%s' % (NetworkCode, StationCode))

            # get a list of the epochs with files but no solutions.
            # This will be shown in the outliers plot as a special marker
            rnx = cnn.query(
                'SELECT r.* FROM rinex_proc as r '
                'LEFT JOIN gamit_soln as p ON '
                'r."NetworkCode" = p."NetworkCode" AND '
                'r."StationCode" = p."StationCode" AND '
                'r."ObservationYear" = p."Year"    AND '
                'r."ObservationDOY"  = p."DOY"'
                'WHERE r."NetworkCode" = \'%s\' AND r."StationCode" = \'%s\' AND '
                'p."NetworkCode" IS NULL' % (NetworkCode, StationCode))

            self.rnx_no_ppp = rnx.dictresult()
            self.ts_ns = np.array([float(item['ObservationFYear']) for item in self.rnx_no_ppp])

            self.completion = 100. - float(len(self.ts_ns)) / float(len(self.ts_ns) + len(self.t)) * 100.

            self.hash = crc32(str(len(self.t) + len(self.blunders)) + ' ' + str(ts[0]) + ' ' + str(ts[-1]))

        else:
            raise pyETMException('Station %s.%s has no valid metadata in the stations table.'
                                 % (NetworkCode, StationCode))


class JumpTable:

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, FitEarthquakes=True, FitGenericJumps=True):

        self.table = []

        # get earthquakes for this station
        self.earthquakes = Earthquakes(cnn, NetworkCode, StationCode, solution, t, FitEarthquakes)

        self.generic_jumps = GenericJumps(cnn, NetworkCode, StationCode, solution, t, FitGenericJumps)

        jumps = self.earthquakes.table + self.generic_jumps.table

        jumps.sort()

        # add the relevant jumps, make sure none are incompatible
        for jump in jumps:
            self.insert_jump(jump)

        # add the "NO_EFFECT" jumps and resort the table
        ne_jumps = [j for j in jumps if j.p.jump_type == NO_EFFECT
                    and j.date > pyDate.Date(fyear=t.min()) < j.date < pyDate.Date(fyear=t.max())]

        self.table += ne_jumps

        self.table.sort()

        self.param_count = sum([jump.param_count for jump in self.table])

        self.constrains = np.array([])

    def insert_jump(self, jump):

        if len(self.table) == 0:
            if jump.p.jump_type != NO_EFFECT:
                self.table.append(jump)
        else:
            # take last jump and compare to adding jump
            jj = self.table[-1]

            if jump.p.jump_type != NO_EFFECT:
                result, decision = jj == jump

                if not result:
                    # jumps are not equal, add it
                    self.table.append(jump)
                else:
                    # decision branches:
                    # 1) decision == jump, remove previous; add jump
                    # 2) decision == jj  , do not add jump (i.e. do nothing)
                    if decision is jump:
                        self.table.pop(-1)
                        self.table.append(jump)

    def get_design_ts(self, t):

        A = np.array([])

        # get the design matrix for the jump table
        for jump in self.table:
            if jump.p.jump_type is not NO_EFFECT:
                a = jump.eval(t)

                if a.size:
                    if A.size:
                        # if A is not empty, verify that this jump will not make the matrix singular
                        tA = np.column_stack((A, a))
                        # getting the condition number might trigger divide_zero warning => turn off
                        np.seterr(divide='ignore', invalid='ignore')
                        if np.linalg.cond(tA) < 1e10:
                            # adding this jumps doesn't make the matrix singular
                            A = tA
                        else:
                            # flag this jump by setting its type = None
                            jump.remove()
                            warnings.warn('%s had to be removed due to high condition number' % str(jump))
                    else:
                        A = a

        return A

    def load_parameters(self, params, sigmas):

        for jump in self.table:
            jump.load_parameters(params=params, sigmas=sigmas)

    def print_parameters(self, lat, lon):

        output_n = ['Year     Relx    [mm]']
        output_e = ['Year     Relx    [mm]']
        output_u = ['Year     Relx    [mm]']

        for jump in self.table:

            if jump.p.jump_type is not NO_EFFECT:

                # relaxation counter
                rx = 0

                for j, p in enumerate(np.arange(jump.param_count)):
                    psc = jump.p.params[:, p]

                    jn, je, ju = ct2lg(psc[0], psc[1], psc[2], lat, lon)

                    if j == 0 and jump.p.jump_type in (GENERIC_JUMP, CO_SEISMIC_JUMP_DECAY):
                        output_n.append('{}      {:>7.1f}'.format(jump.date.yyyyddd(), jn[0] * 1000.0))
                        output_e.append('{}      {:>7.1f}'.format(jump.date.yyyyddd(), je[0] * 1000.0))
                        output_u.append('{}      {:>7.1f}'.format(jump.date.yyyyddd(), ju[0] * 1000.0))
                    else:

                        output_n.append('{} {:4.2f} {:>7.1f}'.format(jump.date.yyyyddd(), jump.p.relaxation[rx],
                                                                     jn[0] * 1000.0))
                        output_e.append('{} {:4.2f} {:>7.1f}'.format(jump.date.yyyyddd(), jump.p.relaxation[rx],
                                                                     je[0] * 1000.0))
                        output_u.append('{} {:4.2f} {:>7.1f}'.format(jump.date.yyyyddd(), jump.p.relaxation[rx],
                                                                     ju[0] * 1000.0))
                        # relaxation counter
                        rx += 1

        if len(output_n) > 22:
            output_n = output_n[0:22] + ['Table too long to print!']
            output_e = output_e[0:22] + ['Table too long to print!']
            output_u = output_u[0:22] + ['Table too long to print!']

        return '\n'.join(output_n), '\n'.join(output_e), '\n'.join(output_u)


class EtmFunction(object):

    def __init__(self, **kwargs):

        self.p = Bunch()

        self.p.NetworkCode = kwargs['NetworkCode']
        self.p.StationCode = kwargs['StationCode']
        self.p.soln = kwargs['solution']

        self.p.params = np.array([])
        self.p.sigmas = np.array([])
        self.p.object = ''
        self.p.metadata = None
        self.p.hash = 0

        self.param_count = 0
        self.column_index = np.array([])
        self.format_str = ''

    def load_parameters(self, **kwargs):

        params = kwargs['params']
        sigmas = kwargs['sigmas']

        if params.ndim == 1:
            # parameters coming from the database, reshape
            params = params.reshape((3, params.shape[0] / 3))

        if sigmas.ndim == 1:
            # parameters coming from the database, reshape
            sigmas = sigmas.reshape((3, sigmas.shape[0] / 3))

        # determine if parameters are coming from the X vector (LSQ) or from the database (solution for self only)
        if params.shape[1] > self.param_count:
            # X vector
            self.p.params = params[:, self.column_index]
            self.p.sigmas = sigmas[:, self.column_index]
        else:
            # database (solution for self only; no need for column_index)
            self.p.params = params
            self.p.sigmas = sigmas


class Jump(EtmFunction):
    """
    generic jump (mechanic jump, frame change, etc) class
    """

    def __init__(self, NetworkCode, StationCode, solution, t, date, metadata, dtype=UNDETERMINED):

        super(Jump, self).__init__(NetworkCode=NetworkCode, StationCode=StationCode, solution=solution)

        # in the future, can load parameters from the db
        self.p.object = 'jump'

        # define initial state variables
        self.date = date

        self.p.jump_date = date.datetime()
        self.p.metadata = metadata
        self.p.jump_type = dtype

        self.design = Jump.eval(self, t)

        if np.any(self.design) and not np.all(self.design):
            self.p.jump_type = GENERIC_JUMP
            self.param_count = 1
        else:
            self.p.jump_type = NO_EFFECT
            self.param_count = 0

        self.p.hash = crc32(str(self.date))

    def remove(self):
        # this method will make this jump type = 0 and adjust its params
        self.p.jump_type = NO_EFFECT
        self.param_count = 0

    def eval(self, t):
        # given a time vector t, return the design matrix column vector(s)
        if self.p.jump_type == NO_EFFECT:
            return np.array([])

        ht = np.zeros((t.shape[0], 1))

        ht[t > self.date.fyear] = 1.

        return ht

    def __eq__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid. Can compare two Jump objects')

        # compare two jumps together and make sure they will not generate a singular (or near singular) system of eq
        c = np.sum(np.logical_xor(self.design[:, 0], jump.design[:, 0]))

        if self.p.jump_type in (CO_SEISMIC_JUMP_DECAY,
                                CO_SEISMIC_DECAY) and jump.p.jump_type in (CO_SEISMIC_JUMP_DECAY, CO_SEISMIC_DECAY):

            # if self is a co-seismic jump and next jump is also co-seismic
            # and there are less than two weeks of data to constrain params, return false
            if c <= EQ_MIN_DAYS:
                return True, jump
            else:
                return False, None

        elif self.p.jump_type in (CO_SEISMIC_JUMP_DECAY,
                                  CO_SEISMIC_DECAY, GENERIC_JUMP) and jump.p.jump_type == GENERIC_JUMP:

            if c <= JP_MIN_DAYS:
                # can't fit the co-seismic or generic jump AND the generic jump after, remove "jump" generic jump
                return True, self
            else:
                return False, None

        elif self.p.jump_type == GENERIC_JUMP and jump.p.jump_type == (CO_SEISMIC_JUMP_DECAY, CO_SEISMIC_DECAY):

            if c <= JP_MIN_DAYS:
                # if generic jump before an earthquake jump and less than 5 days, co-seismic prevails
                return True, jump
            else:
                return False, None

        elif self.p.jump_type == NO_EFFECT and jump.p.jump_type != NO_EFFECT:
            # if comparing to a self that has NO_EFFECT, remove and keep jump
            return True, jump

        elif self.p.jump_type != NO_EFFECT and jump.p.jump_type == NO_EFFECT:
            # if comparing against a jump that has NO_EFFECT, remove jump keep self
            return True, self

        elif self.p.jump_type == NO_EFFECT and jump.p.jump_type == NO_EFFECT:
            # no jump has an effect, return None. This will be interpreted as False (if not result)
            return None, None

    def __str__(self):
        return '(' + str(self.date) + '), ' + str(self.p.jump_type) + ', "' + str(self.p.jump_type) + '"'

    def __repr__(self):
        return 'pyPPPETM.Jump(' + str(self) + ')'

    def __lt__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear < jump.date.fyear

    def __le__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear <= jump.date.fyear

    def __gt__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear > jump.date.fyear

    def __ge__(self, jump):

        if not isinstance(jump, Jump):
            raise pyETMException('type: ' + str(type(jump)) + ' invalid.  Can only compare Jump objects')

        return self.date.fyear >= jump.date.fyear

    def __hash__(self):
        # to make the object hashable
        return hash(self.date.fyear)


class CoSeisJump(Jump):

    def __init__(self, NetworkCode, StationCode, solution, t, date, relaxation, metadata, dtype=UNDETERMINED):

        # super-class initialization
        Jump.__init__(self, NetworkCode, StationCode, solution, t, date, metadata, dtype)

        if dtype is NO_EFFECT:
            # passing default_type == NO_EFFECT, add the jump but make it NO_EFFECT by default
            self.p.jump_type = NO_EFFECT
            self.params_count = 0
            self.p.relaxation = None

            self.design = np.array([])
            return

        if self.p.jump_type == NO_EFFECT:
            # came back from init as NO_EFFECT. May be a jump before t.min()
            # assign just the decay
            self.p.jump_type = CO_SEISMIC_DECAY
        else:
            self.p.jump_type = CO_SEISMIC_JUMP_DECAY

        # if T is an array, it contains the corresponding decays
        # otherwise, it is a single decay
        if not isinstance(relaxation, np.ndarray):
            relaxation = np.array([relaxation])

        self.param_count += relaxation.shape[0]
        self.nr = relaxation.shape[0]
        self.p.relaxation = relaxation

        self.design = self.eval(t)

        # test if earthquake generates at least 10 days of data to adjust
        if self.design.size:
            if np.count_nonzero(self.design[:, -1]) < 10:
                if self.p.jump_type == CO_SEISMIC_JUMP_DECAY:
                    # was a jump and decay, leave the jump
                    self.p.jump_type = GENERIC_JUMP
                    self.p.params = np.zeros((3, 1))
                    self.p.sigmas = np.zeros((3, 1))
                    self.param_count -= 1
                    # reevaluate the design matrix!
                    self.design = self.eval(t)
                else:
                    self.p.jump_type = NO_EFFECT
                    self.p.params = np.array([])
                    self.p.sigmas = np.array([])
                    self.param_count = 0
        else:
            self.p.jump_type = NO_EFFECT
            self.p.params = np.array([])
            self.p.sigmas = np.array([])
            self.param_count = 0

        self.p.hash += crc32(str(self.param_count) + ' ' + str(self.p.jump_type) + ' ' + str(self.p.relaxation))

    def eval(self, t):

        ht = Jump.eval(self, t)

        # if there is nothing in ht, then there is no expected output, return none
        if not np.any(ht):
            return np.array([])

        # if it was determined that this is just a generic jump, return ht
        if self.p.jump_type == GENERIC_JUMP:
            return ht

        # support more than one decay
        hl = np.zeros((t.shape[0], self.nr))

        for i, T in enumerate(self.p.relaxation):
            hl[t > self.date.fyear, i] = np.log10(1. + (t[t > self.date.fyear] - self.date.fyear) / T)

        # if it's both jump and decay, return ht + hl
        if np.any(hl) and self.p.jump_type == CO_SEISMIC_JUMP_DECAY:
            return np.column_stack((ht, hl))

        # if decay only, return hl
        elif np.any(hl) and self.p.jump_type == CO_SEISMIC_DECAY:
            return hl

    def __str__(self):
        return '(' + str(self.date) + '), ' + str(self.p.jump_type) + ', ' + str(self.p.relaxation) + ', "' + str(
            self.p.metadata) + '"'

    def __repr__(self):
        return 'pyPPPETM.CoSeisJump(' + str(self) + ')'


class Earthquakes:

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, FitEarthquakes=True):

        self.StationCode = StationCode
        self.NetworkCode = NetworkCode

        # station location
        stn = cnn.query('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                        % (NetworkCode, StationCode))

        stn = stn.dictresult()[0]

        # load metadata
        lat = float(stn['lat'])
        lon = float(stn['lon'])

        # establish the limit dates. Ignore jumps before 5 years from the earthquake
        sdate = pyDate.Date(fyear=t.min() - 5)
        edate = pyDate.Date(fyear=t.max())

        # get the earthquakes based on Mike's expression
        jumps = cnn.query('SELECT * FROM earthquakes WHERE date BETWEEN \'%s\' AND \'%s\' ORDER BY date'
                          % (sdate.yyyymmdd(), edate.yyyymmdd()))
        jumps = jumps.dictresult()

        # check if data range returned any jumps
        if jumps and FitEarthquakes:
            eq = [[float(jump['lat']), float(jump['lon']), float(jump['mag']),
                   int(jump['date'].year), int(jump['date'].month), int(jump['date'].day),
                   int(jump['date'].hour), int(jump['date'].minute), int(jump['date'].second)] for jump in jumps]

            eq = np.array(list(eq))

            dist = distance(lon, lat, eq[:, 1], eq[:, 0])

            m = -0.8717 * (np.log10(dist) - 2.25) + 0.4901 * (eq[:, 2] - 6.6928)
            # build the earthquake jump table
            # remove event events that happened the same day

            eq_jumps = list(set((float(eq[2]), pyDate.Date(year=int(eq[3]), month=int(eq[4]), day=int(eq[5]),
                                                           hour=int(eq[6]), minute=int(eq[7]), second=int(eq[8])))
                                for eq in eq[m > 0, :]))

            eq_jumps.sort(key=lambda x: (x[1], x[0]))

            # open the jumps table
            jp = cnn.query_float('SELECT * FROM etm_params WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' '
                                 'AND soln = \'%s\' AND jump_type <> 0 AND object = \'jump\''
                                 % (NetworkCode, StationCode, solution), as_dict=True)

            # start by collapsing all earthquakes for the same day.
            # Do not allow more than one earthquake on the same day
            f_jumps = []
            next_date = None

            for mag, date in eq_jumps:

                # jumps are analyzed in windows that are EQ_MIN_DAYS long
                # a date should not be analyzed is it's < next_date
                if next_date is not None:
                    if date < next_date:
                        continue

                # obtain jumps in a EQ_MIN_DAYS window
                jumps = [(m, d) for m, d in eq_jumps if date <= d < date + EQ_MIN_DAYS]

                if len(jumps) > 1:
                    # if more than one jump, get the max magnitude
                    mmag = max([m for m, _ in jumps])

                    # only keep the earthquake with the largest magnitude
                    for m, d in jumps:

                        table = [j['action'] for j in jp if j['Year'] == d.year and j['DOY'] == d.doy]

                        # get a different relaxation for this date
                        relax = [j['relaxation'] for j in jp if j['Year'] == d.year and j['DOY'] == d.doy]

                        if relax:
                            if relax[0] is not None:
                                relaxation = np.array(relax[0])
                            else:
                                relaxation = DEFAULT_RELAXATION
                        else:
                            relaxation = DEFAULT_RELAXATION

                        # if present in jump table, with either + of -, don't use default decay
                        if m == mmag and '-' not in table:
                            f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, d, relaxation,
                                                   'mag=%.1f' % m)]
                            # once the jump was added, exit for loop
                            break
                        else:
                            # add only if in jump list with a '+'
                            if '+' in table:
                                f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, d,
                                                       relaxation, 'mag=%.1f' % m)]
                                # once the jump was added, exit for loop
                                break
                            else:
                                f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, d,
                                                       relaxation, 'mag=%.1f' % m, NO_EFFECT)]
                else:
                    # add, unless marked in table with '-'
                    table = [j['action'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]
                    # get a different relaxation for this date
                    relax = [j['relaxation'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]

                    if relax:
                        if relax[0] is not None:
                            relaxation = np.array(relax[0])
                        else:
                            relaxation = DEFAULT_RELAXATION
                    else:
                        relaxation = DEFAULT_RELAXATION

                    if '-' not in table:
                        f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, date,
                                               relaxation, 'mag=%.1f' % mag)]
                    else:
                        # add it with NO_EFFECT for display purposes
                        f_jumps += [CoSeisJump(NetworkCode, StationCode, solution, t, date,
                                               relaxation, 'mag=%.1f' % mag, NO_EFFECT)]

                next_date = date + EQ_MIN_DAYS

            # final jump table
            self.table = f_jumps
        else:
            self.table = []


class GenericJumps(object):

    def __init__(self, cnn, NetworkCode, StationCode, solution, t, FitGenericJumps=True):

        self.solution = solution
        self.table = []

        if t.size >= 2:
            # analyze if it is possible to add the jumps (based on the available data)
            wt = np.sort(np.unique(t - np.fix(t)))
            # analyze the gaps in the data
            dt = np.diff(wt)
            # max dt (internal)
            dtmax = np.max(dt)
            # dt wrapped around
            dt_interyr = 1 - wt[-1] + wt[0]

            if dt_interyr > dtmax:
                dtmax = dt_interyr

            if dtmax <= 0.2465 and FitGenericJumps:
                # put jumps in
                self.add_metadata_jumps = True
            else:
                # no jumps
                self.add_metadata_jumps = False
        else:
            self.add_metadata_jumps = False

        # open the jumps table
        jp = cnn.query('SELECT * FROM etm_params WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' '
                       'AND soln = \'%s\' AND jump_type = 0 AND object = \'jump\''
                       % (NetworkCode, StationCode, solution))

        jp = jp.dictresult()

        # get station information
        self.stninfo = pyStationInfo.StationInfo(cnn, NetworkCode, StationCode)

        for stninfo in self.stninfo.records[1:]:

            date = pyDate.Date(datetime=stninfo['DateStart'])

            table = [j['action'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]

            # add to list only if:
            # 1) add_meta = True AND there is no '-' OR
            # 2) add_meta = False AND there is a '+'

            if (not self.add_metadata_jumps and '+' in table) or (self.add_metadata_jumps and '-' not in table):
                self.table.append(Jump(NetworkCode, StationCode, solution, t, date,
                                       'Ant-Rec: %s-%s' % (stninfo['AntennaCode'], stninfo['ReceiverCode'])))

        # frame changes if ppp
        if solution == 'ppp':
            frames = cnn.query(
                'SELECT distinct on ("ReferenceFrame") "ReferenceFrame", "Year", "DOY" from ppp_soln WHERE '
                '"NetworkCode" = \'%s\' AND "StationCode" = \'%s\' order by "ReferenceFrame", "Year", "DOY"' %
                (NetworkCode, StationCode))

            frames = frames.dictresult()

            if len(frames) > 1:
                # more than one frame, add a jump
                frames.sort(key=lambda k: k['Year'])

                for frame in frames[1:]:
                    date = pyDate.Date(Year=frame['Year'], doy=frame['DOY'])

                    table = [j['action'] for j in jp if j['Year'] == date.year and j['DOY'] == date.doy]

                    if '-' not in table:
                        self.table.append(Jump(NetworkCode, StationCode, solution, t, date,
                                               'Frame Change: %s' % frame['ReferenceFrame']))

        # now check the jump table to add specific jumps
        jp = cnn.query('SELECT * FROM etm_params WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' '
                       'AND soln = \'%s\' AND jump_type = 0 AND object = \'jump\' '
                       'AND action = \'+\'' % (NetworkCode, StationCode, solution))

        jp = jp.dictresult()

        table = [j.date for j in self.table]

        for j in jp:
            date = pyDate.Date(Year=j['Year'], doy=j['DOY'])

            if date not in table:
                self.table.append(Jump(NetworkCode, StationCode, solution, t, date, 'mechanic-jump'))


class Periodic(EtmFunction):
    """"class to determine the periodic terms to be included in the ETM"""

    def __init__(self, NetworkCode, StationCode, solution, t, FitPeriodic=True):

        super(Periodic, self).__init__(NetworkCode=NetworkCode, StationCode=StationCode, solution=solution)

        # in the future, can load parameters from the db
        self.p.frequencies = DEFAULT_FREQUENCIES
        self.p.object = 'periodic'

        if t.size > 1 and FitPeriodic:
            # wrap around the solutions
            wt = np.sort(np.unique(t - np.fix(t)))

            # analyze the gaps in the data
            dt = np.diff(wt)

            # max dt (internal)
            dtmax = np.max(dt)

            # dt wrapped around
            dt_interyr = 1 - wt[-1] + wt[0]

            if dt_interyr > dtmax:
                dtmax = dt_interyr

            # save the value of the max wrapped delta time
            self.dt_max = dtmax

            # get the 50 % of Nyquist for each component (and convert to average fyear)
            self.nyquist = ((1 / self.p.frequencies) / 2.) * 0.5 * 1 / 365.25

            # frequency count
            self.frequency_count = int(np.sum(self.dt_max <= self.nyquist))

            # redefine the frequencies vector to accommodate only the frequencies that can be fit
            self.p.frequencies = self.p.frequencies[self.dt_max <= self.nyquist]

        else:
            # no periodic terms
            self.frequency_count = 0
            self.dt_max = 1  # one year of delta t

        self.design = self.get_design_ts(t)
        self.param_count = self.frequency_count * 2
        # declare the location of the answer (to be filled by Design object)
        self.column_index = np.array([])

        self.format_str = 'Periodic amp (' + \
                          ', '.join(['%.1f yr' % i for i in (1 / (self.p.frequencies * 365.25)).tolist()]) + \
                          ') N: %s E: %s U: %s [mm]'

        self.p.hash = crc32(str(self.p.frequencies))

    def get_design_ts(self, ts):
        # if dtmax < 3 months (90 days = 0.1232), then we can fit the annual
        # if dtmax < 1.5 months (45 days = 0.24657), then we can fit the semi-annual too

        if self.frequency_count > 0:
            f = self.p.frequencies
            f = np.tile(f, (ts.shape[0], 1))

            As = np.array(sin(2 * pi * f * 365.25 * np.tile(ts[:, np.newaxis], (1, f.shape[1]))))
            Ac = np.array(cos(2 * pi * f * 365.25 * np.tile(ts[:, np.newaxis], (1, f.shape[1]))))

            A = np.column_stack((As, Ac))
        else:
            # no periodic terms
            A = np.array([])

        return A

    def print_parameters(self, lat, lon):

        n = np.array([])
        e = np.array([])
        u = np.array([])

        for p in np.arange(self.param_count):
            psc = self.p.params[:, p]

            sn, se, su = ct2lg(psc[0], psc[1], psc[2], lat, lon)

            n = np.append(n, sn)
            e = np.append(e, se)
            u = np.append(u, su)

        n = n.reshape((2, self.param_count / 2))
        e = e.reshape((2, self.param_count / 2))
        u = u.reshape((2, self.param_count / 2))

        # calculate the amplitude of the components
        an = np.sqrt(np.square(n[0, :]) + np.square(n[1, :]))
        ae = np.sqrt(np.square(e[0, :]) + np.square(e[1, :]))
        au = np.sqrt(np.square(u[0, :]) + np.square(u[1, :]))

        return self.format_str % (np.array_str(an * 1000.0, precision=1),
                                  np.array_str(ae * 1000.0, precision=1),
                                  np.array_str(au * 1000.0, precision=1))


class Polynomial(EtmFunction):
    """"class to build the linear portion of the design matrix"""

    def __init__(self, NetworkCode, StationCode, solution, t, t_ref=0):

        super(Polynomial, self).__init__(NetworkCode=NetworkCode, StationCode=StationCode, solution=solution)

        # t ref (just the beginning of t vector)
        if t_ref == 0:
            t_ref = np.min(t)

        self.p.object = 'polynomial'
        self.p.t_ref = t_ref

        # in the future, can load parameters from the db
        self.terms = DEFAULT_POL_TERMS

        if self.terms == 1:
            self.format_str = 'Ref Position (' + '%.3f' % t_ref + ') X: {:.3f} Y: {:.3f} Z: {:.3f} [m]'

        elif self.terms == 2:
            self.format_str = 'Ref Position (' + '%.3f' % t_ref + ') X: {:.3f} Y: {:.3f} Z: {:.3f} [m]\n' \
                                                                  'Velocity N: {:.2f} E: {:.2f} U: {:.2f} [mm/yr]'

        elif self.terms == 3:
            self.format_str = 'Ref Position (' + '%.3f' % t_ref + ') X: {:.3f} Y: {:.3f} Z: {:.3f} [m]\n' \
                                                                  'Velocity N: {:.3f} E: {:.3f} U: {:.3f} [mm/yr]\n' \
                                                                  'Acceleration N: {:.2f} E: {:.2f} U: {:.2f} [mm/yr^2]'

        self.design = self.get_design_ts(t)

        # always first in the list of A, index columns are fixed
        self.column_index = np.arange(self.terms)
        # param count is the same as terms
        self.param_count = self.terms
        # save the hash of the object
        self.p.hash = crc32(str(self.terms))

    def load_parameters(self, params, sigmas, t_ref):

        super(Polynomial, self).load_parameters(params=params, sigmas=sigmas)

        self.p.t_ref = t_ref

    def print_parameters(self, lat, lon):

        params = np.array([])

        for p in np.arange(self.terms):
            if p == 0:
                params = self.p.params[:, 0]

            elif p > 0:
                n, e, u = ct2lg(self.p.params[0, p], self.p.params[1, p], self.p.params[2, p], lat, lon)
                params = np.append(params, (n * 1000, e * 1000, u * 1000))

        return self.format_str.format(*params.tolist())

    def get_design_ts(self, ts):

        A = np.zeros((ts.size, self.terms))

        for p in np.arange(self.terms):
            A[:, p] = np.power(ts - self.p.t_ref, p)

        return A


class Design(np.ndarray):

    def __new__(subtype, Linear, Jumps, Periodic, dtype=float, buffer=None, offset=0, strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__

        shape = (Linear.design.shape[0], Linear.param_count + Jumps.param_count + Periodic.param_count)
        A = super(Design, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)

        A[:, Linear.column_index] = Linear.design

        # determine the column_index for all objects
        col_index = Linear.param_count

        for jump in Jumps.table:
            # save the column index
            jump.column_index = np.arange(col_index, col_index + jump.param_count)
            # assign the portion of the design matrix
            A[:, jump.column_index] = jump.design
            # increment the col_index
            col_index += jump.param_count

        Periodic.column_index = np.arange(col_index, col_index + Periodic.param_count)

        A[:, Periodic.column_index] = Periodic.design

        # save the object list
        A.objects = (Linear, Jumps, Periodic)

        # save the number of total parameters
        A.linear_params = Linear.param_count
        A.jump_params = Jumps.param_count
        A.periodic_params = Periodic.param_count

        A.params = Linear.param_count + Jumps.param_count + Periodic.param_count

        # save the constrains matrix
        A.constrains = Jumps.constrains

        # Finally, we must return the newly created object:
        return A

    def __call__(self, ts=None, constrains=False):

        if ts is None:
            if constrains:
                if self.constrains.size:
                    A = self.copy()
                    # resize matrix (use A.resize so that it fills with zeros)
                    A.resize((self.shape[0] + self.constrains.shape[0], self.shape[1]), refcheck=False)
                    # apply constrains
                    A[-self.constrains.shape[0]:, self.jump_params] = self.constrains
                    return A

                else:
                    return self

            else:
                return self

        else:

            A = np.array([])

            for obj in self.objects:
                tA = obj.get_design_ts(ts)
                if A.size:
                    A = np.column_stack((A, tA)) if tA.size else A
                else:
                    A = tA

            return A

    def get_l(self, L, constrains=False):

        if constrains:
            if self.constrains.size:
                tL = L.copy()
                tL.resize((L.shape[0] + self.constrains.shape[0]), refcheck=False)
                return tL

            else:
                return L

        else:
            return L

    def get_p(self, constrains=False):
        # return a weight matrix full of ones with or without the extra elements for the constrains
        return np.ones((self.shape[0])) if not constrains else \
            np.ones((self.shape[0] + self.constrains.shape[0]))

    def remove_constrains(self, v):
        # remove the constrains to whatever vector is passed
        if self.constrains.size:
            return v[0:-self.constrains.shape[0]]
        else:
            return v


class ETM:

    def __init__(self, cnn, soln, no_model=False, FitEarthquakes=True, FitGenericJumps=True, FitPeriodic=True):

        # to display more verbose warnings
        # warnings.showwarning = self.warn_with_traceback

        self.C = np.array([])
        self.S = np.array([])
        self.F = np.array([])
        self.R = np.array([])
        self.P = np.array([])
        self.factor = np.array([])
        self.covar = np.zeros((3, 3))
        self.A = None
        self.soln = soln

        self.NetworkCode = soln.NetworkCode
        self.StationCode = soln.StationCode

        # save the function objects
        self.Linear = Polynomial(soln.NetworkCode, soln.StationCode, self.soln.type, soln.t)
        self.Periodic = Periodic(soln.NetworkCode, soln.StationCode, self.soln.type, soln.t, FitPeriodic)
        self.Jumps = JumpTable(cnn, soln.NetworkCode, soln.StationCode, soln.type, soln.t,
                               FitEarthquakes, FitGenericJumps)
        # calculate the hash value for this station
        # now hash also includes the timestamp of the last time pyETM was modified.
        self.hash = soln.hash + crc32(str(getmtime(__file__)))

        # anything less than four is not worth it
        if soln.solutions > 4 and not no_model:

            # to obtain the parameters
            self.A = Design(self.Linear, self.Jumps, self.Periodic)

            # check if problem can be solved!
            if self.A.shape[1] >= soln.solutions:
                self.A = None
                return

            self.As = self.A(soln.ts)

    def run_adjustment(self, cnn, l, plotit=False):

        c = []
        f = []
        s = []
        r = []
        p = []
        factor = []
        cov = np.zeros((3, 1))

        if self.A is not None:
            # try to load the last ETM solution from the database

            etm_objects = cnn.query_float('SELECT * FROM etmsv2 WHERE "NetworkCode" = \'%s\' '
                                          'AND "StationCode" = \'%s\' AND soln = \'%s\''
                                          % (self.NetworkCode, self.StationCode, self.soln.type), as_dict=True)

            db_hash_sum = sum([obj['hash'] for obj in etm_objects])
            ob_hash_sum = sum([o.p.hash for o in self.Jumps.table + [self.Periodic] + [self.Linear]]) + self.hash
            cn_object_sum = len(self.Jumps.table) + 2

            # -1 to account for the var_factor entry
            if len(etm_objects) - 1 == cn_object_sum and db_hash_sum == ob_hash_sum:
                # load the parameters from th db
                self.load_parameters(etm_objects, l)
            else:
                # purge table and recompute
                cnn.query('DELETE FROM etmsv2 WHERE "NetworkCode" = \'%s\' AND '
                          '"StationCode" = \'%s\' AND soln = \'%s\''
                          % (self.NetworkCode, self.StationCode, self.soln.type))

                # use the default parameters from the objects
                t_ref = self.Linear.p.t_ref

                for i in range(3):
                    x, sigma, index, residuals, fact, w = self.adjust_lsq(self.A, l[i])

                    c.append(x)
                    s.append(sigma)
                    f.append(index)
                    r.append(residuals)
                    factor.append(fact)
                    p.append(w)

                self.C = np.array(c)
                self.S = np.array(s)
                self.F = np.array(f)
                self.R = np.array(r)
                self.factor = np.array(factor)
                self.P = np.array(p)

                # load_parameters to the objects
                self.Linear.load_parameters(self.C, self.S, t_ref)
                self.Jumps.load_parameters(self.C, self.S)
                self.Periodic.load_parameters(params=self.C, sigmas=self.S)

                # save the parameters in each object to the db
                self.save_parameters(cnn)

            # save the covariance between X-Y, Y-Z, X-Z to transform to NEU later
            f = self.F[0] * self.F[1] * self.F[2]

            # load the covariances using the correlations
            self.process_covariance()

            if plotit:
                self.plot()

    def process_covariance(self):

        cov = np.zeros((3, 1))

        # save the covariance between N-E, E-U, N-U to transform to NEU later
        f = self.F[0] * self.F[1] * self.F[2]

        # load the covariances using the correlations
        cov[0] = np.corrcoef(self.R[0][f], self.R[1][f])[0, 1] * self.factor[0] * self.factor[1]
        cov[1] = np.corrcoef(self.R[1][f], self.R[2][f])[0, 1] * self.factor[1] * self.factor[2]
        cov[2] = np.corrcoef(self.R[0][f], self.R[2][f])[0, 1] * self.factor[0] * self.factor[2]

        # build a variance-covariance matrix
        self.covar = np.diag(np.square(self.factor))

        self.covar[0, 1] = cov[0]
        self.covar[1, 0] = cov[0]
        self.covar[2, 1] = cov[1]
        self.covar[1, 2] = cov[1]
        self.covar[0, 2] = cov[2]
        self.covar[2, 0] = cov[2]

        if not self.isPD(self.covar):
            self.covar = self.nearestPD(self.covar)

    def save_parameters(self, cnn):

        # insert linear parameters
        cnn.insert('etmsv2', row=to_postgres(self.Linear.p.toDict()))

        # insert jumps
        for jump in self.Jumps.table:
            cnn.insert('etmsv2', row=to_postgres(jump.p.toDict()))

        # insert periodic params
        cnn.insert('etmsv2', row=to_postgres(self.Periodic.p.toDict()))

        cnn.query('INSERT INTO etmsv2 ("NetworkCode", "StationCode", soln, object, params, hash) VALUES '
                  '(\'%s\', \'%s\', \'ppp\', \'var_factor\', \'%s\', %i)'
                  % (self.NetworkCode, self.StationCode, to_postgres(self.factor), self.hash))

    def plot(self, pngfile=None, t_win=None, residuals=False, plot_missing=True, ecef=False):

        import matplotlib.pyplot as plt

        # definitions
        L = [self.soln.x, self.soln.y, self.soln.z]
        p = []
        m = []
        if ecef:
            labels = ('X [mm]', 'Y [mm]', 'Z [mm]')
        else:
            labels = ('North [mm]', 'East [mm]', 'Up [mm]')

        # get filtered observations
        if self.A is not None:
            filt = self.F[0] * self.F[1] * self.F[2]
        else:
            filt = None

        # find the mean of the observations
        for i in range(3):
            if filt is not None:
                p.append(np.mean(L[i][filt]))
                # calculate the modeled ts and remove mean (in mm)
                m.append((np.dot(self.As, self.C[i]) - p[i]) * 1000)
            else:
                p.append(np.mean(L[i]))

        # get geocentric difference w.r.t. the mean in mm
        L = [(self.soln.x - p[0]) * 1000, (self.soln.y - p[1]) * 1000, (self.soln.z - p[2]) * 1000]

        # rotate to NEU
        if ecef:
            lneu = L
        else:
            lneu = self.rotate_2neu(L)

        # determine the window of the plot, if requested
        if t_win is not None:
            if type(t_win) is tuple:
                # data range, with possibly a final value
                if len(t_win) == 1:
                    t_win = (t_win[0], self.soln.t.max())
            else:
                # approximate a day in fyear
                t_win = (self.soln.t.max() - t_win / 365.25, self.soln.t.max())
        # else:
        #    t_win = (self.soln.t.min(), self.soln.t.max())

        # new behaviour: plots the time series even if there is no ETM fit

        if self.A is not None:

            # create the axis
            f, axis = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(15, 10))  # type: plt.subplots

            # rotate modeled ts
            if ecef:
                mneu = m
                rneu = self.R
                fneu = self.factor * 1000
                stdneu = np.array((np.std(self.R[0][self.F[0]]), np.std(self.R[1][self.F[1]]),
                                   np.std(self.R[2][self.F[2]]))) * 1000
            else:
                mneu = self.rotate_2neu(m)
                # rotate residuals
                rneu = self.rotate_2neu(self.R)
                fneu = np.sqrt(np.diag(self.rotate_sig_cov(covar=self.covar))) * 1000

            # ################# FILTERED PLOT #################

            f.suptitle('Station: %s.%s lat: %.5f lon: %.5f\n'
                       '%s completion: %.2f%%\n%s\n%s\n'
                       'NEU rms [mm]: %5.2f %5.2f %5.2f' %
                       (self.NetworkCode, self.StationCode, self.soln.lat, self.soln.lon, self.soln.type.upper(),
                        self.soln.completion,
                        self.Linear.print_parameters(self.soln.lat, self.soln.lon),
                        self.Periodic.print_parameters(self.soln.lat, self.soln.lon),
                        fneu[0], fneu[1], fneu[2]), fontsize=9, family='monospace')

            table_n, table_e, table_u = self.Jumps.print_parameters(self.soln.lat, self.soln.lon)
            tables = (table_n, table_e, table_u)

            for i, ax in enumerate((axis[0][0], axis[1][0], axis[2][0])):

                # plot filtered time series
                if not residuals:
                    ax.plot(self.soln.t[filt], lneu[i][filt], 'ob', markersize=2)
                    ax.plot(self.soln.ts, mneu[i], 'r')
                    # error bars
                    ax.plot(self.soln.ts, mneu[i] - fneu[i] * LIMIT, 'b', alpha=0.1)
                    ax.plot(self.soln.ts, mneu[i] + fneu[i] * LIMIT, 'b', alpha=0.1)
                    ax.fill_between(self.soln.ts, mneu[i] - fneu[i] * LIMIT, mneu[i] + fneu[i] * LIMIT,
                                    antialiased=True, alpha=0.2)
                else:
                    ax.plot(self.soln.t[filt], rneu[i][filt] * 1000, 'ob', markersize=2)
                    # error bars
                    ax.plot(self.soln.ts, - np.repeat(fneu[i], self.soln.ts.shape[0]) * LIMIT, 'b', alpha=0.1)
                    ax.plot(self.soln.ts, np.repeat(fneu[i], self.soln.ts.shape[0]) * LIMIT, 'b', alpha=0.1)
                    ax.fill_between(self.soln.ts, -fneu[i] * LIMIT, fneu[i] * LIMIT, antialiased=True, alpha=0.2)

                ax.grid(True)

                # labels
                ax.set_ylabel(labels[i])
                p = ax.get_position()
                f.text(0.005, p.y0, tables[i], fontsize=8, family='monospace')

                # window data
                self.set_lims(t_win, plt, ax)

                # plot jumps
                self.plot_jumps(ax)

            # ################# OUTLIERS PLOT #################

            for i, ax in enumerate((axis[0][1], axis[1][1], axis[2][1])):
                ax.plot(self.soln.t, lneu[i], 'oc', markersize=2)
                ax.plot(self.soln.t[filt], lneu[i][filt], 'ob', markersize=2)
                ax.plot(self.soln.ts, mneu[i], 'r')
                # error bars
                ax.plot(self.soln.ts, mneu[i] - fneu[i] * LIMIT, 'b', alpha=0.1)
                ax.plot(self.soln.ts, mneu[i] + fneu[i] * LIMIT, 'b', alpha=0.1)
                ax.fill_between(self.soln.ts, mneu[i] - fneu[i] * LIMIT, mneu[i] + fneu[i] * LIMIT,
                                antialiased=True, alpha=0.2)

                self.set_lims(t_win, plt, ax)

                ax.set_ylabel(labels[i])

                ax.grid(True)

                if plot_missing:
                    self.plot_missing_soln(ax)

            f.subplots_adjust(left=0.16)

        else:

            f, axis = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15, 10))  # type: plt.subplots

            f.suptitle('Station: %s.%s lat: %.5f lon: %.5f'
                       % (self.NetworkCode, self.StationCode, self.soln.lat, self.soln.lon) +
                       '\nNot enough solutions to fit an ETM.', fontsize=9, family='monospace')

            for i, ax in enumerate((axis[0], axis[1], axis[2])):
                ax.plot(self.soln.t, lneu[i], 'ob', markersize=2)

                ax.set_ylabel(labels[i])

                ax.grid(True)

                self.set_lims(t_win, plt, ax)

                self.plot_jumps(ax)

                if plot_missing:
                    self.plot_missing_soln(ax)

        if not pngfile:
            plt.show()
        else:
            plt.savefig(pngfile)
            plt.close()

    def plot_hist(self):

        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab
        from scipy.stats import norm

        L = [self.soln.x, self.soln.y, self.soln.z]

        if self.A is not None:
            residuals = np.sqrt(np.square(L[0]) + np.square(L[1]) + np.square(L[2])) - \
                        np.sqrt(np.square(np.dot(self.A, self.C[0])) + np.square(np.dot(self.A, self.C[1])) +
                                np.square(np.dot(self.A, self.C[2])))

            (mu, sigma) = norm.fit(residuals)

            n, bins, patches = plt.hist(residuals, 200, normed=1, alpha=0.75, facecolor='blue')

            y = mlab.normpdf(bins, mu, sigma)
            plt.plot(bins, y, 'r--', linewidth=2)
            plt.title(r'$\mathrm{Histogram\ of\ residuals (mm):}\ \mu=%.3f,\ \sigma=%.3f$' % (mu * 1000, sigma * 1000))
            plt.grid(True)

            plt.show()

    @staticmethod
    def autoscale_y(ax, margin=0.1):
        """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
        ax -- a matplotlib axes object
        margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

        def get_bottom_top(line):
            xd = line.get_xdata()
            yd = line.get_ydata()
            lo, hi = ax.get_xlim()
            y_displayed = yd[((xd > lo) & (xd < hi))]
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed) - margin * h
            top = np.max(y_displayed) + margin * h
            return bot, top

        lines = ax.get_lines()
        bot, top = np.inf, -np.inf

        for line in lines:
            new_bot, new_top = get_bottom_top(line)
            if new_bot < bot:
                bot = new_bot
            if new_top > top:
                top = new_top
        if bot == top:
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.autoscale(enable=False, axis='y', tight=False)
        else:
            ax.set_ylim(bot, top)

    def set_lims(self, t_win, plt, ax):

        if t_win is None:
            # turn on to adjust the limits, then turn off to plot jumps
            ax.autoscale(enable=True, axis='x', tight=False)
            ax.autoscale(enable=False, axis='x', tight=False)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.autoscale(enable=False, axis='y', tight=False)
        else:
            if t_win[0] == t_win[1]:
                t_win[0] = t_win[0] - 1. / 365.25
                t_win[1] = t_win[1] + 1. / 365.25

            plt.xlim(t_win)
            self.autoscale_y(ax)

    def plot_missing_soln(self, ax):

        # plot missing solutions
        for missing in self.soln.ts_ns:
            ax.plot((missing, missing), ax.get_ylim(), color=(1, 0, 1, 0.2), linewidth=1)

        # plot the position of the outliers
        for blunder in self.soln.ts_blu:
            ax.quiver((blunder, blunder), ax.get_ylim(), (0, 0), (-0.01, 0.01), scale_units='height',
                      units='height', pivot='tip', width=0.008, edgecolors='r')

    def plot_jumps(self, ax):

        for jump in self.Jumps.table:
            if jump.p.jump_type == GENERIC_JUMP and 'Frame Change' not in jump.p.metadata:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), 'b:')

            elif jump.p.jump_type == GENERIC_JUMP and 'Frame Change' in jump.p.metadata:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), ':', color='tab:green')

            elif jump.p.jump_type == CO_SEISMIC_JUMP_DECAY:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), 'r:')

            elif jump.p.jump_type == NO_EFFECT:
                ax.plot((jump.date.fyear, jump.date.fyear), ax.get_ylim(), ':', color='tab:gray')

    def todictionary(self, time_series=False):
        # convert the ETM adjustment into a dirtionary
        # optionally, output the whole time series as well

        # start with the parameters
        etm = dict()
        etm['Network'] = self.NetworkCode
        etm['Station'] = self.StationCode
        etm['lat'] = self.soln.lat[0]
        etm['lon'] = self.soln.lon[0]
        etm['Jumps'] = [to_list(jump.p.toDict()) for jump in self.Jumps.table]

        if self.A is not None:
            etm['Polynomial'] = to_list(self.Linear.p.toDict())

            etm['Periodic'] = to_list(self.Periodic.p.toDict())

            etm['rms'] = {'x': self.factor[0], 'y': self.factor[1], 'z': self.factor[2]}

            etm['xyz_covariance'] = self.covar.tolist()

            etm['neu_covariance'] = self.rotate_sig_cov(covar=self.covar).tolist()

        if time_series:
            ts = dict()
            ts['t'] = self.soln.t.tolist()
            ts['x'] = self.soln.x.tolist()
            ts['y'] = self.soln.y.tolist()
            ts['z'] = self.soln.z.tolist()

            if self.A is not None:
                ts['filter'] = np.logical_and(np.logical_and(self.F[0], self.F[1]), self.F[2]).tolist()
            else:
                ts['filter'] = []

            etm['time_series'] = ts

        return etm

    def get_xyz_s(self, year, doy, jmp=None):
        # this function find the requested epochs and returns an X Y Z and sigmas
        # jmp = 'pre' returns the coordinate immediately before a jump
        # jmp = 'post' returns the coordinate immediately after a jump
        # jmp = None returns either the coordinate before or after, depending on the time of the jump.

        # find this epoch in the t vector
        date = pyDate.Date(year=year, doy=doy)
        window = None

        for jump in self.Jumps.table:
            if jump.date == date and jump.p.jump_type in (GENERIC_JUMP, CO_SEISMIC_JUMP_DECAY):
                if np.sqrt(np.sum(np.square(jump.p.params[:, 0]))) > 0.02:
                    window = jump.date
                    # if no pre or post specified, then determine using the time of the jump
                    if jmp is None:
                        if (jump.date.datetime().hour + jump.date.datetime().minute / 60.0) < 12:
                            jmp = 'post'
                        else:
                            jmp = 'pre'
                    # now use what it was determined
                    if jmp == 'pre':
                        date -= 1
                    else:
                        date += 1

        index = np.where(self.soln.mjd == date.mjd)
        index = index[0]

        s = np.zeros((3, 1))
        x = np.zeros((3, 1))

        dneu = [None, None, None]
        source = '?'
        if index.size:
            l = [self.soln.x, self.soln.y, self.soln.z]

            # found a valid epoch in the t vector
            # now see if this epoch was filtered
            for i in range(3):
                if self.A is not None:
                    if self.F[i][index]:
                        # the coordinate is good
                        if np.abs(self.R[i][index]) >= 0.005:
                            # do not allow uncertainties lower than 5 mm (it's just unrealistic)
                            s[i, 0] = self.R[i][index]
                        else:
                            s[i, 0] = 0.005

                        x[i, 0] = l[i][index]
                        source = 'PPP with ETM solution: good'
                    else:
                        # the coordinate is marked as bad
                        # get the requested epoch from the ETM
                        idt = np.argmin(np.abs(self.soln.ts - date.fyear))

                        Ax = np.dot(self.As[idt, :], self.C[i])
                        x[i, 0] = Ax
                        # Use the deviation from the ETM to estimate the error (which will be multiplied by 2.5 later)
                        s[i, 0] = l[i][index] - Ax
                        source = 'PPP with ETM solution: filtered'
                else:
                    # no ETM (too few points), but we have a solution for the requested day
                    x[i, 0] = l[i][index]
                    dneu[i] = 9.99
                    source = 'PPP no ETM solution'

        else:
            if self.A is not None:
                # the coordinate doesn't exist, get it from the ETM
                idt = np.argmin(np.abs(self.soln.ts - date.fyear))
                As = self.As[idt, :]
                source = 'No PPP solution: ETM'

                for i in range(3):
                    x[i, 0] = np.dot(As, self.C[i])
                    # since there is no way to estimate the error,
                    # use the nominal sigma (which will be multiplied by 2.5 later)
                    s[i, 0] = np.std(self.R[i][self.F[i]])
                    dneu[i] = 9.99
            else:
                # no ETM (too few points), get average
                source = 'No PPP solution, no ETM: mean coordinate'
                for i in range(3):
                    if i == 0:
                        x[i, 0] = np.mean(self.soln.x)
                    elif i == 1:
                        x[i, 0] = np.mean(self.soln.y)
                    else:
                        x[i, 0] = np.mean(self.soln.z)
                    # set the uncertainties in NEU by hand
                    dneu[i] = 9.99

        # crude transformation from XYZ to NEU
        if dneu[0] is None:
            dneu = self.rotate_sig_cov(s)

        if self.A is not None:
            # get the velocity of the site
            if np.sqrt(np.square(self.Linear.p.params[0, 1]) +
                       np.square(self.Linear.p.params[1, 1]) +
                       np.square(self.Linear.p.params[2, 1])) > 0.2:
                # fast moving station! bump up the sigma floor
                sigma_h = 99.9
                sigma_v = 99.9
            else:
                sigma_h = 0.1
                sigma_v = 0.15
        else:
            # no ETM solution! bump up the sigma floor
            sigma_h = 9.99
            sigma_v = 9.99

        oneu = np.zeros(3)
        # apply floor sigmas
        oneu[0] = np.sqrt(np.square(dneu[0]) + np.square(sigma_h))
        oneu[1] = np.sqrt(np.square(dneu[1]) + np.square(sigma_h))
        oneu[2] = np.sqrt(np.square(dneu[2]) + np.square(sigma_v))

        s = np.row_stack((oneu[0], oneu[1], oneu[2]))

        return x, s, window, source

    def rotate_2neu(self, ecef):

        return ct2lg(ecef[0], ecef[1], ecef[2], self.soln.lat, self.soln.lon)

    def rotate_2xyz(self, neu):

        return lg2ct(neu[0], neu[1], neu[2], self.soln.lat, self.soln.lon)

    def rotate_sig_cov(self, sigmas=None, covar=None):

        if sigmas is None and covar is None:
            raise pyETMException('Error in rotate_sig_cov: must provide either simgas or covariance matrix')

        R = rotct2lg(self.soln.lat, self.soln.lon)

        if sigmas is not None:
            # build a covariance matrix based on sigmas
            sd = np.diagflat(np.square(sigmas))

            sd[0, 1] = self.covar[0, 1]
            sd[1, 0] = self.covar[1, 0]
            sd[2, 1] = self.covar[2, 1]
            sd[1, 2] = self.covar[1, 2]
            sd[0, 2] = self.covar[0, 2]
            sd[2, 0] = self.covar[2, 0]

            # check that resulting matrix is PSD:
            if not self.isPD(sd):
                sd = self.nearestPD(sd)

            sneu = np.dot(np.dot(R[:, :, 0], sd), R[:, :, 0].transpose())

            dneu = np.sqrt(np.diag(sneu))

        else:
            # covariance matrix given, assume it is a covariance matrix
            dneu = np.dot(np.dot(R[:, :, 0], covar), R[:, :, 0].transpose())

        return dneu

    def nearestPD(self, A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1

        while not self.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    @staticmethod
    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def load_parameters(self, params, l):

        factor = 1
        index = []
        residuals = []
        p = []

        for param in params:
            par = np.array(param['params'])
            sig = np.array(param['sigmas'])

            if param['object'] == 'polynomial':
                self.Linear.load_parameters(par, sig, param['t_ref'])

            if param['object'] == 'periodic':
                self.Periodic.load_parameters(params=par, sigmas=sig)

            if param['object'] == 'jump':
                for jump in self.Jumps.table:
                    if jump.p.hash == param['hash']:
                        jump.load_parameters(params=par, sigmas=sig)

            if param['object'] == 'var_factor':
                # already a vector in the db
                factor = par

        x = self.Linear.p.params
        s = self.Linear.p.sigmas

        for jump in self.Jumps.table:
            x = np.append(x, jump.p.params, axis=1)
            s = np.append(s, jump.p.sigmas, axis=1)

        x = np.append(x, self.Periodic.p.params, axis=1)
        s = np.append(s, self.Periodic.p.sigmas, axis=1)

        for i in range(3):
            residuals.append(l[i, :] - np.dot(self.A(constrains=False), x[i, :]))

            ss = np.abs(np.divide(residuals[i], factor[i]))
            index.append(ss <= LIMIT)

            f = np.ones((l.shape[1],))

            sw = np.power(10, LIMIT - ss[ss > LIMIT])
            sw[sw < np.finfo(np.float).eps] = np.finfo(np.float).eps
            f[ss > LIMIT] = sw

            p.append(np.square(np.divide(f, factor[i])))

        self.C = x
        self.S = s
        self.F = np.array(index)
        self.R = np.array(residuals)
        self.factor = factor
        self.P = np.array(p)

    def adjust_lsq(self, Ai, Li):

        A = Ai(constrains=True)
        L = Ai.get_l(Li, constrains=True)

        cst_pass = False
        iteration = 0
        factor = 1
        So = 1
        dof = (Ai.shape[0] - Ai.shape[1])
        X1 = chi2.ppf(1 - 0.05 / 2, dof)
        X2 = chi2.ppf(0.05 / 2, dof)

        s = np.array([])
        v = np.array([])
        C = np.array([])

        P = Ai.get_p(constrains=True)

        while not cst_pass and iteration <= 10:

            W = np.sqrt(P)

            Aw = np.multiply(W[:, None], A)
            Lw = np.multiply(W, L)

            C = np.linalg.lstsq(Aw, Lw, rcond=-1)[0]

            v = L - np.dot(A, C)

            # unit variance
            So = np.sqrt(np.dot(v, np.multiply(P, v)) / dof)

            x = np.power(So, 2) * dof

            # obtain the overall uncertainty predicted by lsq
            factor = factor * So

            # calculate the normalized sigmas
            s = np.abs(np.divide(v, factor))

            if x < X2 or x > X1:
                # if it falls in here it's because it didn't pass the Chi2 test
                cst_pass = False

                # reweigh by Mike's method of equal weight until 2 sigma
                f = np.ones((v.shape[0],))
                # f[s > LIMIT] = 1. / (np.power(10, LIMIT - s[s > LIMIT]))
                # do not allow sigmas > 100 m, which is basically not putting
                # the observation in. Otherwise, due to a model problem
                # (missing jump, etc) you end up with very unstable inversions
                # f[f > 500] = 500
                sw = np.power(10, LIMIT - s[s > LIMIT])
                sw[sw < np.finfo(np.float).eps] = np.finfo(np.float).eps
                f[s > LIMIT] = sw

                P = np.square(np.divide(f, factor))
            else:
                cst_pass = True

            iteration += 1

        # make sure there are no values below eps. Otherwise matrix becomes singular
        P[P < np.finfo(np.float).eps] = 1e-6

        # some statistics
        SS = np.linalg.inv(np.dot(A.transpose(), np.multiply(P[:, None], A)))

        sigma = So * np.sqrt(np.diag(SS))

        # mark observations with sigma <= LIMIT
        index = Ai.remove_constrains(s <= LIMIT)

        v = Ai.remove_constrains(v)

        return C, sigma, index, v, factor, P

    @staticmethod
    def chi2inv(chi, df):
        """Return prob(chisq >= chi, with df degrees of
        freedom).

        df must be even.
        """
        assert df & 1 == 0
        # XXX If chi is very large, exp(-m) will underflow to 0.
        m = chi / 2.0
        sum = term = np.exp(-m)
        for i in range(1, df // 2):
            term *= m / i
            sum += term
        # With small chi and large df, accumulated
        # roundoff error, plus error in
        # the platform exp(), can cause this to spill
        # a few ULP above 1.0. For
        # example, chi2P(100, 300) on my box
        # has sum == 1.0 + 2.0**-52 at this
        # point.  Returning a value even a teensy
        # bit over 1.0 is no good.
        return np.min(sum)

    @staticmethod
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))


class PPPETM(ETM):

    def __init__(self, cnn, NetworkCode, StationCode, plotit=False, no_model=False):
        # load all the PPP coordinates available for this station
        # exclude ppp solutions in the exclude table and any solution that is more than 100 meters from the auto coord

        self.ppp_soln = PppSoln(cnn, NetworkCode, StationCode)

        l = np.array([self.ppp_soln.x, self.ppp_soln.y, self.ppp_soln.z])

        ETM.__init__(self, cnn, self.ppp_soln, no_model)

        self.run_adjustment(cnn, l, plotit)


class GamitETM(ETM):

    def __init__(self, cnn, NetworkCode, StationCode, plotit=False,
                 no_model=False, gamit_soln=None, project=None):

        if gamit_soln is None:
            self.polyhedrons = cnn.query_float('SELECT "X", "Y", "Z", "Year", "DOY" FROM gamit_soln '
                                               'WHERE "Project" = \'%s\' AND "NetworkCode" = \'%s\' AND '
                                               '"StationCode" = \'%s\' '
                                               'ORDER BY "Year", "DOY", "NetworkCode", "StationCode"'
                                               % (project, NetworkCode, StationCode))

            self.gamit_soln = GamitSoln(cnn, self.polyhedrons, NetworkCode, StationCode)

        else:
            # load the GAMIT polyhedrons
            self.gamit_soln = gamit_soln

        ETM.__init__(self, cnn, self.gamit_soln, no_model)

        l = np.array([self.gamit_soln.x, self.gamit_soln.y, self.gamit_soln.z])

        self.run_adjustment(cnn, l, plotit)

    def get_residuals_dict(self, project):
        # this function return the values of the ETM ONLY

        dict_o = []
        if self.A is not None:
            # find this epoch in the t vector
            # negative to use with stacker
            px = np.divide(1, np.sqrt(self.P[0]))
            py = np.divide(1, np.sqrt(self.P[1]))
            pz = np.divide(1, np.sqrt(self.P[2]))

            # dict_o += [{'x': -x, 'y': -y, 'z': -z,
            #             'sigmax': sigx, 'sigmay': sigy, 'sigmaz': sigz,
            #             'NetworkCode': net, 'StationCode': stn, 'Year': year, 'DOY': doy, 'Project': prj}
            #             for x, y, z, sigx, sigy, sigz, net, stn, year, doy, prj in
            #                 zip(self.R[0].tolist(), self.R[1].tolist(), self.R[2].tolist(),
            #                     px.tolist(), py.tolist(), pz.tolist(),
            #                     repeat(self.NetworkCode), repeat(self.StationCode),
            #                     [date.year for date in self.gamit_soln.date],
            #                     [date.doy for date in self.gamit_soln.date],
            #                     repeat(project))]
            dict_o += [(net, stn, prj, -x, -y, -z, sigx, sigy, sigz, year, doy)
                       for x, y, z, sigx, sigy, sigz, net, stn, year, doy, prj in
                       zip(self.R[0].tolist(), self.R[1].tolist(), self.R[2].tolist(),
                           px.tolist(), py.tolist(), pz.tolist(),
                           repeat(self.NetworkCode), repeat(self.StationCode),
                           [date.year for date in self.gamit_soln.date],
                           [date.doy for date in self.gamit_soln.date],
                           repeat(project))]
        else:
            raise pyETMException_NoDesignMatrix('No design matrix available for %s.%s' %
                                                (self.NetworkCode, self.StationCode))

        return dict_o


class DailyRep(ETM):

    def __init__(self, cnn, NetworkCode, StationCode, plotit=False,
                 no_model=False, gamit_soln=None, project=None):

        if gamit_soln is None:
            self.polyhedrons = cnn.query_float('SELECT "X", "Y", "Z", "Year", "DOY" FROM gamit_soln '
                                               'WHERE "Project" = \'%s\' AND "NetworkCode" = \'%s\' AND '
                                               '"StationCode" = \'%s\' '
                                               'ORDER BY "Year", "DOY", "NetworkCode", "StationCode"'
                                               % (project, NetworkCode, StationCode))

            self.gamit_soln = GamitSoln(cnn, self.polyhedrons, NetworkCode, StationCode)

        else:
            # load the GAMIT polyhedrons
            self.gamit_soln = gamit_soln

        ETM.__init__(self, cnn, self.gamit_soln, no_model, False, False, False)

        l = np.array([self.gamit_soln.x, self.gamit_soln.y, self.gamit_soln.z])

        self.run_adjustment(cnn, l, plotit)

    def get_residuals_dict(self, project):
        # this function return the values of the ETM ONLY

        dict_o = []
        if self.A is not None:
            # find this epoch in the t vector
            # negative to use with stacker
            px = np.divide(1, np.sqrt(self.P[0]))
            py = np.divide(1, np.sqrt(self.P[1]))
            pz = np.divide(1, np.sqrt(self.P[2]))

            # dict_o += [{'x': -x, 'y': -y, 'z': -z,
            #             'sigmax': sigx, 'sigmay': sigy, 'sigmaz': sigz,
            #             'NetworkCode': net, 'StationCode': stn, 'Year': year, 'DOY': doy, 'Project': prj}
            #             for x, y, z, sigx, sigy, sigz, net, stn, year, doy, prj in
            #                 zip(self.R[0].tolist(), self.R[1].tolist(), self.R[2].tolist(),
            #                     px.tolist(), py.tolist(), pz.tolist(),
            #                     repeat(self.NetworkCode), repeat(self.StationCode),
            #                     [date.year for date in self.gamit_soln.date],
            #                     [date.doy for date in self.gamit_soln.date],
            #                     repeat(project))]
            dict_o += [(net, stn, prj, -x, -y, -z, sigx, sigy, sigz, year, doy)
                       for x, y, z, sigx, sigy, sigz, net, stn, year, doy, prj in
                       zip(self.R[0].tolist(), self.R[1].tolist(), self.R[2].tolist(),
                           px.tolist(), py.tolist(), pz.tolist(),
                           repeat(self.NetworkCode), repeat(self.StationCode),
                           [date.year for date in self.gamit_soln.date],
                           [date.doy for date in self.gamit_soln.date],
                           repeat(project))]
        else:
            raise pyETMException_NoDesignMatrix('No design matrix available for %s.%s' %
                                                (self.NetworkCode, self.StationCode))

        return dict_o


"""
Project: Parallel.PPP
Date: 10/25/17 8:53 AM
Author: Demian D. Gomez

Class to manage (insert, create and query) events produced by the Parallel.PPP wrapper
"""
import inspect
import traceback


class Event(dict):

    def __init__(self, **kwargs):

        dict.__init__(self)

        self['EventDate'] = datetime.datetime.now()
        self['EventType'] = 'info'
        self['NetworkCode'] = None
        self['StationCode'] = None
        self['Year'] = None
        self['DOY'] = None
        self['Description'] = ''
        self['node'] = platform.node()
        self['stack'] = None

        module = inspect.getmodule(inspect.stack()[1][0])
        stack = traceback.extract_stack()[0:-2]

        if module is None:
            self['module'] = inspect.stack()[1][3]  # just get the calling module
        else:
            # self['module'] = module.__name__ + '.' + inspect.stack()[1][3]  # just get the calling module
            self['module'] = module.__name__ + '.' + stack[-1][2]  # just get the calling module

        # initialize the dictionary based on the input
        for key in kwargs:
            if key not in list(self.keys()):
                raise Exception('Provided key not in list of valid fields.')

            arg = kwargs[key]
            self[key] = arg

        if self['EventType'] == 'error':
            self['stack'] = ''.join(traceback.format_stack()[0:-2])  # print the traceback until just before this call
        else:
            self['stack'] = None

    def db_dict(self):
        # remove any invalid chars that can cause problems in the database
        # also, remove the timestamp so that we use the default now() in the databasae
        # out of sync clocks in nodes can cause problems.
        val = self.copy()
        val.pop('EventDate')

        for key in val:
            if type(val[key]) is str:
                val[key] = re.sub(r'[^\x00-\x7f]+', '', val[key])
                val[key] = val[key].replace('\'', '"')
                val[key] = re.sub(r'BASH.*', '', val[key])
                val[key] = re.sub(r'PSQL.*', '', val[key])

        return val

    def __repr__(self):
        return 'pyEvent.Event(' + str(self['Description']) + ')'

    def __str__(self):
        return str(self['Description'])


"""
Project: Parallel.PPP
Date: 9/13/17 6:30 PM
Author: Demian D. Gomez

This module handles the cluster nodes and checks all the necessary dependencies before sending jobs to each node
"""

from functools import partial

import dispy
import dispy.httpd
from tqdm import tqdm

DELAY = 5


def test_node(check_gamit_tables=None, software_sync=()):
    # test node: function that makes sure that all required packages and tools are present in the nodes
    import traceback
    import platform
    import os
    import sys

    def check_tab_file(tabfile, date):

        if os.path.isfile(tabfile):
            # file exists, check contents
            with open(tabfile, 'r') as luntab:
                lines = luntab.readlines()
                tabdate = pyDate.Date(mjd=lines[-1].split()[0])
                if tabdate < date:
                    return ' -- %s: Last entry in %s is %s but processing %s' \
                           % (platform.node(), tabfile, tabdate.yyyyddd(), date.yyyyddd())

        else:
            return ' -- %s: Could not find file %s' % (platform.node(), tabfile)

        return []

    # BEFORE ANYTHING! check the python version
    version = sys.version_info
    if version.major < 3:
        return ' -- %s: Incorrect Python version: %i.%i.%i. Recommended version > 3' \
               % (platform.node(), version.major, version.minor, version.micro)

    # start importing the modules needed
    try:
        import pyRinex
        import dbConnection
        import pyStationInfo
        import pyArchiveStruct
        import pyPPP
        import pyBrdc
        import pyOptions
        import Utils
        import pyOTL
        import shutil
        import datetime
        import time
        import uuid
        import pySp3
        import traceback
        import numpy
        import pyETM
        import pyRunWithRetry
        import pyDate
        import pg
        import dirsync

    except Exception:
        return ' -- %s: Problem found while importing modules:\n%s' % (platform.node(), traceback.format_exc())

    try:
        if len(software_sync) > 0:
            # synchronize directories listed in the src and dst arguments
            from dirsync import sync

            for source_dest in software_sync:
                if isinstance(source_dest, str) and ',' in source_dest:
                    s = source_dest.split(',')[0].strip()
                    d = source_dest.split(',')[1].strip()

                    print('    -- Synchronizing %s -> %s' % (s, d))

                    updated = sync(s, d, 'sync', purge=True, create=True)

                    for f in updated:
                        print('    -- Updated %s' % f)

    except Exception:
        return ' -- %s: Problem found while synchronizing software:\n%s ' % (platform.node(), traceback.format_exc())

    # continue with a test SQL connection
    # make sure that the gnss_data.cfg is present
    try:
        cnn = dbConnection.Cnn('gnss_data.cfg')

        q = cnn.query('SELECT count(*) FROM networks')

        if int(pg.version[0]) < 5:
            return ' -- %s: Incorrect PyGreSQL version!: %s' % (platform.node(), pg.version)

    except Exception:
        return ' -- %s: Problem found while connecting to postgres:\n%s ' % (platform.node(), traceback.format_exc())

    # make sure we can create the production folder
    try:
        test_dir = os.path.join('production/node_test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
    except Exception:
        return ' -- %s: Could not create production folder:\n%s ' % (platform.node(), traceback.format_exc())

    # test
    try:
        Config = pyOptions.ReadOptions('gnss_data.cfg')

        # check that all paths exist and can be reached
        if not os.path.exists(Config.archive_path):
            return ' -- %s: Could not reach archive path %s' % (platform.node(), Config.archive_path)

        if not os.path.exists(Config.repository):
            return ' -- %s: Could not reach repository path %s' % (platform.node(), Config.repository)

        # pick a test date to replace any possible parameters in the config file
        date = pyDate.Date(year=2010, doy=1)
    except Exception:
        return ' -- %s: Problem while reading config file and/or testing archive access:\n%s' \
               % (platform.node(), traceback.format_exc())

    try:
        pyBrdc.GetBrdcOrbits(Config.brdc_path, date, test_dir)
    except Exception:
        return ' -- %s: Problem while testing the broadcast ephemeris archive (%s) access:\n%s' \
               % (platform.node(), Config.brdc_path, traceback.format_exc())

    try:
        pySp3.GetSp3Orbits(Config.sp3_path, date, Config.sp3types, test_dir)
    except Exception:
        return ' -- %s: Problem while testing the sp3 orbits archive (%s) access:\n%s' \
               % (platform.node(), Config.sp3_path, traceback.format_exc())

    # check that all executables and GAMIT bins are in the path
    list_of_prgs = ['crz2rnx', 'crx2rnx', 'rnx2crx', 'rnx2crz', 'RinSum', 'teqc', 'svdiff', 'svpos', 'tform',
                    'sh_rx2apr', 'doy', 'RinEdit', 'sed', 'compress']

    for prg in list_of_prgs:
        with pyRunWithRetry.command('which ' + prg) as run:
            run.run()
            if run.stdout == '':
                return ' -- %s: Could not find path to %s' % (platform.node(), prg)

    # check grdtab and ppp from the config file
    if not os.path.isfile(Config.options['grdtab']):
        return ' -- %s: Could not find grdtab in %s' % (platform.node(), Config.options['grdtab'])

    if not os.path.isfile(Config.options['otlgrid']):
        return ' -- %s: Could not find otlgrid in %s' % (platform.node(), Config.options['otlgrid'])

    if not os.path.isfile(Config.options['ppp_exe']):
        return ' -- %s: Could not find ppp_exe in %s' % (platform.node(), Config.options['ppp_exe'])

    if not os.path.isfile(os.path.join(Config.options['ppp_path'], 'gpsppp.stc')):
        return ' -- %s: Could not find gpsppp.stc in %s' % (platform.node(), Config.options['ppp_path'])

    if not os.path.isfile(os.path.join(Config.options['ppp_path'], 'gpsppp.svb_gps_yrly')):
        return ' -- %s: Could not find gpsppp.svb_gps_yrly in %s' % (platform.node(), Config.options['ppp_path'])

    if not os.path.isfile(os.path.join(Config.options['ppp_path'], 'gpsppp.flt')):
        return ' -- %s: Could not find gpsppp.flt in %s' % (platform.node(), Config.options['ppp_path'])

    if not os.path.isfile(os.path.join(Config.options['ppp_path'], 'gpsppp.stc')):
        return ' -- %s: Could not find gpsppp.stc in %s' % (platform.node(), Config.options['ppp_path'])

    if not os.path.isfile(os.path.join(Config.options['ppp_path'], 'gpsppp.met')):
        return ' -- %s: Could not find gpsppp.met in %s' % (platform.node(), Config.options['ppp_path'])

    for frame in Config.options['frames']:
        if not os.path.isfile(frame['atx']):
            return ' -- %s: Could not find atx in %s' % (platform.node(), frame['atx'])

    if check_gamit_tables is not None:
        # check the gamit tables if not none

        date = check_gamit_tables[0]
        eop = check_gamit_tables[1]
        # TODO: Change this so it's not hardwired into the home directory anymore
        tables = os.path.join(Config.options['gg'], 'tables')

        if not os.path.isdir(Config.options['gg']):
            return ' -- %s: Could not GAMIT installation dir (gg)' % (platform.node())

        if not os.path.isdir(tables):
            return ' -- %s: Could not GAMIT tables dir (gg)' % (platform.node())

        # luntab
        luntab = os.path.join(tables, 'luntab.' + date.yyyy() + '.J2000')

        result = check_tab_file(luntab, date)

        if result:
            return result

        # soltab
        soltab = os.path.join(tables, 'soltab.' + date.yyyy() + '.J2000')

        result = check_tab_file(soltab, date)

        if result:
            return result

        # ut
        ut = os.path.join(tables, 'ut1.' + eop)

        result = check_tab_file(ut, date)

        if result:
            return result

        # leapseconds

        # vmf1

        # pole
        pole = os.path.join(tables, 'pole.' + eop)

        result = check_tab_file(pole, date)

        if result:
            return result

        # fes_cmc consistency

    return ' -- %s: Test passed!' % (platform.node())


def setup(modules):
    """
    function to import modules in the nodes
    :return: 0
    """
    for module in modules:
        module_obj = __import__(module)
        # create a global object containing our module
        globals()[module] = module_obj
        print(' >> Importing module %s' % module)

    return 0


class JobServer:

    def __init__(self, Config, check_gamit_tables=None, run_parallel=True, software_sync=()):
        """
        initialize the jobserver
        :param Config: pyOptions.ReadOptions instance
        :param check_gamit_tables: check or not the tables in GAMIT
        :param run_parallel: override the configuration in gnss_data.cfg
        :param software_sync: list of strings with remote and local paths of software to be synchronized
        """
        self.check_gamit_tables = check_gamit_tables
        self.software_sync = software_sync

        self.nodes = []
        self.result = []
        self.jobs = []
        self.run_parallel = Config.run_parallel if run_parallel else False
        self.verbose = False
        self.close = False

        # vars to store the http_server and the progress bar (if needed)
        self.progress_bar = None
        self.http_server = None
        self.callback = None
        self.function = None
        self.modules = []

        print(" ==== Starting JobServer(dispy) ====")

        # check that the run_parallel option is activated
        if self.run_parallel:
            if Config.options['node_list'] is None:
                # no explicit list, find all
                servers = ['*']
            else:
                # use the provided explicit list of nodes
                if Config.options['node_list'].strip() == '':
                    servers = ['*']
                else:
                    servers = [_f for _f in list(Config.options['node_list'].split(',')) if _f]

            # initialize the cluster
            self.cluster = dispy.JobCluster(test_node, servers, recover_file='pg.dat',
                                            ip_addr=servers)
            # discover the available nodes
            self.cluster.discover_nodes(servers)

            # wait for all nodes
            time.sleep(DELAY)

            stop = False

            for r in self.result:
                if 'Test passed!' not in r:
                    print(r)
                    stop = True

            if stop:
                print(' >> Errors were encountered during initialization. Check messages.')
                # terminate execution if problems were found
                self.cluster.close()
                exit()

            self.cluster.close()
        else:
            print(' >> Parallel processing deactivated by user')
            r = test_node(check_gamit_tables)
            if 'Test passed!' not in r:
                print(r)
                print(' >> Errors were encountered during initialization. Check messages.')
                exit()

    def check_cluster(self, status, node, job):

        if status == dispy.DispyNode.Initialized:
            print(' -- Checking node %s (%i CPUs)...' % (node.name, node.avail_cpus))
            # test node to make sure everything works
            self.cluster.send_file('gnss_data.cfg', node)

            j = self.cluster.submit_node(node, self.check_gamit_tables, self.software_sync)

            self.cluster.wait()

            self.result.append(j)

            self.nodes.append(node)

    def create_cluster(self, function, deps=(), callback=None, progress_bar=None, verbose=False, modules=()):

        self.nodes = []
        self.jobs = []
        self.callback = callback
        self.function = function
        self.verbose = verbose
        self.close = True

        if self.run_parallel:
            self.cluster = dispy.JobCluster(function, self.nodes, list(deps), callback, self.cluster_status,
                                            pulse_interval=60, setup=partial(setup, modules),
                                            loglevel=dispy.logger.CRITICAL, reentrant=True)

            self.http_server = dispy.httpd.DispyHTTPServer(self.cluster, poll_sec=2)

            # wait for all nodes to be created
            time.sleep(DELAY)

        self.progress_bar = progress_bar

    def submit(self, *args):
        """
        function to submit jobs to dispy. If run_parallel == False, the jobs are executed
        :param args:
        :return:
        """
        if self.run_parallel:
            self.jobs.append(self.cluster.submit(*args))
        else:
            # if no parallel was invoked, execute the procedure manually
            if self.callback is not None:
                job = dispy.DispyJob(args, (), ())
                try:
                    job.result = self.function(*args)
                    # TODO: Get the error to go into the events database.
                    if self.progress_bar is not None:
                        self.progress_bar.update()
                except Exception as e:
                    job.exception = e
                self.callback(job)
            else:
                self.function(*args)

    def wait(self):
        """
        wrapped function to wait for cluster execution
        :return: none
        """
        if self.run_parallel:
            tqdm.write(' -- Waiting for jobs to finish...')
            try:
                self.cluster.wait()
                # let the process trigger cluster_status before letting the calling proc close the progress bar
                time.sleep(DELAY)
            except KeyboardInterrupt:
                for job in self.jobs:
                    if job.status in (dispy.DispyJob.Running, dispy.DispyJob.Created):
                        self.cluster.cancel(job)
                self.cluster.shutdown()

    def close_cluster(self):
        if self.run_parallel and self.close:
            tqdm.write('')
            self.http_server.shutdown()
            self.cleanup()

    def cluster_status(self, status, node, job):

        # update the status in the http_server
        self.http_server.cluster_status(self.http_server._clusters[self.cluster.name], status, node, job)

        if status == dispy.DispyNode.Initialized:
            tqdm.write(' -- Node %s initialized with %i CPUs' % (node.name, node.avail_cpus))
            # test node to make sure everything works
            self.cluster.send_file('gnss_data.cfg', node)
            self.nodes.append(node)
            return

        if job is not None:
            if status == dispy.DispyJob.Finished and self.verbose:
                tqdm.write(' -- Job %i finished successfully' % job.id)

            elif status == dispy.DispyJob.Abandoned:
                # always print abandoned jobs
                tqdm.write(' -- Job %04i (%s) was reported as abandoned at node %s -> resubmitting'
                           % (job.id, str(job.args), node.name))

            elif status == dispy.DispyJob.Created and self.verbose:
                tqdm.write(' -- Job %i has been created' % job.id)

            elif status == dispy.DispyJob.Terminated:
                tqdm.write(' -- Job %i has been terminated' % job.id)

            if status in (dispy.DispyJob.Finished, dispy.DispyJob.Terminated) and self.progress_bar is not None:
                self.progress_bar.update()

    def cleanup(self):
        if self.run_parallel and self.close:
            self.cluster.print_status()
            self.cluster.close()
            self.close = False

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self


"""
Project: Parallel.Archive
Date: 3/3/17 11:27 AM
Author: Demian D. Gomez

module for robust least squares operations
"""

from Utils import ct2lg
from scipy.stats import chi2


def robust_lsq(A, P, L, max_iter=10, gcc=True, limit=2.5, lat=0, lon=0):
    # goodness of fit test variable
    cst_pass = False
    # iteration count
    iteration = 0
    # current sigma zero
    So = 1
    # degrees of freedom
    dof = (A.shape[0] - A.shape[1])
    # chi2 limits
    X1 = chi2.ppf(1 - 0.05 / 2, dof)
    X2 = chi2.ppf(0.05 / 2, dof)

    # multiplication factor / sigma estimation
    factor = np.ones(3)

    # lists to store the variables of each component
    nsig = [None, None, None]
    v = [None, None, None]
    C = [None, None, None]

    while not cst_pass and iteration <= max_iter:

        # each iteration fits the three axis: x y z
        for i in range(3):
            W = np.sqrt(P)

            Aw = np.multiply(W[:, None], A)
            Lw = np.multiply(W, L[i])

            # adjust
            C[i] = np.linalg.lstsq(Aw, Lw, rcond=-1)[0]

            v[i] = L[i] - np.dot(A, C[i])

        if not gcc:
            # rotate residuals to NEU
            v[0], v[1], v[2] = rotate_vector(v, lat, lon)

        # unit variance
        So = np.sqrt(np.dot(v, np.multiply(P, v)) / dof)

        x = np.power(So, 2) * dof

        # obtain the overall uncertainty predicted by lsq
        factor[i] = factor[i] * So

        # calculate the normalized sigmas
        nsig[i] = np.abs(np.divide(v[i], factor[i]))

        if x < X2 or x > X1:
            # if it falls in here it's because it didn't pass the Chi2 test
            cst_pass = False

            # reweigh by Mike's method of equal weight until 2 sigma
            f = np.ones((v.shape[0],))
            # f[s > LIMIT] = 1. / (np.power(10, LIMIT - s[s > LIMIT]))
            # do not allow sigmas > 100 m, which is basically not putting
            # the observation in. Otherwise, due to a model problem
            # (missing jump, etc) you end up with very unstable inversions
            # f[f > 500] = 500
            sw = np.power(10, LIMIT - s[s > LIMIT])
            sw[sw < np.finfo(np.float).eps] = np.finfo(np.float).eps
            f[s > LIMIT] = 1. / sw

            P = np.diag(np.divide(1, np.square(factor * f)))
        else:
            cst_pass = True

        iteration += 1

    # make sure there are no values below eps. Otherwise matrix becomes singular
    P[P < np.finfo(np.float).eps] = 1e-6
    # some statistics
    SS = np.linalg.inv(np.dot(np.dot(A.transpose(), P), A))

    sigma = So * np.sqrt(np.diag(SS))

    # mark observations with sigma <= LIMIT
    index = Ai.remove_constrains(s <= LIMIT)

    v = Ai.remove_constrains(v)

    return C, sigma, index, v, factor, np.diag(P)


def rotate_vector(ecef, lat, lon):
    return ct2lg(ecef[0], ecef[1], ecef[2], lat, lon)


"""
Project: Parallel.Archive
Date: 3/21/17 5:36 PM
Author: Demian D. Gomez

Class with all the configuration information necessary to run many of the scripts. It loads the config file (gnss_data.cfg).
"""

import configparser

from Utils import process_date


class ReadOptions:
    def __init__(self, configfile):

        self.options = {'path': None,
                        'repository': None,
                        'parallel': False,
                        'cups': None,
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
                        'gg': None}

        config = configparser.ConfigParser()
        config.readfp(open(configfile))

        # get the archive config
        for iconfig, val in dict(config.items('archive')).items():
            self.options[iconfig] = val

        # get the otl config
        for iconfig, val in dict(config.items('otl')).items():
            self.options[iconfig] = val

        # get the ppp config
        for iconfig, val in dict(config.items('ppp')).items():
            self.options[iconfig] = val

        # frames and dates
        frames = [item.strip() for item in self.options['frames'].split(',')]
        atx = [item.strip() for item in self.options['atx'].split(',')]

        self.Frames = []

        for frame, atx in zip(frames, atx):
            date = process_date(self.options[frame.lower()].split(','))
            self.Frames += [{'name': frame, 'atx': atx, 'dates':
                (Date(year=date[0].year, doy=date[0].doy, hour=0, minute=0, second=0),
                 Date(year=date[1].year, doy=date[1].doy, hour=23, minute=59, second=59))}]

        self.options['frames'] = self.Frames

        self.archive_path = self.options['path']
        self.sp3_path = self.options['sp3']
        self.brdc_path = self.options['brdc']
        self.repository = self.options['repository']
        self.gg = self.options['gg']
        self.repository_data_in = os.path.join(self.repository, 'data_in')
        self.repository_data_in_retry = os.path.join(self.repository, 'data_in_retry')
        self.repository_data_reject = os.path.join(self.repository, 'data_rejected')

        self.sp3types = [self.options['sp3_type_1'], self.options['sp3_type_2'], self.options['sp3_type_3']]

        self.sp3types = [sp3type for sp3type in self.sp3types if sp3type is not None]

        # alternative sp3 types
        self.sp3altrn = [self.options['sp3_altr_1'], self.options['sp3_altr_2'], self.options['sp3_altr_3']]

        self.sp3altrn = [sp3alter for sp3alter in self.sp3altrn if sp3alter is not None]

        if self.options['parallel'] == 'True':
            self.run_parallel = True
        else:
            self.run_parallel = False

        return


"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez

Ocean loading coefficients class. It runs and reads grdtab (from GAMIT).
"""


class pyOTLException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class OceanLoading():

    def __init__(self, StationCode, grdtab, otlgrid, x=None, y=None, z=None):

        self.x = None
        self.y = None
        self.z = None

        self.rootdir = os.path.join('production', 'otl_calc')
        # generate a unique id for this instance
        self.rootdir = os.path.join(self.rootdir, str(uuid.uuid4()))
        self.StationCode = StationCode

        try:
            # create a production folder to analyze the rinex file
            if not os.path.exists(self.rootdir):
                os.makedirs(self.rootdir)
        except Exception as excep:
            # could not create production dir! FATAL
            raise

        # verify of link to otl.grid exists
        if not os.path.isfile(os.path.join(self.rootdir, 'otl.grid')):
            # should be configurable
            try:
                os.symlink(otlgrid, os.path.join(self.rootdir, 'otl.grid'))
            except Exception as e:
                raise pyOTLException(e)

        if not os.path.isfile(grdtab):
            raise pyOTLException('grdtab could not be found at the specified location: ' + grdtab)
        else:
            self.grdtab = grdtab

        if not (x is None and y is None and z is None):
            self.x = x;
            self.y = y;
            self.z = z
        return

    def calculate_otl_coeff(self, x=None, y=None, z=None):

        if not self.x and (x is None or y is None or z is None):
            raise pyOTLException('Cartesian coordinates not initialized and not provided in calculate_otl_coef')
        else:
            if not self.x:
                self.x = x
            if not self.y:
                self.y = y
            if not self.z:
                self.z = z

            cmd = pyRunWithRetry.RunCommand(
                self.grdtab + ' ' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.z) + ' ' + self.StationCode, 5,
                self.rootdir)
            out, err = cmd.run_shell()

            if err or os.path.isfile(os.path.join(self.rootdir, 'GAMIT.fatal')) and not os.path.isfile(
                    os.path.join(self.rootdir, 'harpos.' + self.StationCode)):
                if err:
                    raise pyOTLException('grdtab returned an error: ' + err)
                else:
                    with open(os.path.join(self.rootdir, 'GAMIT.fatal'), 'r') as fileio:
                        raise pyOTLException('grdtab returned an error:\n' + fileio.read())
            else:
                # open otl file
                with open(os.path.join(self.rootdir, 'harpos.' + self.StationCode), 'r') as fileio:
                    return fileio.read()

    def __del__(self):
        if os.path.isfile(os.path.join(self.rootdir, 'GAMIT.status')):
            os.remove(os.path.join(self.rootdir, 'GAMIT.status'))

        if os.path.isfile(os.path.join(self.rootdir, 'GAMIT.fatal')):
            os.remove(os.path.join(self.rootdir, 'GAMIT.fatal'))

        if os.path.isfile(os.path.join(self.rootdir, 'grdtab.out')):
            os.remove(os.path.join(self.rootdir, 'grdtab.out'))

        if os.path.isfile(os.path.join(self.rootdir, 'harpos.' + self.StationCode)):
            os.remove(os.path.join(self.rootdir, 'harpos.' + self.StationCode))

        if os.path.isfile(os.path.join(self.rootdir, 'otl.grid')):
            os.remove(os.path.join(self.rootdir, 'otl.grid'))

        if os.path.isfile(os.path.join(self.rootdir, 'ufile.' + self.StationCode)):
            os.remove(os.path.join(self.rootdir, 'ufile.' + self.StationCode))

        if os.path.isdir(self.rootdir):
            os.rmdir(self.rootdir)


"""
Project: Parallel.Archive
Date: 2/25/17 7:15 PM
Author: Demian D. Gomez
"""


class ParseAntexFile():

    def __init__(self, filename):

        try:
            with open(filename, 'r') as fileio:
                antex = fileio.readlines()
        except:
            raise

        self.Antennas = []
        self.Radomes = []

        for line in antex:
            if 'TYPE / SERIAL NO' in line and len(line.split()) <= 6:
                self.Antennas.append(line.split()[0])
                self.Radomes.append(line.split()[1])

        # make a unique list
        self.Antennas = list(set(self.Antennas))
        self.Radomes = list(set(self.Radomes))


"""
Project: Parallel.PPP
Date: 2/21/17 3:34 PM
Author: Demian D. Gomez

Python wrapper for PPP. It runs the NRCAN PPP and loads the information from the summary file. Can be used without a
database connection, except for PPPSpatialCheck

"""
from math import isnan

import pyClk
import pyEOP
import pyRinex
import pySp3
from Utils import determine_frame
from Utils import lg2ct
from pyDate import Date


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


class pyRunPPPException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class pyRunPPPExceptionCoordConflict(pyRunPPPException):
    pass


class pyRunPPPExceptionTooFewAcceptedObs(pyRunPPPException):
    pass


class pyRunPPPExceptionNaN(pyRunPPPException):
    pass


class pyRunPPPExceptionZeroProcEpochs(pyRunPPPException):
    pass


class pyRunPPPExceptionEOPError(pyRunPPPException):
    pass


class PPPSpatialCheck:

    def __init__(self, lat=None, lon=None, h=None, epoch=None):

        self.lat = lat
        self.lon = lon
        self.h = h
        self.epoch = epoch

        return

    def verify_spatial_coherence(self, cnn, StationCode, search_in_new=False):
        # checks the spatial coherence of the resulting coordinate
        # will not make any decisions, just output the candidates
        # if ambiguities are found, the rinex StationCode is used to solve them
        # third output arg is used to return a list with the closest station/s if no match is found
        # or if we had to disambiguate using station name
        # DDG Mar 21 2018: Added the velocity of the station to account for fast moving stations (on ice)
        # the logic is as follows:
        # 1) if etm data is available, then use it to bring the coordinate to self.epoch
        # 2) if no etm parameters are available, default to the coordinate reported in the stations table

        if not search_in_new:
            where_clause = 'WHERE "NetworkCode" not like \'?%%\''
        else:
            where_clause = ''

        # start by reducing the number of stations filtering everything beyond 100 km from the point of interest
        # rs = cnn.query("""
        #     SELECT * FROM
        #     (SELECT *, 2*asin(sqrt(sin((radians(%.8f)-radians(lat))/2)^2 + cos(radians(lat)) * cos(radians(%.8f)) *
        #     sin((radians(%.8f)-radians(lon))/2)^2))*6371000 AS distance
        #     FROM stations %s) as DD
        #     WHERE distance <= %f
        #     """ % (self.lat[0], self.lat[0], self.lon[0], where_clause, 1e3))  # DO NOT RETURN RESULTS
        #     WITH NetworkCode = '?%'

        rs = cnn.query("""
            SELECT st1."NetworkCode", st1."StationCode", st1."StationName", st1."DateStart", st1."DateEnd",
             st1."auto_x", st1."auto_y", st1."auto_z", st1."Harpos_coeff_otl", st1."lat", st1."lon", st1."height",
             st1."max_dist", st1."dome", st1.distance FROM
            (SELECT *, 2*asin(sqrt(sin((radians(%.8f)-radians(lat))/2)^2 + cos(radians(lat)) * 
            cos(radians(%.8f)) * sin((radians(%.8f)-radians(lon))/2)^2))*6371000 AS distance
            FROM stations %s) as st1 left join stations as st2 ON 
                st1."StationCode" = st2."StationCode" and
                st1."NetworkCode" = st2."NetworkCode" and
                st1.distance < coalesce(st2.max_dist, 20)
                WHERE st2."NetworkCode" is not NULL
            """ % (self.lat[0], self.lat[0], self.lon[0], where_clause))  # DO NOT RETURN RESULTS NetworkCode = '?%'

        stn_match = rs.dictresult()

        # using the list of coordinates, check if StationCode exists in the list
        if len(stn_match) == 0:
            # no match, find closest station
            # get the closest station and distance in km to help the caller function
            rs = cnn.query("""
                SELECT * FROM
                    (SELECT *, 2*asin(sqrt(sin((radians(%.8f)-radians(lat))/2)^2 + cos(radians(lat)) * 
                    cos(radians(%.8f)) * sin((radians(%.8f)-radians(lon))/2)^2))*6371000 AS distance
                        FROM stations %s) as DD ORDER BY distance
                """ % (self.lat[0], self.lat[0], self.lon[0], where_clause))

            stn = rs.dictresult()

            return False, [], stn

        if len(stn_match) == 1 and stn_match[0]['StationCode'] == StationCode:
            # one match, same name (return a dictionary)
            return True, stn_match, []

        if len(stn_match) == 1 and stn_match[0]['StationCode'] != StationCode:
            # one match, not the same name (return a list, not a dictionary)
            return False, stn_match, []

        if len(stn_match) > 1:
            # more than one match, same name
            # this is most likely a station that got moved a few meters and renamed
            # or a station that just got renamed.
            # disambiguation might be possible using the name of the station
            min_stn = [stni for stni in stn_match if stni['StationCode'] == StationCode]

            if len(min_stn) > 0:
                # the minimum distance if to a station with same name, we are good:
                # does the name match the closest station to this solution? yes
                return True, min_stn, []
            else:
                return False, stn_match, []


class RunPPP(PPPSpatialCheck):
    def __init__(self, rinexobj, otl_coeff, options, sp3types, sp3altrn, antenna_height, strict=True, apply_met=True,
                 kinematic=False, clock_interpolation=False, hash=0, erase=True, decimate=True):

        assert isinstance(rinexobj, pyRinex.ReadRinex)

        PPPSpatialCheck.__init__(self)

        self.rinex = rinexobj
        self.epoch = rinexobj.date
        self.antH = antenna_height
        self.ppp_path = options['ppp_path']
        self.ppp = options['ppp_exe']
        self.options = options
        self.kinematic = kinematic

        self.ppp_version = None

        self.file_summary = None
        self.proc_parameters = None
        self.observation_session = None
        self.coordinate_estimate = None

        # DDG: do not allow clock interpolation before May 1 2001
        self.clock_interpolation = clock_interpolation if rinexobj.date > Date(year=2001, month=5, day=1) else False

        self.frame = None
        self.atx = None
        self.x = None
        self.y = None
        self.z = None
        self.lat = None
        self.lon = None
        self.h = None
        self.sigmax = None
        self.sigmay = None
        self.sigmaz = None
        self.sigmaxy = None
        self.sigmaxz = None
        self.sigmayz = None
        self.hash = hash

        self.processed_obs = None
        self.rejected_obs = None

        self.orbit_type = None
        self.orbits1 = None
        self.orbits2 = None
        self.clocks1 = None
        self.clocks2 = None
        self.eop_file = None
        self.sp3altrn = sp3altrn
        self.sp3types = sp3types
        self.otl_coeff = otl_coeff
        self.strict = strict
        self.apply_met = apply_met
        self.erase = erase
        self.out = ''
        self.summary = ''
        self.pos = ''

        self.rootdir = os.path.join('production', 'ppp')

        fieldnames = ['NetworkCode', 'StationCode', 'X', 'Y', 'Z', 'Year', 'DOY', 'ReferenceFrame', 'sigmax', 'sigmay',
                      'sigmaz', 'sigmaxy', 'sigmaxz', 'sigmayz', 'hash']

        self.record = dict.fromkeys(fieldnames)

        # determine the atx to use
        self.frame, self.atx = determine_frame(self.options['frames'], self.epoch)

        if os.path.isfile(self.rinex.rinex_path):

            # generate a unique id for this instance
            self.rootdir = os.path.join(self.rootdir, str(uuid.uuid4()))

            try:
                # create a production folder to analyze the rinex file
                if not os.path.exists(self.rootdir):
                    os.makedirs(self.rootdir)
                    os.makedirs(os.path.join(self.rootdir, 'orbits'))
            except Exception:
                # could not create production dir! FATAL
                raise

            try:
                self.get_orbits(self.sp3types)

            except (pySp3.pySp3Exception, pyClk.pyClkException, pyEOP.pyEOPException):

                if sp3altrn:
                    self.get_orbits(self.sp3altrn)
                else:
                    raise

            self.write_otl()
            self.copyfiles()
            self.config_session()

            # make a local copy of the rinex file
            # decimate the rinex file if the interval is < 15 sec.
            # DDG: only decimate when told by caller
            if self.rinex.interval < 15 and decimate:
                self.rinex.decimate(30)

            copyfile(self.rinex.rinex_path, os.path.join(self.rootdir, self.rinex.rinex))

        else:
            raise pyRunPPPException('The file ' + self.rinex.rinex_path + ' could not be found. PPP was not executed.')

        return

    def copyfiles(self):
        # prepare all the files required to run PPP
        if self.apply_met:
            copyfile(os.path.join(self.ppp_path, 'gpsppp.met'), os.path.join(self.rootdir, 'gpsppp.met'))

        copyfile(os.path.join(self.ppp_path, 'gpsppp.stc'), os.path.join(self.rootdir, 'gpsppp.stc'))
        copyfile(os.path.join(self.ppp_path, 'gpsppp.svb_gnss_yrly'),
                 os.path.join(self.rootdir, 'gpsppp.svb_gnss_yrly'))
        copyfile(os.path.join(self.ppp_path, 'gpsppp.flt'), os.path.join(self.rootdir, 'gpsppp.flt'))
        copyfile(os.path.join(self.ppp_path, 'gpsppp.stc'), os.path.join(self.rootdir, 'gpsppp.stc'))
        copyfile(os.path.join(self.ppp_path, 'gpsppp.trf'), os.path.join(self.rootdir, 'gpsppp.trf'))
        copyfile(os.path.join(self.atx), os.path.join(self.rootdir, os.path.basename(self.atx)))

        return

    def write_otl(self):

        otl_file = open(os.path.join(self.rootdir, self.rinex.StationCode + '.olc'), 'w')
        otl_file.write(self.otl_coeff)
        otl_file.close()

        return

    def config_session(self):

        options = self.options

        # create the def file
        def_file = open(os.path.join(self.rootdir, 'gpsppp.def'), 'w')

        def_file_cont = ("'LNG' 'ENGLISH'\n"
                         "'TRF' 'gpsppp.trf'\n"
                         "'SVB' 'gpsppp.svb_gnss_yrly'\n"
                         "'PCV' '%s'\n"
                         "'FLT' 'gpsppp.flt'\n"
                         "'OLC' '%s.olc'\n"
                         "'MET' 'gpsppp.met'\n"
                         "'ERP' '%s'\n"
                         "'GSD' '%s'\n"
                         "'GSD' '%s'\n"
                         % (os.path.basename(self.atx),
                            self.rinex.StationCode,
                            self.eop_file,
                            options['institution'],
                            options['info']))

        def_file.write(def_file_cont)
        def_file.close()

        cmd_file = open(os.path.join(self.rootdir, 'commands.cmd'), 'w')

        cmd_file_cont = ("' UT DAYS OBSERVED                      (1-45)'               1\n"
                         "' USER DYNAMICS         (1=STATIC,2=KINEMATIC)'               %s\n"
                         "' OBSERVATION TO PROCESS         (1=COD,2=C&P)'               2\n"
                         "' FREQUENCY TO PROCESS        (1=L1,2=L2,3=L3)'               3\n"
                         "' SATELLITE EPHEMERIS INPUT     (1=BRD ,2=SP3)'               2\n"
                         "' SATELLITE PRODUCT (1=NO,2=Prc,3=RTCA,4=RTCM)'               2\n"
                         "' SATELLITE CLOCK INTERPOLATION   (1=NO,2=YES)'               %s\n"
                         "' IONOSPHERIC GRID INPUT          (1=NO,2=YES)'               1\n"
                         "' SOLVE STATION COORDINATES       (1=NO,2=YES)'               2\n"
                         "' SOLVE TROP. (1=NO,2-5=RW MM/HR) (+100=grad) '             105\n"
                         "' BACKWARD SUBSTITUTION           (1=NO,2=YES)'               1\n"
                         "' REFERENCE SYSTEM            (1=NAD83,2=ITRF)'               2\n"
                         "' COORDINATE SYSTEM(1=ELLIPSOIDAL,2=CARTESIAN)'               2\n"
                         "' A-PRIORI PSEUDORANGE SIGMA               (m)'           2.000\n"
                         "' A-PRIORI CARRIER PHASE SIGMA             (m)'           0.015\n"
                         "' LATITUDE  (ddmmss.sss,+N) or ECEF X      (m)'          0.0000\n"
                         "' LONGITUDE (ddmmss.sss,+E) or ECEF Y      (m)'          0.0000\n"
                         "' HEIGHT (m)                or ECEF Z      (m)'          0.0000\n"
                         "' ANTENNA HEIGHT                           (m)'          %6.4f\n"
                         "' CUTOFF ELEVATION                       (deg)'          10.000\n"
                         "' GDOP CUTOFF                                 '          20.000\n"
                         % ('1' if not self.kinematic else '2', '1'
                if not self.clock_interpolation else '2', self.antH))

        cmd_file.write(cmd_file_cont)

        cmd_file.close()

        inp_file = open(os.path.join(self.rootdir, 'input.inp'), 'w')

        inp_file_cont = ("%s\n"
                         "commands.cmd\n"
                         "0 0\n"
                         "0 0\n"
                         "%s\n"
                         "%s\n"
                         "%s\n"
                         "%s\n"
                         % (self.rinex.rinex,
                            self.orbits1.sp3_filename,
                            self.clocks1.clk_filename,
                            self.orbits2.sp3_filename,
                            self.clocks2.clk_filename))

        inp_file.write(inp_file_cont)

        inp_file.close()

        return

    def get_orbits(self, type):

        options = self.options

        orbits1 = pySp3.GetSp3Orbits(options['sp3'], self.rinex.date, type, self.rootdir, True)
        orbits2 = pySp3.GetSp3Orbits(options['sp3'], self.rinex.date + 1, type,
                                     self.rootdir, True)

        clocks1 = pyClk.GetClkFile(options['sp3'], self.rinex.date, type, self.rootdir, True)
        clocks2 = pyClk.GetClkFile(options['sp3'], self.rinex.date + 1, type,
                                   self.rootdir, True)

        try:
            eop_file = pyEOP.GetEOP(options['sp3'], self.rinex.date, type, self.rootdir)
            eop_file = eop_file.eop_filename
        except pyEOP.pyEOPException:
            # no eop, continue with out one
            eop_file = 'dummy.eop'

        self.orbits1 = orbits1
        self.orbits2 = orbits2
        self.clocks1 = clocks1
        self.clocks2 = clocks2
        self.eop_file = eop_file
        # get the type of orbit
        self.orbit_type = orbits1.type

    def get_text(self, summary, start, end):
        copy = False

        if type(summary) is str:
            summary = summary.split('\n')

        out = []
        for line in summary:
            if start in line.strip():
                copy = True
            elif end in line.strip():
                copy = False
            elif copy:
                out += [line]

        return '\n'.join(out)

    @staticmethod
    def get_xyz(section):

        x = re.findall(r'X\s\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0][1]
        y = re.findall(r'Y\s\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0][1]
        z = re.findall(r'Z\s\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0][1]

        if '*' not in x and '*' not in y and '*' not in z:
            x = float(x)
            y = float(y)
            z = float(z)
        else:
            raise pyRunPPPExceptionNaN('One or more coordinate is NaN')

        if isnan(x) or isnan(y) or isnan(z):
            raise pyRunPPPExceptionNaN('One or more coordinate is NaN')

        return x, y, z

    @staticmethod
    def get_sigmas(section, kinematic):

        if kinematic:

            sx = re.findall(r'X\s\(m\)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]
            sy = re.findall(r'Y\s\(m\)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]
            sz = re.findall(r'Z\s\(m\)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]

            if '*' not in sx and '*' not in sy and '*' not in sz:
                sx = float(sx)
                sy = float(sy)
                sz = float(sz)
                sxy = 0.0
                sxz = 0.0
                syz = 0.0
            else:
                raise pyRunPPPExceptionNaN('One or more sigma is NaN')

        else:
            sx, sxy, sxz = re.findall(r'X\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)'
                                      r'\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]
            sy, syz = re.findall(r'Y\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]
            sz = re.findall(r'Z\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]

            if '*' in sx or '*' in sy or '*' in sz or '*' in sxy or '*' in sxz or '*' in syz:
                raise pyRunPPPExceptionNaN('Sigmas are NaN')
            else:
                sx = float(sx)
                sy = float(sy)
                sz = float(sz)
                sxy = float(sxy)
                sxz = float(sxz)
                syz = float(syz)

        if isnan(sx) or isnan(sy) or isnan(sz) or isnan(sxy) or isnan(sxz) or isnan(syz):
            raise pyRunPPPExceptionNaN('Sigmas are NaN')

        return sx, sy, sz, sxy, sxz, syz

    def get_pr_observations(self, section, kinematic):

        if self.ppp_version == '1.05':
            processed = re.findall(r'Number of epochs processed\s+\:\s+(\d+)', section)[0]
        else:
            processed = re.findall(r'Number of epochs processed \(%fix\)\s+\:\s+(\d+)', section)[0]

        if kinematic:
            rejected = re.findall(r'Number of epochs rejected\s+\:\s+(\d+)', section)

            if len(rejected) > 0:
                rejected = int(rejected[0])
            else:
                rejected = 0
        else:
            # processed = re.findall('Number of observations processed\s+\:\s+(\d+)', section)[0]

            rejected = re.findall(r'Number of observations rejected\s+\:\s+(\d+)', section)

            if len(rejected) > 0:
                rejected = int(rejected[0])
            else:
                rejected = 0

        return int(processed), int(rejected)

    @staticmethod
    def check_phase_center(section):

        if len(re.findall(r'Antenna phase center.+NOT AVAILABLE', section)) > 0:
            return False
        else:
            return True

    @staticmethod
    def check_otl(section):

        if len(re.findall(r'Ocean loading coefficients.+NOT FOUND', section)) > 0:
            return False
        else:
            return True

    @staticmethod
    def check_eop(section):
        pole = re.findall(r'Pole X\s+.\s+(-?\d+\.\d+|[nN]a[nN])\s+(-?\d+\.\d+|[nN]a[nN])', section)
        if len(pole) > 0:
            if type(pole[0]) is tuple and 'nan' not in pole[0][0].lower():
                return True
            else:
                return False
        else:
            return True

    @staticmethod
    def get_frame(section):
        return re.findall(r'\s+ITRF\s\((\s*\w+\s*)\)', section)[0].strip()

    def parse_summary(self):

        self.summary = ''.join(self.out)

        self.ppp_version = re.findall(r'.*Version\s+(\d.\d+)\/', self.summary)

        if len(self.ppp_version) == 0:
            self.ppp_version = re.findall(r'.*CSRS-PPP ver.\s+(\d.\d+)\/', self.summary)[0]
        else:
            self.ppp_version = self.ppp_version[0]

        self.file_summary = self.get_text(self.summary, 'SECTION 1.', 'SECTION 2.')
        self.proc_parameters = self.get_text(self.summary, 'SECTION 2. ', ' SECTION 3. ')
        self.observation_session = self.get_text(self.summary,
                                                 '3.2 Observation Session', '3.3 Coordinate estimates')
        self.coordinate_estimate = self.get_text(self.summary,
                                                 '3.3 Coordinate estimates', '3.4 Coordinate differences ITRF')

        if self.strict and not self.check_phase_center(self.proc_parameters):
            raise pyRunPPPException(
                'Error while running PPP: could not find the antenna and radome in antex file. '
                'Check RINEX header for formatting issues in the ANT # / TYPE field. RINEX header follows:\n' + ''.join(
                    self.rinex.get_header()))

        if self.strict and not self.check_otl(self.proc_parameters):
            raise pyRunPPPException(
                'Error while running PPP: could not find the OTL coefficients. '
                'Check RINEX header for formatting issues in the APPROX ANT POSITION field. If APR is too far from OTL '
                'coordinates (declared in the HARPOS or BLQ format) NRCAN will reject the coefficients. '
                'OTL coefficients record follows:\n' + self.otl_coeff)

        if not self.check_eop(self.file_summary):
            raise pyRunPPPExceptionEOPError('EOP returned NaN in Pole XYZ.')

        # parse rejected and accepted observations
        self.processed_obs, self.rejected_obs = self.get_pr_observations(self.observation_session, self.kinematic)

        if self.processed_obs == 0:
            raise pyRunPPPExceptionZeroProcEpochs('PPP returned zero processed epochs')

        # if self.strict and (self.processed_obs == 0 or self.rejected_obs > 0.95 * self.processed_obs):
        #    raise pyRunPPPExceptionTooFewAcceptedObs('The processed observations (' + str(self.processed_obs) +
        #                                             ') is zero or more than 95% of the observations were rejected (' +
        #                                             str(self.rejected_obs) + ')')

        # FRAME now comes from the startup process, where the function Utils.determine_frame is called
        # self.frame = self.get_frame(self.coordinate_estimate)

        self.x, self.y, self.z = self.get_xyz(self.coordinate_estimate)
        self.lat, self.lon, self.h = ecef2lla([self.x, self.y, self.z])

        self.sigmax, self.sigmay, self.sigmaz, \
        self.sigmaxy, self.sigmaxz, self.sigmayz = self.get_sigmas(self.coordinate_estimate, self.kinematic)

        # not implemented in PPP: apply NE offset if is NOT zero
        if self.rinex.antOffsetN != 0.0 or self.rinex.antOffsetE != 0.0:
            dx, dy, dz = lg2ct(numpy.array(self.rinex.antOffsetN), numpy.array(self.rinex.antOffsetE),
                               numpy.array([0]), self.lat, self.lon)
            # reduce coordinates
            self.x -= dx[0]
            self.y -= dy[0]
            self.z -= dz[0]
            self.lat, self.lon, self.h = ecef2lla([self.x, self.y, self.z])

    def __exec_ppp__(self, raise_error=True):

        try:
            # DDG: handle the error found in PPP (happens every now and then)
            # Fortran runtime error: End of file
            for i in range(2):
                cmd = pyRunWithRetry.RunCommand(self.ppp, 60, self.rootdir, 'input.inp')
                out, err = cmd.run_shell()

                if '*END - NORMAL COMPLETION' not in out:

                    if 'Fortran runtime error: End of file' in err and i == 0:
                        # error detected, try again!
                        continue

                    msg = 'PPP ended abnormally for ' + self.rinex.rinex_path + ':\n' + err + '\n' + out
                    if raise_error:
                        raise pyRunPPPException(msg)
                    else:
                        return False, msg
                else:
                    f = open(os.path.join(self.rootdir, self.rinex.rinex[:-3] + 'sum'), 'r')
                    self.out = f.readlines()
                    f.close()

                    f = open(os.path.join(self.rootdir, self.rinex.rinex[:-3] + 'pos'), 'r')
                    self.pos = f.readlines()
                    f.close()
                    break

        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            msg = str(e)
            if raise_error:
                raise pyRunPPPException(e)
            else:
                return False, msg
        except IOError as e:
            raise pyRunPPPException(e)

        return True, ''

    def exec_ppp(self):

        while True:
            # execute PPP but do not raise an error if timed out
            result, message = self.__exec_ppp__(False)

            if result:
                try:
                    self.parse_summary()
                    break

                except pyRunPPPExceptionEOPError:
                    # problem with EOP!
                    if self.eop_file != 'dummy.eop':
                        self.eop_file = 'dummy.eop'
                        self.config_session()
                    else:
                        raise

                except (pyRunPPPExceptionNaN, pyRunPPPExceptionTooFewAcceptedObs, pyRunPPPExceptionZeroProcEpochs):
                    # Nan in the result
                    if not self.kinematic:
                        # first retry, turn to kinematic mode
                        self.kinematic = True
                        self.config_session()
                    elif self.kinematic and self.rinex.date.fyear >= 2001.33287 and not self.clock_interpolation:
                        # date has to be > 2001 May 1 (SA deactivation date)
                        self.clock_interpolation = True
                        self.config_session()
                    elif self.kinematic and self.sp3altrn and self.orbit_type not in self.sp3altrn:
                        # second retry, kinematic and alternative orbits (if exist)
                        self.get_orbits(self.sp3altrn)
                        self.config_session()
                    else:
                        # it didn't work in kinematic mode either! raise error
                        raise
            else:
                # maybe a bad orbit, fall back to alternative
                if self.sp3altrn and self.orbit_type not in self.sp3altrn:
                    self.get_orbits(self.sp3altrn)
                    self.config_session()
                else:
                    raise pyRunPPPException(message)

        self.load_record()

        return

    def load_record(self):

        self.record['NetworkCode'] = self.rinex.NetworkCode
        self.record['StationCode'] = self.rinex.StationCode
        self.record['X'] = self.x
        self.record['Y'] = self.y
        self.record['Z'] = self.z
        self.record['Year'] = self.rinex.date.year
        self.record['DOY'] = self.rinex.date.doy
        self.record['ReferenceFrame'] = self.frame
        self.record['sigmax'] = self.sigmax
        self.record['sigmay'] = self.sigmay
        self.record['sigmaz'] = self.sigmaz
        self.record['sigmaxy'] = self.sigmaxy
        self.record['sigmaxz'] = self.sigmaxz
        self.record['sigmayz'] = self.sigmayz
        self.record['hash'] = self.hash

        return

    def cleanup(self):
        if os.path.isdir(self.rootdir) and self.erase:
            # remove all the directory contents
            rmtree(self.rootdir)

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self


"""
Project:
Date: 2/23/17 10:12 AM
Author: Demian D. Gomez
"""


class pyProductsException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class pyProductsExceptionUnreasonableDate(pyProductsException):
    pass


class OrbitalProduct(object):

    def __init__(self, archive, date, filename, copyto):

        if date.gpsWeek < 0 or date > pyDate.Date(datetime=datetime.now()):
            # do not allow negative weeks or future orbit downloads!
            raise pyProductsExceptionUnreasonableDate('Orbit requested for an unreasonable date: week '
                                                      + str(date.gpsWeek) + ' day ' + str(date.gpsWeekDay) +
                                                      ' (' + date.yyyyddd() + ')')

        archive = archive.replace('$year', str(date.year))
        archive = archive.replace('$doy', str(date.doy).zfill(3))
        archive = archive.replace('$gpsweek', str(date.gpsWeek).zfill(4))
        archive = archive.replace('$gpswkday', str(date.gpsWeekDay))

        self.archive = archive
        self.path = None
        self.filename = filename

        # try both zipped and unzipped n files
        archive_file_path = os.path.join(archive, self.filename)

        if os.path.isfile(archive_file_path):
            try:
                copyfile(archive_file_path, os.path.join(copyto, self.filename))
                self.file_path = os.path.join(copyto, self.filename)
            except Exception:
                raise
        else:
            ext = None
            if os.path.isfile(archive_file_path + '.Z'):
                ext = '.Z'
            elif os.path.isfile(archive_file_path + '.gz'):
                ext = '.gz'
            elif os.path.isfile(archive_file_path + '.zip'):
                ext = '.zip'

            if ext is not None:
                copyfile(archive_file_path + ext, os.path.join(copyto, self.filename + ext))
                self.file_path = os.path.join(copyto, self.filename)

                cmd = pyRunWithRetry.RunCommand('gunzip -f ' + self.file_path + ext, 15)
                try:
                    cmd.run_shell()
                except Exception:
                    raise
            else:
                raise pyProductsException('Could not find the archive file for ' + self.filename)


"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez
"""

from shutil import copy

from Utils import ecef2lla
from pyEvents import Event

TYPE_CRINEZ = 0
TYPE_RINEX = 1
TYPE_RINEZ = 2
TYPE_CRINEX = 3


def check_year(year):
    # to check for wrong dates in RinSum

    if int(year) - 1900 < 80 and int(year) >= 1900:
        year = int(year) - 1900 + 2000

    elif int(year) < 1900 and int(year) >= 80:
        year = int(year) + 1900

    elif int(year) < 1900 and int(year) < 80:
        year = int(year) + 2000

    return year


def create_unzip_script(run_file_path):
    # temporary script to uncompress o.Z files
    # requested by RS issue #13
    try:
        run_file = open(run_file_path, 'w')
    except (OSError, IOError):
        raise Exception('could not open file ' + run_file_path)

    contents = """#!/bin/csh -f
        # set default mode
        set out_current = 0
        set del_input = 0
        unset verbose
        set ovrewrite = 0

        set PROGRAM = CRX2RNX

        unset noclobber

        # check options
        foreach var ($argv[*])
        switch ($var)
        case '-c':
        set out_current = 1
        shift; breaksw
        case '-d':
        set del_input = 1
        shift; breaksw
        case '-f':
        set ovrewrite = 1
        shift; breaksw
        case '-v':
        set verbose = 1
        shift; breaksw
        default:
        break
        endsw
        end


        # process files
        foreach file ($argv[*])

        # make command to be issued and name of output file
        set file2 = $file
        set ext   = $file:e
        if ( $out_current ) set file2 = $file2:t
        if( $ext == Z || $ext == gz ) set file2 = $file2:r
        if( $file2 =~ *.??[oO] ) then
        set file2 = `echo $file2 | sed -e 's/d$/o/' -e 's/D$/O/' `
        else if( $file2 !~ *.??[oOnNgGlLpPhHbBmMcC] || ! ($ext == Z || $ext == gz) ) then
        # This is not a compressed RINEX file ... skip it
        continue
        endif
        set file_save = $file2

        # check if the output file is preexisting
        if ( -e "$file_save" && ! $ovrewrite ) then
        echo "The file $file_save already exists. Overwrite?(y/n,default:n)"
        if ( $< !~ [yY] ) continue
        endif

        # issue the command
        if( $file =~ *.??[oO] ) then
            cat $file - > $file_save
        else if( $file =~ *.??[oO].Z || $file =~ *.??[oO].gz ) then
          file $file | grep -q "Zip"
          if ( "$status" == "0" ) then
             unzip -p $file - > $file_save
          else
             file $file | grep -q "ASCII"
             if ( "$status" == "0" ) then
                cat $file > $file_save
             else
                zcat $file > $file_save
             endif
          endif
        else
        zcat $file > $file_save
        endif

        # remove the input file
        if ( $status == 0 && $del_input ) rm $file

        end
    """
    run_file.write(contents)
    run_file.close()

    os.system('chmod +x ' + run_file_path)


class pyRinexException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class pyRinexExceptionBadFile(pyRinexException):
    pass


class pyRinexExceptionSingleEpoch(pyRinexException):
    pass


class pyRinexExceptionNoAutoCoord(pyRinexException):
    pass


class RinexRecord():

    def __init__(self, NetworkCode=None, StationCode=None):
        self.StationCode = StationCode
        self.NetworkCode = NetworkCode

        self.header = None
        self.data = None
        self.firstObs = None
        self.datetime_firstObs = None
        self.datetime_lastObs = None
        self.lastObs = None
        self.antType = None
        self.marker_number = None
        self.marker_name = StationCode
        self.recType = None
        self.recNo = None
        self.recVers = None
        self.antNo = None
        self.antDome = None
        self.antOffset = None
        self.interval = None
        self.size = None
        self.x = None
        self.y = None
        self.z = None
        self.lat = None
        self.lon = None
        self.h = None
        self.date = None
        self.rinex = None
        self.crinez = None
        self.crinez_path = None
        self.rinex_path = None
        self.origin_type = None
        self.obs_types = None
        self.observables = None
        self.system = None
        self.no_cleanup = None
        self.multiday = False
        self.multiday_rnx_list = []
        self.epochs = None
        self.completion = None
        self.rel_completion = None
        self.rinex_version = None

        # log list to append all actions performed to rinex file
        self.log = []

        # list of required header records and a flag to know if they were found or not in the current header
        # also, have a tuple of default values in case there is a missing record
        self.required_records = {'RINEX VERSION / TYPE':
                                     {'format_tuple': ('%9.2f', '%11s', '%1s', '%19s', '%1s', '%19s'),
                                      'found': False,
                                      'default': ('',)},

                                 'PGM / RUN BY / DATE':
                                     {'format_tuple': ('%-20s', '%-20s', '%-20s'),
                                      'found': False,
                                      'default': ('pyRinex: 1.00 000', 'Parallel.PPP', '21FEB17 00:00:00')},

                                 'MARKER NAME':
                                     {'format_tuple': ('%-60s',),
                                      'found': False,
                                      'default': (self.StationCode.upper(),)},

                                 'MARKER NUMBER':
                                     {'format_tuple': ('%-20s',),
                                      'found': False,
                                      'default': (self.StationCode.upper(),)},

                                 'OBSERVER / AGENCY':
                                     {'format_tuple': ('%-20s', '%-40s'),
                                      'found': False,
                                      'default': ('UNKNOWN', 'UNKNOWN')},

                                 'REC # / TYPE / VERS':
                                     {'format_tuple': ('%-20s', '%-20s', '%-20s'),
                                      'found': False,
                                      'default': ('0000000', 'ASHTECH Z-XII3', 'CC00')},

                                 'ANT # / TYPE':
                                     {'format_tuple': ('%-20s', '%-20s'),
                                      'found': False,
                                      'default': ('0000', 'ASH700936C_M SNOW')},

                                 'ANTENNA: DELTA H/E/N':
                                     {'format_tuple': ('%14.4f', '%14.4f', '%14.4f'),
                                      'found': False,
                                      'default': (0.0, 0.0, 0.0)},

                                 'APPROX POSITION XYZ':
                                     {'format_tuple': ('%14.4f', '%14.4f', '%14.4f'),
                                      'found': False,
                                      'default': (0.0, 0.0, 6371000.0)},
                                 # '# / TYPES OF OBSERV' : [('%6i',), False, ('',)],
                                 'TIME OF FIRST OBS':
                                     {'format_tuple': ('%6i', '%6i', '%6i', '%6i', '%6i', '%13.7f', '%8s'),
                                      'found': False,
                                      'default': (1, 1, 1, 1, 1, 0, 'GPS')},
                                 'INTERVAL':
                                     {'format_tuple': ('%10.3f',),
                                      'found': False,
                                      'default': (30,)},  # put a wrong interval when first reading the file so that
                                 # RinSum does not fail to read RINEX if interval record is > 60 chars
                                 # DDG: remove time of last observation all together. It just creates problems and
                                 # is not mandatory
                                 # 'TIME OF LAST OBS'    : [('%6i','%6i','%6i','%6i','%6i','%13.7f','%8s'),
                                 # True, (int(first_obs.year), int(first_obs.month), int(first_obs.day),
                                 # int(23), int(59), float(59), 'GPS')],
                                 'COMMENT':
                                     {'format_tuple': ('%-60s',), 'found': True, 'default': ('',)}}

        fieldnames = ['NetworkCode', 'StationCode', 'ObservationYear', 'ObservationMonth', 'ObservationDay',
                      'ObservationDOY', 'ObservationFYear', 'ObservationSTime', 'ObservationETime', 'ReceiverType',
                      'ReceiverSerial', 'ReceiverFw', 'AntennaType', 'AntennaSerial', 'AntennaDome', 'Filename',
                      'Interval',
                      'AntennaOffset', 'Completion']

        self.record = dict.fromkeys(fieldnames)

    def load_record(self):
        self.record['NetworkCode'] = self.NetworkCode
        self.record['StationCode'] = self.StationCode
        self.record['ObservationYear'] = self.date.year
        self.record['ObservationMonth'] = self.date.month
        self.record['ObservationDay'] = self.date.day
        self.record['ObservationDOY'] = self.date.doy
        self.record['ObservationFYear'] = self.date.fyear
        self.record['ObservationSTime'] = self.firstObs
        self.record['ObservationETime'] = self.lastObs
        self.record['ReceiverType'] = self.recType
        self.record['ReceiverSerial'] = self.recNo
        self.record['ReceiverFw'] = self.recVers
        self.record['AntennaType'] = self.antType
        self.record['AntennaSerial'] = self.antNo
        self.record['AntennaDome'] = self.antDome
        self.record['Filename'] = self.rinex
        self.record['Interval'] = self.interval
        self.record['AntennaOffset'] = self.antOffset
        self.record['Completion'] = self.completion


class ReadRinex(RinexRecord):

    def read_fields(self, line, record, format_tuple):

        # create the parser object
        formatstr = re.sub(r'\..', '',
                           ' '.join(format_tuple).replace('%', '').replace('f', 's').replace('i', 's').replace('-', ''))

        fs = struct.Struct(formatstr)
        parse = fs.unpack_from

        # get the data section by spliting the line using the record text
        data = line.split(record)[0]

        if len(data) < fs.size:
            # line too short, add padding spaces
            f = '%-' + str(fs.size) + 's'
            data = f % line
        elif len(data) > fs.size:
            # line too long! cut
            data = line[0:fs.size]

        fields = [x.decode() for x in parse(data.encode('utf-8'))]

        # convert each element in the list to float if necessary
        for i, field in enumerate(fields):
            if 'f' in format_tuple[i]:
                try:
                    fields[i] = float(fields[i])
                except ValueError:
                    # invalid number in the field!, replace with something harmless
                    fields[i] = float(2.11)
            elif 'i' in format_tuple[i]:
                try:
                    fields[i] = int(fields[i])
                except ValueError:
                    # invalid number in the field!, replace with something harmless
                    fields[i] = int(1)
            elif 's' in format_tuple[i]:
                fields[i] = fields[i].strip()

        return fields, data

    def format_record(self, record_dict, record, values):

        if type(values) is not list and type(values) is not tuple:
            values = [values]

        data = ''.join(record_dict[record]['format_tuple']) % tuple(values)

        if len('%-60s' % data) > 60:
            # field is too long!! cut to 60 chars
            data = data[0:60]
            self.log_event(
                'Found that record data for ' + record + ' was too long (> 60 chars). Replaced with: ' + data)

        data = '%-60s' % data + record

        return data

    def write_rinex(self, new_header):

        if new_header != self.header:

            self.header = new_header

            # add new header
            rinex = new_header + self.data

            try:
                f = open(self.rinex_path, 'w')
                f.writelines(rinex)
                f.close()
            except Exception:
                raise

    def read_data(self):
        try:
            with open(self.rinex_path, 'r') as fileio:
                rinex = fileio.readlines()
        except Exception:
            raise

        if not any("END OF HEADER" in s for s in rinex):
            raise pyRinexExceptionBadFile('Invalid header: could not find END OF HEADER tag.')

        # find the end of header
        index = [i for i, item in enumerate(rinex) if 'END OF HEADER' in item][0]
        # delete header
        del rinex[0:index + 1]

        self.data = rinex

    def replace_record(self, header, record, new_values):

        if record not in list(self.required_records.keys()):
            raise pyRinexException('Record ' + record + ' not implemented!')

        new_header = []
        for line in header:
            if line.strip().endswith(record):
                new_header += [self.format_record(self.required_records, record, new_values) + '\n']
            else:
                new_header += [line]

        if type(new_values) is not list and type(new_values) is not tuple:
            new_values = [new_values]

        self.log_event('RINEX record replaced: ' + record + ' value: ' + ','.join(map(str, new_values)))

        return new_header

    def log_event(self, desc):

        self.log += [Event(StationCode=self.StationCode, NetworkCode=self.NetworkCode, Description=desc)]

    def insert_comment(self, header, comment):

        # remove the end or header
        index = [i for i, item in enumerate(header) if 'END OF HEADER' in item][0]
        del header[index]

        new_header = header + [self.format_record(self.required_records, 'COMMENT', comment) + '\n']

        new_header += [''.ljust(60, ' ') + 'END OF HEADER\n']

        self.log_event('RINEX COMMENT inserted: ' + comment)

        return new_header

    def __purge_comments(self, header):

        new_header = []

        for line in header:
            if not line.strip().endswith('COMMENT'):
                new_header += [line]

        self.log_event('Purged all COMMENTs from RINEX header.')

        return new_header

    def purge_comments(self):

        new_header = self.__purge_comments(self.header)

        self.write_rinex(new_header)

    def check_interval(self):

        interval_record = {'INTERVAL': {'format_tuple': ('%10.3f',), 'found': False, 'default': (30,)}}

        new_header = []

        for line in self.header:

            if line.strip().endswith('INTERVAL'):
                # get the first occurrence only!
                record = [key for key in list(interval_record.keys()) if key in line][0]

                interval_record[record]['found'] = True

                fields, _ = self.read_fields(line, 'INTERVAL', interval_record['INTERVAL']['format_tuple'])

                if fields[0] != self.interval:
                    # interval not equal. Replace record
                    new_header += [self.format_record(interval_record, 'INTERVAL', self.interval) + '\n']
                    self.log_event('Wrong INTERVAL record, setting to %i' % self.interval)
                else:
                    # record matches, leave it untouched
                    new_header += [line]
            else:
                # not a critical field, just put it back in
                if not line.strip().endswith('END OF HEADER'):
                    # leave END OF HEADER until the end to add possible missing records
                    new_header += [line]

        # now check that all the records where included! there's missing ones, then force them
        if not interval_record['INTERVAL']['found']:
            new_header += [self.format_record(interval_record, 'INTERVAL', self.interval) + '\n']
            new_header += [self.format_record(self.required_records, 'COMMENT',
                                              'pyRinex: WARN! added interval to fix file!') + '\n']
            self.log_event('INTERVAL record not found, setting to %i' % self.interval)

        new_header += [''.ljust(60, ' ') + 'END OF HEADER\n']

        self.write_rinex(new_header)

        return

    def check_header(self):

        self.header = self.get_header()
        new_header = []

        self.system = ''

        for line in self.header:

            if any(line.strip().endswith(key) for key in list(self.required_records.keys())):
                # get the first occurrence only!
                record = [key for key in list(self.required_records.keys()) if key in line][0]

                # mark the record as found
                self.required_records[record]['found'] = True

                fields, _ = self.read_fields(line, record, self.required_records[record]['format_tuple'])

                if record == 'RINEX VERSION / TYPE':
                    # read the information about the RINEX type
                    # save the system to use during TIME OF FIRST OBS
                    self.system = fields[4].strip()

                    self.rinex_version = float(fields[0])

                    # now that we know the version, we can get the first obs
                    self.read_data()
                    first_obs = self.get_firstobs()

                    if first_obs is None:
                        raise pyRinexExceptionBadFile(
                            'Could not find a first observation in RINEX file. Truncated file? Header follows:\n' + ''.join(
                                self.header))

                    if not self.system in (' ', 'G', 'R', 'S', 'E', 'M'):
                        # assume GPS
                        self.system = 'G'
                        fields[4] = 'G'
                        self.log_event('System set to (G)PS')

                else:
                    # reformat the header line
                    if record == 'TIME OF FIRST OBS' or record == 'TIME OF LAST OBS':
                        if self.system == 'M' and not fields[6].strip():
                            fields[6] = 'GPS'
                            self.log_event('Adding TIME SYSTEM to TIME OF FIRST OBS')
                        # check if the first observation is meaningful or not
                        if record == 'TIME OF FIRST OBS' and (fields[0] != first_obs.year or
                                                              fields[1] != first_obs.month or
                                                              fields[2] != first_obs.day or
                                                              fields[3] != first_obs.hour or
                                                              fields[4] != first_obs.minute or
                                                              fields[5] != first_obs.second):
                            # bad first observation! replace with the real one
                            fields[0] = first_obs.year
                            fields[1] = first_obs.month
                            fields[2] = first_obs.day
                            fields[3] = first_obs.hour
                            fields[4] = first_obs.minute
                            fields[5] = first_obs.second

                            self.log_event('Bad TIME OF FIRST OBS found -> fixed')

                    if record == 'MARKER NAME':
                        # load the marker name, which RinSum does not return
                        self.marker_name = fields[0].strip().lower()

                # regenerate the fields
                # save to new header
                new_header += [self.format_record(self.required_records, record, fields) + '\n']
            else:
                # not a critical field, just put it back in
                if not line.strip().endswith('END OF HEADER') and not line.strip().endswith(
                        'TIME OF LAST OBS') and line.strip() != '':
                    if line.strip().endswith('COMMENT'):
                        # reformat comments (some come in wrong positions!)
                        fields, _ = self.read_fields(line, 'COMMENT', self.required_records['COMMENT']['format_tuple'])
                        new_header += [self.format_record(self.required_records, 'COMMENT', fields) + '\n']
                    else:
                        # leave END OF HEADER until the end to add possible missing records
                        new_header += [line]

        if self.system == '':
            # if we are out of the loop and we could not determine the system, raise error
            raise pyRinexExceptionBadFile('Unfixable RINEX header: could not find RINEX VERSION / TYPE')

        # now check that all the records where included! there's missing ones, then force them
        if not all([item['found'] for item in list(self.required_records.values())]):
            # get the keys of the missing records
            missing_records = {item: self.required_records[item] for item in self.required_records if
                               self.required_records[item]['found'] == False}

            for record in list(missing_records.keys()):
                if '# / TYPES OF OBSERV' in record:
                    raise pyRinexExceptionBadFile('Unfixable RINEX header: could not find # / TYPES OF OBSERV')

                new_header += [self.format_record(missing_records, record, missing_records[record]['default']) + '\n']
                new_header += [self.format_record(self.required_records, 'COMMENT',
                                                  'pyRinex: WARN! default value to fix file!') + '\n']
                self.log_event('Missing required RINEX record added: ' + record)

        new_header += [''.ljust(60, ' ') + 'END OF HEADER\n']

        self.write_rinex(new_header)

    def IdentifyFile(self, input_file):

        # get the crinez and rinex names
        filename = os.path.basename(input_file)

        self.origin_file = input_file
        self.origin_type = self.identify_type(filename)
        self.local_copy = os.path.abspath(os.path.join(self.rootdir, filename))

        self.rinex = self.to_format(filename, TYPE_RINEX)
        self.crinez = self.to_format(filename, TYPE_CRINEZ)

        # get the paths
        self.crinez_path = os.path.join(self.rootdir, self.crinez)
        self.rinex_path = os.path.join(self.rootdir, self.rinex)

        self.log_event('Origin type is %i' % self.origin_type)

        return

    def CreateTempDirs(self):

        self.rootdir = os.path.join('production', 'rinex')
        self.rootdir = os.path.join(self.rootdir, str(uuid.uuid4()))

        # create a production folder to analyze the rinex file
        if not os.path.exists(self.rootdir):
            os.makedirs(self.rootdir)

        return

    def Uncompress(self):

        if self.origin_type in (TYPE_CRINEZ, TYPE_CRINEX):

            size = os.path.getsize(self.local_copy)

            # run crz2rnx with timeout structure
            cmd = pyRunWithRetry.RunCommand('crz2rnx -f -d ' + self.local_copy, 30)
            try:
                _, err = cmd.run_shell()
            except pyRunWithRetry.RunCommandWithRetryExeception as e:
                # catch the timeout except and pass it as a pyRinexException
                raise pyRinexException(str(e))

            # the uncompressed-unhatanaked file size must be at least > than the crinez
            if os.path.isfile(self.rinex_path):
                if err and os.path.getsize(self.rinex_path) <= size:
                    raise pyRinexExceptionBadFile(
                        "Error in ReadRinex.__init__ -- crz2rnx: error and empty file: " + self.origin_file + ' -> ' + err)
            else:
                if err:
                    raise pyRinexException('Could not create RINEX file. crz2rnx stderr follows: ' + err)
                else:
                    raise pyRinexException(
                        'Could not create RINEX file. Unknown reason. Possible problem with crz2rnx?')

        elif self.origin_type is TYPE_RINEZ:
            # create an unzip script
            create_unzip_script(os.path.join(self.rootdir, 'uncompress.sh'))

            cmd = pyRunWithRetry.RunCommand('./uncompress.sh -f -d ' + self.local_copy, 30, self.rootdir)
            try:
                _, _ = cmd.run_shell()
            except pyRunWithRetry.RunCommandWithRetryExeception as e:
                # catch the timeout except and pass it as a pyRinexException
                raise pyRinexException(str(e))

    def ConvertRinex3to2(self):

        # most programs still don't support RINEX 3 (partially implemented in this code)
        # convert to RINEX 2.11 using RinEdit
        cmd = pyRunWithRetry.RunCommand('RinEdit --IF %s --OF %s.t --ver2' % (self.rinex, self.rinex), 15, self.rootdir)

        try:
            out, _ = cmd.run_shell()

            if 'exception' in out.lower():
                raise pyRinexExceptionBadFile('RinEdit returned error converting to RINEX 2.11:\n' + out)

            if not os.path.exists(self.rinex_path + '.t'):
                raise pyRinexExceptionBadFile('RinEdit failed to convert to RINEX 2.11:\n' + out)

            # if all ok, move converted file to rinex_path
            os.remove(self.rinex_path)
            move(self.rinex_path + '.t', self.rinex_path)
            # change version
            self.rinex_version = 2.11

            self.log_event('Origin file was RINEX 3 -> Converted to 2.11')

        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise pyRinexException(str(e))

    def RunRinSum(self):
        # run RinSum to get file information
        cmd = pyRunWithRetry.RunCommand('RinSum --notable ' + self.rinex_path, 45)  # DDG: increased from 21 to 45.
        try:

            output, _ = cmd.run_shell()
        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise pyRinexException(str(e))

        # write RinSum output to a log file (debug purposes)
        info = open(self.rinex_path + '.log', 'w')
        info.write(output)
        info.close()

        return output

    def isValidRinexName(self, filename):

        filename = os.path.basename(filename)
        sfile = re.findall('(\w{4})(\d{3})(\w{1})\.(\d{2})([do]\.[Z])$', filename)

        if sfile:
            return True
        else:
            sfile = re.findall('(\w{4})(\d{3})(\w{1})\.(\d{2})([od])$', filename)

            if sfile:
                return True
            else:
                return False

    def __init__(self, NetworkCode, StationCode, origin_file, no_cleanup=False, allow_multiday=False):
        """
        pyRinex initialization
        if file is multiday, DO NOT TRUST date object for initial file. Only use pyRinex objects contained in the multiday list
        """
        RinexRecord.__init__(self, NetworkCode, StationCode)

        self.no_cleanup = no_cleanup
        self.origin_file = None

        # check that the rinex file name is valid!
        if not self.isValidRinexName(origin_file):
            raise pyRinexException('File name does not follow the RINEX(Z)/CRINEX(Z) naming convention: %s'
                                   % (os.path.basename(origin_file)))

        self.CreateTempDirs()

        self.IdentifyFile(origin_file)

        copy(origin_file, self.rootdir)

        if self.origin_type in (TYPE_CRINEZ, TYPE_CRINEX, TYPE_RINEZ):
            self.Uncompress()

        # check basic infor in the rinex header to avoid problems with RinSum
        self.check_header()

        if self.rinex_version >= 3:
            self.ConvertRinex3to2()

        # process the output
        self.parse_output(self.RunRinSum())

        # DDG: new interval checking after running RinSum
        # check the sampling interval
        self.check_interval()

        # check for files that have more than one day inside (yes, there are some like this... amazing)
        # condition is: the start and end date don't match AND
        # either there is more than two hours in the second day OR
        # there is more than one day of data
        if self.datetime_lastObs.date() != self.datetime_firstObs.date() and not allow_multiday:
            # more than one day in this file. Is there more than one hour? (at least in principle, based on the time)
            first_obs = datetime.datetime(self.datetime_lastObs.date().year,
                                          self.datetime_lastObs.date().month,
                                          self.datetime_lastObs.date().day)

            if (self.datetime_lastObs - first_obs).total_seconds() >= 3600:
                # the file has more than one day in it...
                # use teqc to window the data
                if not self.multiday_handle(origin_file):
                    return
            else:
                # window the data to remove superfluous epochs
                last_obs = datetime.datetime(self.datetime_firstObs.date().year,
                                             self.datetime_firstObs.date().month,
                                             self.datetime_firstObs.date().day, 23, 59, 59)
                self.window_data(end=last_obs)
                self.log_event('RINEX had incomplete epochs (or < 1 hr) outside of the corresponding UTC day -> '
                               'Data windowed to one UTC day.')

        # reported date for this file is session/2
        self.date = pyDate.Date(datetime=self.datetime_firstObs + (self.datetime_lastObs - self.datetime_firstObs) / 2)

        # DDG: calculate the completion of the file (at sampling rate)
        # completion of day
        # done after binning so that if the file is a multiday we don't get a bad completion
        self.completion = self.epochs * self.interval / 86400
        # completion of time window in file
        self.rel_completion = self.epochs * self.interval / ((self.datetime_lastObs -
                                                              self.datetime_firstObs).total_seconds() + self.interval)

        # load the RinexRecord class
        self.load_record()

        return

    def multiday_handle(self, origin_file):
        # split the file
        self.split_file()

        continue_statements = True

        if len(self.multiday_rnx_list) > 1:
            # truly a multiday file
            self.multiday = True
            # self.log_event('RINEX file is multiday -> generated $i RINEX files' % len(self.multiday_rnx_list))
            continue_statements = True

        elif len(self.multiday_rnx_list) == 1:
            # maybe one of the files has a single epoch in it. Drop the current rinex and use the binned version
            self.cleanup()
            temp_path = self.multiday_rnx_list[0].rootdir
            # keep the log
            self.log_event('RINEX appeared to be multiday but had incomplete epochs (or < 1 hr) -> '
                           'Data windowed to one UTC day.')
            temp_log = self.log
            # set to no cleanup so that the files survive the __init__ statement
            self.multiday_rnx_list[0].no_cleanup = True
            # reinitialize self
            self.__init__(self.multiday_rnx_list[0].NetworkCode, self.multiday_rnx_list[0].StationCode,
                          self.multiday_rnx_list[0].rinex_path)
            # the origin file should still be the rinex passed to init the object, not the multiday file
            self.origin_file = origin_file
            # remove the temp directory
            self.log += temp_log

            rmtree(temp_path)
            # now self points the the binned version of the rinex
            continue_statements = False

        return continue_statements

    def split_file(self):

        # run in the local folder to get the files inside rootdir
        cmd = pyRunWithRetry.RunCommand('teqc -n_GLONASS 64 -n_GPS 64 -n_Galileo 64 -n_SBAS 64 -tbin 1d rnx ' +
                                        self.rinex, 45, self.rootdir)
        try:
            _, err = cmd.run_shell()
        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise pyRinexException(str(e))

        # successfully binned the file
        # delete current file and rename the new files
        os.remove(self.rinex_path)

        # now we should have as many files named rnxDDD0.??o as days inside the RINEX
        for file in os.listdir(self.rootdir):
            if file[0:3] == 'rnx' and self.identify_type(file) is TYPE_RINEX:
                # rename file
                move(os.path.join(self.rootdir, file), os.path.join(self.rootdir,
                                                                    file.replace('rnx', self.StationCode)))
                # get the info for this file
                try:
                    rnx = ReadRinex(self.NetworkCode, self.StationCode,
                                    os.path.join(self.rootdir, file.replace('rnx', self.StationCode)))
                    # append this rinex object to the multiday list
                    self.multiday_rnx_list.append(rnx)
                except (pyRinexException, pyRinexExceptionBadFile):
                    # there was a problem with one of the multiday files. Do not append
                    pass

        return

    def parse_output(self, output):

        try:
            self.x, self.y, self.z = [float(x) for x in
                                      re.findall(r'Position\s+\(XYZ,m\)\s:\s\(\s*(\-?\d+\.\d+)\,\s*(-?\d+\.\d+)\,'
                                                 r'\s*(-?\d+\.\d+)', output, re.MULTILINE)[0]]
            self.lat, self.lon, self.h = ecef2lla([self.x, self.y, self.z])
        except Exception:
            self.x, self.y, self.z = (None, None, None)

        try:
            self.antOffset, self.antOffsetN, self.antOffsetE = [float(x) for x in
                                                                re.findall(r'Antenna\sDelta\s+\(HEN,m\)\s:\s'
                                                                           r'\(\s*(\-?\d+\.\d+)\,\s*(-?\d+\.\d+)'
                                                                           r'\,\s*(-?\d+\.\d+)', output,
                                                                           re.MULTILINE)[0]]
        except Exception:
            self.antOffset, self.antOffsetN, self.antOffsetE = (0, 0, 0)
            self.log_event('Problem parsing ANTENNA OFFSETS, setting to 0')

        try:
            self.recNo, self.recType, self.recVers = [x.strip() for x in
                                                      re.findall(r'Rec#:([^,]*),\s*Type:([^,]*),\s*Vers:(.*)',
                                                                 output, re.MULTILINE)[0]]
        except Exception:
            self.recNo, self.recType, self.recVers = ('', '', '')
            self.log_event('Problem parsing REC # / TYPE / VERS, setting to EMPTY')

        try:
            self.marker_number = re.findall(r'^Marker number\s*:\s*(.*)', output, re.MULTILINE)[0]
        except Exception:
            self.marker_number = 'NOT FOUND'
            self.log_event('No MARKER NUMBER found, setting to NOT FOUND')

        try:
            self.antNo, AntDome = [x.strip() for x in re.findall(r'Antenna\s*#\s*:([^,]*),\s*Type\s*:\s*(.*)',
                                                                 output, re.MULTILINE)[0]]

            if ' ' in AntDome:
                self.antType = AntDome.split()[0]
                self.antDome = AntDome.split()[1]
            else:
                self.antType = AntDome
                self.antDome = 'NONE'
                self.log_event('No dome found, set to NONE')

        except Exception:
            self.antNo, self.antType, self.antDome = ('UNKNOWN', 'UNKNOWN', 'NONE')
            self.log_event('Problem parsing ANT # / TYPE, setting to UNKNOWN NONE')

        try:
            self.interval = float(re.findall(r'^Computed interval\s*(\d+\.\d+)', output, re.MULTILINE)[0])
        except Exception:
            self.interval = 0
            self.log_event('Problem interval, setting to 0')

        try:
            self.epochs = float(re.findall(r'^There were\s*(\d+)\s*epochs', output, re.MULTILINE)[0])
        except Exception:
            self.epochs = 0
            self.log_event('Problem parsing epochs, setting to 0')

        # stop here is epochs of interval is invalid
        if self.interval == 0:
            if self.epochs > 0:
                raise pyRinexExceptionSingleEpoch('RINEX interval equal to zero. Single epoch or bad RINEX file. '
                                                  'Reported epochs in file were %s' % (self.epochs))
            else:
                raise pyRinexExceptionSingleEpoch('RINEX interval equal to zero. Single epoch or bad RINEX file. '
                                                  'No epoch information to report. The output from RinSum was:\n' +
                                                  output)

        elif self.interval > 120:
            raise pyRinexExceptionBadFile('RINEX sampling interval > 120s. The output from RinSum was:\n' + output)

        elif self.epochs * self.interval < 3600:
            raise pyRinexExceptionBadFile('RINEX file with < 1 hr of observation time. '
                                          'The output from RinSum was:\n' + output)

        try:
            yy, mm, dd, hh, MM, ss = [int(x) for x in re.findall(r'^Computed first epoch:\s*(\d+)\/(\d+)\/(\d+)'
                                                                 r'\s(\d+):(\d+):(\d+)', output, re.MULTILINE)[0]]
            yy = check_year(yy)
            self.datetime_firstObs = datetime.datetime(yy, mm, dd, hh, MM, ss)
            self.firstObs = self.datetime_firstObs.strftime('%Y/%m/%d %H:%M:%S')

            yy, mm, dd, hh, MM, ss = [int(x) for x in re.findall(r'^Computed last\s*epoch:\s*(\d+)\/(\d+)'
                                                                 r'\/(\d+)\s(\d+):(\d+):(\d+)', output,
                                                                 re.MULTILINE)[0]]
            yy = check_year(yy)
            self.datetime_lastObs = datetime.datetime(yy, mm, dd, hh, MM, ss)
            self.lastObs = self.datetime_lastObs.strftime('%Y/%m/%d %H:%M:%S')

            if self.datetime_lastObs <= self.datetime_firstObs:
                # bad rinex! first obs > last obs
                raise pyRinexExceptionBadFile('Last observation (' + self.lastObs + ') <= first observation (' +
                                              self.firstObs + ')')

        except Exception:
            raise pyRinexException(self.rinex_path + ': error in ReadRinex.parse_output: the output for first/last obs '
                                                     'is invalid. The output from RinSum was:\n' + output)

        try:
            self.size = int(re.findall(r'^Computed file size:\s*(\d+)', output, re.MULTILINE)[0])
        except Exception:
            self.size = 0
            self.log_event('Problem parsing size, setting to 0')

        try:
            self.obs_types = int(re.findall(r'GPS Observation types\s*\((\d+)\)', output, re.MULTILINE)[0])
        except Exception:
            self.obs_types = 0
            self.log_event('Problem parsing observation types, setting to 0')

        try:
            observables = re.findall(r'System GPS Obs types.*\[v2: (.*)\]', output, re.MULTILINE)[0]
            self.observables = observables.strip().split()
        except Exception as e:
            self.observables = ()
            self.log_event('Problem parsing observables (%s), setting to ()' % str(e))

        warn = re.findall('(.*Warning : Failed to read header: text 0:Incomplete or invalid header.*)', output,
                          re.MULTILINE)
        if warn:
            raise pyRinexException("Warning in ReadRinex.parse_output: " + warn[0])

        warn = re.findall('(.*unexpected exception.*)', output, re.MULTILINE)
        if warn:
            raise pyRinexException("unexpected exception in ReadRinex.parse_output: " + warn[0])

        warn = re.findall('(.*Exception.*)', output, re.MULTILINE)
        if warn:
            raise pyRinexException("Exception in ReadRinex.parse_output: " + warn[0])

        warn = re.findall('(.*no data found. Are time limits wrong.*)', output, re.MULTILINE)
        if warn:
            raise pyRinexException('RinSum: no data found. Are time limits wrong for file ' + self.rinex +
                                   ' details:' + warn[0])

    def get_firstobs(self):

        if self.rinex_version < 3:
            fs = struct.Struct('1s2s1s2s1s2s1s2s1s2s11s2s1s3s')
        else:
            fs = struct.Struct('2s4s1s2s1s2s1s2s1s2s11s2s1s3s')

        parse = fs.unpack_from

        date = None

        skip = 0
        for line in self.data:
            if skip == 0:
                fields = list(parse(line.encode('utf-8')))

                if int(fields[12]) <= 1:  # OK FLAG
                    # read first observation
                    year = int(fields[1])
                    month = int(fields[3])
                    day = int(fields[5])
                    hour = int(fields[7])
                    minute = int(fields[9])
                    second = float(fields[10])

                    try:
                        date = pyDate.Date(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
                    except pyDate.pyDateException as e:
                        raise pyRinexExceptionBadFile(str(e))

                    break
                elif int(fields[12]) > 1:
                    # event, skip lines indicated in next field
                    skip = int(fields[13])
            else:
                skip -= 1

        return date

    def get_header(self):

        header = []
        # retry reading. Every now and then there is a problem during file read.
        for i in range(2):
            try:
                with open(self.rinex_path, 'r') as fileio:

                    for line in fileio:
                        header.append(line)
                        if line.strip().endswith('END OF HEADER'):
                            break
                    break
            except IOError:
                # try again
                if i == 0:
                    continue
                else:
                    raise

        return header

    def auto_coord(self, brdc, chi_limit=3):
        # use gamit's sh_rx2apr to obtain a coordinate of the station

        # do not work with the original file. Decimate and remove other systems (to increase speed)
        try:
            # make a copy to decimate and remove systems to help sh_rx2apr
            # allow multiday files (will not change the answer), just get a coordinate for this file
            rnx = ReadRinex(self.NetworkCode, self.StationCode, self.rinex_path, allow_multiday=True)

            if rnx.interval < 15:
                rnx.decimate(30)
                self.log_event('Decimating to 30 seconds to run auto_coord')

            # remove the other systems that sh_rx2apr does not use
            if rnx.system is 'M':
                rnx.remove_systems()
                self.log_event('Removing systems S, R and E to run auto_coord')

        except pyRinexException as e:
            # print str(e)
            # ooops, something went wrong, try with local file (without removing systems or decimating)
            rnx = self
            # raise pyRinexExceptionBadFile('During decimation or remove_systems (to run auto_coord),
            # teqc returned: %s' + str(e))

        # copy brdc orbit
        copyfile(brdc.brdc_path, os.path.join(rnx.rootdir, brdc.brdc_filename))

        # check if the apr coordinate is zero and iterate more than once if true
        if self.x == 0 and self.y == 0 and self.z == 0:
            max_it = 2
        else:
            max_it = 1

        for i in range(max_it):

            cmd = pyRunWithRetry.RunCommand(
                'sh_rx2apr -site ' + rnx.rinex + ' -nav ' + brdc.brdc_filename + ' -chi ' + str(chi_limit), 40,
                rnx.rootdir)
            # leave errors un-trapped on purpose (will raise an error to the parent)
            out, err = cmd.run_shell()

            if err != '' and err is not None:
                raise pyRinexExceptionNoAutoCoord(str(err) + '\n' + out)
            else:
                # check that the Final chi**2 is < 3
                for line in out.split('\n'):
                    if '* Final sqrt(chi**2/n)' in line:
                        chi = line.split()[-1]

                        if chi == 'NaN':
                            raise pyRinexExceptionNoAutoCoord('chi2 = NaN! ' + str(err) + '\n' + out)

                        elif float(chi) < chi_limit:
                            # open the APR file and read the coordinates
                            if os.path.isfile(os.path.join(rnx.rootdir, rnx.rinex[0:4] + '.apr')):
                                with open(os.path.join(rnx.rootdir, rnx.rinex[0:4] + '.apr')) as apr:
                                    line = apr.readline().split()

                                    self.x = float(line[1])
                                    self.y = float(line[2])
                                    self.z = float(line[3])

                                    self.lat, self.lon, self.h = ecef2lla([self.x, self.y, self.z])

                                # only exit and return coordinate if current iteration == max_it
                                # (minus one due to arrays starting at 0).
                                if i == max_it - 1:
                                    return (float(line[1]), float(line[2]), float(line[3])), \
                                           (self.lat, self.lon, self.h)

                # copy the header to replace with new coordinate
                # note that this piece of code only executes if there is more than one iteration
                new_header = self.header
                new_header = self.replace_record(new_header, 'APPROX POSITION XYZ', (self.x, self.y, self.z))
                # write the rinex file with the new header
                rnx.write_rinex(new_header)

        raise pyRinexExceptionNoAutoCoord(str(out) + '\nLIMIT FOR CHI**2 was %i' % chi_limit)

    def window_data(self, start=None, end=None, copyto=None):
        """
        Window the RINEX data using TEQC
        :param copyto:
        :param start: a start datetime or self.firstObs if None
        :param end: a end datetime or self.lastObs if None
        :return:
        """
        if start is None:
            start = self.datetime_firstObs
            self.log_event('Setting start = first obs in window_data')

        if end is None:
            end = self.datetime_lastObs
            self.log_event('Setting end = last obs in window_data')

        cmd = pyRunWithRetry.RunCommand(
            'teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 -st %i%02i%02i%02i%02i%02i -e %i%02i%02i%02i%02i%02i +obs %s.t %s' % (
                start.year, start.month, start.day, start.hour, start.minute, start.second,
                end.year, end.month, end.day, end.hour, end.minute, end.second, self.rinex_path, self.rinex_path), 5)

        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            # delete the original file and replace with .t
            if copyto is None:
                os.remove(self.rinex_path)
                move(self.rinex_path + '.t', self.rinex_path)
                self.datetime_firstObs = start
                self.datetime_lastObs = end
                self.firstObs = self.datetime_firstObs.strftime('%Y/%m/%d %H:%M:%S')
                self.lastObs = self.datetime_lastObs.strftime('%Y/%m/%d %H:%M:%S')
            else:
                move(self.rinex_path + '.t', copyto)
        else:
            raise pyRinexException(err)

        return

    def decimate(self, decimate_rate, copyto=None):
        # if copy to is passed, then the decimation is done on the copy of the file, not on the current rinex.
        # otherwise, decimation is done in current rinex
        if copyto is not None:
            copyfile(self.rinex_path, copyto)
        else:
            copyto = self.rinex_path
            self.interval = decimate_rate

        if self.rinex_version < 3:
            cmd = pyRunWithRetry.RunCommand('teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 -O.dec %i '
                                            '+obs %s.t %s' % (decimate_rate, copyto, copyto), 5)
            # leave errors un-trapped on purpose (will raise an error to the parent)
        else:
            cmd = pyRunWithRetry.RunCommand('RinEdit --IF %s --OF %s.t --TN %i --TB %i,%i,%i,%i,%i,%i'
                                            % (os.path.basename(copyto), os.path.basename(copyto), decimate_rate,
                                               self.date.year, self.date.month, self.date.day, 0, 0, 0),
                                            15, self.rootdir)
        out, err = cmd.run_shell()

        if 'teqc: failure to read' not in str(err):
            # delete the original file and replace with .t
            os.remove(copyto)
            move(copyto + '.t', copyto)
        else:
            raise pyRinexException(err)

        self.log_event('RINEX decimated to %is (applied to %s)' % (decimate_rate, str(copyto)))

        return

    def remove_systems(self, systems=('R', 'E', 'S'), copyto=None):
        # if copy to is passed, then the system removal is done on the copy of the file, not on the current rinex.
        # other wise, system removal is done to current rinex
        if copyto is not None:
            copyfile(self.rinex_path, copyto)
        else:
            copyto = self.rinex_path

        if self.rinex_version < 3:
            rsys = '-' + ' -'.join(systems)
            cmd = pyRunWithRetry.RunCommand(
                'teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 %s +obs %s.t %s' % (rsys, copyto, copyto), 5)
        else:
            rsys = ' --DS '.join(systems)
            cmd = pyRunWithRetry.RunCommand(
                'RinEdit --IF %s --OF %s.t --DS %s' % (os.path.basename(copyto), os.path.basename(copyto), rsys), 15,
                self.rootdir)

        # leave errors un-trapped on purpose (will raise an error to the parent)
        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            # delete the original file and replace with .t
            os.remove(copyto)
            move(copyto + '.t', copyto)
            # if working on local copy, reload the rinex information
            if copyto == self.rinex_path:
                # reload information from this file
                self.parse_output(self.RunRinSum())
        else:
            raise pyRinexException(err)

        self.log_event('Removed systems %s (applied to %s)' % (','.join(systems), str(copyto)))

        return

    def normalize_header(self, NewValues, brdc=None, x=None, y=None, z=None):
        # this function gets rid of the heaer information and replaces it with the station info (trusted)
        # should be executed before calling PPP or before rebuilding the Archive
        # new function now accepts a dictionary OR a station info object

        if type(NewValues) is pyStationInfo.StationInfo:
            if NewValues.date is not None and NewValues.date != self.date:
                raise pyRinexException('The StationInfo object was initialized for a different date than that of the '
                                       'RINEX file. Date on RINEX: ' + self.date.yyyyddd() +
                                       '; Station Info: ' + NewValues.date.yyyyddd())
            else:
                NewValues = NewValues.currentrecord

        fieldnames = ['AntennaHeight', 'AntennaNorth', 'AntennaEast', 'ReceiverCode', 'ReceiverVers',
                      'ReceiverSerial', 'AntennaCode', 'RadomeCode', 'AntennaSerial']
        rinex_field = ['AntennaOffset', None, None, 'ReceiverType', 'ReceiverFw', 'ReceiverSerial',
                       'AntennaType', 'AntennaDome', 'AntennaSerial']

        new_header = self.header

        # set values
        for i, field in enumerate(fieldnames):
            if field not in list(NewValues.keys()):
                if rinex_field[i] is not None:
                    NewValues[field] = self.record[rinex_field[i]]
                else:
                    NewValues[field] = 0.0

        if self.marker_name != self.StationCode:
            new_header = self.replace_record(new_header, 'MARKER NAME', self.StationCode.upper())
            new_header = self.insert_comment(new_header, 'PREV MARKER NAME: ' + self.marker_name.upper())
            self.marker_name = self.StationCode

        if (NewValues['ReceiverCode'] != self.recType or
                NewValues['ReceiverVers'] != self.recVers or
                NewValues['ReceiverSerial'] != self.recNo):

            new_header = self.replace_record(new_header, 'REC # / TYPE / VERS',
                                             (NewValues['ReceiverSerial'], NewValues['ReceiverCode'],
                                              NewValues['ReceiverVers']))

            if NewValues['ReceiverSerial'] != self.recNo:
                new_header = self.insert_comment(new_header, 'PREV REC #   : ' + self.recNo)
                self.recNo = NewValues['ReceiverSerial']
            if NewValues['ReceiverCode'] != self.recType:
                new_header = self.insert_comment(new_header, 'PREV REC TYPE: ' + self.recType)
                self.recType = NewValues['ReceiverCode']
            if NewValues['ReceiverVers'] != self.recVers:
                new_header = self.insert_comment(new_header, 'PREV REC VERS: ' + self.recVers)
                self.recVers = NewValues['ReceiverVers']

        # if (NewValues['AntennaCode'] != self.antType or
        #    NewValues['AntennaSerial'] != self.antNo or
        #    NewValues['RadomeCode'] != self.antDome):
        if True:

            # DDG: New behaviour, ALWAYS replace the antenna and DOME field due to problems with formats for some
            # stations. Eg:
            # 13072               ASH700936D_M    NONE                    ANT # / TYPE
            # 13072               ASH700936D_M SNOW                       ANT # / TYPE
            new_header = self.replace_record(new_header, 'ANT # / TYPE',
                                             (NewValues['AntennaSerial'],
                                              '%-15s' % NewValues['AntennaCode'] + ' ' + NewValues['RadomeCode']))

            if NewValues['AntennaCode'] != self.antType:
                new_header = self.insert_comment(new_header, 'PREV ANT #   : ' + self.antType)
                self.antType = NewValues['AntennaCode']
            if NewValues['AntennaSerial'] != self.antNo:
                new_header = self.insert_comment(new_header, 'PREV ANT TYPE: ' + self.antNo)
                self.antNo = NewValues['AntennaSerial']
            if NewValues['RadomeCode'] != self.antDome:
                new_header = self.insert_comment(new_header, 'PREV ANT DOME: ' + self.antDome)
                self.antDome = NewValues['RadomeCode']

        if (NewValues['AntennaHeight'] != self.antOffset or
                NewValues['AntennaNorth'] != self.antOffsetN or
                NewValues['AntennaEast'] != self.antOffsetE):

            new_header = self.replace_record(new_header, 'ANTENNA: DELTA H/E/N',
                                             (NewValues['AntennaHeight'], NewValues['AntennaEast'],
                                              NewValues['AntennaNorth']))

            if NewValues['AntennaHeight'] != self.antOffset:
                new_header = self.insert_comment(new_header, 'PREV DELTA H: %.4f' % self.antOffset)
                self.antOffset = float(NewValues['AntennaHeight'])
            if NewValues['AntennaNorth'] != self.antOffsetN:
                new_header = self.insert_comment(new_header, 'PREV DELTA N: %.4f' % self.antOffsetN)
                self.antOffsetN = float(NewValues['AntennaNorth'])
            if NewValues['AntennaEast'] != self.antOffsetE:
                new_header = self.insert_comment(new_header, 'PREV DELTA E: %.4f' % self.antOffsetE)
                self.antOffsetE = float(NewValues['AntennaEast'])

        # always replace the APPROX POSITION XYZ
        if x is None and brdc is None and self.x is None:
            raise pyRinexException(
                'Cannot normalize the header\'s APPROX POSITION XYZ without a coordinate or '
                'a valid broadcast ephemeris object')

        elif self.x is None and brdc is not None:
            self.auto_coord(brdc)

        elif x is not None:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

        new_header = self.replace_record(new_header, 'APPROX POSITION XYZ', (self.x, self.y, self.z))

        new_header = self.insert_comment(new_header, 'APPROX POSITION SET TO AUTONOMOUS SOLUTION')
        new_header = self.insert_comment(new_header, 'HEADER NORMALIZED BY pyRinex ON ' +
                                         datetime.datetime.now().strftime('%Y/%m/%d %H:%M'))

        self.write_rinex(new_header)

        return

    def apply_file_naming_convention(self):
        """
        function to rename a file to make it consistent with the RINEX naming convention
        :return:
        """
        # is the current rinex filename valid?
        fileparts = Utils.parse_crinex_rinex_filename(self.rinex)

        if fileparts:
            doy = int(fileparts[1])
            year = int(Utils.get_norm_year_str(fileparts[3]))
        else:
            # invalid RINEX filename! Assign some values to the variables
            doy = 0
            year = 1900

        if self.record['ObservationDOY'] != doy or self.record['ObservationYear'] != year or \
                fileparts[0].lower() != self.StationCode:
            # this if still remains here but we do not allow this condition any more to happen.
            # See process_crinex_file -> if Result...
            # NO! rename the file before moving to the archive
            filename = self.StationCode + self.date.ddd() + '0.' + self.date.yyyy()[2:4] + 'o'
            # rename file
            self.rename(filename)

    def move_origin_file(self, path, destiny_type=TYPE_CRINEZ):
        # this function moves the ARCHIVE file (or repository) to another location indicated by path
        # can also specify other types, but assumed to be CRINEZ by default
        # it also makes sure that it doesn' overwrite any existing file

        if not os.path.isabs(path):
            raise pyRinexException('Destination must be an absolute path')

        if destiny_type is TYPE_CRINEZ:
            dst = self.crinez
        elif destiny_type is TYPE_RINEX:
            dst = self.rinex
        else:
            dst = self.to_format(self.rinex, destiny_type)

        filename = ''
        # determine action base on origin type
        if self.origin_type == destiny_type:
            # intelligent move (creates folder and checks for file existence)
            # origin and destiny match, do the thing directly
            filename = Utils.move(self.origin_file, os.path.join(path, dst))
        else:
            # if other types are requested, or origin is not the destiny type, then use local file and delete the
            if destiny_type is TYPE_RINEX:
                filename = Utils.move(self.rinex_path, os.path.join(path, dst))

            elif destiny_type is TYPE_CRINEZ:
                filename = self.compress_local_copyto(path)

            elif destiny_type is TYPE_CRINEX:
                cmd = pyRunWithRetry.RunCommand('rnx2crx -f ' + self.rinex_path, 45)
                try:
                    _, err = cmd.run_shell()

                    if os.path.getsize(os.path.join(self.rootdir, self.to_format(self.rinex, TYPE_CRINEX))) == 0:
                        raise pyRinexException(
                            'Error in move_origin_file: compressed version of ' + self.rinex_path + ' has zero size!')
                except pyRunWithRetry.RunCommandWithRetryExeception as e:
                    # catch the timeout except and pass it as a pyRinexException
                    raise pyRinexException(str(e))

            elif destiny_type is TYPE_RINEZ:
                raise pyRinexException('pyRinex will not natively generate a RINEZ file.')

        # to keep everything consistent, also change the local copies of the file
        if filename != '':
            self.rename(filename)
            # delete original (if the dest exists!)
            if os.path.isfile(self.origin_file):
                if os.path.isfile(os.path.join(path, dst)):
                    os.remove(self.origin_file)
                else:
                    raise pyRinexException(
                        'New \'origin_file\' (%s) does not exist!' % os.path.isfile(os.path.join(path, dst)))

            # change origin file reference
            self.origin_file = os.path.join(path, dst)
            self.origin_type = destiny_type

            self.log_event('Origin moved to %s and converted to %i' % (self.origin_file, destiny_type))

        return filename

    def compress_local_copyto(self, path):
        # this function compresses and moves the local copy of the rinex
        # meant to be used when a multiday rinex file is encountered and we need to move it to the repository

        # compress the rinex into crinez. Make the filename
        crinez = self.to_format(self.rinex, TYPE_CRINEZ)

        # we make the crinez again (don't use the existing from the database) to apply any corrections
        # made during the __init__ stage. Notice the -f in rnx2crz
        cmd = pyRunWithRetry.RunCommand('rnx2crz -f ' + self.rinex_path, 45)
        try:
            _, err = cmd.run_shell()

            if os.path.getsize(os.path.join(self.rootdir, crinez)) == 0:
                raise pyRinexException(
                    'Error in compress_local_copyto: compressed version of ' + self.rinex_path + ' has zero size!')
        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise pyRinexException(str(e))

        filename = Utils.copyfile(os.path.join(self.rootdir, crinez), os.path.join(path, crinez))

        self.log_event('Created CRINEZ from local copy and copied to %s' % path)

        return filename

    def rename(self, new_name=None, NetworkCode=None, StationCode=None):

        # function that renames the local crinez and rinex file based on the provided information
        # it also changes the variables in the object to reflect this change
        # new name can be any valid format (??d.Z, .??o, ??d, ??o.Z)

        if new_name:
            rinex = os.path.basename(self.to_format(new_name, TYPE_RINEX))
            # do not continue executing unless there is a REAL change!
            if rinex != self.rinex:
                crinez = os.path.basename(self.to_format(new_name, TYPE_CRINEZ))

                # rename the rinex
                if os.path.isfile(self.rinex_path):
                    move(self.rinex_path, os.path.join(self.rootdir, rinex))

                self.rinex_path = os.path.join(self.rootdir, rinex)

                # rename the files
                # check if local crinez exists (possibly made by compress_local_copyto)
                if os.path.isfile(self.crinez_path):
                    move(self.crinez_path, os.path.join(self.rootdir, crinez))

                self.crinez_path = os.path.join(self.rootdir, crinez)

                # rename the local copy of the origin file (if exists)
                # only cases that need to be renamed (again, IF they exist; they shouldn't, but just in case)
                # are RINEZ and CRINEX since RINEX and CRINEZ are renamed above
                if os.path.isfile(self.local_copy):
                    if self.origin_type is TYPE_RINEZ:
                        local = os.path.basename(self.to_format(new_name, TYPE_RINEZ))
                        move(self.local_copy, os.path.join(self.rootdir, local))
                    elif self.origin_type is TYPE_CRINEX:
                        local = os.path.basename(self.to_format(new_name, TYPE_CRINEX))
                        move(self.local_copy, os.path.join(self.rootdir, local))

                self.crinez = crinez
                self.rinex = rinex

                self.log_event('RINEX/CRINEZ renamed to %s' % rinex)

                # update the database dictionary record
                self.record['Filename'] = self.rinex

        # we don't touch the metadata StationCode and NetworkCode unless explicitly passed
        if NetworkCode:
            self.NetworkCode = NetworkCode.strip().lower()
            self.record['NetworkCode'] = NetworkCode.strip().lower()

        if StationCode:
            self.StationCode = StationCode.strip().lower()
            self.record['StationCode'] = StationCode.strip().lower()

        return

    @staticmethod
    def identify_type(filename):

        # get the type of file passed
        filename = os.path.basename(filename)

        if filename.endswith('d.Z'):
            return TYPE_CRINEZ
        elif filename.endswith('o'):
            return TYPE_RINEX
        elif filename.endswith('o.Z'):
            return TYPE_RINEZ
        elif filename.endswith('d'):
            return TYPE_CRINEX
        else:
            raise pyRinexException('Invalid filename format: ' + filename)

    def to_format(self, filename, to_type):

        path = os.path.dirname(filename)
        filename = os.path.basename(filename)
        type = self.identify_type(filename)

        if type in (TYPE_RINEX, TYPE_CRINEX):
            filename = filename[0:-1]
        elif type in (TYPE_CRINEZ, TYPE_RINEZ):
            filename = filename[0:-3]
        else:
            raise pyRinexException('Invalid filename format: ' + filename)

        # join the path to the file again
        filename = os.path.join(path, filename)

        if to_type is TYPE_CRINEX:
            return filename + 'd'
        elif to_type is TYPE_CRINEZ:
            return filename + 'd.Z'
        elif to_type is TYPE_RINEX:
            return filename + 'o'
        elif to_type is TYPE_RINEZ:
            return filename + 'o.Z'
        else:
            raise pyRinexException(
                'Invalid to_type format. Accepted formats: CRINEX (.??d), CRINEZ (.??d.Z), RINEX (.??o) and RINEZ (.??o.Z)')

    def cleanup(self):
        if self.rinex_path and not self.no_cleanup:
            # remove all the directory contents
            try:
                rmtree(self.rootdir)
            except OSError:
                # something was not found, ignore (we are deleting anyways)
                pass

            # if it's a multiday rinex, delete the multiday objects too
            if self.multiday:
                for Rnx in self.multiday_rnx_list:
                    Rnx.cleanup()

        return

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self

    def __add__(self, other):

        if not isinstance(other, ReadRinex):
            raise pyRinexException('type: ' + type(other) + ' invalid.  Can only splice two RINEX objects.')

        if self.StationCode != other.StationCode:
            raise pyRinexException('Cannot splice together two different stations!')

        # determine which one goes first
        if other.datetime_firstObs > self.datetime_firstObs:
            f1 = self
            f2 = other
        else:
            f1 = other
            f2 = self

        # now splice files
        cmd = pyRunWithRetry.RunCommand('teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 +obs %s.t %s %s' % (
        f1.rinex_path, f1.rinex_path, f2.rinex_path), 5)

        # leave errors un-trapped on purpose (will raise an error to the parent)
        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            filename = Utils.move(f1.rinex_path + '.t', f1.rinex_path)
            return ReadRinex(self.NetworkCode, self.StationCode, filename, allow_multiday=True)
        else:
            raise pyRinexException(err)

    def __repr__(self):
        return 'pyRinex.ReadRinex(' + self.NetworkCode + ', ' + self.StationCode + ', ' + str(
            self.date.year) + ', ' + str(self.date.doy) + ')'


def main():
    # for testing purposes
    rnx = ReadRinex('RNX', 'chac', 'chac0010.17o')


if __name__ == '__main__':
    main()
"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez
"""

import uuid
from shutil import copyfile
from shutil import move
from shutil import rmtree

import pyRunWithRetry
import pyStationInfo


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


class pyRinexException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class pyRinexExceptionBadFile(pyRinexException):
    pass


class pyRinexExceptionSingleEpoch(pyRinexException):
    pass


class pyRinexExceptionNoAutoCoord(pyRinexException):
    pass


class RinexRecord():

    def __init__(self):
        self.firstObs = None
        self.lastObs = None
        self.antType = None
        self.recType = None
        self.recNo = None
        self.recVers = None
        self.antNo = None
        self.antDome = None
        self.antOffset = None
        self.interval = None
        self.size = None
        self.x = None
        self.y = None
        self.z = None
        self.lat = None
        self.lon = None
        self.h = None
        self.date = None
        self.rinex = None
        self.StationCode = None
        self.NetworkCode = None
        self.no_cleanup = None
        self.multiday = False
        self.multiday_rnx_list = []
        self.epochs = None
        self.completion = None
        self.rel_completion = None
        self.rinex_version = None

        fieldnames = ['NetworkCode', 'StationCode', 'ObservationYear', 'ObservationMonth', 'ObservationDay',
                      'ObservationDOY', 'ObservationFYear', 'ObservationSTime', 'ObservationETime', 'ReceiverType',
                      'ReceiverSerial', 'ReceiverFw', 'AntennaType', 'AntennaSerial', 'AntennaDome', 'Filename',
                      'Interval',
                      'AntennaOffset', 'Completion']

        self.record = dict.fromkeys(fieldnames)

    def load_record(self):
        self.record['NetworkCode'] = self.NetworkCode
        self.record['StationCode'] = self.StationCode
        self.record['ObservationYear'] = self.date.year
        self.record['ObservationMonth'] = self.date.month
        self.record['ObservationDay'] = self.date.day
        self.record['ObservationDOY'] = self.date.doy
        self.record['ObservationFYear'] = self.date.fyear
        self.record['ObservationSTime'] = self.firstObs
        self.record['ObservationETime'] = self.lastObs
        self.record['ReceiverType'] = self.recType
        self.record['ReceiverSerial'] = self.recNo
        self.record['ReceiverFw'] = self.recVers
        self.record['AntennaType'] = self.antType
        self.record['AntennaSerial'] = self.antNo
        self.record['AntennaDome'] = self.antDome
        self.record['Filename'] = self.rinex
        self.record['Interval'] = self.interval
        self.record['AntennaOffset'] = self.antOffset
        self.record['Completion'] = self.completion


class ReadRinex(RinexRecord):

    def read_fields(self, line, data, format_tuple):

        # create the parser object
        formatstr = re.sub(r'\..', '',
                           ' '.join(format_tuple).replace('%', '').replace('f', 's').replace('i', 's').replace('-', ''))

        fs = struct.Struct(formatstr)
        parse = fs.unpack_from

        if len(data) < fs.size:
            # line too short, add padding zeros
            f = '%-' + str(fs.size) + 's'
            data = f % line
        elif len(data) > fs.size:
            # line too long! cut
            data = line[0:fs.size]

        fields = list(parse(data))

        # convert each element in the list to float if necessary
        for i, field in enumerate(fields):
            if 'f' in format_tuple[i]:
                try:
                    fields[i] = float(fields[i])
                except ValueError:
                    # invalid number in the field!, replace with something harmless
                    fields[i] = float(2.11)
            elif 'i' in format_tuple[i]:
                try:
                    fields[i] = int(fields[i])
                except ValueError:
                    # invalid number in the field!, replace with something harmless
                    fields[i] = int(1)
            elif 's' in format_tuple[i]:
                fields[i] = fields[i].strip()

        return fields

    def check_interval(self):

        interval_record = {'INTERVAL': [('%10.3f',), False, (30,)]}

        header = self.get_header()
        new_header = []

        for line in header:

            if any(key in line for key in list(interval_record.keys())):
                # get the first occurrence only!
                record = [key for key in list(interval_record.keys()) if key in line][0]

                # get the data section by spliting the line using the record text
                data = line.split(record)[0]
                interval_record[record][1] = True

                fields = self.read_fields(line, data, interval_record[record][0])

                if fields[0] != self.interval:
                    # interval not equal. Replace record
                    data = ''.join(interval_record[record][0]) % self.interval
                    data = '%-60s' % data + record

                    new_header += [data + '\n']
                else:
                    # record matches, leave it untouched
                    new_header += [line]
            else:
                # not a critical field, just put it back in
                if not 'END OF HEADER' in line:
                    # leave END OF HEADER until the end to add possible missing records
                    new_header += [line]

        # now check that all the records where included! there's missing ones, then force them
        if not interval_record['INTERVAL'][1]:
            data = ''.join(interval_record['INTERVAL'][0]) % self.interval
            data = '%-60s' % data + 'INTERVAL'
            new_header = new_header + [data + '\n']
            data = '%-60s' % 'pyRinex: WARN! added interval to fix file!' + 'COMMENT'
            new_header = new_header + [data + '\n']

        new_header += [''.ljust(60, ' ') + 'END OF HEADER\n']

        if new_header != header:
            try:
                with open(self.rinex_path, 'r') as fileio:
                    rinex = fileio.readlines()
            except Exception:
                raise

            if not any("END OF HEADER" in s for s in rinex):
                raise pyRinexExceptionBadFile('Invalid header: could not find END OF HEADER tag.')

            # find the end of header
            index = [i for i, item in enumerate(rinex) if 'END OF HEADER' in item][0]
            # delete header
            del rinex[0:index + 1]
            # add new header
            rinex = new_header + rinex

            try:
                f = open(self.rinex_path, 'w')
                f.writelines(rinex)
                f.close()
            except Exception:
                raise

        return

    def check_header(self):

        header = self.get_header()

        # list of required header records and a flag to know if they were found or not in the current header
        # also, have a tuple of default values in case there is a missing record
        required_records = {'RINEX VERSION / TYPE': [('%9.2f', '%11s', '%1s', '%19s', '%1s', '%19s'), False, ('',)],
                            'PGM / RUN BY / DATE': [('%-20s', '%-20s', '%-20s'), False,
                                                    ('pyRinex: 1.00 000', 'Parallel.PPP', '21FEB17 00:00:00')],
                            'MARKER NAME': [('%-60s',), False, (self.StationCode,)],
                            'MARKER NUMBER': [('%-20s',), False, (self.StationCode,)],
                            'OBSERVER / AGENCY': [('%-20s', '%-40s'), False, ('UNKNOWN', 'UNKNOWN')],
                            'REC # / TYPE / VERS': [('%-20s', '%-20s', '%-20s'), False,
                                                    ('LP00785', 'ASHTECH Z-XII3', 'CC00')],
                            'ANT # / TYPE': [('%-20s', '%-20s'), False, ('12129', 'ASH700936C_M SNOW')],
                            'ANTENNA: DELTA H/E/N': [('%14.4f', '%14.4f', '%14.4f'), False,
                                                     (float(0), float(0), float(0))],
                            'APPROX POSITION XYZ': [('%14.4f', '%14.4f', '%14.4f'), False,
                                                    (float(0), float(0), float(6371000))],
                            # '# / TYPES OF OBSERV' : [('%6i',), False, ('',)],
                            'TIME OF FIRST OBS': [('%6i', '%6i', '%6i', '%6i', '%6i', '%13.7f', '%8s'), False,
                                                  (1, 1, 1, 1, 1, 0, 'GPS')],
                            # DDG: remove time of last observation all together. It just creates problems and is not mandatory
                            # 'TIME OF LAST OBS'    : [('%6i','%6i','%6i','%6i','%6i','%13.7f','%8s'), True, (int(first_obs.year), int(first_obs.month), int(first_obs.day), int(23), int(59), float(59), 'GPS')],
                            'COMMENT': [('%60s',), True, ('',)]}

        new_header = []
        system = ''
        # print ''.join(header)
        for line in header:

            if any(key in line for key in list(required_records.keys())):
                # get the first occurrence only!
                record = [key for key in list(required_records.keys()) if key in line][0]

                # mark the record as found
                required_records[record][1] = True

                # get the data section by spliting the line using the record text
                data = line.split(record)[0]

                fields = self.read_fields(line, data, required_records[record][0])

                if record == 'RINEX VERSION / TYPE':
                    # read the information about the RINEX type
                    # save the system to use during TIME OF FIRST OBS
                    system = fields[4].strip()

                    self.rinex_version = float(fields[0])

                    # now that we know the version, we can get the first obs
                    first_obs = self.get_firstobs()

                    if first_obs is None:
                        raise pyRinexExceptionBadFile(
                            'Could not find a first observation in RINEX file. Truncated file? Header follows:\n' + ''.join(
                                header))

                    if not system in (' ', 'G', 'R', 'S', 'E', 'M'):
                        # assume GPS
                        system = 'G'
                        fields[4] = 'G'

                else:
                    # reformat the header line
                    if record == 'TIME OF FIRST OBS' or record == 'TIME OF LAST OBS':
                        if system == 'M' and not fields[6].strip():
                            fields[6] = 'GPS'
                        # check if the first observation is meaningful or not
                        if record == 'TIME OF FIRST OBS' and (fields[0] != first_obs.year or
                                                              fields[1] != first_obs.month or
                                                              fields[2] != first_obs.day or
                                                              fields[3] != first_obs.hour or
                                                              fields[4] != first_obs.minute or
                                                              fields[5] != first_obs.second):
                            # bad first observation! replace with the real one
                            fields[0] = first_obs.year
                            fields[1] = first_obs.month
                            fields[2] = first_obs.day
                            fields[3] = first_obs.hour
                            fields[4] = first_obs.minute
                            fields[5] = first_obs.second

                # if record == '# / TYPES OF OBSERV':
                # re-read this time with the correct number of fields
                #    required_records[record][0] += ('%6s',)*fields[0]
                #    fields = self.read_fields(line, data, required_records[record][0])

                # regenerate the fields
                data = ''.join(required_records[record][0]) % tuple(fields)
                data = '%-60s' % data + record

                # save to new header
                new_header += [data + '\n']
            else:
                # not a critical field, just put it back in
                if ('END OF HEADER' not in line) and ('TIME OF LAST OBS' not in line):
                    # leave END OF HEADER until the end to add possible missing records
                    new_header += [line]

        if system == '':
            # if we are out of the loop and we could not determine the system, raise error
            raise pyRinexExceptionBadFile('Unfixable RINEX header: could not find RINEX VERSION / TYPE')

        # now check that all the records where included! there's missing ones, then force them
        if not all([item[1] for item in list(required_records.values())]):
            # get the keys of the missing records
            missing_records = {item[0]: item[1] for item in list(required_records.items()) if item[1][1] == False}

            for record in missing_records:
                if '# / TYPES OF OBSERV' in record:
                    raise pyRinexExceptionBadFile('Unfixable RINEX header: could not find # / TYPES OF OBSERV')

                data = ''.join(missing_records[record][0]) % missing_records[record][2]
                data = '%-60s' % data + record
                new_header = new_header + [data + '\n']
                data = '%-60s' % 'pyRinex: WARN! dummy record inserted to fix file!' + 'COMMENT'
                new_header = new_header + [data + '\n']

        new_header += [''.ljust(60, ' ') + 'END OF HEADER\n']
        # print ''.join(new_header)
        if new_header != header:

            try:
                with open(self.rinex_path, 'r') as fileio:
                    rinex = fileio.readlines()
            except Exception:
                raise

            if not any("END OF HEADER" in s for s in rinex):
                raise pyRinexExceptionBadFile('Invalid header: could not find END OF HEADER tag.')

            # find the end of header
            index = [i for i, item in enumerate(rinex) if 'END OF HEADER' in item][0]
            # delete header
            del rinex[0:index + 1]
            # add new header
            rinex = new_header + rinex

            try:
                f = open(self.rinex_path, 'w')
                f.writelines(rinex)
                f.close()
            except Exception:
                raise

    def __init__(self, NetworkCode, StationCode, origin_file, no_cleanup=False):
        """
        pyRinex initialization
        if file is multiday, DO NOT TRUST date object for initial file. Only use pyRinex objects contained in the multiday list
        """
        RinexRecord.__init__(self)

        self.StationCode = StationCode
        self.NetworkCode = NetworkCode
        self.no_cleanup = no_cleanup

        self.origin_file = origin_file

        # check that the rinex file name is valid!
        if not Utils.parse_crinex_rinex_filename(os.path.basename(origin_file)):
            raise pyRinexException(
                'File name does not follow the RINEX/CRINEX naming convention: %s' % (os.path.basename(origin_file)))

        self.rootdir = os.path.join('production', 'rinex')
        self.rootdir = os.path.join(self.rootdir, str(uuid.uuid4()))

        try:
            # create a production folder to analyze the rinex file
            if not os.path.exists(self.rootdir):
                os.makedirs(self.rootdir)
        # except OSError as e:
        #    # folder exists from a concurring instance, ignore the error
        #    sys.exc_clear()
        except Exception:
            raise

        # get the crinex and rinex names
        self.crinex = origin_file.split('/')[-1]

        if self.crinex.endswith('d.Z'):
            self.rinex = self.crinex.replace('d.Z', 'o')
            run_crz2rnx = True
        else:
            # file is not compressed, rinex = crinex
            # I should also add a condition to open just hatanaked files
            self.rinex = self.crinex
            # create the crinex name even if we got a rinex as the input file
            self.crinex = self.crinex_from_rinex(self.crinex)
            run_crz2rnx = False

        # get the paths
        self.crinex_path = os.path.join(self.rootdir, self.crinex)
        self.rinex_path = os.path.join(self.rootdir, self.rinex)

        # copy the rinex file from the archive
        try:
            # copy the file. If the origin name if a crinex, use crinex_path as destiny
            # if it's a rinex, use rinex_path as destiny
            if origin_file.endswith('d.Z'):
                copyfile(origin_file, self.crinex_path)
            else:
                copyfile(origin_file, self.rinex_path)
        except Exception:
            raise

        if run_crz2rnx:

            crinex_size = os.path.getsize(self.crinex_path)

            # run crz2rnx with timeout structure
            cmd = pyRunWithRetry.RunCommand('crz2rnx -f -d ' + self.crinex_path, 30)
            try:
                _, err = cmd.run_shell()
            except pyRunWithRetry.RunCommandWithRetryExeception as e:
                # catch the timeout except and pass it as a pyRinexException
                raise pyRinexException(str(e))
            except Exception:
                raise

            # the uncompressed-unhatanaked file size must be at least > than the crinex
            if os.path.isfile(self.rinex_path):
                if err and os.path.getsize(self.rinex_path) <= crinex_size:
                    raise pyRinexExceptionBadFile(
                        "Error in ReadRinex.__init__ -- crz2rnx: error and empty file: " + self.origin_file + ' -> ' + err)
            else:
                if err:
                    raise pyRinexException('Could not create RINEX file. crz2rnx stderr follows: ' + err)
                else:
                    raise pyRinexException(
                        'Could not create RINEX file. Unknown reason. Possible problem with crz2rnx?')

        # check basic infor in the rinex header to avoid problems with RinSum
        self.check_header()

        if self.rinex_version >= 3:
            # most programs still don't support RINEX 3 (partially implemented in this code)
            # convert to RINEX 2.11 using RinEdit
            cmd = pyRunWithRetry.RunCommand('RinEdit --IF %s --OF %s.t --ver2' % (self.rinex, self.rinex), 15,
                                            self.rootdir)

            try:
                out, _ = cmd.run_shell()

                if 'exception' in out.lower():
                    raise pyRinexExceptionBadFile('RinEdit returned error converting to RINEX 2.11:\n' + out)

                if not os.path.exists(self.rinex_path + '.t'):
                    raise pyRinexExceptionBadFile('RinEdit failed to convert to RINEX 2.11:\n' + out)

                # if all ok, move converted file to rinex_path
                os.remove(self.rinex_path)
                move(self.rinex_path + '.t', self.rinex_path)
                # change version
                self.rinex_version = 2.11

            except pyRunWithRetry.RunCommandWithRetryExeception as e:
                # catch the timeout except and pass it as a pyRinexException
                raise pyRinexException(str(e))

        # run RinSum to get file information
        cmd = pyRunWithRetry.RunCommand('RinSum --notable ' + self.rinex_path, 45)  # DDG: increased from 21 to 45.
        try:
            out, _ = cmd.run_shell()
        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise pyRinexException(str(e))
        except Exception:
            raise

        # write RinSum output to a log file (debug purposes)
        info = open(self.rinex_path + '.log', 'w')
        info.write(out)
        info.close()

        # process the output
        self.process(out)

        # DDG: after process(out), interval should be numeric
        if self.interval is None:
            raise pyRinexExceptionBadFile(
                'RINEX sampling interval could not be determined. The output from RinSum was:\n' + out)
        elif self.interval > 120:
            raise pyRinexExceptionBadFile('RINEX sampling interval > 120s. The output from RinSum was:\n' + out)
        else:
            if self.epochs * self.interval < 3600:
                raise pyRinexExceptionBadFile(
                    'RINEX file with < 1 hr of observation time. The output from RinSum was:\n' + out)

        # DDG: new interval checking after running RinSum
        # check the sampling interval
        self.check_interval()

        if (not self.firstObs or not self.lastObs):
            # first and lastobs cannot be None
            raise pyRinexException(
                self.rinex_path + ': error in ReadRinex.process: the output for first/last obs is empty. The output from RinSum was:\n' + out)
        else:
            # rinsum return dates that are 19xx (1916, for example) when they should be 2016
            # also, some dates from 199x are reported as 0091!
            # handle data format problems seen in the Cluster (only)

            if int(self.firstObs.split('/')[0]) - 1900 < 80 and int(self.firstObs.split('/')[0]) >= 1900:
                # wrong date
                self.firstObs = self.firstObs.replace(self.firstObs.split('/')[0],
                                                      str(int(self.firstObs.split('/')[0]) - 1900 + 2000))

            elif int(self.firstObs.split('/')[0]) < 1900 and int(self.firstObs.split('/')[0]) >= 80:

                self.firstObs = self.firstObs.replace(self.firstObs.split('/')[0],
                                                      str(int(self.firstObs.split('/')[0]) + 1900))

            elif int(self.firstObs.split('/')[0]) < 1900 and int(self.firstObs.split('/')[0]) < 80:

                self.firstObs = self.firstObs.replace(self.firstObs.split('/')[0],
                                                      str(int(self.firstObs.split('/')[0]) + 2000))

            if int(self.lastObs.split('/')[0]) - 1900 < 80 and int(self.lastObs.split('/')[0]) >= 1900:
                # wrong date
                self.lastObs = self.lastObs.replace(self.lastObs.split('/')[0],
                                                    str(int(self.lastObs.split('/')[0]) - 1900 + 2000))

            elif int(self.lastObs.split('/')[0]) < 1900 and int(self.lastObs.split('/')[0]) >= 80:

                self.lastObs = self.lastObs.replace(self.lastObs.split('/')[0],
                                                    str(int(self.lastObs.split('/')[0]) + 1900))

            elif int(self.lastObs.split('/')[0]) < 1900 and int(self.lastObs.split('/')[0]) < 80:

                self.lastObs = self.lastObs.replace(self.lastObs.split('/')[0],
                                                    str(int(self.lastObs.split('/')[0]) + 2000))

            try:
                self.datetime_firstObs = datetime.datetime.strptime(self.firstObs, '%Y/%m/%d %H:%M:%S')
                self.datetime_lastObs = datetime.datetime.strptime(self.lastObs, '%Y/%m/%d %H:%M:%S')
            except ValueError:
                self.datetime_firstObs = datetime.datetime.strptime(self.firstObs, '%y/%m/%d %H:%M:%S')
                self.datetime_lastObs = datetime.datetime.strptime(self.lastObs, '%y/%m/%d %H:%M:%S')
            except Exception:
                raise

            # check for files that have more than one day inside (yes, there are some like this... amazing)
            # condition is: the start and end date don't match AND
            # either there is more than two hours in the second day OR
            # there is more than one day of data
            if self.datetime_lastObs.date() != self.datetime_firstObs.date() and \
                    (self.datetime_lastObs.time().hour > 2 or (
                            self.datetime_lastObs - self.datetime_firstObs).days > 1):
                # the file has more than one day in it...
                # use teqc to window the data
                self.tbin()
                if len(self.multiday_rnx_list) > 1:
                    # truly a multiday file
                    self.multiday = True
                elif len(self.multiday_rnx_list) == 1:
                    # maybe one of the files has a single epoch in it. Drop the current rinex and use the binned version
                    self.cleanup()
                    temp_path = self.multiday_rnx_list[0].rootdir
                    # set to no cleanup so that the files survive the __init__ statement
                    self.multiday_rnx_list[0].no_cleanup = True
                    # reinitialize self
                    self.__init__(self.multiday_rnx_list[0].NetworkCode, self.multiday_rnx_list[0].StationCode,
                                  self.multiday_rnx_list[0].rinex_path)
                    # the origin file should still be the rinex passed to init the object, not the multiday file
                    self.origin_file = origin_file
                    # remove the temp directory
                    rmtree(temp_path)
                    # now self points the the binned version of the rinex
                    return

            # reported date for this file is session/2
            self.date = pyDate.Date(
                datetime=self.datetime_firstObs + (self.datetime_lastObs - self.datetime_firstObs) / 2)

            self.firstObs = self.datetime_firstObs.strftime('%Y/%m/%d %H:%M:%S')
            self.lastObs = self.datetime_lastObs.strftime('%Y/%m/%d %H:%M:%S')

            if self.datetime_lastObs <= self.datetime_firstObs:
                # bad rinex! first obs > last obs
                raise pyRinexExceptionBadFile(
                    'Last observation (' + self.lastObs + ') <= first observation (' + self.firstObs + ')')

            # DDG: calculate the completion of the file (at sampling rate)
            # completion of day
            self.completion = self.epochs * self.interval / 86400
            # completion of time window in file

            self.rel_completion = self.epochs * self.interval / (
                        (self.datetime_lastObs - self.datetime_firstObs).total_seconds() + self.interval)

            # load the RinexRecord class
            self.load_record()

        return

    def tbin(self):

        # run in the local folder to get the files inside rootdir
        cmd = pyRunWithRetry.RunCommand('teqc -tbin 1d rnx ' + self.rinex, 45, self.rootdir)
        try:
            _, err = cmd.run_shell()
        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise pyRinexException(str(e))
        except Exception:
            raise

        # successfully tbinned the file
        # delete current file and rename the new files
        os.remove(self.rinex_path)

        # now we should have as many files named rnxDDD0.??o as there where inside the RINEX
        for file in os.listdir(self.rootdir):
            if file.endswith('o') and file[0:3] == 'rnx':
                # rename file
                move(os.path.join(self.rootdir, file),
                     os.path.join(self.rootdir, file.replace('rnx', self.StationCode)))
                # get the info for this file
                try:
                    rnx = ReadRinex(self.NetworkCode, self.StationCode,
                                    os.path.join(self.rootdir, file.replace('rnx', self.StationCode)))
                    # append this rinex object to the multiday list
                    self.multiday_rnx_list.append(rnx)
                except (pyRinexException, pyRinexExceptionBadFile):
                    # there was a problem with one of the multiday files. Do not append
                    pass

        return

    def process(self, output):

        for line in output.split('\n'):
            if r'Rec#:' in line:
                self.recNo = find_between(line, 'Rec#: ', 'Type:').replace(',', '').strip()
                self.recType = find_between(line, 'Type:', 'Vers:').replace(',', '').strip()
                try:
                    self.recVers = line.split('Vers:')[1].strip()
                except Exception:
                    self.recVers = ''

            if r'Antenna # :' in line:
                self.antNo = find_between(line, 'Antenna # : ', 'Type :').replace(',', '').strip()
                try:
                    self.antType = line.split('Type :')[1].strip()
                    if ' ' in self.antType:
                        self.antDome = self.antType.split(' ')[-1]
                        self.antType = self.antType.split(' ')[0]
                    else:
                        self.antDome = 'NONE'
                except Exception:
                    self.antType = ''
                    self.antDome = ''

            if r'Antenna Delta (HEN,m) :' in line:
                try:
                    self.antOffset = float(find_between(line, 'Antenna Delta (HEN,m) : (', ',').strip())
                except Exception:
                    self.antOffset = 0  # DDG: antenna offset default value set to zero, not to []

            if r'Computed interval' in line and r'Warning' not in line:
                # added condition that skips a warning stating that header does not agree with computed interval
                try:
                    self.interval = float(find_between(line, 'Computed interval', 'seconds.').strip())
                except Exception:
                    self.interval = 0

                self.epochs = [int(find_between(xline, 'There were', 'epochs').strip()) for xline in output.split('\n')
                               if 'There were' in xline]
                if not self.epochs:
                    self.epochs = 0
                else:
                    self.epochs = self.epochs[0]

                if self.interval == 0:
                    # maybe single epoch or bad file. Raise an error
                    if self.epochs > 0:
                        raise pyRinexExceptionSingleEpoch(
                            'RINEX interval equal to zero. Single epoch or bad RINEX file. Reported epochs in file were %s' % (
                                self.epochs))
                    else:
                        raise pyRinexExceptionSingleEpoch(
                            'RINEX interval equal to zero. Single epoch or bad RINEX file. No epoch information to report. The output from RinSum was:\n' + output)

            if r'Computed first epoch:' in line:
                self.firstObs = find_between(line, 'Computed first epoch:', '=').strip()

            if r'Computed last  epoch:' in line:
                self.lastObs = find_between(line, 'Computed last  epoch:', '=').strip()

            if r'Computed file size: ' in line:
                self.size = find_between(line, 'Computed file size:', 'bytes.').strip()

            if r'Warning : Failed to read header: text 0:Incomplete or invalid header' in line:
                # there is a warning in the output, save it
                raise pyRinexException("Warning in ReadRinex.process: " + line)

            if r'unexpected exception' in line:
                raise pyRinexException("unexpected exception in ReadRinex.process: " + line)

            if r'Exception:' in line:
                raise pyRinexException("Exception in ReadRinex.process: " + line)

            if r'no data found. Are time limits wrong' in line:
                raise pyRinexException(
                    'RinSum no data found. Are time limits wrong for file ' + self.rinex + ' details:' + line)

        # remove non-utf8 chars
        if self.recNo:
            self.recNo = re.sub(r'[^\x00-\x7f]+', '', self.recNo).strip()
        if self.recType:
            self.recType = re.sub(r'[^\x00-\x7f]+', '', self.recType).strip()
        if self.recVers:
            self.recVers = re.sub(r'[^\x00-\x7f]+', '', self.recVers).strip()
        if self.antNo:
            self.antNo = re.sub(r'[^\x00-\x7f]+', '', self.antNo).strip()
        if self.antType:
            self.antType = re.sub(r'[^\x00-\x7f]+', '', self.antType).strip()
        if self.antDome:
            self.antDome = re.sub(r'[^\x00-\x7f]+', '', self.antDome).strip()

    def get_firstobs(self):

        if self.rinex_version < 3:
            fs = struct.Struct('1s2s1s2s1s2s1s2s1s2s11s2s1s3s')
        else:
            fs = struct.Struct('2s4s1s2s1s2s1s2s1s2s11s2s1s3s')

        parse = fs.unpack_from

        date = None
        with open(self.rinex_path, 'r') as fileio:

            found = False
            for line in fileio:
                if 'END OF HEADER' in line:
                    found = True
                    break

            if found:
                skip = 0
                for line in fileio:
                    if skip == 0:
                        fields = list(parse(line))

                        if int(fields[12]) <= 1:  # OK FLAG
                            # read first observation
                            year = int(fields[1])
                            month = int(fields[3])
                            day = int(fields[5])
                            hour = int(fields[7])
                            minute = int(fields[9])
                            second = float(fields[10])

                            try:
                                date = pyDate.Date(year=year, month=month, day=day, hour=hour, minute=minute,
                                                   second=second)
                            except pyDate.pyDateException as e:
                                raise pyRinexExceptionBadFile(str(e))

                            break
                        elif int(fields[12]) > 1:
                            # event, skip lines indicated in next field
                            skip = int(fields[13])
                    else:
                        skip -= 1

        return date

    def get_header(self):

        header = []
        # retry reading. Every now and then there is a problem during file read.
        for i in range(2):
            try:
                with open(self.rinex_path, 'r') as fileio:

                    for line in fileio:
                        header.append(line)
                        if 'END OF HEADER' in line:
                            break
                    break
            except IOError:
                # try again
                if i == 0:
                    continue
                else:
                    raise

        return header

    def ecef2lla(self, ecefArr):
        # convert ECEF coordinates to LLA
        # test data : test_coord = [2297292.91, 1016894.94, -5843939.62]
        # expected result : -66.8765400174 23.876539914 999.998386689

        x = float(ecefArr[0])
        y = float(ecefArr[1])
        z = float(ecefArr[2])

        a = 6378137
        e = 8.1819190842622e-2

        asq = numpy.power(a, 2)
        esq = numpy.power(e, 2)

        b = numpy.sqrt(asq * (1 - esq))
        bsq = numpy.power(b, 2)

        ep = numpy.sqrt((asq - bsq) / bsq)
        p = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
        th = numpy.arctan2(a * z, b * p)

        lon = numpy.arctan2(y, x)
        lat = numpy.arctan2((z + numpy.power(ep, 2) * b * numpy.power(numpy.sin(th), 3)),
                            (p - esq * a * numpy.power(numpy.cos(th), 3)))
        N = a / (numpy.sqrt(1 - esq * numpy.power(numpy.sin(lat), 2)))
        alt = p / numpy.cos(lat) - N

        lon = lon * 180 / numpy.pi
        lat = lat * 180 / numpy.pi

        return numpy.array([lat]), numpy.array([lon]), numpy.array([alt])

    def auto_coord(self, brdc, chi_limit=3):
        # use gamit's sh_rx2apr to obtain a coordinate of the station

        # do not work with the original file. Decimate and remove other systems (to increase speed)
        try:
            dst_gen = Utils._increment_filename(self.rinex_path)
            dst = next(dst_gen)
            while os.path.isfile(dst):
                dst = next(dst_gen)  # loop until we find a "free" rinex name

            # decimate in a copy
            self.decimate(30, dst)
            # read the decimated copy
            rnx = ReadRinex(self.NetworkCode, self.StationCode, dst)
            # remove the systems
            rnx.remove_systems()
            # copy brdc orbit
            copyfile(brdc.brdc_path, os.path.join(rnx.rootdir, brdc.brdc_filename))

        except pyRinexException as e:
            # ooops, something went wrong, try with local file (without removing systems)
            rnx = self
            # raise pyRinexExceptionBadFile('During decimation or remove_systems (to run auto_coord), teqc returned: %s' + str(e))

        cmd = pyRunWithRetry.RunCommand(
            'sh_rx2apr -site ' + rnx.rinex + ' -nav ' + brdc.brdc_filename + ' -chi ' + str(chi_limit), 40, rnx.rootdir)
        # leave errors un-trapped on purpose (will raise an error to the parent)
        out, err = cmd.run_shell()

        if err != '':
            raise pyRinexExceptionNoAutoCoord(err + '\n' + out)
        else:
            # check that the Final chi**2 is < 3
            for line in out.split('\n'):
                if '* Final sqrt(chi**2/n)' in line:
                    chi = line.split()[-1]

                    if chi == 'NaN':
                        raise pyRinexExceptionNoAutoCoord('chi2 = NaN! ' + err + '\n' + out)

                    elif float(chi) < chi_limit:
                        # open the APR file and read the coordinates
                        if os.path.isfile(os.path.join(rnx.rootdir, rnx.rinex[0:4] + '.apr')):
                            with open(os.path.join(rnx.rootdir, rnx.rinex[0:4] + '.apr')) as apr:
                                line = apr.readline().split()

                                self.x = float(line[1])
                                self.y = float(line[2])
                                self.z = float(line[3])

                                self.lat, self.lon, self.h = self.ecef2lla([self.x, self.y, self.z])

                            return (float(line[1]), float(line[2]), float(line[3])), (self.lat, self.lon, self.h)

            raise pyRinexExceptionNoAutoCoord(out + '\nLIMIT FOR CHI**2 was %i' % chi_limit)

    def auto_coord_teqc(self, brdc):
        # calculate an autonomous coordinate using broadcast orbits
        # expects to find the orbits next to the file in question

        cmd = pyRunWithRetry.RunCommand('teqc +qcq -nav ' + brdc.brdc_path + ' ' + self.rinex_path, 5)
        # leave errors un-trapped on purpose (will raise an error to the parent)
        out, err = cmd.run_shell()

        if err:
            # this part needs to handle all possible outcomes of teqc
            for line in err.split('\n'):
                # find if the err is saying there was a large antenna change
                if r'! Warning ! ... antenna position change of ' in line:
                    # bad coordinate
                    change = float(find_between(line, '! Warning ! ... antenna position change of ', ' meters'))
                    if change > 100:
                        # don't trust this RINEX for coordinates!
                        return None

                if r'currently cannot deal with an applied clock offset in QC mode' in line:
                    # handle clock stuff (see rufi)
                    # remove RCV CLOCK OFFS APPL from header and rerun
                    return None

        # if no significant problem was found, continue
        for line in out.split('\n'):
            if r'  antenna WGS 84 (xyz)  :' in line:
                xyz = find_between(line, '  antenna WGS 84 (xyz)  : ', ' (m)').split(' ')
                x = xyz[0].strip();
                y = xyz[1].strip();
                z = xyz[2].strip()
                self.x = x;
                self.y = y;
                self.z = z
                return (float(x), float(y), float(z))

        return None

    def window_data(self, start=None, end=None, copyto=None):
        """
        Window the RINEX data using TEQC
        :param copyto:
        :param start: a start datetime or self.firstObs if None
        :param end: a end datetime or self.lastObs if None
        :return:
        """
        if start is None:
            start = self.datetime_firstObs

        if end is None:
            end = self.datetime_lastObs

        cmd = pyRunWithRetry.RunCommand(
            'teqc -igs -st %i%02i%02i%02i%02i%02i -e %i%02i%02i%02i%02i%02i +obs %s.t %s' % (
                start.year, start.month, start.day, start.hour, start.minute, start.second,
                end.year, end.month, end.day, end.hour, end.minute, end.second, self.rinex_path, self.rinex_path), 5)

        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            # delete the original file and replace with .t
            if copyto is None:
                os.remove(self.rinex_path)
                move(self.rinex_path + '.t', self.rinex_path)
                self.datetime_firstObs = start
                self.datetime_lastObs = end
                self.firstObs = self.datetime_firstObs.strftime('%Y/%m/%d %H:%M:%S')
                self.lastObs = self.datetime_lastObs.strftime('%Y/%m/%d %H:%M:%S')
            else:
                move(self.rinex_path + '.t', copyto)
        else:
            raise pyRinexException(err)

        return

    def decimate(self, decimate_rate, copyto=None):
        # if copy to is passed, then the decimation is done on the copy of the file, not on the current rinex.
        # other wise, decimation is done in current rinex
        if copyto is not None:
            copyfile(self.rinex_path, copyto)
        else:
            copyto = self.rinex_path
            self.interval = decimate_rate

        if self.rinex_version < 3:
            cmd = pyRunWithRetry.RunCommand('teqc -igs -O.dec %i +obs %s.t %s' % (decimate_rate, copyto, copyto), 5)
            # leave errors un-trapped on purpose (will raise an error to the parent)
        else:
            cmd = pyRunWithRetry.RunCommand('RinEdit --IF %s --OF %s.t --TN %i --TB %i,%i,%i,%i,%i,%i' % (
            os.path.basename(copyto), os.path.basename(copyto), decimate_rate, self.date.year, self.date.month,
            self.date.day, 0, 0, 0), 15, self.rootdir)
        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            # delete the original file and replace with .t
            os.remove(copyto)
            move(copyto + '.t', copyto)
        else:
            raise pyRinexException(err)

        return

    def remove_systems(self, systems=('R', 'E', 'S'), copyto=None):
        # if copy to is passed, then the system removal is done on the copy of the file, not on the current rinex.
        # other wise, system removal is done to current rinex
        if copyto is not None:
            copyfile(self.rinex_path, copyto)
        else:
            copyto = self.rinex_path

        if self.rinex_version < 3:
            rsys = '-' + ' -'.join(systems)
            cmd = pyRunWithRetry.RunCommand('teqc -igs %s +obs %s.t %s' % (rsys, copyto, copyto), 5)
        else:
            rsys = ' --DS '.join(systems)
            cmd = pyRunWithRetry.RunCommand(
                'RinEdit --IF %s --OF %s.t --DS %s' % (os.path.basename(copyto), os.path.basename(copyto), rsys), 15,
                self.rootdir)

        # leave errors un-trapped on purpose (will raise an error to the parent)
        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            # delete the original file and replace with .t
            os.remove(copyto)
            move(copyto + '.t', copyto)
        else:
            raise pyRinexException(err)

        return

    def normalize_header(self, StationInfo, brdc=None, x=None, y=None, z=None):
        assert isinstance(StationInfo, pyStationInfo.StationInfo)
        # this function gets rid of the heaer information and replaces it with the station info (trusted)
        # should be executed before calling PPP or before rebuilding the Archive

        if StationInfo.date is not None and StationInfo.date != self.date:
            raise pyRinexException(
                'The StationInfo object was initialized for a different date than that of the RINEX file')

        if StationInfo.AntennaCode is not None and StationInfo.ReceiverCode is not None:
            # make sure that there is infornation in the provided StationInfo object
            try:
                with open(self.rinex_path, 'r') as fileio:
                    rinex = fileio.readlines()
            except Exception:
                raise

            insert_comment_antcode = False
            insert_comment_antheight = False
            insert_comment_receiever = False
            del_lines = []

            # remove all comments from the header
            for i, line in enumerate(rinex):
                if line.strip().endswith('COMMENT'):
                    del_lines.append(i)
                if line.strip().endswith('END OF HEADER'):
                    break

            rinex = [i for j, i in enumerate(rinex) if j not in del_lines]

            for i, line in enumerate(rinex):
                if line.strip().endswith('ANT # / TYPE'):
                    AntNo = line[0:20].strip()
                    AntCode = line[20:35].strip()
                    AntDome = line[36:60].strip()
                    # make sure that the ANTENNA and DOME fields are correctly separated
                    # (antenna should take 15 chars and DOME should start after the 16th place)
                    # otherwise PPP won't read the ANTENNA MODEL and DOME correctly (piece of sh.t)
                    if (
                            StationInfo.AntennaCode != AntCode or StationInfo.AntennaSerial != AntNo or StationInfo.RadomeCode != AntDome):
                        del rinex[i]
                        rinex.insert(i, str(StationInfo.AntennaSerial).ljust(20) + str(StationInfo.AntennaCode).ljust(
                            15) + ' ' + str(StationInfo.RadomeCode).ljust(24) + 'ANT # / TYPE\n')
                        insert_comment_antcode = True
                    break

            if (
                    StationInfo.ReceiverCode != self.recType or StationInfo.ReceiverSerial != self.recNo or StationInfo.ReceiverVers != self.recVers):
                for i, line in enumerate(rinex):
                    if line.strip().endswith('REC # / TYPE / VERS'):
                        del rinex[i]
                        rinex.insert(i, str(StationInfo.ReceiverSerial).ljust(20) + str(StationInfo.ReceiverCode).ljust(
                            20) + str(StationInfo.ReceiverVers).ljust(20) + 'REC # / TYPE / VERS\n')
                        insert_comment_receiever = True
                        break

            if StationInfo.AntennaHeight != self.antOffset:
                for i, line in enumerate(rinex):
                    if line.strip().endswith('ANTENNA: DELTA H/E/N'):
                        del rinex[i]
                        rinex.insert(i, ("{0:.4f}".format(StationInfo.AntennaHeight).rjust(14) + "{0:.4f}".format(
                            StationInfo.AntennaEast).rjust(14) + "{0:.4f}".format(StationInfo.AntennaNorth).rjust(
                            14)).ljust(60) + 'ANTENNA: DELTA H/E/N\n')
                        insert_comment_antheight = True
                        break

            # always replace the APPROX POSITION XYZ
            if x is None and brdc is None and self.x is None:
                raise pyRinexException(
                    'Cannot normalize the header\'s APPROX POSITION XYZ without a coordinate or a valid broadcast ephemeris object')

            elif self.x is None and brdc is not None:
                self.auto_coord(brdc)

            elif x is not None:
                self.x = x
                self.y = y
                self.z = z

            for i, line in enumerate(rinex):
                if line.strip().endswith('APPROX POSITION XYZ'):
                    del rinex[i]
                    rinex.insert(i, ("{0:.4f}".format(self.x).rjust(14) + "{0:.4f}".format(self.y).rjust(
                        14) + "{0:.4f}".format(self.z).rjust(14)).ljust(60) + 'APPROX POSITION XYZ\n')
                    break

            for i, line in enumerate(rinex):
                if line.strip().endswith('END OF HEADER'):
                    if insert_comment_antcode:
                        rinex.insert(i, ('PREV ANT    #: ' + str(self.antNo)).ljust(60) + 'COMMENT\n')
                        rinex.insert(i, ('PREV ANT TYPE: ' + str(self.antType)).ljust(60) + 'COMMENT\n')
                        rinex.insert(i, ('PREV ANT RADM: ' + str(self.antDome)).ljust(60) + 'COMMENT\n')

                    if insert_comment_antheight:
                        rinex.insert(i, (
                                    'PREV DELTAS: ' + "{0:.4f}".format(self.antOffset).rjust(14) + "{0:.4f}".format(
                                0).rjust(14) + "{0:.4f}".format(0).rjust(14)).ljust(60) + 'COMMENT\n')

                    if insert_comment_receiever:
                        rinex.insert(i, ('PREV REC    #: ' + str(self.recNo)).ljust(60) + 'COMMENT\n')
                        rinex.insert(i, ('PREV REC TYPE: ' + str(self.recType)).ljust(60) + 'COMMENT\n')
                        rinex.insert(i, ('PREV REC VERS: ' + str(self.recVers)).ljust(60) + 'COMMENT\n')

                    rinex.insert(i, 'APPROX POSITION SET TO AUTONOMOUS COORDINATE'.ljust(60) + 'COMMENT\n')

                    rinex.insert(i, ('HEADER NORMALIZED BY PARALLEL.ARCHIVE ON ' + datetime.datetime.now().strftime(
                        '%Y/%m/%d %H:%M')).ljust(60) + 'COMMENT\n')
                    break

            try:
                f = open(self.rinex_path, 'w')
                f.writelines(rinex)
                f.close()
            except Exception:
                raise
        else:
            raise pyRinexException('The StationInfo object was not initialized correctly.')

        return

    def apply_file_naming_convention(self):
        """
        function to rename a file to make it consistent with the RINEX naming convention
        :return:
        """
        # is the current rinex filename valid?
        fileparts = Utils.parse_crinex_rinex_filename(self.rinex)

        if fileparts:
            doy = int(fileparts[1])
            year = int(Utils.get_norm_year_str(fileparts[3]))
        else:
            # invalid RINEX filename! Assign some values to the variables
            doy = 0
            year = 1900

        if self.record['ObservationDOY'] != doy or self.record['ObservationYear'] != year:
            # this if still remains here but we do not allow this condition any more to happen. See process_crinex_file -> if Result...
            # NO! rename the file before moving to the archive
            filename = self.StationCode + self.date.ddd() + '0.' + self.date.yyyy()[2:4] + 'o'
            # rename file
            self.rename_crinex_rinex(filename)

    def move_origin_file(self, path):
        # this function moves the ARCHIVE file out to another location indicated by path
        # it also makes sure that it doesn' overwrite any existing file

        # intelligent move (creates folder and checks for file existence)
        filename = Utils.move(self.origin_file, os.path.join(path, self.crinex))

        # to keep everything consistent, also change the local copies of the file
        self.rename_crinex_rinex(os.path.basename(filename))

        return

    def compress_local_copyto(self, path):
        # this function compresses and moves the local copy of the rinex
        # meant to be used when a multiday rinex file is encountered and we need to move it to the repository

        # compress the rinex into crinex. Make the filename
        crinex = self.crinex_from_rinex(self.rinex)

        # we make the crinex again (don't use the existing from the database) to apply any corrections
        # made during the __init__ stage. Notice the -f in rnx2crz
        cmd = pyRunWithRetry.RunCommand('rnx2crz -f ' + self.rinex_path, 45)
        try:
            _, err = cmd.run_shell()

            if os.path.getsize(os.path.join(self.rootdir, crinex)) == 0:
                raise pyRinexException(
                    'Error in compress_local_copyto: compressed version of ' + self.rinex_path + ' has zero size!')
        except pyRunWithRetry.RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise pyRinexException(str(e))
        except Exception:
            raise

        filename = Utils.copyfile(os.path.join(self.rootdir, crinex), os.path.join(path, crinex))

        return filename

    def rename_crinex_rinex(self, new_name=None, NetworkCode=None, StationCode=None):

        # function that renames the local crinex and rinex file based on the provided information
        # it also changes the variables in the object to reflect this change
        # new name can be either a d.Z or .??o

        if new_name:
            if new_name.endswith('d.Z'):
                crinex_new_name = new_name
                rinex_new_name = self.rinex_from_crinex(new_name)
            elif new_name.endswith('o'):
                crinex_new_name = self.crinex_from_rinex(new_name)
                rinex_new_name = new_name
            else:
                raise pyRinexException('%s: Invalid name for rinex or crinex file.' % (new_name))

            # rename the files
            # check if local crinex exists (possibly made by compress_local_copyto)
            if os.path.isfile(self.crinex_path):
                move(self.crinex_path, os.path.join(self.rootdir, crinex_new_name))
            # update the crinex record
            self.crinex_path = os.path.join(self.rootdir, crinex_new_name)
            self.crinex = crinex_new_name

            move(self.rinex_path, os.path.join(self.rootdir, rinex_new_name))
            self.rinex_path = os.path.join(self.rootdir, rinex_new_name)
            self.rinex = rinex_new_name

            # update the database dictionary record
            self.record['Filename'] = self.rinex

        # we don't touch the metadata StationCode and NetworkCode unless explicitly passed
        if NetworkCode:
            self.NetworkCode = NetworkCode.strip().lower()
            self.record['NetworkCode'] = NetworkCode.strip().lower()

        if StationCode:
            self.StationCode = StationCode.strip().lower()
            self.record['StationCode'] = StationCode.strip().lower()

        return

    def crinex_from_rinex(self, name):

        return name.replace(name.split('.')[-1], name.split('.')[-1].replace('o', 'd.Z'))

    def rinex_from_crinex(self, name):

        return name.replace('d.Z', 'o')

    def cleanup(self):
        if self.rinex_path and not self.no_cleanup:
            # remove all the directory contents
            try:
                rmtree(self.rootdir)
            except OSError:
                # something was not found, ignore (we are deleting anyways)
                pass

            # if it's a multiday rinex, delete the multiday objects too
            if self.multiday:
                for Rnx in self.multiday_rnx_list:
                    Rnx.cleanup()

        return

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self


def main():
    # for testing purposes
    rnx = ReadRinex('RNX', 'chac', 'chac0010.17o')


if __name__ == '__main__':
    main()

    # BACK UP OF OLD check_time_sys
    # print ''.join(new_header)
    #
    # add_time_sys = False
    # check_time_sys = False
    # add_obs_agen = True
    # add_marker_name = True
    # add_pgm_runby = True
    # replace_pgm_runby = False
    # replace_ant_type = False
    # bad_header = False
    #
    # for line in header:
    #
    #     if len(line) < 60:
    #         if 'TIME OF FIRST OBS' in line or 'TIME OF LAST OBS' in line:
    #             bad_header = True
    #
    #     if 'RINEX VERSION / TYPE' in line:
    #         if line[40:41] == 'M':
    #             # mixed system, should check for GPS in time of first obs
    #             check_time_sys = True
    #
    #     if 'TIME OF FIRST OBS' in line and check_time_sys:
    #         if line[48:51].strip() == '':
    #             add_time_sys = True
    #
    #     if 'OBSERVER / AGENCY' in line:
    #         add_obs_agen = False
    #
    #     if 'PGM / RUN BY / DATE' in line:
    #         # an error detected in some rinex files:
    #         # 04JAN100 18:03:33 GTMPGM / RUN BY / DATE
    #         # the M of GTM moves one char the PGM / RUN BY / DATE
    #         if line[60:].strip() != 'PGM / RUN BY / DATE':
    #             replace_pgm_runby = True
    #         add_pgm_runby = False
    #
    #     if 'MARKER NAME' in line:
    #         add_marker_name = False
    #
    #     if 'ANT # / TYPE' in line:
    #         if line[60:71].strip() != 'ANT # / TYPE':
    #             # bad header in some RINEX files
    #             # fix it
    #             replace_ant_type = True
    #
    # if add_time_sys or add_obs_agen or add_marker_name or add_pgm_runby or replace_pgm_runby or replace_ant_type or bad_header:
    #     try:
    #         with open(self.rinex_path, 'r') as fileio:
    #             rinex = fileio.readlines()
    #     except:
    #         raise
    #
    #     for i, line in enumerate(rinex):
    #         if len(line) < 60:
    #             # if the line is < 60 chars, replace with a bogus time and date (RinSum ignores it anyways)
    #             # but requires it to continue
    #             # notice that the code only arrives here if non-compulsory bad fields are found e.g. TIME OF FIRST OBS
    #             if 'TIME OF FIRST OBS' in line:
    #                 rinex[i] = '  2000    12    27    00    00    0.000                     TIME OF FIRST OBS\n'
    #
    #             if 'TIME OF LAST OBS' in line:
    #                 rinex[i] = '  2000    12    27    23    59   59.000                     TIME OF LAST OBS\n'
    #
    #         if 'TIME OF FIRST OBS' in line and add_time_sys:
    #             rinex[i] = line.replace('            TIME OF FIRST OBS', 'GPS         TIME OF FIRST OBS')
    #
    #         if 'PGM / RUN BY / DATE' in line and replace_pgm_runby:
    #             rinex[i] = line.replace(line,
    #                                     'pyRinex: 1.00 000   Parallel.Archive    21FEB17 00:00:00    PGM / RUN BY / DATE\n')
    #
    #         if 'ANT # / TYPE' in line and replace_ant_type:
    #             rinex[i] = rinex[i].replace(rinex[i][60:], 'ANT # / TYPE\n')
    #
    #         if 'END OF HEADER' in line:
    #             if add_obs_agen:
    #                 rinex.insert(i, 'IGN                 IGN                                     OBSERVER / AGENCY\n')
    #             if add_marker_name:
    #                 rinex.insert(i,
    #                              self.StationCode + '                                                        MARKER NAME\n')
    #             if add_pgm_runby:
    #                 rinex.insert(i, 'pyRinex: 1.00 000   Parallel.Archive    21FEB17 00:00:00    PGM / RUN BY / DATE\n')
    #             break
    #
    #     try:
    #         f = open(self.rinex_path, 'w')
    #         f.writelines(rinex)
    #         f.close()
    #     except:
    #         raise

"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez
"""

import os
import platform
import threading


class RunCommandWithRetryExeception(Exception):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class command(threading.Thread):

    def __init__(self, command, cwd=os.getcwd(), cat_file=None):
        self.stdout = None
        self.stderr = None
        self.cmd = command
        self.cwd = cwd
        self.cat_file = cat_file

        threading.Thread.__init__(self)

    def run(self):
        retry = 0
        while True:
            try:
                if self.cat_file:
                    # TODO: What is this?
                    cat = subprocess.Popen(['cat', self.cat_file], shell=False, stdout=subprocess.PIPE,
                                           cwd=self.cwd, close_fds=True, bufsize=-1, universal_newlines=True)

                    self.p = subprocess.Popen(self.cmd.split(), shell=False, stdin=cat.stdout,
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd,
                                              close_fds=True, bufsize=-1, universal_newlines=True)
                else:
                    self.p = subprocess.Popen(self.cmd.split(), shell=False, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE, cwd=self.cwd, close_fds=True,
                                              bufsize=-1, universal_newlines=True)

                self.stdout, self.stderr = self.p.communicate()
                break
            except OSError as e:
                if str(e) == '[Errno 35] Resource temporarily unavailable':
                    if retry <= 2:
                        retry += 1
                        # wait a moment
                        time.sleep(0.5)
                        continue
                    else:
                        print(self.cmd)
                        raise OSError(str(e) + ' after 3 retries on node: ' + platform.node())
                else:
                    print(self.cmd)
                    raise
            except Exception:
                print(self.cmd)
                raise

    def wait(self, timeout=None):

        self.join(timeout=timeout)
        if self.is_alive():
            try:
                self.p.kill()
            except Exception:
                # the process was done
                return False

            return True

        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.p.terminate()
        except Exception:
            pass

        self.p = None

    def __enter__(self):
        return self


class RunCommand():
    def __init__(self, command, time_out, cwd=os.getcwd(), cat_file=None):
        self.stdout = None
        self.stderr = None
        self.cmd = command
        self.time_out = time_out
        self.cwd = cwd
        self.cat_file = cat_file

    def run_shell(self):
        retry = 0
        while True:
            with command(self.cmd, self.cwd, self.cat_file) as cmd:
                cmd.start()
                timeout = cmd.wait(self.time_out)
                if timeout:
                    if retry <= 2:
                        retry += 1
                        continue
                    else:
                        raise RunCommandWithRetryExeception(
                            "Error in RunCommand.run_shell -- (" + self.cmd + "): Timeout after 3 retries")
                else:
                    # remove non-ASCII chars
                    if not cmd.stderr is None:
                        cmd.stderr = ''.join([i if ord(i) < 128 else ' ' for i in cmd.stderr])

                    return cmd.stdout, cmd.stderr


"""
Project: Parallel.Archive
Date: 2/22/17 3:27 PM
Author: Demian D. Gomez
"""

import pyProducts


class pySp3Exception(pyProducts.pyProductsException):
    pass


class GetSp3Orbits(pyProducts.OrbitalProduct):

    def __init__(self, sp3archive, date, sp3types, copyto, no_cleanup=False):

        # try both compressed and non-compressed sp3 files
        # loop through the types of sp3 files to try
        self.sp3_path = None
        self.RF = None
        self.no_cleanup = no_cleanup

        for sp3type in sp3types:
            self.sp3_filename = sp3type + date.wwwwd() + '.sp3'

            try:
                pyProducts.OrbitalProduct.__init__(self, sp3archive, date, self.sp3_filename, copyto)
                self.sp3_path = self.file_path
                self.type = sp3type
                break
            except pyProducts.pyProductsExceptionUnreasonableDate:
                raise
            except pyProducts.pyProductsException:
                # if the file was not found, go to next
                pass

        # if we get here and self.sp3_path is still none, then no type of sp3 file was found
        if self.sp3_path is None:
            raise pySp3Exception(
                'Could not find a valid orbit file (types: ' + ', '.join(sp3types) + ') for week ' + str(
                    date.gpsWeek) + ' day ' + str(date.gpsWeekDay) + ' using any of the provided sp3 types')
        else:
            # parse the RF of the orbit file
            try:
                with open(self.sp3_path, 'r') as fileio:
                    line = fileio.readline()

                    self.RF = line[46:51].strip()
            except Exception:
                raise

        return

    def cleanup(self):
        if self.sp3_path and not self.no_cleanup:
            # delete files
            if os.path.isfile(self.sp3_path):
                os.remove(self.sp3_path)

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self


"""
Project: Parallel.Archive
Date: 02/16/2017
Author: Demian D. Gomez
"""

import datetime
from json import JSONEncoder

import numpy as np
import pyBunch
import pyEvents


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default


class pyStationInfoException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class pyStationInfoHeightCodeNotFound(pyStationInfoException):
    pass


class StationInfoRecord(pyBunch.Bunch):
    def __init__(self, NetworkCode=None, StationCode=None, record=None):

        pyBunch.Bunch.__init__(self)

        self.NetworkCode = NetworkCode
        self.StationCode = StationCode
        self.ReceiverCode = ''
        self.ReceiverSerial = None
        self.ReceiverFirmware = None
        self.AntennaCode = ''
        self.AntennaSerial = None
        self.AntennaHeight = 0
        self.AntennaNorth = 0
        self.AntennaEast = 0
        self.HeightCode = ''
        self.RadomeCode = ''
        self.DateStart = None
        self.DateEnd = None
        self.ReceiverVers = None
        self.Comments = None
        self.hash = None

        if record is not None:
            self.parse_station_record(record)

        # create a hash record using the station information
        # use only the information that can actually generate a change in the antenna position
        self.hash = zlib.crc32('%.4f %.4f %.4f %s %s %s %s'.encode('utf-8') %
                               (
                               self.AntennaNorth, self.AntennaEast, self.AntennaHeight, self.HeightCode.encode('utf-8'),
                               self.AntennaCode.encode('utf-8'), self.RadomeCode.encode('utf-8'),
                               self.ReceiverCode.encode('utf-8')))

        self.record_format = ' %-4s  %-16s  %-19s%-19s%7.4f  %-5s  %7.4f  %7.4f  %-20s  ' \
                             '%-20s  %5s  %-20s  %-15s  %-5s  %-20s'

    def database(self):

        fieldnames = ['StationCode', 'NetworkCode', 'DateStart', 'DateEnd', 'AntennaHeight', 'HeightCode',
                      'AntennaNorth', 'AntennaEast', 'ReceiverCode', 'ReceiverVers', 'ReceiverFirmware',
                      'ReceiverSerial', 'AntennaCode', 'RadomeCode', 'AntennaSerial', 'Comments']

        return_fields = dict()

        for field in fieldnames:
            if field == 'DateStart':
                return_fields[field] = self[field].datetime()
            elif field == 'DateEnd':
                if self[field].year is None:
                    return_fields[field] = None
                else:
                    return_fields[field] = self[field].datetime()
            else:
                return_fields[field] = self[field]

        return return_fields

    def to_json(self):

        fields = self.database()
        fields['DateStart'] = str(self.DateStart)
        fields['DateEnd'] = str(self.DateEnd)

        return fields

    def parse_station_record(self, record):

        if isinstance(record, str):

            fieldnames = ['StationCode', 'StationName', 'DateStart', 'DateEnd', 'AntennaHeight', 'HeightCode',
                          'AntennaNorth', 'AntennaEast', 'ReceiverCode', 'ReceiverVers', 'ReceiverFirmware',
                          'ReceiverSerial', 'AntennaCode', 'RadomeCode', 'AntennaSerial']

            fieldwidths = (
                1, 6, 18, 19, 19, 9, 7, 9, 9, 22, 22, 7, 22, 17, 7,
                20)  # negative widths represent ignored padding fields
            fmtstring = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's') for fw in fieldwidths)

            fieldstruct = struct.Struct(fmtstring)
            parse = fieldstruct.unpack_from

            if record[0] == ' ' and len(record) >= 77:
                record = dict(list(zip(fieldnames, list(map(str.strip, parse(record.ljust(fieldstruct.size))[1:])))))
            else:
                return

        for key in list(self.keys()):
            try:
                if key == 'AntennaNorth' or key == 'AntennaEast' or key == 'AntennaHeight':
                    self[key] = float(record[key])
                else:
                    self[key] = record[key]
            except KeyError:
                # if key not found in the record, may be an added field (like hash)
                pass

        try:
            # if initializing with a RINEX record, some of these may not exist in the dictionary
            self.DateStart = pyDate.Date(stninfo=record['DateStart'])
            self.DateEnd = pyDate.Date(stninfo=record['DateEnd'])
            self.StationCode = record['StationCode'].lower()
        except KeyError:
            pass

    def __repr__(self):
        return 'pyStationInfo.StationInfoRecord(' + str(self) + ')'

    def __str__(self):

        return self.record_format % (self.StationCode.upper(), '', str(self.DateStart), str(self.DateEnd),
                                     self.AntennaHeight, self.HeightCode,
                                     self.AntennaNorth, self.AntennaEast,
                                     self.ReceiverCode, self.ReceiverVers,
                                     self.ReceiverFirmware, self.ReceiverSerial,
                                     self.AntennaCode, self.RadomeCode,
                                     self.AntennaSerial)


class StationInfo(object):
    """
    New parameter: h_tolerance makes the station info more tolerant to gaps. This is because station info in the old
    days had a break in the middle and the average epoch was falling right in between the gap
    """

    def __init__(self, cnn, NetworkCode=None, StationCode=None, date=None, allow_empty=False, h_tolerance=0):

        self.record_count = 0
        self.NetworkCode = NetworkCode
        self.StationCode = StationCode
        self.allow_empty = allow_empty
        self.date = None
        self.records = []
        self.currentrecord = StationInfoRecord(NetworkCode, StationCode)

        self.header = '*SITE  Station Name      Session Start      Session Stop       Ant Ht   HtCod  Ant N    ' \
                      'Ant E    Receiver Type         Vers                  SwVer  Receiver SN           ' \
                      'Antenna Type     Dome   Antenna SN          '

        # connect to the db and load the station info table
        if NetworkCode is not None and StationCode is not None:

            self.cnn = cnn

            if self.load_stationinfo_records():
                # find the record that matches the given date
                if date is not None:
                    self.date = date

                    pDate = date.datetime()

                    for record in self.records:

                        DateStart = record['DateStart'].datetime()
                        DateEnd = record['DateEnd'].datetime()

                        # make the gap-tolerant comparison
                        if DateStart - datetime.timedelta(hours=h_tolerance) <= pDate <= \
                                DateEnd + datetime.timedelta(hours=h_tolerance):
                            # found the record that corresponds to this date
                            self.currentrecord = record
                            break

                    if self.currentrecord.DateStart is None:
                        raise pyStationInfoException('Could not find a matching station.info record for ' +
                                                     NetworkCode + '.' + StationCode + ' ' +
                                                     date.yyyymmdd() + ' (' + date.yyyyddd() + ')')

    def load_stationinfo_records(self):
        # function to load the station info records in the database
        # returns true if records found
        # returns false if none found, unless allow_empty = False in which case it raises an error.
        stninfo = self.cnn.query('SELECT * FROM stationinfo WHERE "NetworkCode" = \'' + self.NetworkCode +
                                 '\' AND "StationCode" = \'' + self.StationCode + '\' ORDER BY "DateStart"')

        if stninfo.ntuples() == 0:
            if not self.allow_empty:
                # allow no station info if explicitly requested by the user.
                # Purpose: insert a station info for a new station!
                raise pyStationInfoException('Could not find ANY valid station info entry for ' +
                                             self.NetworkCode + '.' + self.StationCode)
            self.record_count = 0
            return False
        else:
            for record in stninfo.dictresult():
                self.records.append(StationInfoRecord(self.NetworkCode, self.StationCode, record))

            self.record_count = stninfo.ntuples()
            return True

    def parse_station_info(self, stninfo_file_list):
        """
        function used to parse a station information file
        :param stninfo_file_list: a station information file or list containing station info records
        :return: a list of StationInformationRecords
        """

        if isinstance(stninfo_file_list, list):
            # a list is comming in
            stninfo = stninfo_file_list
        else:
            # a file is comming in
            with open(stninfo_file_list, 'r') as fileio:
                stninfo = fileio.readlines()

        records = []
        for line in stninfo:

            if line[0] == ' ' and len(line) >= 77:
                record = StationInfoRecord(self.NetworkCode, self.StationCode, line)

                if record.DateStart is not None:
                    records.append(record)

        return records

    def to_dharp(self, record):
        """
        function to convert the current height code to DHARP
        :return: DHARP height
        """

        if record.HeightCode == 'DHARP':
            return record
        else:
            htc = self.cnn.query_float('SELECT * FROM gamit_htc WHERE "AntennaCode" = \'%s\' AND "HeightCode" = \'%s\''
                                       % (record.AntennaCode, record.HeightCode), as_dict=True)

            if len(htc):

                record.AntennaHeight = np.sqrt(np.square(float(record.AntennaHeight)) -
                                               np.square(float(htc[0]['h_offset']))) - float(htc[0]['v_offset'])
                if record.Comments is not None:
                    record.Comments = record.Comments + '\nChanged from %s to DHARP by pyStationInfo.\n' \
                                      % record.HeightCode
                else:
                    record.Comments = 'Changed from %s to DHARP by pyStationInfo.\n' % record.HeightCode

                record.HeightCode = 'DHARP'

                return record
            else:
                # TODO: Could not translate height code DHPAB to DHARP (ter.inmn: TRM59800.00).
                #  Check the height codes table.
                raise pyStationInfoHeightCodeNotFound('Could not translate height code %s to DHARP (%s.%s: %s). '
                                                      'Check the height codes table.'
                                                      % (record.HeightCode, self.NetworkCode, self.StationCode,
                                                         record.AntennaCode))

    def return_stninfo(self, record=None):
        """
        return a station information string to write to a file (without header
        :param record: to print a specific record, pass a record, otherwise, leave empty to print all records
        :return: a string in station information format
        """
        stninfo = []

        # from the records struct, return a station info file
        if record is not None:
            records = [record]
        else:
            records = self.records

        if records is not None:
            for record in records:
                stninfo.append(str(self.to_dharp(record)))

        return '\n'.join(stninfo)

    def return_stninfo_short(self, record=None):
        """
        prints a simplified version of the station information to better fit screens
        :param record: to print a specific record, pass a record, otherwise, leave empty to print all records
        :return: a string in station information format. It adds the NetworkCode dot StationCode
        """
        stninfo_lines = self.return_stninfo(record=record).split('\n')

        stninfo_lines = [' %s.%s [...] %s' % (self.NetworkCode.upper(), l[1:110], l[160:]) for l in stninfo_lines]

        return '\n'.join(stninfo_lines)

    def overlaps(self, qrecord):

        # check if the incoming record is between any existing record
        overlaps = []

        q_start = qrecord['DateStart'].datetime()
        q_end = qrecord['DateEnd'].datetime()

        if self.records:
            for record in self.records:

                r_start = record['DateStart'].datetime()
                r_end = record['DateEnd'].datetime()

                earliest_end = min(q_end, r_end)
                latest_start = max(q_start, r_start)

                if (earliest_end - latest_start).total_seconds() > 0:
                    overlaps.append(record)

        return overlaps

    def DeleteStationInfo(self, record):

        event = pyEvents.Event(Description=record['DateStart'].strftime() +
                                           ' has been deleted:\n' + str(record), StationCode=self.StationCode,
                               NetworkCode=self.NetworkCode)

        self.cnn.insert_event(event)

        self.cnn.delete('stationinfo', record.database())
        self.load_stationinfo_records()

    def UpdateStationInfo(self, record, new_record):

        # avoid problems with trying to insert records from other stations. Force this NetworkCode
        record['NetworkCode'] = self.NetworkCode
        new_record['NetworkCode'] = self.NetworkCode

        if self.NetworkCode and self.StationCode:

            # check the possible overlaps. This record will probably overlap with itself, so check that the overlap has
            # the same DateStart as the original record (that way we know it's an overlap with itself)
            overlaps = self.overlaps(new_record)

            for overlap in overlaps:
                if overlap['DateStart'].datetime() != record['DateStart'].datetime():
                    # it's overlapping with another record, raise error

                    raise pyStationInfoException('Record %s -> %s overlaps with existing station.info records: %s -> %s'
                                                 % (str(record['DateStart']), str(record['DateEnd']),
                                                    str(overlap['DateStart']), str(overlap['DateEnd'])))

            # insert event (before updating to save all information)
            event = pyEvents.Event(Description=record['DateStart'].strftime() +
                                               ' has been updated:\n' + str(new_record) +
                                               '\n+++++++++++++++++++++++++++++++++++++\n' +
                                               'Previous record:\n' +
                                               str(record) + '\n',
                                   NetworkCode=self.NetworkCode,
                                   StationCode=self.StationCode)

            self.cnn.insert_event(event)

            if new_record['DateStart'] != record['DateStart']:
                self.cnn.query('UPDATE stationinfo SET "DateStart" = \'%s\' '
                               'WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND "DateStart" = \'%s\'' %
                               (new_record['DateStart'].strftime(), self.NetworkCode,
                                self.StationCode, record['DateStart'].strftime()))

            self.cnn.update('stationinfo', new_record.database(), NetworkCode=self.NetworkCode,
                            StationCode=self.StationCode, DateStart=new_record['DateStart'].datetime())

            self.load_stationinfo_records()

    def InsertStationInfo(self, record):

        # avoid problems with trying to insert records from other stations. Force this NetworkCode
        record['NetworkCode'] = self.NetworkCode

        if self.NetworkCode and self.StationCode:
            # check existence of station in the db
            rs = self.cnn.query(
                'SELECT * FROM stationinfo WHERE "NetworkCode" = \'%s\' '
                'AND "StationCode" = \'%s\' AND "DateStart" = \'%s\'' %
                (self.NetworkCode, self.StationCode, record['DateStart'].strftime()))

            if rs.ntuples() == 0:
                # can insert because it's not the same record
                # 1) verify the record is not between any two existing records
                overlaps = self.overlaps(record)

                if overlaps:
                    # if it overlaps all records and the DateStart < self.records[0]['DateStart']
                    # see if we have to extend the initial date
                    if len(overlaps) == len(self.records) and \
                            record['DateStart'].datetime() < self.records[0]['DateStart'].datetime():
                        if self.records_are_equal(record, self.records[0]):
                            # just modify the start date to match the incoming record
                            # self.cnn.update('stationinfo', self.records[0], DateStart=record['DateStart'])
                            # the previous statement seems not to work because it updates a primary key!
                            self.cnn.query(
                                'UPDATE stationinfo SET "DateStart" = \'%s\' WHERE "NetworkCode" = \'%s\' '
                                'AND "StationCode" = \'%s\' AND "DateStart" = \'%s\'' %
                                (record['DateStart'].strftime(),
                                 self.NetworkCode, self.StationCode,
                                 self.records[0]['DateStart'].strftime()))

                            # insert event
                            event = pyEvents.Event(Description='The start date of the station information record ' +
                                                               self.records[0]['DateStart'].strftime() +
                                                               ' has been been modified to ' +
                                                               record['DateStart'].strftime(),
                                                   StationCode=self.StationCode,
                                                   NetworkCode=self.NetworkCode)
                            self.cnn.insert_event(event)
                        else:
                            # new and different record, stop the Session with
                            # EndDate = self.records[0]['DateStart'] - datetime.timedelta(seconds=1) and insert
                            record['DateEnd'] = pyDate.Date(datetime=self.records[0]['DateStart'].datetime() -
                                                                     datetime.timedelta(seconds=1))

                            self.cnn.insert('stationinfo', record.database())

                            # insert event
                            event = pyEvents.Event(
                                Description='A new station information record was added:\n'
                                            + str(record),
                                StationCode=self.StationCode,
                                NetworkCode=self.NetworkCode)

                            self.cnn.insert_event(event)

                    elif len(overlaps) == 1 and overlaps[0] == self.records[-1] and \
                            not self.records[-1]['DateEnd'].year:
                        # overlap with the last session
                        # stop the current valid session
                        self.cnn.update('stationinfo', self.records[-1].database(),
                                        DateEnd=record['DateStart'].datetime() - datetime.timedelta(seconds=1))

                        # create the incoming session
                        self.cnn.insert('stationinfo', record.database())

                        # insert event
                        event = pyEvents.Event(
                            Description='A new station information record was added:\n' +
                                        self.return_stninfo(record) +
                                        '\nThe previous DateEnd value was updated to ' +
                                        self.records[-1]['DateEnd'].strftime(),
                            StationCode=self.StationCode,
                            NetworkCode=self.NetworkCode)

                        self.cnn.insert_event(event)

                    else:
                        stroverlap = []
                        for overlap in overlaps:
                            stroverlap.append(' -> '.join([str(overlap['DateStart']), str(overlap['DateEnd'])]))

                        raise pyStationInfoException('Record %s -> %s overlaps with existing station.info records: %s'
                                                     % (str(record['DateStart']), str(record['DateEnd']),
                                                        ' '.join(stroverlap)))

                else:
                    # no overlaps, insert the record
                    self.cnn.insert('stationinfo', record.database())

                    # insert event
                    event = pyEvents.Event(Description='A new station information record was added:\n' +
                                                       str(record),
                                           StationCode=self.StationCode,
                                           NetworkCode=self.NetworkCode)
                    self.cnn.insert_event(event)

                # reload the records
                self.load_stationinfo_records()
            else:
                raise pyStationInfoException('Record %s -> %s already exists in station.info' %
                                             (str(record['DateStart']), str(record['DateEnd'])))
        else:
            raise pyStationInfoException('Cannot insert record without initializing pyStationInfo '
                                         'with NetworkCode and StationCode')

    def rinex_based_stninfo(self, ignore):
        # build a station info based on the information from the RINEX headers
        rs = self.cnn.query('SELECT * FROM rinex WHERE "NetworkCode" = \'' + self.NetworkCode +
                            '\' AND "StationCode" = \'' + self.StationCode + '\' ORDER BY "ObservationSTime"')

        rnxtbl = rs.dictresult()

        rnx = rnxtbl[0]

        RecSerial = rnx['ReceiverSerial']
        AntSerial = rnx['AntennaSerial']
        AntHeig = rnx['AntennaOffset']
        RadCode = rnx['AntennaDome']
        StartDate = rnx['ObservationSTime']

        stninfo = []
        count = 0
        for i, rnx in enumerate(rnxtbl):

            if RecSerial != rnx['ReceiverSerial'] or AntSerial != rnx['AntennaSerial'] or \
                    AntHeig != rnx['AntennaOffset'] or RadCode != rnx['AntennaDome']:
                # start the counter
                count += 1

                if count > ignore:
                    Vers = rnx['ReceiverFw'][:22]

                    record = StationInfoRecord(self.NetworkCode, self.StationCode, rnx)
                    record.DateStart = pyDate.Date(datetime=StartDate)
                    record.DateEnd = pyDate.Date(datetime=rnxtbl[i - count]['ObservationETime'])
                    record.HeightCode = 'DHARP'
                    record.ReceiverVers = Vers[:5]

                    stninfo.append(str(record))

                    RecSerial = rnx['ReceiverSerial']
                    AntSerial = rnx['AntennaSerial']
                    AntHeig = rnx['AntennaOffset']
                    RadCode = rnx['AntennaDome']
                    StartDate = rnxtbl[i - count + 1]['ObservationSTime']
                    count = 0
            elif RecSerial == rnx['ReceiverSerial'] and AntSerial == rnx['AntennaSerial'] and \
                    AntHeig == rnx['AntennaOffset'] and RadCode == rnx['AntennaDome'] and count > 0:
                # we started counting records that where different, but we didn't make it past > ignore, reset counter
                count = 0

        # insert the last record with 9999
        record = StationInfoRecord(self.NetworkCode, self.StationCode, None)
        record.DateStart = pyDate.Date(datetime=StartDate)
        record.DateEnd = pyDate.Date(stninfo=None)
        record.HeightCode = 'DHARP'

        stninfo.append(str(record))

        return '\n'.join(stninfo) + '\n'

    def to_json(self):
        return [r.to_json() for r in self.records]

    @staticmethod
    def records_are_equal(record1, record2):

        if record1['ReceiverCode'] != record2['ReceiverCode']:
            return False

        if record1['ReceiverSerial'] != record2['ReceiverSerial']:
            return False

        if record1['AntennaCode'] != record2['AntennaCode']:
            return False

        if record1['AntennaSerial'] != record2['AntennaSerial']:
            return False

        if record1['AntennaHeight'] != record2['AntennaHeight']:
            return False

        if record1['AntennaNorth'] != record2['AntennaNorth']:
            return False

        if record1['AntennaEast'] != record2['AntennaEast']:
            return False

        if record1['HeightCode'] != record2['HeightCode']:
            return False

        if record1['RadomeCode'] != record2['RadomeCode']:
            return False

        return True

    def __eq__(self, stninfo):

        if not isinstance(stninfo, StationInfo):
            raise pyStationInfoException('type: ' + str(type(stninfo))
                                         + ' is invalid. Can only compare pyStationInfo.StationInfo objects')

        if self.currentrecord.AntennaCode != stninfo.currentrecord.AntennaCode:
            return False

        if self.currentrecord.AntennaHeight != stninfo.currentrecord.AntennaHeight:
            return False

        if self.currentrecord.AntennaNorth != stninfo.currentrecord.AntennaNorth:
            return False

        if self.currentrecord.AntennaEast != stninfo.currentrecord.AntennaEast:
            return False

        if self.currentrecord.AntennaSerial != stninfo.currentrecord.AntennaSerial:
            return False

        if self.currentrecord.ReceiverCode != stninfo.currentrecord.ReceiverCode:
            return False

        if self.currentrecord.ReceiverSerial != stninfo.currentrecord.ReceiverSerial:
            return False

        if self.currentrecord.RadomeCode != stninfo.currentrecord.RadomeCode:
            return False

        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.records = None

    def __enter__(self):
        return self


import sys


class TerminalController:
    """
    A class that can be used to portably generate formatted output to
    a terminal.

    `TerminalController` defines a set of instance variables whose
    values are initialized to the control sequence necessary to
    perform a given action.  These can be simply included in normal
    output to the terminal:

        >>> term = TerminalController()
        >>> print 'This is '+term.GREEN+'green'+term.NORMAL

    Alternatively, the `render()` method can used, which replaces
    '${action}' with the string required to perform 'action':

        >>> term = TerminalController()
        >>> print term.render('This is ${GREEN}green${NORMAL}')

    If the terminal doesn't support a given action, then the value of
    the corresponding instance variable will be set to ''.  As a
    result, the above code will still work on terminals that do not
    support color, except that their output will not be colored.
    Also, this means that you can test whether the terminal supports a
    given action by simply testing the truth value of the
    corresponding instance variable:

        >>> term = TerminalController()
        >>> if term.CLEAR_SCREEN:
        ...     print 'This terminal supports clearning the screen.'

    Finally, if the width and height of the terminal are known, then
    they will be stored in the `COLS` and `LINES` attributes.
    """
    # Cursor movement:
    BOL = ''  #: Move the cursor to the beginning of the line
    UP = ''  #: Move the cursor up one line
    DOWN = ''  #: Move the cursor down one line
    LEFT = ''  #: Move the cursor left one char
    RIGHT = ''  #: Move the cursor right one char

    # Deletion:
    CLEAR_SCREEN = ''  #: Clear the screen and move to home position
    CLEAR_EOL = ''  #: Clear to the end of the line.
    CLEAR_BOL = ''  #: Clear to the beginning of the line.
    CLEAR_EOS = ''  #: Clear to the end of the screen

    # Output modes:
    BOLD = ''  #: Turn on bold mode
    BLINK = ''  #: Turn on blink mode
    DIM = ''  #: Turn on half-bright mode
    REVERSE = ''  #: Turn on reverse-video mode
    NORMAL = ''  #: Turn off all modes

    # Cursor display:
    HIDE_CURSOR = ''  #: Make the cursor invisible
    SHOW_CURSOR = ''  #: Make the cursor visible

    # Terminal size:
    COLS = None  #: Width of the terminal (None for unknown)
    LINES = None  #: Height of the terminal (None for unknown)

    # Foreground colors:
    BLACK = BLUE = GREEN = CYAN = RED = MAGENTA = YELLOW = WHITE = ''

    # Background colors:
    BG_BLACK = BG_BLUE = BG_GREEN = BG_CYAN = ''
    BG_RED = BG_MAGENTA = BG_YELLOW = BG_WHITE = ''

    _STRING_CAPABILITIES = """
    BOL=cr UP=cuu1 DOWN=cud1 LEFT=cub1 RIGHT=cuf1
    CLEAR_SCREEN=clear CLEAR_EOL=el CLEAR_BOL=el1 CLEAR_EOS=ed BOLD=bold
    BLINK=blink DIM=dim REVERSE=rev UNDERLINE=smul NORMAL=sgr0
    HIDE_CURSOR=cinvis SHOW_CURSOR=cnorm""".split()
    _COLORS = """BLACK BLUE GREEN CYAN RED MAGENTA YELLOW WHITE""".split()
    _ANSICOLORS = "BLACK RED GREEN YELLOW BLUE MAGENTA CYAN WHITE".split()

    def __init__(self, term_stream=sys.stdout):
        """
        Create a `TerminalController` and initialize its attributes
        with appropriate values for the current terminal.
        `term_stream` is the stream that will be used for terminal
        output; if this stream is not a tty, then the terminal is
        assumed to be a dumb terminal (i.e., have no capabilities).
        """
        # Curses isn't available on all platforms
        try:
            import curses
        except:
            return

        # If the stream isn't a tty, then assume it has no capabilities.
        if not term_stream.isatty(): return

        # Check the terminal type.  If we fail, then assume that the
        # terminal has no capabilities.
        try:
            curses.setupterm()
        except:
            return

        # Look up numeric capabilities.
        self.COLS = curses.tigetnum('cols')
        self.LINES = curses.tigetnum('lines')

        # Look up string capabilities.
        for capability in self._STRING_CAPABILITIES:
            (attrib, cap_name) = capability.split('=')
            setattr(self, attrib, self._tigetstr(cap_name) or '')

        # Colors
        set_fg = self._tigetstr('setf')
        if set_fg:
            for i, color in zip(list(range(len(self._COLORS))), self._COLORS):
                setattr(self, color, curses.tparm(set_fg, i) or '')
        set_fg_ansi = self._tigetstr('setaf')
        if set_fg_ansi:
            for i, color in zip(list(range(len(self._ANSICOLORS))), self._ANSICOLORS):
                setattr(self, color, curses.tparm(set_fg_ansi, i) or '')
        set_bg = self._tigetstr('setb')
        if set_bg:
            for i, color in zip(list(range(len(self._COLORS))), self._COLORS):
                setattr(self, 'BG_' + color, curses.tparm(set_bg, i) or '')
        set_bg_ansi = self._tigetstr('setab')
        if set_bg_ansi:
            for i, color in zip(list(range(len(self._ANSICOLORS))), self._ANSICOLORS):
                setattr(self, 'BG_' + color, curses.tparm(set_bg_ansi, i) or '')

    def _tigetstr(self, cap_name):
        # String capabilities can include "delays" of the form "$<2>".
        # For any modern terminal, we should be able to just ignore
        # these, so strip them out.
        import curses
        cap = curses.tigetstr(cap_name) or ''
        return re.sub(r'\$<\d+>[/*]?', '', cap)

    def render(self, template):
        """
        Replace each $-substitutions in the given template string with
        the corresponding terminal control string (if it's defined) or
        '' (if it's not).
        """
        return re.sub(r'\$\$|\${\w+}', self._render_sub, template)

    def _render_sub(self, match):
        s = match.group()
        if s == '$$':
            return s
        else:
            return getattr(self, s[2:-1])


#######################################################################
# Example use case: progress bar
#######################################################################

class ProgressBar:
    """
    A 3-line progress bar, which looks like::

                                Header
        20% [===========----------------------------------]
                           progress message

    The progress bar is colored, if the terminal supports color
    output; and adjusts to the width of the terminal.
    """
    BAR = '%3d%% ${GREEN}[${BOLD}%s%s${NORMAL}${GREEN}]${NORMAL}\n'
    HEADER = '${BOLD}${CYAN}%s${NORMAL}\n\n'

    def __init__(self, term, header):
        self.term = term
        if not (self.term.CLEAR_EOL and self.term.UP and self.term.BOL):
            raise ValueError("Terminal isn't capable enough -- you "
                             "should use a simpler progress dispaly.")
        self.width = self.term.COLS or 75
        self.bar = term.render(self.BAR)
        self.header = self.term.render(self.HEADER % header.center(self.width))
        self.cleared = 1  #: true if we haven't drawn the bar yet.
        self.update(0, '')

    def update(self, percent, message):
        if self.cleared:
            sys.stdout.write(self.header)
            self.cleared = 0
        n = int((self.width - 10) * percent)
        sys.stdout.write(
            self.term.BOL + self.term.UP + self.term.CLEAR_EOL +
            (self.bar % (100 * percent, '=' * n, '-' * (self.width - 10 - n))) +
            self.term.CLEAR_EOL + message.center(self.width))

    def clear(self):
        if not self.cleared:
            sys.stdout.write(self.term.BOL + self.term.CLEAR_EOL +
                             self.term.UP + self.term.CLEAR_EOL +
                             self.term.UP + self.term.CLEAR_EOL)
            self.cleared = 1


"""
Author: Tyler Reddy

The purpose of this Python module is to provide utility code for handling spherical Voronoi Diagrams.
"""

import math

import numpy.linalg
import numpy.random
import scipy
import scipy.spatial


class IntersectionError(Exception):
    pass


def filter_tetrahedron_to_triangle(current_tetrahedron_coord_array):
    current_triangle_coord_array = []  # initialize as a list
    for row in current_tetrahedron_coord_array:  # ugly to use for loop for this, but ok for now!
        if row[0] == 0 and row[1] == 0 and row[2] == 0:  # filter out origin row
            continue
        else:
            current_triangle_coord_array.append(row)
    current_triangle_coord_array = numpy.array(current_triangle_coord_array)
    return current_triangle_coord_array


def test_polygon_for_self_intersection(array_ordered_Voronoi_polygon_vertices_2D):
    '''Test an allegedly properly-ordered numpy array of Voronoi region vertices in 2D for self-intersection of edges
    based on algorithm described at http://algs4.cs.princeton.edu/91primitives/'''
    total_vertices = array_ordered_Voronoi_polygon_vertices_2D.shape[0]
    total_edges = total_vertices

    def intersection_test(a, b, c, d):
        # code in r & s equations provided on above website, which operate on the 2D coordinates of the edge
        # vertices for edges a - b and c - d
        # so: a, b, c, d are numpy arrays of vertex coordinates -- presumably with shape (2,)
        intersection = False
        denominator = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        r = ((a[1] - c[1]) * (d[0] - c[0]) - (a[0] - c[0]) * (d[1] - c[1])) / denominator
        s = ((a[1] - c[1]) * (b[0] - a[0]) - (a[0] - c[0]) * (b[1] - a[1])) / denominator
        if (r >= 0 and r <= 1) and (s >= 0 and s <= 1):  # conditions for intersection
            intersection = True
        if intersection:
            raise IntersectionError("Voronoi polygon line intersection !")

    # go through and test all possible non-consecutive edge combinations for intersection
    list_vertex_indices_in_edges = [[vertex_index, vertex_index + 1] for vertex_index in range(total_vertices)]
    # for the edge starting from the last point in the Voronoi polygon the index of the final point should be switched to the starting index -- to close the polygon
    filtered_list_vertex_indices_in_edges = []
    for list_vertex_indices_in_edge in list_vertex_indices_in_edges:
        if list_vertex_indices_in_edge[1] == total_vertices:
            filtered_list_vertex_indices_in_edge = [list_vertex_indices_in_edge[0], 0]
        else:
            filtered_list_vertex_indices_in_edge = list_vertex_indices_in_edge
        filtered_list_vertex_indices_in_edges.append(filtered_list_vertex_indices_in_edge)

    for edge_index, list_vertex_indices_in_edge in enumerate(filtered_list_vertex_indices_in_edges):
        for edge_index_2, list_vertex_indices_in_edge_2 in enumerate(filtered_list_vertex_indices_in_edges):
            if (list_vertex_indices_in_edge[0] not in list_vertex_indices_in_edge_2) and (
                    list_vertex_indices_in_edge[1] not in list_vertex_indices_in_edge_2):  # non-consecutive edges
                a = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge[0]]
                b = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge[1]]
                c = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge_2[0]]
                d = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge_2[1]]
                intersection_test(a, b, c, d)


def calculate_Vincenty_distance_between_spherical_points(cartesian_array_1, cartesian_array_2, sphere_radius):
    '''Apparently, the special case of the Vincenty formula (http://en.wikipedia.org/wiki/Great-circle_distance) may be the most accurate method for calculating great-circle distances.'''
    spherical_array_1 = convert_cartesian_array_to_spherical_array(cartesian_array_1)
    spherical_array_2 = convert_cartesian_array_to_spherical_array(cartesian_array_2)
    lambda_1 = spherical_array_1[1]
    lambda_2 = spherical_array_2[1]
    phi_1 = spherical_array_1[2]
    phi_2 = spherical_array_2[2]
    delta_lambda = abs(lambda_2 - lambda_1)
    delta_phi = abs(phi_2 - phi_1)
    radian_angle = math.atan2(math.sqrt((math.sin(phi_2) * math.sin(delta_lambda)) ** 2 + (
                math.sin(phi_1) * math.cos(phi_2) - math.cos(phi_1) * math.sin(phi_2) * math.cos(delta_lambda)) ** 2), (
                                          math.cos(phi_1) * math.cos(phi_2) + math.sin(phi_1) * math.sin(
                                      phi_2) * math.cos(delta_lambda)))
    spherical_distance = sphere_radius * radian_angle
    return spherical_distance


def calculate_haversine_distance_between_spherical_points(cartesian_array_1, cartesian_array_2, sphere_radius):
    '''Calculate the haversine-based distance between two points on the surface of a sphere. Should be more accurate than the arc cosine strategy. See, for example: http://en.wikipedia.org/wiki/Haversine_formula'''
    spherical_array_1 = convert_cartesian_array_to_spherical_array(cartesian_array_1)
    spherical_array_2 = convert_cartesian_array_to_spherical_array(cartesian_array_2)
    lambda_1 = spherical_array_1[1]
    lambda_2 = spherical_array_2[1]
    phi_1 = spherical_array_1[2]
    phi_2 = spherical_array_2[2]
    # we rewrite the standard Haversine slightly as long/lat is not the same as spherical coordinates - phi differs by pi/4
    spherical_distance = 2.0 * sphere_radius * math.asin(math.sqrt(
        ((1 - math.cos(phi_2 - phi_1)) / 2.) + math.sin(phi_1) * math.sin(phi_2) * (
                    (1 - math.cos(lambda_2 - lambda_1)) / 2.)))
    return spherical_distance


def filter_polygon_vertex_coordinates_for_extreme_proximity(array_ordered_Voronoi_polygon_vertices, sphere_radius):
    '''Merge (take the midpoint of) polygon vertices that are judged to be extremely close together and return the filtered polygon vertex array. The purpose is to alleviate numerical complications that may arise during surface area calculations involving polygons with ultra-close / nearly coplanar vertices.'''
    while 1:
        distance_matrix = scipy.spatial.distance.cdist(array_ordered_Voronoi_polygon_vertices,
                                                       array_ordered_Voronoi_polygon_vertices, 'euclidean')
        maximum_euclidean_distance_between_any_vertices = numpy.amax(distance_matrix)
        vertex_merge_threshold = 0.02  # merge any vertices that are separated by less than 1% of the longest inter-vertex distance (may have to play with this value a bit)
        threshold_assessment_matrix = distance_matrix / maximum_euclidean_distance_between_any_vertices
        row_indices_that_violate_threshold, column_indices_that_violate_threshold = numpy.where(
            (threshold_assessment_matrix < vertex_merge_threshold) & (threshold_assessment_matrix > 0))
        if len(row_indices_that_violate_threshold) > 0 and len(column_indices_that_violate_threshold) > 0:
            for row, column in zip(row_indices_that_violate_threshold, column_indices_that_violate_threshold):
                if not row == column:  # ignore diagonal values
                    first_violating_vertex_index = row
                    associated_vertex_index = column
                    new_vertex_at_midpoint = (array_ordered_Voronoi_polygon_vertices[row] +
                                              array_ordered_Voronoi_polygon_vertices[column]) / 2.0
                    spherical_polar_coords_new_vertex = convert_cartesian_array_to_spherical_array(
                        new_vertex_at_midpoint)
                    spherical_polar_coords_new_vertex[0] = sphere_radius  # project back to surface of sphere
                    new_vertex_at_midpoint = convert_spherical_array_to_cartesian_array(
                        spherical_polar_coords_new_vertex)
                    array_ordered_Voronoi_polygon_vertices[row] = new_vertex_at_midpoint
                    array_ordered_Voronoi_polygon_vertices = numpy.delete(array_ordered_Voronoi_polygon_vertices,
                                                                          column, 0)
                    break
        else:
            break  # no more violating vertices
    return array_ordered_Voronoi_polygon_vertices


def calculate_surface_area_of_planar_polygon_in_3D_space(array_ordered_Voronoi_polygon_vertices):
    '''Based largely on: http://stackoverflow.com/a/12653810
    Use this function when spherical polygon surface area calculation fails (i.e., lots of nearly-coplanar vertices and negative surface area).'''

    # unit normal vector of plane defined by points a, b, and c
    def unit_normal(a, b, c):
        x = numpy.linalg.det([[1, a[1], a[2]],
                              [1, b[1], b[2]],
                              [1, c[1], c[2]]])
        y = numpy.linalg.det([[a[0], 1, a[2]],
                              [b[0], 1, b[2]],
                              [c[0], 1, c[2]]])
        z = numpy.linalg.det([[a[0], a[1], 1],
                              [b[0], b[1], 1],
                              [c[0], c[1], 1]])
        magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
        return (x / magnitude, y / magnitude, z / magnitude)

    # area of polygon poly
    def poly_area(poly):
        '''Accepts a list of xyz tuples.'''
        assert len(poly) >= 3, "Not a polygon (< 3 vertices)."
        total = [0, 0, 0]
        N = len(poly)
        for i in range(N):
            vi1 = poly[i]
            vi2 = poly[(i + 1) % N]
            prod = numpy.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = numpy.dot(total, unit_normal(poly[0], poly[1], poly[2]))
        return abs(result / 2)

    list_vertices = []  # need a list of tuples for above function
    for coord in array_ordered_Voronoi_polygon_vertices:
        list_vertices.append(tuple(coord))
    planar_polygon_surface_area = poly_area(list_vertices)
    return planar_polygon_surface_area


def calculate_surface_area_of_a_spherical_Voronoi_polygon(array_ordered_Voronoi_polygon_vertices, sphere_radius):
    '''Calculate the surface area of a polygon on the surface of a sphere. Based on equation provided here: http://mathworld.wolfram.com/LHuiliersTheorem.html
    Decompose into triangles, calculate excess for each'''
    # have to convert to unit sphere before applying the formula
    spherical_coordinates = convert_cartesian_array_to_spherical_array(array_ordered_Voronoi_polygon_vertices)
    spherical_coordinates[..., 0] = 1.0
    array_ordered_Voronoi_polygon_vertices = convert_spherical_array_to_cartesian_array(spherical_coordinates)
    # handle nearly-degenerate vertices on the unit sphere by returning an area close to 0 -- may be better options, but this is my current solution to prevent crashes, etc.
    # seems to be relatively rare in my own work, but sufficiently common to cause crashes when iterating over large amounts of messy data
    if scipy.spatial.distance.pdist(array_ordered_Voronoi_polygon_vertices).min() < (10 ** -7):
        return 10 ** -8
    else:
        n = array_ordered_Voronoi_polygon_vertices.shape[0]
        # point we start from
        root_point = array_ordered_Voronoi_polygon_vertices[0]
        totalexcess = 0
        # loop from 1 to n-2, with point 2 to n-1 as other vertex of triangle
        # this could definitely be written more nicely
        b_point = array_ordered_Voronoi_polygon_vertices[1]
        root_b_dist = calculate_haversine_distance_between_spherical_points(root_point, b_point, 1.0)
        for i in 1 + numpy.arange(n - 2):
            a_point = b_point
            b_point = array_ordered_Voronoi_polygon_vertices[i + 1]
            root_a_dist = root_b_dist
            root_b_dist = calculate_haversine_distance_between_spherical_points(root_point, b_point, 1.0)
            a_b_dist = calculate_haversine_distance_between_spherical_points(a_point, b_point, 1.0)
            s = (root_a_dist + root_b_dist + a_b_dist) / 2
            totalexcess += 4 * math.atan(math.sqrt(
                math.tan(0.5 * s) * math.tan(0.5 * (s - root_a_dist)) * math.tan(0.5 * (s - root_b_dist)) * math.tan(
                    0.5 * (s - a_b_dist))))
        return totalexcess * (sphere_radius ** 2)


def calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(array_ordered_Voronoi_polygon_vertices,
                                                                     sphere_radius):
    '''Takes an array of ordered Voronoi polygon vertices (for a single generator) and calculates the sum of the inner angles on the sphere surface. The resulting value is theta in the equation provided here: http://mathworld.wolfram.com/SphericalPolygon.html '''
    # if sphere_radius != 1.0:
    # try to deal with non-unit circles by temporarily normalizing the data to radius 1:
    # spherical_polar_polygon_vertices = convert_cartesian_array_to_spherical_array(array_ordered_Voronoi_polygon_vertices)
    # spherical_polar_polygon_vertices[...,0] = 1.0
    # array_ordered_Voronoi_polygon_vertices = convert_spherical_array_to_cartesian_array(spherical_polar_polygon_vertices)

    num_vertices_in_Voronoi_polygon = array_ordered_Voronoi_polygon_vertices.shape[
        0]  # the number of rows == number of vertices in polygon

    # some debugging here -- I'm concerned that some sphere radii are demonstrating faulty projection of coordinates (some have r = 1, while others have r = sphere_radius -- see workflowy for more detailed notes)
    spherical_polar_polygon_vertices = convert_cartesian_array_to_spherical_array(
        array_ordered_Voronoi_polygon_vertices)
    min_vertex_radius = spherical_polar_polygon_vertices[..., 0].min()
    # print 'before array projection check'
    assert sphere_radius - min_vertex_radius < 0.1, "The minimum projected Voronoi vertex r value should match the sphere_radius of {sphere_radius}, but got {r_min}.".format(
        sphere_radius=sphere_radius, r_min=min_vertex_radius)
    # print 'after array projection check'

    # two edges (great circle arcs actually) per vertex are needed to calculate tangent vectors / inner angle at that vertex
    current_vertex_index = 0
    list_Voronoi_poygon_angles_radians = []
    while current_vertex_index < num_vertices_in_Voronoi_polygon:
        current_vertex_coordinate = array_ordered_Voronoi_polygon_vertices[current_vertex_index]
        if current_vertex_index == 0:
            previous_vertex_index = num_vertices_in_Voronoi_polygon - 1
        else:
            previous_vertex_index = current_vertex_index - 1
        if current_vertex_index == num_vertices_in_Voronoi_polygon - 1:
            next_vertex_index = 0
        else:
            next_vertex_index = current_vertex_index + 1
        # try using the law of cosines to produce the angle at the current vertex (basically using a subtriangle, which is a common strategy anyway)
        current_vertex = array_ordered_Voronoi_polygon_vertices[current_vertex_index]
        previous_vertex = array_ordered_Voronoi_polygon_vertices[previous_vertex_index]
        next_vertex = array_ordered_Voronoi_polygon_vertices[next_vertex_index]
        # produce a,b,c for law of cosines using spherical distance (http://mathworld.wolfram.com/SphericalDistance.html)
        # old_a = math.acos(numpy.dot(current_vertex,next_vertex))
        # old_b = math.acos(numpy.dot(next_vertex,previous_vertex))
        # old_c = math.acos(numpy.dot(previous_vertex,current_vertex))
        # print 'law of cosines a,b,c:', old_a,old_b,old_c
        # a = calculate_haversine_distance_between_spherical_points(current_vertex,next_vertex,sphere_radius)
        # b = calculate_haversine_distance_between_spherical_points(next_vertex,previous_vertex,sphere_radius)
        # c = calculate_haversine_distance_between_spherical_points(previous_vertex,current_vertex,sphere_radius)
        a = calculate_Vincenty_distance_between_spherical_points(current_vertex, next_vertex, sphere_radius)
        b = calculate_Vincenty_distance_between_spherical_points(next_vertex, previous_vertex, sphere_radius)
        c = calculate_Vincenty_distance_between_spherical_points(previous_vertex, current_vertex, sphere_radius)
        # print 'law of haversines a,b,c:', a,b,c
        # print 'Vincenty edge lengths a,b,c:', a,b,c
        pre_acos_term = (math.cos(b) - math.cos(a) * math.cos(c)) / (math.sin(a) * math.sin(c))
        if abs(pre_acos_term) > 1.0:
            print('angle calc vertex coords (giving acos violation):',
                  [convert_cartesian_array_to_spherical_array(vertex) for vertex in
                   [current_vertex, previous_vertex, next_vertex]])
            print('Vincenty edge lengths (giving acos violation) a,b,c:', a, b, c)
            print('pre_acos_term:', pre_acos_term)
            # break
        current_vertex_inner_angle_on_sphere_surface = math.acos(pre_acos_term)

        list_Voronoi_poygon_angles_radians.append(current_vertex_inner_angle_on_sphere_surface)

        current_vertex_index += 1

    if abs(pre_acos_term) > 1.0:
        theta = 0
    else:
        theta = numpy.sum(numpy.array(list_Voronoi_poygon_angles_radians))

    return theta


def convert_cartesian_array_to_spherical_array(coord_array, angle_measure='radians'):
    '''Take shape (N,3) cartesian coord_array and return an array of the same shape in spherical polar form (r, theta, phi). Based on StackOverflow response: http://stackoverflow.com/a/4116899
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    spherical_coord_array = numpy.zeros(coord_array.shape)
    xy = coord_array[..., 0] ** 2 + coord_array[..., 1] ** 2
    spherical_coord_array[..., 0] = numpy.sqrt(xy + coord_array[..., 2] ** 2)
    spherical_coord_array[..., 1] = numpy.arctan2(coord_array[..., 1], coord_array[..., 0])
    spherical_coord_array[..., 2] = numpy.arccos(coord_array[..., 2] / spherical_coord_array[..., 0])
    if angle_measure == 'degrees':
        spherical_coord_array[..., 1] = numpy.degrees(spherical_coord_array[..., 1])
        spherical_coord_array[..., 2] = numpy.degrees(spherical_coord_array[..., 2])
    return spherical_coord_array


def convert_spherical_array_to_cartesian_array(spherical_coord_array, angle_measure='radians'):
    '''Take shape (N,3) spherical_coord_array (r,theta,phi) and return an array of the same shape in cartesian coordinate form (x,y,z). Based on the equations provided at: http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    cartesian_coord_array = numpy.zeros(spherical_coord_array.shape)
    # convert to radians if degrees are used in input (prior to Cartesian conversion process)
    if angle_measure == 'degrees':
        spherical_coord_array[..., 1] = numpy.deg2rad(spherical_coord_array[..., 1])
        spherical_coord_array[..., 2] = numpy.deg2rad(spherical_coord_array[..., 2])
    # now the conversion to Cartesian coords
    cartesian_coord_array[..., 0] = spherical_coord_array[..., 0] * numpy.cos(
        spherical_coord_array[..., 1]) * numpy.sin(spherical_coord_array[..., 2])
    cartesian_coord_array[..., 1] = spherical_coord_array[..., 0] * numpy.sin(
        spherical_coord_array[..., 1]) * numpy.sin(spherical_coord_array[..., 2])
    cartesian_coord_array[..., 2] = spherical_coord_array[..., 0] * numpy.cos(spherical_coord_array[..., 2])
    return cartesian_coord_array


def produce_triangle_vertex_coordinate_array_Delaunay_sphere(hull_instance):
    '''Return shape (N,3,3) numpy array of the Delaunay triangle vertex coordinates on the surface of the sphere.'''
    list_points_vertices_Delaunay_triangulation = []
    for simplex in hull_instance.simplices:  # for each simplex (face; presumably a triangle) of the convex hull
        convex_hull_triangular_facet_vertex_coordinates = hull_instance.points[simplex]
        assert convex_hull_triangular_facet_vertex_coordinates.shape == (3,
                                                                         3), "Triangular facet of convex hull should be a triangle in 3D space specified by coordinates in a shape (3,3) numpy array."
        list_points_vertices_Delaunay_triangulation.append(convex_hull_triangular_facet_vertex_coordinates)
    array_points_vertices_Delaunay_triangulation = numpy.array(list_points_vertices_Delaunay_triangulation)
    return array_points_vertices_Delaunay_triangulation


def produce_array_Voronoi_vertices_on_sphere_surface(facet_coordinate_array_Delaunay_triangulation, sphere_radius,
                                                     sphere_centroid):
    '''Return shape (N,3) array of coordinates for the vertices of the Voronoi diagram on the sphere surface given a shape (N,3,3) array of Delaunay triangulation vertices.'''
    assert facet_coordinate_array_Delaunay_triangulation.shape[1:] == (
    3, 3), "facet_coordinate_array_Delaunay_triangulation should have shape (N,3,3)."
    # draft numpy vectorized workflow to avoid Python for loop
    facet_normals_array = numpy.cross(
        facet_coordinate_array_Delaunay_triangulation[..., 1, ...] - facet_coordinate_array_Delaunay_triangulation[
            ..., 0, ...],
        facet_coordinate_array_Delaunay_triangulation[..., 2, ...] - facet_coordinate_array_Delaunay_triangulation[
            ..., 0, ...])
    facet_normal_magnitudes = numpy.linalg.norm(facet_normals_array, axis=1)
    facet_normal_unit_vector_array = facet_normals_array / numpy.column_stack(
        (facet_normal_magnitudes, facet_normal_magnitudes, facet_normal_magnitudes))
    # try to ensure that facet normal faces the correct direction (i.e., out of sphere)
    triangle_centroid_array = numpy.average(facet_coordinate_array_Delaunay_triangulation, axis=1)
    # normalize the triangle_centroid to unit sphere distance for the purposes of the following directionality check
    array_triangle_centroid_spherical_coords = convert_cartesian_array_to_spherical_array(triangle_centroid_array)
    array_triangle_centroid_spherical_coords[..., 0] = 1.0
    triangle_centroid_array = convert_spherical_array_to_cartesian_array(array_triangle_centroid_spherical_coords)
    # the Euclidean distance between the triangle centroid and the facet normal should be smaller than the sphere centroid to facet normal distance, otherwise, need to invert the vector
    triangle_to_normal_distance_array = numpy.linalg.norm(triangle_centroid_array - facet_normal_unit_vector_array,
                                                          axis=1)
    sphere_centroid_to_normal_distance_array = numpy.linalg.norm(sphere_centroid - facet_normal_unit_vector_array,
                                                                 axis=1)
    delta_value_array = sphere_centroid_to_normal_distance_array - triangle_to_normal_distance_array
    facet_normal_unit_vector_array[
        delta_value_array < -0.1] *= -1.0  # need to rotate the vector so that it faces out of the circle
    facet_normal_unit_vector_array *= sphere_radius  # adjust for radius of sphere
    array_Voronoi_vertices = facet_normal_unit_vector_array
    assert array_Voronoi_vertices.shape[1] == 3, "The array of Voronoi vertices on the sphere should have shape (N,3)."
    return array_Voronoi_vertices


# !/usr/bin/python

import Utils


class StationData:

    def __init__(self, x=None, y=None, z=None):

        self.X = x
        self.Y = y
        self.Z = z

        self.sigX = None
        self.sigY = None
        self.sigZ = None
        self.sigXY = None
        self.sigXZ = None
        self.sigYZ = None

        self.velX = None
        self.velY = None
        self.velZ = None

        self.sigVelX = None
        self.sigVelY = None
        self.sigVelZ = None

        self.refEpoch = None

        self.domesNumber = None

    def Print(self):
        if self.X != None:
            print('%13.4f %13.4f %13.4f ' % (self.X, self.Y, self.Z), end=' ')

        if self.sigX != None:
            print('%2.10f %2.10f %2.10f\n' % (self.sigX, self.sigY, self.sigZ), end=' ')
        else:
            print()

    def __repr__(self):

        string = ""

        if self.X != None:
            string += '%13.4f %13.4f %13.4f ' % (self.X, self.Y, self.Z)

        if self.sigX != None:
            string += '%10.8f %10.8f %10.8f ' % (self.sigX, self.sigY, self.sigZ)

        if self.velX != None:
            string += '%8.5f %8.5f %8.5f ' % (self.velX, self.velY, self.velZ)

        if self.sigVelX != None:
            string += '%10.8f %10.8f %10.8f ' % (self.sigVelX, self.sigVelY, self.sigVelZ)

        if self.refEpoch != None:
            string += '%11.6f' % (self.refEpoch)

        return string

    def __str__(self):

        string = ""

        if self.X != None:
            string += '%13.4f %13.4f %13.4f ' % (self.X, self.Y, self.Z)

        if self.sigX != None:
            string += '%2.10f %2.10f %2.10f ' % (self.sigX, self.sigY, self.sigZ)

        if self.velX != None:
            string += '%8.5f %8.5f %8.5f ' % (self.velX, self.velY, self.velZ)

        if self.sigVelX != None:
            string += '%10.8f %10.8f %10.8f ' % (self.sigVelX, self.sigVelY, self.sigVelZ)

        if self.refEpoch != None:
            string += '%11.6f' % (self.refEpoch)

        return string


class mergedSinexStationData():

    # Decorator Pattern to avoid subclassing ...

    def __init__(self, snxStnData=None):
        self.snxStnData = snxStnData
        self.orgNameSet = set()

    def Print(self):

        # don't want to print the sigmaXYZ
        if self.snxStnData.X != None:
            print('%13.4f %13.4f %13.4f ' % (self.snxStnData.X, self.snxStnData.Y, self.snxStnData.Z), end=' ')

        # print additional org info also
        if len(self.orgNameSet) > 0:
            for orgName in self.orgNameSet:
                print(orgName, end=' ')
            print()
        else:
            print()


class snxFileParser(object):

    def __init__(self, snxFilePath=None):

        self.snxFilePath = snxFilePath
        self.snxFileName = os.path.basename(snxFilePath)
        self.stationDict = dict()
        self.orgName = None
        self.varianceFactor = 1
        self.observations = 0
        self.unknowns = 0

        # iter protocol shit
        self.iterIndx = 0
        self.iterList = None

    def parse(self):

        foundSolutionEstimate = False
        inSolutionEstimateSection = False
        inSolutionMatrixEstimateSection = False
        isHeaderLine = False

        foundSiteId = False
        inSiteIdSection = False

        dictID = dict()
        stn_ID = dict()

        # if there's a file to parse
        if self.snxFilePath != None:

            # flag to rezip at end
            wasZipped = False
            wasCompressed = False

            # check for gzip
            if self.snxFilePath[-2:] == "gz":
                # file_ops.gunzip(self.snxFilePath)
                self.snxFilePath = self.snxFilePath[0:-3]
                wasZipped = True

            # check for unix compression
            elif self.snxFilePath[-1:] == "Z":
                # file_ops.uncompress(self.snxFilePath)
                self.snxFilePath = self.snxFilePath[0:-2]
                wasCompressed = True

            # open the file
            try:
                snxFileHandle = open(self.snxFilePath, 'r')
            except:
                print("snxFileParser ERROR:  Could not open file " + self.snxFilePath + " !!!")
                raise

            # make pattern to match to snx organization ...
            self.snxFileName = os.path.basename(self.snxFilePath)

            orgPattern = re.compile('^([a-zA-Z]+).*\.f?snx$')
            orgMatch = orgPattern.findall(self.snxFileName)
            self.orgName = orgMatch[0].upper()

            # make pattern to look for SiteId start tag
            siteIdStartPattern = re.compile('^\+SITE\/ID$')

            # make pattern to look for end of siteId section
            siteIdEndPattern = re.compile('^\-SITE\/ID$')

            # make pattern to parse the siteId lines
            # Example:
            #
            #     TROM  A 82397M001 P , USA                   18 56 18.0  69 39 45.9   135.4
            #
            siteIdPattern = re.compile('^\s+(\w+)\s+\w\s+(\w+).*$')

            # variance factor patther
            # Example:
            #
            # VARIANCE FACTOR                    0.048618461936712
            #
            #
            varianceFactorPattern = re.compile('^ VARIANCE FACTOR\s+([\d+]?\.\d+)$')

            observationsPattern = re.compile('^ NUMBER OF OBSERVATIONS\s+(\d+)$')

            unknownsPattern = re.compile('^ NUMBER OF UNKNOWNS\s+(\d+)$')

            # Make pattern to look for solution estimate start tag
            startSolutionEstimatePattern = re.compile('^\+SOLUTION\/ESTIMATE.*')

            # make pattern to look for solution estimate end tag
            endSolutionEstimatePattern = re.compile('^\-SOLUTION\/ESTIMATE.*')

            # make pattern to look for the L COVA start tag (+SOLUTION/MATRIX_ESTIMATE L COVA)
            startSolutionMatrixEstimate = re.compile('^\+SOLUTION\/MATRIX_ESTIMATE.*')

            # make pattern to look for the L COVA end tag (-SOLUTION/MATRIX_ESTIMATE L COVA)
            endSolutionMatrixEstimate = re.compile('^\-SOLUTION\/MATRIX_ESTIMATE.*')

            # make pattern to look for station coordinates
            # Example:
            #
            #   1 STAX   ALGO  A ---- 05:180:43200 m    2 .91812936331043008E+6 .2511266E-2
            #
            stationCoordinatePattern = re.compile(
                '^\s+(\d+)+\s+STA(\w)\s+(\w+)\s+(\w).*\d+\s+(-?[\d+]?\.\d+[Ee][+-]?\d+)\s+(-?[\d+]?\.\d+[Ee][+-]?\d+)$')

            # make pattern to look for station velocities
            # Example:
            #
            # 916 VELX   YAKA  A    1 00:001:00000 m/y  2 -.219615010076079E-01 0.13728E-03
            #
            stationVelocityPattern = re.compile(
                '^\s+\d+\s+VEL(\w)\s+(\w+)\s+\w\s+....\s+(\d\d:\d\d\d).*\d+\s+(-?[\d+]?\.\d+[Ee][+-]?\d+)\s+(-?[\d+]?\.\d+[Ee][+-]?\d+)$')

            for line in snxFileHandle:

                varianceFactorMatch = varianceFactorPattern.findall(line)

                observationsMatch = observationsPattern.findall(line)

                unknownsMatch = unknownsPattern.findall(line)

                siteIdStartMatch = siteIdStartPattern.findall(line)
                siteIdEndMatch = siteIdEndPattern.findall(line)

                startSolutionEstimateMatch = startSolutionEstimatePattern.findall(line)
                endSolutionEstimateMatch = endSolutionEstimatePattern.findall(line)

                startSolutionMatrixEstimateMatch = startSolutionMatrixEstimate.findall(line)
                endSolutionMatrixEstimateMatch = endSolutionMatrixEstimate.findall(line)

                if siteIdStartMatch:
                    inSiteIdSection = True
                    continue

                if siteIdEndMatch:
                    inSiteIdSection = False
                    continue

                # check for solution estimate section
                if startSolutionEstimateMatch:
                    inSolutionEstimateSection = True
                    continue

                elif endSolutionEstimateMatch:
                    inSolutionEstimateSection = False
                    continue

                if startSolutionMatrixEstimateMatch:
                    inSolutionMatrixEstimateSection = True
                    continue

                elif endSolutionMatrixEstimateMatch:
                    inSolutionMatrixEstimateSection = False
                    break

                if varianceFactorMatch:
                    self.varianceFactor = float(varianceFactorMatch[0])

                if unknownsMatch:
                    self.unknowns = float(unknownsMatch[0])

                if observationsMatch:
                    self.observations = float(observationsMatch[0])

                if inSiteIdSection:

                    # parse the siteID line
                    siteIdMatch = siteIdPattern.findall(line)

                    # blab about it
                    # print siteIdMatch

                    # if the line does not contain a match then move along
                    if not siteIdMatch:
                        continue

                    # extract the parsed info
                    (stationName, domesNumber) = siteIdMatch[0]

                    # make sure the name is upper case
                    stationName = stationName.upper()

                    # initialize station data if not seen this station before
                    if not stationName in list(self.stationDict.keys()):
                        self.stationDict[stationName] = StationData()

                    self.stationDict[stationName].domesNumber = domesNumber

                    # print "set domes number "+ domesNumber +" for station "+stationName

                # if in the solution estimate section
                if inSolutionEstimateSection:

                    # check for station coordinate match
                    stationCoordinateMatch = stationCoordinatePattern.findall(line)

                    # check for station velocity match
                    stationVelocityMatch = stationVelocityPattern.findall(line)

                    # print line
                    # print stationCoordinateMatch

                    # if match then store result
                    if stationCoordinateMatch:
                        (ID,
                         coordID,
                         stationName,
                         pointCode,
                         coord,
                         sigCoord
                         ) = stationCoordinateMatch[0]

                        if pointCode != 'A':
                            os.sys.stderr.write('ignoring solution/estimate STA' + coordID \
                                                + ' for station: ' + stationName \
                                                + ', point code = ' + pointCode \
                                                + ', file = ' + self.snxFileName \
                                                + '\n' \
                                                )
                            continue

                        # make sure station name is upper case
                        stationName = stationName.upper()

                        # save the correspondance between the ID and the coordID and stationName
                        dictID[ID] = coordID
                        stn_ID[ID] = stationName

                        if not stationName in list(self.stationDict.keys()):
                            self.stationDict[stationName] = StationData()

                        if coordID == 'X':
                            self.stationDict[stationName].X = float(coord)
                            self.stationDict[stationName].sigX = float(sigCoord)

                        elif coordID == 'Y':
                            self.stationDict[stationName].Y = float(coord)
                            self.stationDict[stationName].sigY = float(sigCoord)

                        else:
                            self.stationDict[stationName].Z = float(coord)
                            self.stationDict[stationName].sigZ = float(sigCoord)

                    if stationVelocityMatch:

                        ( \
                            coordID, \
                            stationName, \
                            refEpoch, \
                            vel, \
                            sigVel \
                            ) = stationVelocityMatch[0]

                        stationName = stationName.upper()

                        # parse refEpoch String
                        (year, doy) = refEpoch.split(':')

                        # convert from string
                        doy = float(doy)

                        # normalize the year and convert to float
                        year = float(Utils.get_norm_year_str(year))

                        # compute fractional year to match matlab round off
                        fractionalYear = year + ((doy - 1) / 366.0) + 0.001413

                        # init if not already in dict
                        if not stationName in list(self.stationDict.keys()):
                            self.stationDict[stationName] = StationData()

                        # set the reference epoch for the velocity
                        self.stationDict[stationName].refEpoch = fractionalYear

                        if coordID == 'X':
                            self.stationDict[stationName].velX = float(vel)
                            self.stationDict[stationName].sigVelX = float(sigVel)

                        elif coordID == 'Y':
                            self.stationDict[stationName].velY = float(vel)
                            self.stationDict[stationName].sigVelY = float(sigVel)

                        else:
                            self.stationDict[stationName].velZ = float(vel)
                            self.stationDict[stationName].sigVelZ = float(sigVel)

                # regzip the file is was .gz
                # if wasZipped:
                # file_ops.gzip(self.snxFilePath)

                # recompress the file if was .Z
                # if wasCompressed:
                # file_ops.compress(self.snxFilePath)

                if inSolutionMatrixEstimateSection:
                    matrixLine = line.split()

                    ID1 = matrixLine[0]

                    # check that the key is actually a station variance-covariance item
                    if ID1 in list(stn_ID.keys()):

                        for i, ID2 in enumerate(range(len(matrixLine) - 2)):
                            ID2 = str(int(matrixLine[1]) + i)

                            if stn_ID[ID1] == stn_ID[ID2] and dictID[ID1] != dictID[ID2]:
                                # we already have the variance, we want the covariance
                                if (dictID[ID1] == 'X' and dictID[ID2] == 'Y') or (
                                        dictID[ID1] == 'Y' and dictID[ID2] == 'X'):
                                    self.stationDict[stn_ID[ID1]].sigXY = float(matrixLine[i + 2])

                                elif (dictID[ID1] == 'X' and dictID[ID2] == 'Z') or (
                                        dictID[ID1] == 'Z' and dictID[ID2] == 'X'):
                                    self.stationDict[stn_ID[ID1]].sigXZ = float(matrixLine[i + 2])

                                elif (dictID[ID1] == 'Y' and dictID[ID2] == 'Z') or (
                                        dictID[ID1] == 'Z' and dictID[ID2] == 'Y'):
                                    self.stationDict[stn_ID[ID1]].sigYZ = float(matrixLine[i + 2])

        return self

    def Print(self, key=None, fid=None):

        if key != None and self.contains(key):
            print(key, self.orgName, end=' ')
            self.stationDict[key].Print()

        # loop through each station print info
        for stationName in list(self.stationDict.keys()):
            print(stationName, self.orgName, end=' ')
            self.stationDict[stationName].Print()

    def size(self):
        return len(self.stationDict)

    def __iter__(self):
        self.iterList = list(self.stationDict.keys())
        return self

    # iterator protocol
    def __next__(self):

        # if self.iterList ==None:
        #    self.iterList = list(self.stnIdSet)

        if self.iterIndx > len(self.iterList) - 1:

            # reset iteration parameters
            self.iterList = None
            self.iterIndx = 0

            # halt iteration
            raise StopIteration
        else:
            key = self.iterList[self.iterIndx]
            self.iterIndx += 1
            return key

    def contains(self, key):
        return key.upper() in self.stationDict

    def get(self, key):
        if self.contains(key):
            return self.stationDict[key.upper()]
        else:
            return None


class snxStationMerger:

    def __init__(self):
        self.snxObjectList = list()
        self.mergedStationDict = dict()
        self.orgList = set()
        self.maxMetersApart = 1

        # init the stationDict iwht level one stations
        # level one holds all stations that appear only once
        self.mergedStationDict['1'] = dict()

    def compareUsingCoordinates(self, snxStationDataA, snxStationDataB):

        diffX = snxStationDataB.X - snxStationDataA.X
        diffY = snxStationDataB.Y - snxStationDataA.Y
        diffZ = snxStationDataB.Z - snxStationDataA.Z

        radialDistanceApart = (diffX ** 2 + diffY ** 2 + diffZ ** 2) ** (0.5)

        if radialDistanceApart <= self.maxMetersApart:
            return True
        else:
            # print 'station with same name found but different coords!!!  Radial Distance Apart:',radialDistanceApart
            return False

    def stationExistsWithNumberOfOccurrences(self, stationName, numberOfOccurences):

        # first make sure numberOfOccurences is an integer at least zero
        try:
            numberOfOccurrences = int(numberOfOccurences)
        except Exception:
            print('number of occurrences is not an integer!!!')
            raise

        if numberOfOccurrences < 0:
            print('number of occurrences must be at least zero!!')
            raise Exception

        # finally, convert to string for merged station dictionary key
        numberOfOccurrences = str(numberOfOccurrences)

        # empty station dictionary case
        if len(list(self.mergedStationDict.keys())) == 0:
            return False

        # check that numberOfOccurances even is in dictionary
        if numberOfOccurrences not in list(self.mergedStationDict.keys()):
            return False

        elif stationName not in self.mergedStationDict[numberOfOccurrences]:
            return False

        else:
            return True

    def maxLevel(self):

        return int(sorted(list(self.mergedStationDict.keys()))[-1])

    def addStation(self, level, orgName, stationName, snxStnData):

        # add instance of mergedSinexStationData to mergedStationDict at level
        self.mergedStationDict[level][stationName] = \
            mergedSinexStationData(snxStnData)

        # add org data
        self.mergedStationDict[level][stationName].orgNameSet.add(orgName)

    def addStationsFromSinexObject(self, snxObj):

        stationAdded = False

        # add sinex to list
        self.snxObjectList.append(snxObj)

        for stationName in list(snxObj.stationDict.keys()):

            for level in list(self.mergedStationDict.keys()):

                # first check if station is in level one station dictionary
                if not stationName in list(self.mergedStationDict[level].keys()):

                    # add the station to this level since it does not exist
                    self.addStation(level, snxObj.orgName, stationName, snxObj.stationDict[stationName])

                    # update list of all unique orgs associated with station merge
                    self.orgList.add(snxObj.orgName)

                    # note that we added the station here
                    stationAdded = True

                    # no point looking through other levels since we've added it here
                    break

                else:

                    # station with same name should be compared by coordinate(s)
                    if self.compareUsingCoordinates( \
                            self.mergedStationDict[level][stationName].snxStnData, \
                            snxObj.stationDict[stationName] \
                            ):
                        # update org list for station to reflect the orgs that process this station
                        self.mergedStationDict[level][stationName].orgNameSet.add(snxObj.orgName)

                        # nothing left to do, since they are the same station within 1 meter
                        # Notice here that if false then station check goes to next level
                        stationAdded = True
                        break

            # if station not added at any level
            # then need to make new level and add station
            if not stationAdded:
                # get max level + 1
                nextLevel = str(self.maxLevel() + 1)

                # init the level/numberOfOccurrences
                self.mergedStationDict[nextLevel] = dict()

                # add the station with correct number of unique occurrences
                self.addStation(nextLevel, snxObj.orgName, stationName, snxObj.stationDict[stationName])

            # reset station added for next station
            stationAdded = False

        return self

    def Print(self):
        for level in list(self.mergedStationDict.keys()):
            for stationName in list(self.mergedStationDict[level].keys()):
                print(stationName, level, end=' ')
                self.mergedStationDict[level][stationName].Print()


def main():
    #    esaFile = '../esa13297.snx'
    #    emrFile = '../emr13297.snx'
    #    gfzFile = '../gfz13297.snx'
    #    jplFile = '../jpl13297.snx'
    #    ngsFile = '../ngs13297.snx'
    #    sioFile = '../sio13293.snx'
    #
    #    # init the sinex objects
    #    snxParserESA = snxFileParser(esaFile)
    #    snxParserEMR = snxFileParser(emrFile)
    #    snxParserGFZ = snxFileParser(gfzFile)
    #    snxParserJPL = snxFileParser(jplFile)
    #    snxParserNGS = snxFileParser(ngsFile)
    #    snxParserSIO = snxFileParser(sioFile)
    #

    #
    #    # add stations from each sinxe object
    #    snxMerge.addStationsFromSinexObject(snxParserESA.parse())
    #    snxMerge.addStationsFromSinexObject(snxParserEMR.parse())
    #    snxMerge.addStationsFromSinexObject(snxParserGFZ.parse())
    #    snxMerge.addStationsFromSinexObject(snxParserJPL.parse())
    #    snxMerge.addStationsFromSinexObject(snxParserNGS.parse())
    #    snxMerge.addStationsFromSinexObject(snxParserSIO.parse())

    # get list of sinex files
    # snxFileList = glob('/Users/abel/itrf/itrf2008_trf.snx')

    # snxParser = snxFileParser(snxFileList[0]).parse()

    # for stn in snxParser:
    #    print 'COORD',stn, snxParser.get(stn)

    # init the sinex station merger
    # snxMerge = snxStationMerger()

    # uncomment the following to set
    # the position fileter manually
    # snxMerge.maxMetersApart = 1.1

    #    for snxFile in snxFileList:
    #        os.sys.stdout.write("Processing snx: "+snxFile+"\n")
    #        snxMerge.addStationsFromSinexObject(snxFileParser(snxFile).parse())
    #
    #    # print the data
    #    snxMerge.Print()

    file = '/media/fugu/processing/projects/osuGLOBAL/napeos/orbit/solutions/2006/304/o12/o1213992.snx.gz'
    file = '/Users/gomez.124/mounts/qnap/ign/procesamientos/gamit/2010/001/igs-sirgas/glbf/IGN15645.snx'

    sigXYZ = 1.0

    snxParser = snxFileParser(file).parse()

    # print the header line
    os.sys.stdout.write('STATION COORDINATES AT EPOCH  2002.0\n')

    # organize the coordinates for the data object
    for stn in snxParser:
        # get the snx data from the parser
        stnData = snxParser.get(stn)

        # print stn, stnData.domesNumber, stnData.X, stnData.Y, stnData.Z

        xyzStr = "%9s %4s %9s     GPS %4s                  %12.3f %12.3f %12.3f %5.3f %5.3f %5.3f OSU" % (
            stnData.domesNumber, stn, stnData.domesNumber, stn, stnData.X, stnData.Y, stnData.Z, sigXYZ, sigXYZ, sigXYZ)
        velStr = "%9s %4s %9s         %4s                  %12.3f %12.3f %12.3f %5.3f %5.3f %5.3f OSU" % (
            stnData.domesNumber, "", "", "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        os.sys.stdout.write(xyzStr + "\n")
        os.sys.stdout.write(velStr + "\n")


import argparse
import filecmp
import subprocess
from datetime import datetime

import numpy
import pyDate


class UtilsException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


def parse_atx_antennas(atx_file):
    f = open(atx_file, 'r')
    output = f.readlines()
    f.close()

    return re.findall(r'START OF ANTENNA\s+(\w+[.-\/+]?\w*[.-\/+]?\w*)\s+(\w+)', ''.join(output), re.MULTILINE)


def smallestN_indices(a, N):
    """
    Function to return the row and column of the N smallest values
    :param a: array to search (any dimension)
    :param N: number of values to search
    :return: array with the rows-cols of min values
    """
    idx = a.ravel().argsort()[:N]
    return numpy.stack(numpy.unravel_index(idx, a.shape)).T


def ll2sphere_xyz(ell):
    r = 6371000.0
    x = []
    for lla in ell:
        x.append((r * numpy.cos(lla[0] * numpy.pi / 180) * numpy.cos(lla[1] * numpy.pi / 180),
                  r * numpy.cos(lla[0] * numpy.pi / 180) * numpy.sin(lla[1] * numpy.pi / 180),
                  r * numpy.sin(lla[0] * numpy.pi / 180)))

    return numpy.array(x)


def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = 'argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest, nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredLength


def parse_crinex_rinex_filename(filename):
    # parse a crinex filename
    sfile = re.findall('(\w{4})(\d{3})(\w{1})\.(\d{2})([d]\.[Z])$', filename)

    if sfile:
        return sfile[0]
    else:
        sfile = re.findall('(\w{4})(\d{3})(\w{1})\.(\d{2})([o])$', filename)

        if sfile:
            return sfile[0]
        else:
            return []


def _increment_filename(filename):
    """
    Returns a generator that yields filenames with a counter. This counter
    is placed before the file extension, and incremented with every iteration.
    For example:
        f1 = increment_filename("myimage.jpeg")
        f1.next() # myimage-1.jpeg
        f1.next() # myimage-2.jpeg
        f1.next() # myimage-3.jpeg
    If the filename already contains a counter, then the existing counter is
    incremented on every iteration, rather than starting from 1.
    For example:
        f2 = increment_filename("myfile-3.doc")
        f2.next() # myfile-4.doc
        f2.next() # myfile-5.doc
        f2.next() # myfile-6.doc
    The default marker is an underscore, but you can use any string you like:
        f3 = increment_filename("mymovie.mp4", marker="_")
        f3.next() # mymovie_1.mp4
        f3.next() # mymovie_2.mp4
        f3.next() # mymovie_3.mp4
    Since the generator only increments an integer, it is practically unlimited
    and will never raise a StopIteration exception.
    """
    # First we split the filename into three parts:
    #
    #  1) a "base" - the part before the counter
    #  2) a "counter" - the integer which is incremented
    #  3) an "extension" - the file extension

    sessions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [chr(x) for x in range(ord('a'), ord('z') + 1)]

    path = os.path.dirname(filename)
    filename = os.path.basename(filename)
    fileparts = parse_crinex_rinex_filename(filename)

    if not fileparts:
        raise ValueError('Invalid file naming convention: {}'.format(filename))

    # Check if there's a counter in the filename already - if not, start a new
    # counter at 0.
    value = 0

    filename = os.path.join(path, '%s%03i%s.%02i%s' % (
        fileparts[0].lower(), int(fileparts[1]), sessions[value], int(fileparts[3]), fileparts[4]))

    # The counter is just an integer, so we can increment it indefinitely.
    while True:
        if value == 0:
            yield filename

        value += 1

        if value == len(sessions):
            raise ValueError('Maximum number of sessions reached: %s%03i%s.%02i%s' % (
                fileparts[0].lower(), int(fileparts[1]), sessions[value - 1], int(fileparts[3]), fileparts[4]))

        yield os.path.join(path, '%s%03i%s.%02i%s' % (
            fileparts[0].lower(), int(fileparts[1]), sessions[value], int(fileparts[3]), fileparts[4]))


def copyfile(src, dst):
    """
    Copies a file from path src to path dst.
    If a file already exists at dst, it will not be overwritten, but:
     * If it is the same as the source file, do nothing
     * If it is different to the source file, pick a new name for the copy that
       is distinct and unused, then copy the file there.
    Returns the path to the copy.
    """
    if not os.path.exists(src):
        raise ValueError('Source file does not exist: {}'.format(src))

    # make the folders if they don't exist
    # careful! racing condition between different workers
    try:
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
    except OSError:
        # some other process created the folder an instant before
        pass

    # Keep trying to copy the file until it works
    dst_gen = _increment_filename(dst)

    while True:

        dst = next(dst_gen)

        # Check if there is a file at the destination location
        if os.path.exists(dst):

            # If the namesake is the same as the source file, then we don't
            # need to do anything else.
            if filecmp.cmp(src, dst):
                return dst

        else:

            # If there is no file at the destination, then we attempt to write
            # to it. There is a risk of a race condition here: if a file
            # suddenly pops into existence after the `if os.path.exists()`
            # check, then writing to it risks overwriting this new file.
            #
            # We write by transferring bytes using os.open(). Using the O_EXCL
            # flag on the dst file descriptor will cause an OSError to be
            # raised if the file pops into existence; the O_EXLOCK stops
            # anybody else writing to the dst file while we're using it.
            try:
                src_fd = os.open(src, os.O_RDONLY)
                dst_fd = os.open(dst, os.O_WRONLY | os.O_EXCL | os.O_CREAT)

                # Read 100 bytes at a time, and copy them from src to dst
                while True:
                    data = os.read(src_fd, 100)
                    os.write(dst_fd, data)

                    # When there are no more bytes to read from the source
                    # file, 'data' will be an empty string
                    if not data:
                        break

                os.close(src_fd)
                os.close(dst_fd)
                # If we get to this point, then the write has succeeded
                return dst

            # An OSError errno 17 is what happens if a file pops into existence
            # at dst, so we print an error and try to copy to a new location.
            # Any other exception is unexpected and should be raised as normal.
            except OSError as e:
                if e.errno != 17 or e.strerror != 'File exists':
                    raise


def move(src, dst):
    """
    Moves a file from path src to path dst.
    If a file already exists at dst, it will not be overwritten, but:
     * If it is the same as the source file, do nothing
     * If it is different to the source file, pick a new name for the copy that
       is distinct and unused, then copy the file there.
    Returns the path to the new file.
    """
    dst = copyfile(src, dst)
    os.remove(src)
    return dst


def ct2lg(dX, dY, dZ, lat, lon):
    n = dX.size

    R = rotct2lg(lat, lon, n)

    dxdydz = numpy.column_stack((numpy.column_stack((dX, dY)), dZ))

    RR = numpy.reshape(R[0, :, :], (3, n))
    dx = numpy.sum(numpy.multiply(RR, dxdydz.transpose()), axis=0)
    RR = numpy.reshape(R[1, :, :], (3, n))
    dy = numpy.sum(numpy.multiply(RR, dxdydz.transpose()), axis=0)
    RR = numpy.reshape(R[2, :, :], (3, n))
    dz = numpy.sum(numpy.multiply(RR, dxdydz.transpose()), axis=0)

    return dx, dy, dz


def rotct2lg(lat, lon, n=1):
    R = numpy.zeros((3, 3, n))

    R[0, 0, :] = -numpy.multiply(numpy.sin(numpy.deg2rad(lat)), numpy.cos(numpy.deg2rad(lon)))
    R[0, 1, :] = -numpy.multiply(numpy.sin(numpy.deg2rad(lat)), numpy.sin(numpy.deg2rad(lon)))
    R[0, 2, :] = numpy.cos(numpy.deg2rad(lat))
    R[1, 0, :] = -numpy.sin(numpy.deg2rad(lon))
    R[1, 1, :] = numpy.cos(numpy.deg2rad(lon))
    R[1, 2, :] = numpy.zeros((1, n))
    R[2, 0, :] = numpy.multiply(numpy.cos(numpy.deg2rad(lat)), numpy.cos(numpy.deg2rad(lon)))
    R[2, 1, :] = numpy.multiply(numpy.cos(numpy.deg2rad(lat)), numpy.sin(numpy.deg2rad(lon)))
    R[2, 2, :] = numpy.sin(numpy.deg2rad(lat))

    return R


def lg2ct(dN, dE, dU, lat, lon):
    n = dN.size

    R = rotlg2ct(lat, lon, n)

    dxdydz = numpy.column_stack((numpy.column_stack((dN, dE)), dU))

    RR = numpy.reshape(R[0, :, :], (3, n))
    dx = numpy.sum(numpy.multiply(RR, dxdydz.transpose()), axis=0)
    RR = numpy.reshape(R[1, :, :], (3, n))
    dy = numpy.sum(numpy.multiply(RR, dxdydz.transpose()), axis=0)
    RR = numpy.reshape(R[2, :, :], (3, n))
    dz = numpy.sum(numpy.multiply(RR, dxdydz.transpose()), axis=0)

    return dx, dy, dz


def rotlg2ct(lat, lon, n=1):
    R = numpy.zeros((3, 3, n))

    R[0, 0, :] = -numpy.multiply(numpy.sin(numpy.deg2rad(lat)), numpy.cos(numpy.deg2rad(lon)))
    R[1, 0, :] = -numpy.multiply(numpy.sin(numpy.deg2rad(lat)), numpy.sin(numpy.deg2rad(lon)))
    R[2, 0, :] = numpy.cos(numpy.deg2rad(lat))
    R[0, 1, :] = -numpy.sin(numpy.deg2rad(lon))
    R[1, 1, :] = numpy.cos(numpy.deg2rad(lon))
    R[2, 1, :] = numpy.zeros((1, n))
    R[0, 2, :] = numpy.multiply(numpy.cos(numpy.deg2rad(lat)), numpy.cos(numpy.deg2rad(lon)))
    R[1, 2, :] = numpy.multiply(numpy.cos(numpy.deg2rad(lat)), numpy.sin(numpy.deg2rad(lon)))
    R[2, 2, :] = numpy.sin(numpy.deg2rad(lat))

    return R


def parseIntSet(nputstr=""):
    selection = []
    invalid = []
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        if len(i) > 0:
            if i[:1] == "<":
                i = "1-%s" % (i[1:])
        try:
            # typically tokens are plain old integers
            selection.append(int(i))
        except Exception:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.append(x)
            except:
                # not an int and not a range...
                invalid.append(i)
    # Report invalid tokens before returning valid selection
    if len(invalid) > 0:
        print(("Invalid set: " + str(invalid)))
        sys.exit(2)
    return selection


def ecef2lla(ecefArr):
    # convert ECEF coordinates to LLA
    # test data : test_coord = [2297292.91, 1016894.94, -5843939.62]
    # expected result : -66.8765400174 23.876539914 999.998386689

    x = float(ecefArr[0])
    y = float(ecefArr[1])
    z = float(ecefArr[2])

    a = 6378137
    e = 8.1819190842622e-2

    asq = numpy.power(a, 2)
    esq = numpy.power(e, 2)

    b = numpy.sqrt(asq * (1 - esq))
    bsq = numpy.power(b, 2)

    ep = numpy.sqrt((asq - bsq) / bsq)
    p = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
    th = numpy.arctan2(a * z, b * p)

    lon = numpy.arctan2(y, x)
    lat = numpy.arctan2((z + numpy.power(ep, 2) * b * numpy.power(numpy.sin(th), 3)),
                        (p - esq * a * numpy.power(numpy.cos(th), 3)))
    N = a / (numpy.sqrt(1 - esq * numpy.power(numpy.sin(lat), 2)))
    alt = p / numpy.cos(lat) - N

    lon = lon * 180 / numpy.pi
    lat = lat * 180 / numpy.pi

    return numpy.array([lat]), numpy.array([lon]), numpy.array([alt])


def process_date(arg, missing_input='fill', allow_days=True):
    # function to handle date input from PG.
    # Input: arg = arguments from command line
    #        missing_input = a string specifying if vector should be filled when something is missing
    #        allow_day = allow a single argument which represents an integer N expressed in days, to compute now()-N

    if missing_input == 'fill':
        dates = [pyDate.Date(year=1980, doy=1), pyDate.Date(datetime=datetime.now())]
    else:
        dates = [None, None]

    if arg:
        for i, arg in enumerate(arg):
            try:
                if '.' in arg:
                    dates[i] = pyDate.Date(fyear=float(arg))
                elif '_' in arg:
                    dates[i] = pyDate.Date(year=int(arg.split('_')[0]), doy=int(arg.split('_')[1]))
                elif '/' in arg:
                    dates[i] = pyDate.Date(year=int(arg.split('/')[0]), month=int(arg.split('/')[1]),
                                           day=int(arg.split('/')[2]))
                elif '-' in arg:
                    dates[i] = pyDate.Date(gpsWeek=int(arg.split('-')[0]), gpsWeekDay=int(arg.split('-')[1]))
                elif len(arg) > 0:
                    if allow_days and i == 0:
                        dates[i] = pyDate.Date(datetime=datetime.now()) - int(arg)
                    else:
                        raise ValueError('Invalid input date: allow_days was set to False.')
            except Exception as e:
                raise ValueError('Could not decode input date (valid entries: '
                                 'fyear, yyyy_ddd, yyyy/mm/dd, gpswk-wkday). '
                                 'Error while reading the date start/end parameters: ' + str(e))

    return tuple(dates)


def determine_frame(frames, date):
    for frame in frames:
        if frame['dates'][0] <= date <= frame['dates'][1]:
            return frame['name'], frame['atx']

    raise Exception('No valid frame was found for the specified date.')


def print_columns(l):
    for a, b, c, d, e, f, g, h in zip(l[::8], l[1::8], l[2::8], l[3::8], l[4::8], l[5::8], l[6::8], l[7::8]):
        print(('    {:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<}'.format(a, b, c, d, e, f, g, h)))

    if len(l) % 8 != 0:
        sys.stdout.write('    ')
        for i in range(len(l) - len(l) % 8, len(l)):
            sys.stdout.write('{:<10}'.format(l[i]))
        sys.stdout.write('\n')


def get_resource_delimiter():
    return '.'


def process_stnlist(cnn, stnlist_in, print_summary=True):
    if len(stnlist_in) == 1 and os.path.isfile(stnlist_in[0]):
        print((' >> Station list read from file: ' + stnlist_in[0]))
        stnlist_in = [line.strip() for line in open(stnlist_in[0], 'r')]

    stnlist = []

    if len(stnlist_in) == 1 and stnlist_in[0] == 'all':
        # all stations
        rs = cnn.query('SELECT * FROM stations WHERE "NetworkCode" NOT LIKE \'?%%\' '
                       'ORDER BY "NetworkCode", "StationCode"')

        for rstn in rs.dictresult():
            stnlist += [{'NetworkCode': rstn['NetworkCode'], 'StationCode': rstn['StationCode']}]

    else:
        for stn in stnlist_in:
            rs = None
            if '.' in stn and '-' not in stn:
                # a net.stnm given
                if stn.split('.')[1] == 'all':
                    # all stations from a network
                    rs = cnn.query('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND '
                                   '"NetworkCode" NOT LIKE \'?%%\' ORDER BY "NetworkCode", "StationCode"'
                                   % (stn.split('.')[0]))

                else:
                    rs = cnn.query(
                        'SELECT * FROM stations WHERE "NetworkCode" NOT LIKE \'?%%\' AND "NetworkCode" = \'%s\' '
                        'AND "StationCode" = \'%s\' ORDER BY "NetworkCode", "StationCode"'
                        % (stn.split('.')[0], stn.split('.')[1]))

            elif '.' not in stn and '-' not in stn:
                # just a station name
                rs = cnn.query(
                    'SELECT * FROM stations WHERE "NetworkCode" NOT LIKE \'?%%\' AND '
                    '"StationCode" = \'%s\' ORDER BY "NetworkCode", "StationCode"' % stn)

            if rs is not None:
                for rstn in rs.dictresult():
                    if {'NetworkCode': rstn['NetworkCode'], 'StationCode': rstn['StationCode']} not in stnlist:
                        stnlist += [{'NetworkCode': rstn['NetworkCode'], 'StationCode': rstn['StationCode']}]

    # deal with station removals (-)
    for stn in [stn.replace('-', '') for stn in stnlist_in if '-' in stn]:
        # if netcode not given, remove everybody with that station code
        if '.' in stn.lower():
            stnlist = [stnl for stnl in stnlist if stnl['NetworkCode'] + '.' + stnl['StationCode'] != stn.lower()]
        else:
            stnlist = [stnl for stnl in stnlist if stnl['StationCode'] != stn.lower()]

    if print_summary:
        print(' >> Selected station list:')
        print_columns([item['NetworkCode'] + '.' + item['StationCode'] for item in stnlist])

    return stnlist


def get_norm_year_str(year):
    # mk 4 digit year
    try:
        year = int(year)
    except Exception:
        raise UtilsException('must provide a positive integer year YY or YYYY');

    # defensively, make sure that the year is positive
    if year < 0:
        raise UtilsException('must provide a positive integer year YY or YYYY');

    if 80 <= year <= 99:
        year += 1900
    elif 0 <= year < 80:
        year += 2000

    return str(year)


def get_norm_doy_str(doy):
    try:
        doy = int(doy)
    except Exception:
        raise UtilsException('must provide an integer day of year');

        # create string version up fround
    doy = str(doy);

    # mk 3 diit doy
    if len(doy) == 1:
        doy = "00" + doy
    elif len(doy) == 2:
        doy = "0" + doy
    return doy


def parseIntSet(nputstr=""):
    selection = []
    invalid = []
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        if len(i) > 0:
            if i[:1] == "<":
                i = "1-%s" % (i[1:])
        try:
            # typically tokens are plain old integers
            selection.append(int(i))
        except Exception:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.append(x)
            except Exception:
                # not an int and not a range...
                invalid.append(i)
    # Report invalid tokens before returning valid selection
    if len(invalid) > 0:
        print(("Invalid set: " + str(invalid)))
        sys.exit(2)
    return selection


def parse_stnId(stnId):
    # parse the station id
    parts = re.split('\.', stnId);

    # make sure at least two components here
    if len(parts) < 2:
        raise UtilsException('invalid station id: ' + stnId);

    # get station name space
    ns = '.'.join(parts[:-1]);

    # get the station code
    code = parts[-1];

    # that's it
    return ns, code;


def get_platform_id():
    # ask the os for platform information
    uname = os.uname();

    # combine to form the platform identification
    return '.'.join((uname[0], uname[2], uname[4]));


def get_processor_count():
    # init to null
    num_cpu = None;

    # ok, lets get some operating system info
    uname = os.uname();

    if uname[0].lower() == 'linux':

        # open the system file and read the lines
        with open('/proc/cpuinfo') as fid:
            nstr = sum([l.strip().replace('\t', '').split(':')[0] == 'core id' for l in fid.readlines()]);

    elif uname[0].lower() == 'darwin':
        nstr = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'], stdout=subprocess.PIPE).communicate()[0];
    else:
        raise UtilsException('Unrecognized/Unsupported operating system');

        # try to turn the process response into an integer
    try:
        num_cpu = int(nstr)
    except Exception:
        # nothing else we can do here
        num_cpu = None

    # that's all folks
    # return the number of PHYSICAL CORES, not the logical number (usually double)
    return num_cpu / 2


def human_readable_time(secs):
    # start with work time in seconds
    unit = 'secs';
    time = secs

    # make human readable work time with units
    if time > 60 and time < 3600:
        time = time / 60.0;
        unit = 'mins'
    elif time > 3600:
        time = time / 3600.0;
        unit = 'hours';

    return time, unit


def fix_gps_week(file_path):
    # example:  g017321.snx.gz --> g0107321.snx.gz

    # extract the full file name
    path, full_file_name = os.path.split(file_path);

    # init
    file_name = full_file_name;
    file_ext = '';
    ext = None;

    # remove all file extensions
    while ext != '':
        file_name, ext = os.path.splitext(file_name);
        file_ext = ext + file_ext;

    # if the name is short 1 character then add zero
    if len(file_name) == 7:
        file_name = file_name[0:3] + '0' + file_name[3:];

    # reconstruct file path
    return os.path.join(path, file_name + file_ext);


"""
Read and write ZIP files.
"""
import binascii
import io
import os
import re
import shutil
import stat
import string
import struct
import sys
import time

try:
    import zlib  # We may need its compression method

    crc32 = zlib.crc32
except ImportError:
    zlib = None
    crc32 = binascii.crc32

__all__ = ["BadZipfile", "error", "ZIP_STORED", "ZIP_DEFLATED", "is_zipfile",
           "ZipInfo", "ZipFile", "PyZipFile", "LargeZipFile"]


class BadZipfile(Exception):
    pass


class LargeZipFile(Exception):
    """
    Raised when writing a zipfile, the zipfile requires ZIP64 extensions
    and those extensions are disabled.
    """


error = BadZipfile  # The exception raised by this module

ZIP64_LIMIT = (1 << 31) - 1
ZIP_FILECOUNT_LIMIT = (1 << 16) - 1
ZIP_MAX_COMMENT = (1 << 16) - 1

# constants for Zip file compression methods
ZIP_STORED = 0
ZIP_DEFLATED = 8
# Other ZIP compression methods not supported

# Below are some formats and associated data for reading/writing headers using
# the struct module.  The names and structures of headers/records are those used
# in the PKWARE description of the ZIP file format:
#     http://www.pkware.com/documents/casestudies/APPNOTE.TXT
# (URL valid as of January 2008)

# The "end of central directory" structure, magic number, size, and indices
# (section V.I in the format document)
structEndArchive = "<4s4H2LH"
stringEndArchive = "PK\005\006"
sizeEndCentDir = struct.calcsize(structEndArchive)

_ECD_SIGNATURE = 0
_ECD_DISK_NUMBER = 1
_ECD_DISK_START = 2
_ECD_ENTRIES_THIS_DISK = 3
_ECD_ENTRIES_TOTAL = 4
_ECD_SIZE = 5
_ECD_OFFSET = 6
_ECD_COMMENT_SIZE = 7
# These last two indices are not part of the structure as defined in the
# spec, but they are used internally by this module as a convenience
_ECD_COMMENT = 8
_ECD_LOCATION = 9

# The "central directory" structure, magic number, size, and indices
# of entries in the structure (section V.F in the format document)
structCentralDir = "<4s4B4HL2L5H2L"
stringCentralDir = "PK\001\002"
sizeCentralDir = struct.calcsize(structCentralDir)

# indexes of entries in the central directory structure
_CD_SIGNATURE = 0
_CD_CREATE_VERSION = 1
_CD_CREATE_SYSTEM = 2
_CD_EXTRACT_VERSION = 3
_CD_EXTRACT_SYSTEM = 4
_CD_FLAG_BITS = 5
_CD_COMPRESS_TYPE = 6
_CD_TIME = 7
_CD_DATE = 8
_CD_CRC = 9
_CD_COMPRESSED_SIZE = 10
_CD_UNCOMPRESSED_SIZE = 11
_CD_FILENAME_LENGTH = 12
_CD_EXTRA_FIELD_LENGTH = 13
_CD_COMMENT_LENGTH = 14
_CD_DISK_NUMBER_START = 15
_CD_INTERNAL_FILE_ATTRIBUTES = 16
_CD_EXTERNAL_FILE_ATTRIBUTES = 17
_CD_LOCAL_HEADER_OFFSET = 18

# The "local file header" structure, magic number, size, and indices
# (section V.A in the format document)
structFileHeader = "<4s2B4HL2L2H"
stringFileHeader = "PK\003\004"
sizeFileHeader = struct.calcsize(structFileHeader)

_FH_SIGNATURE = 0
_FH_EXTRACT_VERSION = 1
_FH_EXTRACT_SYSTEM = 2
_FH_GENERAL_PURPOSE_FLAG_BITS = 3
_FH_COMPRESSION_METHOD = 4
_FH_LAST_MOD_TIME = 5
_FH_LAST_MOD_DATE = 6
_FH_CRC = 7
_FH_COMPRESSED_SIZE = 8
_FH_UNCOMPRESSED_SIZE = 9
_FH_FILENAME_LENGTH = 10
_FH_EXTRA_FIELD_LENGTH = 11

# The "Zip64 end of central directory locator" structure, magic number, and size
structEndArchive64Locator = "<4sLQL"
stringEndArchive64Locator = "PK\x06\x07"
sizeEndCentDir64Locator = struct.calcsize(structEndArchive64Locator)

# The "Zip64 end of central directory" record, magic number, size, and indices
# (section V.G in the format document)
structEndArchive64 = "<4sQ2H2L4Q"
stringEndArchive64 = "PK\x06\x06"
sizeEndCentDir64 = struct.calcsize(structEndArchive64)

_CD64_SIGNATURE = 0
_CD64_DIRECTORY_RECSIZE = 1
_CD64_CREATE_VERSION = 2
_CD64_EXTRACT_VERSION = 3
_CD64_DISK_NUMBER = 4
_CD64_DISK_NUMBER_START = 5
_CD64_NUMBER_ENTRIES_THIS_DISK = 6
_CD64_NUMBER_ENTRIES_TOTAL = 7
_CD64_DIRECTORY_SIZE = 8
_CD64_OFFSET_START_CENTDIR = 9


def _check_zipfile(fp):
    try:
        if _EndRecData(fp):
            return True  # file has correct magic number
    except IOError:
        pass
    return False


def is_zipfile(filename):
    """Quickly see if a file is a ZIP file by checking the magic number.

    The filename argument may be a file or file-like object too.
    """
    result = False
    try:
        if hasattr(filename, "read"):
            result = _check_zipfile(fp=filename)
        else:
            with open(filename, "rb") as fp:
                result = _check_zipfile(fp)
    except IOError:
        pass
    return result


def _EndRecData64(fpin, offset, endrec):
    """
    Read the ZIP64 end-of-archive records and use that to update endrec
    """
    try:
        fpin.seek(offset - sizeEndCentDir64Locator, 2)
    except IOError:
        # If the seek fails, the file is not large enough to contain a ZIP64
        # end-of-archive record, so just return the end record we were given.
        return endrec

    data = fpin.read(sizeEndCentDir64Locator)
    if len(data) != sizeEndCentDir64Locator:
        return endrec
    sig, diskno, reloff, disks = struct.unpack(structEndArchive64Locator, data)
    if sig != stringEndArchive64Locator:
        return endrec

    if diskno != 0 or disks != 1:
        raise BadZipfile("zipfiles that span multiple disks are not supported")

    # Assume no 'zip64 extensible data'
    fpin.seek(offset - sizeEndCentDir64Locator - sizeEndCentDir64, 2)
    data = fpin.read(sizeEndCentDir64)
    if len(data) != sizeEndCentDir64:
        return endrec
    sig, sz, create_version, read_version, disk_num, disk_dir, \
    dircount, dircount2, dirsize, diroffset = \
        struct.unpack(structEndArchive64, data)
    if sig != stringEndArchive64:
        return endrec

    # Update the original endrec using data from the ZIP64 record
    endrec[_ECD_SIGNATURE] = sig
    endrec[_ECD_DISK_NUMBER] = disk_num
    endrec[_ECD_DISK_START] = disk_dir
    endrec[_ECD_ENTRIES_THIS_DISK] = dircount
    endrec[_ECD_ENTRIES_TOTAL] = dircount2
    endrec[_ECD_SIZE] = dirsize
    endrec[_ECD_OFFSET] = diroffset
    return endrec


def _EndRecData(fpin):
    """Return data from the "End of Central Directory" record, or None.

    The data is a list of the nine items in the ZIP "End of central dir"
    record followed by a tenth item, the file seek offset of this record."""

    # Determine file size
    fpin.seek(0, 2)
    filesize = fpin.tell()

    # Check to see if this is ZIP file with no archive comment (the
    # "end of central directory" structure should be the last item in the
    # file if this is the case).
    try:
        fpin.seek(-sizeEndCentDir, 2)
    except IOError:
        return None
    data = fpin.read()
    if (len(data) == sizeEndCentDir and
            data[0:4] == stringEndArchive and
            data[-2:] == b"\000\000"):
        # the signature is correct and there's no comment, unpack structure
        endrec = struct.unpack(structEndArchive, data)
        endrec = list(endrec)

        # Append a blank comment and record start offset
        endrec.append("")
        endrec.append(filesize - sizeEndCentDir)

        # Try to read the "Zip64 end of central directory" structure
        return _EndRecData64(fpin, -sizeEndCentDir, endrec)

    # Either this is not a ZIP file, or it is a ZIP file with an archive
    # comment.  Search the end of the file for the "end of central directory"
    # record signature. The comment is the last item in the ZIP file and may be
    # up to 64K long.  It is assumed that the "end of central directory" magic
    # number does not appear in the comment.
    maxCommentStart = max(filesize - (1 << 16) - sizeEndCentDir, 0)
    fpin.seek(maxCommentStart, 0)
    data = fpin.read()
    start = data.rfind(stringEndArchive)
    if start >= 0:
        # found the magic number; attempt to unpack and interpret
        recData = data[start:start + sizeEndCentDir]
        if len(recData) != sizeEndCentDir:
            # Zip file is corrupted.
            return None
        endrec = list(struct.unpack(structEndArchive, recData))
        commentSize = endrec[_ECD_COMMENT_SIZE]  # as claimed by the zip file
        comment = data[start + sizeEndCentDir:start + sizeEndCentDir + commentSize]
        endrec.append(comment)
        endrec.append(maxCommentStart + start)

        # Try to read the "Zip64 end of central directory" structure
        return _EndRecData64(fpin, maxCommentStart + start - filesize,
                             endrec)

    # Unable to find a valid end of central directory structure
    return None


class ZipInfo(object):
    """Class with attributes describing each file in the ZIP archive."""

    __slots__ = (
        'orig_filename',
        'filename',
        'date_time',
        'compress_type',
        'comment',
        'extra',
        'create_system',
        'create_version',
        'extract_version',
        'reserved',
        'flag_bits',
        'volume',
        'internal_attr',
        'external_attr',
        'header_offset',
        'CRC',
        'compress_size',
        'file_size',
        '_raw_time',
    )

    def __init__(self, filename="NoName", date_time=(1980, 1, 1, 0, 0, 0)):
        self.orig_filename = filename  # Original file name in archive

        # Terminate the file name at the first null byte.  Null bytes in file
        # names are used as tricks by viruses in archives.
        null_byte = filename.find(chr(0))
        if null_byte >= 0:
            filename = filename[0:null_byte]
        # This is used to ensure paths in generated ZIP files always use
        # forward slashes as the directory separator, as required by the
        # ZIP format specification.
        if os.sep != "/" and os.sep in filename:
            filename = filename.replace(os.sep, "/")

        self.filename = filename  # Normalized file name
        self.date_time = date_time  # year, month, day, hour, min, sec

        if date_time[0] < 1980:
            raise ValueError('ZIP does not support timestamps before 1980')

        # Standard values:
        self.compress_type = ZIP_STORED  # Type of compression for the file
        self.comment = ""  # Comment for each file
        self.extra = ""  # ZIP extra data
        if sys.platform == 'win32':
            self.create_system = 0  # System which created ZIP archive
        else:
            # Assume everything else is unix-y
            self.create_system = 3  # System which created ZIP archive
        self.create_version = 20  # Version which created ZIP archive
        self.extract_version = 20  # Version needed to extract archive
        self.reserved = 0  # Must be zero
        self.flag_bits = 0  # ZIP flag bits
        self.volume = 0  # Volume number of file header
        self.internal_attr = 0  # Internal attributes
        self.external_attr = 0  # External file attributes
        # Other attributes are set by class ZipFile:
        # header_offset         Byte offset to the file header
        # CRC                   CRC-32 of the uncompressed file
        # compress_size         Size of the compressed file
        # file_size             Size of the uncompressed file

    def FileHeader(self, zip64=None):
        """Return the per-file header as a string."""
        dt = self.date_time
        dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]
        dostime = dt[3] << 11 | dt[4] << 5 | (dt[5] // 2)
        if self.flag_bits & 0x08:
            # Set these to zero because we write them after the file data
            CRC = compress_size = file_size = 0
        else:
            CRC = self.CRC
            compress_size = self.compress_size
            file_size = self.file_size

        extra = self.extra

        if zip64 is None:
            zip64 = file_size > ZIP64_LIMIT or compress_size > ZIP64_LIMIT
        if zip64:
            fmt = '<HHQQ'
            extra = extra + struct.pack(fmt,
                                        1, struct.calcsize(fmt) - 4, file_size, compress_size)
        if file_size > ZIP64_LIMIT or compress_size > ZIP64_LIMIT:
            if not zip64:
                raise LargeZipFile("Filesize would require ZIP64 extensions")
            # File is larger than what fits into a 4 byte integer,
            # fall back to the ZIP64 extension
            file_size = 0xffffffff
            compress_size = 0xffffffff
            self.extract_version = max(45, self.extract_version)
            self.create_version = max(45, self.extract_version)

        filename, flag_bits = self._encodeFilenameFlags()
        header = struct.pack(structFileHeader, stringFileHeader,
                             self.extract_version, self.reserved, flag_bits,
                             self.compress_type, dostime, dosdate, CRC,
                             compress_size, file_size,
                             len(filename), len(extra))
        return header + filename + extra

    def _encodeFilenameFlags(self):
        if isinstance(self.filename, str):
            try:
                return self.filename.encode('ascii'), self.flag_bits
            except UnicodeEncodeError:
                return self.filename.encode('utf-8'), self.flag_bits | 0x800
        else:
            return self.filename, self.flag_bits

    def _decodeFilename(self):
        if self.flag_bits & 0x800:
            return self.filename.decode('utf-8')
        else:
            return self.filename

    def _decodeExtra(self):
        # Try to decode the extra field.
        extra = self.extra
        unpack = struct.unpack
        while len(extra) >= 4:
            tp, ln = unpack('<HH', extra[:4])
            if tp == 1:
                if ln >= 24:
                    counts = unpack('<QQQ', extra[4:28])
                elif ln == 16:
                    counts = unpack('<QQ', extra[4:20])
                elif ln == 8:
                    counts = unpack('<Q', extra[4:12])
                elif ln == 0:
                    counts = ()
                else:
                    raise RuntimeError("Corrupt extra field %s" % (ln,))

                idx = 0

                # ZIP64 extension (large files and/or large archives)
                if self.file_size in (0xffffffffffffffff, 0xffffffff):
                    self.file_size = counts[idx]
                    idx += 1

                if self.compress_size == 0xFFFFFFFF:
                    self.compress_size = counts[idx]
                    idx += 1

                if self.header_offset == 0xffffffff:
                    old = self.header_offset
                    self.header_offset = counts[idx]
                    idx += 1

            extra = extra[ln + 4:]


class _ZipDecrypter:
    """Class to handle decryption of files stored within a ZIP archive.

    ZIP supports a password-based form of encryption. Even though known
    plaintext attacks have been found against it, it is still useful
    to be able to get data out of such a file.

    Usage:
        zd = _ZipDecrypter(mypwd)
        plain_char = zd(cypher_char)
        plain_text = map(zd, cypher_text)
    """

    def _GenerateCRCTable():
        """Generate a CRC-32 table.

        ZIP encryption uses the CRC32 one-byte primitive for scrambling some
        internal keys. We noticed that a direct implementation is faster than
        relying on binascii.crc32().
        """
        poly = 0xedb88320
        table = [0] * 256
        for i in range(256):
            crc = i
            for j in range(8):
                if crc & 1:
                    crc = ((crc >> 1) & 0x7FFFFFFF) ^ poly
                else:
                    crc = ((crc >> 1) & 0x7FFFFFFF)
            table[i] = crc
        return table

    crctable = _GenerateCRCTable()

    def _crc32(self, ch, crc):
        """Compute the CRC32 primitive on one byte."""
        return ((crc >> 8) & 0xffffff) ^ self.crctable[(crc ^ ord(ch)) & 0xff]

    def __init__(self, pwd):
        self.key0 = 305419896
        self.key1 = 591751049
        self.key2 = 878082192
        for p in pwd:
            self._UpdateKeys(p)

    def _UpdateKeys(self, c):
        self.key0 = self._crc32(c, self.key0)
        self.key1 = (self.key1 + (self.key0 & 255)) & 4294967295
        self.key1 = (self.key1 * 134775813 + 1) & 4294967295
        self.key2 = self._crc32(chr((self.key1 >> 24) & 255), self.key2)

    def __call__(self, c):
        """Decrypt a single character."""
        c = ord(c)
        k = self.key2 | 2
        c = c ^ (((k * (k ^ 1)) >> 8) & 255)
        c = chr(c)
        self._UpdateKeys(c)
        return c


compressor_names = {
    0: 'store',
    1: 'shrink',
    2: 'reduce',
    3: 'reduce',
    4: 'reduce',
    5: 'reduce',
    6: 'implode',
    7: 'tokenize',
    8: 'deflate',
    9: 'deflate64',
    10: 'implode',
    12: 'bzip2',
    14: 'lzma',
    18: 'terse',
    19: 'lz77',
    97: 'wavpack',
    98: 'ppmd',
}


class ZipExtFile(io.BufferedIOBase):
    """File-like object for reading an archive member.
       Is returned by ZipFile.open().
    """

    # Max size supported by decompressor.
    MAX_N = 1 << 31 - 1

    # Read from compressed files in 4k blocks.
    MIN_READ_SIZE = 4096

    # Search for universal newlines or line chunks.
    PATTERN = re.compile(r'^(?P<chunk>[^\r\n]+)|(?P<newline>\n|\r\n?)')

    def __init__(self, fileobj, mode, zipinfo, decrypter=None,
                 close_fileobj=False):
        self._fileobj = fileobj
        self._decrypter = decrypter
        self._close_fileobj = close_fileobj

        self._compress_type = zipinfo.compress_type
        self._compress_size = zipinfo.compress_size
        self._compress_left = zipinfo.compress_size

        if self._compress_type == ZIP_DEFLATED:
            self._decompressor = zlib.decompressobj(-15)
        elif self._compress_type != ZIP_STORED:
            descr = compressor_names.get(self._compress_type)
            if descr:
                raise NotImplementedError("compression type %d (%s)" % (self._compress_type, descr))
            else:
                raise NotImplementedError("compression type %d" % (self._compress_type,))
        self._unconsumed = ''

        self._readbuffer = ''
        self._offset = 0

        self._universal = 'U' in mode
        self.newlines = None

        # Adjust read size for encrypted files since the first 12 bytes
        # are for the encryption/password information.
        if self._decrypter is not None:
            self._compress_left -= 12

        self.mode = mode
        self.name = zipinfo.filename

        if hasattr(zipinfo, 'CRC'):
            self._expected_crc = zipinfo.CRC
            self._running_crc = crc32(b'') & 0xffffffff
        else:
            self._expected_crc = None

    def readline(self, limit=-1):
        """Read and return a line from the stream.

        If limit is specified, at most limit bytes will be read.
        """

        if not self._universal and limit < 0:
            # Shortcut common case - newline found in buffer.
            i = self._readbuffer.find('\n', self._offset) + 1
            if i > 0:
                line = self._readbuffer[self._offset: i]
                self._offset = i
                return line

        if not self._universal:
            return io.BufferedIOBase.readline(self, limit)

        line = ''
        while limit < 0 or len(line) < limit:
            readahead = self.peek(2)
            if readahead == '':
                return line

            #
            # Search for universal newlines or line chunks.
            #
            # The pattern returns either a line chunk or a newline, but not
            # both. Combined with peek(2), we are assured that the sequence
            # '\r\n' is always retrieved completely and never split into
            # separate newlines - '\r', '\n' due to coincidental readaheads.
            #
            match = self.PATTERN.search(readahead)
            newline = match.group('newline')
            if newline is not None:
                if self.newlines is None:
                    self.newlines = []
                if newline not in self.newlines:
                    self.newlines.append(newline)
                self._offset += len(newline)
                return line + '\n'

            chunk = match.group('chunk')
            if limit >= 0:
                chunk = chunk[: limit - len(line)]

            self._offset += len(chunk)
            line += chunk

        return line

    def peek(self, n=1):
        """Returns buffered bytes without advancing the position."""
        if n > len(self._readbuffer) - self._offset:
            chunk = self.read(n)
            if len(chunk) > self._offset:
                self._readbuffer = chunk + self._readbuffer[self._offset:]
                self._offset = 0
            else:
                self._offset -= len(chunk)

        # Return up to 512 bytes to reduce allocation overhead for tight loops.
        return self._readbuffer[self._offset: self._offset + 512]

    def readable(self):
        return True

    def read(self, n=-1):
        """Read and return up to n bytes.
        If the argument is omitted, None, or negative, data is read and returned until EOF is reached..
        """
        buf = ''
        if n is None:
            n = -1
        while True:
            if n < 0:
                data = self.read1(n)
            elif n > len(buf):
                data = self.read1(n - len(buf))
            else:
                return buf
            if len(data) == 0:
                return buf
            buf += data

    def _update_crc(self, newdata, eof):
        # Update the CRC using the given data.
        if self._expected_crc is None:
            # No need to compute the CRC if we don't have a reference value
            return
        self._running_crc = crc32(newdata, self._running_crc) & 0xffffffff
        # Check the CRC if we're at the end of the file
        if eof and self._running_crc != self._expected_crc:
            raise BadZipfile("Bad CRC-32 for file %r" % self.name)

    def read1(self, n):
        """Read up to n bytes with at most one read() system call."""

        # Simplify algorithm (branching) by transforming negative n to large n.
        if n < 0 or n is None:
            n = self.MAX_N

        # Bytes available in read buffer.
        len_readbuffer = len(self._readbuffer) - self._offset

        # Read from file.
        if self._compress_left > 0 and n > len_readbuffer + len(self._unconsumed):
            nbytes = n - len_readbuffer - len(self._unconsumed)
            nbytes = max(nbytes, self.MIN_READ_SIZE)
            nbytes = min(nbytes, self._compress_left)

            data = self._fileobj.read(nbytes)
            self._compress_left -= len(data)

            if data and self._decrypter is not None:
                data = ''.join(map(self._decrypter, data))

            if self._compress_type == ZIP_STORED:
                self._update_crc(data, eof=(self._compress_left == 0))
                self._readbuffer = self._readbuffer[self._offset:] + data
                self._offset = 0
            else:
                # Prepare deflated bytes for decompression.
                self._unconsumed += data

        # Handle unconsumed data.
        if (len(self._unconsumed) > 0 and n > len_readbuffer and
                self._compress_type == ZIP_DEFLATED):
            data = self._decompressor.decompress(
                self._unconsumed,
                max(n - len_readbuffer, self.MIN_READ_SIZE)
            )

            self._unconsumed = self._decompressor.unconsumed_tail
            eof = len(self._unconsumed) == 0 and self._compress_left == 0
            if eof:
                data += self._decompressor.flush()

            self._update_crc(data, eof=eof)
            self._readbuffer = self._readbuffer[self._offset:] + data
            self._offset = 0

        # Read from buffer.
        data = self._readbuffer[self._offset: self._offset + n]
        self._offset += len(data)
        return data

    def close(self):
        try:
            if self._close_fileobj:
                self._fileobj.close()
        finally:
            super(ZipExtFile, self).close()


class ZipFile(object):
    """ Class with methods to open, read, write, close, list zip files.

    z = ZipFile(file, mode="r", compression=ZIP_STORED, allowZip64=False)

    file: Either the path to the file, or a file-like object.
          If it is a path, the file will be opened and closed by ZipFile.
    mode: The mode can be either read "r", write "w" or append "a".
    compression: ZIP_STORED (no compression) or ZIP_DEFLATED (requires zlib).
    allowZip64: if True ZipFile will create files with ZIP64 extensions when
                needed, otherwise it will raise an exception when this would
                be necessary.

    """

    fp = None  # Set here since __del__ checks it

    def __init__(self, file, mode="r", compression=ZIP_STORED, allowZip64=False):
        """Open the ZIP file with mode read "r", write "w" or append "a"."""
        if mode not in ("r", "w", "a"):
            raise RuntimeError('ZipFile() requires mode "r", "w", or "a"')

        if compression == ZIP_STORED:
            pass
        elif compression == ZIP_DEFLATED:
            if not zlib:
                raise RuntimeError("Compression requires the (missing) zlib module")
        else:
            raise RuntimeError("That compression method is not supported")

        self._allowZip64 = allowZip64
        self._didModify = False
        self.debug = 0  # Level of printing: 0 through 3
        self.NameToInfo = {}  # Find file info given name
        self.filelist = []  # List of ZipInfo instances for archive
        self.compression = compression  # Method of compression
        self.mode = key = mode.replace('b', '')[0]
        self.pwd = None
        self._comment = ''

        # Check if we were passed a file-like object
        if isinstance(file, str):
            self._filePassed = 0
            self.filename = file
            modeDict = {'r': 'rb', 'w': 'wb', 'a': 'r+b'}
            try:
                self.fp = open(file, modeDict[mode])
            except IOError:
                if mode == 'a':
                    mode = key = 'w'
                    self.fp = open(file, modeDict[mode])
                else:
                    raise
        else:
            self._filePassed = 1
            self.fp = file
            self.filename = getattr(file, 'name', None)

        try:
            if key == 'r':
                self._RealGetContents()
            elif key == 'w':
                # set the modified flag so central directory gets written
                # even if no files are added to the archive
                self._didModify = True
                self._start_disk = 0
            elif key == 'a':
                try:
                    # See if file is a zip file
                    self._RealGetContents()
                    # seek to start of directory and overwrite
                    self.fp.seek(self.start_dir, 0)
                except BadZipfile:
                    # file is not a zip file, just append
                    self.fp.seek(0, 2)

                    # set the modified flag so central directory gets written
                    # even if no files are added to the archive
                    self._didModify = True
                    self._start_disk = self.fp.tell()
            else:
                raise RuntimeError('Mode must be "r", "w" or "a"')
        except:
            fp = self.fp
            self.fp = None
            if not self._filePassed:
                fp.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _RealGetContents(self):
        """Read in the table of contents for the ZIP file."""
        fp = self.fp
        try:
            endrec = _EndRecData(fp)
        except IOError:
            raise BadZipfile("File is not a zip file")
        if not endrec:
            raise BadZipfile("File is not a zip file")
        if self.debug > 1:
            print(endrec)
        size_cd = endrec[_ECD_SIZE]  # bytes in central directory
        offset_cd = endrec[_ECD_OFFSET]  # offset of central directory
        self._comment = endrec[_ECD_COMMENT]  # archive comment

        # self._start_disk:  Position of the start of ZIP archive
        # It is zero, unless ZIP was concatenated to another file
        self._start_disk = endrec[_ECD_LOCATION] - size_cd - offset_cd
        if endrec[_ECD_SIGNATURE] == stringEndArchive64:
            # If Zip64 extension structures are present, account for them
            self._start_disk -= (sizeEndCentDir64 + sizeEndCentDir64Locator)

        if self.debug > 2:
            inferred = self._start_disk + offset_cd
            print("given, inferred, offset", offset_cd, inferred, self._start_disk)
        # self.start_dir:  Position of start of central directory
        self.start_dir = offset_cd + self._start_disk
        fp.seek(self.start_dir, 0)
        data = fp.read(size_cd)
        fp = io.StringIO(data)
        total = 0
        while total < size_cd:
            centdir = fp.read(sizeCentralDir)
            if len(centdir) != sizeCentralDir:
                raise BadZipfile("Truncated central directory")
            centdir = struct.unpack(structCentralDir, centdir)
            if centdir[_CD_SIGNATURE] != stringCentralDir:
                raise BadZipfile("Bad magic number for central directory")
            if self.debug > 2:
                print(centdir)
            filename = fp.read(centdir[_CD_FILENAME_LENGTH])
            # Create ZipInfo instance to store file information
            x = ZipInfo(filename)
            x.extra = fp.read(centdir[_CD_EXTRA_FIELD_LENGTH])
            x.comment = fp.read(centdir[_CD_COMMENT_LENGTH])
            x.header_offset = centdir[_CD_LOCAL_HEADER_OFFSET]
            (x.create_version, x.create_system, x.extract_version, x.reserved,
             x.flag_bits, x.compress_type, t, d,
             x.CRC, x.compress_size, x.file_size) = centdir[1:12]
            x.volume, x.internal_attr, x.external_attr = centdir[15:18]
            # Convert date/time code to (year, month, day, hour, min, sec)
            x._raw_time = t
            x.date_time = ((d >> 9) + 1980, (d >> 5) & 0xF, d & 0x1F,
                           t >> 11, (t >> 5) & 0x3F, (t & 0x1F) * 2)

            x._decodeExtra()
            x.header_offset = x.header_offset + self._start_disk
            x.filename = x._decodeFilename()
            self.filelist.append(x)
            self.NameToInfo[x.filename] = x

            # update total bytes read from central directory
            total = (total + sizeCentralDir + centdir[_CD_FILENAME_LENGTH]
                     + centdir[_CD_EXTRA_FIELD_LENGTH]
                     + centdir[_CD_COMMENT_LENGTH])

            if self.debug > 2:
                print("total", total)

    def namelist(self):
        """Return a list of file names in the archive."""
        l = []
        for data in self.filelist:
            l.append(data.filename)
        return l

    def infolist(self):
        """Return a list of class ZipInfo instances for files in the
        archive."""
        return self.filelist

    def printdir(self):
        """Print a table of contents for the zip file."""
        print("%-46s %19s %12s" % ("File Name", "Modified    ", "Size"))
        for zinfo in self.filelist:
            date = "%d-%02d-%02d %02d:%02d:%02d" % zinfo.date_time[:6]
            print("%-46s %s %12d" % (zinfo.filename, date, zinfo.file_size))

    def testzip(self):
        """Read all the files and check the CRC."""
        chunk_size = 2 ** 20
        for zinfo in self.filelist:
            try:
                # Read by chunks, to avoid an OverflowError or a
                # MemoryError with very large embedded files.
                with self.open(zinfo.filename, "r") as f:
                    while f.read(chunk_size):  # Check CRC-32
                        pass
            except BadZipfile:
                return zinfo.filename

    def getinfo(self, name):
        """Return the instance of ZipInfo given 'name'."""
        info = self.NameToInfo.get(name)
        if info is None:
            raise KeyError(
                'There is no item named %r in the archive' % name)

        return info

    def setpassword(self, pwd):
        """Set default password for encrypted files."""
        self.pwd = pwd

    @property
    def comment(self):
        """The comment text associated with the ZIP file."""
        return self._comment

    @comment.setter
    def comment(self, comment):
        # check for valid comment length
        if len(comment) > ZIP_MAX_COMMENT:
            import warnings
            warnings.warn('Archive comment is too long; truncating to %d bytes'
                          % ZIP_MAX_COMMENT, stacklevel=2)
            comment = comment[:ZIP_MAX_COMMENT]
        self._comment = comment
        self._didModify = True

    def read(self, name, pwd=None):
        """Return file bytes (as a string) for name."""
        return self.open(name, "r", pwd).read()

    def open(self, name, mode="r", pwd=None):
        """Return file-like object for 'name'."""
        if mode not in ("r", "U", "rU"):
            raise RuntimeError('open() requires mode "r", "U", or "rU"')
        if not self.fp:
            raise RuntimeError("Attempt to read ZIP archive that was already closed")

        # Only open a new file for instances where we were not
        # given a file object in the constructor
        if self._filePassed:
            zef_file = self.fp
            should_close = False
        else:
            zef_file = open(self.filename, 'rb')
            should_close = True

        try:
            # Make sure we have an info object
            if isinstance(name, ZipInfo):
                # 'name' is already an info object
                zinfo = name
            else:
                # Get info object for name
                zinfo = self.getinfo(name)

            zef_file.seek(zinfo.header_offset, 0)

            # Skip the file header:
            fheader = zef_file.read(sizeFileHeader)
            if len(fheader) != sizeFileHeader:
                raise BadZipfile("Truncated file header")
            fheader = struct.unpack(structFileHeader, fheader)
            if fheader[_FH_SIGNATURE] != stringFileHeader:
                raise BadZipfile("Bad magic number for file header")

            fname = zef_file.read(fheader[_FH_FILENAME_LENGTH])
            if fheader[_FH_EXTRA_FIELD_LENGTH]:
                zef_file.read(fheader[_FH_EXTRA_FIELD_LENGTH])

            if fname != zinfo.orig_filename:
                raise BadZipfile('File name in directory "%s" and header "%s" differ.' % (
                    zinfo.orig_filename, fname))

            # check for encrypted flag & handle password
            is_encrypted = zinfo.flag_bits & 0x1
            zd = None
            if is_encrypted:
                if not pwd:
                    pwd = self.pwd
                if not pwd:
                    raise RuntimeError("File %s is encrypted, " \
                                       "password required for extraction" % name)

                zd = _ZipDecrypter(pwd)
                # The first 12 bytes in the cypher stream is an encryption header
                #  used to strengthen the algorithm. The first 11 bytes are
                #  completely random, while the 12th contains the MSB of the CRC,
                #  or the MSB of the file time depending on the header type
                #  and is used to check the correctness of the password.
                bytes = zef_file.read(12)
                h = list(map(zd, bytes[0:12]))
                if zinfo.flag_bits & 0x8:
                    # compare against the file type from extended local headers
                    check_byte = (zinfo._raw_time >> 8) & 0xff
                else:
                    # compare against the CRC otherwise
                    check_byte = (zinfo.CRC >> 24) & 0xff
                if ord(h[11]) != check_byte:
                    raise RuntimeError("Bad password for file", name)

            return ZipExtFile(zef_file, mode, zinfo, zd,
                              close_fileobj=should_close)
        except:
            if should_close:
                zef_file.close()
            raise

    def extract(self, member, path=None, pwd=None):
        """Extract a member from the archive to the current working directory,
           using its full name. Its file information is extracted as accurately
           as possible. `member' may be a filename or a ZipInfo object. You can
           specify a different directory using `path'.
        """
        if not isinstance(member, ZipInfo):
            member = self.getinfo(member)

        if path is None:
            path = os.getcwd()

        return self._extract_member(member, path, pwd)

    def extractall(self, path=None, members=None, pwd=None):
        """Extract all members from the archive to the current working
           directory. `path' specifies a different directory to extract to.
           `members' is optional and must be a subset of the list returned
           by namelist().
        """
        if members is None:
            members = self.namelist()

        for zipinfo in members:
            self.extract(zipinfo, path, pwd)

    def _extract_member(self, member, targetpath, pwd):
        """Extract the ZipInfo object 'member' to a physical
           file on the path targetpath.
        """
        # build the destination pathname, replacing
        # forward slashes to platform specific separators.
        arcname = member.filename.replace('/', os.path.sep)

        if os.path.altsep:
            arcname = arcname.replace(os.path.altsep, os.path.sep)
        # interpret absolute pathname as relative, remove drive letter or
        # UNC path, redundant separators, "." and ".." components.
        arcname = os.path.splitdrive(arcname)[1]
        arcname = os.path.sep.join(x for x in arcname.split(os.path.sep)
                                   if x not in ('', os.path.curdir, os.path.pardir))
        if os.path.sep == '\\':
            # filter illegal characters on Windows
            illegal = ':<>|"?*'
            if isinstance(arcname, str):
                table = {ord(c): ord('_') for c in illegal}
            else:
                table = string.maketrans(illegal, '_' * len(illegal))
            arcname = arcname.translate(table)
            # remove trailing dots
            arcname = (x.rstrip('.') for x in arcname.split(os.path.sep))
            arcname = os.path.sep.join(x for x in arcname if x)

        targetpath = os.path.join(targetpath, arcname)
        targetpath = os.path.normpath(targetpath)

        # Create all upper directories if necessary.
        upperdirs = os.path.dirname(targetpath)
        if upperdirs and not os.path.exists(upperdirs):
            os.makedirs(upperdirs)

        if member.filename[-1] == '/':
            if not os.path.isdir(targetpath):
                os.mkdir(targetpath)
            return targetpath

        with self.open(member, pwd=pwd) as source, \
                file(targetpath, "wb") as target:
            shutil.copyfileobj(source, target)

        return targetpath

    def _writecheck(self, zinfo):
        """Check for errors before writing a file to the archive."""
        if zinfo.filename in self.NameToInfo:
            import warnings
            warnings.warn('Duplicate name: %r' % zinfo.filename, stacklevel=3)
        if self.mode not in ("w", "a"):
            raise RuntimeError('write() requires mode "w" or "a"')
        if not self.fp:
            raise RuntimeError("Attempt to write ZIP archive that was already closed")
        if zinfo.compress_type == ZIP_DEFLATED and not zlib:
            raise RuntimeError("Compression requires the (missing) zlib module")
        if zinfo.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
            raise RuntimeError("That compression method is not supported")
        if not self._allowZip64:
            requires_zip64 = None
            if len(self.filelist) >= ZIP_FILECOUNT_LIMIT:
                requires_zip64 = "Files count"
            elif zinfo.file_size > ZIP64_LIMIT:
                requires_zip64 = "Filesize"
            elif zinfo.header_offset > ZIP64_LIMIT:
                requires_zip64 = "Zipfile size"
            if requires_zip64:
                raise LargeZipFile(requires_zip64 +
                                   " would require ZIP64 extensions")

    def write(self, filename, arcname=None, compress_type=None):
        """Put the bytes from filename into the archive under the name
        arcname."""
        if not self.fp:
            raise RuntimeError(
                "Attempt to write to ZIP archive that was already closed")

        st = os.stat(filename)
        isdir = stat.S_ISDIR(st.st_mode)
        mtime = time.localtime(st.st_mtime)
        date_time = mtime[0:6]
        # Create ZipInfo instance to store file information
        if arcname is None:
            arcname = filename
        arcname = os.path.normpath(os.path.splitdrive(arcname)[1])
        while arcname[0] in (os.sep, os.altsep):
            arcname = arcname[1:]
        if isdir:
            arcname += '/'
        zinfo = ZipInfo(arcname, date_time)
        zinfo.external_attr = (st[0] & 0xFFFF) << 16  # Unix attributes
        if isdir:
            zinfo.compress_type = ZIP_STORED
        elif compress_type is None:
            zinfo.compress_type = self.compression
        else:
            zinfo.compress_type = compress_type

        zinfo.file_size = st.st_size
        zinfo.flag_bits = 0x00
        zinfo.header_offset = self.fp.tell()  # Start of header bytes

        self._writecheck(zinfo)
        self._didModify = True

        if isdir:
            zinfo.file_size = 0
            zinfo.compress_size = 0
            zinfo.CRC = 0
            zinfo.external_attr |= 0x10  # MS-DOS directory flag
            self.filelist.append(zinfo)
            self.NameToInfo[zinfo.filename] = zinfo
            self.fp.write(zinfo.FileHeader(False))
            return

        with open(filename, "rb") as fp:
            # Must overwrite CRC and sizes with correct data later
            zinfo.CRC = CRC = 0
            zinfo.compress_size = compress_size = 0
            # Compressed size can be larger than uncompressed size
            zip64 = self._allowZip64 and \
                    zinfo.file_size * 1.05 > ZIP64_LIMIT
            self.fp.write(zinfo.FileHeader(zip64))
            if zinfo.compress_type == ZIP_DEFLATED:
                cmpr = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION,
                                        zlib.DEFLATED, -15)
            else:
                cmpr = None
            file_size = 0
            while 1:
                buf = fp.read(1024 * 8)
                if not buf:
                    break
                file_size = file_size + len(buf)
                CRC = crc32(buf, CRC) & 0xffffffff
                if cmpr:
                    buf = cmpr.compress(buf)
                    compress_size = compress_size + len(buf)
                self.fp.write(buf)
        if cmpr:
            buf = cmpr.flush()
            compress_size = compress_size + len(buf)
            self.fp.write(buf)
            zinfo.compress_size = compress_size
        else:
            zinfo.compress_size = file_size
        zinfo.CRC = CRC
        zinfo.file_size = file_size
        if not zip64 and self._allowZip64:
            if file_size > ZIP64_LIMIT:
                raise RuntimeError('File size has increased during compressing')
            if compress_size > ZIP64_LIMIT:
                raise RuntimeError('Compressed size larger than uncompressed size')
        # Seek backwards and write file header (which will now include
        # correct CRC and file sizes)
        position = self.fp.tell()  # Preserve current position in file
        self.fp.seek(zinfo.header_offset, 0)
        self.fp.write(zinfo.FileHeader(zip64))
        self.fp.seek(position, 0)
        self.filelist.append(zinfo)
        self.NameToInfo[zinfo.filename] = zinfo

    def writestr(self, zinfo_or_arcname, bytes, compress_type=None):
        """Write a file into the archive.  The contents is the string
        'bytes'.  'zinfo_or_arcname' is either a ZipInfo instance or
        the name of the file in the archive."""
        if not isinstance(zinfo_or_arcname, ZipInfo):
            zinfo = ZipInfo(filename=zinfo_or_arcname,
                            date_time=time.localtime(time.time())[:6])

            zinfo.compress_type = self.compression
            if zinfo.filename[-1] == '/':
                zinfo.external_attr = 0o40775 << 16  # drwxrwxr-x
                zinfo.external_attr |= 0x10  # MS-DOS directory flag
            else:
                zinfo.external_attr = 0o600 << 16  # ?rw-------
        else:
            zinfo = zinfo_or_arcname

        if not self.fp:
            raise RuntimeError(
                "Attempt to write to ZIP archive that was already closed")

        if compress_type is not None:
            zinfo.compress_type = compress_type

        zinfo.file_size = len(bytes)  # Uncompressed size
        zinfo.header_offset = self.fp.tell()  # Start of header bytes
        self._writecheck(zinfo)
        self._didModify = True
        zinfo.CRC = crc32(bytes) & 0xffffffff  # CRC-32 checksum
        if zinfo.compress_type == ZIP_DEFLATED:
            co = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION,
                                  zlib.DEFLATED, -15)
            bytes = co.compress(bytes) + co.flush()
            zinfo.compress_size = len(bytes)  # Compressed size
        else:
            zinfo.compress_size = zinfo.file_size
        zip64 = zinfo.file_size > ZIP64_LIMIT or \
                zinfo.compress_size > ZIP64_LIMIT
        if zip64 and not self._allowZip64:
            raise LargeZipFile("Filesize would require ZIP64 extensions")
        self.fp.write(zinfo.FileHeader(zip64))
        self.fp.write(bytes)
        if zinfo.flag_bits & 0x08:
            # Write CRC and file sizes after the file data
            fmt = '<LQQ' if zip64 else '<LLL'
            self.fp.write(struct.pack(fmt, zinfo.CRC, zinfo.compress_size,
                                      zinfo.file_size))
        self.fp.flush()
        self.filelist.append(zinfo)
        self.NameToInfo[zinfo.filename] = zinfo

    def __del__(self):
        """Call the "close()" method in case the user forgot."""
        self.close()

    def close(self):
        """Close the file, and for mode "w" and "a" write the ending
        records."""
        if self.fp is None:
            return

        try:
            if self.mode in ("w", "a") and self._didModify:  # write ending records
                pos1 = self.fp.tell()
                for zinfo in self.filelist:  # write central directory
                    dt = zinfo.date_time
                    dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]
                    dostime = dt[3] << 11 | dt[4] << 5 | (dt[5] // 2)
                    extra = []
                    if zinfo.file_size > ZIP64_LIMIT \
                            or zinfo.compress_size > ZIP64_LIMIT:
                        extra.append(zinfo.file_size)
                        extra.append(zinfo.compress_size)
                        file_size = 0xffffffff
                        compress_size = 0xffffffff
                    else:
                        file_size = zinfo.file_size
                        compress_size = zinfo.compress_size

                    header_offset = zinfo.header_offset - self._start_disk
                    if header_offset > ZIP64_LIMIT:
                        extra.append(header_offset)
                        header_offset = 0xffffffff

                    extra_data = zinfo.extra
                    if extra:
                        # Append a ZIP64 field to the extra's
                        extra_data = struct.pack(
                            '<HH' + 'Q' * len(extra),
                            1, 8 * len(extra), *extra) + extra_data

                        extract_version = max(45, zinfo.extract_version)
                        create_version = max(45, zinfo.create_version)
                    else:
                        extract_version = zinfo.extract_version
                        create_version = zinfo.create_version

                    try:
                        filename, flag_bits = zinfo._encodeFilenameFlags()
                        centdir = struct.pack(structCentralDir,
                                              stringCentralDir, create_version,
                                              zinfo.create_system, extract_version, zinfo.reserved,
                                              flag_bits, zinfo.compress_type, dostime, dosdate,
                                              zinfo.CRC, compress_size, file_size,
                                              len(filename), len(extra_data), len(zinfo.comment),
                                              0, zinfo.internal_attr, zinfo.external_attr,
                                              header_offset)
                    except DeprecationWarning:
                        print((structCentralDir,
                               stringCentralDir, create_version,
                               zinfo.create_system, extract_version, zinfo.reserved,
                               zinfo.flag_bits, zinfo.compress_type, dostime, dosdate,
                               zinfo.CRC, compress_size, file_size,
                               len(zinfo.filename), len(extra_data), len(zinfo.comment),
                               0, zinfo.internal_attr, zinfo.external_attr,
                               header_offset), file=sys.stderr)
                        raise
                    self.fp.write(centdir)
                    self.fp.write(filename)
                    self.fp.write(extra_data)
                    self.fp.write(zinfo.comment)

                pos2 = self.fp.tell()
                # Write end-of-zip-archive record
                centDirCount = len(self.filelist)
                centDirSize = pos2 - pos1
                centDirOffset = pos1 - self._start_disk
                requires_zip64 = None
                if centDirCount > ZIP_FILECOUNT_LIMIT:
                    requires_zip64 = "Files count"
                elif centDirOffset > ZIP64_LIMIT:
                    requires_zip64 = "Central directory offset"
                elif centDirSize > ZIP64_LIMIT:
                    requires_zip64 = "Central directory size"
                if requires_zip64:
                    # Need to write the ZIP64 end-of-archive records
                    if not self._allowZip64:
                        raise LargeZipFile(requires_zip64 +
                                           " would require ZIP64 extensions")
                    zip64endrec = struct.pack(
                        structEndArchive64, stringEndArchive64,
                        44, 45, 45, 0, 0, centDirCount, centDirCount,
                        centDirSize, centDirOffset)
                    self.fp.write(zip64endrec)

                    zip64locrec = struct.pack(
                        structEndArchive64Locator,
                        stringEndArchive64Locator, 0, pos2, 1)
                    self.fp.write(zip64locrec)
                    centDirCount = min(centDirCount, 0xFFFF)
                    centDirSize = min(centDirSize, 0xFFFFFFFF)
                    centDirOffset = min(centDirOffset, 0xFFFFFFFF)

                endrec = struct.pack(structEndArchive, stringEndArchive,
                                     0, 0, centDirCount, centDirCount,
                                     centDirSize, centDirOffset, len(self._comment))
                self.fp.write(endrec)
                self.fp.write(self._comment)
                self.fp.flush()
        finally:
            fp = self.fp
            self.fp = None
            if not self._filePassed:
                fp.close()


class PyZipFile(ZipFile):
    """Class to create ZIP archives with Python library files and packages."""

    def writepy(self, pathname, basename=""):
        """Add all files from "pathname" to the ZIP archive.

        If pathname is a package directory, search the directory and
        all package subdirectories recursively for all *.py and enter
        the modules into the archive.  If pathname is a plain
        directory, listdir *.py and enter all modules.  Else, pathname
        must be a Python *.py file and the module will be put into the
        archive.  Added modules are always module.pyo or module.pyc.
        This method will compile the module.py into module.pyc if
        necessary.
        """
        dir, name = os.path.split(pathname)
        if os.path.isdir(pathname):
            initname = os.path.join(pathname, "__init__.py")
            if os.path.isfile(initname):
                # This is a package directory, add it
                if basename:
                    basename = "%s/%s" % (basename, name)
                else:
                    basename = name
                if self.debug:
                    print("Adding package in", pathname, "as", basename)
                fname, arcname = self._get_codename(initname[0:-3], basename)
                if self.debug:
                    print("Adding", arcname)
                self.write(fname, arcname)
                dirlist = os.listdir(pathname)
                dirlist.remove("__init__.py")
                # Add all *.py files and package subdirectories
                for filename in dirlist:
                    path = os.path.join(pathname, filename)
                    root, ext = os.path.splitext(filename)
                    if os.path.isdir(path):
                        if os.path.isfile(os.path.join(path, "__init__.py")):
                            # This is a package directory, add it
                            self.writepy(path, basename)  # Recursive call
                    elif ext == ".py":
                        fname, arcname = self._get_codename(path[0:-3],
                                                            basename)
                        if self.debug:
                            print("Adding", arcname)
                        self.write(fname, arcname)
            else:
                # This is NOT a package directory, add its files at top level
                if self.debug:
                    print("Adding files from directory", pathname)
                for filename in os.listdir(pathname):
                    path = os.path.join(pathname, filename)
                    root, ext = os.path.splitext(filename)
                    if ext == ".py":
                        fname, arcname = self._get_codename(path[0:-3],
                                                            basename)
                        if self.debug:
                            print("Adding", arcname)
                        self.write(fname, arcname)
        else:
            if pathname[-3:] != ".py":
                raise RuntimeError('Files added with writepy() must end with ".py"')
            fname, arcname = self._get_codename(pathname[0:-3], basename)
            if self.debug:
                print("Adding file", arcname)
            self.write(fname, arcname)

    def _get_codename(self, pathname, basename):
        """Return (filename, archivename) for the path.

        Given a module name path, return the correct file path and
        archive name, compiling if necessary.  For example, given
        /python/lib/string, return (/python/lib/string.pyc, string).
        """
        file_py = pathname + ".py"
        file_pyc = pathname + ".pyc"
        file_pyo = pathname + ".pyo"
        if os.path.isfile(file_pyo) and \
                os.stat(file_pyo).st_mtime >= os.stat(file_py).st_mtime:
            fname = file_pyo  # Use .pyo file
        elif not os.path.isfile(file_pyc) or \
                os.stat(file_pyc).st_mtime < os.stat(file_py).st_mtime:
            import py_compile
            if self.debug:
                print("Compiling", file_py)
            try:
                py_compile.compile(file_py, file_pyc, None, True)
            except py_compile.PyCompileError as err:
                print(err.msg)
            fname = file_pyc
        else:
            fname = file_pyc
        archivename = os.path.split(fname)[1]
        if basename:
            archivename = "%s/%s" % (basename, archivename)
        return (fname, archivename)


def main(args=None):
    import textwrap
    USAGE = textwrap.dedent("""\
        Usage:
            zip_file.py -l zipfile.zip        # Show listing of a zipfile
            zip_file.py -t zipfile.zip        # Test if a zipfile is valid
            zip_file.py -e zipfile.zip target # Extract zipfile into target dir
            zip_file.py -c zipfile.zip src ... # Create zipfile from sources
        """)
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] not in ('-l', '-c', '-e', '-t'):
        print(USAGE)
        sys.exit(1)

    if args[0] == '-l':
        if len(args) != 2:
            print(USAGE)
            sys.exit(1)
        with ZipFile(args[1], 'r') as zf:
            zf.printdir()

    elif args[0] == '-t':
        if len(args) != 2:
            print(USAGE)
            sys.exit(1)
        with ZipFile(args[1], 'r') as zf:
            badfile = zf.testzip()
        if badfile:
            print(("The following enclosed file is corrupted: {!r}".format(badfile)))
        print("Done testing")

    elif args[0] == '-e':
        if len(args) != 3:
            print(USAGE)
            sys.exit(1)

        with ZipFile(args[1], 'r') as zf:
            zf.extractall(args[2])

    elif args[0] == '-c':
        if len(args) < 3:
            print(USAGE)
            sys.exit(1)

        def addToZip(zf, path, zippath):
            if os.path.isfile(path):
                zf.write(path, zippath, ZIP_DEFLATED)
            elif os.path.isdir(path):
                if zippath:
                    zf.write(path, zippath)
                for nm in os.listdir(path):
                    addToZip(zf,
                             os.path.join(path, nm), os.path.join(zippath, nm))
            # else: ignore

        with ZipFile(args[1], 'w', allowZip64=True) as zf:
            for path in args[2:]:
                zippath = os.path.basename(path)
                if not zippath:
                    zippath = os.path.basename(os.path.dirname(path))
                if zippath in ('', os.curdir, os.pardir):
                    zippath = ''
                addToZip(zf, path, zippath)
