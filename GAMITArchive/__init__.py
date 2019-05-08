import argparse
import configparser
import datetime as dt
import filecmp
import inspect
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zlib
import functools
import dirsync
import dispy
import dispy.httpd
import numpy
import psycopg2
import scandir
from collections import defaultdict
from tqdm import tqdm
from psycopg2 import sql
import logging

DELAY = 5
TYPE_CRINEZ = 0
TYPE_RINEX = 1
TYPE_RINEZ = 2
TYPE_CRINEX = 3


class UtilsException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class ProductsException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class DBErrInsert(Exception):
    pass


class RunPPPException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class DBErrUpdate(Exception):
    pass


class DBErrConnect(Exception):
    pass


class DBErrDelete(Exception):
    pass


class OTLException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class RinexException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class StationInfoException(Exception):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error')

    def __str__(self):
        return str(self.value)


class RunCommandWithRetryExeception(Exception):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class DateException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class Runpppexceptioncoordconflict(RunPPPException):
    pass


class Runpppexceptiontoofewacceptedobs(RunPPPException):
    pass


class Runpppexceptionnan(RunPPPException):
    pass


class Runpppexceptionzeroprocepochs(RunPPPException):
    pass


class Runpppexceptioneoperror(RunPPPException):
    pass


class Productsexceptionunreasonabledate(ProductsException):
    pass


class Rinexexceptionbadfile(RinexException):
    pass


class Rinexexceptionsingleepoch(RinexException):
    pass


class Rinexexceptionnoautocoord(RinexException):
    pass


class Stationinfoheightcodenotfound(StationInfoException):
    pass


class Clkexception(ProductsException):
    pass


class EOPException(ProductsException):
    def __init__(self, value):
        self.value = value
        self.event = Event(Description=value, EventType='error', module=type(self).__name__)

    def __str__(self):
        return str(self.value)


class Sp3exception(ProductsException):
    pass


class Brdcexception(ProductsException):
    pass


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


def copy_file(src, dst):
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
            os.makedirs(os.path.dirname(dst), exist_ok=True)
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


def move_new(src, dst):
    """
    Moves a file from path src to path dst.
    If a file already exists at dst, it will not be overwritten, but:
     * If it is the same as the source file, do nothing
     * If it is different to the source file, pick a new name for the copy that
       is distinct and unused, then copy the file there.
    Returns the path to the new file.
    """
    dst = shutil.copyfile(src, dst)
    os.remove(src)
    return dst


def lg2ct(dn, de, du, lat, lon):
    n = dn.size

    R = rotlg2ct(lat, lon, n)

    dxdydz = numpy.column_stack((numpy.column_stack((dn, de)), du))

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


def ecef2lla(ecef_arr):
    # convert ECEF coordinates to LLA
    # test data : test_coord = [2297292.91, 1016894.94, -5843939.62]
    # expected result : -66.8765400174 23.876539914 999.998386689

    x = float(ecef_arr[0])
    y = float(ecef_arr[1])
    z = float(ecef_arr[2])

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
    # TODO: Might want to move this into the Config class since it's only used there.
    if missing_input == 'fill':
        dates = [Date(year=1980, doy=1), Date(datetime=dt.datetime.now())]
    else:
        dates = [None, None]

    if arg:
        for i, arg in enumerate(arg):
            try:
                if '.' in arg:
                    dates[i] = Date(fyear=float(arg))
                elif '_' in arg:
                    dates[i] = Date(year=int(arg.split('_')[0]), doy=int(arg.split('_')[1]))
                elif '/' in arg:
                    dates[i] = Date(year=int(arg.split('/')[0]), month=int(arg.split('/')[1]),
                                    day=int(arg.split('/')[2]))
                elif '-' in arg:
                    dates[i] = Date(gpsWeek=int(arg.split('-')[0]), gpsWeekDay=int(arg.split('-')[1]))
                elif len(arg) > 0:
                    if allow_days and i == 0:
                        dates[i] = Date(datetime=dt.datetime.now()) - int(arg)
                    else:
                        raise ValueError('Invalid input date: allow_days was set to False.')
            except Exception as e:
                raise ValueError('Could not decode input date (valid entries: '
                                 'fyear, yyyy_ddd, yyyy/mm/dd, gpswk-wkday). '
                                 'Error while reading the date start/end parameters: {}'.format(e))

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


def process_stnlist(cnn, stnlist_in, print_summary=True):
    if len(stnlist_in) == 1 and os.path.isfile(stnlist_in[0]):
        print((' >> Station list read from file: ' + stnlist_in[0]))
        stnlist_in = [line.strip() for line in open(stnlist_in[0])]

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
        raise UtilsException('must provide a positive integer year YY or YYYY')

    # defensively, make sure that the year is positive
    if year < 0:
        raise UtilsException('must provide a positive integer year YY or YYYY')

    if 80 <= year <= 99:
        year += 1900
    elif 0 <= year < 80:
        year += 2000

    return str(year)


def get_norm_doy_str(doy):
    try:
        doy = int(doy)
    except Exception:
        raise UtilsException('must provide an integer day of year')

        # create string version up fround
    doy = str(doy)

    # mk 3 diit doy
    if len(doy) == 1:
        doy = "00" + doy
    elif len(doy) == 2:
        doy = "0" + doy
    return doy


def test_node(check_gamit_tables=None, software_sync=(), cfg_file='gnss_data.cfg'):
    # test node: function that makes sure that all required packages and tools are present in the nodes

    def check_tab_file(tabfile, date):

        if os.path.isfile(tabfile):
            # file exists, check contents
            with open(tabfile) as luntab:
                lines = luntab.readlines()
                tabdate = Date(mjd=lines[-1].split()[0])
                if tabdate < date:
                    return ' -- %s: Last entry in %s is %s but processing %s' \
                           % (platform.node(), tabfile, tabdate.yyyyddd(), date.yyyyddd())

        else:
            return ' -- %s: Could not find file %s' % (platform.node(), tabfile)

        return []

    # BEFORE ANYTHING! check the python version
    version = sys.version_info
    if version.major < 3:
        return ' -- {}: Incorrect Python version: {}.{}.{}. Recommended version > 3'.format(platform.node(),
                                                                                            version.major,
                                                                                            version.minor,
                                                                                            version.micro)

    # start importing the modules needed

    try:
        if len(software_sync) > 0:
            # synchronize directories listed in the src and dst arguments

            for source_dest in software_sync:
                if isinstance(source_dest, str) and ',' in source_dest:
                    s = source_dest.split(',')[0].strip()
                    d = source_dest.split(',')[1].strip()

                    print('    -- Synchronizing %s -> %s' % (s, d))

                    updated = dirsync.sync(s, d, 'sync', purge=True, create=True)

                    for f in updated:
                        print('    -- Updated %s' % f)

    except Exception:
        return ' -- %s: Problem found while synchronizing software:\n%s ' % (platform.node(), traceback.format_exc())

    # continue with a test SQL connection
    # make sure that the gnss_data.cfg is present
    try:
        Connection(cfg_file)
    except Exception:
        return ' -- %s: Problem found while connecting to postgres:\n%s ' % (platform.node(), traceback.format_exc())

    # make sure we can create the production folder
    try:
        test_dir = os.path.join('production/node_test')
        os.makedirs(test_dir, exist_ok=True)
    except Exception:
        return ' -- %s: Could not create production folder:\n%s ' % (platform.node(), traceback.format_exc())

    # test
    try:
        config = ReadOptions(cfg_file)

        # check that all paths exist and can be reached
        if not os.path.exists(config.options['path']):
            return ' -- %s: Could not reach archive path %s' % (platform.node(), config.options['path'])

        if not os.path.exists(config.options['repository']):
            return ' -- %s: Could not reach repository path %s' % (platform.node(), config.options['repository'])

        # pick a test date to replace any possible parameters in the config file
        date = Date(year=2010, doy=1)
    except Exception:
        return ' -- %s: Problem while reading config file and/or testing archive access:\n%s' \
               % (platform.node(), traceback.format_exc())

    try:
        GetBrdcOrbits(config.options['brdc'], date, test_dir)
    except Exception:
        return ' -- %s: Problem while testing the broadcast ephemeris archive (%s) access:\n%s' \
               % (platform.node(), config.options['brdc'], traceback.format_exc())

    try:
        GetSp3Orbits(config.options['sp3'], date, config.sp3types, test_dir)
    except Exception:
        return ' -- %s: Problem while testing the sp3 orbits archive (%s) access:\n%s' \
               % (platform.node(), config.options['sp3'], traceback.format_exc())

    # check that all executables and GAMIT bins are in the path
    # TODO: Determine which of these programs we can convert into pure python.
    list_of_prgs = ['crz2rnx', 'crx2rnx', 'rnx2crx', 'rnx2crz', 'RinSum', 'teqc', 'svdiff', 'svpos', 'tform',
                    'sh_rx2apr', 'doy', 'RinEdit', 'sed', 'compress']

    for prg in list_of_prgs:
        out = shutil.which(prg)
        if not out:
            return ' -- %s: Could not find path to %s' % (platform.node(), prg)

    # check grdtab and ppp from the config file
    if not os.path.isfile(config.options['grdtab']):
        return ' -- %s: Could not find grdtab in %s' % (platform.node(), config.options['grdtab'])

    if not os.path.isfile(config.options['otlgrid']):
        return ' -- %s: Could not find otlgrid in %s' % (platform.node(), config.options['otlgrid'])

    if not os.path.isfile(config.options['ppp_exe']):
        return ' -- %s: Could not find ppp_exe in %s' % (platform.node(), config.options['ppp_exe'])

    if not os.path.isfile(os.path.join(config.options['ppp_path'], 'gpsppp.stc')):
        return ' -- %s: Could not find gpsppp.stc in %s' % (platform.node(), config.options['ppp_path'])

    if not os.path.isfile(os.path.join(config.options['ppp_path'], 'gpsppp.svb_gps_yrly')):
        return ' -- %s: Could not find gpsppp.svb_gps_yrly in %s' % (platform.node(), config.options['ppp_path'])

    if not os.path.isfile(os.path.join(config.options['ppp_path'], 'gpsppp.flt')):
        return ' -- %s: Could not find gpsppp.flt in %s' % (platform.node(), config.options['ppp_path'])

    if not os.path.isfile(os.path.join(config.options['ppp_path'], 'gpsppp.stc')):
        return ' -- %s: Could not find gpsppp.stc in %s' % (platform.node(), config.options['ppp_path'])

    if not os.path.isfile(os.path.join(config.options['ppp_path'], 'gpsppp.met')):
        return ' -- %s: Could not find gpsppp.met in %s' % (platform.node(), config.options['ppp_path'])

    for frame in config.options['frames']:
        if not os.path.isfile(frame['atx']):
            return ' -- %s: Could not find atx in %s' % (platform.node(), frame['atx'])

    if check_gamit_tables is not None:
        # check the gamit tables if not none

        date = check_gamit_tables[0]
        eop = check_gamit_tables[1]
        # TODO: Change this so it's not hardwired into the home directory anymore
        tables = os.path.join(config.options['gg'], 'tables')

        if not os.path.isdir(config.options['gg']):
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


def check_year(year):
    # to check for wrong dates in RinSum

    if int(year) - 1900 < 80 and int(year) >= 1900:
        year = int(year) - 1900 + 2000

    elif 1900 > int(year) >= 80:
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
        raise DateException('invalid day of year')

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
    fday = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    lday = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    month = None
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
        raise DateException('day of year input is invalid')

    # localized days
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


def date2gpsdate(year, month, day):
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


def gpsdate2mjd(gpsweek, gpsweekday):
    # parse to integers
    gpsweek = int(gpsweek)
    gpsweekday = int(gpsweekday)

    mjd = (gpsweek * 7.) + 44244. + gpsweekday

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
            elif key == 'datetime':  # DDG 03-28-2017: handle conversion from date_time to pyDate
                if isinstance(arg, dt.datetime):
                    self.day = arg.day
                    self.month = arg.month
                    self.year = arg.year
                    self.hour = arg.hour
                    self.minute = arg.minute
                    self.second = arg.second
                else:
                    raise DateException('invalid type for ' + key + '\n')
            elif key == 'stninfo':  # DDG: handle station information records

                self.from_stninfo = True

                if isinstance(arg, str) or isinstance(arg, str):
                    self.year, self.doy, self.hour, self.minute, self.second = parse_stninfo(arg)
                elif isinstance(arg, dt.datetime):
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
                    raise DateException('invalid type ' + str(type(arg)) + ' for ' + key + '\n')
            else:
                raise DateException('unrecognized input arg: ' + key + '\n')

        # make due with what we gots
        if self.year is not None and self.doy is not None:

            # compute the month and day of month
            self.month, self.day = doy2date(self.year, self.doy)

            # compute the fractional year
            self.fyear = yeardoy2fyear(self.year, self.doy, self.hour, self.minute, self.second)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsdate(self.year, self.month, self.day)

            self.mjd = gpsdate2mjd(self.gpsWeek, self.gpsWeekDay)

        elif self.gpsWeek is not None and self.gpsWeekDay is not None:

            # initialize modified julian day from gps date
            self.mjd = gpsdate2mjd(self.gpsWeek, self.gpsWeekDay)

            # compute year, month, and day of month from modified julian day
            self.year, self.month, self.day = mjd2date(self.mjd)

            # compute day of year from month and day of month
            self.doy, self.fyear = date2doy(self.year, self.month, self.day, self.hour, self.minute, self.second)

        elif self.year is not None and self.month is not None and self.day:

            # initialize day of year and fractional year from date
            self.doy, self.fyear = date2doy(self.year, self.month, self.day, self.hour, self.minute, self.second)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsdate(self.year, self.month, self.day)

            # init the modified julian date
            self.mjd = gpsdate2mjd(self.gpsWeek, self.gpsWeekDay)

        elif self.fyear is not None:

            # initialize year and day of year
            self.year, self.doy, self.hour, self.minute, self.second = fyear2yeardoy(self.fyear)

            # set the month and day of month
            # compute the month and day of month
            self.month, self.day = doy2date(self.year, self.doy)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsdate(self.year, self.month, self.day)

            # finally, compute modified jumlian day
            self.mjd = gpsdate2mjd(self.gpsWeek, self.gpsWeekDay)

        elif self.mjd is not None:

            # compute year, month, and day of month from modified julian day
            self.year, self.month, self.day = mjd2date(self.mjd)

            # compute day of year from month and day of month
            self.doy, self.fyear = date2doy(self.year, self.month, self.day, self.hour, self.minute, self.second)

            # compute the gps date
            self.gpsWeek, self.gpsWeekDay = date2gpsdate(self.year, self.month, self.day)

        else:
            if not self.from_stninfo:
                # if empty Date object from a station info, it means that it should be printed as 9999 999 00 00 00
                raise DateException('not enough independent input args to compute full date')

    def strftime(self):
        return self.date_time().strftime('%Y-%m-%d %H:%M:%S')

    def to_json(self):
        if self.from_stninfo:
            return {'stninfo': str(self)}
        else:
            return {'year': self.year, 'doy': self.doy, 'hour': self.hour, 'minute': self.minute, 'second': self.second}

    def __repr__(self):
        return 'Date(' + str(self.year) + ', ' + str(self.doy) + ')'

    def __str__(self):
        if self.year is None:
            return '9999 999 00 00 00'
        else:
            return '%04i %03i %02i %02i %02i' % (self.year, self.doy, self.hour, self.minute, self.second)

    def __lt__(self, date):

        if not isinstance(date, Date):
            raise DateException('type: ' + str(type(date)) + ' invalid. Can only compare Date objects')

        return self.fyear < date.fyear

    def __le__(self, date):

        if not isinstance(date, Date):
            raise DateException('type: ' + str(type(date)) + ' invalid. Can only compare Date objects')

        return self.fyear <= date.fyear

    def __gt__(self, date):

        if not isinstance(date, Date):
            raise DateException('type: ' + str(type(date)) + ' invalid. Can only compare Date objects')

        return self.fyear > date.fyear

    def __ge__(self, date):

        if not isinstance(date, Date):
            raise DateException('type: ' + str(type(date)) + ' invalid.  Can only compare Date objects')

        return self.fyear >= date.fyear

    def __eq__(self, date):

        if not isinstance(date, Date):
            raise DateException('type: ' + str(type(date)) + ' invalid.  Can only compare Date objects')

        return self.mjd == date.mjd

    def __ne__(self, date):

        if not isinstance(date, Date):
            raise DateException('type: ' + str(type(date)) + ' invalid.  Can only compare Date objects')

        return self.mjd != date.mjd

    def __add__(self, ndays):

        if not isinstance(ndays, int):
            raise DateException('type: ' + str(type(ndays)) + ' invalid.  Can only add integer number of days')

        return Date(mjd=self.mjd + ndays)

    def __sub__(self, ndays):

        if not (isinstance(ndays, int) or isinstance(ndays, Date)):
            raise DateException('type: ' + str(type(ndays)) + ' invalid. Can only subtract integer number of days')

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

    def date_time(self):
        if self.year is None:
            return dt.datetime(year=9999, month=1, day=1,
                               hour=1, minute=1, second=1)
        else:
            return dt.datetime(year=self.year, month=self.month, day=self.day,
                               hour=self.hour, minute=self.minute, second=self.second)

    def first_epoch(self, out_format='date_time'):
        if out_format == 'date_time':
            return dt.datetime(year=self.year, month=self.month, day=self.day, hour=0, minute=0, second=0).strftime(
                '%Y-%m-%d %H:%M:%S')
        else:
            _, fyear = date2doy(self.year, self.month, self.day, 0)
            return fyear

    def last_epoch(self, out_format='date_time'):
        if out_format == 'date_time':
            return dt.datetime(year=self.year, month=self.month, day=self.day, hour=23, minute=59, second=59).strftime(
                '%Y-%m-%d %H:%M:%S')
        else:
            _, fyear = date2doy(self.year, self.month, self.day, 23, 59, 59)
            return fyear


class OrbitalProduct(object):

    def __init__(self, archive, date, filename, copyto):

        if date.gpsWeek < 0 or date > Date(datetime=dt.datetime.now()):
            # do not allow negative weeks or future orbit downloads!
            raise Productsexceptionunreasonabledate('Orbit requested for an unreasonable date: week '
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
                shutil.copyfile(archive_file_path, os.path.join(copyto, self.filename))
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
                shutil.copyfile(archive_file_path + ext, os.path.join(copyto, self.filename + ext))
                self.file_path = os.path.join(copyto, self.filename)

                cmd = RunCommand('gunzip -f ' + self.file_path + ext, 15)
                try:
                    cmd.run_shell()
                except Exception:
                    raise
            else:
                raise ProductsException('Could not find the archive file for ' + self.filename)


class StationInfoRecord(dict):
    # TODO: This class needs cleaning up, right now there are some type errors when trying to execute
    # some of the class functions.
    def __init__(self, networkcode=None, stationcode=None, record=None, **kwargs):
        super().__init__(**kwargs)
        self.NetworkCode = networkcode
        self.StationCode = stationcode
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
                                   self.AntennaNorth, self.AntennaEast, self.AntennaHeight,
                                   self.HeightCode.encode('utf-8'),
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
                return_fields[field] = self[field].date_time()
            elif field == 'DateEnd':
                if self[field].year is None:
                    return_fields[field] = None
                else:
                    return_fields[field] = self[field].date_time()
            else:
                return_fields[field] = self[field]

        return return_fields

    def to_json(self):

        fields = self.database()
        fields['DateStart'] = str(self.DateStart)
        fields['DateEnd'] = str(self.DateEnd)

        return fields

    def parse_station_record(self, record):
        # TODO: Deal with a dict record input.
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
            self.DateStart = Date(stninfo=record['DateStart'])
            self.DateEnd = Date(stninfo=record['DateEnd'])
            self.StationCode = record['StationCode'].lower()
        except KeyError:
            pass

    def __repr__(self):
        return 'StationInfoRecord(' + str(self) + ')'

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

                    pDate = date.date_time()

                    for record in self.records:

                        DateStart = record.DateStart.date_time()
                        DateEnd = record.DateEnd.date_time()

                        # make the gap-tolerant comparison
                        if DateStart - dt.timedelta(hours=h_tolerance) <= pDate <= \
                                DateEnd + dt.timedelta(hours=h_tolerance):
                            # found the record that corresponds to this date
                            self.currentrecord = record
                            break

                    if self.currentrecord.DateStart is None:
                        raise StationInfoException('Could not find a matching station.info record for ' +
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
                raise StationInfoException('Could not find ANY valid station info entry for ' +
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
            with open(stninfo_file_list) as fileio:
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

                record.AntennaHeight = numpy.sqrt(numpy.square(float(record.AntennaHeight)) -
                                                  numpy.square(float(htc[0]['h_offset']))) - float(
                    htc[0]['v_offset'])
                if record.Comments is not None:
                    record.Comments += '\nChanged from %s to DHARP by pyStationInfo.\n' \
                                       % record.HeightCode
                else:
                    record.Comments = 'Changed from %s to DHARP by pyStationInfo.\n' % record.HeightCode

                record.HeightCode = 'DHARP'

                return record
            else:
                # TODO: Could not translate height code DHPAB to DHARP (ter.inmn: TRM59800.00).
                #  Check the height codes table.
                raise Stationinfoheightcodenotfound('Could not translate height code %s to DHARP (%s.%s: %s). '
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

        q_start = qrecord['DateStart'].date_time()
        q_end = qrecord['DateEnd'].date_time()

        if self.records:
            for record in self.records:

                r_start = record.DateStart.date_time()
                r_end = record.DateEnd.date_time()

                earliest_end = min(q_end, r_end)
                latest_start = max(q_start, r_start)

                if (earliest_end - latest_start).total_seconds() > 0:
                    overlaps.append(record)

        return overlaps

    def DeleteStationInfo(self, record):

        event = Event(Description=record.DateStart.strftime() +
                                  ' has been deleted:\n' + str(record), StationCode=self.StationCode,
                      NetworkCode=self.NetworkCode)

        self.cnn.insert_event(event)

        self.cnn.delete('stationinfo', record.database())
        self.load_stationinfo_records()

    def UpdateStationInfo(self, record, new_record):

        # avoid problems with trying to insert records from other stations. Force this NetworkCode
        record.NetworkCode = self.NetworkCode
        new_record.NetworkCode = self.NetworkCode

        if self.NetworkCode and self.StationCode:

            # check the possible overlaps. This record will probably overlap with itself, so check that the overlap has
            # the same DateStart as the original record (that way we know it's an overlap with itself)
            overlaps = self.overlaps(new_record)

            for overlap in overlaps:
                if overlap['DateStart'].date_time() != record.DateStart.date_time():
                    # it's overlapping with another record, raise error

                    raise StationInfoException('Record %s -> %s overlaps with existing station.info records: %s -> %s'
                                               % (str(record.DateStart), str(record.DateEnd),
                                                    str(overlap['DateStart']), str(overlap['DateEnd'])))

            # insert event (before updating to save all information)
            event = Event(Description=record.DateStart.strftime() +
                                      ' has been updated:\n' + str(new_record) +
                                      '\n+++++++++++++++++++++++++++++++++++++\n' +
                                      'Previous record:\n' +
                                      str(record) + '\n',
                          NetworkCode=self.NetworkCode,
                          StationCode=self.StationCode)

            self.cnn.insert_event(event)

            if new_record['DateStart'] != record.DateStart:
                self.cnn.query('UPDATE stationinfo SET "DateStart" = \'%s\' '
                               'WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND "DateStart" = \'%s\'' %
                               (new_record['DateStart'].strftime(), self.NetworkCode,
                                self.StationCode, record.DateStart.strftime()))

            self.cnn.update('stationinfo', new_record.database(), NetworkCode=self.NetworkCode,
                            StationCode=self.StationCode, DateStart=new_record['DateStart'].date_time())

            self.load_stationinfo_records()

    def InsertStationInfo(self, record):

        # avoid problems with trying to insert records from other stations. Force this NetworkCode
        record.NetworkCode = self.NetworkCode

        if self.NetworkCode and self.StationCode:
            # check existence of station in the db
            rs = self.cnn.query(
                'SELECT * FROM stationinfo WHERE "NetworkCode" = \'%s\' '
                'AND "StationCode" = \'%s\' AND "DateStart" = \'%s\'' %
                (self.NetworkCode, self.StationCode, record.DateStart.strftime()))

            if rs.ntuples() == 0:
                # can insert because it's not the same record
                # 1) verify the record is not between any two existing records
                overlaps = self.overlaps(record)

                if overlaps:
                    # if it overlaps all records and the DateStart < self.records[0]['DateStart']
                    # see if we have to extend the initial date
                    if len(overlaps) == len(self.records) and \
                            record.DateStart.date_time() < self.records[0].DateStart.date_time():
                        if self.records_are_equal(record, self.records[0]):
                            # just modify the start date to match the incoming record
                            # self.cnn.update('stationinfo', self.records[0], DateStart=record.DateStart)
                            # the previous statement seems not to work because it updates a primary key!
                            self.cnn.query(
                                'UPDATE stationinfo SET "DateStart" = \'%s\' WHERE "NetworkCode" = \'%s\' '
                                'AND "StationCode" = \'%s\' AND "DateStart" = \'%s\'' %
                                (record.DateStart.strftime(),
                                 self.NetworkCode, self.StationCode,
                                 self.records[0]['DateStart'].strftime()))

                            # insert event
                            event = Event(Description='The start date of the station information record ' +
                                                      self.records[0]['DateStart'].strftime() +
                                                      ' has been been modified to ' +
                                                      record.DateStart.strftime(),
                                          StationCode=self.StationCode,
                                          NetworkCode=self.NetworkCode)
                            self.cnn.insert_event(event)
                        else:
                            # new and different record, stop the Session with
                            # EndDate = self.records[0]['DateStart'] - date_time.timedelta(seconds=1) and insert
                            record.DateEnd = Date(datetime=self.records[0].DateStart.date_time() -
                                                           dt.timedelta(seconds=1))

                            self.cnn.insert('stationinfo', record.database())

                            # insert event
                            event = Event(
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
                                        DateEnd=record.DateStart.date_time() - dt.timedelta(seconds=1))

                        # create the incoming session
                        self.cnn.insert('stationinfo', record.database())

                        # insert event
                        event = Event(
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

                        raise StationInfoException('Record %s -> %s overlaps with existing station.info records: %s'
                                                   % (str(record.DateStart), str(record.DateEnd),
                                                        ' '.join(stroverlap)))

                else:
                    # no overlaps, insert the record
                    self.cnn.insert('stationinfo', record.database())

                    # insert event
                    event = Event(Description='A new station information record was added:\n' +
                                              str(record),
                                  StationCode=self.StationCode,
                                  NetworkCode=self.NetworkCode)
                    self.cnn.insert_event(event)

                # reload the records
                self.load_stationinfo_records()
            else:
                raise StationInfoException('Record %s -> %s already exists in station.info' %
                                           (str(record.DateStart), str(record.DateEnd)))
        else:
            raise StationInfoException('Cannot insert record without initializing pyStationInfo '
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
                    record.DateStart = Date(datetime=StartDate)
                    record.DateEnd = Date(datetime=rnxtbl[i - count]['ObservationETime'])
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
        record = StationInfoRecord(self.NetworkCode, self.StationCode)
        record.DateStart = Date(datetime=StartDate)
        record.DateEnd = Date(stninfo=None)
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
            raise StationInfoException('type: ' + str(type(stninfo))
                                       + ' is invalid. Can only compare StationInfo objects')

        if self.currentrecord['AntennaCode'] != stninfo.currentrecord['AntennaCode']:
            return False

        if self.currentrecord['AntennaHeight'] != stninfo.currentrecord['AntennaHeight']:
            return False

        if self.currentrecord['AntennaNorth'] != stninfo.currentrecord['AntennaNorth']:
            return False

        if self.currentrecord['AntennaEast'] != stninfo.currentrecord['AntennaEast']:
            return False

        if self.currentrecord['AntennaSerial'] != stninfo.currentrecord['AntennaSerial']:
            return False

        if self.currentrecord['ReceiverCode'] != stninfo.currentrecord['ReceiverCode']:
            return False

        if self.currentrecord['ReceiverSerial'] != stninfo.currentrecord['ReceiverSerial']:
            return False

        if self.currentrecord['RadomeCode'] != stninfo.currentrecord['RadomeCode']:
            return False

        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.records = None

    def __enter__(self):
        return self


class Event(dict):

    def __init__(self, **kwargs):

        super().__init__()

        self['EventDate'] = dt.datetime.now()
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


class Connection:
    """
    Ideally this class is to be used as a direct connection to the gnss_data database.  As such it should initiate to
    a pgdb.Connection object through the uses of pgdb.connect().  Once the object has established a connection to the
    database, class methods should be used to interact with the database.  The goal is to abstract all the SQL commands
    so we can deal with the errors here instead of out in the wild.
    """
    __slots__ = ['active_transaction', 'options', 'conn',
                 'table_cursor']

    def __init__(self, configfile='gnss_data.cfg'):

        self.options = {'hostname': 'localhost',
                        'username': 'postgres',
                        'password': '',
                        'database': 'gnss_data'}
        self.conn = None
        self.active_transaction = False
        # parse session config file
        config = configparser.ConfigParser()
        with open(configfile) as cf:
            config.read_file(cf)
        # get the database config
        for key in config['postgres']:
            self.options[key] = config.get('postgres', key)
        connect_dsn = 'dbname={} host={} user={} password={}'.format(self.options['database'],
                                                                     self.options['hostname'],
                                                                     self.options['username'],
                                                                     self.options['password'])
        # open connection to server
        try:
            self.conn = psycopg2.connect(connect_dsn)
            if self.conn.closed == 1:
                raise DBErrConnect
        except Exception as e:
            raise e

    def __del__(self):
        try:
            self.conn.close()
        except Exception as e:
            print(e)
            pass
        del self.conn

    def _execute_wrapper(self, sql_statement, values=None, retval=False, return_dict=False):
        try:
            with self.conn.cursor() as curs:
                if values is not None:
                    curs.execute(sql_statement, values)
                else:
                    curs.execute(sql_statement)
                if retval:
                    if return_dict:
                        qresults = curs.fetchall()
                        keys = [name[0] for name in curs.description]
                        d = defaultdict(list)
                        for n, key in enumerate(keys):
                            for rec in qresults:
                                d[key].append(rec[n])
                        return d
                    else:
                        return curs.fetchall()

            self.conn.commit()
        except Exception as e:
            raise DBErrInsert(e)

    def insert(self, table, record: dict):
        x = sql.SQL('{}').format(sql.Identifier(table))
        y = sql.SQL(', ').join([sql.Identifier(key) for key in record.keys()])
        z = sql.SQL(', ').join(sql.Placeholder() * len(record))
        insert_statement = sql.SQL("INSERT INTO {0} ({1}) VALUES ({2});").format(x, y, z)
        self._execute_wrapper(insert_statement, [v for v in record.values()])

    def clear_locks(self):

        clear_statement = sql.SQL("DELETE FROM locks WHERE {} NOT LIKE {};").format(sql.Identifier('NetworkCode'),
                                                                                    sql.Placeholder())
        self._execute_wrapper(clear_statement, ('?%',))

    def load_table(self, table):
        select_statement = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        return self._execute_wrapper(select_statement, retval=True)

    def clear_locked(self, table):

        clear_statement = sql.SQL("DELETE FROM {} WHERE {} LIKE {};").format(sql.Identifier(table),
                                                                             sql.Identifier('NetworkCode'),
                                                                             sql.Placeholder())
        self._execute_wrapper(clear_statement, ('?%',))

    def load_tankstruct(self):
        sql_statement = sql.SQL('SELECT * FROM {0} INNER JOIN {1} '
                                'USING ({2}) ORDER BY {3}').format(sql.Identifier('rinex_tank_struct'),
                                                                   sql.Identifier('keys'),
                                                                   sql.Identifier('KeyCode'),
                                                                   sql.Identifier('Level'))
        return self._execute_wrapper(sql_statement, retval=True, return_dict=True)

    def insert_event(self, event):
        event_dict = event.db_dict()
        y = sql.SQL(', ').join([sql.Identifier(key) for key in event_dict.keys()])
        z = sql.SQL(', ').join(sql.Placeholder() * len(event_dict))
        insert_statement = sql.SQL("insert into {0} ({1}) VALUES ({2});").format(sql.Identifier('events'), y, z)
        self._execute_wrapper(insert_statement, [v for v in event_dict.values()])

    def print_summary(self, script):
        script_start = sql.SQL('SELECT MAX({}) AS mx FROM {} WHERE {} = {}').format(sql.Identifier('exec_date'),
                                                                                    sql.Identifier('executions'),
                                                                                    sql.Identifier('script'),
                                                                                    sql.Placeholder())
        st = self._execute_wrapper(script_start, (script,), retval=True)

        counter = sql.SQL(
            'SELECT COUNT(*) AS cc FROM {0} WHERE {1} >= {2} AND {3} = {2}').format(sql.Identifier('events'),
                                                                                    sql.Identifier('EventDate'),
                                                                                    sql.Placeholder(),
                                                                                    sql.Identifier('EventType'))
        info = self._execute_wrapper(counter, (st[0][0], 'info'), retval=True)
        erro = self._execute_wrapper(counter, (st[0][0], 'error'), retval=True)
        warn = self._execute_wrapper(counter, (st[0][0], 'warning'), retval=True)

        print(' >> Summary of events for this run:')
        print(' -- info    : %i' % info[0][0])
        print(' -- errors  : %i' % erro[0][0])
        print(' -- warnings: %i' % warn[0][0])

    def spatial_check(self, vals, search_in_new=False):
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
        return self._execute_wrapper(sql_select, vals, retval=True)

    def nearest_station(self, vals, search_in_new=False):
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
        return self._execute_wrapper(sql_select, vals, retval=True)

    def similar_locked(self, vals):
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

    def update(self, table: str, row: dict, **kw):
        a = sql.SQL('{}').format(sql.Identifier(table))
        b = sql.SQL(', ').join([sql.Identifier(key) for key in kw.keys()])
        c = sql.SQL(', ').join(sql.Placeholder() * len(kw))
        d = sql.SQL(', ').join([sql.Identifier(key) for key in row.keys()])
        e = sql.SQL(', ').join(sql.Placeholder() * len(row))
        insert_statement = sql.SQL("UPDATE {0} SET ({1}) = ({2}) WHERE ({3}) LIKE ({4})").format(a, b, c, d, e)
        vals = []
        for v in kw.values():
            vals.append(v)
        for v in row.values():
            vals.append(v)

        self._execute_wrapper(insert_statement, vals)

    def load_table_matching(self, table: str, where_dict: dict):
        select_statement = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        where_statement = sql.SQL("WHERE")
        if len(where_dict) > 1:
            like_statement = sql.SQL(' AND ').join([sql.SQL('{} = {}').format(sql.Identifier(k), sql.Placeholder()) for k in
                                                    where_dict.keys()])
        else:
            like_statement = [sql.SQL('{} LIKE {}').format(sql.Identifier(k), sql.Placeholder()) for k in where_dict.keys()]
            like_statement = like_statement[0]

        full_statement = sql.SQL(' ').join([select_statement, where_statement, like_statement])
        return self._execute_wrapper(full_statement, [v for v in where_dict.values()], retval=True)


class RinexStruct:

    def __init__(self, cnn, cfg_file='gnss_data.cfg'):

        self.cnn = cnn

        # read the structure definition table
        self.levels = cnn.load_tankstruct()

        self.keys = cnn.load_table('keys')

        # read the station and network tables
        self.networks = cnn.load_table('networks')

        self.stations = cnn.load_table('stations')

        self.config = ReadOptions(cfg_file)
        self.archiveroot = None

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

            try:
                self.cnn.insert('rinex', record)

                if rinexobj is not None:
                    # a rinexobj was passed, copy it into the archive.

                    path2archive = os.path.join(self.config.options['path'],
                                                self.build_rinex_path(record['NetworkCode'], record['StationCode'],
                                                                      record['ObservationYear'],
                                                                      record['ObservationDOY'],
                                                                      with_filename=False, rinexobj=rinexobj))

                    # copy fixed version into the archive
                    archived_crinex = rinexobj.compress_local_copyto(path2archive)
                    copy_succeeded = True
                    # get the rinex filename to update the database
                    rnx = rinexobj.to_format(os.path.basename(archived_crinex), TYPE_RINEX)

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

                    event = Event(Description='A new RINEX was added to the archive: %s' % record['Filename'],
                                  NetworkCode=record['NetworkCode'],
                                  StationCode=record['StationCode'],
                                  Year=record['ObservationYear'],
                                  DOY=record['ObservationDOY'])
                else:
                    event = Event(Description='Archived CRINEX file %s added to the database.' %
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

                    filename = shutil.move(rinex_path, os.path.join(move_to_dir, os.path.basename(rinex_path)))
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
            event = Event(
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

        return self.cnn.load_table_matching(table, kwargs)

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
                            exit()

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
                            exit()

        return stninfo, path2stninfo

    def build_rinex_path(self, network_code, station_code, observation_year, observation_doy,
                         with_filename=True, filename=None, rinexobj=None):
        """
        Function to get the location in the archive of a rinex file. It has two modes of operation:
        1) retrieve an existing rinex file, either specific or the rinex for processing
        (most complete, largest interval) or a specific rinex file (already existing in the rinex table).
        2) To get the location of a potential file (probably used for injecting a new file in the archive. No this mode,
        filename has no effect.
        :param network_code: NetworkCode of the station being retrieved
        :param station_code: StationCode of the station being retrieved
        :param observation_year: Year of the rinex file being retrieved
        :param observation_doy: DOY of the rinex file being retrieved
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
                                    network_code + '\' AND "StationCode" = \'' + station_code +
                                    '\' AND "ObservationYear" = ' + str(observation_year) + ' AND "ObservationDOY" = ' +
                                    str(observation_doy) + ' AND "Filename" = \'' + filename + '\'')
            else:
                # if filename is NOT set, user requesting a the processing file: query rinex_proc
                rs = self.cnn.query(
                    'SELECT ' + sql_string + ' FROM rinex_proc WHERE "NetworkCode" = \'' + network_code +
                    '\' AND "StationCode" = \'' + station_code + '\' AND "ObservationYear" = ' + str(
                        observation_year) + ' AND "ObservationDOY" = ' + str(observation_doy))

            if rs.ntuples() != 0:
                field = rs.dictresult()[0]
                keys = []
                for level in self.levels:
                    keys.append(str(field[level['rinex_col_in']]).zfill(level['TotalChars']))

                if with_filename:
                    # database stores rinex, we want crinez
                    retval = os.sep.join(keys) + os.sep + \
                             field['Filename'].replace(field['Filename'].split('.')[-1],
                                                       field['Filename'].split('.')[-1].replace('o', 'd.Z'))
                    if retval[0] == os.path.sep:
                        return retval[1:]
                    else:
                        return retval

                else:
                    return os.sep.join(keys)
            else:
                return None
        else:
            # new file (get the path where it's supposed to go)
            pathlist = []
            for n, key in enumerate(self.levels['rinex_col_in']):
                pathlist.append(str(rinexobj.record[key]).zfill(self.levels['TotalChars'][n]))
            path = os.sep.join(pathlist)

            valid, _ = self.parse_archive_keys(os.path.join(path, rinexobj.crinez), self.levels['KeyCode'])

            if valid:
                if with_filename:
                    return os.path.join(path, rinexobj.crinez)
                else:
                    return path
            else:
                raise ValueError('Invalid path result: %s' % path)

    def parse_archive_keys(self, path, key_filter=()):
        """
        Checks the path to make sure that the levels (e.g. NetworkCode) correspond to the correct naming convetions
        defined in the "keys" table of the database.
        :param path: The path to parse for correctness
        :param key_filter: Optional parameter such that only path levels listed in the key_filter are returned.
        :return: Path is correct (bool), Dictionary with key value pairs (dict)
        """
        try:
            pathparts = path.split(os.sep)
            filename = os.path.basename(path)

            # check the number of levels in pathparts against the number of expected levels
            # subtract one for the filename
            if len(pathparts) - 1 != len(self.levels['KeyCode']):
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
                # Correct the database keys entries such that they can correctly be used to generate a file structure.
                # now look in the different levels to match more data (or replace filename keys)

                # At this point the pathparts list is already made up of strings that correspond to the database entries
                # so there doesn't seem to be a need to double check them and run error handling.

                # check date is valid and also fill day and month keys
                date = Date(year=keys['year'], doy=keys['doy'])
                keys['day'] = date.day
                keys['month'] = date.month
                return True, {key: keys[key] for key in list(keys.keys()) if key in key_filter}
            else:
                return False, {}

        except Exception:
            return False, {}


class GetBrdcOrbits(OrbitalProduct):

    def __init__(self, brdc_archive, date, copyto, no_cleanup=False):

        self.brdc_archive = brdc_archive
        self.brdc_path = None
        self.no_cleanup = no_cleanup

        # try both zipped and unzipped n files
        self.brdc_filename = 'brdc' + str(date.doy).zfill(3) + '0.' + str(date.year)[2:4] + 'n'

        try:
            super().__init__(self.brdc_archive, date, self.brdc_filename, copyto)
            self.brdc_path = self.file_path

        except Productsexceptionunreasonabledate:
            raise
        except ProductsException:
            raise Brdcexception(
                'Could not find the broadcast ephemeris file for ' + str(date.year) + ' ' + str(date.doy))

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


class JobServer(dispy.JobCluster):
    # TODO: Make this into a subclass of the dispy jobcluster

    def __init__(self, config, check_gamit_tables=None, software_sync=(), cfg_file='gnss_data.cfg'):
        """
        initialize the jobserver
        :param config: pyOptions.ReadOptions instance
        :param check_gamit_tables: check or not the tables in GAMIT
        :param software_sync: list of strings with remote and local paths of software to be synchronized
        """
        self.cfg_file = cfg_file
        self.check_gamit_tables = check_gamit_tables
        self.software_sync = software_sync

        self.nodes = None
        self.result = None
        self.jobs = None
        self.run_parallel = config.options['parallel']
        self.verbose = False
        self.close = False

        # vars to store the http_server and the progress bar (if needed)
        self.progress_bar = None
        self.http_server = None
        self.callback = None
        self.function = None
        self.modules = None

        print(" ==== Starting JobServer(dispy) ====")

        # check that the run_parallel option is activated
        if config.options['parallel']:
            if config.options['node_list'] is None:
                # no explicit list, find all
                servers = ['*']
            else:
                # use the provided explicit list of nodes
                if config.options['node_list'].strip() == '':
                    servers = ['*']
                else:
                    servers = [_f for _f in list(config.options['node_list'].split(',')) if _f]

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
                sys.exit()

    def check_cluster(self, status, node):

        if status == dispy.DispyNode.Initialized:
            print(' -- Checking node %s (%i CPUs)...' % (node.name, node.avail_cpus))
            # test node to make sure everything works
            self.cluster.send_file(self.cfg_file, node)

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
                                            pulse_interval=60, setup=functools.partial(setup, modules),
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
            self.cluster.send_file(self.cfg_file, node)
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


class OceanLoading:

    def __init__(self, station_code, grdtab, otlgrid, x=None, y=None, z=None):

        self.x = None
        self.y = None
        self.z = None

        self.rootdir = os.path.join('production', 'otl_calc')
        # generate a unique id for this instance
        self.rootdir = os.path.join(self.rootdir, str(uuid.uuid4()))
        self.StationCode = station_code
        os.makedirs(self.rootdir, exist_ok=True)

        # verify of link to otl.grid exists
        if not os.path.isfile(os.path.join(self.rootdir, 'otl.grid')):
            # should be configurable
            try:
                os.symlink(otlgrid, os.path.join(self.rootdir, 'otl.grid'))
            except Exception as e:
                raise OTLException(e)

        if not os.path.isfile(grdtab):
            raise OTLException('grdtab could not be found at the specified location: ' + grdtab)
        else:
            self.grdtab = grdtab

        if not (x is None and y is None and z is None):
            self.x = x
            self.y = y
            self.z = z
        return

    def calculate_otl_coeff(self, x=None, y=None, z=None):

        if not self.x and (x is None or y is None or z is None):
            raise OTLException('Cartesian coordinates not initialized and not provided in calculate_otl_coef')
        else:
            if not self.x:
                self.x = x
            if not self.y:
                self.y = y
            if not self.z:
                self.z = z

            cmd = RunCommand('{} {} {} {} {}'.format(self.grdtab, self.x, self.y, self.z, self.StationCode),
                             5, self.rootdir)
            out, err = cmd.run_shell()

            if err or os.path.isfile(os.path.join(self.rootdir, 'GAMIT.fatal')) and not os.path.isfile(
                    os.path.join(self.rootdir, 'harpos.' + self.StationCode)):
                if err:
                    raise OTLException('grdtab returned an error: ' + err)
                else:
                    with open(os.path.join(self.rootdir, 'GAMIT.fatal')) as fileio:
                        raise OTLException('grdtab returned an error:\n' + fileio.read())
            else:
                # open otl file
                with open(os.path.join(self.rootdir, 'harpos.' + self.StationCode)) as fileio:
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


class ReadOptions:
    """
    Class that deals with reading in the default configuration file gnss_data.cfg
    """

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
        config.read_file(open(configfile))

        # get the archive config
        for section in config.sections():
            for key in config[section]:
                self.options[key] = config[section][key]

        del config
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
        self.repository_data_in = os.path.join(self.options['repository'], 'data_in')
        self.repository_data_in_retry = os.path.join(self.options['repository'], 'data_in_retry')
        self.repository_data_reject = os.path.join(self.options['repository'], 'data_rejected')

        self.sp3types = [self.options['sp3_type_1'], self.options['sp3_type_2'], self.options['sp3_type_3']]
        self.sp3types = [sp3type for sp3type in self.sp3types if sp3type is not None]

        # alternative sp3 types
        self.sp3altrn = [self.options['sp3_altr_1'], self.options['sp3_altr_2'], self.options['sp3_altr_3']]
        self.sp3altrn = [sp3alter for sp3alter in self.sp3altrn if sp3alter is not None]
        if self.options['parallel'] == 'True':
            self.options['parallel'] = True
        else:
            self.options['parallel'] = False


class PPPSpatialCheck:

    def __init__(self, lat=None, lon=None, h=None, epoch=None):
        # TODO: Why is this a separate class?
        self.lat = lat
        self.lon = lon
        self.h = h
        self.epoch = epoch

        return

    def verify_spatial_coherence(self, cnn, station_code):
        # checks the spatial coherence of the resulting coordinate
        # will not make any decisions, just output the candidates
        # if ambiguities are found, the rinex StationCode is used to solve them
        # third output arg is used to return a list with the closest station/s if no match is found
        # or if we had to disambiguate using station name
        # DDG Mar 21 2018: Added the velocity of the station to account for fast moving stations (on ice)
        # the logic is as follows:
        # 1) if etm data is available, then use it to bring the coordinate to self.epoch
        # 2) if no etm parameters are available, default to the coordinate reported in the stations table

        stn_match = cnn.spatial_check((self.lat[0], self.lat[0], self.lon[0]))

        # using the list of coordinates, check if StationCode exists in the list
        if len(stn_match) == 0:
            # no match, find closest station
            # get the closest station and distance in km to help the caller function
            stn = cnn.nearest_station((self.lat[0], self.lat[0], self.lon[0]))

            return False, [], stn

        if len(stn_match) == 1 and stn_match[0][1] == station_code:
            # one match, same name (return a dictionary)
            return True, stn_match, []

        if len(stn_match) == 1 and stn_match[0][1] != station_code:
            # one match, not the same name (return a list, not a dictionary)
            return False, stn_match, []

        if len(stn_match) > 1:
            # more than one match, same name
            # this is most likely a station that got moved a few meters and renamed
            # or a station that just got renamed.
            # disambiguation might be possible using the name of the station
            min_stn = [stni for stni in stn_match if stni[1] == station_code]

            if len(min_stn) > 0:
                # the minimum distance if to a station with same name, we are good:
                # does the name match the closest station to this solution? yes
                return True, min_stn, []
            else:
                return False, stn_match, []


class RunPPP(PPPSpatialCheck):
    def __init__(self, rinexobj, otl_coeff, options, sp3types, sp3altrn, antenna_height, strict=True, apply_met=True,
                 kinematic=False, clock_interpolation=False, hash=0, erase=True, decimate=True):

        assert isinstance(rinexobj, ReadRinex)

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

            except (Sp3exception, Clkexception, EOPException):

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

            copy_file(self.rinex.rinex_path, os.path.join(self.rootdir, self.rinex.rinex))

        else:
            raise RunPPPException('The file ' + self.rinex.rinex_path + ' could not be found. PPP was not executed.')

        return

    def copyfiles(self):
        # prepare all the files required to run PPP
        if self.apply_met:
            shutil.copyfile(os.path.join(self.ppp_path, 'gpsppp.met'), os.path.join(self.rootdir, 'gpsppp.met'))

        shutil.copyfile(os.path.join(self.ppp_path, 'gpsppp.stc'), os.path.join(self.rootdir, 'gpsppp.stc'))
        shutil.copyfile(os.path.join(self.ppp_path, 'gpsppp.svb_gnss_yrly'),
                        os.path.join(self.rootdir, 'gpsppp.svb_gnss_yrly'))
        shutil.copyfile(os.path.join(self.ppp_path, 'gpsppp.flt'), os.path.join(self.rootdir, 'gpsppp.flt'))
        shutil.copyfile(os.path.join(self.ppp_path, 'gpsppp.stc'), os.path.join(self.rootdir, 'gpsppp.stc'))
        shutil.copyfile(os.path.join(self.ppp_path, 'gpsppp.trf'), os.path.join(self.rootdir, 'gpsppp.trf'))
        shutil.copyfile(os.path.join(self.atx), os.path.join(self.rootdir, os.path.basename(self.atx)))

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

        orbits1 = GetSp3Orbits(options['sp3'], self.rinex.date, type, self.rootdir, True)
        orbits2 = GetSp3Orbits(options['sp3'], self.rinex.date + 1, type,
                               self.rootdir, True)

        clocks1 = GetClkFile(options['sp3'], self.rinex.date, type, self.rootdir, True)
        clocks2 = GetClkFile(options['sp3'], self.rinex.date + 1, type,
                             self.rootdir, True)

        try:
            eop_file = GetEOP(options['sp3'], self.rinex.date, type, self.rootdir)
            eop_file = eop_file.eop_filename
        except EOPException:
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
            raise Runpppexceptionnan('One or more coordinate is NaN')

        if numpy.isnan(x) or numpy.isnan(y) or numpy.isnan(z):
            raise Runpppexceptionnan('One or more coordinate is NaN')

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
                raise Runpppexceptionnan('One or more sigma is NaN')

        else:
            sx, sxy, sxz = re.findall(r'X\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)'
                                      r'\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]
            sy, syz = re.findall(r'Y\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]
            sz = re.findall(r'Z\(m\)\s+(-?\d+\.\d+|[nN]a[nN]|\*+)', section)[0]

            if '*' in sx or '*' in sy or '*' in sz or '*' in sxy or '*' in sxz or '*' in syz:
                raise Runpppexceptionnan('Sigmas are NaN')
            else:
                sx = float(sx)
                sy = float(sy)
                sz = float(sz)
                sxy = float(sxy)
                sxz = float(sxz)
                syz = float(syz)

        if numpy.isnan(sx) or numpy.isnan(sy) or numpy.isnan(sz) or numpy.isnan(sxy) or numpy.isnan(sxz) or numpy.isnan(
                syz):
            raise Runpppexceptionnan('Sigmas are NaN')

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
            raise RunPPPException(
                'Error while running PPP: could not find the antenna and radome in antex file. '
                'Check RINEX header for formatting issues in the ANT # / TYPE field. RINEX header follows:\n' + ''.join(
                    self.rinex.get_header()))

        if self.strict and not self.check_otl(self.proc_parameters):
            raise RunPPPException(
                'Error while running PPP: could not find the OTL coefficients. '
                'Check RINEX header for formatting issues in the APPROX ANT POSITION field. If APR is too far from OTL '
                'coordinates (declared in the HARPOS or BLQ format) NRCAN will reject the coefficients. '
                'OTL coefficients record follows:\n' + self.otl_coeff)

        if not self.check_eop(self.file_summary):
            raise Runpppexceptioneoperror('EOP returned NaN in Pole XYZ.')

        # parse rejected and accepted observations
        self.processed_obs, self.rejected_obs = self.get_pr_observations(self.observation_session, self.kinematic)

        if self.processed_obs == 0:
            raise Runpppexceptionzeroprocepochs('PPP returned zero processed epochs')

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
                cmd = RunCommand(self.ppp, 30, self.rootdir, 'input.inp')
                out, err = cmd.run_shell()
                # TODO: Program becomes a zombie after this step.
                if '*END - NORMAL COMPLETION' not in out:

                    if 'Fortran runtime error: End of file' in err and i == 0:
                        # error detected, try again!
                        continue

                    msg = 'PPP ended abnormally for ' + self.rinex.rinex_path + ':\nerr: ' + err + '\nout: ' + out
                    if raise_error:
                        raise RunPPPException(msg)
                    else:
                        return False, msg
                else:
                    with open(os.path.join(self.rootdir, self.rinex.rinex[:-3] + 'sum')) as f:
                        self.out = f.readlines()

                    with open(os.path.join(self.rootdir, self.rinex.rinex[:-3] + 'pos')) as f:
                        self.pos = f.readlines()
                    break

        except RunCommandWithRetryExeception as e:
            msg = str(e)
            if raise_error:
                raise RunPPPException(e)
            else:
                return False, msg
        except IOError as e:
            raise RunPPPException(e)

        return True, ''

    def exec_ppp(self):

        while True:
            # execute PPP but do not raise an error if timed out
            result, message = self.__exec_ppp__(False)

            if result:
                try:
                    self.parse_summary()
                    break

                except Runpppexceptioneoperror:
                    # problem with EOP!
                    if self.eop_file != 'dummy.eop':
                        self.eop_file = 'dummy.eop'
                        self.config_session()
                    else:
                        raise

                except (Runpppexceptionnan, Runpppexceptiontoofewacceptedobs, Runpppexceptionzeroprocepochs):
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
                    raise RunPPPException(message)

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
            shutil.rmtree(self.rootdir)

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self


class RinexRecord:

    def __init__(self, network_code=None, station_code=None):
        self.StationCode = station_code
        self.NetworkCode = network_code

        self.header = None
        self.data = None
        self.firstObs = None
        self.datetime_firstObs = None
        self.datetime_lastObs = None
        self.lastObs = None
        self.antType = None
        self.marker_number = None
        self.marker_name = station_code
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

    def __init__(self, network_code, station_code, origin_file, no_cleanup=False, allow_multiday=False):
        """
        pyRinex initialization
        if file is multiday, DO NOT TRUST date object for initial file. Only use pyRinex objects contained in the multiday list
        """
        super().__init__(network_code, station_code)

        self.no_cleanup = no_cleanup
        self.origin_file = origin_file

        # check that the rinex file name is valid!
        if not self.is_rinex_name(origin_file):
            raise RinexException('File name does not follow the RINEX(Z)/CRINEX(Z) naming convention: %s'
                                 % (os.path.basename(origin_file)))

        rootdir = os.path.join('production', 'rinex')
        self.rootdir = os.path.join(rootdir, str(uuid.uuid4()))

        # create a production folder to analyze the rinex file
        os.makedirs(self.rootdir, exist_ok=True)

        filename = os.path.basename(origin_file)

        self.origin_type = self.identify_type(filename)
        self.local_copy = os.path.abspath(os.path.join(self.rootdir, filename))

        self.rinex = self.to_format(filename, TYPE_RINEX)
        self.crinez = self.to_format(filename, TYPE_CRINEZ)

        # get the paths
        self.crinez_path = os.path.join(self.rootdir, self.crinez)
        self.rinex_path = os.path.join(self.rootdir, self.rinex)

        self.log += [Event(StationCode=self.StationCode,
                           NetworkCode=self.NetworkCode,
                           Description='Origin type is %i' % self.origin_type)]

        shutil.copy(origin_file, self.rootdir)

        if self.origin_type in (TYPE_CRINEZ, TYPE_CRINEX, TYPE_RINEZ):
            self.uncompress()

        # check basic infor in the rinex header to avoid problems with RinSum

        self.check_header()

        if self.rinex_version >= 3:
            self.convert_rinex3to2()

        # process the output
        self.parse_output(self.run_rinsum())

        # DDG: new interval checking after running RinSum
        # check the sampling interval
        self.check_interval()

        # check for files that have more than one day inside (yes, there are some like this... amazing)
        # condition is: the start and end date don't match AND
        # either there is more than two hours in the second day OR
        # there is more than one day of data
        if self.datetime_lastObs.date() != self.datetime_firstObs.date() and not allow_multiday:
            # more than one day in this file. Is there more than one hour? (at least in principle, based on the time)
            first_obs = dt.datetime(self.datetime_lastObs.date().year,
                                    self.datetime_lastObs.date().month,
                                    self.datetime_lastObs.date().day)

            if (self.datetime_lastObs - first_obs).total_seconds() >= 3600:
                # the file has more than one day in it...
                # use teqc to window the data
                if not self.multiday_handle(origin_file):
                    return
            else:
                # window the data to remove superfluous epochs
                last_obs = dt.datetime(self.datetime_firstObs.date().year,
                                       self.datetime_firstObs.date().month,
                                       self.datetime_firstObs.date().day, 23, 59, 59)
                self.window_data(end=last_obs)
                self.log_event('RINEX had incomplete epochs (or < 1 hr) outside of the corresponding UTC day -> '
                               'Data windowed to one UTC day.')

        # reported date for this file is session/2
        self.date = Date(datetime=self.datetime_firstObs + (self.datetime_lastObs - self.datetime_firstObs) / 2)

        # DDG: calculate the completion of the file (at sampling rate)
        # completion of day
        # done after binning so that if the file is a multiday we don't get a bad completion
        self.completion = self.epochs * self.interval / 86400
        # completion of time window in file
        self.rel_completion = self.epochs * self.interval / ((self.datetime_lastObs -
                                                              self.datetime_firstObs).total_seconds() + self.interval)

        # load the RinexRecord class
        self.load_record()

    def __del__(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __enter__(self):
        return self

    def __add__(self, other):

        if not isinstance(other, ReadRinex):
            raise RinexException('type: ' + type(other) + ' invalid.  Can only splice two RINEX objects.')

        if self.StationCode != other.StationCode:
            raise RinexException('Cannot splice together two different stations!')

        # determine which one goes first
        if other.datetime_firstObs > self.datetime_firstObs:
            f1 = self
            f2 = other
        else:
            f1 = other
            f2 = self

        # now splice files
        cmd = RunCommand('teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 +obs %s.t %s %s' % (
            f1.rinex_path, f1.rinex_path, f2.rinex_path), 5)

        # leave errors un-trapped on purpose (will raise an error to the parent)
        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            filename = shutil.move(f1.rinex_path + '.t', f1.rinex_path)
            return ReadRinex(self.NetworkCode, self.StationCode, filename, allow_multiday=True)
        else:
            raise RinexException(err)

    def __repr__(self):
        return 'ReadRinex(' + self.NetworkCode + ', ' + self.StationCode + ', ' + str(self.date.year) + ', ' + str(
            self.date.doy) + ')'

    def log_event(self, desc):
        self.log += [Event(StationCode=self.StationCode, NetworkCode=self.NetworkCode, Description=desc)]

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
            with open(self.rinex_path) as fileio:
                rinex = fileio.readlines()
        except Exception:
            raise

        if not any("END OF HEADER" in s for s in rinex):
            raise Rinexexceptionbadfile('Invalid header: could not find END OF HEADER tag.')

        # find the end of header
        index = [i for i, item in enumerate(rinex) if 'END OF HEADER' in item][0]
        # delete header
        del rinex[0:index + 1]

        self.data = rinex

    def replace_record(self, header, record, new_values):

        if record not in list(self.required_records.keys()):
            raise RinexException('Record ' + record + ' not implemented!')

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
                        raise Rinexexceptionbadfile(
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
                        first_obs = self.get_firstobs()
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
            raise Rinexexceptionbadfile('Unfixable RINEX header: could not find RINEX VERSION / TYPE')

        # now check that all the records where included! there's missing ones, then force them
        if not all([item['found'] for item in list(self.required_records.values())]):
            # get the keys of the missing records
            missing_records = {item: self.required_records[item] for item in self.required_records if
                               self.required_records[item]['found'] == False}

            for record in list(missing_records.keys()):
                if '# / TYPES OF OBSERV' in record:
                    raise Rinexexceptionbadfile('Unfixable RINEX header: could not find # / TYPES OF OBSERV')

                new_header += [self.format_record(missing_records, record, missing_records[record]['default']) + '\n']
                new_header += [self.format_record(self.required_records, 'COMMENT',
                                                  'pyRinex: WARN! default value to fix file!') + '\n']
                self.log_event('Missing required RINEX record added: ' + record)

        new_header += [''.ljust(60, ' ') + 'END OF HEADER\n']

        self.write_rinex(new_header)

    def uncompress(self):

        if self.origin_type in (TYPE_CRINEZ, TYPE_CRINEX):

            size = os.path.getsize(self.local_copy)

            # run crz2rnx with timeout structure
            cmd = RunCommand('crz2rnx -f -d ' + self.local_copy, 30)
            try:
                _, err = cmd.run_shell()
            except RunCommandWithRetryExeception as e:
                # catch the timeout except and pass it as a pyRinexException
                raise RinexException(str(e))

            # the uncompressed-unhatanaked file size must be at least > than the crinez
            if os.path.isfile(self.rinex_path):
                if err and os.path.getsize(self.rinex_path) <= size:
                    raise Rinexexceptionbadfile(
                        "Error in ReadRinex.__init__ -- crz2rnx: error and empty file: " + self.origin_file + ' -> ' + err)
            else:
                if err:
                    raise RinexException('Could not create RINEX file. crz2rnx stderr follows: ' + err)
                else:
                    raise RinexException(
                        'Could not create RINEX file. Unknown reason. Possible problem with crz2rnx?')

        elif self.origin_type is TYPE_RINEZ:
            # create an unzip script
            create_unzip_script(os.path.join(self.rootdir, 'uncompress.sh'))

            cmd = RunCommand('./uncompress.sh -f -d ' + self.local_copy, 30, self.rootdir)
            try:
                _, _ = cmd.run_shell()
            except RunCommandWithRetryExeception as e:
                # catch the timeout except and pass it as a pyRinexException
                raise RinexException(str(e))

    def convert_rinex3to2(self):

        # most programs still don't support RINEX 3 (partially implemented in this code)
        # convert to RINEX 2.11 using RinEdit
        cmd = RunCommand('RinEdit --IF %s --OF %s.t --ver2' % (self.rinex, self.rinex), 15, self.rootdir)

        try:
            out, _ = cmd.run_shell()

            if 'exception' in out.lower():
                raise Rinexexceptionbadfile('RinEdit returned error converting to RINEX 2.11:\n' + out)

            if not os.path.exists(self.rinex_path + '.t'):
                raise Rinexexceptionbadfile('RinEdit failed to convert to RINEX 2.11:\n' + out)

            # if all ok, move_new converted file to rinex_path
            os.remove(self.rinex_path)
            shutil.move(self.rinex_path + '.t', self.rinex_path)
            # change version
            self.rinex_version = 2.11

            self.log_event('Origin file was RINEX 3 -> Converted to 2.11')

        except RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise RinexException(str(e))

    def run_rinsum(self):
        # run RinSum to get file information
        cmd = RunCommand('RinSum --notable ' + self.rinex_path, 45)  # DDG: increased from 21 to 45.
        try:

            output, _ = cmd.run_shell()
        except RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise RinexException(str(e))

        # write RinSum output to a log file (debug purposes)
        info = open(self.rinex_path + '.log', 'w')
        info.write(output)
        info.close()

        return output

    def is_rinex_name(self, filename):

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

            shutil.rmtree(temp_path)
            # now self points the the binned version of the rinex
            continue_statements = False

        return continue_statements

    def split_file(self):

        # run in the local folder to get the files inside rootdir
        cmd = RunCommand('teqc -n_GLONASS 64 -n_GPS 64 -n_Galileo 64 -n_SBAS 64 -tbin 1d rnx ' +
                         self.rinex, 45, self.rootdir)
        try:
            _, err = cmd.run_shell()
        except RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise RinexException(str(e))

        # successfully binned the file
        # delete current file and rename the new files
        os.remove(self.rinex_path)

        # now we should have as many files named rnxDDD0.??o as days inside the RINEX
        for file in os.listdir(self.rootdir):
            if file[0:3] == 'rnx' and self.identify_type(file) is TYPE_RINEX:
                # rename file
                shutil.move(os.path.join(self.rootdir, file), os.path.join(self.rootdir,
                                                                           file.replace('rnx', self.StationCode)))
                # get the info for this file
                try:
                    rnx = ReadRinex(self.NetworkCode, self.StationCode,
                                    os.path.join(self.rootdir, file.replace('rnx', self.StationCode)))
                    # append this rinex object to the multiday list
                    self.multiday_rnx_list.append(rnx)
                except (RinexException, Rinexexceptionbadfile):
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
                raise Rinexexceptionsingleepoch('RINEX interval equal to zero. Single epoch or bad RINEX file. '
                                                  'Reported epochs in file were %s' % self.epochs)
            else:
                raise Rinexexceptionsingleepoch('RINEX interval equal to zero. Single epoch or bad RINEX file. '
                                                  'No epoch information to report. The output from RinSum was:\n' +
                                                output)

        elif self.interval > 120:
            raise Rinexexceptionbadfile('RINEX sampling interval > 120s. The output from RinSum was:\n' + output)

        elif self.epochs * self.interval < 3600:
            raise Rinexexceptionbadfile('RINEX file with < 1 hr of observation time. '
                                          'The output from RinSum was:\n' + output)

        try:
            yy, mm, dd, hh, MM, ss = [int(x) for x in re.findall(r'^Computed first epoch:\s*(\d+)\/(\d+)\/(\d+)'
                                                                 r'\s(\d+):(\d+):(\d+)', output, re.MULTILINE)[0]]
            yy = check_year(yy)
            self.datetime_firstObs = dt.datetime(yy, mm, dd, hh, MM, ss)
            self.firstObs = self.datetime_firstObs.strftime('%Y/%m/%d %H:%M:%S')

            yy, mm, dd, hh, MM, ss = [int(x) for x in re.findall(r'^Computed last\s*epoch:\s*(\d+)\/(\d+)'
                                                                 r'\/(\d+)\s(\d+):(\d+):(\d+)', output,
                                                                 re.MULTILINE)[0]]
            yy = check_year(yy)
            self.datetime_lastObs = dt.datetime(yy, mm, dd, hh, MM, ss)
            self.lastObs = self.datetime_lastObs.strftime('%Y/%m/%d %H:%M:%S')

            if self.datetime_lastObs <= self.datetime_firstObs:
                # bad rinex! first obs > last obs
                raise Rinexexceptionbadfile('Last observation (' + self.lastObs + ') <= first observation (' +
                                            self.firstObs + ')')

        except Exception:
            raise RinexException(self.rinex_path + ': error in ReadRinex.parse_output: the output for first/last obs '
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
            raise RinexException("Warning in ReadRinex.parse_output: " + warn[0])

        warn = re.findall('(.*unexpected exception.*)', output, re.MULTILINE)
        if warn:
            raise RinexException("unexpected exception in ReadRinex.parse_output: " + warn[0])

        warn = re.findall('(.*Exception.*)', output, re.MULTILINE)
        if warn:
            raise RinexException("Exception in ReadRinex.parse_output: " + warn[0])

        warn = re.findall('(.*no data found. Are time limits wrong.*)', output, re.MULTILINE)
        if warn:
            raise RinexException('RinSum: no data found. Are time limits wrong for file ' + self.rinex +
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
                        date = Date(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
                    except DateException as e:
                        raise Rinexexceptionbadfile(str(e))

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
                with open(self.rinex_path) as fileio:
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
        out = None
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

        except RinexException:
            # print str(e)
            # ooops, something went wrong, try with local file (without removing systems or decimating)
            rnx = self
            # raise pyRinexExceptionBadFile('During decimation or remove_systems (to run auto_coord),
            # teqc returned: %s' + str(e))

        # copy brdc orbit
        shutil.copyfile(brdc.brdc_path, os.path.join(rnx.rootdir, brdc.brdc_filename))

        # check if the apr coordinate is zero and iterate more than once if true
        if self.x == 0 and self.y == 0 and self.z == 0:
            max_it = 2
        else:
            max_it = 1

        for i in range(max_it):

            cmd = RunCommand(
                'sh_rx2apr -site ' + rnx.rinex + ' -nav ' + brdc.brdc_filename + ' -chi ' + str(chi_limit), 40,
                rnx.rootdir)
            # leave errors un-trapped on purpose (will raise an error to the parent)
            out, err = cmd.run_shell()

            if err != '' and err is not None:
                raise Rinexexceptionnoautocoord(str(err) + '\n' + out)
            else:
                # check that the Final chi**2 is < 3
                for line in out.split('\n'):
                    if '* Final sqrt(chi**2/n)' in line:
                        chi = line.split()[-1]

                        if chi == 'NaN':
                            raise Rinexexceptionnoautocoord('chi2 = NaN! ' + str(err) + '\n' + out)

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

        raise Rinexexceptionnoautocoord(str(out) + '\nLIMIT FOR CHI**2 was %i' % chi_limit)

    def window_data(self, start=None, end=None, copyto=None):
        """
        Window the RINEX data using TEQC
        :param copyto:
        :param start: a start date_time or self.firstObs if None
        :param end: a end date_time or self.lastObs if None
        :return:
        """
        if start is None:
            start = self.datetime_firstObs
            self.log_event('Setting start = first obs in window_data')

        if end is None:
            end = self.datetime_lastObs
            self.log_event('Setting end = last obs in window_data')

        cmd = RunCommand(
            'teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 -st %i%02i%02i%02i%02i%02i -e %i%02i%02i%02i%02i%02i +obs %s.t %s' % (
                start.year, start.month, start.day, start.hour, start.minute, start.second,
                end.year, end.month, end.day, end.hour, end.minute, end.second, self.rinex_path, self.rinex_path), 5)

        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            # delete the original file and replace with .t
            if copyto is None:
                os.remove(self.rinex_path)
                shutil.move(self.rinex_path + '.t', self.rinex_path)
                self.datetime_firstObs = start
                self.datetime_lastObs = end
                self.firstObs = self.datetime_firstObs.strftime('%Y/%m/%d %H:%M:%S')
                self.lastObs = self.datetime_lastObs.strftime('%Y/%m/%d %H:%M:%S')
            else:
                shutil.move(self.rinex_path + '.t', copyto)
        else:
            raise RinexException(err)

        return

    def decimate(self, decimate_rate, copyto=None):
        # if copy to is passed, then the decimation is done on the copy of the file, not on the current rinex.
        # otherwise, decimation is done in current rinex
        if copyto is not None:
            copy_file(self.rinex_path, copyto)
        else:
            copyto = self.rinex_path
            self.interval = decimate_rate

        if self.rinex_version < 3:
            cmd = RunCommand('teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 -O.dec %i '
                             '+obs %s.t %s' % (decimate_rate, copyto, copyto), 5)
            # leave errors un-trapped on purpose (will raise an error to the parent)
        else:
            cmd = RunCommand('RinEdit --IF %s --OF %s.t --TN %i --TB %i,%i,%i,%i,%i,%i'
                             % (os.path.basename(copyto), os.path.basename(copyto), decimate_rate,
                                self.date.year, self.date.month, self.date.day, 0, 0, 0),
                             15, self.rootdir)
        out, err = cmd.run_shell()

        if 'teqc: failure to read' not in str(err):
            # delete the original file and replace with .t
            os.remove(copyto)
            shutil.move(copyto + '.t', copyto)
        else:
            raise RinexException(err)

        self.log_event('RINEX decimated to %is (applied to %s)' % (decimate_rate, str(copyto)))

        return

    def remove_systems(self, systems=('R', 'E', 'S'), copyto=None):
        # if copy to is passed, then the system removal is done on the copy of the file, not on the current rinex.
        # other wise, system removal is done to current rinex
        if copyto is not None:
            copy_file(self.rinex_path, copyto)
        else:
            copyto = self.rinex_path

        if self.rinex_version < 3:
            rsys = '-' + ' -'.join(systems)
            cmd = RunCommand(
                'teqc -n_GLONASS 64 -n_GPS 64 -n_SBAS 64 -n_Galileo 64 %s +obs %s.t %s' % (rsys, copyto, copyto), 5)
        else:
            rsys = ' --DS '.join(systems)
            cmd = RunCommand(
                'RinEdit --IF %s --OF %s.t --DS %s' % (os.path.basename(copyto), os.path.basename(copyto), rsys), 15,
                self.rootdir)

        # leave errors un-trapped on purpose (will raise an error to the parent)
        out, err = cmd.run_shell()

        if not 'teqc: failure to read' in str(err):
            # delete the original file and replace with .t
            os.remove(copyto)
            shutil.move(copyto + '.t', copyto)
            # if working on local copy, reload the rinex information
            if copyto == self.rinex_path:
                # reload information from this file
                self.parse_output(self.run_rinsum())
        else:
            raise RinexException(err)

        self.log_event('Removed systems %s (applied to %s)' % (','.join(systems), str(copyto)))

        return

    def normalize_header(self, NewValues, brdc=None, x=None, y=None, z=None):
        # this function gets rid of the heaer information and replaces it with the station info (trusted)
        # should be executed before calling PPP or before rebuilding the Archive
        # new function now accepts a dictionary OR a station info object

        if type(NewValues) is StationInfo:
            if NewValues.date is not None and NewValues.date != self.date:
                raise RinexException('The StationInfo object was initialized for a different date than that of the '
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
            raise RinexException(
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
                                         dt.datetime.now().strftime('%Y/%m/%d %H:%M'))

        self.write_rinex(new_header)

        return

    def apply_file_naming_convention(self):
        """
        function to rename a file to make it consistent with the RINEX naming convention
        :return:
        """
        # is the current rinex filename valid?
        fileparts = parse_crinex_rinex_filename(self.rinex)

        if fileparts:
            doy = int(fileparts[1])
            year = int(get_norm_year_str(fileparts[3]))
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
            raise RinexException('Destination must be an absolute path')

        if destiny_type is TYPE_CRINEZ:
            dst = self.crinez
        elif destiny_type is TYPE_RINEX:
            dst = self.rinex
        else:
            dst = self.to_format(self.rinex, destiny_type)

        filename = ''
        # determine action base on origin type
        if self.origin_type == destiny_type:
            # intelligent move_new (creates folder and checks for file existence)
            # origin and destiny match, do the thing directly
            filename = shutil.move(self.origin_file, os.path.join(path, dst))
        else:
            # if other types are requested, or origin is not the destiny type, then use local file and delete the
            if destiny_type is TYPE_RINEX:
                filename = shutil.move(self.rinex_path, os.path.join(path, dst))

            elif destiny_type is TYPE_CRINEZ:
                filename = self.compress_local_copyto(path)

            elif destiny_type is TYPE_CRINEX:
                cmd = RunCommand('rnx2crx -f ' + self.rinex_path, 45)
                try:
                    _, err = cmd.run_shell()

                    if os.path.getsize(os.path.join(self.rootdir, self.to_format(self.rinex, TYPE_CRINEX))) == 0:
                        raise RinexException(
                            'Error in move_origin_file: compressed version of ' + self.rinex_path + ' has zero size!')
                except RunCommandWithRetryExeception as e:
                    # catch the timeout except and pass it as a pyRinexException
                    raise RinexException(str(e))

            elif destiny_type is TYPE_RINEZ:
                raise RinexException('pyRinex will not natively generate a RINEZ file.')

        # to keep everything consistent, also change the local copies of the file
        if filename != '':
            self.rename(filename)
            # delete original (if the dest exists!)
            if os.path.isfile(self.origin_file):
                if os.path.isfile(os.path.join(path, dst)):
                    os.remove(self.origin_file)
                else:
                    raise RinexException(
                        'New \'origin_file\' (%s) does not exist!' % os.path.isfile(os.path.join(path, dst)))

            # change origin file reference
            self.origin_file = os.path.join(path, dst)
            self.origin_type = destiny_type

            self.log_event('Origin moved to %s and converted to %i' % (self.origin_file, destiny_type))

        return filename

    def compress_local_copyto(self, path):
        # this function compresses and moves the local copy of the rinex
        # meant to be used when a multiday rinex file is encountered and we need to move_new it to the repository

        # compress the rinex into crinez. Make the filename
        crinez = self.to_format(self.rinex, TYPE_CRINEZ)

        # we make the crinez again (don't use the existing from the database) to apply any corrections
        # made during the __init__ stage. Notice the -f in rnx2crz
        cmd = RunCommand('rnx2crz -f ' + self.rinex_path, 45)
        try:
            _, err = cmd.run_shell()

            if os.path.getsize(os.path.join(self.rootdir, crinez)) == 0:
                raise RinexException(
                    'Error in compress_local_copyto: compressed version of ' + self.rinex_path + ' has zero size!')
        except RunCommandWithRetryExeception as e:
            # catch the timeout except and pass it as a pyRinexException
            raise RinexException(str(e))

        filename = copy_file(os.path.join(self.rootdir, crinez), os.path.join(path, crinez))

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
                    shutil.move(self.rinex_path, os.path.join(self.rootdir, rinex))

                self.rinex_path = os.path.join(self.rootdir, rinex)

                # rename the files
                # check if local crinez exists (possibly made by compress_local_copyto)
                if os.path.isfile(self.crinez_path):
                    shutil.move(self.crinez_path, os.path.join(self.rootdir, crinez))

                self.crinez_path = os.path.join(self.rootdir, crinez)

                # rename the local copy of the origin file (if exists)
                # only cases that need to be renamed (again, IF they exist; they shouldn't, but just in case)
                # are RINEZ and CRINEX since RINEX and CRINEZ are renamed above
                if os.path.isfile(self.local_copy):
                    if self.origin_type is TYPE_RINEZ:
                        local = os.path.basename(self.to_format(new_name, TYPE_RINEZ))
                        shutil.move(self.local_copy, os.path.join(self.rootdir, local))
                    elif self.origin_type is TYPE_CRINEX:
                        local = os.path.basename(self.to_format(new_name, TYPE_CRINEX))
                        shutil.move(self.local_copy, os.path.join(self.rootdir, local))

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
            raise RinexException('Invalid filename format: ' + filename)

    def to_format(self, filename, to_type):

        path = os.path.dirname(filename)
        filename = os.path.basename(filename)
        type = self.identify_type(filename)

        if type in (TYPE_RINEX, TYPE_CRINEX):
            filename = filename[0:-1]
        elif type in (TYPE_CRINEZ, TYPE_RINEZ):
            filename = filename[0:-3]
        else:
            raise RinexException('Invalid filename format: ' + filename)

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
            raise RinexException(
                'Invalid to_type format. Accepted formats: '
                'CRINEX (.??d), CRINEZ (.??d.Z), RINEX (.??o) and RINEZ (.??o.Z)')

    def cleanup(self):
        if self.rinex_path and not self.no_cleanup:
            # remove all the directory contents
            try:
                shutil.rmtree(self.rootdir)
            except OSError:
                # something was not found, ignore (we are deleting anyways)
                pass

            # if it's a multiday rinex, delete the multiday objects too
            if self.multiday:
                for Rnx in self.multiday_rnx_list:
                    Rnx.cleanup()

        return


class GetClkFile(OrbitalProduct):

    def __init__(self, clk_archive, date, sp3types, copyto, no_cleanup=False):

        # try both compressed and non-compressed sp3 files
        # loop through the types of sp3 files to try
        self.clk_path = None
        self.no_cleanup = no_cleanup

        for sp3type in sp3types:
            self.clk_filename = sp3type + date.wwwwd() + '.clk'

            try:
                OrbitalProduct.__init__(self, clk_archive, date, self.clk_filename, copyto)
                self.clk_path = self.file_path
                break
            except Productsexceptionunreasonabledate:
                raise
            except ProductsException:
                # if the file was not found, go to next
                pass

        # if we get here and self.sp3_path is still none, then no type of sp3 file was found
        if self.clk_path is None:
            raise Clkexception(
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


class GetEOP(OrbitalProduct):

    def __init__(self, sp3archive, date, sp3types, copyto):

        # try both compressed and non-compressed sp3 files
        # loop through the types of sp3 files to try
        self.eop_path = None

        for sp3type in sp3types:

            self.eop_filename = sp3type + date.wwww() + '7.erp'

            try:
                OrbitalProduct.__init__(self, sp3archive, date, self.eop_filename, copyto)
                self.eop_path = self.file_path
                self.type = sp3type
                break

            except Productsexceptionunreasonabledate:
                raise

            # rapid EOP files do not work in NRCAN PPP
            # except pyProductsException:
            #    # rapid orbits do not have 7.erp, try wwwwd.erp

            #    self.eop_filename = sp3type + date.wwwwd() + '.erp'

            #    OrbitalProduct.__init__(self, sp3archive, date, self.eop_filename, copyto)
            #    self.eop_path = self.file_path

            except ProductsException:
                # if the file was not found, go to next
                pass

        # if we get here and self.sp3_path is still none, then no type of sp3 file was found
        if self.eop_path is None:
            raise EOPException(
                'Could not find a valid earth orientation parameters file for gps week ' + date.wwww() +
                ' using any of the provided sp3 types')

        return


class GetSp3Orbits(OrbitalProduct):

    def __init__(self, sp3archive, date, sp3types, copyto, no_cleanup=False):

        # try both compressed and non-compressed sp3 files
        # loop through the types of sp3 files to try
        self.sp3_path = None
        self.RF = None
        self.no_cleanup = no_cleanup

        for sp3type in sp3types:
            self.sp3_filename = sp3type + date.wwwwd() + '.sp3'

            try:
                super().__init__(sp3archive, date, self.sp3_filename, copyto)
                self.sp3_path = self.file_path
                self.type = sp3type
                break
            except Productsexceptionunreasonabledate:
                raise
            except ProductsException:
                # if the file was not found, go to next
                pass

        # if we get here and self.sp3_path is still none, then no type of sp3 file was found
        if self.sp3_path is None:
            raise Sp3exception(
                'Could not find a valid orbit file (types: ' + ', '.join(sp3types) + ') for week ' + str(
                    date.gpsWeek) + ' day ' + str(date.gpsWeekDay) + ' using any of the provided sp3 types')
        else:
            # parse the RF of the orbit file
            try:
                with open(self.sp3_path) as fileio:
                    line = fileio.readline()

                    self.RF = line[46:51].strip()
            except Exception:
                raise

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


class shell_cmd(threading.Thread):
    """
    This class is implemented in order to run shell scripts and external programs.
    """

    def __init__(self, cmd, cwd=os.getcwd(), cat_file=None):
        self.stdout = None
        self.stderr = None
        self.cmd = cmd
        self.cwd = cwd
        self.cat_file = cat_file
        self.p = None
        super().__init__()

    def run(self):
        while True:
            logging.debug(self.cmd)
            try:
                if self.cat_file:
                    with open(os.path.join(self.cwd, self.cat_file)) as fin:
                        logging.debug(self.cat_file)
                        self.p = subprocess.run(self.cmd.split(), stdin=fin,
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd,
                                                universal_newlines=True)
                        logging.debug(self.p.check_returncode())
                else:
                    self.p = subprocess.run(self.cmd.split(), stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, cwd=self.cwd,
                                            universal_newlines=True)
                    logging.debug(self.p.check_returncode())

                self.stdout, self.stderr = self.p.stdout, self.p.stderr
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.p.terminate()
        except Exception:
            pass

        self.p = None

    def __enter__(self):
        return self


class RunCommand:
    def __init__(self, cmd, time_out, cwd=os.getcwd(), cat_file=None):
        self.stdout = None
        self.stderr = None
        self.cmd = cmd
        self.time_out = time_out
        self.cwd = cwd
        self.cat_file = cat_file

    def run_shell(self):
        while True:
            with shell_cmd(self.cmd, self.cwd, self.cat_file) as cmd:
                try:
                    cmd.start()
                    cmd.join(timeout=self.time_out)
                    # remove non-ASCII chars
                    if cmd.stderr is not None:
                        cmd.stderr = ''.join([i if ord(i) < 128 else ' ' for i in cmd.stderr])
                    return str(cmd.stdout), str(cmd.stderr)
                except Exception as e:
                    print(e)
                    sys.exit()
