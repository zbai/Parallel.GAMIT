"""
archive/__init__.py

Used to add RINEX files to the archive.
"""
import gpys
import argparse
import logging
import shutil
import sys
import dispy
import subprocess

# Globals


def node_setup() -> int:
    # TODO: Import the packages required to run parse_data_in
    return 0


def parse_data_in(filepath: str, config, n: int) -> str:
    """
    Runs PPP on a RINEX and either send it to the rejected folder, retry folder, lock it or add it to an existing
    station record.  The archive variable is used to pass what is already in the archive and the file structure.
    :param filepath: Path to the RINEX we're working on.
    :param config: gpys.ReadOptions object containing parameters from the .cfg file.
    :param n: The job number.
    :return: A string?... TBD
    """
    import logging
    import socket
    import os
    import pathlib
    import datetime as dt
    import shutil
    # Internal variable set up and definition for later reference.
    nodelogger = None               # Internal logger for the function.
    ofile = None                    # Original file converted to a pathlib.Path object.
    file = None                     # pathlib.Path object for the RINEX we're working on, updated as it is operated on.
    prodpath = None                 # pathlib.Path object defining where the production folder is.
    start_date = None               # datetime.date object for the RINEX as determined by TEQC
    orbit = None                    # pathlib.Path object with the archive path to the BRDC orbit file.
    sp3path = None                  # Path to the IGS sp3, also used for the clk files
    nextweek = None                 # Used to gather up the next day's clk/sp3  file
    nextday = None                  # Ditto but the next day
    ppp_input_string = 'BadFilename'  # The STDIN input to the PPP command.
    try:
        ofile = pathlib.Path(filepath)
        nodelogger = logging.getLogger('parse_data_in.{}'.format(n))
        nodelogger.setLevel(logging.DEBUG)
        nodestream = logging.StreamHandler()
        nodestream.setFormatter(logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s -'
                                                  ' %(threadName)s %(message)s',
                                                  '%Y-%m-%d %H:%M:%S'))
        nodestream.setLevel(logging.DEBUG)
        nodelogger.addHandler(nodestream)
        nodelogger.debug('Logger setup on {}.'.format(socket.gethostname()))
    except Exception as e:
        # TODO: Figure out what kind of command we can use here to quit the function on an exception.
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    try:
        # TODO: Set up the environment (copy files & stuff)...
        prodpath = pathlib.Path('production/job{}'.format(n))
        prodpath = os.sep.join([config.options['working_dir'], str(prodpath)])
        prodpath = pathlib.Path(prodpath)
        nodelogger.debug('Creating folder: {}'.format(prodpath.name))
        os.makedirs(str(prodpath), exist_ok=True)
        shutil.move(str(ofile), str(prodpath))
        file = prodpath / ofile.name
        nodelogger.debug('Working with {}'.format(file.name))
        os.chdir(str(prodpath))
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # TODO: Convert into an uncompressed obs file.
    try:
        # TODO: Add some error checking to the subprocess runs.
        if '.Z' in file.suffixes:
            nodelogger.debug('Decompressing {}'.format(file.name))
            subprocess.run(['uncompress', '-f', file.name])
            # Remove the .Z suffix
            file = file.with_suffix('')
        if file.match('*.??d'):
            subprocess.run(['crx2rnx', '-f', file])
            os.remove(file)
            # Change the internal path suffix to o from d
            file = file.with_suffix(file.suffix.replace('d', 'o'))
        else:
            raise Exception('Unrecognized file extension: {}'.format(file.suffix))
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # First get the date information from the RINEX file using teqc +meta:
    try:
        nodelogger.debug('Loading metadata from {}'.format(file.name))
        metadata = subprocess.run(['teqc',
                                   '+meta',
                                   file.name],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  encoding='utf-8',
                                  check=True)
        # Parse out the 'start date & time' field of the TEQC metadata record.
        start_date = [x for x in metadata.stdout.splitlines() if 'start date & time:' in x][0]
        start_date = [int(x) for x in start_date.partition(':')[-1].strip().partition(' ')[0].split('-')]
        start_date = dt.date(start_date[0], start_date[1], start_date[2])
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # TODO: Move the orbits into the production folder.
    try:
        orbit = pathlib.Path(config.options['brdc'])\
            .with_name(str(start_date.year)) / \
            pathlib.Path('brdc{}0.{}n'.format(start_date.strftime('%j'),
                                              start_date.strftime('%y')))
        shutil.copy(str(orbit), str(prodpath))
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    try:
        # Get the metadata from the RINEX via teqc +meta inputfile
        # Run teqc with the QC flags, no GLONASS (-R), no SBAS (-S), no GALILEO (-E), no BEIDOU (-C) and no QZSS (-J).
        # Use -no_orbit G so that no error is raised about missing nav files for GPS.
        nodelogger.debug('Running TEQC on {}'.format(file.name))
        qcresults = subprocess.run(['teqc',
                                    '+qcq',
                                    '-nav',
                                    orbit.name,
                                    '-R', '-S', '-E', '-C', '-J',
                                    file.name],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   check=True,
                                   encoding='utf-8').stdout.splitlines()
        completion = qcresults[-1].split(' ')
        completion = [x for x in completion if x != '']
        completion = completion[-4]
        # teqc_xyz = [x for x in qcresults if 'antenna WGS 84 (xyz)' in x][0].partition(':')[-1].strip().split(' ')[0:3]
        if float(completion) < 90:
            raise Exception('Less than 90% completion found in {} with {}%'.format(file.name, completion))
        # Check the result of the qc run.
        # Right now we're just going to check the percentage completion and make sure it's above 90%
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # TODO: Move IGS orbits...
    # TODO: Add ability to go through the heirarchy of orbits.
    try:
        gpsepoch = dt.date(1980, 1, 6)
        gpsweek = abs(start_date - gpsepoch).days / 7
        gpsweekday = str(int((gpsweek - int(gpsweek))*7))
        gpsweek = str(int(gpsweek))
        if gpsweekday != '6':
            nextday = str(int(gpsweekday) + 1)
            nextweek = gpsweek
        else:
            nextday = '0'
            nextweek = str(int(gpsweek) + 1)
        sp3path = [pathlib.Path(config.options['sp3']).with_name(gpsweek)
                   / pathlib.Path('igr{}{}.sp3'.format(gpsweek, gpsweekday))][0]
        nextsp3path = [pathlib.Path(config.options['sp3']).with_name(nextweek)
                       / pathlib.Path(f'igr{nextweek}{nextday}.sp3')][0]
        nodelogger.debug('Copying {} to {}'.format(sp3path, prodpath))
        shutil.copy(sp3path, prodpath)
        shutil.copy(sp3path.with_suffix('.clk'), prodpath)
        shutil.copy(nextsp3path, prodpath)
        shutil.copy(nextsp3path.with_suffix('.clk'), prodpath)
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    try:

        nodelogger.debug('Creating files for the PPP run.')
        # TODO: Implement the OTL correction either using grdtab during processing or
        #  http://holt.oso.chalmers.se/loading/hfo.html after the stations are added to the database.
        # Generate the ppp.def file.
        # TODO: Find a newer version of the .svb_gps_yrly PPP file.
        atxfile = pathlib.Path(config.options['atx']).name
        defstring = "'LNG' '{}'\n" \
                    "'TRF' '{}'\n" \
                    "'SVB' '{}'\n" \
                    "'PCV' '{}'\n" \
                    "'FLT' '{}'\n" \
                    "'OLC' '{}'\n" \
                    "'MET' '{}'\n" \
                    "'ERP' '{}'\n" \
                    "'GSD' '{}'\n" \
                    "'GSD' '{}'\n".format('ENGLISH',
                                          f'gpsppp.trf',
                                          f'gpsppp.svb_gps_yrly',
                                          atxfile,
                                          f'gpsppp.flt',
                                          f'gpsppp.olc',
                                          f'gpsppp.met',
                                          f'gpsppp.eop',
                                          'The Ohio State University',
                                          '--')
        with open(f'gpsppp.def', 'w') as f:
            f.write(defstring)
        shutil.copy(os.sep.join([config.options['ppp_path'], 'gpsppp.svb_gnss_yrly']),
                    str(prodpath))
        shutil.copy(config.options['atx'],
                    str(prodpath))
        fltstring = 'FLT    8.1    130    150   5000    300  1     0        2        1        1     0       2'
        with open(f'gpsppp.flt', 'w') as f:
            f.write(fltstring)
        with open(f'gpsppp.olc', 'w') as f:
            f.write('')
        with open(f'gpsppp.met', 'w') as f:
            f.write('')
        with open(f'gpsppp.eop', 'w') as f:
            f.write('')
        cmdfile = f'gpsppp.cmd'
        # TODO: Add command file customization?
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
        with open(cmdfile, 'w') as f:
            f.write(cmdstring)
        ppp_input_string = '{}\n{}\n0 0\n0 0\n{}\n{}\n{}\n{}'.format(file.name,
                                                                     cmdfile,
                                                                     sp3path.name,
                                                                     sp3path.with_suffix('.clk').name,
                                                                     'igr{}{}.sp3'.format(nextweek, nextday),
                                                                     'igr{}{}.clk'.format(nextweek, nextday))
        nodelogger.debug('Done creating PPP files.')
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # TODO: Run PPP on it...
    # TODO: Write PPP in python, it's a nightmare to work with!!!!
    # TODO: The PPP metadata files have to be named gpsppp.ext ....
    try:
        nodelogger.debug('Running PPP.')
        subprocess.run('ppp',
                       input=ppp_input_string,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True,
                       encoding='utf-8')
        nodelogger.debug('Done running PPP')
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # TODO: Parse the .sum file for the coordinates.
    try:
        nodelogger.debug('Parsing {}'.format(file.with_suffix('.sum').name))
        with open(file.with_suffix('.sum').name) as f:
            summary = f.read()
        coordindex = summary.splitlines().index(' 3.3 Coordinate estimates')
        pppcoords = [float(x.split()[3]) for x in summary.splitlines()[coordindex + 3:coordindex + 6]]
        nodelogger.debug('Got coordinates: {}'.format(pppcoords))
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # TODO: Determine if we should lock it (and lock it)...
    try:
        nodelogger.debug('Checking against locked entries.')
        nodelogger.debug('Done checking if it should be locked.')
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    # TODO: Add it to the archive.
    try:
        nodelogger.debug('Checking if it can go into the archive.')
        nodelogger.debug('Done checking if it can go into the archive.')
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    return 'parse_data_in complete: {}'.format(file.name)

# TODO: Make the callback function that helps reduce the amount of submissions and then deals with the output.


def main():
    logger_name = 'archive'
    logger = logging.getLogger(logger_name)
    stream = logging.StreamHandler()

    parser = argparse.ArgumentParser(description='Archive operations Main Program')

    parser.add_argument('-purge', '--purge_locks', action='store_true',
                        help="Delete any network starting with '?' from the stations table and purge the contents of "
                             "the locks table, deleting the associated files from data_in.")

    parser.add_argument('-dup', '--duplicate', type=str,
                        help='Duplicate the archive as it is seen by the database')

    parser.add_argument('-config', '--config_file', type=str, default='gnss_data.cfg',
                        help='Specify the config file, defaults to gnss_data.cfg in the current directory')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable extra messaging by setting the log level to DEBUG')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        stream.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        stream.setLevel(logging.INFO)
    form = logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s - %(threadName)s %(message)s',
                             '%Y-%m-%d %H:%M:%S')
    stream.setFormatter(form)
    logger.addHandler(stream)
    logger.debug('Verbose output enabled')

    logger.debug('Running `{}` with options: {}'.format(logger_name, args))
    config = gpys.ReadOptions(args.config_file)

    # TODO: Implement purge locks
    archive = gpys.RinexArchive(config)
    # TODO: Implement duplicate archive.
    retry_files = archive.scan_archive_struct(config.data_in_retry)
    # Move files from retry into the data_in folder.
    logger.info('Found {} files in {}'.format(len(retry_files), config.data_in_retry))
    try:
        for file in retry_files:
            shutil.move(file.path, '/'.join([config.data_in.as_posix(), file.name]))
        logger.info('Sucessfully moved {} files to {}.'.format(len(retry_files), config.data_in))
    except Exception as e:
        logger.error('Uncaught Exception: {} {}'.format(type(e), e))
        sys.exit(2)
    logger.debug('Filtering out the files that were found in the locks table.')
    try:
        # TODO: Evaluate how to determine the files that are found in the locks and in data_in
        data_in_files = archive.scan_archive_struct(config.data_in)
        logger.debug('Finished parsing the locks table. List of data_in files built.')
    except Exception as e:
        logger.error('Uncaught Exception: {} {}'.format(type(e), e))
        sys.exit(2)
    # TODO: Set up the full jobserver and run parse_data_in.
    # TODO: Need to make sure that all classes sent via a dispy submission are 'picklable'
    # TODO: Loggers aren't picklable
    # Parallel run.
    if config.options['parallel']:
        logger.debug('Parallel run.')
        cluster = None
        job = None
        try:
            jobs = list()
            # TODO: Using a setup function causes the jobcluster to fail?
            cluster = dispy.JobCluster(parse_data_in,
                                       ip_addr=config.options['head_node'],
                                       ping_interval=int(config.options['ping_interval']),
                                       setup=node_setup,
                                       pulse_interval=6,
                                       loglevel=logger.getEffectiveLevel())
            # Dump all jobs onto the cluster and save the DispyJob objects to read later.
            for n, file in enumerate(data_in_files):
                logger.debug('Submitting {}, {}'.format(file.path, archive))
                job = cluster.submit(file.path, config, n)
                jobs.append(job)
            # Wait for the jobs to be submitted.
            cluster.wait()
            logger.debug('Jobs have been submitted.')
            for job in jobs:
                logger.debug('Reading Job {}'.format(job.id))
                if job.status == dispy.DispyJob.Terminated:
                    logger.error('Job was terminated.')
                    raise SystemError(job.exception)
                elif job.status == dispy.DispyJob.Finished:
                    logger.info('Job {} has finished'.format(job.id))
                    logger.debug('STDOUT:\n{}'.format(job.stdout))
                    logger.debug('STDERR:\n{}'.format(job.stderr))
                    logger.debug('Result:\n{}'.format(job.result))
        except SystemError as e:
            logger.error('Broken code: {}'.format(e))
            logger.error('STDERR: {}'.format(job.stderr))
            sys.exit(2)
        except Exception as e:
            logger.error('Uncaught Exception: {} {}'.format(type(e), e))
            sys.exit(2)
        finally:
            cluster.shutdown()
    # Serial run.
    else:
        logger.debug('Serial run.')
        try:
            for n, file in enumerate(data_in_files):
                logger.debug('Running {}, Job #{}'.format(file.path, n))
                result = parse_data_in(file.path, config, n)
                logger.debug('{}'.format(result))
        except Exception as e:
            logger.error('Uncaught Exception: {} {}'.format(type(e), e))
            sys.exit(2)
    logger.info('End of `archive` reached.')
    sys.exit(0)


if __name__ == '__main__':
    main()
