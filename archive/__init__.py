"""
archive/__init__.py

Required directory structure:


Used to add RINEX files to the archive.
TODO: Read parse_data_in output into the postgres table.
TODO: Add else statements to the end of try blocks.
TODO: Sort the data_in list so that it does the days in order.
TODO: Add some error checking to the subprocess runs.
TODO: Delete completed jobs.
TODO: Add percentage completion to the config file?
TODO: Abstract all the file names.
TODO: Hardwire all the paths and then create another class/module/function that checks/installs/updates the executables
      and the path.
"""
import gpys
import argparse
import logging
import shutil
import sys
import dispy
import os
import threading

# Globals
loglevel = None  # Allows for consistent logging levels.
form = logging.Formatter('%(asctime)-15s %(name)-50s %(levelname)-5s:%(lineno)4s %(message)s',
                         '%Y-%m-%d %H:%M:%S')  # Logging format
strform = '%(asctime)-15s %(name)-50s %(levelname)-5s:%(lineno)4s %(message)s'
status = {'Created': dispy.DispyJob.Created,
          'Running': dispy.DispyJob.Running,
          'Terminated': dispy.DispyJob.Terminated,
          'Finished': dispy.DispyJob.Finished,
          'Abandoned': dispy.DispyJob.Abandoned,
          'Cancelled': dispy.DispyJob.Cancelled}  # Used to shorten some if statements.
jobs_cond = threading.Condition()
pending_jobs = dict()
lower_bound, upper_bound = 1350, 2700


def node_setup() -> int:
    """
    Placeholder if in the future using a setup function is deemed to be useful.
    :return:
    """
    return 0


def parse_data_in(filepath: str, config, n: int) -> dict:
    """
    Runs PPP on a RINEX and either send it to the rejected folder, retry folder, lock it or add it to an existing
    station record.  Most parameters from the run are stored in the rinex_dict variable and returned at the end of
    the function.  The function should run through each try block unless: there is a problem importing modules,
    adding global variables or assigning local variables.  Otherwise everything happens within a try block with a
    general exception catch that will just log an ERROR with the exception type and message.
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
    :param config: gpys.ReadOptions object containing parameters from the .cfg file.  Requires these properties:
                   options dict(): working_dir
                                   brdc
                                   sp3
                                   atx
                   sp3types list(str)
    :param n: The job number.
    :return: Dictionary containing the parameters used during the run as well as the results with the following
             keys:
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

    ntimeout = 10
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
    try:
        ofile = Path(filepath)
        nodelogger = logging.getLogger(f'parse_data_in.{n}')
        nodelogger.setLevel(config.nodelevel)
        nodestream = logging.StreamHandler()
        nodestream.setFormatter(config.nodeform)
        nodestream.setLevel(config.nodelevel)
        nodelogger.addHandler(nodestream)
        nodelogger.debug(f'Logger setup on {socket.gethostname()}.')
        rinex_dict['ofile'] = ofile
    except Exception as e:
        print(f'Exception raised during logger setup {type(e)} {e}', file=sys.stderr)
        complete = False
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
            [x for x in qcresults if 'Antenna type' in x][0].partition(':')[2].strip().partition('(')[2].partition(
                ')')[
                0].split()[2]
        observationfyear = float(start_date.year) + float(start_date.strftime('%j')) / \
                           float(dt.date(start_date.year, 12, 31).strftime('%j'))

        rinex_dict['ObservationFYear'] = Decimal(observationfyear).quantize(Decimal('0.001'))
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
    try:
        if not complete:
            raise Exception('Skipping PPP as previous blocks have failed.')
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
    try:
        if complete:
            nodelogger.debug(f'No errors found, deleting {prodpath}')
            shutil.rmtree(prodpath.as_posix())
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
        complete = False
    rinex_dict['completed'] = complete
    program_endtime = dt.datetime.now()
    rinex_dict['runtime'] = (program_endtime - program_starttime).total_seconds()
    return rinex_dict


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
        nodelogger.setLevel(config.nodelevel)
        nodestream = logging.StreamHandler()
        nodestream.setFormatter(config.nodeform)
        nodestream.setLevel(config.nodelevel)
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


def callback(job):
    """
    Simple callback function that helps reduce the amount of submissions.  Typing hints don't work here.
    :param job: An instance of dispy.DispyJob
    :return:
    """
    if job.status in [status['Finished'],
                      status['Terminated'],
                      status['Abandoned'],
                      status['Cancelled']]:
        jobs_cond.acquire()
        if job.id:
            pending_jobs.pop(job.id)
            if len(pending_jobs) <= lower_bound:
                jobs_cond.notify()
            jobs_cond.release()


def gamit(gamit_dict, config, n):
    """
    Run GAMIT
    :return:

    Runs GAMIT for a set of stations on a given day.
    """
    import logging
    import socket
    from pathlib import Path
    nodelogger = None  # Logger
    """Set_Logger-------------------------------------------------------------------------------------------------------
    Set up the logger internal to parse_data_in in order to make debugging any problems much easier.  Also convert the 
    string filepath into a pathlib.Path object.

    Previous errors:
    None (yay!)
    """
    try:
        nodelogger = logging.getLogger(f"gamit")
        nodelogger.setLevel(config.nodelevel)
        nodestream = logging.StreamHandler()
        nodestream.setFormatter(config.nodeform)
        nodestream.setLevel(config.nodelevel)
        nodelogger.addHandler(nodestream)
        nodelogger.debug(f'Logger setup on {socket.gethostname()}.')
    except Exception as e:
        print(f'Exception raised during logger setup {type(e)} {e}', file=sys.stderr)
    try:
        prodpath = Path(f'production/job{n}')
        prodpath = os.sep.join([config.options['working_dir'], str(prodpath)])
        prodpath = Path(prodpath)
        os.makedirs(str(prodpath), exist_ok=True)
        os.chdir(str(prodpath))
    except PermissionError as e:
        nodelogger.error(f'Permission error: {type(e)} {e}')
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')


def main() -> None:
    """
    Parse the commandline arguments
    Load the options file into an object.
    Move files from retry into the data_in folder.
    Submit jobs to the cluster.
    First submit using cluster.submit, then add it to [jobs] so we can use the results later.  Once that is
    done we lock the MainThread using jobs_cond.acquire()
    Wait for the jobs to be submitted.

    TODO: Need to make sure that all classes sent via a dispy submission are 'picklable'
    TODO: Loggers aren't picklable
    TODO: Implement purge locks
    TODO: Implement duplicate archive.
    TODO: Evaluate how to determine the files that are found in the locks and in data_in
    TODO: Add data to the PostgreSQL database.
    TODO: Atomize the submission process.
    :return:
    """
    # Add the globals.
    global loglevel, strform
    try:
        parser = argparse.ArgumentParser(description='Archive operations Main Program')
        parser.add_argument('-purge', '--purge_locks', action='store_true',
                            help="Delete any network starting with '?' from the stations table and purge the contents "
                                 "of the locks table, deleting the associated files from data_in.")
        parser.add_argument('-dup', '--duplicate', type=str,
                            help='Duplicate the archive as it is seen by the database')

        parser.add_argument('-config', '--config_file', type=str, default='gnss_data.cfg',
                            help='Specify the config file, defaults to gnss_data.cfg in the current directory')

        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Enable extra messaging by setting the log level to DEBUG')
        args = parser.parse_args()
    except Exception as e:
        print(f'Incorrect command line arguments {type(e)} {e}', file=sys.stderr)
    try:
        logger_name = 'archive'
        logger = logging.getLogger(logger_name)
        stream = logging.StreamHandler()
        if args.verbose:
            loglevel = logging.DEBUG
            logger.setLevel(loglevel)
            stream.setLevel(loglevel)
        else:
            loglevel = logging.INFO
            logger.setLevel(loglevel)
            stream.setLevel(loglevel)
        stream.setFormatter(form)
        logger.addHandler(stream)
        logger.debug('Verbose output enabled')
        logger.debug(f'Running `{logger_name}` with options: {args}')
    except Exception as e:
        print(f'Logger failed to set up: {type(e)} {e}', file=sys.stderr)
    """Clear locks table------------------------------------------------------------------------------------------------
    Remove any files in the locks table that have a network code that doesn't equal ???
    """
    try:
        logger.debug('Reading the config file and archive layout.')
        config = gpys.ReadOptions(args.config_file)
        cluster_options = {'ip_addr': config.options['head_node'],
                           'ping_interval': int(config.options['ping_interval']),
                           'setup': node_setup,
                           'callback': callback,
                           'pulse_interval': 6,
                           'loglevel': 60}
        retry_files = config.scan_archive_struct(config.data_in_retry)
        logger.info(f'Found {len(retry_files)} files in {config.data_in_retry}')
        logger.debug('Clearing locks table')
        update_locks = gpys.Connection(config.options)
        update_locks.update_locks()
        locked_files = update_locks.load_table('locks', ['filename'])
        logger.debug(f"Found locked files: {locked_files}")
        del update_locks
    except Exception as e:
        logger.error(f'Uncaught Exception {type(e)} {e}')
    """Parse_data_in_retry----------------------------------------------------------------------------------------------
    """
    try:
        for file in retry_files:
            shutil.move(file.path, os.sep.join([config.data_in.as_posix(), file.name]))
        logger.info(f'Sucessfully moved {len(retry_files)} files to {config.data_in}.')
    except Exception as e:
        logger.error(f'Uncaught Exception {type(e)} {e}')
    """Parse_data_in----------------------------------------------------------------------------------------------
    """
    logger.debug('Filtering out the files that were found in the locks table.')
    try:
        data_in_files = config.scan_archive_struct(config.data_in)
        cleared_files = list()
        if locked_files:
            for f in data_in_files:
                if f.name not in locked_files['filename']:
                    logger.debug(f"{f.name} not found in locked files, adding to cleared files.")
                    cleared_files.append(f)
        else:
            cleared_files = data_in_files
        logger.debug(f'Cleared out {len(data_in_files)-len(cleared_files)} files.')
        logger.info(f'Found {len(cleared_files)} file(s)')
    except Exception as e:
        logger.error(f'Uncaught Exception {type(e)} {e}')
    if config.options['parallel']:
        """Parallel run-------------------------------------------------------------------------------------------------
        
        Previous errors:
        Using a setup function causes the jobcluster to fail sometimes, not sure what the cause is.
        To remove any logging by  dispy, set the loglevel >50 as seen at line 125 in the pycos source code.
        """
        try:
            jobs = list()
            runtimes = list()
            processed_rinex = list()
            parse_data_cluster = dispy.JobCluster(parse_data_in, **cluster_options)
            for n, file in enumerate(cleared_files):
                logger.debug(f'Submitting job#{n}: {file.as_posix()}')
                job = parse_data_cluster.submit(file.as_posix(), config, n)
                jobs.append(job)
                jobs_cond.acquire()
                if job.status in [status['Created'], status['Running']]:
                    pending_jobs[job.id] = job
                    if len(pending_jobs) >= upper_bound:
                        logger.debug(f'More pending_jobs than {upper_bound}, '
                                         f'waiting for the next jobs_cond.notify.')
                        while len(pending_jobs) > lower_bound:
                            jobs_cond.wait()
                        logger.debug(f'')
                jobs_cond.release()
            logger.debug('Waiting for jobs to finish up submission.')
            parse_data_cluster.wait()
            logger.info(f'{len(jobs)} jobs have been submitted.')
            for job in jobs:
                logger.debug(f'Reading Job {job.id}')
                if job.status in [status['Terminated']]:
                    logger.error('Job was terminated.')
                    logger.error(f'STDOUT:\n{job.stdout}')
                    logger.error(f'STDERR:\n{job.stderr}')
                    logger.error(f'Result:\n{job.result}')
                    logger.error(job.exception)
                elif job.status in [status['Finished']]:
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
        except Exception as e:
            logger.error(f'Uncaught Exception {type(e)} {e}')
    logger.info('End of `archive` reached.')
    sys.exit(0)


if __name__ == '__main__':
    main()
