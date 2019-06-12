"""
archive/__init__.py

Required directory structure:


Used to add RINEX files to the archive.
TODO: Read parse_data_in output into the postgres table.
TODO: Add more docstrings around the try blocks.
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
import threading
import os


# Globals
lower_bound, upper_bound = 50, 100                  # Lower and upper limits to the number of pending jobs.
jobs_cond = threading.Condition()                   # The condition lock used to limit the number of jobs submitted.
pending_jobs = dict()                               # Contains running jobs.
status = {'Created': dispy.DispyJob.Created,
          'Running': dispy.DispyJob.Running,
          'Terminated': dispy.DispyJob.Terminated,
          'Finished': dispy.DispyJob.Finished,
          'Abandoned': dispy.DispyJob.Abandoned,
          'Cancelled': dispy.DispyJob.Cancelled}    # Used to shorten some if statements.
loglevel = None                                     # Allows for consistent logging levels.


def node_setup() -> int:
    """
    Placeholder if in the future using a setup function is deemed to be useful.
    :return:
    """
    return 0


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
    TODO: Figure out what kind of command we can use here to quit the function on an exception.
    TODO: Implement the OTL correction either using grdtab during processing or
          http://holt.oso.chalmers.se/loading/hfo.html after the stations are added to the database.
    TODO: Find a newer version of the .svb_gps_yrly PPP file.
    TODO: Add command (gpsppp.cmd) file customization?
    TODO: Write PPP in python, it's a nightmare to work with!!!!
    :param filepath: Path to the RINEX we're working on.
    :param config: gpys.ReadOptions object containing parameters from the .cfg file.
    :param n: The job number.
    :return: Dictionary containing the parameters used during the run as well as the results.
    """
    import logging
    import socket
    import os
    from pathlib import Path
    import datetime as dt
    import shutil
    import subprocess
    import sys

    def fileopts(orig_file: Path) -> Path:
        """
        Remove the .Z suffix
        Change the internal path suffix to o from d
        Raise exception if the last letter in the extention doesn't match [sp]3, [cl]k, [##]n, [##]o
        :param orig_file:
        :return:
        """
        if orig_file.suffix in '.Z':
            nodelogger.debug(f'Decompressing {orig_file.name}')
            subprocess.run(['uncompress', '-f', orig_file.name])
            orig_file = orig_file.with_suffix('')
        if orig_file.match('*.??d'):
            subprocess.run(['crx2rnx', '-f', orig_file])
            os.remove(orig_file.as_posix())
            orig_file = orig_file.with_suffix(orig_file.suffix.replace('d', 'o'))
        if orig_file.suffix[-1] not in ['3', 'k', 'n', 'o']:
            raise Exception(f'Unrecognized file extension: {orig_file.suffix}')
        return orig_file

    # Globals
    global loglevel
    # Internal variable set up and definition for later reference.
    nodelogger = None               # Internal logger for the function.
    ofile = None                    # Original file converted to a pathlib.Path object.
    file = None                     # pathlib.Path object for the RINEX we're working on, updated as it is operated on.
    prodpath = None                 # pathlib.Path object defining where the production folder is.
    start_date = None               # datetime.date object for the RINEX as determined by TEQC
    orbit = None                    # pathlib.Path object with the archive path to the BRDC orbit file.
    sp3path = None                  # Path to the IGS sp3, also used for the clk files
    nextsp3path = None              # Path to next day's sp3 file.
    ppp_input_string = 'BadFilename'  # The STDIN input to the PPP command.
    rinex_dict = dict()             # Where all the info about the run is stored.
    tanktype = {'<year>': None,
                '<month>': None,
                '<day>': None,
                '<gpsweek>': None}  # Helps define the path to the brdc or igs tanks.
    gpsweek = None                  # Normally just a string
    gpsweekday = None               # Just a string.
    nextweek = None                 # The week of the next day's orbits
    nextday = None                  # Day of the next orbit
    igsdir = None                   # Shorthand for the archive location for igs orbits
    brdcdir = None                  # Shorthand for brdc location
    clkpath = None                  # Same but for the clk files
    nextclkpath = None              # Ditto, next day though.
    pppref = None                   # The reference frame that PPP uses, sometimes it swaps between IGS14 and IGb08
    mincompletion = 90              # Integer representing the percentage of completion (# of obs/total possible obs)
    """Set_Logger-------------------------------------------------------------------------------------------------------
    Set up the logger internal to parse_data_in in order to make debugging any problems much easier.  Also convert the 
    string filepath into a pathlib.Path.
    
    Previous errors:
    None (yay!)
    """
    try:
        ofile = Path(filepath)
        nodelogger = logging.getLogger(f'parse_data_in.{n}')
        nodelogger.setLevel(loglevel)
        nodestream = logging.StreamHandler()
        nodestream.setFormatter(logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s -'
                                                  ' %(threadName)s %(message)s',
                                                  '%Y-%m-%d %H:%M:%S'))
        nodestream.setLevel(loglevel)
        nodelogger.addHandler(nodestream)
        nodelogger.debug(f'Logger setup on {socket.gethostname()}.')
        rinex_dict['ofile'] = ofile
    except Exception as e:
        print(f'Exception raised during logger setup {type(e)} {e}', file=sys.stderr)
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
        file = fileopts(file)
        nodelogger.debug(f'Working with {file.name}')
        nodelogger.debug('Created paths to dependencies.')
        teqc = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'teqc']))
        ppp = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'ppp']))
        atxfile = Path(os.sep.join([config.options['working_dir'], 'dependencies', config.options['atx']]))
        trffile = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'gpsppp.trf']))
        svbfile = Path(os.sep.join([config.options['working_dir'], 'dependencies', 'gpsppp.svb_gnss_yrly']))
        nodelogger.debug(f'Using {teqc}, {atxfile}, {ppp}')
        for f in [atxfile, trffile, svbfile]:
            shutil.copy(str(f), str(prodpath))
        rinex_dict['file'] = file
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
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
                                  check=True)
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
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
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
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
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
                                   encoding='utf-8').stdout.splitlines()
        completion = qcresults[-1].split(' ')
        completion = [x for x in completion if x != '']
        completion = completion[-4]
        teqc_xyz = [x for x in qcresults if 'antenna WGS 84 (xyz)' in x][0].partition(':')[-1].strip().split(' ')[0:3]
        rinex_dict['completion'] = completion
        rinex_dict['teqc_xyz'] = teqc_xyz
    except AttributeError as e:
        nodelogger.error(f'A variable was incorrectly assigned, did a previous block fail?: {type(e)} {e}')
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
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
                sp3path = fileopts(sp3path)
                clkpath = fileopts(clkpath)
                nextsp3path = fileopts(nextsp3path)
                nextclkpath = fileopts(nextclkpath)
                break
            elif sp3type == config.sp3types[-1]:
                raise Exception("Didn't find any valid orbits.")
        rinex_dict['sp3path'] = sp3path
        rinex_dict['nextsp3path'] = nextsp3path
    except TypeError as e:
        nodelogger.error(f'A variable was incorrectly assigned, did a previous block fail?: {type(e)} {e}')
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
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
                       encoding='utf-8')
        nodelogger.debug('Done running PPP')
    except FileNotFoundError as e:
        nodelogger.error(f'Make sure that PPP is on system path and executable: {type(e)} {e}')
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
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
            coordindex = [summary.splitlines().index(' 3.3 Coordinate estimates'),
                          summary.splitlines().index(' 3.4 Coordinate differences ITRF (IGS14)')]
            pppref = 'IGS14'
    except ValueError as e:
        if str(e) == "' 3.4 Coordinate differences ITRF (IGS14)' is not in list":
            nodelogger.debug('Using IGb08')
            coordindex = [summary.splitlines().index(' 3.3 Coordinate estimates'),
                          summary.splitlines().index(' 3.4 Coordinate differences ITRF (IGb08)')]
            pppref = 'IGb08'
        else:
            nodelogger.error(f'Uncaught ValueError {type(e)} {e}')
    except FileNotFoundError as e:
        nodelogger.error(f"Couldn't find the .sum file, PPP probably failed: {type(e)} {e}")
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
    """Parse .sum into variables----------------------------------------------------------------------------------------
    """
    try:
        resultblock = summary.splitlines()[coordindex[0]:coordindex[1] - 1]
        cartind = resultblock.index(f' CARTESIAN           NAD83(CSRS )     ITRF ({pppref})   Sigma(m) NAD-ITR(m)')
        pppcoords = [float(x.split()[3]) for x in resultblock[cartind + 1:cartind + 4]]
        nodelogger.debug(f'Got coordinates: {pppcoords}')
        rinex_dict['pppcoords'] = pppcoords
        rinex_dict['pppref'] = pppref
    except Exception as e:
        nodelogger.error(f'Uncaught Exception {type(e)} {e}')
    '------------------------------------------------------------------------------------------------------------------'
    return rinex_dict


def callback(job):
    """
    Simple callback function that helps reduce the amount of submissions.  Typing hints don't work here.
    :param job: An instance of dispy.DispyJob
    :return:
    """
    # Add the globals.
    global pending_jobs, jobs_cond, lower_bound, upper_bound, status
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


def database_ops(rinex_dict: dict = None):
    """
    Either locks the file or adds it to an existing record.
    :return:
    """
    import logging
    import socket

    global loglevel
    """Set_Logger-------------------------------------------------------------------------------------------------------
    Set up the logger internal to parse_data_in in order to make debugging any problems much easier.  Also convert the 
    string filepath into a pathlib.Path.

    Previous errors:
    None (yay!)
    """
    try:
        nodelogger = logging.getLogger(f"database_ops.{rinex_dict['name']}-{rinex_dict['start_date']}")
        nodelogger.setLevel(loglevel)
        nodestream = logging.StreamHandler()
        nodestream.setFormatter(logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s -'
                                                  ' %(threadName)s %(message)s',
                                                  '%Y-%m-%d %H:%M:%S'))
        nodestream.setLevel(loglevel)
        nodelogger.addHandler(nodestream)
        nodelogger.debug(f'Logger setup on {socket.gethostname()}.')
    except Exception as e:
        print(f'Exception raised during logger setup {type(e)} {e}', file=sys.stderr)
    """Check the locks table--------------------------------------------------------------------------------------------
    """
    try:
        nodelogger.debug('Checking the locks table.')
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
    global pending_jobs, jobs_cond, lower_bound, upper_bound, status, loglevel
    # Local vars.
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
        sys.exit(2)
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
        form = logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s - %(threadName)s %(message)s',
                                 '%Y-%m-%d %H:%M:%S')
        stream.setFormatter(form)
        logger.addHandler(stream)
        logger.debug('Verbose output enabled')

        logger.debug(f'Running `{logger_name}` with options: {args}')
        config = gpys.ReadOptions(args.config_file)

        archive = gpys.RinexArchive(config)
        retry_files = archive.scan_archive_struct(config.data_in_retry)
        logger.info(f'Found {len(retry_files)} files in {config.data_in_retry}')
    except Exception as e:
        print(f'Logger failed to set up: {type(e)} {e}', file=sys.stderr)
        sys.exit(2)
    """Parse_data_in_retry----------------------------------------------------------------------------------------------
    """
    try:
        for file in retry_files:
            shutil.move(file.path, os.sep.join([config.data_in.as_posix(), file.name]))
        logger.info(f'Sucessfully moved {len(retry_files)} files to {config.data_in}.')
    except Exception as e:
        logger.error(f'Uncaught Exception {type(e)} {e}')
        sys.exit(2)
    """Parse_data_in----------------------------------------------------------------------------------------------
    """
    logger.debug('Filtering out the files that were found in the locks table.')
    try:
        data_in_files = archive.scan_archive_struct(config.data_in)
        logger.debug('Finished parsing the locks table. List of data_in files built.')
        logger.info(f'Found {len(data_in_files)} file(s)')
    except Exception as e:
        logger.error(f'Uncaught Exception {type(e)} {e}')
        sys.exit(2)
    if config.options['parallel']:
        """Parallel run-------------------------------------------------------------------------------------------------
        
        Previous errors:
        Using a setup function causes the jobcluster to fail sometimes, not sure what the cause is.
        """
        logger.debug('Parallel run.')
        cluster = None
        job = None
        try:
            jobs = list()
            cluster = dispy.JobCluster(parse_data_in,
                                       ip_addr=config.options['head_node'],
                                       ping_interval=int(config.options['ping_interval']),
                                       setup=node_setup,
                                       callback=callback,
                                       pulse_interval=6,
                                       loglevel=logger.getEffectiveLevel())
            for n, file in enumerate(data_in_files):
                logger.debug('Submitting {}, {}'.format(file.path, archive))
                job = cluster.submit(file.path, config, n)
                jobs.append(job)
                jobs_cond.acquire()
                logger.debug(f'Locking thread {threading.get_ident()}...')
                if job.status in [status['Created'], status['Running']]:
                    pending_jobs[job.id] = job
                    if len(pending_jobs) >= upper_bound:
                        while len(pending_jobs) > lower_bound:
                            logger.debug(f'More pending_jobs than {lower_bound}, '
                                         f'waiting for the next jobs_cond.notify.')
                            jobs_cond.wait()
                jobs_cond.release()
            logger.debug('Waiting for jobs to finish up submission.')
            cluster.wait()
            logger.debug(f'{len(jobs)} jobs have been submitted.')
            for job in jobs:
                logger.debug(f'Reading Job {job.id}')
                if job.status in [status['Terminated']]:
                    logger.error('Job was terminated.')
                    raise SystemError(job.exception)
                elif job.status in [status['Finished']]:
                    logger.info(f'Job {job.id} has finished')
                    logger.debug(f'STDOUT:\n{job.stdout}')
                    logger.debug(f'STDERR:\n{job.stderr}')
                    logger.debug(f'Result:\n{job.result}')
        except SystemError as e:
            logger.error(f'Broken code: {e}')
            logger.error(f'STDERR: {job.stderr}')
            sys.exit(2)
        except Exception as e:
            logger.error(f'Uncaught Exception {type(e)} {e}')
            sys.exit(2)
        finally:
            cluster.shutdown()
    else:
        """Serial Run---------------------------------------------------------------------------------------------------
        
        Previous errors:
        None
        """
        logger.debug('Serial run.')
        try:
            for n, file in enumerate(data_in_files):
                logger.debug(f'Running {file.path}, Job #{n}')
                result = parse_data_in(file.path, config, n)
                logger.debug(f'{result}')
        except Exception as e:
            logger.error(f'Uncaught Exception {type(e)} {e}')
            sys.exit(2)
    """Add to database--------------------------------------------------------------------------------------------------
    Similar structure to the above PPP section, (serial vs parallel)
    
    Previous Errors
    """
    """Serial Run---------------------------------------------------------------------------------------------------

    Previous errors:
    None
    """
    try:
        logger.debug('Serial database_ops  run.')
    except Exception as e:
        logger.error(f'Uncaught Exception {type(e)} {e}')
        sys.exit(2)
    logger.info('End of `archive` reached.')
    sys.exit(0)


if __name__ == '__main__':
    main()
