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

# Globals
loglevel = None  # Allows for consistent logging levels.
form = logging.Formatter('%(asctime)-15s %(name)-50s %(levelname)-5s:%(lineno)4s %(message)s',
                         '%Y-%m-%d %H:%M:%S')  # Logging format
strform = '%(asctime)-15s %(name)-50s %(levelname)-5s:%(lineno)4s %(message)s'


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
            pppserver = gpys.Distribute(config)
            processed_rinex = pppserver.dist(cleared_files, pppserver.parse_data_in)
            pppserver.dist(processed_rinex, pppserver.database_ops)
        except Exception as e:
            logger.error(f'Uncaught Exception {type(e)} {e}')
    logger.info('End of `archive` reached.')
    sys.exit(0)


if __name__ == '__main__':
    main()
