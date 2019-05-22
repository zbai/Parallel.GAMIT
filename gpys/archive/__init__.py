import gpys
import argparse
import logging
import shutil
import sys
import dispy
import time

logger_name = 'archive'
logger = logging.getLogger(logger_name)
stream = logging.StreamHandler()


def node_setup() -> int:
    # TODO: Import the packages required to run parse_data_in
    return 2


def parse_data_in(filename: str) -> str:
    # TODO: Add a logger
    return 'complete: {}'.format(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Archive operations Main Program')

    parser.add_argument('-purge', '--purge_locks', action='store_true',
                        help="Delete any network starting with '?' from the stations table and purge the contents of "
                             "the locks table, deleting the associated files from data_in.")

    parser.add_argument('-dup', '--duplicate', type=str,
                        help='Duplicate the archive as it is seen by the database')

    parser.add_argument('-config', '--config_file', type=str, default='gnss_data.cfg',
                        help='Specify the config file, defaults to gnss_data.cfg in the current directory')

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable extra messaging.')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        stream.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        stream.setLevel(logging.INFO)
    form = logging.Formatter('%(asctime)-15s - %(name)-25s %(levelname)s - %(message)s',
                             '%Y-%m-%d %H:%M:%S')
    stream.setFormatter(form)
    logger.addHandler(stream)
    logger.debug('Verbose output enabled')

    logger.debug('Running `{}` with options: {}'.format(logger_name, args))
    config = gpys.ReadOptions(args.config_file)

    # TODO: Implement purge locks
    archive = gpys.RinexArchive(config)
    # TODO: Implement duplicate the archive.
    retry_files = archive.scan_archive_struct(config.data_in_retry)
    # Move files from retry into the data_in folder.
    logger.info('Found {} files in {}'.format(len(retry_files), config.data_in_retry))
    try:
        for file in retry_files:
            shutil.move(file.path, '/'.join([config.data_in.as_posix(), file.name]))
        logger.info('Sucessfully moved {} files to {}.'.format(len(retry_files), config.data_in))
    except Exception as e:
        logger.error('Uncaught exception: {}'.format(e))
        sys.exit(2)
    logger.debug('Filtering out the files that were found in the locks table.')
    try:
        locked_files = config.conn.load_table('locks')
        # TODO: Evaluate how to determine the files that are found in the locks and in data_in
        data_in_files = archive.scan_archive_struct(config.data_in)
        logger.debug('Finished parsing the locks table. List of data_in files built.')
    except Exception as e:
        logger.error('Uncaught exception: {}'.format(e))
        sys.exit(2)
    # TODO: Set up the full jobserver and run parse_data_in.

    cluster = gpys.JobServer(config.options)
    cluster.connect(parse_data_in)
    logger.info('End of `archive` reached.')
    sys.exit(0)
