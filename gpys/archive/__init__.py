import gpys
import argparse
import logging
import shutil
import sys
import threading
import dispy
import time

# Globals
logger_name = 'archive'
logger = logging.getLogger(logger_name)
stream = logging.StreamHandler()
lower_bound, upper_bound = 1, 10
jobs_cond = threading.Condition()
pending_jobs = dict()


def node_setup() -> int:
    # TODO: Import the packages required to run parse_data_in
    import gpys
    print('Running node_setup.')
    return 0


def parse_data_in(filename: str, archive, n: int) -> str:
    """
    Runs PPP on a RINEX and either send it to the rejected folder, retry folder, lock it or add it to an existing
    station record.  The archive variable is used to pass what is already in the archive and the file structure.
    :param filename:
    :param archive:
    :param n:
    :return:
    """
    import logging
    import time
    try:
        nodelogger = logging.getLogger('parse_data_in.{}'.format(n))
        nodelogger.setLevel(logging.DEBUG)
        nodestream = logging.StreamHandler()
        nform = logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s - %(threadName)s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
        nodestream.setLevel(logging.DEBUG)
        nodestream.setFormatter(nform)
        nodelogger.addHandler(nodestream)
        # TODO: Add a logger
        # TODO: Check if the RINEX is bad...
        # TODO: Run PPP on it...
        # TODO: Determine if we should lock it (and lock it)...
        # TODO: Add it to the archive.
        time.sleep(1)
        nodelogger.debug('Logger setup.')
    except Exception as e:
        nodelogger.error('Uncaught Exception {} {}'.format(type(e), e))
    return 'complete: {}'.format(filename)


def job_callback(job):
    global pending_jobs, jobs_cond, lower_bound
    cbacklogger = logging.getLogger('archive.job_callback')
    cbacklogger.debug('Running the callback function: job_callback(#{}).'.format(job.id))
    try:
        statdict = {dispy.DispyJob.Finished: 'Finished',
                    dispy.DispyJob.Terminated: 'Terminated',
                    dispy.DispyJob.Cancelled: 'Cancelled',
                    dispy.DispyJob.Abandoned: 'Abandoned'}
        if job.status == dispy.DispyJob.Finished:
            cbacklogger.debug('Job has a status: {}'.format(statdict[job.status]))
            jobs_cond.acquire()
            if job.id:
                cbacklogger.debug('Job #{} has finished, removing from pending job dict.'.format(job.id))
                pending_jobs.pop(job.id)
                cbacklogger.debug('Job #{} STDOUT:\n {}'.format(job.id, job.stdout))
                cbacklogger.debug('Job #{} STDERR:\n {}'.format(job.id, job.stderr))
                cbacklogger.debug('Currently {} jobs running.'.format(len(pending_jobs)))
                if len(pending_jobs) <= lower_bound:
                    cbacklogger.debug('Pending jobs less than the lower bound, releasing lock.')
                    jobs_cond.notify()
            jobs_cond.release()
        elif job.status == dispy.DispyJob.Terminated:
            # TODO: Something is wrong with the code, we should exit the program
            #  at this point so we're not wasting time running broken code.
            cbacklogger.debug('Job status {}'.format(statdict[job.status]))
            raise SystemError('Terminated')
        else:
            cbacklogger.debug('Job status: {}'.format(statdict[job.status]))
    except SystemError as e:
        logger.error(e)
        sys.exit(2)
    except Exception as e:
        logger.error('Uncaught Exception: {} {}'.format(type(e), e))
        sys.exit(2)


def main():
    global pending_jobs, jobs_cond, lower_bound, upper_bound, logger

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
    form = logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s - %(threadName)s %(message)s',
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
        try:
            # TODO: Using a setup function causes the jobcluster to fail.
            cluster = dispy.JobCluster(parse_data_in,
                                       ip_addr=config.options['head_node'],
                                       callback=job_callback,
                                       ping_interval=int(config.options['ping_interval']),
                                       setup=node_setup,
                                       pulse_interval=6,
                                       loglevel=logger.getEffectiveLevel())
            for file in data_in_files:
                logger.debug('Submitting {}, {}'.format(file.path, archive))
                job = cluster.submit(file.path, archive)
                jobs_cond.acquire()
                if job.status == dispy.DispyJob.Created or job.status == dispy.DispyJob.Running:
                    pending_jobs[job.id] = job
                    if len(pending_jobs) >= upper_bound:
                        while len(pending_jobs) > upper_bound:
                            tstart = time.time()
                            logger.debug('Waiting for pending jobs to finish.')
                            jobs_cond.wait()
                            logger.debug('Waited {:.3f} seconds.'.format(time.time()-tstart))
                jobs_cond.release()
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
                logger.debug('Running {}, {}'.format(file.path, archive))
                result = parse_data_in(file.path, archive, n)
                logger.debug('{}'.format(result))
        except Exception as e:
            logger.error('Uncaught Exception: {} {}'.format(type(e), e))
            sys.exit(2)
    logger.info('End of `archive` reached.')
    sys.exit(0)


if __name__ == '__main__':
    main()
