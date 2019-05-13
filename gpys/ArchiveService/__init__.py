"""
Project: Parallel.Archive
Date: 3/19/17 11:41 AM
Author: Demian D. Gomez

Main script that scans the repository for new rinex files.
It PPPs the rinex files and searches the database for stations (within 100 m) with the same station 4 letter code.
If the station exists in the db, it moves the file to the archive and adds the new file to the "rinex" table.
if the station doesn't exist, then it incorporates the station with a special NetworkCode (???) and leaves the
file in the repo until you assign the correct NetworkCode and add the station information.

It is invoked just by calling python ArchiveService.py
Requires the config file gnss_data.cfg (in the running folder)

TODO: Move just about everything into the gpys init file.
"""

import argparse
import datetime as dt
import os
import platform
import shutil
import sys
import traceback
import uuid

from tqdm import tqdm

import gpys


def insert_station_w_lock(cnn, stationcode, filename, lat, lon, h, x, y, z, otl):

    similar_stations = cnn.similar_locked([lat, lat, lon, stationcode])
    if len(similar_stations) != 0:
        network_code = similar_stations[0][0]
        cnn.update('locks', {'filename': filename}, NetworkCode=network_code, StationCode=stationcode)
    else:
        # insert this new record in the stations table using a default network name (???)
        # this network name is a flag that tells ArchiveService that no data should be added to this station
        # until a NetworkCode is assigned.

        # check if network code exists
        network_code = '???'
        index = 0
        while len(cnn.load_table_matching('stations', {'NetworkCode': network_code, 'StationCode': stationcode})) != 0:
            network_code = hex(index).replace('0x', '').rjust(3, '?')
            index += 1
            if index > 255:
                # FATAL ERROR! the networkCode exceed FF
                raise Exception('While looking for a temporary network code, ?ff was reached! '
                                'Cannot continue executing pyArchiveService. Please free some temporary network codes.')

        rs = cnn.load_table_matching('networks', {'NetworkCode': network_code})

        if len(rs) == 0:
            # create network code
            cnn.insert('networks', NetworkCode=network_code, NetworkName='Temporary network for new stations')

        # insert record in stations with temporary NetworkCode
        try:
            cnn.insert('stations', {'NetworkCode': network_code,
                                    'StationCode': stationcode,
                                    'auto_x': x,
                                    'auto_y': y,
                                    'auto_z': z,
                                    'Harpos_coeff_otl': otl,
                                    'lat': round(lat, 8),
                                    'lon': round(lon, 8),
                                    'height': round(h, 3)})
        except gpys.DBErrInsert:
            # another process did the insert before, ignore the error
            pass

        # update the lock information for this station
        cnn.update('locks', {'filename': filename}, {'NetworkCode': network_code, 'StationCode': stationcode})


def check_rinex_timespan_int(rinex, station):
    # how many seconds difference between the rinex file and the record in the db
    stime_diff = abs((station['ObservationSTime'] - rinex.datetime_firstObs).total_seconds())
    etime_diff = abs((station['ObservationETime'] - rinex.datetime_lastObs).total_seconds())

    # at least four minutes different on each side
    if stime_diff <= 240 and etime_diff <= 240 and station['Interval'] == rinex.interval:
        return False
    else:
        return True


def write_error(folder, filename, msg):
    # do append just in case...
    count = 0
    while True:
        try:
            with open(os.path.join(folder, filename), 'a') as file:
                file.write(msg)
            break
        except IOError as e:
            if count < 3:
                count += 1
            else:
                raise IOError(str(e) + ' after 3 retries')
            continue

    return


def error_handle(cnn, event, crinez, folder, filename, db_log=True):
    # move_new to the folder indicated
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        # racing condition of two processes trying to create the same folder
        pass

    message = event['Description']

    mfile = filename
    try:
        mfile = os.path.basename(shutil.move(crinez, os.path.join(folder, filename)))
    except (OSError, ValueError) as e:
        message = 'could not move_new file into this folder!' + str(e) + '\n. Original error: ' + event['Description']

    error_file = mfile.replace('d.Z', '.log')
    write_error(folder, error_file, message)

    if db_log:
        cnn.insert_event(event)


def insert_data(cnn, archix, rinexinfo):
    inserted = archix.insert_rinex(rinexobj=rinexinfo)
    # if archive.insert_rinex has a dbInserErr, it will be catched by the calling function
    # always remove original file
    os.remove(rinexinfo.origin_file)

    if not inserted:
        # insert an event to account for the file (otherwise is weird to have a missing rinex in the events table
        event = gpys.Event(
            Description=rinexinfo.crinez + ' had the same interval and completion as an existing file. '
                                           'CRINEZ deleted from data_in.',
            NetworkCode=rinexinfo.NetworkCode,
            StationCode=rinexinfo.StationCode,
            Year=int(rinexinfo.date.year),
            DOY=int(rinexinfo.date.doy))

        cnn.insert_event(event)


def verify_rinex_multiday(cnn, rinexinfo, cfg):
    # function to verify if rinex is multiday
    # returns true if parent process can continue with insert
    # returns false if file had to be moved to the retry

    # check if rinex is a multiday file (rinex with more than one day of observations)
    if rinexinfo.multiday:

        # move_new all the files to the repository
        rnxlist = []
        for rnx in rinexinfo.multiday_rnx_list:
            rnxlist.append(rnx.rinex)
            # some other file, move_new it to the repository
            retry_folder = os.path.join(cfg.repository_data_in_retry,
                                        'multidays_found/' + rnx.date.yyyy() + '/' + rnx.date.ddd())
            rnx.compress_local_copyto(retry_folder)

        # if the file corresponding to this session is found, assign its object to rinexinfo
        event = gpys.Event(
            Description='%s was a multi-day rinex file. The following rinex files where generated '
                        'and moved to the repository/data_in_retry: %s. The file %s did not enter '
                        'the database at this time.' %
                        (rinexinfo.origin_file, ','.join(rnxlist), rinexinfo.crinez),
            NetworkCode=rinexinfo.NetworkCode,
            StationCode=rinexinfo.StationCode,
            Year=int(rinexinfo.date.year),
            DOY=int(rinexinfo.date.doy))

        cnn.insert_event(event)

        # remove crinez from the repository (origin_file points to the repository, not to the archive in this case)
        os.remove(rinexinfo.origin_file)

        return False

    return True


def process_crinex_file(crinez, filename, data_rejected, data_retry, cfg_file='gnss_data.cfg'):
    """
    :param cfg_file:
    :param crinez:
    :param filename:
    :param data_rejected:
    :param data_retry:
    :return:
    """

    # create a uuid temporary folder in case we cannot read the year and doy from the file (and gets rejected)
    reject_folder = os.path.join(data_rejected, str(uuid.uuid4()))

    try:
        cnn = gpys.Connection(cfg_file)
        cfg = gpys.ReadOptions(cfg_file)
        archix = gpys.RinexStruct(cnn, cfg_file=cfg_file)
        # apply local configuration (path to repo) in the executing node
        crinez = os.path.join(cfg.repository_data_in, crinez)

    except Exception:

        return traceback.format_exc() + ' while opening the database to process file ' + \
               crinez + ' node ' + platform.node(), None

    # assume a default networkcode
    network_code = 'rnx'
    # get the station code year and doy from the filename
    fileparts = archix.parse_crinex_filename(filename)

    if fileparts:
        station_code = fileparts[0].lower()
        doy = int(fileparts[1])
        year = int(gpys.get_norm_year_str(fileparts[3]))
    else:
        event = gpys.Event(
            Description='Could not read the station code, year or doy for file ' + crinez,
            EventType='error')
        error_handle(cnn, event, crinez, reject_folder, filename, db_log=True)
        return event['Description'], None

    # we can now make better reject and retry folders
    # TODO: We can probably clean up this string building a bit.
    reject_folder = os.path.join(data_rejected,
                                 '%reason%/' + gpys.get_norm_year_str(
                                     year) + '/' + gpys.get_norm_doy_str(doy))
    # TODO: We can probably clean up this string building a bit.
    retry_folder = os.path.join(data_retry,
                                '%reason%/' + gpys.get_norm_year_str(
                                    year) + '/' + gpys.get_norm_doy_str(doy))

    try:
        # main try except block
        with gpys.ReadRinex(network_code, station_code, crinez) as rinexinfo:  # type: gpys.ReadRinex

            # STOP! see if rinexinfo is a multiday rinex file
            if not verify_rinex_multiday(cnn, rinexinfo, cfg):  # pragma: no cover
                # was a multiday rinex. verify_rinex_date_multiday took care of it
                return None, None

            # DDG: we don't use otl coefficients because we need an approximated coordinate
            # we therefore just calculate the first coordinate without otl
            # NOTICE that we have to trust the information coming in the RINEX header (receiver type, antenna type, etc)
            # we don't have station info data! Still, good enough
            # the final PPP coordinate will be calculated by pyScanArchive on a different process

            # make sure that the file has the appropriate coordinates in the header for PPP.
            # put the correct APR coordinates in the header.
            # ppp didn't work, try using sh_rx2apr
            brdc = gpys.GetBrdcOrbits(cfg.options['brdc'], rinexinfo.date, rinexinfo.rootdir)

            # inflate the chi**2 limit to make sure it will pass (even if we get a crappy coordinate)
            try:
                rinexinfo.auto_coord(brdc, chi_limit=1000)

                # normalize header to add the APR coordinate
                # empty dict since nothing extra to change (other than the APR coordinate)
                rinexinfo.normalize_header(dict())
            except gpys.Rinexexceptionnoautocoord:
                # could not determine an autonomous coordinate, try PPP anyways. 50% chance it will work
                pass

            with gpys.RunPPP(rinexinfo, '', cfg.options, cfg.sp3types, cfg.sp3altrn,
                             rinexinfo.antOffset,
                             strict=False, apply_met=False,
                             clock_interpolation=True) as ppp:  # type: gpys.RunPPP

                try:
                    ppp.exec_ppp()
                # TODO: What exactly would cause an exception here that we haven't already checked for?
                except gpys.RunPPPException as ePPP:

                    # inflate the chi**2 limit to make sure it will pass (even if we get a crappy coordinate)
                    # if coordinate is TOO bad it will get kicked off by the unreasonable geodetic height
                    try:
                        auto_coords_xyz, auto_coords_lla = rinexinfo.auto_coord(brdc, chi_limit=1000)

                    except gpys.Rinexexceptionnoautocoord as e:
                        # catch pyRinexExceptionNoAutoCoord and convert it into a pyRunPPPException

                        raise gpys.RunPPPException(
                            'Both PPP and sh_rx2apr failed to obtain a coordinate for %s.\n'
                            'The file has been moved into the rejection folder. '
                            'Summary PPP file and error (if exists) follows:\n%s\n\n'
                            'ERROR section:\n%s\ngpys.auto_coord error follows:\n%s'
                            % (crinez.replace(cfg.repository_data_in, ''),
                               ppp.summary, str(ePPP).strip(), str(e).strip()))

                    # DDG: this is correct - auto_coord returns a numpy array (calculated in ecef2lla),
                    # so ppp.lat = auto_coords_lla is consistent.
                    ppp.lat = auto_coords_lla[0]
                    ppp.lon = auto_coords_lla[1]
                    ppp.h = auto_coords_lla[2]
                    ppp.x = auto_coords_xyz[0]
                    ppp.y = auto_coords_xyz[1]
                    ppp.z = auto_coords_xyz[2]

                # check for unreasonable heights
                if ppp.h[0] > 9000 or ppp.h[0] < -400:
                    raise gpys.RinexException(os.path.relpath(crinez, cfg.repository_data_in) +
                                                        ' : unreasonable geodetic height (%.3f). '
                                                        'RINEX file will not enter the archive.' % (ppp.h[0]))

                result, match, _ = ppp.verify_spatial_coherence(cnn, station_code)

                if result:
                    # insert: there is only 1 match with the same StationCode.
                    rinexinfo.rename(NetworkCode=match[0][0])
                    insert_data(cnn, archix, rinexinfo)
                else:

                    if len(match) == 1:
                        error = "%s matches the coordinate of %s.%s (distance = %8.3f m) but the filename " \
                                "indicates it is %s. Please verify that this file belongs to %s.%s, rename it and " \
                                "try again. The file was moved to the retry folder. " \
                                "Rename script and pSQL sentence follows:\n" \
                                "BASH# mv %s %s\n" \
                                "PSQL# INSERT INTO stations (\"NetworkCode\", \"StationCode\", \"auto_x\", " \
                                "\"auto_y\", \"auto_z\", \"lat\", \"lon\", \"height\") VALUES " \
                                "('???','%s', %12.3f, %12.3f, %12.3f, " \
                                "%10.6f, %10.6f, %8.3f)\n" \
                                % (os.path.relpath(crinez, cfg.repository_data_in), match[0]['NetworkCode'],
                                   match[0]['StationCode'], float(match[0]['distance']), station_code,
                                   match[0]['NetworkCode'], match[0]['StationCode'],
                                   os.path.join(retry_folder, filename),
                                   os.path.join(retry_folder, filename.replace(station_code, match[0]['StationCode'])),
                                   station_code, ppp.x, ppp.y, ppp.z, ppp.lat[0], ppp.lon[0], ppp.h[0])

                        raise gpys.Runpppexceptioncoordconflict(error)

                    elif len(match) > 1:
                        # a number of things could have happened:
                        # 1) wrong station code, and more than one matching stations
                        #    (that do not match the station code, of course)
                        #    see rms.lhcl 2007 113 -> matches rms.igm0: 34.293 m, rms.igm1: 40.604 m, rms.byns: 4.819 m
                        # 2) no entry in the database for this solution -> add a lock and populate the exit args

                        # no match, but we have some candidates

                        error = "Solution for RINEX in repository (%s %s) did not match a unique station location " \
                                "(and station code) within 5 km. Possible cantidate(s): %s. This file has been moved " \
                                "to data_in_retry. pSQL sentence follows:\n" \
                                "PSQL# INSERT INTO stations (\"NetworkCode\", \"StationCode\", \"auto_x\", " \
                                "\"auto_y\", \"auto_z\", \"lat\", \"lon\", \"height\") VALUES " \
                                "('???','%s', %12.3f, %12.3f, %12.3f, %10.6f, %10.6f, %8.3f)\n" \
                                % (os.path.relpath(crinez, cfg.repository_data_in), rinexinfo.date.yyyyddd(),
                                   ', '.join(['%s.%s: %.3f m' %
                                              (m['NetworkCode'], m['StationCode'], m['distance']) for m in match]),
                                   station_code, ppp.x, ppp.y, ppp.z, ppp.lat[0], ppp.lon[0], ppp.h[0])

                        raise gpys.Runpppexceptioncoordconflict(error)

                    else:
                        # only found a station removing the distance limit (could be thousands of km away!)

                        # The user will have to add the metadata to the database before the file can be added,
                        # but in principle no problem was detected by the process. This file will stay in this folder
                        # so that it gets analyzed again but a "lock" will be added to the file that will have to be
                        # removed before the service analyzes again.
                        # if the user inserted the station by then, it will get moved to the appropriate place.
                        # we return all the relevant metadata to ease the insert of the station in the database

                        otl = gpys.OceanLoading(station_code, cfg.options['grdtab'],
                                                cfg.options['otlgrid'])
                        # use the ppp coordinates to calculate the otl
                        coeff = otl.calculate_otl_coeff(ppp.x, ppp.y, ppp.z)

                        # add the file to the locks table so that it doesn't get processed over and over
                        # this will be removed by user so that the file gets reprocessed once all the metadata is ready
                        cnn.insert('locks', {'filename': os.path.relpath(crinez, cfg.repository_data_in),
                                   'StationCode': station_code})

                        return None, [station_code, (ppp.x, ppp.y, ppp.z), coeff, (ppp.lat[0], ppp.lon[0],
                                                                                   ppp.h[0]), crinez]
    # TODO: This is a giant try statement, I think we can pare it down a bit.
    except (gpys.Rinexexceptionbadfile, gpys.Rinexexceptionsingleepoch,
            gpys.Rinexexceptionnoautocoord) as e:  # pragma: no cover

        reject_folder = reject_folder.replace('%reason%', 'bad_rinex')

        # add more verbose output
        e.event['Description'] = e.event['Description'] + '\n' + os.path.relpath(crinez, cfg.repository_data_in) + \
                                 ': (file moved to ' + reject_folder + ')'
        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy
        # error, move_new the file to rejected folder
        error_handle(cnn, e.event, crinez, reject_folder, filename)

        return None, None

    except gpys.RinexException as e:  # pragma: no cover

        retry_folder = retry_folder.replace('%reason%', 'rinex_issues')

        # add more verbose output
        e.event['Description'] = e.event['Description'] + '\n' + os.path.relpath(crinez, cfg.repository_data_in) + \
                                 ': (file moved to ' + retry_folder + ')'
        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy
        # error, move_new the file to rejected folder
        error_handle(cnn, e.event, crinez, retry_folder, filename)

        return None, None

    except gpys.Runpppexceptioncoordconflict as e:  # pragma: no cover

        retry_folder = retry_folder.replace('%reason%', 'coord_conflicts')

        e.event['Description'] = e.event['Description'].replace('%reason%', 'coord_conflicts')

        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy

        error_handle(cnn, e.event, crinez, retry_folder, filename)

        return None, None

    except gpys.RunPPPException as e:  # pragma: no cover

        reject_folder = reject_folder.replace('%reason%', 'no_ppp_solution')

        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy

        error_handle(cnn, e.event, crinez, reject_folder, filename)

        return None, None

    except gpys.StationInfoException as e:  # pragma: no cover

        retry_folder = retry_folder.replace('%reason%', 'station_info_exception')

        e.event['Description'] = e.event['Description'] + '. The file will stay in the repository and will be ' \
                                                          'processed during the next cycle of pyArchiveService.'
        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy

        error_handle(cnn, e.event, crinez, retry_folder, filename)

        return None, None

    except gpys.OTLException as e:  # pragma: no cover

        retry_folder = retry_folder.replace('%reason%', 'otl_exception')

        e.event['Description'] = e.event['Description'] + ' while calculating OTL for %s. ' \
                                                          'The file has been moved into the retry folder.' \
                                 % os.path.relpath(crinez, cfg.repository_data_in)
        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy

        error_handle(cnn, e.event, crinez, retry_folder, filename)

        return None, None

    except gpys.Productsexceptionunreasonabledate as e:  # pragma: no cover
        # a bad RINEX file requested an orbit for a date < 0 or > now()
        reject_folder = reject_folder.replace('%reason%', 'bad_rinex')

        e.event['Description'] = e.event['Description'] + ' during %s. The file has been moved to the rejected ' \
                                                          'folder. Most likely bad RINEX header/data.' \
                                 % os.path.relpath(crinez, cfg.repository_data_in)
        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy

        error_handle(cnn, e.event, crinez, reject_folder, filename)

        return None, None

    except gpys.ProductsException as e:  # pragma: no cover

        # if PPP fails and ArchiveService tries to run sh_rnx2apr and it doesn't find the orbits, send to retry
        retry_folder = retry_folder.replace('%reason%', 'sp3_exception')

        e.event['Description'] = e.event['Description'] + ': %s. Check the brdc/sp3/clk files and also check that ' \
                                                          'the RINEX data is not corrupt.' \
                                 % os.path.relpath(crinez, cfg.repository_data_in)
        e.event['StationCode'] = station_code
        e.event['NetworkCode'] = '???'
        e.event['Year'] = year
        e.event['DOY'] = doy

        error_handle(cnn, e.event, crinez, retry_folder, filename)

        return None, None

    except gpys.DBErrInsert as e:  # pragma: no cover

        reject_folder = reject_folder.replace('%reason%', 'duplicate_insert')

        # insert duplicate values: two parallel processes tried to insert different filenames
        # (or the same) of the same station to the db: move_new it to the rejected folder.
        # The user might want to retry later. Log it in events
        # this case should be very rare
        event = gpys.Event(Description='Duplicate rinex insertion attempted while processing ' +
                                       os.path.relpath(crinez, cfg.repository_data_in) +
                                               ' : (file moved to rejected folder)\n' + str(e),
                           EventType='warn',
                           StationCode=station_code,
                           NetworkCode='???',
                           Year=year,
                           DOY=doy)

        error_handle(cnn, event, crinez, reject_folder, filename)

        return None, None

    except Exception:  # pragma: no cover

        retry_folder = retry_folder.replace('%reason%', 'general_exception')

        event = gpys.Event(Description=traceback.format_exc() + ' processing: ' +
                                       os.path.relpath(crinez, cfg.repository_data_in) + ' in node ' +
                                       platform.node() + ' (file moved to retry folder)', EventType='error')

        error_handle(cnn, event, crinez, retry_folder, filename, db_log=False)

        return event['Description'], None

    return None, None


def main(args):
    """
    Runs the main part of the script.
    """
    # Parse the config arguments.
    config = gpys.ReadOptions(args.config_file)

    if not os.path.isdir(config.options['repository']):  # pragma: no cover
        raise FileNotFoundError("The provided repository path in {} is not a folder.".format(args.config_file))

    jobserver = gpys.JobServer(config,
                               software_sync=[config.options['ppp_remote_local']],
                               cfg_file=args.config_file)  # type: gpys.JobServer

    conn = gpys.Connection(args.config_file)
    # Log the execution of the script
    conn.insert('executions', {'script': 'ArchiveService.py'})

    # set the data_xx directories
    data_in = os.path.join(config.options['repository'], 'data_in')
    data_in_retry = os.path.join(config.options['repository'], 'data_in_retry')
    data_reject = os.path.join(config.options['repository'], 'data_rejected')

    # if if the subdirs exist
    os.makedirs(data_in, exist_ok=True)
    os.makedirs(data_in_retry, exist_ok=True)
    os.makedirs(data_reject, exist_ok=True)

    archive = gpys.RinexStruct(conn, cfg_file=args.config_file)

    pbar = tqdm(desc='%-30s' % ' >> Scanning data_in_retry',
                ncols=160,
                unit='crz')
    rfiles, paths, _ = archive.scan_archive_struct(data_in_retry, pbar)
    pbar.close()

    pbar = tqdm(desc='%-30s' % ' -- Moving files to data_in',
                total=len(rfiles),
                ncols=160,
                unit='crz')

    for rfile, path in zip(rfiles, paths):

        # move_new the file into the folder
        shutil.move(path, data_in)

        pbar.set_postfix(crinez=rfile)
        pbar.update()

        # remove folder from data_in_retry (also removes the log file)
        try:
            # remove the log file that accompanies this Z file
            os.remove(path.replace('d.Z', '.log'))
        except OSError:
            sys.exit()

    pbar.close()

    conn.update_locks()
    # get the locks to avoid reprocessing files that had no metadata in the database
    locks = conn.load_table('locks')

    if args.purge_locks:
        for table in ['stations', 'networks']:
            conn.clear_locked(table)

    files_path = []
    files_list = []
    pbar = tqdm(desc='%-30s' % ' >> Repository crinez scan', ncols=160)
    rpaths, _, files = archive.scan_archive_struct(data_in, pbar)
    pbar.close()
    pbar = tqdm(desc='%-30s' % ' -- Checking the locks table', total=len(files), ncols=130, unit='crz')

    for file, path in zip(files, rpaths):
        pbar.set_postfix(crinez=file)
        pbar.update()
        if path not in [lock[0] for lock in locks]:
            files_path.append(path)
            files_list.append(file)

    pbar.close()

    tqdm.write(" -- Found %i files in the lock list..." % (len(locks)))
    tqdm.write(" -- Found %i files (matching format [stnm][doy][s].[yy]d.Z) to process..." % (len(files_list)))
    pbar = tqdm(desc='%-30s' % ' >> Processing repository', total=len(files_path), ncols=160, unit='crz')

    depfuncs = (check_rinex_timespan_int, write_error, error_handle, insert_data, verify_rinex_multiday)
    modules = ('gpys', 'os', 'datetime', 'uuid', 'numpy', 'traceback', 'platform')

    fn = process_crinex_file

    def callback_handle(job):
        if job.result is not None:
            out_message = job.result[0]
            new_station = job.result[1]

            if out_message:
                tqdm.write(' -- There were unhandled errors during this batch. '
                           'Please check errors_pyArchiveService.log for details')

                # function to print any error that are encountered during parallel execution
                with open('errors_pyArchiveService.log', 'a') as f:
                    f.write('ON {}  an unhandled error occurred:\n{}\n'
                            'END OF ERROR =================== \n\n'.format(
                            dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            out_message))

            if new_station:
                tqdm.write(' -- New stations were found in the repository. Please assign a network to each new station '
                           'and remove the locks from the files before running again ArchiveService')

                station_code = new_station[0]
                x = new_station[1][0]
                y = new_station[1][1]
                z = new_station[1][2]
                otl = new_station[2]
                lat = new_station[3][0]
                lon = new_station[3][1]
                h = new_station[3][2]

                filename = os.path.relpath(new_station[4], config.repository_data_in)

                insert_station_w_lock(conn, station_code, filename, lat, lon, h, x, y, z, otl)

        elif job.exception:
            tqdm.write(' -- There were unhandled errors during this batch. '
                       'Please check errors_pyArchiveService.log for details')

            # function to print any error that are encountered during parallel execution
            with open('errors_pyArchiveService.log', 'a') as f:
                f.write('ON ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' an unhandled error occurred:\n')
                f.write('{0}\n'.format(job.exception))
                f.write('END OF ERROR =================== \n\n')

    jobserver.create_cluster(fn, depfuncs, callback_handle, pbar, modules=modules)

    for file_to_process, sfile in zip(files_path, files_list):
        jobserver.submit(file_to_process, sfile, data_reject, data_in_retry, args.config_file)

    jobserver.wait()
    pbar.close()
    jobserver.close_cluster()
    conn.print_summary('ArchiveService.py')
    # TODO: Remove the folders in the production folder.


if __name__ == '__main__':  # pragma: no cover
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(description='Archive operations Main Program')

    parser.add_argument('-purge', '--purge_locks', action='store_true',
                        help="Delete any network starting with '?' from the stations table and purge the contents of "
                             "the locks table, deleting the associated files from data_in.")

    parser.add_argument('-dup', '--duplicate', type=str,
                        help='Duplicate the archive as it is seen by the database')

    parser.add_argument('-config', '--config_file', type=str, default='gnss_data.cfg',
                        help='Specify the config file, defaults to gnss_data.cfg in the current directory')

    arguments = parser.parse_args()

    main(arguments)
