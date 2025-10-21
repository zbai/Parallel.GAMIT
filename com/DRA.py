#!/usr/bin/env python
"""
Project: Parallel.Stacker
Date: 6/12/18 10:28 AM
Author: Demian D. Gomez
"""

import argparse
import logging
import os
from datetime import datetime
import json

# deps
import numpy as np
from tqdm import tqdm
import matplotlib

from geode.etm.core.etm_engine import EtmEngine

if not os.environ.get('DISPLAY', None):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy

# app
from geode import dbConnection
from geode.etm.core import etm_engine
from geode.etm.core import etm_config
from geode.etm.core import type_declarations
from geode import pyDate
from geode import pyJobServer
from geode import pyOptions
from geode.Utils import process_date, file_write, json_converter, add_version_argument
from geode.pyStack import Polyhedron, np_array_vertices
from geode.pyDate import Date

stn_stats = []
wrms_n = []
wrms_e = []
wrms_u = []
project = 'default'

LIMIT = 2.5


def sql_select(project, fields, date2):
    return '''SELECT %s from gamit_soln
          WHERE "Project" = \'%s\' AND "Year" = %i AND "DOY" = %i
          ORDER BY "NetworkCode", "StationCode"''' % (
              fields, project, date2.year, date2.doy)


def save_excluded_soln(cnn, etm, network_code: str, station_code: str, project: str):

    res_mag = numpy.sqrt(numpy.sum(numpy.square([r.residuals for r in etm.fit.results]), axis=0))

    for date, f, r in zip(etm.solution_data.coordinates.dates,
                          etm.fit.outlier_flags,
                          res_mag):

        if not cnn.query_float('SELECT * FROM gamit_soln_excl WHERE "NetworkCode" = \'%s\' AND '
                               '"StationCode" = \'%s\' AND "Project" = \'%s\' AND '
                               '"Year" = %i AND "DOY" = %i'
                               % (network_code, station_code, project,
                                  date.year, date.doy)) \
                and not f:

            cnn.query('INSERT INTO gamit_soln_excl ("NetworkCode", "StationCode", "Project", '
                      '"Year", "DOY", residual) VALUES (\'%s\', \'%s\', \'%s\', %i ,%i, %.4f)'
                      % (network_code, station_code, project, date.year, date.doy, r))


def compute_dra(ts, NetworkCode, StationCode,
                pdates, project, histogram=False, save_excluded=False):

    try:
        from geode.etm.core.logging_config import setup_etm_logging
        # deactivate messages
        setup_etm_logging(level=logging.CRITICAL)

        # load from the db
        cnn = dbConnection.Cnn('gnss_data.cfg')

        config = etm_config.EtmConfig(NetworkCode, StationCode, cnn=cnn)
        config.plotting_config.plot_show_outliers = True
        config.solution.solution_type = type_declarations.SolutionType.DRA
        config.solution.project = project
        # force linear components only
        config.modeling.poly_terms = 1

        # to pass the filename back to the callback_handler
        filename = project + '_dra/' + NetworkCode + '.' + StationCode

        if ts.size:

            if ts.shape[0] > 2:
                dts = numpy.append(numpy.diff(ts[:, 0:3], axis=0),
                                   ts[1:, -3:], axis=1)

                etm = etm_engine.EtmEngine(config=config, cnn=cnn, polyhedrons=dts)
                etm.run_adjustment(cnn=cnn, try_save_to_db=False, try_loading_db=False)

                fig_file = ''
                his_file = ''

                if etm.config.modeling.status == etm.config.modeling.status.POSTFIT:
                    # got a DRA back
                    config.plotting_config.file_io = io.BytesIO()
                    config.plotting_config.plot_time_window = pdates

                    fig_file = etm.plot()

                    if histogram:
                        his_file = etm.plot_hist()

                    if save_excluded:
                        save_excluded_soln(cnn, etm, NetworkCode, StationCode, project)

                    # save the wrms
                    return (
                        etm.fit.results[0].wrms * 1000, etm.fit.results[1].wrms * 1000,
                        etm.fit.results[2].wrms * 1000, fig_file, his_file, filename, NetworkCode,
                        StationCode, config.metadata.lat[0], config.metadata.lon[0], config.metadata.height[0]
                    )
                else:
                    return (
                        None, None, None, fig_file, his_file,
                        filename, NetworkCode, StationCode,
                        config.metadata.lat[0], config.metadata.lon[0], config.metadata.height[0]
                    )

    except Exception as e:
        raise Exception('While working on %s.%s' % (NetworkCode,
                                                    StationCode) + '\n') from e

    return None, None, None, '', '', filename, NetworkCode, StationCode, 0, 0, 0


class DRA(list):

    def __init__(self, cnn, project, start_date, end_date, verbose=False):

        super(DRA, self).__init__()

        self.project = project
        self.cnn = cnn
        self.transformations = []
        self.stats = {}
        self.verbose = verbose

        if end_date is None:
            end_date = Date(datetime=datetime.now())

        print(' >> Loading GAMIT solutions for project %s...' % project)

        gamit_vertices = self.cnn.query_float(
            'SELECT "NetworkCode" || \'.\' || "StationCode", "X", "Y", "Z",'
            ' "Year", "DOY", "FYear" '
            'FROM gamit_soln WHERE "Project" = \'%s\' AND ("Year", "DOY")'
            ' BETWEEN (%i, %i) AND (%i, %i) '
            'ORDER BY "NetworkCode", "StationCode"' % (
                project, start_date.year, start_date.doy,
                end_date.year, end_date.doy))

        self.gamit_vertices = np_array_vertices(gamit_vertices)

        dates = self.cnn.query_float(
            'SELECT "Year", "DOY" FROM gamit_soln WHERE "Project" = \'%s\' '
            'AND ("Year", "DOY") BETWEEN (%i, %i) AND (%i, %i) '
            'GROUP BY "Year", "DOY" ORDER BY "Year", "DOY"'
            % (project, start_date.year, start_date.doy,
               end_date.year, end_date.doy))

        self.dates = [Date(year=int(d[0]), doy=int(d[1])) for d in dates]

        self.stations = self.cnn.query_float(
            'SELECT "NetworkCode", "StationCode" FROM gamit_soln '
            'WHERE "Project" = \'%s\' AND ("Year", "DOY") '
            'BETWEEN (%i, %i) AND (%i, %i) '
            'GROUP BY "NetworkCode", "StationCode" '
            'ORDER BY "NetworkCode", "StationCode"'
            % (project, start_date.year, start_date.doy,
               end_date.year, end_date.doy), as_dict=True)

        i = 0
        for d in tqdm(self.dates, ncols=160,
                      desc=' >> Initializing the stack polyhedrons'):
            self.append(Polyhedron(self.gamit_vertices, project, d))
            if i < len(self.dates) - 1:
                if d != self.dates[i + 1] - 1:
                    for dd in [Date(mjd=md) for md in list(range(
                            d.mjd + 1, self.dates[i + 1].mjd))]:
                        tqdm.write(' -- Missing DOY detected: %s'
                                   % dd.yyyyddd())
            i += 1

    def stack_dra(self):

        for j in tqdm(list(range(len(self) - 1)),
                      desc=' >> Daily repeatability analysis progress',
                      ncols=160):

            # should only attempt to align a polyhedron that is unaligned
            # do not set the polyhedron as aligned
            # unless we are in the max iteration step
            self[j + 1].align(self[j], scale=False, verbose=self.verbose)
            # write info to the screen
            tqdm.write(' -- %s (%04i) %2i it wrms: %4.1f T %6.1f %6.1f %6.1f '
                       'R (%6.1f %6.1f %6.1f)*1e-9 D-W: %5.3f IQR: %4.1f' %
                       (self[j + 1].date.yyyyddd(),
                        self[j + 1].stations_used,
                        self[j + 1].iterations,
                        self[j + 1].wrms * 1000,
                        self[j + 1].helmert[3] * 1000,
                        self[j + 1].helmert[4] * 1000,
                        self[j + 1].helmert[5] * 1000,
                        self[j + 1].helmert[0],
                        self[j + 1].helmert[1],
                        self[j + 1].helmert[2],
                        self[j + 1].down_frac,
                        self[j + 1].iqr * 1000
                        ))

        self.transformations.append([poly.info() for poly in self[1:]])

    def get_station(self, NetworkCode, StationCode):
        """
        Obtains the time series for a given station
        :param NetworkCode:
        :param StationCode:
        :return: a numpy array with the time series [x, y, z, yr, doy, fyear]
        """

        stnstr = NetworkCode + '.' + StationCode

        ts = []

        for poly in self:
            p = poly.vertices[poly.vertices['stn'] == stnstr]
            if p.size:
                ts.append([p['x'][0],
                           p['y'][0],
                           p['z'][0],
                           p['yr'][0],
                           p['dd'][0],
                           p['fy'][0]])

        return np.array(ts)

    def to_json(self, json_file):
        # print(repr(self.transformations))
        file_write(json_file,
                   json.dumps({'transformations': self.transformations,
                              'stats': self.stats},
                              indent=4, sort_keys=False, default=json_converter
                              ))


def callback_handler(job):

    global stn_stats, wrms_n, wrms_e, wrms_u

    if job.exception:
        tqdm.write(
            ' -- Fatal error on node %s message from node follows -> \n%s' % (
                job.ip_addr, job.exception))
    elif job.result is not None:
        if job.result[0] is not None:
            # pass information about the station's stats to save in the json
            stn_stats.append({'NetworkCode': job.result[6],
                              'StationCode': job.result[7],
                              'lat': job.result[8],
                              'lon': job.result[9],
                              'height': job.result[10],
                              'wrms_n': job.result[0],
                              'wrms_e': job.result[1],
                              'wrms_u': job.result[2]})

            wrms_n.append(job.result[0])
            wrms_e.append(job.result[1])
            wrms_u.append(job.result[2])

            # save the figures, if any
            if job.result[3]:
                with open('%s.png' % job.result[5], "wb") as fh:
                    fh.write(base64.b64decode(job.result[3]))

            if job.result[4]:
                with open('%s_hist.png' % job.result[5], "wb") as fh:
                    fh.write(base64.b64decode(job.result[4]))
        else:
            tqdm.write(' -- Station %s.%s did not produce valid statistics' % (
                job.result[6], job.result[7]))


def main():

    global stn_stats, wrms_n, wrms_e, wrms_u, project

    parser = argparse.ArgumentParser(
        description='GNSS daily repeatability analysis (DRA)')

    parser.add_argument('project', type=str, nargs=1, metavar='{project name}',
                        help='''Specify the project name used to process
                             the GAMIT solutions in GeoDE.''')

    parser.add_argument('-d', '--date_filter', nargs='+', metavar='date',
                        help="Date range filter. Can be specified in yyyy/mm/dd yyyy_doy  wwww-d format")

    parser.add_argument('-w', '--plot_window', nargs='+', metavar='date',
                        help="Date window range to plot. Can be specified in yyyy/mm/dd yyyy_doy  wwww-d format")

    parser.add_argument('-hist', '--histogram', action='store_true',
                        help="Plot a histogram of the daily repeatabilities")

    parser.add_argument('-se', '--save_excluded', action='store_true',
                        help="Save the DRA outliers to gamit_soln_excl for use with WeeklyCombination")

    parser.add_argument('-verb', '--verbose', action='store_true',
                        help="Provide additional information during the alignment process (for debugging purposes)")

    parser.add_argument('-np', '--noparallel', action='store_true',
                        help="Execute command without parallelization.")

    add_version_argument(parser)

    args = parser.parse_args()

    cnn = dbConnection.Cnn("gnss_data.cfg")

    project = args.project[0]

    dates = [pyDate.Date(year=1980, doy=1),
             pyDate.Date(year=2100, doy=1)]

    Config = pyOptions.ReadOptions("gnss_data.cfg")

    JobServer = pyJobServer.JobServer(
        Config, check_archive=False, check_atx=False, check_executables=False,
        run_parallel=not args.noparallel)

    try:
        dates = process_date(args.date_filter)
    except ValueError as e:
        parser.error(str(e))

    pdates = None
    if args.plot_window is not None:
        if len(args.plot_window) == 1:
            try:
                pdates = process_date(args.plot_window,
                                      missing_input=None,
                                      allow_days=False)
                pdates = (pdates[0].fyear,)
            except ValueError:
                # an integer value
                pdates = float(args.plot_window[0])
        else:
            pdates = process_date(args.plot_window)
            pdates = (pdates[0].fyear,
                      pdates[1].fyear)

    # create folder for plots
    path_plot = project + '_dra'
    if not os.path.isdir(path_plot):
        os.makedirs(path_plot)

    ########################################
    # load polyhedrons
    # create the DRA object
    dra = DRA(cnn, args.project[0], dates[0], dates[1], args.verbose)

    dra.stack_dra()

    tqdm.write(" >> Daily repeatability analysis done. DOYs with wrms > 8 mm are shown below:")
    for i, d in enumerate(dra):
        if d.wrms is not None:
            if d.wrms > 0.008:
                tqdm.write(
                    " -- %s (%04i) %2i it wrms: %4.1f D-W: %5.3f IQR: %4.1f" % (
                        d.date.yyyyddd(), d.stations_used, d.iterations, d.wrms * 1000,
                        d.down_frac, d.iqr * 1000)
                )

    qbar = tqdm(total=len(dra.stations), desc=' >> Computing DRAs',
                ncols=160, disable=None)


    modules = ('geode.dbConnection',
               'geode.etm.core.etm_engine',
               'geode.etm.core.etm_config',
               'geode.etm.core.type_declarations',
               'io', 'numpy', 'logging')

    JobServer.create_cluster(compute_dra, progress_bar=qbar,
                             callback=callback_handler,
                             modules=modules, deps=(save_excluded_soln,))

    # plot each DRA
    for stn in dra.stations:
        NetworkCode = stn['NetworkCode']
        StationCode = stn['StationCode']

        ts = dra.get_station(NetworkCode, StationCode)

        JobServer.submit(ts, NetworkCode, StationCode, pdates,
                         project, args.histogram, args.save_excluded)

    JobServer.wait()
    qbar.close()
    JobServer.close_cluster()

    # add the station stats to the json output
    dra.stats = stn_stats
    dra.to_json(project + '_dra.json')

    wrms_n = np.array(wrms_n)
    wrms_e = np.array(wrms_e)
    wrms_u = np.array(wrms_u)

    # plot the WRM of the DRA stack and number of stations
    # type: plt.subplots
    f, axis = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))

    # WRMS
    ax = axis[0][0]
    ax.plot([t['fyear'] for t in dra.transformations[0]],
            [t['wrms'] * 1000 for t in dra.transformations[0]], 'ob',
            markersize=2)
    ax.set_ylabel('WRMS [mm]')
    ax.grid(True)
    ax.set_ylim(0, 10)

    # station count
    ax = axis[1][0]
    ax.plot([t['fyear'] for t in dra.transformations[0]],
            [t['stations_used'] for t in dra.transformations[0]], 'ob',
            markersize=2)
    ax.set_ylabel('Station count')
    ax.grid(True)

    # d-w fraction
    ax = axis[2][0]
    ax.plot([t['fyear'] for t in dra.transformations[0]],
            [t['downweighted_fraction'] for t in dra.transformations[0]],
            'ob', markersize=2)
    ax.set_ylabel('DW fraction')
    ax.grid(True)

    ax = axis[0][1]
    ax.hist(wrms_n[wrms_n <= 8], 40, alpha=0.75, facecolor='blue')
    ax.grid(True)
    ax.set_ylabel('# stations')
    ax.set_xlabel('WRMS misfit N [mm]')
    ax.set_title('Daily repetitivities NEU')

    ax = axis[1][1]
    ax.hist(wrms_e[wrms_e <= 8], 40, alpha=0.75, facecolor='blue')
    ax.grid(True)
    ax.set_xlim(0, 8)
    ax.set_ylabel('# stations')
    ax.set_xlabel('WRMS misfit E [mm]')

    ax = axis[2][1]
    ax.hist(wrms_u[wrms_u <= 10], 40, alpha=0.75, facecolor='blue')
    ax.grid(True)
    ax.set_xlim(0, 10)
    ax.set_ylabel('# stations')
    ax.set_xlabel('WRMS misfit U [mm]')

    f.suptitle('Daily repetitivity analysis for project %s\n'
               'Solutions with WRMS > 10 mm are not shown' % project,
               fontsize=12, family='monospace')
    plt.savefig(project + '_dra.png')
    plt.close()

    ax.set_xlim(0, 8)


if __name__ == '__main__':
    main()
