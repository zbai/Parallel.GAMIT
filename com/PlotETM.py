#!/usr/bin/env python

"""
Project: Parallel.PPP
Date: 10/10/17 9:10 AM
Author: Demian D. Gomez

User interface to plot and save JSON files of ETM objects.
Type python pyPlotETM.py -h for usage help
"""
import argparse
import os
import traceback
import json
import logging

# deps
import numpy as np
import zipfile
from io import BytesIO
import xml.etree.ElementTree as ET

# app
from pgamit import dbConnection
from pgamit.pyDate import Date
from pgamit.etm.core.logging_config import setup_etm_logging
from pgamit.etm.core.etm_engine import EtmEngine
from pgamit.etm.core.etm_config import EtmConfig
from pgamit.etm.core.data_classes import AdjustmentModels, SolutionType, CovarianceFunction
from pgamit.Utils import (process_date,
                          process_stnlist,
                          file_write,
                          station_list_help,
                          stationID,
                          print_columns,
                          add_version_argument)


# Map verbosity to logging levels
VERBOSITY_MAP = {
    'quiet': logging.CRITICAL,  # or logging.NOTSET to disable all
    'info': logging.INFO,
    'debug': logging.DEBUG
}

ADJUSTMENT_MODEL_MAP = {
    'rls': AdjustmentModels.ROBUST_LEAST_SQUARES,
    'lsc': AdjustmentModels.LSQ_COLLOCATION
}

COVARIANCE_MAP = {
    'arma': CovarianceFunction.ARMA,
    'gaussian': CovarianceFunction.GAUSSIAN
}

def process_plot_dates(args):
    dates = None
    if args.plot_window is not None:
        if len(args.plot_window) == 1:
            try:
                dates = process_date(args.plot_window, missing_input=None, allow_days=False)
                dates = (dates[0].fyear, )
            except ValueError:
                # an integer value
                dates = float(args.plot_window[0])
        else:
            dates = process_date(args.plot_window)
            dates = (dates[0].fyear, dates[1].fyear)

    return dates


def process_fit_dates(args):
    dates = []
    if args.fit_window is not None:
        if np.mod(args.fit_window, 2):
            raise ValueError('Fit dates must be given in start/end pairs')

    for i in range(0, len(args.fit_window), 2):
        try:
            dates.append(process_date(args.fit_window[i:i+1]))
        except ValueError:
            raise ValueError('Invalid fit dates')

    return None if not dates else dates


def print_query(args, etm: EtmEngine):
    model = (args.query[0] == 'model')
    q_date = Date(fyear=float(args.query[1]))

    xyz, _, _, txt = etm.get_position(q_date.year, q_date.doy, force_model=model)

    strp = ''
    # if user requests velocity too, output it
    if etm.config.modeling.status == etm.config.modeling.status.POSTFIT:
        if 'vel' in args.query:
            vxyz = [x.p.params[:, 1].tolist() for x in
                              etm.design_matrix.functions if x.p.object == 'polynomial']
            strp = '%8.5f %8.5f %8.5f ' % (vxyz[0], vxyz[0], vxyz[0])

        if 'per' in args.query:
            # also output seasonal terms, if requested
            strp += ' '.join(['%8.5f' % (x.p.params.flatten() * 1000).tolist() for x in
                              etm.design_matrix.functions if x.p.object == 'periodic'])

    print(' %s %14.5f %14.5f %14.5f %8.3f %s -> %s' \
          % (etm.config.get_station_id(), xyz[0], xyz[1], xyz[2], q_date.fyear, strp, txt))

def main():
    parser = argparse.ArgumentParser(description='Plot extended trajectory models (ETMs) '
                                                 'for station data stored in the database, json files, or text files')

    parser.add_argument('stnlist', type=str, nargs='+',
                        help=station_list_help())

    parser.add_argument('-sol', '--solution', type=str, metavar='{stack|ppp}', default='ppp',
                        help="Required: specify the GAMIT stack name or ppp to use GPSPACE solutions.")

    parser.add_argument('-lsq', '--least_squares_strategy',
                        choices=['lsc', 'rls'], default='rls',
                        help="Adjustment strategy to fit the time series. Choose between least squares collocation "
                             "(lsc) and robust least squares (rls)")

    parser.add_argument('-cova', '--covariance', nargs='+',
                        choices=['arma', 'gaussian'], default=['arma', 49, 200],
                        help="To use with strategy lsc, provide the method to estimate the covariance matrix. "
                             "Choose between Autoregressive Moving Average (arma, default) and gaussian covariance "
                             "function. If arma is used, optionally provide the number of roots for the decomposition "
                             "(49 by default) and the number of points to estimate the AR process (200 by default)")

    parser.add_argument('-fit_win', '--fit_window', nargs='+', metavar='interval',
                        help='Date range to window data fit. Can be specified in yyyy/mm/dd, yyyy_doy '
                             'in pairs for data ranges to be included in the ETM fit. Data outside '
                             'the range will be plotted but not included in the fit')

    parser.add_argument('-sigma', '--sigma_limit', type=float, default=2.5,
                        help="Number of sigmas for the limit of outlier detection. Default is 2.5 sigma")

    parser.add_argument('-iter', '--max_iterations', type=int, default=10,
                        help="Maximum number of iterations during outlier detection. Default is 10")

    parser.add_argument('-options', '--plot_options', nargs='*',
                        choices=['out', 'hist', 'missing', 'residuals', 'no-model', 'no-plots'],
                        default=[],
                        help="Plotting options: "
                             "out -> plot right panel with the outliers marked in cyan; "
                             "hist -> plot additional file with histograms of residuals; "
                             "residuals -> show only residuals (remove all ETM functions); "
                             "missing -> show missing solution days as magenta lines in the plot; "
                             "no-model -> plot time series without fitting a model (do not estimate parameters); "
                             "no-plots -> to use for querying, do not produce plots but estimate parameters")

    parser.add_argument('-dir', '--directory', type=str, default='production/',
                        help="Directory to save the time series PNG files. If not specified, "
                             "files will be saved in the production directory")

    parser.add_argument('-json', '--json',
                        choices=['dmx', 'obs', 'raw', 'mod'], nargs='*',
                        help="Export ETM to JSON. By default the json will contain the "
                             "station metadata and the functions fit by the ETM (with "
                             "parameters and sigmas). Append additional output options: "
                             "dmx (design matrix), obs (observations), raw (raw results), "
                             "mod (model)")

    parser.add_argument('-gui', '--interactive', action='store_true',
                        help="Interactive mode: allows to zoom and view the plot interactively")

    parser.add_argument('-rm', '--remove', nargs='+', choices=['poly', 'per', 'jumps', 'stoch'],
                        default=[],
                        help="Remove components from model and time series before plotting. Options are: "
                             "poly -> polynomial; per -> periodic; jumps -> mechanical and geophysical jumps; "
                             "stoch -> stochastic noise if adjustment strategy is 'least squares collocation'")

    parser.add_argument('-plot_win', '--plot_window', nargs='+', metavar='interval',
                        default=[],
                        help='Date range to window plot. Can be specified in yyyy/mm/dd, yyyy_doy or as a single '
                             'integer value (N) which shall be interpreted as last epoch minus N days')

    parser.add_argument('-lang', '--language', type=str, default='ENG',
                        help="Change the language of the plots. Default is English. "
                        "Use ESP to select Spanish. To add more languages, "
                        "include the ISO 639-1 code in pyETM.py")

    parser.add_argument('-file', '--filename', type=str, default=None,
                        help="Obtain data from an external source (filename, json or text format). This name accepts "
                             "variables for {net} and {stn} to specify more than one file based on a list of "
                             "stations. If a single file is used (no variables), then only the first station "
                             "is processed. File column format should be specified with -format (required) "
                             "unless files are in json format")

    parser.add_argument('-format', '--format', nargs='+', type=str,
                        help="To be used together with --filename. Specify order of the fields as found in the input "
                             "file. Format strings are gpsWeek, gpsWeekDay, year, doy, fyear, month, day, mjd, "
                             "x, y, z, na. Use 'na' to specify a field that should be ignored. If fields to be ignored "
                             "are at the end of the line, then there is no need to specify those")

    parser.add_argument('-kmz', '--kmz', nargs=1, type=str, default=None,
                        help="To be used together with --filename and --format. Do not fetch station metadata from "
                             "the database but rather use the provided kmz/kml file to obtain station coordinates "
                             "and other relevant metadata. When using this option, the station list is ignored and "
                             "only one station is processed.")

    parser.add_argument('-fit_auto', '--fit_auto_jumps', nargs='?',
                        choices=['angry', 'dbscan'], default=['angy'],
                        help="Fit unmodeled but automatically detected jumps. "
                             "Choose between two algorithms: "
                             "angry (used and provided by JPL) and dbscan")

    parser.add_argument('-q', '--query', nargs='*',
                        metavar='{type} {date} ', default=[],
                        help='Specify "model" or "solution" to get the '
                             'ETM value or the value of the daily solution (if exists). '
                             'Specify the date of the desired output. Append "vel" and "per" '
                             'to also include the velocity and seasonal (periodic) components. '
                             'Output is in XYZ.')

    parser.add_argument('-no_save', '--no_save_database', action='store_true', default=False,
                        help="Do not fetch / save ETM solution from / to database")

    parser.add_argument('-verbosity', '--verbosity',
                        choices=['quiet', 'info', 'debug'], default='info',
                        help="Determine how detailed the execution messages should be. "
                             "Default is 'info'")

    add_version_argument(parser)

    args = parser.parse_args()

    cnn = dbConnection.Cnn('gnss_data.cfg', write_cfg_file=True)

    setup_etm_logging(level=VERBOSITY_MAP[args.verbosity])

    stnlist = process_stnlist(cnn, args.stnlist)

    plot_dates = process_plot_dates(args)

    for stn in stnlist:

        config = EtmConfig(stn['NetworkCode'], stn['StationCode'], cnn=cnn)

        config.solution.solution_type = SolutionType.PPP if args.solution == 'ppp' else SolutionType.GAMIT
        config.solution.stack_name = args.solution

        config.plotting_config.plot_time_window = plot_dates

        config.plotting_config.plot_remove_polynomial = 'poly' in args.remove
        config.plotting_config.plot_remove_jumps = 'jumps' in args.remove
        config.plotting_config.plot_remove_periodic = 'per' in args.remove
        config.plotting_config.plot_remove_stochastic = 'stoch' in args.remove

        config.plotting_config.plot_show_outliers = 'out' in args.plot_options
        config.plotting_config.plot_residuals_mode = 'residuals' in args.plot_options

        config.modeling.data_model_window = process_fit_dates(args)

        config.modeling.least_squares_strategy.adjustment_model = ADJUSTMENT_MODEL_MAP[args.least_squares_strategy]
        config.modeling.least_squares_strategy.iterations = args.max_iterations
        config.modeling.least_squares_strategy.sigma_filter_limit = args.sigma_limit
        config.modeling.least_squares_strategy.covariance_function = COVARIANCE_MAP[args.covariance[0]]
        if args.covariance[0] == 'arma' and len(args.covariance) > 1:
            config.modeling.least_squares_strategy.arma_roots = args.covariance[1]
        if args.covariance[0] == 'arma' and len(args.covariance) > 2:
            config.modeling.least_squares_strategy.arma_points = args.covariance[2]

        if not os.path.exists(args.directory):
            os.mkdir(args.directory)

        etm = EtmEngine(config, cnn=cnn)
        etm.run_adjustment(cnn=cnn,
                           try_save_to_db=not args.save_database,
                           try_loading_db=not args.save_database)

        etm.save_etm(filename=args.directory,
                     dump_observations='obs' in args.json,
                     dump_design_matrix='dmx' in args.json,
                     dump_raw_results='raw' in args.json,
                     dump_model='mod' in args.json)

        if 'no-plots' not in args.plot_options:
            etm.plot()
            if 'hist' in args.plot_options:
                etm.plot_hist()

        if args.query:
            print_query(args, etm)

        print('Successfully plotted ' + stn['NetworkCode'] + '.' + stn['StationCode'])


if __name__ == '__main__':
    main()
