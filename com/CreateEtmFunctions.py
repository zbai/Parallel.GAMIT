#!/usr/bin/env python

"""
Project: Geodesy Database Engine (GeoDE)
Date: 10/10/17 9:10 AM
Author: Demian D. Gomez

User interface to plot and save JSON files of ETM objects.
Type python pyPlotETM.py -h for usage help
"""
import argparse
import numpy as np
import os
import json
import platform
from dataclasses import asdict
from datetime import datetime
from scipy.interpolate import griddata

from geode.etm.etm_functions.etm_function import EtmFunction
from geode.dbConnection import Cnn
from geode.etm.etm_functions.polynomial import PolynomialFunction
from geode.etm.etm_functions.periodic import PeriodicFunction
from geode.etm.etm_functions.jumps import JumpFunction
from geode.etm.core.etm_config import EtmConfig
from geode.etm.core.etm_engine import enum_dict_factory, EtmEncoder
from geode.pyDate import Date
from geode.etm.core.type_declarations import JumpType
from geode.Utils import (add_version_argument, load_json,
                          process_date_str, file_write, azimuthal_equidistant)


function_map_str = {'poly': 'polynomial', 'per': 'periodic', 'jump': 'jump'}
function_map = {'poly': PolynomialFunction, 'per': PeriodicFunction, 'jump': JumpFunction}


def get_prefit_models(config, args):

    user_prefit_models = []

    if args:
        # begin by loading the json
        functions = load_json(args[0])['functions']
        # read current function
        current_funct = args[1]
        # set i at the correct location
        i = 2
        while i < len(args):
            # find the json info for this function
            print(' -- Adding ' + function_map_str[current_funct] + ' as prefit model')
            if current_funct == 'jump':
                # if it is a jump assimilate the date also
                mjd = process_date_str(args[i]).mjd
                # find the correct jump
                jf = [funct for funct in functions if funct['object'] == function_map_str[current_funct]
                      and funct['jump_date']['mjd'] == mjd]
                # create instance of function
                funct = function_map[current_funct](config,
                                                    time_vector=np.array([0]),
                                                    date=Date(**jf[0]['jump_date']),
                                                    jump_type=JumpType.UNDETERMINED,
                                                    fit=True)
                i += 1
            else:
                jf = [funct for funct in functions if funct['object'] == function_map_str[current_funct]]
                # create instance of function
                funct = function_map[current_funct](config)

            if not len(jf):
                raise ValueError('Constraint function ' + function_map_str[current_funct] +
                                 ' with matching arguments could not be found')

            funct.load_from_json(jf[0])
            # append to constraints
            user_prefit_models.append(funct)
            i += 1

        for funct in user_prefit_models:
            print(funct.p.params)

    return user_prefit_models


def process_model(config: EtmConfig, model) -> EtmFunction:

    # load the model file
    model_grid = np.loadtxt(model[0], dtype=float)

    lat = config.metadata.lat
    lon = config.metadata.lon

    # project grid onto station position
    x, y = azimuthal_equidistant(lon, lat, model_grid[:, 0], model_grid[:, 1])

    points = np.column_stack([x, y])

    neu = [np.array([]), np.array([]), np.nan]
    neu[0] = griddata(points, model_grid[:, 3] / 1000, (0, 0), method='cubic')
    neu[1] = griddata(points, model_grid[:, 2] / 1000, (0, 0), method='cubic')

    if model[1] == 'jump':
        # if it is a jump assimilate the date also
        mjd = process_date_str(model[2]).mjd
        # create instance of function
        funct = function_map[model[1]](config,
                                       time_vector=np.array([0]),
                                       date=Date(mjd=mjd),
                                       jump_type=JumpType.UNDETERMINED,
                                       fit=True)
    else:
        # create instance of function
        funct = function_map[model[1]](config)

    # load the json file with the station coordinates
    if isinstance(funct, PolynomialFunction):
        for i in range(3):
            funct.p.params[i] = np.array([np.nan, neu[i]])
            # default uncertainty for now
            funct.p.sigmas[i] = np.array([0.00025])

    elif isinstance(funct, JumpFunction):

        if len(model) > 2:
            # relaxation was given
            funct.p.jump_type = JumpType.POSTSEISMIC_ONLY
            funct.p.relaxation = np.array([float(model[3])])
        else:
            # no relaxation
            funct.p.jump_type = JumpType.COSEISMIC_ONLY

        for i in range(3):
            funct.p.params[i] = np.array([neu[i]])
            # default uncertainty for now
            funct.p.sigmas[i] = np.array([0.001])

    return funct

def main():
    parser = argparse.ArgumentParser(description='Create, extract, and combine ETM functions')

    parser.add_argument('station', type=str,
                        help='Target station for the output json function file. Provide a net.stnm to fetcha metadata '
                             'from the database or, alternatively, a json file to read the station metadata from. If '
                             'a net.stnm is given, provide both the station and network codes')

    parser.add_argument('-o', '--output', type=str,
                        help='Output json file with all functions created. If exists, functions will be appended')

    parser.add_argument('-i', '--input_json', nargs=3, default=None, action='append',
                        metavar=('json_file', 'function', 'date'),
                        help="Extract ETM function from json file. Provide list of functions "
                             "to extract using keywords: poly -> polynomial; per -> periodic; jumps -> mechanical and "
                             "geophysical jumps")

    parser.add_argument('-model', '--model', nargs='+', default=None, action='append',
                        metavar=('{model_file} {function}', '[args]'),
                        help="Sample model grid to obtain a velocity or log amplitude model at the location "
                             "determined from the {station} metadata. The ETM {function} "
                             "determines which function will be created (poly or jump). If jump is invoked, then "
                             "provide the date of the jump and, optionally, the [relaxation] term. If "
                             "[relaxation] is given, the jump_type is assumed to be POSTSEIMIC-ONLY. If no relaxation "
                             "is provided, then the jump_type is assumed to be COSEISMIC-ONLY. To produce a full "
                             "coseismic model (CO+POSTSEISMIC), use two separate grids using two -model calls")

    add_version_argument(parser)

    args = parser.parse_args()

    if os.path.exists(args.station):
        # a json file provided
        station = load_json(args.station)
        config = EtmConfig(station['network_code'], station['station_code'], json_file=args.station)
    else:
        # just net.stnm, use database
        cnn = Cnn('gnss_data.cfg', write_cfg_file=True)
        net, stn = args.station.split('.')
        station = {'network_code': net, 'station_code': stn}
        config = EtmConfig(station['network_code'], station['station_code'], cnn=cnn)

    # read the output json to append to it if exists
    config.metadata.station_information = [item.to_json() for item in config.metadata.station_information]
    output_json = {
        'station_meta': asdict(config.metadata, dict_factory=enum_dict_factory),
        'functions': load_json(args.output)['functions'] if os.path.exists(args.output) else [] ,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'localhost':  platform.node()
                   }

    if args.input_json:
        for input_opt in args.input_json:
            output_json['functions'].extend(
                [asdict(funct.p, dict_factory=enum_dict_factory) for funct in get_prefit_models(config, input_opt)]
            )

    if args.model:
        for model in args.model:
            output_json['functions'].append(asdict(process_model(config, model).p, dict_factory=enum_dict_factory))

    file_write(args.output, json.dumps(output_json, indent=4, sort_keys=False, cls=EtmEncoder))


if __name__ == '__main__':
    main()