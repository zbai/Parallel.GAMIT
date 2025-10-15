#!/usr/bin/env python

"""
Project: Parallel.PPP
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

from pgamit.etm.etm_functions.polynomial import PolynomialFunction
from pgamit.etm.etm_functions.periodic import PeriodicFunction
from pgamit.etm.etm_functions.jumps import JumpFunction
from pgamit.etm.core.etm_config import EtmConfig
from pgamit.etm.core.etm_engine import enum_dict_factory, EtmEncoder
from pgamit.pyDate import Date
from pgamit.etm.core.type_declarations import JumpType
from pgamit.Utils import (add_version_argument, load_json,
                          process_date_str, file_write)

def get_prefit_models(config, args):

    user_prefit_models = []

    if args:
        function_map_str = {'poly': 'polynomial', 'per': 'periodic', 'jump': 'jump'}
        function_map = {'poly': PolynomialFunction, 'per': PeriodicFunction, 'jump': JumpFunction}
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


def main():
    parser = argparse.ArgumentParser(description='Create, extract, and combine ETM functions')

    parser.add_argument('-o', '--output', type=str,
                        help='Output json file with all functions created. If exists, functions will be appended')

    parser.add_argument('-i', '--input_json', nargs=3, default=None, action='append',
                        metavar=('FILE', 'TYPE', 'VALUE'),
                        help="Extract ETM function from json file. Provide list of functions "
                             "to extract using keywords: poly -> polynomial; per -> periodic; jumps -> mechanical and "
                             "geophysical jumps")

    add_version_argument(parser)

    args = parser.parse_args()

    output_json = {
        'functions': load_json(args.output)['functions'] if os.path.exists(args.output) else [] ,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'localhost':  platform.node()
                   }

    if args.input_json:
        for input_opt in args.input_json:
            stn = load_json(input_opt[0])
            config = EtmConfig(stn['network_code'], stn['station_code'], json_file=stn)

            output_json['functions'].extend(
                [asdict(funct.p, dict_factory=enum_dict_factory) for funct in get_prefit_models(config, input_opt)]
            )

    file_write(args.output, json.dumps(output_json, indent=4, sort_keys=False, cls=EtmEncoder))


if __name__ == '__main__':
    main()