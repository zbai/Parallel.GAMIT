#!/usr/bin/env python

"""
Project: Geodesy Database Engine (GeoDE)
Date: 10/10/17 9:10 AM
Author: Demian D. Gomez

Routines to stack ETMs and jointly estimate parameters.
"""

import argparse
import numpy as np
import logging

import numpy.linalg

# app
from geode.etm.core.etm_config import EtmConfig
from geode.etm.core.data_classes import ModelingParameters
from geode.etm.core.logging_config import setup_etm_logging
from geode.etm.core.etm_engine import EtmEngine
from geode.etm.core.etm_stacker import EtmStacker
from geode.dbConnection import Cnn
from geode.Utils import add_version_argument, station_list_help
from geode.Utils import stationID, process_stnlist, azimuthal_equidistant
from geode.elasticity.elastic_interpolation import get_qpw, get_radius
from geode.elasticity.diskload import compute_diskload, load_love_numbers
from geode.etm.etm_functions.polynomial import PolynomialFunction

# Map verbosity to logging levels
VERBOSITY_MAP = {
    'quiet': logging.CRITICAL,  # or logging.NOTSET to disable all
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def main():
    parser = argparse.ArgumentParser(description='Routines to stack ETMs and jointly estimate parameters')

    parser.add_argument('stnlist', type=str, nargs='+',
                        help=station_list_help())

    parser.add_argument('-s_score', '--s_score_mag_limit', type=float, default=6.0, metavar='magnitude',
                        help="Limit the s-score search to earthquakes with magnitude >= {magnitude}. Default is 6.0")

    parser.add_argument('-force', '--force_earthquakes', nargs='+', default=[], metavar='event_id',
                        help="Add cherry-picked seismic earthquake (that fall outside of s_score_mag_limit) to the "
                             "list of jump functions to fit (using the USGS event id). Event needs to have an "
                             "s-score > 0 to be considered, even if it has been cherry-picked")

    parser.add_argument('-relax', '--default_relax', type=float, nargs='+',
                        default=ModelingParameters().relaxation,
                        help="Relaxation value(s) to use during the fit. Default as defined by the station in "
                             "the database or in the ETM module (0.05 and 1 years)")

    parser.add_argument('-verbosity', '--verbosity',
                        choices=['quiet', 'info', 'debug'], default='info',
                        help="Determine how detailed the execution messages should be. "
                             "Default is 'info'")

    add_version_argument(parser)

    args = parser.parse_args()

    cnn = Cnn('gnss_data.cfg', write_cfg_file=True)

    setup_etm_logging(level=VERBOSITY_MAP[args.verbosity])

    stnlist = process_stnlist(cnn, args.stnlist)

    etm_stack = EtmStacker(
        earthquake_magnitude_limit=args.s_score_mag_limit,
        relaxation=np.array(args.default_relax),
        earthquakes_cherry_picked=args.force_earthquakes
    )

    for stn in stnlist:
        # processing station
        etm_stack.add_station(cnn, stn['NetworkCode'], stn['StationCode'])

    etm_stack.build_normal_eq(cnn)
    velocities, postseismic = etm_stack.fit_stack_parameters()

    with open('velo_inter.txt', 'w') as f:
        for geo, vel in zip(etm_stack.interpolation_geographic, etm_stack.grid_velocities.T):
            lon, lat = geo
            ve, vn, vu = vel * 1000

            f.write(f' gp '
                    f'{lon:15.8f} {lat:15.8f} '
                    f'{ve:6.2f} '
                    f'{vn:6.2f} '
                    f'{vu:6.2f}\n')

    for event in etm_stack.earthquakes:
        with open(event.event.id + '.txt', 'w') as f:
            for i, relax in enumerate(event.relaxation):
                if len(event.grid_amplitudes):
                    for geo, amp in zip(etm_stack.interpolation_geographic,
                                        event.grid_amplitudes[i].T):
                        lon, lat = geo
                        ae, an, au = amp * 1000

                        f.write(f' gp '
                                f'{lon:15.8f} {lat:15.8f} '
                                f'{event.event.id:32s} '
                                f'{relax:6.3f} '
                                f'{ae:6.2f} '
                                f'{an:6.2f} '
                                f'{au:6.2f}\n')

    with open('velo.txt', 'w') as f:
        for stn in velocities:
            name = stn['station']
            vc = stn['vc']
            vp = stn['vp']
            lon = stn['lon']
            lat = stn['lat']
            inter = stn['c_interseismic']

            f.write(f'{name} '
                    f'{lon:15.8f} {lat:15.8f} '
                    f'{vc[0]:6.2f} '
                    f'{vc[1]:6.2f} '
                    f'{vc[2]:6.2f} '
                    f'{vp[0]:6.2f} '
                    f'{vp[1]:6.2f} '
                    f'{vp[2]:6.2f} '
                    f'{inter}\n')

    with open('earthquakes_post.txt', 'w') as f:
        for stn in postseismic:
            name = stn['station']
            event_id = stn['event_id']
            relax = stn['relax']
            vc = stn['rc']
            vp = stn['rp']
            lon = stn['lon']
            lat = stn['lat']
            is_const = stn['c_relax']

            f.write(f'{name} '
                    f'{lon:15.8f} {lat:15.8f} '
                    f'{event_id:32s}'
                    f'{relax:6.3f}'
                    f'{vc[0]:10.2f} '
                    f'{vc[1]:10.2f} '
                    f'{vc[2]:10.2f} '
                    f'{vp[0]:10.2f} '
                    f'{vp[1]:10.2f} '
                    f'{vp[2]:10.2f} '
                    f'{is_const}\n')

    for stn in stnlist:
        # redo with constrained velocity
        etm_stack.plot_constrained_etm(cnn, stn['NetworkCode'], stn['StationCode'], './velo_const/')


if __name__ == '__main__':
    main()
