"""
Project: Geodesy Database Engine (GeoDE)
Date: 3/31/17 3:39 PM
Author: Demian D. Gomez

Class that holds the station metadata needed to process in GAMIT
"""

import random
import string
import os

# deps
import numpy as np
from tqdm import tqdm

# app
from geode.metadata.station_info import StationInfo, StationInfoHeightCodeNotFound, StationInfoNoRecordFound
from geode.etm.core.etm_config import EtmConfig
from geode.etm.core.etm_engine import EtmEngine
from geode.etm.core.type_declarations import SolutionType, EtmSolutionType
from geode.etm.core.data_classes import SolutionOptions
from geode import pyBunch
from geode import pyDate
from geode.Utils import stationID

COMPLETION = 0.5
INTERVAL   = 120


class StationException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class StationCollectionException(StationException):
    pass


class Station(object):

    def __init__(self, cnn, NetworkCode, StationCode, dates, StationAlias=None):
        """
        Station object to manage metadata and APRs for GAMIT run. Class allows overriding StationAlias but now the
        stations have a default alias if there is a station duplicate. StationAlias override left for backwards
        compatibility but it should not be used.
        """

        self.NetworkCode  = NetworkCode
        self.StationCode  = StationCode
        self.netstn = stationID(self)

        if StationAlias is None:
            rs = cnn.query_float('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' '
                                 'AND "StationCode" = \'%s\' AND alias IS NOT NULL'
                                 % (NetworkCode, StationCode), as_dict=True)
            if len(rs):
                self.StationAlias = rs[0]['alias']
            else:
                # if no record, then Alias = StationCode
                self.StationAlias = StationCode
        else:
            self.StationAlias = StationAlias

        self.record       = None
        self.etm          = None
        self.StationInfo  = None
        self.lat          = None
        self.lon          = None
        self.height       = None
        self.X            = None
        self.Y            = None
        self.Z            = None
        self.otl_H        = None

        rs = cnn.query_float('SELECT * FROM stations WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                             % (NetworkCode, StationCode), as_dict=True)

        if len(rs) != 0:
            self.record = pyBunch.Bunch().fromDict(rs[0])

            self.otl_H  = self.record.Harpos_coeff_otl
            self.lat    = self.record.lat
            self.lon    = self.record.lon
            self.height = self.record.height
            self.X      = self.record.auto_x
            self.Y      = self.record.auto_y
            self.Z      = self.record.auto_z

            # get the available dates for the station (RINEX files with conditions to be processed)
            rs = cnn.query(
                'SELECT "ObservationYear" as y, "ObservationDOY" as d, "Completion" as c FROM rinex_proc '
                'WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND '
                '("ObservationYear", "ObservationDOY") BETWEEN (%s) AND (%s) AND "Interval" <= %i'
                % (NetworkCode, StationCode, dates[0].yyyy() + ', ' + dates[0].ddd(),
                   dates[1].yyyy() + ', ' + dates[1].ddd(), INTERVAL))

            rs = rs.dictresult()

            self.good_rinex = [pyDate.Date(year=r['y'], doy=r['d']) for r in rs if r['c'] >= COMPLETION]

            # create a set of the missing days
            good_rinex = {d.mjd for d in self.good_rinex}

            self.missing_rinex = [pyDate.Date(mjd=d) for d in range(dates[0].mjd, dates[1].mjd+1)
                                  if d not in good_rinex]

            solution_options = SolutionOptions()
            solution_options.solution_type = SolutionType.PPP

            config = EtmConfig(NetworkCode, StationCode, cnn=cnn,
                               solution_options=solution_options, silent=True)


            self.etm = EtmEngine(config=config, cnn=cnn)
            self.etm.run_adjustment(cnn=cnn, try_save_to_db=True, try_loading_db=True)

            self.StationInfo = StationInfo(cnn, NetworkCode, StationCode)

            valid_rinex = []
            for date in self.good_rinex:
                try:
                    self.StationInfo.check_coverage(date)
                    valid_rinex.append(date)
                except StationInfoNoRecordFound:
                    tqdm.write(f'    WARNING: Observations for day {date.yyyymmdd()} ({date.yyyyddd()}) '
                               f'have a gap or no station information. Day will be removed from the processing '
                               f'to avoid a GAMIT FATAL during the run.')

            self.good_rinex = valid_rinex

            # DDG: report RINEX files with Completion < 0.5
            rs = [r for r in rs if r['c'] < COMPLETION]

            if len(rs):
                tqdm.write('    WARNING: The requested date interval has %i days with < 50%% of observations. '
                           'These days will not be processed.' % len(rs))
        else:
            raise ValueError('Specified station %s.%s could not be found' % (NetworkCode, StationCode))

    def check_gamit_soln(self, cnn, project, date):
        """
        Function to check if a gamit solution exists for this station, project and date
        :param cnn: database connection
        :param project: name of the project to search
        :param date: date to check if a solution exists
        :return: True if solution exists, otherwise False
        """
        soln = cnn.query_float('SELECT * FROM gamit_soln WHERE "Project" = \'%s\' AND "Year" = %i AND '
                               '"DOY" = %i AND "NetworkCode" = \'%s\' AND "StationCode" = \'%s\''
                               % (project, date.year, date.doy, self.NetworkCode, self.StationCode))
        
        return bool(len(soln))

    # DDG: deprecated, aliases are now fixed and kept constant. Left for backwards compatibility
    def generate_alias(self):
        self.StationAlias = self.id_generator()

    # DDG: deprecated, aliases are now fixed and kept constant. Left for backwards compatibility
    @staticmethod
    def id_generator(size=4, chars=string.ascii_lowercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def __eq__(self, station):

        if not isinstance(station, Station):
            raise StationException('type: ' + str(type(station)) + ' invalid. Can only compare pyStation.Station')

        return self.NetworkCode == station.NetworkCode and self.StationCode == station.StationCode

    def __hash__(self):
        # to make the object hashable
        return hash(str(self))

    def __str__(self):
        return self.NetworkCode + '.' + self.StationCode

    def __repr__(self):
        return 'pyStation.Station(' + str(self) + ', ' + self.StationAlias + ')'

    def __iter__(self):
        return self


class StationInstance(object):

    def __init__(self, cnn, archive, station: Station, date, GamitConfig, is_tie=False):

        self.NetworkCode  = station.NetworkCode
        self.StationCode  = station.StationCode
        self.StationAlias = station.StationAlias
        self.lat          = station.record.lat
        self.lon          = station.record.lon
        self.height       = station.record.height
        self.X            = station.record.auto_x
        self.Y            = station.record.auto_y
        self.Z            = station.record.auto_z
        self.otl_H        = station.otl_H
        # save in the station instance if it was intended as a tie station or not
        self.is_tie       = is_tie

        # save the station information as text
        try:
            self.StationInfo = StationInfo(
                cnn, station.NetworkCode, station.StationCode, date).return_stninfo()

        except StationInfoHeightCodeNotFound as e:
            tqdm.write(' -- WARNING: ' + str(e) + '. Antenna height will be used as is and GAMIT may produce a fatal.')
            self.StationInfo = StationInfo(cnn, station.NetworkCode,
                                           station.StationCode, date).return_stninfo(no_dharp_translate=True)

        self.date         = date  # type: pyDate.Date
        self.Archive_path = GamitConfig.archive_path

        # get the APR and sigmas for this date
        station.etm.config.modeling.sigma_floor_h = float(GamitConfig.gamitopt['sigma_floor_h'])
        station.etm.config.modeling.sigma_floor_v = float(GamitConfig.gamitopt['sigma_floor_v'])
        etm_output = station.etm.get_position(self.date, EtmSolutionType.OBSERVATION)

        self.apr = [item[0] for item in etm_output['position']]
        self.source = etm_output['source']
        self.sigmas = [item[0] for item in etm_output['sigmas']]
        self.window = station.etm.query_jump(self.date)

        # rinex file
        self.archive_file = archive.build_rinex_path(self.NetworkCode, self.StationCode,
                                                     self.date.year, self.date.doy)

        # DDG: force RINEX 2 filenames even with RINEX 3 data
        self.filename = self.StationAlias + self.date.ddd() + '0.' + self.date.yyyy()[2:4] + 'd.Z'

        # save some information for debugging purposes
        rs = cnn.query_float('SELECT * FROM ppp_soln WHERE "NetworkCode" = \'%s\' AND "StationCode" = \'%s\' AND '
                             '"Year" = %s AND "DOY" = %s'
                             % (self.NetworkCode, self.StationCode, self.date.yyyy(), self.date.ddd()), as_dict=True)

        self.ppp = rs[0] if len(rs) > 0 else None

    def GetRinexFilename(self):

        return {'NetworkCode' : self.NetworkCode,
                'StationCode' : self.StationCode,
                'StationAlias': self.StationAlias,
                'source'      : os.path.join(self.Archive_path, self.archive_file),
                'destiny'     : self.filename,
                'lat'         : self.lat,
                'lon'         : self.lon,
                'height'      : self.height,
                'jump'        : self.window,
                'is_tie'      : self.is_tie}

    def GetApr(self):
        x = self.apr

        return ' ' + self.StationAlias.upper() + '_GPS ' + '{:12.3f}'.format(x[0]) + ' ' + '{:12.3f}'.format(
            x[1]) + ' ' + '{:12.3f}'.format(x[2]) + ' 0.000 0.000 0.000 ' + '{:8.4f}'.format(
            self.date.fyear)

    def GetSittbl(self):
        s = self.sigmas

        return (self.StationAlias.upper() + ' ' + self.StationAlias.upper() +
                '_GPS' + 'NNN'.rjust(8) +
                '    {:.5}'.format('%5.3f' % (s[0])) + ' ' +
                '{:.5}'.format('%5.3f' % (s[1])) + ' ' +
                '{:.5}'.format('%5.3f' % (s[2])))

    def DebugCoord(self):
        x = self.apr
        s = self.sigmas

        if self.ppp is not None:
            return '%s %s_GPS %8.3f %8.3f %8.3f %14.3f %14.3f %14.3f %8.3f %8.3f %8.3f %8.4f %s' % \
                   (self.StationAlias.upper(), self.StationAlias.upper(),
                    self.ppp['X'] - self.X,
                    self.ppp['Y'] - self.Y,
                    self.ppp['Z'] - self.Z,
                    x[0], x[1], x[2],
                    s[0], s[1], s[2],
                    self.date.fyear, self.source)
        else:
            return '%s %s_GPS %-26s %14.3f %14.3f %14.3f %8.3f %8.3f %8.3f %8.4f %s' % \
                   (self.StationAlias.upper(), self.StationAlias.upper(), 'NO PPP COORDINATE',
                    x[0], x[1], x[2],
                    s[0], s[1], s[2],
                    self.date.fyear, self.source)

    def GetStationInformation(self):

        return self.StationInfo.replace(self.StationCode.upper(), self.StationAlias.upper())


class StationCollection(list):
    """
    StationCollection object accumulates Station objects verifying there is no collision in StationCodes
    It is essentially a list with an overloaded append method that triggers verification of the StationCodes and
    makes changes to StationAliases as needed
    """
    def __init__(self, stations=None):
        super(StationCollection, self).__init__()
        if stations:
            self.append(stations)
            
    def labels_array(self):
        # pyNetwork already filters to active stations, so check isn't needed...
      	# return np.array([stn.netstn for stn in self if date in stn.good_rinex])
        return np.array([stn.netstn for stn in self])

    def append(self, station):
        # DDG: deprecated, aliases are now fixed and kept constant
        # DDG: removed arg check_aliases=True

        if not (isinstance(station, Station) or isinstance(station, list)):
            raise StationException('type: ' + str(type(station)) +
                                     ' invalid. Can only append Station objects or lists of Station objects')

        if isinstance(station, Station):
            station = [station]

        for stn in station:
            # check that the incoming stations is not already in the list
            if not self.ismember(stn):
                # DDG: deprecated, aliases are now fixed and kept constant
                # verify the incoming Station against all StationCodes
                # if check_aliases:
                #     self.check_station_codes(stn)
                super(StationCollection, self).append(stn)

    # DDG: deprecated, aliases are now fixed and kept constant
    # def check_station_codes(self, station):
    #
    #     for stn in self:
    #         if stn.NetworkCode != station.NetworkCode and \
    #            stn.StationCode == station.StationCode:
    #             # duplicate StationCode (different Network), produce Alias
    #             unique = False
    #             while not unique:
    #                 station.generate_alias()
    #                 # compare again to make sure this name is unique
    #                 unique = self.compare_aliases(station)
    #
    #             tqdm.write('    Station renamed due to name conflict: ' + station.StationAlias)

    def compare_aliases(self, station):

        # make sure alias does not exists as alias and station code
        for stn in self:
            if stn != station and station.StationAlias in (stn.StationAlias, stn.StationCode):
                # not unique!
                return False

        return True

    def get_active_stations(self, date, check_aliases=False):
        """
        create a collection with the stations that actually have observations for a given day
        by default, the aliases are leaved untouched
        :param date: to check if observations are available or not
        :param check_aliases: boolean, check the aliases of the stations and change them if necessary
        :return: a collection with the stations that have observations
        """
        collection = StationCollection()
        for stn in self:
            if date in stn.good_rinex:
                collection.append(stn)

        return collection

    def get_active_coordinates(self, date):
        """
        obtain a numpy array of the coordinates for the active stations in the collection
        :param date:
        :return: numpy array
        """
        return np.array([[stn.X, stn.Y, stn.Z] for stn in self if date in stn.good_rinex])

    # DDG: deprecated, aliases are now fixed and kept constant
    # def replace_alias(self, stations, aliases=None):
    #     """
    #     replace alias for station(s) provided in the list
    #     :param stations: can be a station object, a list of station objects, a StationCollection or a list of strings
    #     :param aliases: new aliases to apply to the stations (must be string or list of strings). If it's not provided
    #     then aliases are pulled from the stations objects (in which case, stations must be a Station object)
    #     :return: None
    #     """
    #
    #     if isinstance(stations, StationCollection) or isinstance(stations, list):
    #         if isinstance(stations, list):
    #             for stn in stations:
    #                 if not (isinstance(stn, Station) or isinstance(stn, str)):
    #                     raise pyStationException('type: ' + str(type(stn)) +
    #                                              ' invalid. Can only pass Station or String objects.')
    #
    #         if isinstance(aliases, list):
    #             if len(stations) != len(aliases):
    #                 raise pyStationCollectionException('Length of stations and aliases arguments must match')
    #
    #         elif isinstance(aliases, str) and len(stations) > 1:
    #             raise pyStationCollectionException('More than one station for a single alias string')
    #
    #         elif aliases is None:
    #             aliases = [stn.StationAlias for stn in stations]
    #
    #         elif not (isinstance(aliases, str) or isinstance(aliases, list)):
    #             raise pyStationException('type: ' + str(type(aliases)) +
    #                                      ' invalid. Can only pass List or String objects.')
    #
    #     elif isinstance(stations, Station) or isinstance(stations, str):
    #         if isinstance(aliases, list) and len(aliases) > 1:
    #             raise pyStationCollectionException('More than one alias for a single station object')
    #         elif aliases is None:
    #             if isinstance(stations, Station):
    #                 aliases = stations.StationAlias
    #             else:
    #                 raise pyStationException('No aliases provided. Argument stations must be a Station object')
    #         elif not (isinstance(aliases, list) or isinstance(aliases, str)):
    #             raise pyStationException('type: ' + str(type(aliases)) +
    #                                      ' invalid. Can only pass List or String objects.')
    #
    #     else:
    #         raise pyStationException('type: ' + str(type(stations)) +
    #                                  ' invalid. Can only pass List, StationCollection or String objects.')
    #
    #     if isinstance(aliases, str):
    #         aliases = [aliases]
    #
    #     if isinstance(stations, str) or isinstance(stations, Station):
    #         stations = [stations]
    #
    #     for stn, alias in zip(stations, aliases):
    #         if self[stn].StationCode == alias:
    #             # if alias == StationCode continue to next one
    #             continue
    #         try:
    #             self[stn].StationAlias = alias
    #             # make sure there is no alias overlap
    #             while not self.compare_aliases(self[stn]):
    #                 self[stn].generate_alias()
    #
    #         except pyStationCollectionException as e:
    #             print(str(e))
    #             pass  # not found in collection

    def ismember(self, station):
        """
        determines if a station is part of the collection or not
        :param station: station object or string
        :return: boolean
        """
        try:
            _ = self[station]
            return True
        except StationCollectionException:
            return False

    def __getitem__(self, item):
        """
        return the station object that matches a given another station object or net.stn string
        :param item: station object or string
        :return: station object
        """
        if isinstance(item, str):
            for stn in self:
                if stn.netstn == item.lower():
                    return stn
            raise StationCollectionException('Requested station code ' + item.lower() + ' could not be found')

        elif isinstance(item, Station):
            for stn in self:
                if stn.netstn == item.netstn:
                    return stn
            raise StationCollectionException('Requested station code ' + item.netstn + ' could not be found')

        else:
            raise StationException('type: ' + str(type(item)) + ' invalid. Can only pass Station or String objects.')

    def __contains__(self, item):
        return self.ismember(item)