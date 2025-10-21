from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("geode")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

__all__ = [
    'cluster',
    'network',
    'plots',
    'pyRinexName',
    'Utils',
    'pyJobServer',
    'pyParseAntex',
    'pyStatic1d',
    'snxParse',
    'pyLeastSquares',
    'pyProducts',
    'metadata.station_info',
    'dbConnection',
    'pyDate',
    'pyOTL',
    'pyRinex',
    'pyTerminal',
    'pyArchiveStruct',
    'pyOkada',
    'ConvertRaw',
    'pyETM',
    'pyOptions',
    'pyRunWithRetry',
    'pyBunch',
    'pyEvents',
    'pyPPP',
    'pyProducts',
    'pyZTD',
    'pyStack',
    'pyGamitConfig',
    'pyGamitSession',
    'pyGamitTask',
    'pyGlobkTask',
    'pyParseZTD',
    'pyStation'
]

from importlib import import_module

for _name in __all__:
    try:
        globals()[_name] = import_module(f'.{_name}', __name__)
    except Exception:
        pass
