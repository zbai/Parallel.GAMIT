from setuptools import setup
import logging
import shutil
import os

logger = logging.getLogger('gpys.setup')
stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
form = logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s - %(threadName)s %(message)s',
                         '%Y-%m-%d %H:%M:%S')
stream.setFormatter(form)
logger.addHandler(stream)


def find_reqs() -> None:
    """
    Finds instances of subprocess.run within the gpys package and then checks to make sure they are located on the path.
    :return:
    """
    global logger
    for script in ['gpys/__init__.py', 'gpys/archive/__init__.py']:
        with open(script) as f:
            code = f.read().splitlines()
        progs = set([x.split('\'')[1] for x in code if 'subprocess.run' in x])
        for p in progs:
            fpath = shutil.which(p, mode=os.X_OK)
            if not fpath:
                logger.error(f'{p} not found or not executable.')
            else:
                logger.info(f'{p} found at {fpath}')


if __name__ == '__main__':
    find_reqs()
    setup(
        name = 'gpys',
        version = '0.0.0',
        packages = ['gpys', 'gpys.archive'],
        url = 'https://github.com/demiangomez/Parallel.GAMIT',
        license = '',
        author = 'Demian Gomez & Peter Matheny',
        author_email = '',
        description = '',
        install_requires = ['dispy', 'psycopg2'],
        entry_points = {'console_scripts': ['archive = gpys.archive:main']}
    )
