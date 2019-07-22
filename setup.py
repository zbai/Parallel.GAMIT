from setuptools import setup
import logging
import datetime as dt
logger = logging.getLogger('gpys.setup')
stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
form = logging.Formatter('%(asctime)-15s %(name)-25s %(levelname)s - %(threadName)s %(message)s',
                         '%Y-%m-%d %H:%M:%S')
stream.setFormatter(form)
logger.addHandler(stream)
date = dt.date.today().strftime('%y%m%d')

if __name__ == '__main__':
    setup(
        name='gpys',
        version=f'0.0.post{date}',
        packages=['gpys', 'archive'],
        url='https://github.com/demiangomez/Parallel.GAMIT',
        license='',
        author='Demian Gomez & Peter Matheny',
        author_email='',
        description='',
        install_requires=['dispy', 'psycopg2'],
        entry_points={'console_scripts': ['archive = gpys.archive:main']}
    )
