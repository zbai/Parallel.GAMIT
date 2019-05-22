from setuptools import setup

setup(
    name='gpys',
    version='0.0.0',
    packages=['gpys', 'gpys.archive'],
    url='https://github.com/demiangomez/Parallel.GAMIT',
    license='',
    author='Demian Gomez & Peter Matheny',
    author_email='',
    description='',
    install_requires=['dispy', 'numpy', 'tqdm', 'psycopg2'],
    entry_points={'console_scripts': ['archive = gpys.archive:main']}
)
