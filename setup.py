from setuptools import setup

setup(
    name='gpys',
    version='0.0.0',
    packages=['gpys', 'gpys.ArchiveService', 'gpys.test'],
    url='https://github.com/demiangomez/Parallel.GAMIT',
    license='',
    author='Demian Gomez & Peter Matheny',
    author_email='',
    description='',
    install_requires=['ray', 'numpy', 'tqdm', 'psycopg2'],
)
