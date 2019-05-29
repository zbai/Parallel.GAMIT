"""
The testing script to help debug functions and classes.  The classes are split up into the script that they call.
gpys is implicitly tested via the ArchiveService routines.
"""
import unittest
from gpys.archive import main
import sys
import shutil
import subprocess
import os


class ArchiveTest(unittest.TestCase):

    def setUp(self) -> None:
        shutil.rmtree('/tmp/pfsdir')
        shutil.rmtree('/tmp/tmpdir')
        # Start building the run environment.
        os.makedirs('/tmp/pfsdir')
        os.makedirs('/tmp/pfsdir/repository/data_in')
        os.makedirs('/tmp/pfsdir/archive')
        os.makedirs('/tmp/pfsdir/orbits/igs')
        os.makedirs('/tmp/pfsdir/orbits/igs/1980')
        os.makedirs('/tmp/pfsdir/orbits/brdc')
        os.makedirs('/tmp/pfsdir/orbits/brdc/2017')
        os.makedirs('/tmp/tmpdir')
        os.makedirs('/tmp/tmpdir/dependencies')
        os.makedirs('/tmp/tmpdir/dependencies/GPSPACE')
        os.makedirs('/tmp/tmpdir/dependencies/gg')
        os.makedirs('/tmp/tmpdir/dependencies/gg/tables')
        shutil.copy('test_files/brdc3560.17n', '/tmp/pfsdir/orbits/brdc/2017')
        shutil.copy('test_files/inmn3560.17d.Z', '/tmp/pfsdir/repository/data_in')
        shutil.copy('test_files/igr19805.sp3', '/tmp/pfsdir/orbits/igs/1980')
        shutil.copy('test_files/igr19806.sp3', '/tmp/pfsdir/orbits/igs/1980')
        shutil.copy('test_files/igr19805.clk', '/tmp/pfsdir/orbits/igs/1980')
        shutil.copy('test_files/igr19806.clk', '/tmp/pfsdir/orbits/igs/1980')
        shutil.copy('test_files/igr19806.clk', '/tmp/pfsdir/orbits/igs/1980')
        shutil.copy('test_files/gpsppp.svb_gnss_yrly', '/tmp/tmpdir/dependencies/GPSPACE')
        shutil.copy('test_files/igs14_1992_plus.atx', '/tmp/tmpdir/dependencies/gg/tables')

    def test_serial(self):
        # First create a simulated shared archive/repository/orbits folder like the PFS on the HPC.
        # Here we'll keep it in the /tmp/gpys directory as defined in debug.cfg
        # First clear out the data from previous runs.

        sys.argv = ['archive',
                    '-v',
                    '-config',
                    'test_files/debug_serial.cfg']
        with self.assertRaisesRegex(SystemExit, '0'):
            main()

    def test_parallel(self):
        # First create a simulated shared archive/repository/orbits folder like the PFS on the HPC.
        # Here we'll keep it in the /tmp/gpys directory as defined in debug.cfg
        commands = ['rm -rf /tmp/tmpdir',
                    'rm -rf /tmp/pfsdir',
                    'mkdir /tmp/tmpdir',
                    'mkdir /tmp/tmpdir/dependencies',
                    'mkdir /tmp/tmpdir/dependencies/GPSPACE',
                    'mkdir /tmp/tmpdir/dependencies/gg',
                    'mkdir /tmp/tmpdir/dependencies/gg/tables',
                    'mkdir /tmp/pfsdir',
                    'mkdir /tmp/pfsdir/repository',
                    'mkdir /tmp/pfsdir/archive',
                    'mkdir /tmp/pfsdir/orbits',
                    'mkdir /tmp/pfsdir/repository/data_in',
                    'mkdir /tmp/pfsdir/orbits/igs',
                    'mkdir /tmp/pfsdir/orbits/igs/1980',
                    'mkdir /tmp/pfsdir/orbits/brdc',
                    'mkdir /tmp/pfsdir/orbits/brdc/2017']
        for c in commands:
            subprocess.run(['ssh', 'peter@192.168.1.17', f'{c}'])

        files = {'test_files/inmn3560.17d.Z': '/tmp/pfsdir/repository/data_in',
                 'test_files/brdc3560.17n': '/tmp/pfsdir/orbits/brdc/2017',
                 'test_files/igr19805.sp3': '/tmp/pfsdir/orbits/igs/1980',
                 'test_files/igr19805.clk': '/tmp/pfsdir/orbits/igs/1980',
                 'test_files/igr19806.sp3': '/tmp/pfsdir/orbits/igs/1980',
                 'test_files/igr19806.clk': '/tmp/pfsdir/orbits/igs/1980',
                 'test_files/gpsppp.svb_gnss_yrly': '/tmp/tmpdir/dependencies/GPSPACE',
                 'test_files/igs14_1992_plus.atx': '/tmp/tmpdir/dependencies/gg/tables'}

        for f, d in files.items():
            subprocess.run(['scp', f, f'peter@192.168.1.17:{d}'], check=True)

        with open('test_files/debug_serial.cfg') as inp:
            with open('test_files/debug_parallel.cfg', 'w') as output:
                original = inp.read()
                new = original.replace('parallel = False', 'parallel = True')
                output.write(new)

        sys.argv = ['archive',
                    '-v',
                    '-config',
                    'test_files/debug_parallel.cfg']
        with self.assertRaisesRegex(SystemExit, '0'):
            main()


if __name__ == '__main__':
    unittest.main()
