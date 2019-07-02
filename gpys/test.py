"""
The testing script to help debug functions and classes.  The classes are split up into the script that they call.
gpys is implicitly tested via the archive routines.
Current required directory structure:
-----------------------
test.py
test_files/
├── pfsdir
│   ├── archive
│   ├── orbits
│   │   ├── brdc
│   │   │   └── <year>
│   │   └── igs
│   │       └── <gpsweek>
│   └── repository
│       └── data_in
└── tmpdir
    └── dependencies
        ├── crx2rnx*
        ├── gpsppp.svb_gnss_yrly
        ├── gpsppp.trf
        ├── igs14_1992_plus.atx
        ├── ppp*
        └── teqc*
-----------------------
TODO: Copy jobs that failed to /tmp/errors
TODO: Add logger
TODO: Add function that does all the file moving.
TODO: Add better docstrings throughout.
"""
import gpys
import unittest
from gpys.archive import main
import shutil
import subprocess
import os
from pathlib import Path
import sys
year, yr, doy, gpsweek, nextgpsweek, orbit, station = '2017', '17', '355', '19804', '19805', 'igr', 'inmn'


def singlefile() -> None:
    global year, yr, doy, gpsweek, nextgpsweek, orbit, station
    srcdir = 'test_files/pfsdir'
    dstdir = '/tmp/pfsdir'
    mdir = ['/tmp/pfsdir', '/tmp/pfsdir/repository/data_in', '/tmp/pfsdir/archive',
            '/tmp/pfsdir/orbits/igs', f'/tmp/pfsdir/orbits/igs/{gpsweek[:-1]}',
            '/tmp/pfsdir/orbits/brdc', f'/tmp/pfsdir/orbits/brdc/{year}']
    if gpsweek[:-1] != nextgpsweek[:-1]:
        mdir.append(f'/tmp/pfsdir/orbits/igs/{nextgpsweek[:-1]}')
    for d in mdir:
        os.makedirs(d)
    cfile = {f'{srcdir}/orbits/brdc/{year}/brdc{doy}0.{yr}n.Z':
             f'/tmp/pfsdir/orbits/brdc/{year}',
             f'{srcdir}/repository/data_in/{station}{doy}0.{yr}d.Z':
             '/tmp/pfsdir/repository/data_in',
             f'{srcdir}/orbits/igs/{gpsweek[:-1]}/{orbit}{gpsweek}.sp3.Z':
             f'{dstdir}/orbits/igs/{gpsweek[:-1]}',
             f'{srcdir}/orbits/igs/{gpsweek[:-1]}/{orbit}{gpsweek}.clk.Z':
             f'{dstdir}/orbits/igs/{gpsweek[:-1]}',
             f'{srcdir}/orbits/igs/{nextgpsweek[:-1]}/{orbit}{nextgpsweek}.sp3.Z':
             f'{dstdir}/orbits/igs/{nextgpsweek[:-1]}',
             f'{srcdir}/orbits/igs/{nextgpsweek[:-1]}/{orbit}{nextgpsweek}.clk.Z':
             f'{dstdir}/orbits/igs/{nextgpsweek[:-1]}'}
    for f, d in cfile.items():
        shutil.copy(f, d)


class ArchiveTest(unittest.TestCase):

    def setUp(self) -> None:
        """
        First delete the existing working directories, then copy the tmpdir into /tmp since it's required for all runs.
        :return:
        """
        rmdir = ['/tmp/pfsdir', '/tmp/tmpdir']
        for rm in rmdir:
            if Path(rm).exists():
                shutil.rmtree(rm)
        shutil.copytree('test_files/tmpdir_macosx/tmpdir', '/tmp/tmpdir', ignore_dangling_symlinks=True)
        opts = {'database': 'gnss_data',
                'hostname': 'localhost',
                'username': 'postgres',
                'password': ''}
        setupcnn = gpys.Connection(opts)


    def test_serial_singlefile(self) -> None:
        """
        First create a simulated shared archive/repository/orbits folder like the PFS on the HPC.
        Here we'll keep it in the /tmp/gpys directory as defined in debug.cfg
        Start building the run environment.
        :return:
        """
        singlefile()
        sys.argv = ['archive',
                    '-v',
                    '-config',
                    'test_files/debug_serial.cfg']
        nexcept = 0
        with self.assertLogs(level='DEBUG') as cm:
            with self.assertRaisesRegex(SystemExit, '0'):
                main()
        for line in cm.output:
            if 'ERROR' in line.split(':'):
                nexcept += 1
        self.assertEqual(0, nexcept)

    def test_parallel_singlefile(self) -> None:
        """
        First create a simulated shared archive/repository/orbits folder like the PFS on the HPC.
        Here we'll keep it in the /tmp/gpys directory as defined in debug.cfg
        :return:
        """
        singlefile()
        commands = ['rm -rf /tmp/tmpdir',
                    'rm -rf /tmp/pfsdir']
        for c in commands:
            subprocess.run(['ssh', 'peter@192.168.1.17', f'{c}'])
        subprocess.run(['rsync', '-avz',
                        '/Users/pmatheny/PycharmProjects/Parallel.GAMIT/gpys/test_files/tmpdir_ubuntu/tmpdir',
                        'peter@192.168.1.17::tmp'])
        subprocess.run(['rsync', '-avz', '/tmp/pfsdir', 'peter@192.168.1.17::tmp'])
        with open('test_files/debug_serial.cfg') as inp:
            with open('test_files/debug_parallel.cfg', 'w') as output:
                original = inp.read()
                new = original.replace('parallel = False', 'parallel = True')
                output.write(new)
        sys.argv = ['archive',
                    '-v',
                    '-config',
                    'test_files/debug_parallel.cfg']
        nexcept = 0
        with self.assertLogs(level='DEBUG') as cm:
            with self.assertRaisesRegex(SystemExit, '0'):
                main()
        for line in cm.output:
            if 'ERROR' in line.split(':'):
                print(line)
                nexcept += 1
        self.assertEqual(nexcept, 0)

    def test_serial_multifile(self) -> None:
        """
        First create a simulated shared archive/repository/orbits folder like the PFS on the HPC.
        Here we'll keep it in the /tmp/gpys directory as defined in debug.cfg
        First clear out the data from previous runs.
        Start building the run environment.
        :return:
        """

        shutil.copytree('test_files/pfsdir', '/tmp/pfsdir')
        sys.argv = ['archive',
                    '-v',
                    '-config',
                    'test_files/debug_serial.cfg']
        nexcept = 0
        with self.assertLogs(level='DEBUG') as cm:
            with self.assertRaisesRegex(SystemExit, '0'):
                main()
        for line in cm.output:
            if 'ERROR' in line.split(':'):
                print(line)
                nexcept += 1
        self.assertEqual(nexcept, 0)

    def test_parallel_multifile(self) -> None:
        """
        Same as the test serial multifile but after creating the directories in /tmp we rsync them over to the desktop.
        Start building the run environment.
        :return:
        """
        shutil.copytree('test_files/pfsdir', '/tmp/pfsdir')
        subprocess.run(['rsync', '-avz',
                        '/Users/pmatheny/PycharmProjects/Parallel.GAMIT/gpys/test_files/tmpdir_ubuntu/tmpdir',
                        'peter@192.168.1.17::tmp'])
        subprocess.run(['rsync', '-avz', '/tmp/pfsdir', 'peter@192.168.1.17::tmp'])
        with open('test_files/debug_serial.cfg') as inp:
            with open('test_files/debug_parallel.cfg', 'w') as output:
                original = inp.read()
                new = original.replace('parallel = False', 'parallel = True')
                output.write(new)
        sys.argv = ['archive',
                    '-v',
                    '-config',
                    'test_files/debug_parallel.cfg']
        nexcept = 0
        with self.assertLogs(level='DEBUG') as cm:
            with self.assertRaisesRegex(SystemExit, '0'):
                main()
        for line in cm.output:
            if 'ERROR' in line.split(':'):
                print(line)
                nexcept += 1
        self.assertEqual(nexcept, 0)


if __name__ == '__main__':
    unittest.main()
