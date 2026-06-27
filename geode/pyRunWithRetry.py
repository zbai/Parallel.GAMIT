"""
Project: Geodetic Database Engine (GeoDE)
Date: 02/16/2017
Author: Demian D. Gomez
"""

import os
import platform
import subprocess
import time

# app
from . import pyEvents


class RunCommandWithRetryExeception(Exception):
    def __init__(self, value):
        self.value = value
        self.event = pyEvents.Event(Description = value,
                                    EventType   = 'error',
                                    module      = type(self).__name__)
    def __str__(self):
        return str(self.value)


class RunCommand:
    def __init__(self, command, time_out, cwd=None, cat_file=None):
        self.cmd      = command
        self.time_out = time_out
        # Evaluate cwd at instantiation time, not at import time.
        self.cwd      = cwd or os.getcwd()
        self.cat_file = cat_file

    def run_shell(self):
        retry = 0
        while True:
            cat_fh = None
            try:
                if self.cat_file:
                    cat_fh = open(os.path.join(self.cwd, self.cat_file), 'r',
                                  encoding='utf-8', errors='ignore')
                    stdin = cat_fh
                else:
                    # Use DEVNULL so the subprocess never blocks waiting for
                    # interactive input, regardless of how the parent's stdin
                    # is connected (TTY, pipe, /dev/null in batch jobs, etc.).
                    stdin = subprocess.DEVNULL

                result = subprocess.run(
                    self.cmd.split(),
                    stdin   = stdin,
                    stdout  = subprocess.PIPE,
                    stderr  = subprocess.PIPE,
                    cwd     = self.cwd,
                    timeout = self.time_out,
                    encoding = 'utf-8',
                    errors   = 'ignore',
                )
                stdout = result.stdout
                # Strip non-ASCII characters from stderr (preserve original behaviour)
                stderr = ''.join(c if ord(c) < 128 else ' '
                                 for c in (result.stderr or ''))
                return stdout, stderr

            except subprocess.TimeoutExpired:
                if retry < 2:
                    retry += 1
                    continue
                raise RunCommandWithRetryExeception(
                    'Error in RunCommand.run_shell -- (%s): Timeout after 3 retries'
                    % self.cmd)

            except OSError as e:
                if '[Errno 35]' in str(e):
                    if retry < 2:
                        retry += 1
                        time.sleep(0.5)
                        continue
                    raise OSError(str(e) + ' after 3 retries on node: '
                                  + platform.node())
                raise

            finally:
                if cat_fh is not None:
                    cat_fh.close()
