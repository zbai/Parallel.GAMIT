=======
archive
=======

Required directory structure:

Used to add RINEX files to the archive.

.. py:function:: main() -> None:

    :return:

    #. Parse the commandline arguments
    #. Load the options file into an object.
    #. Move files from retry into the data_in folder.
    #. Submit jobs to the cluster.
    #. First submit using cluster.submit, then add it to [jobs] so we can use the results later.  Once that is
       done we lock the MainThread using jobs_cond.acquire()
    #. Wait for the jobs to be submitted.
