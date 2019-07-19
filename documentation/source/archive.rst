=======
archive
=======

Required directory structure:

Used to add RINEX files to the archive.

.. todo:: Read parse_data_in output into the postgres table.
.. todo:: Add else statements to the end of try blocks.
.. todo:: Sort the data_in list so that it does the days in order.
.. todo:: Add some error checking to the subprocess runs.
.. todo:: Delete completed jobs.
.. todo:: Add percentage completion to the config file?
.. todo:: Abstract all the file names.
.. todo:: Hardwire all the paths and then create another class/module/function that checks/installs/updates the executables
          and the path.

.. py:function:: node_setup() -> int

    :return:

    Placeholder if in the future using a setup function is deemed to be useful.

.. py:function:: callback(job)

    :param dispy.DispyJob job: An instance of dispy.DispyJob

    Simple callback function that helps reduce the amount of submissions.  Typing hints don't work here.

.. py:function:: parse_data_in(filepath, config, n) -> dict

    :param str filepath: Path to the RINEX we're working on.
    :param config: gpys.ReadOptions object containing parameters from the .cfg file.
    :param int n: The job number.
    :return: Dictionary containing the parameters used during the run as well as the results with the following keys:

             **ofile**: (*pathlib.Path*) Path object of the original file in data_in

             **file**: Path object to the location of the file in the production folder

             **start_date**: (*datetime.date*) Date of the RINEX as determined by TEQC +qc

             **name**: (*str*) 4-character name of the station as determined by TEQC +qc

             **orbit**: (*pathlib.Path*) Path object for the broadcast orbit file.

             **completion**: (*decimal.Decimal*) The percentage completion as a string between 0-100 as reported by TEQC

             **teqc_xyz**: (*list[decimal.Decimal]*) 3-element list with strings representing the X, Y and Z  coordinates found by TEQC

             **sp3path**: (*pathlib.Path*) Path object to the precise orbit file in the production folder for the current
             day of the RINEX.

             **nextsp3path**: (*pathlib.Path*) Same as sp3path but for the next day's precise orbit file.

             **ppp_input_string**: (*str*) The string read into PPP via STDIN.

             **pppcoords**: (*list[decimal.Decimal]*) 3-element list containing float values for the ITRF X, Y and Z.

             **latlonh**: (*list[decimal.Decimal]*) 3-element list containing float values for the ITRF latitude, longitude and height.

             **pppref**: (*str*) String representing the ITRF datum as reported by the PPP summary file.

             **completed**: (*bool*) Bool that is False if any exception was raised during the operation of this function.

             **runtime**: (*float*) Float representing how long it took to complete the program.

             **ObservationFYear**: (*float*) Decimal year format.

             **ObservationSTime**: (*datetime.datetime*) Start of the observation.

             **ObservationETime**: (*datetime.datetime*) End of observation.

             **ReceiverType**: (*str*) The type of receiver.

             **ReceiverSerial**: (*str*) The receiver serial.

             **ReceiverFw**: (*str*) The receiver firmware.

             **AntennaType**: (*str*) The type of antenna.

             **AntennaSerial**: (*str*) The antenna serial.

             **AntennaDome**: (*str*) The antenna dome code.

             **Interval**: (*decimal.Decimal*) The sampling rate.

    Runs PPP on a RINEX and either send it to the rejected folder, retry folder, lock it or add it to an existing
    station record.  Most parameters from the run are stored in the rinex_dict variable and returned at the end of the
    function.  The function should run through each try block unless: there is a problem importing modules, adding
    global variables or assigning local variables.  Otherwise everything happens within a try block with a general
    exception catch that will just log an ERROR with the exception type and message.
    Steps in program:

    1. Import packages.
    2. Custom functions

       i. fileopts

    3. Add globals
    4. Define local variables.
    5. Set up the logger & create the working directory structure.
    6. Create the working directory.
    7. Pull the metadata.
    8. Set up the PPP run
    9. Run PPP
    10. Parse the PPP .sum file.

    .. todo:: Add file existence checking before PPP run?

    .. todo:: Implement the OTL correction either using grdtab during processing or
              http://holt.oso.chalmers.se/loading/hfo.html after the stations are added to the database.

    .. todo:: Find a newer version of the .svb_gps_yrly PPP file.

    .. todo:: Add command (gpsppp.cmd) file customization?

    .. todo:: Write PPP in python, it's a nightmare to work with!!!!

    .. py:function:: fileopts(orig_file) -> pathlib.Path:

        :param pathlib.Path orig_file:
        :rtype: pathlib.Path
        :return:

        #. First determine which compression was used by invoking UNIX :manpage:`file(1)`
        #. Remove the .Z suffix
        #. Change the internal path suffix to o from d

        Raise exception if the last letter in the extention doesn't match [sp]3, [cl]k, [##]n, [##]o

.. py:function:: database_ops(rinex_dict=None, options=None)

    :param dict rinex_dict:
    :param options:
    :return:

    Compares the information in the rinex_dict object returned by parse_data_in with the information in the database
    collected by the head node.  Returns an string that indicates what should be done with the file.
    Possible outcomes:

    * File is not close to any stations and is added with the ??? network name.
    * File is close to another station but has a different name added with the ??? network name.
    * File matches the location of another station and has the same name and it has a network code not matching ???,
      it is moved into the archive.

    Locked files remain in data_in until they are unlocked by adding a network code to the station in the stations table
    of the database.

.. py:function:: main() -> None:

    :return:

    #. Parse the commandline arguments
    #. Load the options file into an object.
    #. Move files from retry into the data_in folder.
    #. Submit jobs to the cluster.
    #. First submit using cluster.submit, then add it to [jobs] so we can use the results later.  Once that is
       done we lock the MainThread using jobs_cond.acquire()
    #. Wait for the jobs to be submitted.

    .. todo:: Need to make sure that all classes sent via a dispy submission are 'picklable'

    .. todo:: Loggers aren't picklable

    .. todo:: Implement purge locks

    .. todo:: Implement duplicate archive.

    .. todo:: Evaluate how to determine the files that are found in the locks and in data_in

    .. todo:: Add data to the PostgreSQL database.

    .. todo:: Atomize the submission process.

