=======================
gpys
=======================
Contains a few classes that make organizing the executable scripts easier.  Most repetative tasks will get migrated into
this module.

.. py:class:: Connection

    Ideally this class is to be used as a direct connection to the gnss_data database.  As such it should initiate to
    a pgdb.Connection object through the uses of pgdb.connect().  Once the object has established a connection to the
    database, class methods should be used to interact with the database.  The goal is to abstract all the SQL commands
    so we can deal with the errors here instead of out in the wild.

    .. py:function:: __init__(config, parent_logger='archive')

        :param config:
        :param str parent_logger:

    .. py:function:: execute_wrapper(sql_statement=None, values=None, retval=False, return_dict=True)

        :param sql.Composed sql_statement: A composable object.
        :param list values: List or tuple of values for the sql statement.
        :param bool retval: Whether to return a value or not.
        :param bool return_dict: Return it as a dictionary or not.
        :return: Returns either a list of tuples containing the results or a dictionary or None.

        Deal with all the actual database interactions here and deal with the related error possibilities.

    .. py:function:: insert(table, record):

        :param str table:
        :param dict record:

    .. py:function:: update_locks()

    .. py:function:: load_table(table=None, columns=None)

        :param str table:
        :param list columns:


    .. py:function:: load_tankstruct()

        :return:

        Determines the archive structure based on the two tables, rinex_tank_struct and keys.  Returns a dictionary with
        the following keys:

        **KeyCode**: ( :py:obj:`list` [ :py:obj:`str` ]) Property of a file e.g. 'network' or 'doy'.

        **Level**: ( :py:obj:`list` [ :py:obj:`int` ]) The heirachy for the properties, the first entry will be the highest level in the
        archive.

        **TotalChars**: ( :py:obj:`list` [ :py:obj:`int` ]) The number of characters in the keycode.

    .. py:function:: insert_event(event)

        :param event:

    .. py:function:: print_summary(script)

        :param script:

    .. py:function:: spatial_check(vals, search_in_new=False)

        :param vals:
        :param bool search_in_new:
        :return: The following keys, NetworkCode, StationCode, StationName, DateStart, DateEnd, auto_x, auto_y, auto_z, Harpos_coeff_otl, lat, lon, height, max_dist, dome, distance.
        :rtype: defaultdict(list)

        Used to find the nearest station to a given RINEX file.  It only goes out to a range of 20 meters or the value
        listed in the max_dist column of the stations table.  It will return None if there are no matching stations
        and if there is a match a defaultdict(list) object containing every column from the stations table in addition
        to the distance between the RINEX and the given stations entry.  The distance calculation is performed using the
        haversine formula.

    .. py:function:: nearest_station(vals, search_in_new=False)

        :param vals: list with format [lattitude, longitude] in decimal degrees
        :param bool search_in_new: Whether to also search stations that are not yet assigned to a network.
        :return: A single entry from the stations table of the database.

        Sorts all stations by  distance from the given [lat, lon] in the vals variable, always returns a station unless
        the database is empty.

    .. py:function:: similar_locked(vals)

        :param vals:

    .. py:function:: update(table=None, row=None, record=None)

        :param str table:
        :param dict row:
        :param dict record:

    .. py:function:: load_table_matching(table=None, where_dict=None)

        :param str table:
        :param dict where_dict:


.. py:class:: ReadOptions

    Class that deals with reading in the default configuration file gnss_data.cfg

    .. py:attribute:: options

        A dictionary containing the values read in by :py:mod:`configparser`.  Not all values are used currently, here
        are the currently used keys:

    .. py:attribute:: data_in

    .. py:attribute:: data_in_retry

    .. py:attribute:: data_reject

    .. py:attribute:: sp3types

    .. py:attribute:: rinex_struct

    .. py:function:: __init__(configfile='gnss_data.cfg', parent_logger='archive')

        :param str configfile:
        :param str parent_logger:

        Initialize the logger.

    .. py:function:: scan_archive_struct(rootdir = None) -> list:

        :param rootdir:

        Recursive member method of RinexArcvhive that searches through the given rootdir
        to find files matching a compressed rinex file e.g. ending with d.Z.  The method
        self.scan_archive_struct() is used to determine the file type.

.. py:class:: JobServer

    .. py:function:: __init__(self, options, parent_logger='archive'):

        :param options: gpys.ReadOptions instance
        :param parent_logger: Name of the function creating a new instance of JobServer

        Initialize the the dispy scheduler and test the connection to the expected nodes.

    .. py:function:: _connect(compute)

    .. py:function:: cluster_test()

.. py:class:: Distribute

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

        .. py:function:: fileopts(orig_file) -> pathlib.Path:

            :param pathlib.Path orig_file:
            :rtype: pathlib.Path
            :return:

            #. First determine which compression was used by invoking UNIX :manpage:`file(1)`
            #. Remove the .Z suffix
            #. Change the internal path suffix to o from d

            Raise exception if the last letter in the extention doesn't match [sp]3, [cl]k, [##]n, [##]o

    .. py:function:: database_ops(rinex_dict=None, options=None)

        :param dict rinex_dict: Output from
        :param options:

        Compares the information in the rinex_dict object returned by parse_data_in with the information in the database
        collected by the head node.  Returns an string that indicates what should be done with the file.
        Possible outcomes:

        * File is not close to any stations and is added with the ??? network name.
        * File is close to another station but has a different name added with the ??? network name.
        * File matches the location of another station and has the same name and it has a network code not matching ???,
          it is moved into the archive.

        Locked files remain in data_in until they are unlocked by adding a network code to the station in the stations table
        of the database.