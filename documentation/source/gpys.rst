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

