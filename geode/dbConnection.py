"""
Project: Geodetic Database Engine (GeoDE)
Date: 02/16/2017
Author: Demian D. Gomez

This class is used to connect to the database and handles inserts, updates and selects
It also handles the error, info and warning messages
"""

import sys
import platform
import configparser
import inspect
import re
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import numpy as np
from decimal import Decimal

# app
from .Utils import file_read_all, file_append, create_empty_cfg


DB_HOST = 'localhost'
DB_USER = 'postgres'
DB_PASS = ''
DB_NAME = 'gnss_data'


DEBUG = False


def cast_array_to_float(recordset):

    if len(recordset) > 0:
        if not isinstance(recordset[0], dict):
            result = []
            for record in recordset:
                new_record = []
                for field in record:
                    if isinstance(field, list):
                        new_record.append([float(value) if isinstance(value, Decimal) else value for value in field])
                    else:
                        if isinstance(field, Decimal):
                            new_record.append(float(field))
                        else:
                            new_record.append(field)

                result.append(tuple(new_record))

            return result
        else:
            # Convert any DECIMAL values to float
            for record in recordset:
                for key, value in record.items():
                    if isinstance(value, Decimal):
                        record[key] = float(value)
                    elif isinstance(value, list) and all(isinstance(i, Decimal) for i in value):
                        record[key] = [float(i) for i in value]

    return recordset


# class to match the pygreSQl structure using psycopg2
class query_obj(object):
    def __init__(self, cursor):
        self.rows = []
        # to maintain backwards compatibility
        try:
            self.rows = cast_array_to_float(cursor.fetchall())
        except psycopg2.ProgrammingError as e:
            if 'no results to fetch' in str(e):
                pass
            else:
                raise e

    def dictresult(self):
        return self.rows

    def ntuples(self):
        return len(self.rows)

    def getresult(self):
        return [tuple(d.values()) for d in self.rows]

    def __len__(self):
        return len(self.rows)


def debug(s):
    if DEBUG:
        file_append('/tmp/db.log', "DB: %s\n" % s)


def run_db_migrations(cnn: 'Cnn'):
    ##################################################################
    # New field in table api_visitgnssdatafiles
    if 'rinexed' not in cnn.get_columns('api_visitgnssdatafiles').keys():
        print(' >> Adding rinexed field to api_visitgnssdatafiles')
        cnn.begin_transac()
        cnn.query("""
        ALTER TABLE api_visitgnssdatafiles
        ADD COLUMN rinexed BOOLEAN DEFAULT FALSE;
        """)
        cnn.commit_transac()

    ##################################################################
    # New AntennaDAZ field in table stationinfo
    if 'AntennaDAZ' not in cnn.get_columns('stationinfo').keys():
        print(' >> Adding AntennaDAZ field to stationinfo')
        cnn.begin_transac()
        cnn.query("""
        ALTER TABLE stationinfo 
        ADD COLUMN "AntennaDAZ" NUMERIC(4,1) DEFAULT 0.0
        CHECK ("AntennaDAZ" >= 0.0 AND "AntennaDAZ" <= 360.0);
        """)
        cnn.commit_transac()

    ##################################################################
    # New plate field in table stations
    from .station_selector import get_tectonic_plate

    if 'plate' not in cnn.get_columns('stations').keys():
        print(' >> Adding plate field to stations, may take a few seconds')
        cnn.begin_transac()
        cnn.query("""
        ALTER TABLE stations
        ADD COLUMN plate VARCHAR(2) DEFAULT NULL;
        """)
        cnn.commit_transac()
        # now add tectonic plates to all stations
        stations = cnn.query_float('SELECT lat, lon, api_id FROM stations', as_dict=True)
        for stn in stations:
            if stn['lon'] is not None and stn['lat'] is not None:
                plate, _ = get_tectonic_plate(stn['lon'], stn['lat'])
                if plate:
                    cnn.update('stations', {'plate': plate}, api_id=stn['api_id'])
    else:
        # check that all stations
        stations = cnn.query_float('SELECT lat, lon, api_id FROM stations '
                                   'WHERE "NetworkCode" NOT LIKE \'?%%\' AND plate IS NULL', as_dict=True)
        for stn in stations:
            if stn['lon'] is not None and stn['lat'] is not None:
                plate, _ = get_tectonic_plate(stn['lon'], stn['lat'])
                if plate:
                    cnn.update('stations', {'plate': plate}, api_id=stn['api_id'])

    ##################################################################
    # modifications to ppp_soln to store big int values
    fields = cnn.get_columns('ppp_soln')

    if 'orbit' not in fields.keys():
        print(' >> Adding orbit field to ppp_soln')
        # New field in table ppp_soln present, no need to migrate.
        cnn.begin_transac()
        cnn.query("""
        ALTER TABLE ppp_soln
        ADD COLUMN orbit VARCHAR(100) DEFAULT '';
        """)
        cnn.commit_transac()

    ##################################################################
    if fields['hash'].lower() != 'bigint':
        # check the database to modify the ppp_soln table hash column from integer to bigint
        print(' >> Converting hash column in ppp_soln to BIGINT. This operation might take a while...')
        cnn.begin_transac()
        cnn.query("""
        ALTER TABLE ppp_soln
        ALTER COLUMN hash TYPE BIGINT;
        """)
        cnn.commit_transac()

    ##################################################################
    # check precision of lat lon height and auto_[x|y|z] in stations table
    stn_types = cnn.query_float("""
    SELECT 
        column_name,
        data_type,
        numeric_precision,
        numeric_scale
    FROM information_schema.columns 
        WHERE table_name = 'stations' 
        AND column_name = 'auto_x';
    """, as_dict=True)
    if stn_types[0]['numeric_precision'] is None:
        print(' >> Converting lat lon height and auto_[x|y|z] types')
        cnn.begin_transac()
        cnn.query("""
        ALTER TABLE stations ALTER COLUMN auto_x TYPE NUMERIC(16,5) USING ROUND(auto_x::numeric, 5);
        ALTER TABLE stations ALTER COLUMN auto_y TYPE NUMERIC(16,5) USING ROUND(auto_y::numeric, 5);
        ALTER TABLE stations ALTER COLUMN auto_z TYPE NUMERIC(16,5) USING ROUND(auto_z::numeric, 5);
        ALTER TABLE stations ALTER COLUMN lat    TYPE NUMERIC(12,9) USING ROUND(lat::numeric, 9);
        ALTER TABLE stations ALTER COLUMN lon    TYPE NUMERIC(12,9) USING ROUND(lon::numeric, 9);
        ALTER TABLE stations ALTER COLUMN height TYPE NUMERIC(10,5) USING ROUND(height::numeric, 5);
        """)
        cnn.commit_transac()

    ##################################################################
    # For the Mask object: check that the new fields exist or create them
    if 'density' not in cnn.get_columns('earthquakes').keys():
        cnn.begin_transac()
        cnn.query("""
                    ALTER TABLE earthquakes
                    ADD COLUMN density INTEGER   DEFAULT NULL,
                    ADD COLUMN c_kml   TEXT      DEFAULT NULL,
                    ADD COLUMN cp_kml  TEXT      DEFAULT NULL;
                    """)
        cnn.commit_transac()

    # check that the index exists
    idx = cnn.query("SELECT * FROM pg_indexes WHERE tablename = 'earthquakes' "
                    "AND indexname = 'earthquake_id_key'")

    if not len(idx):
        cnn.begin_transac()
        cnn.query("""CREATE UNIQUE INDEX earthquake_id_key ON earthquakes (id);""")
        cnn.commit_transac()

    ##################################################################
    # s_score_cache for storing the s_score values and not calculate them all the time

    s_score_cache = cnn.query_float("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name = 's_score_cache');
        """, as_dict=True)

    if not s_score_cache[0]['exists']:
        cnn.begin_transac()
        cnn.query("""
            CREATE TABLE s_score_cache (
                network_code VARCHAR(3) NOT NULL,
                station_code VARCHAR(4) NOT NULL,
                event_id VARCHAR(40) NOT NULL,
                coseismic NUMERIC(10,6),
                postseismic NUMERIC(10,6),
                hash BIGINT,
                PRIMARY KEY (network_code, station_code, event_id),
                FOREIGN KEY (network_code, station_code) 
                    REFERENCES stations("NetworkCode", "StationCode") 
                    ON DELETE CASCADE,
                FOREIGN KEY (event_id) 
                    REFERENCES earthquakes(id) 
                    ON DELETE CASCADE
            );
            CREATE INDEX idx_s_score_cache_hash ON s_score_cache(hash);
            CREATE INDEX idx_s_score_cache_station ON s_score_cache(network_code, station_code);
            CREATE INDEX idx_s_score_cache_event ON s_score_cache(event_id);
                """)
        cnn.commit_transac()

    ##################################################################
    # gamit_antenna_residuals for storing the residual values after DD processing

    antenna_residuals = cnn.query_float("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name = 'gamit_antenna_residuals');
        """, as_dict=True)

    if not antenna_residuals[0]['exists']:
        cnn.begin_transac()
        cnn.query("""
            CREATE TABLE gamit_antenna_residuals (
                network_code VARCHAR(3) NOT NULL,
                station_code VARCHAR(4) NOT NULL,
                project      VARCHAR(20),
                system       CHARACTER(1),
                subnet       SMALLINT NOT NULL,
                year         SMALLINT NOT NULL,
                doy          SMALLINT NOT NULL,
                antenna_code VARCHAR(22) NOT NULL,
                radome_code  VARCHAR(7) NOT NULL,
                residuals    DOUBLE PRECISION[91],  -- elevation-dependent residuals, index 1=0deg to 91=90deg
                CONSTRAINT gamit_antenna_residuals_pkey 
                    PRIMARY KEY (network_code, station_code, project, subnet, year, doy, system),
                FOREIGN KEY (network_code, station_code) 
                    REFERENCES stations("NetworkCode", "StationCode") 
                    ON DELETE CASCADE,
                FOREIGN KEY (project, subnet, year, doy, system) 
                    REFERENCES gamit_stats("Project", subnet, "Year", "DOY", system) 
                    ON DELETE CASCADE
            ) WITH (
                autovacuum_enabled = TRUE);
            CREATE INDEX idx_gamit_antenna_residuals_station ON gamit_antenna_residuals(network_code, station_code);
            CREATE INDEX idx_gamit_antenna_residuals_date ON gamit_antenna_residuals(year, doy);
            CREATE INDEX idx_gamit_antenna_residuals_antenna ON gamit_antenna_residuals(antenna_code, radome_code);
                """)
        cnn.commit_transac()

    ##################################################################
    # ppp_antenna_residuals for storing the residual values after PPP processing

    antenna_residuals = cnn.query_float("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'ppp_antenna_residuals');
        """, as_dict=True)

    if not antenna_residuals[0]['exists']:
        cnn.begin_transac()
        cnn.query("""
            CREATE TABLE ppp_antenna_residuals (
                network_code VARCHAR(3) NOT NULL,
                station_code VARCHAR(4) NOT NULL,
                system       CHARACTER(1),
                year         SMALLINT NOT NULL,
                doy          SMALLINT NOT NULL,
                antenna_code VARCHAR(22) NOT NULL,
                radome_code  VARCHAR(7) NOT NULL,
                residuals    DOUBLE PRECISION[91],  -- elevation-dependent residuals, index 1=0deg to 91=90deg
                CONSTRAINT ppp_antenna_residuals_pkey
                    PRIMARY KEY (network_code, station_code, year, doy),
                FOREIGN KEY (network_code, station_code)
                    REFERENCES stations("NetworkCode", "StationCode")
                    ON DELETE CASCADE,
                FOREIGN KEY (network_code, station_code, year, doy)
                    REFERENCES ppp_soln("NetworkCode", "StationCode", "Year", "DOY")
                    ON DELETE CASCADE
            ) WITH (
                autovacuum_enabled = TRUE);
            CREATE INDEX idx_ppp_antenna_residuals_station ON ppp_antenna_residuals(network_code, station_code);
            CREATE INDEX idx_ppp_antenna_residuals_date ON ppp_antenna_residuals(year, doy);
            CREATE INDEX idx_ppp_antenna_residuals_antenna ON ppp_antenna_residuals(antenna_code, radome_code);
                """)
        cnn.commit_transac()

    ##################################################################
    # Migrate antennas table: extend primary key to include RadomeCode,
    # and update stationinfo FK to enforce both AntennaCode + RadomeCode.
    #
    # Before this migration:
    #   - antennas PK:  (AntennaCode)             <-- radome-blind
    #   - stationinfo FK: AntennaCode → antennas  <-- radome unconstrained
    #
    # After this migration:
    #   - antennas PK:  (AntennaCode, RadomeCode) <-- full IGS pair
    #   - stationinfo FK: (AntennaCode, RadomeCode) → antennas
    #
    # Side-effect: gamit_htc.antenna_fk references the old single-column PK
    # and cannot survive the PK change. It is dropped here. gamit_htc data
    # is untouched; HTC data is radome-independent so no FK is re-added.
    if 'RadomeCode' not in cnn.get_columns('antennas').keys():
        print(' >> Migrating antennas: adding RadomeCode to primary key '
              'and updating stationinfo FK to enforce (AntennaCode, RadomeCode)')
        cnn.begin_transac()
        cnn.query("""
        -- Step 1: Add RadomeCode to antennas.
        --         Existing rows default to 'NONE' (IGS code for no radome).
        ALTER TABLE antennas
            ADD COLUMN "RadomeCode" VARCHAR(7) NOT NULL DEFAULT 'NONE';

        -- Step 2: Drop stationinfo's FK FIRST — it depends on antennas_pkey,
        --         so the PK cannot be touched while this constraint is alive.
        ALTER TABLE stationinfo
            DROP CONSTRAINT "stationinfo_AntennaCode_fkey";

        -- Step 3: Drop gamit_htc's FK (also references antennas_pkey).
        ALTER TABLE gamit_htc
            DROP CONSTRAINT antenna_fk;

        -- Step 4: Now that no FKs depend on it, drop the old single-column PK.
        ALTER TABLE antennas
            DROP CONSTRAINT antennas_pkey;

        -- Step 5: Establish the new composite primary key.
        ALTER TABLE antennas
            ADD CONSTRAINT antennas_pkey
                PRIMARY KEY ("AntennaCode", "RadomeCode");

        -- Step 6: Back-fill any (AntennaCode, RadomeCode) pairs that already
        --         exist in stationinfo but are missing from antennas.
        --         api_id is omitted — the sequence fills it automatically.
        INSERT INTO antennas ("AntennaCode", "RadomeCode")
        SELECT DISTINCT si."AntennaCode", si."RadomeCode"
        FROM   stationinfo si
        WHERE  NOT EXISTS (
            SELECT 1 FROM antennas a
            WHERE  a."AntennaCode" = si."AntennaCode"
              AND  a."RadomeCode"  = si."RadomeCode"
        );

        -- Step 7: Re-add stationinfo's FK against the new composite PK.
        ALTER TABLE stationinfo
            ADD CONSTRAINT "stationinfo_AntennaCode_RadomeCode_fkey"
                FOREIGN KEY ("AntennaCode", "RadomeCode")
                REFERENCES antennas ("AntennaCode", "RadomeCode")
                ON UPDATE CASCADE
                ON DELETE RESTRICT;

        -- Step 8: Drop the DEFAULT now that the schema is stable.
        --         Future inserts into antennas must supply RadomeCode explicitly.
        ALTER TABLE antennas
            ALTER COLUMN "RadomeCode" DROP DEFAULT;
        """)
        cnn.commit_transac()

        ##################################################################
        # ATX calibration tables: atx_files, antenna_calibrations,
        # antenna_calibration_freq, antenna_calibration_pcv.
        #
        # These four tables store the complete content of ANTEX 1.4 files
        # so that multiple ATX sources can be held simultaneously and every
        # calibration value is traceable back to a specific file.
        #
        # Dependency: the antennas table must already have the composite
        # PK (AntennaCode, RadomeCode) added by the earlier migration above.

        atx_files_exists = cnn.query_float("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND   table_name   = 'atx_files'
            )
        """, as_dict=True)

        if not atx_files_exists[0]['exists']:
            print(' >> Creating ATX calibration tables '
                  '(atx_files, antenna_calibrations, antenna_calibration_freq, '
                  'antenna_calibration_pcv)')
            cnn.begin_transac()
            cnn.query("""
                -- ----------------------------------------------------------------
                -- 1. ATX source files registry
                -- ----------------------------------------------------------------
                CREATE TABLE atx_files (
                    atx_file_id        SERIAL      PRIMARY KEY,
                    filename           VARCHAR(255) NOT NULL,
                    pcv_type           CHAR(1)      NOT NULL
                                       CHECK (pcv_type IN ('A', 'R')),
                    ref_antenna        VARCHAR(20),
                    ref_antenna_serial VARCHAR(20),
                    loaded_at          TIMESTAMP    NOT NULL DEFAULT NOW(),
                    CONSTRAINT atx_files_filename_key UNIQUE (filename)
                );

                COMMENT ON TABLE  atx_files IS
                    'Registry of ATX (ANTEX 1.4) source files loaded into the database.';
                COMMENT ON COLUMN atx_files.pcv_type IS
                    '''A'' = absolute, ''R'' = relative phase center variations.';

                -- ----------------------------------------------------------------
                -- 2. Antenna calibration blocks
                --    One row per (AntennaCode, RadomeCode, serial_no, atx_file).
                --    Stores everything needed to reconstruct the ATX antenna section.
                -- ----------------------------------------------------------------
                CREATE TABLE antenna_calibrations (
                    calibration_id   SERIAL       PRIMARY KEY,

                    "AntennaCode"    VARCHAR(20)  NOT NULL,
                    "RadomeCode"     VARCHAR(4)   NOT NULL,
                    serial_no        VARCHAR(20)  NOT NULL DEFAULT '',

                    atx_file_id      INTEGER      NOT NULL
                        REFERENCES atx_files (atx_file_id) ON DELETE CASCADE,

                    -- METH / BY / # / DATE
                    method           VARCHAR(20),
                    calibrated_by    VARCHAR(20),
                    num_calibrations INTEGER,
                    cal_date         VARCHAR(10),

                    -- DAZI  (0.0 → no azimuth dependence)
                    dazi             NUMERIC(6,1) NOT NULL DEFAULT 0.0,

                    -- ZEN1 / ZEN2 / DZEN
                    zen1             NUMERIC(6,1) NOT NULL,
                    zen2             NUMERIC(6,1) NOT NULL,
                    dzen             NUMERIC(6,1) NOT NULL,

                    -- # OF FREQUENCIES
                    num_frequencies  INTEGER      NOT NULL,

                    -- VALID FROM / VALID UNTIL (optional in ANTEX 1.4)
                    valid_from       TIMESTAMP,
                    valid_until      TIMESTAMP,

                    -- SINEX CODE (optional)
                    sinex_code       VARCHAR(10),

                    -- All COMMENT lines inside the antenna block (preserves order)
                    comments         TEXT[],

                    CONSTRAINT antenna_calibrations_antenna_fk
                        FOREIGN KEY ("AntennaCode", "RadomeCode")
                        REFERENCES antennas ("AntennaCode", "RadomeCode")
                        ON UPDATE CASCADE ON DELETE RESTRICT,

                    CONSTRAINT antenna_calibrations_unique
                        UNIQUE ("AntennaCode", "RadomeCode", serial_no, atx_file_id)
                );

                CREATE INDEX idx_ant_cal_antenna
                    ON antenna_calibrations ("AntennaCode", "RadomeCode");
                CREATE INDEX idx_ant_cal_atxfile
                    ON antenna_calibrations (atx_file_id);

                COMMENT ON TABLE  antenna_calibrations IS
                    'One row per antenna/radome/serial/ATX-file. '
                    'Contains the full antenna block header needed to reconstruct the ATX section.';
                COMMENT ON COLUMN antenna_calibrations.serial_no IS
                    'Empty string means the calibration applies to all representatives of this antenna type.';
                COMMENT ON COLUMN antenna_calibrations.dazi IS
                    '0.0 means no azimuth-dependent corrections are stored.';
                COMMENT ON COLUMN antenna_calibrations.comments IS
                    'Array of COMMENT lines found inside the antenna block, in file order.';

                -- ----------------------------------------------------------------
                -- 3. Phase centre offsets (PCO)
                --    One row per calibration × GNSS frequency.
                -- ----------------------------------------------------------------
                CREATE TABLE antenna_calibration_freq (
                    freq_id          SERIAL        PRIMARY KEY,
                    calibration_id   INTEGER       NOT NULL
                        REFERENCES antenna_calibrations (calibration_id) ON DELETE CASCADE,

                    frequency        VARCHAR(3)    NOT NULL,  -- G01, G02, R01, E01 …

                    -- NORTH / EAST / UP eccentricities in mm (relative to ARP)
                    north_offset     NUMERIC(10,4) NOT NULL,
                    east_offset      NUMERIC(10,4) NOT NULL,
                    up_offset        NUMERIC(10,4) NOT NULL,

                    -- Optional RMS from START OF FREQ RMS
                    north_offset_rms NUMERIC(10,4),
                    east_offset_rms  NUMERIC(10,4),
                    up_offset_rms    NUMERIC(10,4),

                    CONSTRAINT antenna_calibration_freq_unique
                        UNIQUE (calibration_id, frequency)
                );

                CREATE INDEX idx_ant_cal_freq_cal
                    ON antenna_calibration_freq (calibration_id);

                COMMENT ON TABLE  antenna_calibration_freq IS
                    'Phase centre offsets (PCO) per calibration and GNSS frequency.';
                COMMENT ON COLUMN antenna_calibration_freq.frequency IS
                    'Three-character ANTEX frequency code: G01=L1, G02=L2, R01=G1, E01=E1, etc.';

                -- ----------------------------------------------------------------
                -- 4. Phase centre variations (PCV)
                --    One row per freq × azimuth bin.
                --    azimuth IS NULL  → NOAZI (non-azimuth-dependent) pattern.
                --    azimuth NOT NULL → azimuth-dependent row (degrees, 0–360).
                --    pcv_values holds PCV values [mm] from ZEN1 to ZEN2 step DZEN.
                -- ----------------------------------------------------------------
                CREATE TABLE antenna_calibration_pcv (
                    pcv_id          BIGSERIAL           PRIMARY KEY,
                    freq_id         INTEGER             NOT NULL
                        REFERENCES antenna_calibration_freq (freq_id) ON DELETE CASCADE,

                    -- NULL = NOAZI;  numeric = azimuth angle in degrees (0.0–360.0)
                    azimuth         NUMERIC(6,1),

                    -- PCV array [mm], length = (zen2 - zen1) / dzen + 1
                    pcv_values      DOUBLE PRECISION[]  NOT NULL,

                    -- Optional RMS array from START OF FREQ RMS
                    pcv_rms_values  DOUBLE PRECISION[]
                );

                -- Enforce uniqueness of the NOAZI row per frequency.
                -- A partial index is used for NULL azimuth (compatible with PG < 15).
                CREATE UNIQUE INDEX idx_ant_pcv_noazi
                    ON antenna_calibration_pcv (freq_id)
                    WHERE azimuth IS NULL;

                -- Unique index for azimuth-dependent rows.
                CREATE UNIQUE INDEX idx_ant_pcv_azi
                    ON antenna_calibration_pcv (freq_id, azimuth)
                    WHERE azimuth IS NOT NULL;

                CREATE INDEX idx_ant_pcv_freq
                    ON antenna_calibration_pcv (freq_id);

                COMMENT ON TABLE  antenna_calibration_pcv IS
                    'Phase centre variations (PCV) per frequency and azimuth bin. '
                    'azimuth IS NULL for NOAZI (non-azimuth-dependent) rows.';
                COMMENT ON COLUMN antenna_calibration_pcv.azimuth IS
                    'NULL = NOAZI pattern; otherwise azimuth angle in degrees (0.0–360.0).';
                COMMENT ON COLUMN antenna_calibration_pcv.pcv_values IS
                    'Array of PCV values [mm] from ZEN1 to ZEN2 in DZEN steps '
                    '(length = (zen2 - zen1) / dzen + 1).';
            """)
            cnn.commit_transac()

    ##################################################################
    # Widen ReceiverDescription in receivers from VARCHAR(22) to VARCHAR(256).
    # The original schema allocated only 22 chars — identical to ReceiverCode —
    # which is far too short for the IGS equipment descriptions in rcvr_ant.txt.
    rcv_desc = cnn.query_float("""
        SELECT character_maximum_length
        FROM information_schema.columns
        WHERE table_name  = 'receivers'
          AND column_name = 'ReceiverDescription';
    """, as_dict=True)

    if rcv_desc and rcv_desc[0]['character_maximum_length'] is not None \
            and rcv_desc[0]['character_maximum_length'] < 256:
        print(' >> Widening receivers."ReceiverDescription" from '
              f'VARCHAR({rcv_desc[0]["character_maximum_length"]}) to VARCHAR(256)')
        cnn.begin_transac()
        cnn.query("""
        ALTER TABLE receivers
            ALTER COLUMN "ReceiverDescription" TYPE VARCHAR(256);
        """)
        cnn.commit_transac()

    ##################################################################
    # New table: sources_metadata
    # Stores URL/path structure for metadata files (IGS logs, stninfo).
    # Same fields as sources_servers; each sources_servers record can
    # reference one sources_metadata record via metadata_source_id.

    sources_metadata_exists = cnn.query_float("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'sources_metadata');
        """, as_dict=True)

    if not sources_metadata_exists[0]['exists']:
        print(' >> Creating sources_metadata table')
        cnn.begin_transac()
        cnn.query("""
            CREATE TABLE sources_metadata (
                id         SERIAL PRIMARY KEY,
                protocol   VARCHAR NOT NULL CHECK (protocol IN ('ftp', 'http', 'sftp',
                           'https', 'ftpa', 'FTP', 'HTTP', 'SFTP', 'HTTPS', 'FTPA')),
                fqdn       VARCHAR NOT NULL,
                username   VARCHAR,
                "password" VARCHAR,
                "path"     VARCHAR,
                "format"   VARCHAR REFERENCES sources_formats(format)
                           DEFAULT 'DEFAULT_FORMAT'
            );

            COMMENT ON TABLE sources_metadata IS
                'URL/path templates for metadata files (IGS site logs, station info). '
                'Referenced by sources_servers.metadata_source_id.';
        """)
        cnn.commit_transac()

    ##################################################################
    # New column: sources_servers.metadata_source_id
    # Foreign key to sources_metadata for metadata download paths.

    if 'metadata_source_id' not in cnn.get_columns('sources_servers').keys():
        print(' >> Adding metadata_source_id to sources_servers')
        cnn.begin_transac()
        cnn.query("""
            ALTER TABLE sources_servers
                ADD COLUMN metadata_source_id INTEGER REFERENCES sources_metadata(id);
        """)
        cnn.commit_transac()

    ##################################################################
    # New table: stationinfo_audit
    # Tracks audit findings from metadata comparisons, keyed by session hash.
    # Used to prevent re-flagging findings that humans have already reviewed.

    stationinfo_audit_exists = cnn.query_float("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'stationinfo_audit');
        """, as_dict=True)

    if not stationinfo_audit_exists[0]['exists']:
        print(' >> Creating stationinfo_audit table')
        cnn.begin_transac()
        cnn.query("""
            CREATE TABLE stationinfo_audit (
                api_id          SERIAL PRIMARY KEY,

                -- which station
                "NetworkCode"   VARCHAR(3)   NOT NULL,
                "StationCode"   VARCHAR(4)   NOT NULL,

                -- CRC32 fingerprint of the external session (or DB record for ORPHAN)
                session_hash    BIGINT       NOT NULL,

                -- what Claude found
                finding_type    VARCHAR(30)  NOT NULL,
                action_required VARCHAR(10)  NOT NULL,

                -- db_record: {"DateStart": "YYYY-MM-DD HH:MM:SS"} to identify DB session
                db_record       JSONB,
                claude_summary  TEXT,

                -- structured field values for programmatic updates (only differing fields)
                db_field_values   JSONB,
                file_field_values JSONB,

                -- human disposition (NULL = not yet reviewed)
                reviewed_by     VARCHAR(80),
                reviewed_at     TIMESTAMP,
                disposition     VARCHAR(10),
                review_notes    TEXT,

                -- audit trail
                created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
            );

            -- Prevents duplicate audit rows for the same session content
            CREATE UNIQUE INDEX stationinfo_audit_unique
                ON stationinfo_audit ("NetworkCode", "StationCode", session_hash);

            -- Index for fast lookups by station
            CREATE INDEX idx_stationinfo_audit_station
                ON stationinfo_audit ("NetworkCode", "StationCode");

            COMMENT ON TABLE stationinfo_audit IS
                'Tracks metadata comparison findings per station session. '
                'session_hash is the CRC32 fingerprint of the external session content.';
            COMMENT ON COLUMN stationinfo_audit.session_hash IS
                'CRC32 of the canonical stninfo-format line for the external session, '
                'or the DB record for ORPHAN_SESSION findings. Matches StationInfoRecord.hash.';
            COMMENT ON COLUMN stationinfo_audit.disposition IS
                'Human decision: APPLIED, DISMISSED, DEFERRED, or NO_ACTION (auto-set for matches).';
            COMMENT ON COLUMN stationinfo_audit.db_field_values IS
                'JSONB object with field names as keys and current DB values. '
                'Only contains fields that differ between DB and external file. '
                'Field names match StationInfoRecord attributes (e.g., ReceiverCode, AntennaHeight).';
            COMMENT ON COLUMN stationinfo_audit.file_field_values IS
                'JSONB object with field names as keys and recommended values from external file. '
                'Only contains fields that differ between DB and external file. '
                'Field names match StationInfoRecord attributes (e.g., ReceiverCode, AntennaHeight).';
        """)
        cnn.commit_transac()

    ##################################################################
    # New column: sources_stations.metadata_hash
    # Stores the CRC32 hash of the last downloaded metadata file for each station.
    # Used to detect changes without re-parsing and calling the API.

    if 'metadata_hash' not in cnn.get_columns('sources_stations').keys():
        print(' >> Adding metadata_hash to sources_stations')
        cnn.begin_transac()
        cnn.query("""
            ALTER TABLE sources_stations
                ADD COLUMN metadata_hash BIGINT;

            COMMENT ON COLUMN sources_stations.metadata_hash IS
                'CRC32 hash of the last downloaded metadata file for this station. '
                'Used to detect file changes without re-parsing.';
        """)
        cnn.commit_transac()


def adapt_numpy_array(numpy_array):
    return psycopg2.extensions.adapt(numpy_array.tolist())


class dbErrInsert (psycopg2.errors.UniqueViolation): pass


class dbErrUpdate (Exception): pass


class dbErrConnect(Exception): pass


class dbErrDelete (Exception): pass


class DatabaseError(psycopg2.DatabaseError): pass


class Cnn(object):

    def __init__(self, configfile, use_float=False, write_cfg_file=False):

        options = {'hostname': DB_HOST,
                   'username': DB_USER,
                   'password': DB_PASS,
                   'database': DB_NAME}

        self.active_transaction = False
        self.options            = options
        
        # parse session config file
        config = configparser.ConfigParser()

        try:
            config.read_string(file_read_all(configfile))
        except FileNotFoundError:
            if write_cfg_file:
                create_empty_cfg()
                print(' >> No gnss_data.cfg file found, an empty one has been created. Replace all the necessary '
                      'config and try again.')
                exit(1)
            else:
                raise
        # get the database config
        options.update(dict(config.items('postgres')))

        # register an adapter to convert decimal to float
        # see: https://www.psycopg.org/docs/faq.html#faq-float
        DEC2FLOAT = psycopg2.extensions.new_type(
            psycopg2.extensions.DECIMAL.values,
            'DEC2FLOAT',
            lambda value, curs: float(value) if value is not None else None)

        # Define the custom type for an array of decimals
        DECIMAL_ARRAY_TYPE = psycopg2.extensions.new_type(
            (psycopg2.extensions.DECIMAL.values,),  # This matches the type codes for DECIMAL
            'DECIMAL_ARRAY',  # Name of the type
            lambda value, curs: [float(d) for d in value] if value is not None else None
        )

        psycopg2.extensions.register_type(DEC2FLOAT)
        psycopg2.extensions.register_type(DECIMAL_ARRAY_TYPE)
        psycopg2.extensions.register_adapter(np.ndarray, adapt_numpy_array)

        # open connection to server
        err = None
        for i in range(3):
            try:
                self.cnn = psycopg2.connect(host=options['hostname'], user=options['username'],
                                            password=options['password'], dbname=options['database'],
                                            connect_timeout=10)

                self.cnn.autocommit = True
                self.cursor = self.cnn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                debug("Database connection established")

                run_db_migrations(self)

            except psycopg2.Error as e:
                raise e
            else:
                break
        else:
            raise dbErrConnect(err)

    def query(self, command):
        try:
            self.cursor.execute(command)

            debug(" QUERY: command=%r" % command)

            # passing a query object to match response from pygresql
            return query_obj(self.cursor)
        except Exception as e:
            raise DatabaseError(e)

    def query_float(self, command, as_dict=False):
        # deprecated: using psycopg2 now solves the problem of returning float numbers
        # still in to maintain backwards compatibility
        if not as_dict:
            cursor = self.cnn.cursor()
            cursor.execute(command)
            recordset = cast_array_to_float(cursor.fetchall())
        else:
            # return results as a dictionary
            self.cursor.execute(command)
            recordset = cast_array_to_float(self.cursor.fetchall())

        return recordset


    def get(self, table, filter_fields, return_fields=None, limit=None):
        """
        Selects from the given table the records that match filter_fields and returns ONE dictionary.
        Method should not be used to retrieve more than one single record.
        Parameters:
        table (str): The table to select from.
        filter_fields (dict): The dictionary where the keys are the field names and the values are the filter values.
        return_fields (list of str): The fields to return. If empty return all columns
        limit (int): sets a limit for rows in case it is a query to determine if records exist

        Returns:
        list: A list of dictionaries, each representing a record that matches the filter.
        """

        if return_fields is None:
            return_fields = list(self.get_columns(table).keys())

        where_clause = ' AND '.join([f'"{key}" = %s' if val is not None else f'"{key}" IS %s'
                                     for key, val in zip(filter_fields.keys(), filter_fields.values())])
        fields_clause = ', '.join([f'"{field}"' for field in return_fields])
        if where_clause:
            query = f'SELECT {fields_clause} FROM {table} WHERE {where_clause}'
        else:
            query = f'SELECT {fields_clause} FROM {table}'
        values = list(filter_fields.values())
        # new feature to limit the results
        if limit:
            query += ' LIMIT %i' % limit

        try:
            self.cursor.execute(query, values)
            records = self.cursor.fetchall()
            debug(f"SELECT: query={query}, values={values}")

            if len(records) > 0:
                return records[0]
            else:
                raise DatabaseError('query returned no records: ' + query)

        except psycopg2.Error as e:
            raise e

    def get_columns(self, table):
        tblinfo = self.query('select column_name, data_type from information_schema.columns where table_name=\'%s\''
                             % table).dictresult()

        return {field['column_name']: field['data_type'] for field in tblinfo}

    def begin_transac(self):
        # do not begin a new transaction with another one active.
        if self.active_transaction:
            self.rollback_transac()

        self.active_transaction = True
        self.cursor.execute('BEGIN TRANSACTION')

    def commit_transac(self):
        self.active_transaction = False
        self.cursor.execute('COMMIT')

    def rollback_transac(self):
        self.active_transaction = False
        self.cursor.execute('ROLLBACK')

    def insert(self, table: str, **kw):
        debug("INSERT: table=%r kw=%r" % (table, kw))

        # figure out any extra columns and remove them from the incoming **kw
        cols = list(self.get_columns(table).keys())

        # assuming fields are passed through kw which are keyword arguments
        fields = [k for k in kw.keys() if k in cols]
        values = [v for v, k in zip(kw.values(), kw.keys()) if k in cols]

        # form the insert query dynamically
        placeholders = ', '.join(['%s'] * len(fields))
        columns = '", "'.join(fields)
        query = f'INSERT INTO {table} ("{columns}") VALUES ({placeholders})'
        try:
            self.cursor.execute(query, values)
            self.cnn.commit()
        except psycopg2.errors.UniqueViolation as e:
            self.cnn.rollback()
            raise dbErrInsert(e)

    def update(self, table: str, set_clause_dict: dict, **kwargs):
        """
        Updates the specified table with new field values. The row(s) are updated based on the primary key(s)
        indicated in the 'row' dictionary. New values are specified in kwargs. Field names must be enclosed
        with double quotes to handle camel case names.

        Parameters:
        table (str): The table to update.
        set_row (dict): New field values for the row.
        kwargs: The dictionary where the keys are the primary key fields and the values are the row's identifiers.
        """
        # Build the SET clause of the query
        cols = list(self.get_columns(table))
        set_clause = ', '.join([f'"{field}" = %s' for field in set_clause_dict.keys() if field in cols])

        # Build the WHERE clause based on the row dictionary
        where_clause = ' AND '.join([f'"{key}" = %s' if val is not None else f'"{key}" IS %s'
                                     for key, val in zip(kwargs.keys(), kwargs.values())])
        # Construct query
        query = f'UPDATE {table} SET {set_clause} WHERE {where_clause}'

        # Values to use in the query
        values = (list([value for field, value in set_clause_dict.items() if field in cols])
                  + list(kwargs.values()))

        try:
            self.cursor.execute(query, values)
            self.cnn.commit()
            debug(f"UPDATE {table}: set={set_clause_dict}, where={kwargs}")
            debug(query)
        except psycopg2.Error as e:
            self.cnn.rollback()
            raise dbErrUpdate(e)

    def delete(self, table, **kw):
        """
        Deletes row(s) from the specified table based on the provided keyword arguments.

        Parameters:
        table (str): The table to delete from.
        kw: Keywords to identify the row(s) to be deleted.
        """
        debug("DELETE: table=%r kw=%r" % (table, kw))

        if not kw:
            raise ValueError("No conditions provided for deletion")

        where_clause = ' AND '.join([f'"{key}" = %s' if val is not None else f'"{key}" IS %s'
                                     for key, val in zip(kw.keys(), kw.values())])
        query = f'DELETE FROM {table} WHERE {where_clause}'
        values = list(kw.values())

        try:
            self.cursor.execute(query, values)
            self.cnn.commit()
            debug(f"DELETE FROM {table}: kw={kw}")
        except psycopg2.Error as e:
            self.cnn.rollback()
            raise dbErrDelete(e)

    def insert_event(self, event):
        debug("EVENT: event=%r" % (event.db_dict()))

        self.insert('events', **event.db_dict())

    def insert_event_bak(self, type, module, desc):
        debug("EVENT_BAK: type=%r module=%r desc=%r" % (type, module, desc))

        # do not insert if record exists
        desc = '%s%s' % (module, desc.replace('\'', ''))
        desc = re.sub(r'[^\x00-\x7f]+', '', desc)
        # remove commands from events
        # modification introduced by DDG (suggested by RS)
        desc = re.sub(r'BASH.*', '', desc)
        desc = re.sub(r'PSQL.*', '', desc)

        # warn = self.query('SELECT * FROM events WHERE "EventDescription" = \'%s\'' % (desc))

        # if warn.ntuples() == 0:
        self.insert('events', EventType=type, EventDescription=desc)

    def insert_warning(self, desc):
        self.insert_event_bak('warn', _caller_str(), desc)

    def insert_error(self, desc):
        self.insert_event_bak('error', _caller_str(), desc)

    def insert_info(self, desc):
        self.insert_event_bak('info', _caller_str(), desc)

    def close(self):
        self.cursor.close()
        self.cnn.close()

    def __del__(self):
        if self.active_transaction:
            self.cnn.rollback()


def _caller_str():
    # get the module calling to make clear how is logging this message
    frame = inspect.stack()[2]
    line   = frame[2]
    caller = frame[3]
    
    return '[%s:%s(%s)]\n' % (platform.node(), caller, str(line))

