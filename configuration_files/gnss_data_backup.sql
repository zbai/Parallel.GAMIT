--
-- PostgreSQL database dump
--

-- Dumped from database version 11.2
-- Dumped by pg_dump version 11.2

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

DROP DATABASE gnss_data;
--
-- Name: gnss_data; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE gnss_data WITH TEMPLATE = template0 ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';


ALTER DATABASE gnss_data OWNER TO postgres;

\connect gnss_data

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: ecef2neu(numeric, numeric, numeric, numeric, numeric); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.ecef2neu(dx numeric, dy numeric, dz numeric, lat numeric, lon numeric) RETURNS double precision[]
    LANGUAGE sql
    AS $_$
select 
array[-sin(radians($4))*cos(radians($5))*$1 - sin(radians($4))*sin(radians($5))*$2 + cos(radians($4))*$3::numeric,
      -sin(radians($5))*$1 + cos(radians($5))*$2::numeric,
      cos(radians($4))*cos(radians($5))*$1 + cos(radians($4))*sin(radians($5))*$2 + sin(radians($4))*$3::numeric];

$_$;


ALTER FUNCTION public.ecef2neu(dx numeric, dy numeric, dz numeric, lat numeric, lon numeric) OWNER TO postgres;

--
-- Name: fyear(numeric, numeric, numeric, numeric, numeric); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.fyear("Year" numeric, "DOY" numeric, "Hour" numeric DEFAULT 12, "Minute" numeric DEFAULT 0, "Second" numeric DEFAULT 0) RETURNS numeric
    LANGUAGE sql
    AS $_$
SELECT CASE 
WHEN isleapyear(cast($1 as integer)) = True  THEN $1 + ($2 + $3/24 + $4/1440 + $5/86400)/366
WHEN isleapyear(cast($1 as integer)) = False THEN $1 + ($2 + $3/24 + $4/1440 + $5/86400)/365
END;

$_$;


ALTER FUNCTION public.fyear("Year" numeric, "DOY" numeric, "Hour" numeric, "Minute" numeric, "Second" numeric) OWNER TO postgres;

--
-- Name: horizdist(double precision[]); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.horizdist(neu double precision[]) RETURNS double precision
    LANGUAGE sql
    AS $_$

select 
sqrt(($1)[1]^2 + ($1)[2]^2 + ($1)[3]^2)

$_$;


ALTER FUNCTION public.horizdist(neu double precision[]) OWNER TO postgres;

--
-- Name: isleapyear(integer); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.isleapyear(year integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
SELECT ($1 % 4 = 0) AND (($1 % 100 <> 0) or ($1 % 400 = 0))
$_$;


ALTER FUNCTION public.isleapyear(year integer) OWNER TO postgres;

--
-- Name: stationalias_check(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.stationalias_check() RETURNS trigger
    LANGUAGE plpgsql
    AS $$DECLARE
	stnalias BOOLEAN;
BEGIN
SELECT (SELECT "StationCode" FROM stations WHERE "StationCode" = new."StationAlias") IS NULL INTO stnalias;
IF stnalias THEN
    RETURN NEW;
ELSE
	RAISE EXCEPTION 'Invalid station alias: already exists as a station code';
END IF;
END
$$;


ALTER FUNCTION public.stationalias_check() OWNER TO postgres;

--
-- Name: update_station_timespan(character varying, character varying); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_station_timespan("NetworkCode" character varying, "StationCode" character varying) RETURNS void
    LANGUAGE sql
    AS $_$
update stations set 
"DateStart" = 
    (SELECT MIN("ObservationFYear") as MINN 
     FROM rinex WHERE "NetworkCode" = $1 AND
     "StationCode" = $2),
"DateEnd" = 
    (SELECT MAX("ObservationFYear") as MAXX 
     FROM rinex WHERE "NetworkCode" = $1 AND
     "StationCode" = $2)
WHERE "NetworkCode" = $1 AND "StationCode" = $2
$_$;


ALTER FUNCTION public.update_station_timespan("NetworkCode" character varying, "StationCode" character varying) OWNER TO postgres;

--
-- Name: update_timespan_trigg(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_timespan_trigg() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    update stations set 
"DateStart" = 
    (SELECT MIN("ObservationFYear") as MINN 
     FROM rinex 
     WHERE "NetworkCode" = new."NetworkCode" AND
           "StationCode" = new."StationCode"),
"DateEnd" = 
    (SELECT MAX("ObservationFYear") as MAXX 
     FROM rinex 
     WHERE "NetworkCode" = new."NetworkCode" AND
           "StationCode" = new."StationCode")
WHERE "NetworkCode" = new."NetworkCode" 
  AND "StationCode" = new."StationCode";

           RETURN new;
END;
$$;


ALTER FUNCTION public.update_timespan_trigg() OWNER TO postgres;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: antennas; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.antennas (
    "AntennaCode" character varying(22) NOT NULL,
    "AntennaDescription" character varying
);


ALTER TABLE public.antennas OWNER TO postgres;

--
-- Name: apr_coords; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.apr_coords (
    "NetworkCode" character varying NOT NULL,
    "StationCode" character varying NOT NULL,
    "FYear" numeric,
    x numeric,
    y numeric,
    z numeric,
    sn numeric,
    se numeric,
    su numeric,
    "ReferenceFrame" character varying(20),
    "Year" integer NOT NULL,
    "DOY" integer NOT NULL
);


ALTER TABLE public.apr_coords OWNER TO postgres;

--
-- Name: aws_sync; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.aws_sync (
    "NetworkCode" character varying NOT NULL,
    "StationCode" character varying NOT NULL,
    "StationAlias" character varying(4) NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL,
    sync_date timestamp without time zone
);


ALTER TABLE public.aws_sync OWNER TO postgres;

--
-- Name: earthquakes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.earthquakes (
    date timestamp without time zone NOT NULL,
    lat numeric NOT NULL,
    lon numeric NOT NULL,
    depth numeric,
    mag numeric
);


ALTER TABLE public.earthquakes OWNER TO postgres;

--
-- Name: etm_params_uid_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.etm_params_uid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.etm_params_uid_seq OWNER TO postgres;

--
-- Name: etm_params; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.etm_params (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    soln character varying(10) NOT NULL,
    object character varying(10) NOT NULL,
    terms numeric,
    frequencies numeric[],
    jump_type numeric,
    relaxation numeric[],
    "Year" numeric,
    "DOY" numeric,
    action character varying(1),
    uid integer DEFAULT nextval('public.etm_params_uid_seq'::regclass) NOT NULL
);


ALTER TABLE public.etm_params OWNER TO postgres;

--
-- Name: etms; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.etms (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "Name" character varying(20) NOT NULL,
    "Value" numeric NOT NULL,
    hash integer
);


ALTER TABLE public.etms OWNER TO postgres;

--
-- Name: etmsv2_uid_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.etmsv2_uid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.etmsv2_uid_seq OWNER TO postgres;

--
-- Name: etmsv2; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.etmsv2 (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    soln character varying(10) NOT NULL,
    object character varying(10) NOT NULL,
    t_ref numeric,
    jump_type numeric,
    relaxation numeric[],
    frequencies numeric[],
    params numeric[],
    sigmas numeric[],
    metadata text,
    hash numeric,
    jump_date timestamp without time zone,
    uid integer DEFAULT nextval('public.etmsv2_uid_seq'::regclass) NOT NULL
);


ALTER TABLE public.etmsv2 OWNER TO postgres;

--
-- Name: events_event_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.events_event_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.events_event_id_seq OWNER TO postgres;

--
-- Name: events; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.events (
    event_id bigint DEFAULT nextval('public.events_event_id_seq'::regclass) NOT NULL,
    "EventDate" timestamp without time zone DEFAULT now() NOT NULL,
    "EventType" character varying(6),
    "NetworkCode" character varying(3),
    "StationCode" character varying(4),
    "Year" integer,
    "DOY" integer,
    "Description" text,
    stack text,
    module text,
    node text
);


ALTER TABLE public.events OWNER TO postgres;

--
-- Name: executions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.executions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.executions_id_seq OWNER TO postgres;

--
-- Name: executions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.executions (
    id integer DEFAULT nextval('public.executions_id_seq'::regclass) NOT NULL,
    script character varying(40),
    exec_date timestamp without time zone DEFAULT now()
);


ALTER TABLE public.executions OWNER TO postgres;

--
-- Name: gamit_htc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gamit_htc (
    "AntennaCode" character varying(22) NOT NULL,
    "HeightCode" character varying(5) NOT NULL,
    v_offset numeric,
    h_offset numeric
);


ALTER TABLE public.gamit_htc OWNER TO postgres;

--
-- Name: gamit_soln; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gamit_soln (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "Project" character varying(20) NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL,
    "FYear" numeric,
    "X" numeric,
    "Y" numeric,
    "Z" numeric,
    sigmax numeric,
    sigmay numeric,
    sigmaz numeric,
    "VarianceFactor" numeric,
    sigmaxy numeric,
    sigmayz numeric,
    sigmaxz numeric
);


ALTER TABLE public.gamit_soln OWNER TO postgres;

--
-- Name: gamit_subnets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gamit_subnets (
    "Project" character varying(20) NOT NULL,
    subnet numeric NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL,
    centroid numeric[],
    stations character varying[],
    alias character varying[],
    ties character varying[]
);


ALTER TABLE public.gamit_subnets OWNER TO postgres;

--
-- Name: keys; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.keys (
    "KeyCode" character varying(7) NOT NULL,
    "TotalChars" integer,
    rinex_col_out character varying,
    rinex_col_in character varying(60),
    isnumeric bit(1)
);


ALTER TABLE public.keys OWNER TO postgres;

--
-- Name: locks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.locks (
    filename text NOT NULL,
    "NetworkCode" character varying(3),
    "StationCode" character varying(4) NOT NULL
);


ALTER TABLE public.locks OWNER TO postgres;

--
-- Name: networks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.networks (
    "NetworkCode" character varying NOT NULL,
    "NetworkName" character varying
);


ALTER TABLE public.networks OWNER TO postgres;

--
-- Name: ppp_soln; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ppp_soln (
    "NetworkCode" character varying NOT NULL,
    "StationCode" character varying NOT NULL,
    "X" numeric(12,4),
    "Y" numeric(12,4),
    "Z" numeric(12,4),
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL,
    "ReferenceFrame" character varying(20) NOT NULL,
    sigmax numeric,
    sigmay numeric,
    sigmaz numeric,
    sigmaxy numeric,
    sigmaxz numeric,
    sigmayz numeric,
    hash integer
);


ALTER TABLE public.ppp_soln OWNER TO postgres;

--
-- Name: ppp_soln_excl; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ppp_soln_excl (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL
);


ALTER TABLE public.ppp_soln_excl OWNER TO postgres;

--
-- Name: receivers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.receivers (
    "ReceiverCode" character varying(22) NOT NULL,
    "ReceiverDescription" character varying(22)
);


ALTER TABLE public.receivers OWNER TO postgres;

--
-- Name: rinex; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rinex (
    "NetworkCode" character varying NOT NULL,
    "StationCode" character varying NOT NULL,
    "ObservationYear" numeric NOT NULL,
    "ObservationMonth" numeric NOT NULL,
    "ObservationDay" numeric NOT NULL,
    "ObservationDOY" numeric NOT NULL,
    "ObservationFYear" numeric NOT NULL,
    "ObservationSTime" timestamp without time zone,
    "ObservationETime" timestamp without time zone,
    "ReceiverType" character varying(20),
    "ReceiverSerial" character varying(20),
    "ReceiverFw" character varying(20),
    "AntennaType" character varying(20),
    "AntennaSerial" character varying(20),
    "AntennaDome" character varying(20),
    "Filename" character varying(20),
    "Interval" numeric NOT NULL,
    "AntennaOffset" numeric,
    "Completion" numeric(7,3) NOT NULL
);


ALTER TABLE public.rinex OWNER TO postgres;

--
-- Name: rinex_proc; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.rinex_proc AS
 SELECT rnx."NetworkCode",
    rnx."StationCode",
    rnx."ObservationYear",
    rnx."ObservationMonth",
    rnx."ObservationDay",
    rnx."ObservationDOY",
    rnx."ObservationFYear",
    rnx."ObservationSTime",
    rnx."ObservationETime",
    rnx."ReceiverType",
    rnx."ReceiverSerial",
    rnx."ReceiverFw",
    rnx."AntennaType",
    rnx."AntennaSerial",
    rnx."AntennaDome",
    rnx."Filename",
    rnx."Interval",
    rnx."AntennaOffset",
    rnx."Completion",
    rnx."mI"
   FROM ( SELECT aa."NetworkCode",
            aa."StationCode",
            aa."ObservationYear",
            aa."ObservationMonth",
            aa."ObservationDay",
            aa."ObservationDOY",
            aa."ObservationFYear",
            aa."ObservationSTime",
            aa."ObservationETime",
            aa."ReceiverType",
            aa."ReceiverSerial",
            aa."ReceiverFw",
            aa."AntennaType",
            aa."AntennaSerial",
            aa."AntennaDome",
            aa."Filename",
            aa."Interval",
            aa."AntennaOffset",
            aa."Completion",
            min(aa."Interval") OVER (PARTITION BY aa."NetworkCode", aa."StationCode", aa."ObservationYear", aa."ObservationDOY") AS "mI"
           FROM (public.rinex aa
             LEFT JOIN public.rinex bb ON ((((aa."NetworkCode")::text = (bb."NetworkCode")::text) AND ((aa."StationCode")::text = (bb."StationCode")::text) AND (aa."ObservationYear" = bb."ObservationYear") AND (aa."ObservationDOY" = bb."ObservationDOY") AND (aa."Completion" < bb."Completion"))))
          WHERE (bb."NetworkCode" IS NULL)
          ORDER BY aa."NetworkCode", aa."StationCode", aa."ObservationYear", aa."ObservationDOY", aa."Interval", aa."Completion") rnx
  WHERE (rnx."Interval" = rnx."mI");


ALTER TABLE public.rinex_proc OWNER TO postgres;

--
-- Name: rinex_tank_struct; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rinex_tank_struct (
    "Level" integer NOT NULL,
    "KeyCode" character varying(7)
);


ALTER TABLE public.rinex_tank_struct OWNER TO postgres;

--
-- Name: stationalias; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stationalias (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "StationAlias" character varying(4) NOT NULL
);


ALTER TABLE public.stationalias OWNER TO postgres;

--
-- Name: stationinfo; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stationinfo (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "ReceiverCode" character varying(22) NOT NULL,
    "ReceiverSerial" character varying(22),
    "ReceiverFirmware" character varying(10),
    "AntennaCode" character varying(22) NOT NULL,
    "AntennaSerial" character varying(20),
    "AntennaHeight" numeric(6,4),
    "AntennaNorth" numeric(12,4),
    "AntennaEast" numeric(12,4),
    "HeightCode" character varying,
    "RadomeCode" character varying(7) NOT NULL,
    "DateStart" timestamp without time zone NOT NULL,
    "DateEnd" timestamp without time zone,
    "ReceiverVers" character varying(22),
    "Comments" text
);


ALTER TABLE public.stationinfo OWNER TO postgres;

--
-- Name: stations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stations (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "StationName" character varying(40),
    "DateStart" numeric(7,3),
    "DateEnd" numeric(7,3),
    auto_x numeric,
    auto_y numeric,
    auto_z numeric,
    "Harpos_coeff_otl" text,
    lat numeric,
    lon numeric,
    height numeric,
    max_dist numeric,
    dome character varying(9)
);


ALTER TABLE public.stations OWNER TO postgres;

--
-- Data for Name: antennas; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.antennas ("AntennaCode", "AntennaDescription") FROM stdin;
\.


--
-- Data for Name: apr_coords; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.apr_coords ("NetworkCode", "StationCode", "FYear", x, y, z, sn, se, su, "ReferenceFrame", "Year", "DOY") FROM stdin;
\.


--
-- Data for Name: aws_sync; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.aws_sync ("NetworkCode", "StationCode", "StationAlias", "Year", "DOY", sync_date) FROM stdin;
\.


--
-- Data for Name: earthquakes; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.earthquakes (date, lat, lon, depth, mag) FROM stdin;
\.


--
-- Data for Name: etm_params; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.etm_params ("NetworkCode", "StationCode", soln, object, terms, frequencies, jump_type, relaxation, "Year", "DOY", action, uid) FROM stdin;
\.


--
-- Data for Name: etms; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.etms ("NetworkCode", "StationCode", "Name", "Value", hash) FROM stdin;
\.


--
-- Data for Name: etmsv2; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.etmsv2 ("NetworkCode", "StationCode", soln, object, t_ref, jump_type, relaxation, frequencies, params, sigmas, metadata, hash, jump_date, uid) FROM stdin;
\.


--
-- Data for Name: events; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.events (event_id, "EventDate", "EventType", "NetworkCode", "StationCode", "Year", "DOY", "Description", stack, module, node) FROM stdin;
\.


--
-- Data for Name: executions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.executions (id, script, exec_date) FROM stdin;
\.


--
-- Data for Name: gamit_htc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gamit_htc ("AntennaCode", "HeightCode", v_offset, h_offset) FROM stdin;
\.


--
-- Data for Name: gamit_soln; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gamit_soln ("NetworkCode", "StationCode", "Project", "Year", "DOY", "FYear", "X", "Y", "Z", sigmax, sigmay, sigmaz, "VarianceFactor", sigmaxy, sigmayz, sigmaxz) FROM stdin;
\.


--
-- Data for Name: gamit_subnets; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gamit_subnets ("Project", subnet, "Year", "DOY", centroid, stations, alias, ties) FROM stdin;
\.


--
-- Data for Name: keys; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.keys ("KeyCode", "TotalChars", rinex_col_out, rinex_col_in, isnumeric) FROM stdin;
\.


--
-- Data for Name: locks; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.locks (filename, "NetworkCode", "StationCode") FROM stdin;
\.


--
-- Data for Name: networks; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.networks ("NetworkCode", "NetworkName") FROM stdin;
\.


--
-- Data for Name: ppp_soln; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ppp_soln ("NetworkCode", "StationCode", "X", "Y", "Z", "Year", "DOY", "ReferenceFrame", sigmax, sigmay, sigmaz, sigmaxy, sigmaxz, sigmayz, hash) FROM stdin;
\.


--
-- Data for Name: ppp_soln_excl; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ppp_soln_excl ("NetworkCode", "StationCode", "Year", "DOY") FROM stdin;
\.


--
-- Data for Name: receivers; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.receivers ("ReceiverCode", "ReceiverDescription") FROM stdin;
\.


--
-- Data for Name: rinex; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.rinex ("NetworkCode", "StationCode", "ObservationYear", "ObservationMonth", "ObservationDay", "ObservationDOY", "ObservationFYear", "ObservationSTime", "ObservationETime", "ReceiverType", "ReceiverSerial", "ReceiverFw", "AntennaType", "AntennaSerial", "AntennaDome", "Filename", "Interval", "AntennaOffset", "Completion") FROM stdin;
\.


--
-- Data for Name: rinex_tank_struct; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.rinex_tank_struct ("Level", "KeyCode") FROM stdin;
\.


--
-- Data for Name: stationalias; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.stationalias ("NetworkCode", "StationCode", "StationAlias") FROM stdin;
\.


--
-- Data for Name: stationinfo; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.stationinfo ("NetworkCode", "StationCode", "ReceiverCode", "ReceiverSerial", "ReceiverFirmware", "AntennaCode", "AntennaSerial", "AntennaHeight", "AntennaNorth", "AntennaEast", "HeightCode", "RadomeCode", "DateStart", "DateEnd", "ReceiverVers", "Comments") FROM stdin;
\.


--
-- Data for Name: stations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.stations ("NetworkCode", "StationCode", "StationName", "DateStart", "DateEnd", auto_x, auto_y, auto_z, "Harpos_coeff_otl", lat, lon, height, max_dist, dome) FROM stdin;
\.


--
-- Name: etm_params_uid_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.etm_params_uid_seq', 1, false);


--
-- Name: etmsv2_uid_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.etmsv2_uid_seq', 1, false);


--
-- Name: events_event_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.events_event_id_seq', 2973, true);


--
-- Name: executions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.executions_id_seq', 24, true);


--
-- Name: antennas antennas_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.antennas
    ADD CONSTRAINT antennas_pk PRIMARY KEY ("AntennaCode");


--
-- Name: apr_coords apr_coords_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.apr_coords
    ADD CONSTRAINT apr_coords_pk PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY");


--
-- Name: aws_sync aws_sync_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.aws_sync
    ADD CONSTRAINT aws_sync_pk PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY");


--
-- Name: earthquakes earthquakes_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earthquakes
    ADD CONSTRAINT earthquakes_pk PRIMARY KEY (date, lat, lon);


--
-- Name: etm_params etm_params_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etm_params
    ADD CONSTRAINT etm_params_pk PRIMARY KEY (uid);


--
-- Name: etms etms_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etms
    ADD CONSTRAINT etms_pk PRIMARY KEY ("NetworkCode", "StationCode", "Name");


--
-- Name: etmsv2 etmsv2_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etmsv2
    ADD CONSTRAINT etmsv2_pk PRIMARY KEY (uid);


--
-- Name: events events_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_pk PRIMARY KEY (event_id, "EventDate");


--
-- Name: gamit_htc gamit_htc_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gamit_htc
    ADD CONSTRAINT gamit_htc_pk PRIMARY KEY ("AntennaCode", "HeightCode");


--
-- Name: gamit_soln gamit_soln_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gamit_soln
    ADD CONSTRAINT gamit_soln_pk PRIMARY KEY ("NetworkCode", "StationCode", "Project", "Year", "DOY");


--
-- Name: gamit_subnets gamit_subnets_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gamit_subnets
    ADD CONSTRAINT gamit_subnets_pk PRIMARY KEY ("Project", subnet, "Year", "DOY");


--
-- Name: keys keys_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.keys
    ADD CONSTRAINT keys_pk PRIMARY KEY ("KeyCode");


--
-- Name: locks locks_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.locks
    ADD CONSTRAINT locks_pk PRIMARY KEY (filename);


--
-- Name: networks networks_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.networks
    ADD CONSTRAINT networks_pk PRIMARY KEY ("NetworkCode");


--
-- Name: ppp_soln_excl ppp_soln_excl_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ppp_soln_excl
    ADD CONSTRAINT ppp_soln_excl_pk PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY");


--
-- Name: ppp_soln ppp_soln_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ppp_soln
    ADD CONSTRAINT ppp_soln_pk PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY", "ReferenceFrame");


--
-- Name: receivers receivers_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.receivers
    ADD CONSTRAINT receivers_pk PRIMARY KEY ("ReceiverCode");


--
-- Name: rinex rinex_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rinex
    ADD CONSTRAINT rinex_pk PRIMARY KEY ("NetworkCode", "StationCode", "ObservationYear", "ObservationDOY", "Interval", "Completion");


--
-- Name: rinex_tank_struct rinex_tank_struct_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rinex_tank_struct
    ADD CONSTRAINT rinex_tank_struct_pk PRIMARY KEY ("Level");


--
-- Name: stationalias stationalias_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stationalias
    ADD CONSTRAINT stationalias_pk PRIMARY KEY ("NetworkCode", "StationCode");


--
-- Name: stationinfo stationinfo_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT stationinfo_pk PRIMARY KEY ("NetworkCode", "StationCode", "DateStart");


--
-- Name: stations stations_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stations
    ADD CONSTRAINT stations_pk PRIMARY KEY ("NetworkCode", "StationCode");


--
-- Name: apr_coords apr_coords_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.apr_coords
    ADD CONSTRAINT apr_coords_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode");


--
-- Name: etm_params etm_params_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etm_params
    ADD CONSTRAINT etm_params_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode");


--
-- Name: etms etms_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etms
    ADD CONSTRAINT etms_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: etmsv2 etmsv2_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etmsv2
    ADD CONSTRAINT etmsv2_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: gamit_htc gamit_htc_antennas_antennacode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gamit_htc
    ADD CONSTRAINT gamit_htc_antennas_antennacode_fk FOREIGN KEY ("AntennaCode") REFERENCES public.antennas("AntennaCode") ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: gamit_soln gamit_soln_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gamit_soln
    ADD CONSTRAINT gamit_soln_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: locks locks_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.locks
    ADD CONSTRAINT locks_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: ppp_soln_excl ppp_soln_excl_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ppp_soln_excl
    ADD CONSTRAINT ppp_soln_excl_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: ppp_soln ppp_soln_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ppp_soln
    ADD CONSTRAINT ppp_soln_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: rinex rinex_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rinex
    ADD CONSTRAINT rinex_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: rinex_tank_struct rinex_tank_struct_keys_keycode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rinex_tank_struct
    ADD CONSTRAINT rinex_tank_struct_keys_keycode_fk FOREIGN KEY ("KeyCode") REFERENCES public.keys("KeyCode");


--
-- Name: stationalias stationalias_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stationalias
    ADD CONSTRAINT stationalias_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: stationinfo stationinfo_antennas_antennacode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT stationinfo_antennas_antennacode_fk FOREIGN KEY ("NetworkCode") REFERENCES public.antennas("AntennaCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: stationinfo stationinfo_receivers_receivercode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT stationinfo_receivers_receivercode_fk FOREIGN KEY ("ReceiverCode") REFERENCES public.receivers("ReceiverCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: stationinfo stationinfo_stations_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT stationinfo_stations_networkcode_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: stations stations_networks_networkcode_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stations
    ADD CONSTRAINT stations_networks_networkcode_fk FOREIGN KEY ("NetworkCode") REFERENCES public.networks("NetworkCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- PostgreSQL database dump complete
--

