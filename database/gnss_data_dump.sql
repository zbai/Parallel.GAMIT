--
-- PostgreSQL database dump
--

-- Dumped from database version 13.14
-- Dumped by pg_dump version 14.11 (Ubuntu 14.11-0ubuntu0.22.04.1)

-- Started on 2024-05-22 16:15:25 EDT

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 237 (class 1255 OID 86855)
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
-- TOC entry 238 (class 1255 OID 86856)
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
-- TOC entry 239 (class 1255 OID 86857)
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
-- TOC entry 240 (class 1255 OID 86858)
-- Name: isleapyear(integer); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.isleapyear(year integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
SELECT ($1 % 4 = 0) AND (($1 % 100 <> 0) or ($1 % 400 = 0))
$_$;


ALTER FUNCTION public.isleapyear(year integer) OWNER TO postgres;

--
-- TOC entry 241 (class 1255 OID 86859)
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
-- TOC entry 242 (class 1255 OID 86860)
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
-- TOC entry 243 (class 1255 OID 86861)
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

SET default_table_access_method = heap;

--
-- TOC entry 200 (class 1259 OID 86862)
-- Name: antennas; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.antennas (
    "AntennaCode" character varying(22) NOT NULL,
    "AntennaDescription" character varying
);


ALTER TABLE public.antennas OWNER TO gnss_data_osu;

--
-- TOC entry 201 (class 1259 OID 86868)
-- Name: apr_coords; Type: TABLE; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.apr_coords OWNER TO gnss_data_osu;

--
-- TOC entry 202 (class 1259 OID 86874)
-- Name: aws_sync; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.aws_sync (
    "NetworkCode" character varying NOT NULL,
    "StationCode" character varying NOT NULL,
    "StationAlias" character varying(4) NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL,
    sync_date timestamp without time zone
);


ALTER TABLE public.aws_sync OWNER TO gnss_data_osu;

--
-- TOC entry 232 (class 1259 OID 556348)
-- Name: data_source; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.data_source (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    try_order numeric NOT NULL,
    protocol character varying NOT NULL,
    fqdn character varying NOT NULL,
    username character varying,
    password character varying,
    path character varying,
    format character varying
);


ALTER TABLE public.data_source OWNER TO gnss_data_osu;

--
-- TOC entry 203 (class 1259 OID 86880)
-- Name: earthquakes; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.earthquakes (
    date timestamp without time zone NOT NULL,
    lat numeric NOT NULL,
    lon numeric NOT NULL,
    depth numeric,
    mag numeric,
    strike1 numeric,
    dip1 numeric,
    rake1 numeric,
    strike2 numeric,
    dip2 numeric,
    rake2 numeric,
    id character varying(40),
    location character varying(120)
);


ALTER TABLE public.earthquakes OWNER TO gnss_data_osu;

--
-- TOC entry 204 (class 1259 OID 86886)
-- Name: etm_params_uid_seq; Type: SEQUENCE; Schema: public; Owner: gnss_data_osu
--

CREATE SEQUENCE public.etm_params_uid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.etm_params_uid_seq OWNER TO gnss_data_osu;

--
-- TOC entry 205 (class 1259 OID 86888)
-- Name: etm_params; Type: TABLE; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.etm_params OWNER TO gnss_data_osu;

--
-- TOC entry 206 (class 1259 OID 86895)
-- Name: etmsv2_uid_seq; Type: SEQUENCE; Schema: public; Owner: gnss_data_osu
--

CREATE SEQUENCE public.etmsv2_uid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.etmsv2_uid_seq OWNER TO gnss_data_osu;

--
-- TOC entry 207 (class 1259 OID 86897)
-- Name: etms; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.etms (
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
    uid integer DEFAULT nextval('public.etmsv2_uid_seq'::regclass) NOT NULL,
    stack character varying(20)
);


ALTER TABLE public.etms OWNER TO gnss_data_osu;

--
-- TOC entry 208 (class 1259 OID 86904)
-- Name: events_event_id_seq; Type: SEQUENCE; Schema: public; Owner: gnss_data_osu
--

CREATE SEQUENCE public.events_event_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.events_event_id_seq OWNER TO gnss_data_osu;

--
-- TOC entry 209 (class 1259 OID 86906)
-- Name: events; Type: TABLE; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.events OWNER TO gnss_data_osu;

--
-- TOC entry 210 (class 1259 OID 86914)
-- Name: executions_id_seq; Type: SEQUENCE; Schema: public; Owner: gnss_data_osu
--

CREATE SEQUENCE public.executions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.executions_id_seq OWNER TO gnss_data_osu;

--
-- TOC entry 211 (class 1259 OID 86916)
-- Name: executions; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.executions (
    id integer DEFAULT nextval('public.executions_id_seq'::regclass) NOT NULL,
    script character varying(40),
    exec_date timestamp without time zone DEFAULT now()
);


ALTER TABLE public.executions OWNER TO gnss_data_osu;

--
-- TOC entry 212 (class 1259 OID 86921)
-- Name: gamit_htc; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.gamit_htc (
    "AntennaCode" character varying(22) NOT NULL,
    "HeightCode" character varying(5) NOT NULL,
    v_offset numeric,
    h_offset numeric
);


ALTER TABLE public.gamit_htc OWNER TO gnss_data_osu;

--
-- TOC entry 213 (class 1259 OID 86927)
-- Name: gamit_soln; Type: TABLE; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.gamit_soln OWNER TO gnss_data_osu;

--
-- TOC entry 214 (class 1259 OID 86933)
-- Name: gamit_soln_excl; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.gamit_soln_excl (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "Project" character varying(20) NOT NULL,
    "Year" bigint NOT NULL,
    "DOY" bigint NOT NULL,
    residual numeric
);


ALTER TABLE public.gamit_soln_excl OWNER TO gnss_data_osu;

--
-- TOC entry 215 (class 1259 OID 86939)
-- Name: gamit_stats; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.gamit_stats (
    "Project" character varying(20) NOT NULL,
    subnet numeric NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL,
    "FYear" numeric,
    wl numeric,
    nl numeric,
    nrms numeric,
    relaxed_constrains text,
    max_overconstrained character varying(8),
    updated_apr text,
    iterations numeric,
    node character varying(50),
    execution_time numeric,
    execution_date timestamp without time zone,
    system character(1) NOT NULL
);


ALTER TABLE public.gamit_stats OWNER TO gnss_data_osu;

--
-- TOC entry 216 (class 1259 OID 86945)
-- Name: gamit_subnets; Type: TABLE; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.gamit_subnets OWNER TO gnss_data_osu;

--
-- TOC entry 217 (class 1259 OID 86951)
-- Name: gamit_ztd; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.gamit_ztd (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "Date" timestamp without time zone NOT NULL,
    "Project" character varying(20) NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL,
    "ZTD" numeric NOT NULL,
    model numeric,
    sigma numeric
);


ALTER TABLE public.gamit_ztd OWNER TO gnss_data_osu;

--
-- TOC entry 218 (class 1259 OID 86957)
-- Name: keys; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.keys (
    "KeyCode" character varying(7) NOT NULL,
    "TotalChars" integer,
    rinex_col_out character varying,
    rinex_col_in character varying(60),
    isnumeric bit(1)
);


ALTER TABLE public.keys OWNER TO gnss_data_osu;

--
-- TOC entry 219 (class 1259 OID 86963)
-- Name: locks; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.locks (
    filename text NOT NULL,
    "NetworkCode" character varying(3),
    "StationCode" character varying(4)
);


ALTER TABLE public.locks OWNER TO gnss_data_osu;

--
-- TOC entry 220 (class 1259 OID 86969)
-- Name: networks; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.networks (
    "NetworkCode" character varying NOT NULL,
    "NetworkName" character varying
);


ALTER TABLE public.networks OWNER TO gnss_data_osu;

--
-- TOC entry 221 (class 1259 OID 86975)
-- Name: ppp_soln; Type: TABLE; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.ppp_soln OWNER TO gnss_data_osu;

--
-- TOC entry 222 (class 1259 OID 86981)
-- Name: ppp_soln_excl; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.ppp_soln_excl (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "Year" numeric NOT NULL,
    "DOY" numeric NOT NULL
);


ALTER TABLE public.ppp_soln_excl OWNER TO gnss_data_osu;

--
-- TOC entry 223 (class 1259 OID 86987)
-- Name: receivers; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.receivers (
    "ReceiverCode" character varying(22) NOT NULL,
    "ReceiverDescription" character varying(22)
);


ALTER TABLE public.receivers OWNER TO gnss_data_osu;

--
-- TOC entry 224 (class 1259 OID 86990)
-- Name: rinex; Type: TABLE; Schema: public; Owner: gnss_data_osu
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
    "Filename" character varying(50),
    "Interval" numeric NOT NULL,
    "AntennaOffset" numeric,
    "Completion" numeric(7,3) NOT NULL
);


ALTER TABLE public.rinex OWNER TO gnss_data_osu;

--
-- TOC entry 225 (class 1259 OID 86996)
-- Name: rinex_proc; Type: VIEW; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.rinex_proc OWNER TO gnss_data_osu;

--
-- TOC entry 226 (class 1259 OID 87001)
-- Name: rinex_sources_info; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.rinex_sources_info (
    name character varying(20) NOT NULL,
    fqdn character varying NOT NULL,
    protocol character varying NOT NULL,
    username character varying,
    password character varying,
    path character varying,
    format character varying
);


ALTER TABLE public.rinex_sources_info OWNER TO gnss_data_osu;

--
-- TOC entry 227 (class 1259 OID 87007)
-- Name: rinex_tank_struct; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.rinex_tank_struct (
    "Level" integer NOT NULL,
    "KeyCode" character varying(7)
);


ALTER TABLE public.rinex_tank_struct OWNER TO gnss_data_osu;

--
-- TOC entry 233 (class 1259 OID 556361)
-- Name: sources_formats; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.sources_formats (
    format character varying NOT NULL
);


ALTER TABLE public.sources_formats OWNER TO gnss_data_osu;

--
-- TOC entry 235 (class 1259 OID 556371)
-- Name: sources_servers; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.sources_servers (
    server_id integer NOT NULL,
    protocol character varying NOT NULL,
    fqdn character varying NOT NULL,
    username character varying,
    password character varying,
    path character varying,
    format character varying DEFAULT 'DEFAULT_FORMAT'::character varying NOT NULL,
    CONSTRAINT sources_servers_protocol_check CHECK (((protocol)::text = ANY ((ARRAY['ftp'::character varying, 'http'::character varying, 'sftp'::character varying, 'https'::character varying, 'ftpa'::character varying, 'FTP'::character varying, 'HTTP'::character varying, 'SFTP'::character varying, 'HTTPS'::character varying, 'FTPA'::character varying])::text[])))
);


ALTER TABLE public.sources_servers OWNER TO gnss_data_osu;

--
-- TOC entry 234 (class 1259 OID 556369)
-- Name: sources_servers_server_id_seq; Type: SEQUENCE; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE public.sources_servers ALTER COLUMN server_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.sources_servers_server_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 236 (class 1259 OID 556386)
-- Name: sources_stations; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.sources_stations (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    try_order smallint DEFAULT 1 NOT NULL,
    server_id integer NOT NULL,
    path character varying,
    format character varying
);


ALTER TABLE public.sources_stations OWNER TO gnss_data_osu;

--
-- TOC entry 228 (class 1259 OID 87010)
-- Name: stacks; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.stacks (
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
    sigmaxz numeric,
    name character varying(20) NOT NULL
);


ALTER TABLE public.stacks OWNER TO gnss_data_osu;

--
-- TOC entry 229 (class 1259 OID 87016)
-- Name: stationalias; Type: TABLE; Schema: public; Owner: gnss_data_osu
--

CREATE TABLE public.stationalias (
    "NetworkCode" character varying(3) NOT NULL,
    "StationCode" character varying(4) NOT NULL,
    "StationAlias" character varying(4) NOT NULL
);


ALTER TABLE public.stationalias OWNER TO gnss_data_osu;

--
-- TOC entry 230 (class 1259 OID 87019)
-- Name: stationinfo; Type: TABLE; Schema: public; Owner: gnss_data_osu
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


ALTER TABLE public.stationinfo OWNER TO gnss_data_osu;

--
-- TOC entry 231 (class 1259 OID 87025)
-- Name: stations; Type: TABLE; Schema: public; Owner: gnss_data_osu
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
    dome character varying(9),
    country_code character varying(3),
    marker integer,
    alias character varying(4)
);


ALTER TABLE public.stations OWNER TO gnss_data_osu;

--
-- TOC entry 3019 (class 2606 OID 522474)
-- Name: antennas antennas_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.antennas
    ADD CONSTRAINT antennas_pkey PRIMARY KEY ("AntennaCode");


--
-- TOC entry 3022 (class 2606 OID 522476)
-- Name: apr_coords apr_coords_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.apr_coords
    ADD CONSTRAINT apr_coords_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY");


--
-- TOC entry 3025 (class 2606 OID 522478)
-- Name: aws_sync aws_sync_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.aws_sync
    ADD CONSTRAINT aws_sync_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY");


--
-- TOC entry 3086 (class 2606 OID 556355)
-- Name: data_source data_source_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.data_source
    ADD CONSTRAINT data_source_pkey PRIMARY KEY ("NetworkCode", "StationCode", try_order);


--
-- TOC entry 3014 (class 2606 OID 522479)
-- Name: stationinfo date_chk; Type: CHECK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE public.stationinfo
    ADD CONSTRAINT date_chk CHECK (("DateEnd" > "DateStart")) NOT VALID;


--
-- TOC entry 3027 (class 2606 OID 522481)
-- Name: earthquakes earthquakes_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.earthquakes
    ADD CONSTRAINT earthquakes_pkey PRIMARY KEY (date, lat, lon);


--
-- TOC entry 3030 (class 2606 OID 522483)
-- Name: etm_params etm_params_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.etm_params
    ADD CONSTRAINT etm_params_pkey PRIMARY KEY (uid);


--
-- TOC entry 3032 (class 2606 OID 522485)
-- Name: etms etmsv2_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.etms
    ADD CONSTRAINT etmsv2_pkey PRIMARY KEY (uid);


--
-- TOC entry 3035 (class 2606 OID 522487)
-- Name: events events_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_pkey PRIMARY KEY (event_id, "EventDate");


--
-- TOC entry 3037 (class 2606 OID 522489)
-- Name: gamit_htc gamit_htc_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_htc
    ADD CONSTRAINT gamit_htc_pkey PRIMARY KEY ("AntennaCode", "HeightCode");


--
-- TOC entry 3041 (class 2606 OID 522491)
-- Name: gamit_soln_excl gamit_soln_excl_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_soln_excl
    ADD CONSTRAINT gamit_soln_excl_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Project", "Year", "DOY");


--
-- TOC entry 3039 (class 2606 OID 522493)
-- Name: gamit_soln gamit_soln_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_soln
    ADD CONSTRAINT gamit_soln_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Project", "Year", "DOY");


--
-- TOC entry 3043 (class 2606 OID 565050)
-- Name: gamit_stats gamit_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_stats
    ADD CONSTRAINT gamit_stats_pkey PRIMARY KEY ("Project", subnet, "Year", "DOY", system);


--
-- TOC entry 3045 (class 2606 OID 522497)
-- Name: gamit_subnets gamit_subnets_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_subnets
    ADD CONSTRAINT gamit_subnets_pkey PRIMARY KEY ("Project", subnet, "Year", "DOY");


--
-- TOC entry 3048 (class 2606 OID 522499)
-- Name: gamit_ztd gamit_ztd_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_ztd
    ADD CONSTRAINT gamit_ztd_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Date", "Project", "Year", "DOY");


--
-- TOC entry 3050 (class 2606 OID 522509)
-- Name: keys keys_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.keys
    ADD CONSTRAINT keys_pkey PRIMARY KEY ("KeyCode");


--
-- TOC entry 3052 (class 2606 OID 522511)
-- Name: locks locks_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.locks
    ADD CONSTRAINT locks_pkey PRIMARY KEY (filename);


--
-- TOC entry 3054 (class 2606 OID 522513)
-- Name: networks networks_NetworkCode_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.networks
    ADD CONSTRAINT "networks_NetworkCode_pkey" PRIMARY KEY ("NetworkCode");


--
-- TOC entry 3060 (class 2606 OID 522515)
-- Name: ppp_soln_excl ppp_soln_excl_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.ppp_soln_excl
    ADD CONSTRAINT ppp_soln_excl_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY");


--
-- TOC entry 3058 (class 2606 OID 522517)
-- Name: ppp_soln ppp_soln_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.ppp_soln
    ADD CONSTRAINT ppp_soln_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY", "ReferenceFrame");


--
-- TOC entry 3062 (class 2606 OID 522519)
-- Name: receivers receivers_ReceiverCode_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.receivers
    ADD CONSTRAINT "receivers_ReceiverCode_pkey" PRIMARY KEY ("ReceiverCode");


--
-- TOC entry 3068 (class 2606 OID 522521)
-- Name: rinex rinex_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.rinex
    ADD CONSTRAINT rinex_pkey PRIMARY KEY ("NetworkCode", "StationCode", "ObservationYear", "ObservationDOY", "Interval", "Completion");


--
-- TOC entry 3070 (class 2606 OID 522523)
-- Name: rinex_sources_info rinex_sources_info_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.rinex_sources_info
    ADD CONSTRAINT rinex_sources_info_pkey PRIMARY KEY (name);


--
-- TOC entry 3072 (class 2606 OID 522525)
-- Name: rinex_tank_struct rinex_tank_struct_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.rinex_tank_struct
    ADD CONSTRAINT rinex_tank_struct_pkey PRIMARY KEY ("Level");


--
-- TOC entry 3088 (class 2606 OID 556368)
-- Name: sources_formats sources_formats_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.sources_formats
    ADD CONSTRAINT sources_formats_pkey PRIMARY KEY (format);


--
-- TOC entry 3090 (class 2606 OID 556380)
-- Name: sources_servers sources_servers_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.sources_servers
    ADD CONSTRAINT sources_servers_pkey PRIMARY KEY (server_id);


--
-- TOC entry 3092 (class 2606 OID 556394)
-- Name: sources_stations sources_stations_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.sources_stations
    ADD CONSTRAINT sources_stations_pkey PRIMARY KEY ("NetworkCode", "StationCode", try_order);


--
-- TOC entry 3075 (class 2606 OID 522527)
-- Name: stacks stacks_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stacks
    ADD CONSTRAINT stacks_pkey PRIMARY KEY ("NetworkCode", "StationCode", "Year", "DOY", name);


--
-- TOC entry 3077 (class 2606 OID 522529)
-- Name: stationalias stationalias_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stationalias
    ADD CONSTRAINT stationalias_pkey PRIMARY KEY ("NetworkCode", "StationCode");


--
-- TOC entry 3079 (class 2606 OID 522531)
-- Name: stationalias stationalias_uniq; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stationalias
    ADD CONSTRAINT stationalias_uniq UNIQUE ("StationAlias");


--
-- TOC entry 3081 (class 2606 OID 522533)
-- Name: stationinfo stationinfo_NetworkCode_StationCode_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT "stationinfo_NetworkCode_StationCode_pkey" PRIMARY KEY ("NetworkCode", "StationCode", "DateStart");


--
-- TOC entry 3084 (class 2606 OID 522535)
-- Name: stations stations_NetworkCode_StationCode_pkey; Type: CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stations
    ADD CONSTRAINT "stations_NetworkCode_StationCode_pkey" PRIMARY KEY ("NetworkCode", "StationCode");


--
-- TOC entry 3063 (class 1259 OID 522536)
-- Name: Filename; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX "Filename" ON public.rinex USING btree ("Filename" varchar_ops);


--
-- TOC entry 3020 (class 1259 OID 522537)
-- Name: apr_coords_date_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX apr_coords_date_idx ON public.apr_coords USING btree ("NetworkCode", "StationCode", "Year", "DOY");


--
-- TOC entry 3023 (class 1259 OID 522538)
-- Name: aws_sync_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX aws_sync_idx ON public.aws_sync USING btree ("NetworkCode", "StationCode", "StationAlias", "Year", "DOY");


--
-- TOC entry 3028 (class 1259 OID 522539)
-- Name: etm_params_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX etm_params_idx ON public.etm_params USING btree ("NetworkCode", "StationCode", soln, object);


--
-- TOC entry 3033 (class 1259 OID 522540)
-- Name: events_index; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX events_index ON public.events USING btree ("NetworkCode", "StationCode", "Year", "DOY");


--
-- TOC entry 3046 (class 1259 OID 522541)
-- Name: gamit_ztd_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX gamit_ztd_idx ON public.gamit_ztd USING btree ("Project", "Year", "DOY");


--
-- TOC entry 3064 (class 1259 OID 522542)
-- Name: network_station; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX network_station ON public.rinex USING btree ("NetworkCode" varchar_ops, "StationCode" varchar_ops);


--
-- TOC entry 3055 (class 1259 OID 522543)
-- Name: ppp_soln_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX ppp_soln_idx ON public.ppp_soln USING btree ("NetworkCode" COLLATE "C" varchar_ops, "StationCode" COLLATE "C" varchar_ops);


--
-- TOC entry 3056 (class 1259 OID 522544)
-- Name: ppp_soln_order; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX ppp_soln_order ON public.ppp_soln USING btree ("NetworkCode", "StationCode", "Year", "DOY");


--
-- TOC entry 3065 (class 1259 OID 522545)
-- Name: rinex_obs_comp_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX rinex_obs_comp_idx ON public.rinex USING btree ("NetworkCode", "StationCode", "ObservationYear", "ObservationDOY", "Completion");


--
-- TOC entry 3066 (class 1259 OID 522546)
-- Name: rinex_obs_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX rinex_obs_idx ON public.rinex USING btree ("NetworkCode", "StationCode", "ObservationYear", "ObservationDOY");


--
-- TOC entry 3073 (class 1259 OID 522547)
-- Name: stacks_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE INDEX stacks_idx ON public.stacks USING btree ("Year", "DOY");


--
-- TOC entry 3082 (class 1259 OID 522548)
-- Name: stations_NetworkCode_StationCode_idx; Type: INDEX; Schema: public; Owner: gnss_data_osu
--

CREATE UNIQUE INDEX "stations_NetworkCode_StationCode_idx" ON public.stations USING btree ("NetworkCode", "StationCode");


--
-- TOC entry 3117 (class 2620 OID 522549)
-- Name: rinex update_stations; Type: TRIGGER; Schema: public; Owner: gnss_data_osu
--

CREATE TRIGGER update_stations AFTER INSERT ON public.rinex FOR EACH ROW EXECUTE FUNCTION public.update_timespan_trigg();


--
-- TOC entry 3118 (class 2620 OID 522550)
-- Name: stationalias verify_stationalias; Type: TRIGGER; Schema: public; Owner: gnss_data_osu
--

CREATE TRIGGER verify_stationalias BEFORE INSERT OR UPDATE ON public.stationalias FOR EACH ROW EXECUTE FUNCTION public.stationalias_check();


--
-- TOC entry 3111 (class 2606 OID 522551)
-- Name: stations NetworkCode; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stations
    ADD CONSTRAINT "NetworkCode" FOREIGN KEY ("NetworkCode") REFERENCES public.networks("NetworkCode") MATCH FULL ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3096 (class 2606 OID 522556)
-- Name: gamit_htc antenna_fk; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_htc
    ADD CONSTRAINT antenna_fk FOREIGN KEY ("AntennaCode") REFERENCES public.antennas("AntennaCode") ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3093 (class 2606 OID 522561)
-- Name: apr_coords apr_coords_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.apr_coords
    ADD CONSTRAINT "apr_coords_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode");


--
-- TOC entry 3112 (class 2606 OID 556356)
-- Name: data_source data_source_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.data_source
    ADD CONSTRAINT "data_source_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3095 (class 2606 OID 522566)
-- Name: etms etms_fk; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.etms
    ADD CONSTRAINT etms_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3097 (class 2606 OID 522571)
-- Name: gamit_soln gamit_soln_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_soln
    ADD CONSTRAINT "gamit_soln_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3098 (class 2606 OID 522576)
-- Name: gamit_soln_excl gamit_soln_excl_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_soln_excl
    ADD CONSTRAINT "gamit_soln_excl_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode", "Project", "Year", "DOY") REFERENCES public.gamit_soln("NetworkCode", "StationCode", "Project", "Year", "DOY") ON UPDATE CASCADE ON DELETE CASCADE NOT VALID;


--
-- TOC entry 3099 (class 2606 OID 522581)
-- Name: gamit_ztd gamit_ztd_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.gamit_ztd
    ADD CONSTRAINT "gamit_ztd_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode");


--
-- TOC entry 3100 (class 2606 OID 522586)
-- Name: locks locks_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.locks
    ADD CONSTRAINT "locks_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3101 (class 2606 OID 522591)
-- Name: ppp_soln ppp_soln_NetworkName_StationCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.ppp_soln
    ADD CONSTRAINT "ppp_soln_NetworkName_StationCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3102 (class 2606 OID 522596)
-- Name: ppp_soln_excl ppp_soln_excl_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.ppp_soln_excl
    ADD CONSTRAINT "ppp_soln_excl_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3103 (class 2606 OID 522601)
-- Name: rinex rinex_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.rinex
    ADD CONSTRAINT "rinex_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3104 (class 2606 OID 522606)
-- Name: rinex_tank_struct rinex_tank_struct_key_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.rinex_tank_struct
    ADD CONSTRAINT rinex_tank_struct_key_fkey FOREIGN KEY ("KeyCode") REFERENCES public.keys("KeyCode");


--
-- TOC entry 3113 (class 2606 OID 556381)
-- Name: sources_servers sources_servers_format_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.sources_servers
    ADD CONSTRAINT sources_servers_format_fkey FOREIGN KEY (format) REFERENCES public.sources_formats(format);


--
-- TOC entry 3116 (class 2606 OID 556405)
-- Name: sources_stations sources_stations_NetworkCode_StationCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.sources_stations
    ADD CONSTRAINT "sources_stations_NetworkCode_StationCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode");


--
-- TOC entry 3115 (class 2606 OID 556400)
-- Name: sources_stations sources_stations_format_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.sources_stations
    ADD CONSTRAINT sources_stations_format_fkey FOREIGN KEY (format) REFERENCES public.sources_formats(format);


--
-- TOC entry 3114 (class 2606 OID 556395)
-- Name: sources_stations sources_stations_server_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.sources_stations
    ADD CONSTRAINT sources_stations_server_id_fkey FOREIGN KEY (server_id) REFERENCES public.sources_servers(server_id);


--
-- TOC entry 3105 (class 2606 OID 522611)
-- Name: stacks stacks_NetworkCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stacks
    ADD CONSTRAINT "stacks_NetworkCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3106 (class 2606 OID 522616)
-- Name: stacks stacks_gamit_soln_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stacks
    ADD CONSTRAINT stacks_gamit_soln_fkey FOREIGN KEY ("Year", "DOY", "StationCode", "Project", "NetworkCode") REFERENCES public.gamit_soln("Year", "DOY", "StationCode", "Project", "NetworkCode") ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3107 (class 2606 OID 522621)
-- Name: stationalias stationalias_fk; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stationalias
    ADD CONSTRAINT stationalias_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3108 (class 2606 OID 522626)
-- Name: stationinfo stationinfo_AntennaCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT "stationinfo_AntennaCode_fkey" FOREIGN KEY ("AntennaCode") REFERENCES public.antennas("AntennaCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3109 (class 2606 OID 522631)
-- Name: stationinfo stationinfo_NetworkCode_StationCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT "stationinfo_NetworkCode_StationCode_fkey" FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3110 (class 2606 OID 522636)
-- Name: stationinfo stationinfo_ReceiverCode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.stationinfo
    ADD CONSTRAINT "stationinfo_ReceiverCode_fkey" FOREIGN KEY ("ReceiverCode") REFERENCES public.receivers("ReceiverCode") ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- TOC entry 3094 (class 2606 OID 522641)
-- Name: etm_params stations_fk; Type: FK CONSTRAINT; Schema: public; Owner: gnss_data_osu
--

ALTER TABLE ONLY public.etm_params
    ADD CONSTRAINT stations_fk FOREIGN KEY ("NetworkCode", "StationCode") REFERENCES public.stations("NetworkCode", "StationCode");


-- Completed on 2024-05-22 16:15:25 EDT

--
-- PostgreSQL database dump complete
--

