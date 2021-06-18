# -*- coding: utf-8 -*-
"""
USGS Archive
==============

    * Collect z3d files into logical scheduled blocks
    * Merge Z3D files into USGS ascii format
    * Collect metadata information
    * make .csv, .xml, .shp files.

Created on Tue Aug 29 16:38:28 2017

@author: jpeacock
"""
# ==============================================================================

import os
import time
import datetime
import sys
import glob
from io import StringIO
import collections

import urllib as url
import xml.etree.ElementTree as ET

import numpy as np

import pandas as pd

# for writing shape file
import geopandas as gpd
from shapely.geometry import Point

# science base
import sciencebasepy as sb

# =============================================================================
# data base error
# =============================================================================
class ArchiveError(Exception):
    pass


# =============================================================================
# class for capturing the output to store in a file
# =============================================================================
# this should capture all the print statements
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


# ==============================================================================
# Need a dummy utc time zone for the date time format
# ==============================================================================
class UTC(datetime.tzinfo):
    def utcoffset(self, df):
        return datetime.timedelta(hours=0)

    def dst(self, df):
        return datetime.timedelta(0)

    def tzname(self, df):
        return "UTC"

# =============================================================================
#  define an empty metadata dataframe
# =============================================================================
def create_empty_meta_arr():
    """
    Create an empty pandas Series
    """
    dtypes = [
        ("station", "U10"),
        ("latitude", np.float),
        ("longitude", np.float),
        ("elevation", np.float),
        ("start", np.int64),
        ("stop", np.int64),
        ("sampling_rate", np.float),
        ("n_chan", np.int),
        ("instrument_id", "U10"),
        ("collected_by", "U30"),
        ("notes", "U200"),
    ]
    ch_types = np.dtype(
        [
            ("ch_azimuth", np.float32),
            ("ch_length", np.float32),
            ("ch_num", np.int32),
            ("ch_sensor", "U10"),
            ("n_samples", np.int32),
            ("t_diff", np.int32),
            ("standard_deviation", np.float32),
        ]
    )

    for cc in ["ex", "ey", "hx", "hy", "hz"]:
        for name, n_dtype in ch_types.fields.items():
            if "ch" in name:
                m_name = name.replace("ch", cc)
            else:
                m_name = "{0}_{1}".format(cc, name)
            dtypes.append((m_name, n_dtype[0]))
    ### make an empy data frame, for now just 1 set.
    df = pd.DataFrame(np.zeros(1, dtype=dtypes))

    ### return a pandas series, easier to access than dataframe
    return df.iloc[0]


# =============================================================================
# Get national map elevation data from internet
# =============================================================================
def get_nm_elev(lat, lon):
    """
    Get national map elevation for a given lat and lon.

    Queries the national map website for the elevation value.

    :param lat: latitude in decimal degrees
    :type lat: float

    :param lon: longitude in decimal degrees
    :type lon: float

    :return: elevation (meters)
    :rtype: float

    :Example: ::

        >>> import mtpy.usgs.usgs_archive as archive
        >>> archive.get_nm_elev(35.467, -115.3355)
        >>> 809.12

    .. note:: Needs an internet connection to work.

    """
    nm_url = r"https://nationalmap.gov/epqs/pqs.php?x={0:.5f}&y={1:.5f}&units=Meters&output=xml"
    print(lat, lon)
    # call the url and get the response
    try:
        response = url.request.urlopen(nm_url.format(lon, lat))
    except (url.error.HTTPError, url.request.http.client.RemoteDisconnected):
        print("xxx GET_ELEVATION_ERROR: Could not connect to internet")
        return -666

    # read the xml response and convert to a float
    try:
        info = ET.ElementTree(ET.fromstring(response.read()))
        info = info.getroot()
        nm_elev = 0.0
        for elev in info.iter("Elevation"):
            nm_elev = float(elev.text)
        return nm_elev
    except ET.ParseError as error:
        print(
            "xxx Something wrong with xml elevation for lat = {0:.5f}, lon = {1:.5f}".format(
                lat, lon
            )
        )
        print(error)
        return -666


# =============================================================================
# Functions to analyze csv files
# =============================================================================
def read_pd_series(csv_fn):
    """
    read a pandas series and turn it into a dataframe
    
    :param csv_fn: full path to schedule csv
    :type csv_fn: string
    
    :returns: pandas dataframe 
    :rtype: pandas.DataFrame
    """
    series = pd.read_csv(csv_fn, index_col=0, header=None, squeeze=True)

    return pd.DataFrame(
        dict([(key, [value]) for key, value in zip(series.index, series.values)])
    )


def combine_station_runs(csv_dir):
    """
    combine all scheduled runs into a single data frame
    
    :param csv_dir: full path the station csv files
    :type csv_dir: string
    
    """
    station = os.path.basename(csv_dir)

    csv_fn_list = sorted(
        [
            os.path.join(csv_dir, fn)
            for fn in os.listdir(csv_dir)
            if "runs" not in fn and fn.endswith(".csv")
        ]
    )

    count = 0
    for ii, csv_fn in enumerate(csv_fn_list):
        if ii == 0:
            run_df = read_pd_series(csv_fn)

        else:
            run_df = run_df.append(read_pd_series(csv_fn), ignore_index=True)
            count += 1

    ### replace any None with 0, makes it easier
    try:
        run_df = run_df.replace("None", "0")
    except UnboundLocalError:
        return None, None

    ### make lat and lon floats
    run_df.latitude = run_df.latitude.astype(np.float)
    run_df.longitude = run_df.longitude.astype(np.float)

    ### write combined csv file
    csv_fn = os.path.join(csv_dir, "{0}_runs.csv".format(station))
    run_df.to_csv(csv_fn, index=False)
    return run_df, csv_fn


def summarize_station_runs(run_df):
    """
    summarize all runs into a pandas.Series to be appended to survey df
    
    :param run_df: combined run dataframe for a single station
    :type run_df: pd.DataFrame
    
    :returns: single row data frame with summarized information
    :rtype: pd.Series
    """
    station_dict = collections.OrderedDict()
    for col in run_df.columns:
        if "_fn" in col:
            continue
        if col == "start":
            value = run_df["start"].min()
            start_date = datetime.datetime.utcfromtimestamp(float(value))
            station_dict["start_date"] = start_date.isoformat() + " UTC"
        elif col == "stop":
            value = run_df["stop"].max()
            stop_date = datetime.datetime.utcfromtimestamp(float(value))
            station_dict["stop_date"] = stop_date.isoformat() + " UTC"
        else:
            try:
                value = run_df[col].median()
            except (TypeError, ValueError):
                value = list(set(run_df[col]))[0]
        station_dict[col] = value

    return pd.Series(station_dict)


def combine_survey_csv(survey_dir, skip_stations=None):
    """
    Combine all stations into a single data frame
    
    :param survey_dir: full path to survey directory
    :type survey_dir: string
    
    :param skip_stations: list of stations to skip
    :type skip_stations: list
    
    :returns: data frame with all information summarized
    :rtype: pandas.DataFrame
    
    :returns: full path to csv file
    :rtype: string
    """

    if not isinstance(skip_stations, list):
        skip_stations = [skip_stations]

    count = 0
    for station in os.listdir(survey_dir):
        station_dir = os.path.join(survey_dir, station)
        if not os.path.isdir(station_dir):
            continue
        if station in skip_stations:
            continue

        # get the database and write a csv file
        run_df, run_fn = combine_station_runs(station_dir)
        if run_df is None:
            print("*** No Information for {0} ***".format(station))
            continue
        if count == 0:
            survey_df = pd.DataFrame(summarize_station_runs(run_df)).T
            count += 1
        else:
            survey_df = survey_df.append(pd.DataFrame(summarize_station_runs(run_df)).T)
            count += 1

    survey_df.latitude = survey_df.latitude.astype(np.float)
    survey_df.longitude = survey_df.longitude.astype(np.float)

    csv_fn = os.path.join(survey_dir, "survey_summary.csv")
    survey_df.to_csv(csv_fn, index=False)

    return survey_df, csv_fn


def read_survey_csv(survey_csv):
    """
    Read in a survey .csv file that will overwrite existing metadata
    parameters.

    :param survey_csv: full path to survey_summary.csv file
    :type survey_csv: string

    :return: survey summary database
    :rtype: pandas dataframe
    """
    db = pd.read_csv(survey_csv, dtype={"latitude": np.float, "longitude": np.float})
    for key in ["hx_sensor", "hy_sensor", "hz_sensor"]:
        db[key] = db[key].fillna(0)
        db[key] = db[key].astype(np.int)

    return db


def get_station_info_from_csv(survey_csv, station):
    """
    get station information from a survey .csv file

    :param survey_csv: full path to survey_summary.csv file
    :type survey_csv: string

    :param station: station name
    :type station: string

    :return: station database
    :rtype: pandas dataframe

    .. note:: station must be verbatim for whats in summary.
    """

    db = read_survey_csv(survey_csv)
    try:
        station_index = db.index[db.station == station].tolist()[0]
    except IndexError:
        raise ArchiveError("Could not find {0}, check name".format(station))

    return db.iloc[station_index]


def summarize_log_files(station_dir):
    """
    summarize the log files to see if there are any errors
    """
    lines = []
    summary_fn = os.path.join(station_dir, "archive_log_summary.log")
    for folder in os.listdir(station_dir):
        try:
            log_fn = glob.glob(os.path.join(station_dir, folder, "*.log"))[0]
        except IndexError:
            print("xxx Did not find log file in {0}".format(folder))
            continue
        lines.append("{0}{1}{0}".format("-" * 10, folder))
        with open(log_fn, "r") as fid:
            log_lines = fid.readlines()
            for log_line in log_lines:
                if "xxx" in log_line:
                    lines.append(log_line)
                elif "error" in log_line.lower():
                    lines.append(log_line)

        with open(summary_fn, "w") as fid:
            fid.write("\n".join(lines))

    return summary_fn


def write_shp_file(survey_csv_fn, save_path=None):
    """
        write a shape file with important information

        :param survey_csv_fn: full path to survey_summary.csv
        :type survey_csf_fn: string

        :param save_path: directory to save shape file to
        :type save_path: string

        :return: full path to shape files
        :rtype: string
        """
    if save_path is not None:
        save_fn = save_path
    else:
        save_fn = os.path.join(os.path.dirname(survey_csv_fn), "survey_sites.shp")

    survey_db = pd.read_csv(survey_csv_fn)
    geometry = [Point(x, y) for x, y in zip(survey_db.longitude, survey_db.latitude)]
    crs = {"init": "epsg:4326"}
    # survey_db = survey_db.drop(['latitude', 'longitude'], axis=1)
    survey_db = survey_db.rename(
        columns={
            "collected_by": "operator",
            "instrument_id": "instr_id",
            "station": "siteID",
        }
    )

    # list of columns to take from the database
    col_list = [
        "siteID",
        "latitude",
        "longitude",
        "elevation",
        "hx_azimuth",
        "hy_azimuth",
        "hz_azimuth",
        "hx_sensor",
        "hy_sensor",
        "hz_sensor",
        "ex_length",
        "ey_length",
        "ex_azimuth",
        "ey_azimuth",
        "n_chan",
        "instr_id",
        "operator",
        "type",
        "quality",
        "start_date",
        "stop_date",
    ]

    survey_db = survey_db[col_list]

    geo_db = gpd.GeoDataFrame(survey_db, crs=crs, geometry=geometry)

    geo_db.to_file(save_fn)

    print("*** Wrote survey shapefile to {0}".format(save_fn))
    return survey_db, save_fn


# =============================================================================
# Science Base Functions
# =============================================================================
def sb_locate_child_item(sb_session, station, sb_page_id):
    """
    See if there is a child item already for the given station.  If there is
    not an existing child item returns False.

    :param sb_session: sciencebase session object
    :type sb_session: sciencebasepy.SbSession

    :param station: station to archive
    :type station: string

    :param sb_page_id: page id of the sciencebase database to download to
    :type sb_page_id: string

    :returns: page id for the station child item
    :rtype: string or False
    """
    for item_id in sb_session.get_child_ids(sb_page_id):
        ### for some reason there is a child item that doesn't play nice
        ### so have to skip it
        try:
            item_title = sb_session.get_item(item_id, {"fields": "title"})["title"]
        except:
            continue
        if station in item_title:
            return item_id

    return False


def sb_sort_fn_list(fn_list):
    """
    sort the file name list to xml, edi, png

    :param fn_list: list of files to sort
    :type fn_list: list

    :returns: sorted list ordered by xml, edi, png, zip files
    """

    fn_list_sort = [None, None, None]
    index_dict = {"xml": 0, "edi": 1, "png": 2}

    for ext in ["xml", "edi", "png"]:
        for fn in fn_list:
            if fn.endswith(ext):
                fn_list_sort[index_dict[ext]] = fn
                fn_list.remove(fn)
                break
    fn_list_sort += sorted(fn_list)

    # check to make sure all the files are there
    if fn_list_sort[0] is None:
        print("\t\t!! No .xml file found !!")
    if fn_list_sort[1] is None:
        print("\t\t!! No .edi file found !!")
    if fn_list_sort[2] is None:
        print("\t\t!! No .png file found !!")

    # get rid of any Nones in the list in case there aren't all the files
    fn_list_sort[:] = (value for value in fn_list_sort if value is not None)

    return fn_list_sort


def sb_session_login(sb_session, sb_username, sb_password=None):
    """
    login in to sb session using the input credentials.  Checks to see if
    you are already logged in.  If no password is given, the password will be
    requested through the command prompt.

    .. note:: iPython shells will echo your password.  Use a Python command
              shell to not have your password echoed.

    :param sb_session: sciencebase session object
    :type sb_session: sciencebasepy.SbSession

    :param sb_username: sciencebase username, typically your full USGS email
    :type sb_username: string

    :param sb_password: AD password
    :type sb_password: string

    :returns: logged in sciencebasepy.SbSession
    """

    if not sb_session.is_logged_in():
        if sb_password is None:
            sb_session.loginc(sb_username)
        else:
            sb_session.login(sb_username, sb_password)
        time.sleep(5)

    return sb_session


def sb_get_fn_list(archive_dir, f_types=[".zip", ".edi", ".png", ".xml", ".mth5"]):
    """
    Get the list of files to archive looking for .zip, .edi, .png within the
    archive directory.  Sorts in the order of xml, edi, png, zip

    :param archive_dir: full path to the directory to be archived
    :type archive_dir: string

    :returns: list of files to archive ordered by xml, edi, png, zip

    """
    fn_list = []
    for f_type in f_types:
        fn_list += glob.glob(os.path.join(archive_dir, "*{0}".format(f_type)))

    return sb_sort_fn_list(fn_list)


def sb_upload_data(
    sb_page_id,
    archive_station_dir,
    sb_username,
    sb_password=None,
    f_types=[".zip", ".edi", ".png", ".xml", ".mth5"],
    child_xml=None,
):
    """
    Upload a given archive station directory to a new child item of the given
    sciencebase page.

    .. note:: iPython shells will echo your password.  Use a Python command
              shell to not have your password echoed.


    :param sb_page_id: page id of the sciencebase database to download to
    :type sb_page_id: string

    :param archive_station_dir: full path to the station directory to archive
    :type archive_station_dir: string

    :param sb_username: sciencebase username, typically your full USGS email
    :type sb_username: string

    :param sb_password: AD password
    :type sb_password: string

    :returns: child item created on the sciencebase page
    :rtype: dictionary

    :Example: ::
        >>> import mtpy.usgs.usgs_archive as archive
        >>> sb_page = 'sb_page_number'
        >>> child_item = archive.sb_upload_data(sb_page,
                                                r"/home/mt/archive_station",
                                                'jdoe@usgs.gov')
    """
    ### initialize a session
    session = sb.SbSession()

    ### login to session, note if you run this in a console your password will
    ### be visible, otherwise run from a command line > python sciencebase_upload.py
    sb_session_login(session, sb_username, sb_password)

    station = os.path.basename(archive_station_dir)

    ### File to upload
    if child_xml:
        f_types.remove(".xml")
    upload_fn_list = sb_get_fn_list(archive_station_dir, f_types=f_types)

    ### check if child item is already created
    child_id = sb_locate_child_item(session, station, sb_page_id)
    ## it is faster to remove the child item and replace it all
    if child_id:
        session.delete_item(session.get_item(child_id))
        sb_action = "Updated"

    else:
        sb_action = "Created"

    ### make a new child item
    new_child_dict = {
        "title": "station {0}".format(station),
        "parentId": sb_page_id,
        "summary": "Magnetotelluric data",
    }
    new_child = session.create_item(new_child_dict)

    if child_xml:
        child_xml.update_child(new_child)
        child_xml.save(os.path.join(archive_station_dir, f"{station}.xml"))
        upload_fn_list.append(child_xml.fn.as_posix())

    # sort list so that xml, edi, png, zip files
    # upload data
    try:
        item = session.upload_files_and_upsert_item(new_child, upload_fn_list)
    except:
        sb_session_login(session, sb_username, sb_password)
        # if you want to keep the order as read on the sb page,
        # need to reverse the order cause it sorts by upload date.
        for fn in upload_fn_list[::-1]:
            try:
                item = session.upload_file_to_item(new_child, fn)
            except:
                print("\t +++ Could not upload {0} +++".format(fn))
                continue

    print("==> {0} child for {1}".format(sb_action, station))

    session.logout()

    return item
