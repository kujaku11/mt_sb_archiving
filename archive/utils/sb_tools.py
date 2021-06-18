# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:46:42 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
from pathlib import Path
import time
import sciencebasepy as sb

import urllib as url
import xml.etree.ElementTree as ET

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


def sb_get_fn_list(archive_dir, f_types=["zip", "edi", "png", "xml", "h5"]):
    """
    Get the list of files to archive looking for .zip, .edi, .png within the
    archive directory.  Sorts in the order of xml, edi, png, zip

    :param archive_dir: full path to the directory to be archived
    :type archive_dir: string

    :returns: list of files to archive ordered by xml, edi, png, zip

    """
    fn_list = []
    for f_type in f_types:
        fn_path = Path(archive_dir)
        fn_list += fn_path.glob(f"*.{f_type}")

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
    archive_station_dir = Path(archive_station_dir)
    station = archive_station_dir.parent

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
        child_xml.save(archive_station_dir.joinpath(f"{station}.xml"))
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