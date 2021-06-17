# -*- coding: utf-8 -*-
"""
Z3D Collection
==============

    * Collect z3d files into logical scheduled blocks
    * Merge Z3D files into USGS ascii format
    * Collect metadata information
    * make .csv, .xml, .shp files.

Created on Tue Aug 29 16:38:28 2017

@author: jpeacock
"""
# ==============================================================================

from pathlib import Path
import datetime


import urllib as url
import xml.etree.ElementTree as ET

import numpy as np
import scipy.signal as sps
import pandas as pd

from mth5.io import zen

# =============================================================================
class Z3DCollectionError(Exception):
    pass


# =============================================================================
# Collect Z3d files
# =============================================================================
class Z3DCollection(object):
    """
    Collects .z3d files into useful arrays and lists

    ================= ============================= ===========================
    Attribute         Description                   Default
    ================= ============================= ===========================
    chn_order         list of the order of channels [hx, ex, hy, ey, hz]
    meta_notes        extraction of notes from      None
                      the .z3d files
    leap_seconds      number of leap seconds for    16 [2016]
                      a given year
    ================= ============================= ===========================

    ===================== =====================================================
    Methods               Description
    ===================== =====================================================
    get_time_blocks       Get a list of files for each schedule action
    check_sampling_rate   Check the sampling rate a given time block
    check_time_series     Get information for a given time block
    merge_ts              Merge a given schedule block making sure that they
                          line up in time.
    get_chn_order         Get the appropriate channels, in case some are
                          missing
    ===================== =====================================================

    :Example: ::

        >>> import mtpy.usgs.usgs_archive as archive
        >>> z3d_path = r"/Data/Station_00"
        >>> zc = archive.Z3DCollection()
        >>> fn_list = zc.get_time_blocks(z3d_path)

    """

    def __init__(self):

        self.chn_order = ["hx", "ex", "hy", "ey", "hz"]
        self.meta_notes = None
        self.verbose = True
        self._pd_dt_fmt = "%Y-%m-%d %H:%M:%S.%f"
        self._meta_dtype = np.dtype(
            [
                ("comp", "U3"),
                ("start", np.int64),
                ("stop", np.int64),
                ("fn", "U140"),
                ("sampling_rate", np.float32),
                ("latitude", np.float32),
                ("longitude", np.float32),
                ("elevation", np.float32),
                ("ch_azimuth", np.float32),
                ("ch_length", np.float32),
                ("ch_num", np.int32),
                ("ch_sensor", "U10"),
                ("n_samples", np.int32),
                ("t_diff", np.int32),
                ("standard_deviation", np.float32),
                ("station", "U12"),
            ]
        )

    def _empty_meta_arr(self):
        """
        Create an empty pandas Series
        """
        dtype_list = [
            ("station", "U10"),
            ("latitude", np.float),
            ("longitude", np.float),
            ("elevation", np.float),
            ("start", np.int64),
            ("stop", np.int64),
            ("sampling_rate", np.float),
            ("n_chan", np.int),
            ("n_samples", np.int),
            ("instrument_id", "U10"),
            ("collected_by", "U30"),
            ("notes", "U200"),
            ("comp", "U24"),
        ]

        for cc in ["ex", "ey", "hx", "hy", "hz"]:
            for name, n_dtype in self._meta_dtype.fields.items():
                if name in [
                    "station",
                    "latitude",
                    "longitude",
                    "elevation",
                    "sampling_rate",
                    "comp",
                ]:
                    continue
                elif "ch" in name:
                    m_name = name.replace("ch", cc)
                else:
                    m_name = "{0}_{1}".format(cc, name)
                dtype_list.append((m_name, n_dtype[0]))
        ### make an empy data frame, for now just 1 set.
        df = pd.DataFrame(np.zeros(1, dtype=dtype_list))

        ### return a pandas series, easier to access than dataframe
        return df.iloc[0]

    def get_time_blocks(self, z3d_dir):
        """
        Organize z3d files into blocks based on start time and sampling rate
        in the file name.

        .. note:: This assumes the z3d file is named
                  * station_date_time_samplingrate_chn.z3d

        :param z3d_dir: full path to z3d files
        :type z3d_dir: string

        :returns: nested list of files for each time block, sorted by time

        :Example: ::

            >>> import mtpy.usgs.usgs_archive as archive
            >>> zc = archive.Z3DCollection()
            >>> fn_list = zc.get_time_blocks(r"/home/mt_data/station_01")

        """
        z3d_dir = Path(z3d_dir)
        fn_list = z3d_dir.glob("*.z3d")
        merge_list = []

        for fn in fn_list:
            z3d_obj = zen.Zen3D(fn=fn)
            z3d_obj.read_all_info()
            merge_list.append(
                {"fn": fn, "start_date": z3d_obj.zen_schedule, "df": z3d_obj.df}
            )

        if merge_list == []:
            raise Z3DCollectionError((f"No .z3d files in {z3d_dir}"))

        df = pd.DataFrame(merge_list)

        start_list = list(set(df["start_date"]))

        merge_fn_list = []
        for start in start_list:
            merge_fn_list.append(df[df.start_date == start]["fn"].tolist())

        return merge_fn_list

    # ==================================================
    def merge_z3d_block(self, fn_list, decimate=1):
        """
        Merge a block of z3d files into a MTH5.Schedule Object
        
        :param fn_list: list of z3d files from same schedule action
        :type fn_list: list of strings
        
        :returns: Schedule object that contains metadata and TS dataframes
        :rtype: mth5.Schedule
        
        :Example: ::
            >>> zc = archive.Z3DCollection()
            >>> fn_blocks = zc.get_time_blocks(r"/home/mt/station_00")
            >>> sch_obj = zc.merge_z3d(fn_blocks[0])
        """
        # length of file list
        n_fn = len(fn_list)

        ### get empty series to fill
        meta_df = self._empty_meta_arr()

        ### need to get some statistics from the files, sometimes a file can
        ### be corrupt so we can make some of these lists
        lat = np.zeros(n_fn)
        lon = np.zeros(n_fn)
        elev = np.zeros(n_fn)
        station = np.zeros(n_fn, dtype="U12")
        sampling_rate = np.zeros(n_fn)
        zen_num = np.zeros(n_fn, dtype="U6")
        start = []
        stop = []
        n_samples = []
        ts_list = []

        print("-" * 50)
        for ii, fn in enumerate(fn_list):
            z3d_obj = zen.Zen3D(fn)
            try:
                z3d_obj.read_z3d()
            except zen.ZenGPSError:
                print("xxxxxx BAD FILE: Skipping {0} xxxx".format(fn))
                continue
            # get the components from the file
            comp = z3d_obj.metadata.ch_cmp.lower()
            # convert the time index into an integer
            dt_index = z3d_obj.ts_obj.ts.data.index.astype(np.int64) / 10.0 ** 9

            # extract useful data that will be for the full station
            sampling_rate[ii] = z3d_obj.df
            lat[ii] = z3d_obj.latidted
            lon[ii] = z3d_obj.longitude
            elev[ii] = z3d_obj.elevation
            station[ii] = z3d_obj.station
            zen_num[ii] = int(z3d_obj.header.box_number)

            #### get channel setups
            meta_df["comp"] += f"{comp} "
            meta_df[f"{comp}_start"] = dt_index[0]
            meta_df[f"{comp}_stop"] = dt_index[-1]
            start.append(dt_index[0])
            stop.append(dt_index[-1])
            meta_df[f"{comp}_fn"] = fn
            meta_df[f"{comp}_azimuth"] = z3d_obj.metadata.ch_azimuth
            if "e" in comp:
                meta_df[f"{comp}_length"] = z3d_obj.metadata.ch_length
            ### get sensor number
            elif "h" in comp:
                meta_df[f"{comp}_sensor"] = int(z3d_obj.coil_num)
            meta_df[f"{comp}_num"] = ii + 1
            meta_df[f"{comp}_n_samples"] = z3d_obj.ts_obj.ts.shape[0]
            n_samples.append(z3d_obj.ts_obj.ts.shape[0])
            meta_df[f"{comp}_t_diff".format(comp, "t_diff")] = (
                int((dt_index[-1] - dt_index[0]) * z3d_obj.df)
                - z3d_obj.ts_obj.ts.shape[0]
            )
            # give deviation in percent
            meta_df[f"{comp}_standard_deviation"] = 100 * abs(
                z3d_obj.ts_obj.ts.std()[0] / z3d_obj.ts_obj.ts.median()[0]
            )
            try:
                meta_df["notes"] = (
                    z3d_obj.metadata.notes.replace("\r", " ")
                    .replace("\x00", "")
                    .rstrip()
                )
            except AttributeError:
                pass

            ts_list.append(z3d_obj.ts_obj)

        ### fill in meta data for the station
        meta_df.latitude = self._median_value(lat)
        meta_df.longitude = self._median_value(lon)
        meta_df.elevation = get_nm_elev(meta_df.latitude, meta_df.longitude)
        meta_df.station = self._median_value(station)
        meta_df.instrument_id = "ZEN{0}".format(self._median_value(zen_num))
        meta_df.sampling_rate = self._median_value(sampling_rate)

        ### merge time series into a single data base
        sch_obj = self.merge_ts_list(ts_list, decimate=decimate)
        meta_df.start = sch_obj.start_seconds_from_epoch
        meta_df.stop = sch_obj.stop_seconds_from_epoch
        meta_df.n_chan = sch_obj.n_channels
        meta_df.n_samples = sch_obj.n_samples
        ### add metadata DataFrame to the schedule object
        sch_obj.meta_df = meta_df

        return sch_obj

    def check_start_times(self, ts_list, tol=10):
        """
        check to make sure the start times align
        """
        dt_index_list = [
            ts_obj.ts.data.index.astype(np.int64) / 10.0 ** 9 for ts_obj in ts_list
        ]

        ### check for unique start times
        start_list = np.array([dt[0] for dt in dt_index_list])
        starts, counts = np.unique(start_list, return_counts=True)
        if len(np.unique(start_list)) > 1:
            start = starts[np.where(counts == counts.max())][0]
            off_index = np.where(
                (start_list < start - tol) | (start_list > start + tol)
            )[0]
            if len(off_index) > 0:
                for off in off_index:
                    off = int(off)
                    print(
                        "xxx TS for {0} {1} is off xxx".format(
                            ts_list[off].station, ts_list[off].component
                        )
                    )
                    print("xxx Setting time index to match rest of block xxx")
                    ts_list[off].start_time_epoch_sec = start

        dt_index_list = [
            ts_obj.ts.data.index.astype(np.int64) / 10.0 ** 9 for ts_obj in ts_list
        ]
        # get start and stop times
        start = max([dt[0] for dt in dt_index_list])
        stop = min([dt[-1] for dt in dt_index_list])

        return ts_list, start, stop

    def merge_ts_list(self, ts_list, decimate=1):
        """
        Merge time series from a list of TS objects.
        
        Looks for the maximum start time and the minimum stop time to align
        the time series.  Indexed by UTC time.
        
        :param ts_list: list of mtpy.core.ts.TS objects from z3d files
        :type ts_list: list
        
        :param decimate: factor to decimate the data by
        :type decimate: int
        
        :returns: merged time series
        :rtype: mth5.Schedule object
        """
        comp_list = [ts_obj.component.lower() for ts_obj in ts_list]
        df = ts_list[0].sampling_rate

        ts_list, start, stop = self.check_start_times(ts_list)

        ### make start time in UTC
        dt_struct = datetime.datetime.utcfromtimestamp(start)
        start_time_utc = datetime.datetime.strftime(dt_struct, self._pd_dt_fmt)

        # figure out the max length of the array, getting the time difference into
        # seconds and then multiplying by the sampling rate
        max_ts_len = int((stop - start) * df)
        if max_ts_len < 0:
            print("Times are odd start = {0}, stop = {1}".format(start, stop))
            max_ts_len = abs(max_ts_len)

        ts_len = min([ts_obj.ts.size for ts_obj in ts_list] + [max_ts_len])

        if decimate > 1:
            ts_len /= decimate

        ### make an empty pandas dataframe to put data into, seems like the
        ### fastes way so far.
        ts_db = pd.DataFrame(
            np.zeros((ts_len, len(comp_list))), columns=comp_list, dtype=np.float32
        )

        for ts_obj in ts_list:
            comp = ts_obj.component.lower()
            dt_index = ts_obj.ts.data.index.astype(np.int64) / 10 ** 9
            try:
                index_0 = np.where(dt_index == start)[0][0]
            except IndexError:
                try:
                    index_0 = np.where(np.round(dt_index, decimals=4) == start)[0][0]
                    print(
                        "Start time of {0} is off by {1} seconds".format(
                            ts_obj.fn, abs(start - dt_index[index_0])
                        )
                    )
                except IndexError:
                    raise Z3DCollectionError(
                        "Could not find start time {0} in index of {1}".format(
                            start, ts_obj.fn
                        )
                    )
            index_1 = min([ts_len - index_0, ts_obj.ts.shape[0] - index_0])

            ### check to see what the time difference is, should be 0,
            ### but sometimes not, then need to account for that.
            t_diff = ts_len - (index_1 - index_0)
            if decimate > 1:
                ts_db[comp][0 : (ts_len - t_diff) / decimate] = sps.resample(
                    ts_obj.ts.data[index_0:index_1], ts_len, window="hanning"
                )

            else:
                ts_db[comp][0 : ts_len - t_diff] = ts_obj.ts.data[index_0:index_1]

        # reorder the columns
        ts_db = ts_db[self.get_chn_order(comp_list)]

        # set the index to be UTC time
        dt_freq = "{0:.0f}N".format(1.0 / (df) * 1e9)
        dt_index = pd.date_range(
            start=start_time_utc, periods=ts_db.shape[0], freq=dt_freq
        )
        ts_db.index = dt_index

        # schedule_obj = mth5.Schedule()
        # schedule_obj.from_dataframe(ts_db)

        # return schedule_obj

    def get_chn_order(self, chn_list):
        """
        Get the order of the array channels according to the components.

        .. note:: If you want to change the channel order, change the
                  parameter Z3DCollection.chn_order

        :param chn_list: list of channels in data
        :type chn_list: list

        :return: channel order list
        """

        if len(chn_list) == 5:
            return self.chn_order
        else:
            chn_order = []
            for chn_00 in self.chn_order:
                for chn_01 in chn_list:
                    if chn_00.lower() == chn_01.lower():
                        chn_order.append(chn_00.lower())
                        continue

            return chn_order

    def _median_value(self, value_array):
        """
        get the median value from a metadata entry
        """
        try:
            median = np.median(value_array[np.nonzero(value_array)])
        except TypeError:
            median = list(set(value_array))[0]
        if type(median) in [np.bytes_, bytes]:
            median = median.decode()
        return median


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