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
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd

# for writing shape file
import geopandas as gpd
from shapely.geometry import Point

from mth5.io import zen

# from mth5.io.reader import read_file
from mth5.timeseries import RunTS

from mt_metadata.timeseries.filters import FrequencyResponseTableFilter
from mt_metadata.utils.mttime import MTTimeError

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

    def __init__(self, z3d_path=None):
        self.z3d_path = z3d_path
        self.chn_order = ["hx", "ex", "hy", "ey", "hz"]

        self._keys_dict = {
            "station": "station",
            "start": "start",
            "end": "end",
            "sample_rate": "sample_rate",
            "component": "component",
            "fn_z3d": "fn",
            "azimuth": "azimuth",
            "dipole_length": "dipole_length",
            "coil_number": "coil_num",
            "latitude": "latitude",
            "longitude": "longitude",
            "elevation": "elevation",
            "n_samples": "n_samples",
            "block": "block",
            "run": "run",
            "zen_num": "zen_num",
            "cal_fn": "cal_fn",
            "channel_number": "channel_number",
        }

        self._dtypes = {
            "station": str,
            "start": str,
            "end": str,
            "sample_rate": float,
            "component": str,
            "fn_z3d": str,
            "azimuth": float,
            "dipole_length": float,
            "coil_number": str,
            "latitude": float,
            "longitude": float,
            "elevation": float,
            "n_samples": int,
            "block": int,
            "run": int,
            "zen_num": str,
            "cal_fn": str,
            "channel_number": str,
            "operator": str,
            "quality": int,
        }

        self._summary_dtypes = {
            "station": str,
            "start": str,
            "end": str,
            "latitude": float,
            "longitude": float,
            "elevation": float,
            "components": str,
            "n_runs": int,
            "n_runs_4096": int,
            "n_runs_1024": int,
            "n_runs_256": int,
            "ex_length": float,
            "ex_azimuth": float,
            "ex_ch_num": int,
            "ex_cres_start": float,
            "ex_cres_end": float,
            "ex_id": str,
            "ey_length": float,
            "ey_azimuth": float,
            "ey_ch_num": int,
            "ey_cres_start": float,
            "ey_cres_end": float,
            "ey_id": str,
            "hx_sensor": str,
            "hx_azimuth": float,
            "hx_ch_num": int,
            "hx_cal_fn": str,
            "hy_sensor": str,
            "hy_azimuth": float,
            "hy_ch_num": int,
            "hy_cal_fn": str,
            "hz_sensor": str,
            "hz_azimuth": float,
            "hz_ch_num": int,
            "hz_cal_fn": str,
            "sample_rates": str,
            "data_logger": str,
            "notes": str,
            "battery": str,
            "battery_start": float,
            "battery_end": float,
            "quality": int,
            "n_chan": int,
            "operator": str,
            "type": str,
        }

        self._csv_ch_translation = {
            "ex_length": "ex.dipole_length",
            "ex_azimuth": "ex.measurement_azimuth",
            "ex_ch_num": "ex.channel_number",
            "ex_cres_start": "ex.contact_resistance.start",
            "ex_cres_end": "ex.contact_resistance.end",
            "ex_id": ["ex.positive.id", "ex.negative.id"],
            "ey_azimuth": "ey.measurement_azimuth",
            "ey_ch_num": "ey.channel_number",
            "ey_cres_start": "ey.contact_resistance.start",
            "ey_cres_end": "ey.contact_resistance.end",
            "ey_id": ["ey.positive.id", "ey.negative.id"],
            "hx_sensor": "hx.sensor.id",
            "hx_azimuth": "hx.measurement_azimuth",
            "hx_ch_num": "hx.channel_number",
            "hy_sensor": "hy.sensor.id",
            "hy_azimuth": "hy.measurement_azimuth",
            "hy_ch_num": "hy.channel_number",
            "hz_sensor": "hz.sensor.id",
            "hz_azimuth": "hz.measurement_azimuth",
            "hz_ch_num": "hz.channel_number",
            "data_logger": "run.data_logger.id",
            "battery": "run.data_logger.power_source.id",
            "battery_start": "run.data_logger.power_source.voltage.start",
            "battery_end": "run.data_logger.power_source.voltage.end",
            "operator": [
                "run.acquired_by.author",
                "station.acquired_by.author",
                "run.metadata_by.author",
            ],
            "type": ["station.data_type", "run.data_type"],
        }

    @property
    def z3d_path(self):
        """
        Path object to z3d directory
        """
        return self._z3d_path

    @z3d_path.setter
    def z3d_path(self, z3d_path):
        """
        :param z3d_path: path to z3d files
        :type z3d_path: string or Path object

        sets z3d_path as a Path object
        """
        if z3d_path is None:
            self._z3d_path = None
        else:
            self._z3d_path = Path(z3d_path)

    def get_z3d_fn_list(self, z3d_path=None):
        """
        Get a list of z3d files in a given directory

        :param z3d_path: Path to z3d files
        :type z3d_path: [ str | pathlib.Path object]
        :return: list of z3d files
        :rtype: list

        :Example: ::

            >>> zc = Z3DCollection()
            >>> z3d_fn_list = zc.get_z3d_fn_list(z3d_path=r"/home/z3d_files")
        """
        if z3d_path is not None:
            self.z3d_path = z3d_path
        if not self.z3d_path.exists():
            raise ValueError(
                "Error: Directory {0} does not exist".format(self.z3d_path)
            )

        z3d_list = [
            fn_path
            for fn_path in self.z3d_path.rglob("*")
            if fn_path.suffix in [".z3d", ".Z3D"]
        ]
        return z3d_list

    def get_calibrations(self, calibration_path):
        """
        get coil calibrations
        """
        if calibration_path is None:
            print("WARNING: Calibration path is None")
            return {}

        if not isinstance(calibration_path, Path):
            calibration_path = Path(calibration_path)

        if not calibration_path.exists():
            print(
                "WARNING: could not find calibration path: "
                "{0}".format(calibration_path)
            )
            return {}

        calibration_dict = {}
        for cal_fn in calibration_path.glob("*.csv"):
            cal_num = cal_fn.stem
            calibration_dict[cal_num] = cal_fn

        return calibration_dict

    def get_z3d_df(self, z3d_path=None, calibration_path=None):
        """
        Get general z3d information and put information in a dataframe

        :param z3d_fn_list: List of files Paths to z3d files
        :type z3d_fn_list: list

        :return: Dataframe of z3d information
        :rtype: Pandas.DataFrame

        :Example: ::

            >>> zc_obj = zc.Z3DCollection(r"/home/z3d_files")
            >>> z3d_fn_list = zc.get_z3d_fn_list()
            >>> z3d_df = zc.get_z3d_info(z3d_fn_list)
            >>> # write dataframe to a file to use later
            >>> z3d_df.to_csv(r"/home/z3d_files/z3d_info.csv")

        """
        z3d_fn_list = self.get_z3d_fn_list(z3d_path)

        if len(z3d_fn_list) < 1:
            raise ValueError("No Z3D files found")

        cal_dict = self.get_calibrations(calibration_path)
        z3d_info_list = []
        for z3d_fn in z3d_fn_list:
            z3d_obj = zen.Z3D(z3d_fn)
            try:
                z3d_obj.read_all_info()
                # z3d_obj.start = z3d_obj.zen_schedule.isoformat()
                # set some attributes to null to fill later
                z3d_obj.n_samples = 0
                z3d_obj.block = 0
                z3d_obj.run = -666
                z3d_obj.zen_num = z3d_obj.header.data_logger
                try:
                    z3d_obj.cal_fn = cal_dict[z3d_obj.coil_num]
                except KeyError:
                    z3d_obj.cal_fn = 0
                # make a dictionary of values to put into data frame
                entry = dict(
                    [
                        (key, getattr(z3d_obj, value))
                        for key, value in self._keys_dict.items()
                    ]
                )
                try:
                    entry["coil_number"] = int(entry["coil_number"])
                except (ValueError, TypeError):
                    entry["coil_number"] = 0
                entry["start"] = z3d_obj.zen_schedule.isoformat()
                entry["operator"] = z3d_obj.metadata.gdp_operator
                entry["quality"] = 0
                z3d_info_list.append(entry)
            except MTTimeError:
                print(f"WARNING: Skipping {z3d_fn}")

        # make pandas dataframe and set data types
        z3d_df = pd.DataFrame(z3d_info_list)
        z3d_df = z3d_df.astype(self._dtypes)
        z3d_df.start = pd.to_datetime(z3d_df.start, errors="coerce")
        z3d_df.end = pd.to_datetime(z3d_df.end, errors="coerce")

        # assign block numbers
        for sr in z3d_df.sample_rate.unique():
            starts = sorted(z3d_df[z3d_df.sample_rate == sr].start.unique())
            for block_num, start in enumerate(starts):
                z3d_df.loc[(z3d_df.start == start), "block"] = block_num

        # assign run number
        for ii, start in enumerate(sorted(z3d_df.start.unique())):
            z3d_df.loc[(z3d_df.start == start), "run"] = ii

        return z3d_df

    def make_runts(
        self, run_df, logger_file_handler=None, config_dict={}, survey_csv_fn=None
    ):
        """
        Create a RunTS object given a Dataframe of channels

        :param run_df: DESCRIPTION
        :type run_df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        ch_list = []
        filter_object_list = []
        for entry in run_df.itertuples():
            ch_obj = zen.read_z3d(entry.fn_z3d, logger_file_handler=logger_file_handler)

            if survey_csv_fn:
                cfg_dict = self.get_station_from_csv(survey_csv_fn, entry.station)
                config_dict[ch_obj.component].update(cfg_dict[ch_obj.component])
                config_dict["run"].update(cfg_dict["run"])
                config_dict["station"].update(cfg_dict["station"])

            try:
                ch_dict = config_dict[ch_obj.component]
                ch_obj.channel_metadata.from_dict(ch_dict, skip_none=True)
            except KeyError:
                pass

            if entry.cal_fn not in [0, "0"]:
                if not ch_obj.channel_response_filter:
                    fap_obj = self._make_fap_filter(entry.cal_fn)
                    filter_object_list.append(fap_obj)
                    ch_obj.channel_metadata.filter.name.append(fap_obj.name)
                    ch_obj.channel_metadata.filter.applied.append(False)

            ch_obj.run_metadata.id = f"{run_df.run.unique()[0]:03d}"
            ch_list.append(ch_obj)

            filter_object_list += ch_obj.channel_response_filter.filters_list

        run_obj = RunTS(array_list=ch_list)
        try:
            run_dict = config_dict["run"]
            run_obj.run_metadata.from_dict(run_dict, skip_none=True)
        except KeyError:
            pass

        try:
            station_dict = config_dict["station"]
            run_obj.station_metadata.from_dict(station_dict, skip_none=True)
        except KeyError:
            pass

        return run_obj, filter_object_list

    def _make_fap_filter(self, cal_fn):
        """
        make a FAP filter object from calibration file
        """

        cal_fn = Path(cal_fn)
        if not cal_fn.exists():
            raise IOError(f"Could not find {cal_fn}")

        fap_df = pd.read_csv(cal_fn)
        fap = FrequencyResponseTableFilter()
        fap.units_in = "millivolts"
        fap.units_out = "nanotesla"
        fap.name = f"ant4_{cal_fn.stem}_response"
        fap.amplitudes = np.sqrt(fap_df.real ** 2 + fap_df.imaginary ** 2).to_numpy()
        fap.phases = np.rad2deg(np.arctan2(fap_df.imaginary, fap_df.real).to_numpy())
        fap.frequencies = fap_df.frequency.to_numpy()

        return fap

    def summarize_survey(self, z3d_path=None, calibration_path=None, output_fn=None):
        """
        summarize a directory of z3d files into a survey summary
        with one entry per station.

        Parameters
        ----------
        z3d_path : TYPE, optional
            DESCRIPTION. The default is None.
        output_fn : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if z3d_path is not None:
            self.z3d_path = Path(z3d_path)
        survey_df = self.get_z3d_df(calibration_path=calibration_path)

        entry_list = []
        for station in survey_df.station.unique():
            station_df = survey_df.loc[survey_df.station == station]
            entry = {"station": station}
            entry["start"] = station_df.start.min()
            entry["end"] = station_df.end.max()
            entry["latitude"] = station_df.latitude.median()
            entry["longitude"] = station_df.longitude.median()
            entry["elevation"] = station_df.elevation.median()
            entry["components"] = ",".join(list(station_df.component.unique()))
            entry["n_runs"] = len(station_df.run.unique())
            for sr in [4096, 1024, 256]:
                entry[f"n_runs_{sr}"] = len(
                    station_df.loc[station_df.sample_rate == sr].block.unique()
                )

            for ec in ["ex", "ey"]:
                e_df = station_df.loc[station_df.component == ec]
                if len(e_df) == 0:
                    print(f"WARNING: No {ec} information for {station}")
                entry[f"{ec}_length"] = e_df.dipole_length.median()
                entry[f"{ec}_azimuth"] = e_df.azimuth.median()
                entry[f"{ec}_ch_num"] = np.nan_to_num(e_df.channel_number.median())
                entry[f"{ec}_cres_start"] = 0
                entry[f"{ec}_cres_end"] = 0
                entry[f"{ec}_id"] = 0

            for hc in ["hx", "hy", "hz"]:
                h_df = station_df.loc[station_df.component == hc]
                if len(h_df) == 0:
                    print(f"WARNING: No {hc} information for {station}")
                    entry[f"{hc}_sensor"] = 0
                    entry[f"{hc}_azimuth"] = 0
                    entry[f"{hc}_ch_num"] = 0
                    entry[f"{hc}_cal_fn"] = 0
                    continue
                entry[f"{hc}_sensor"] = h_df.coil_number.median()
                entry[f"{hc}_azimuth"] = h_df.azimuth.median()
                entry[f"{hc}_ch_num"] = np.nan_to_num(h_df.channel_number.median())
                entry[f"{hc}_cal_fn"] = h_df.cal_fn.mode()[0]

            entry["sample_rates"] = ",".join(
                [f"{ff}" for ff in station_df.sample_rate.unique()]
            )
            entry["data_logger"] = station_df.zen_num.mode()[0]
            for col in ["notes", "battery"]:
                entry[col] = ""
            for col in ["battery_start", "battery_end"]:
                entry[col] = 0
            entry["quality"] = 0
            entry["operator"] = station_df.operator.mode()[0]
            entry["n_chan"] = len(station_df.component.unique())
            entry["type"] = "WBMT"

            entry_list.append(entry)

        # make pandas dataframe and set data types
        summary_df = pd.DataFrame(entry_list)
        summary_df = summary_df.astype(self._summary_dtypes)
                
        summary_df.start = pd.to_datetime(summary_df.start, errors="coerce")
        summary_df.end = pd.to_datetime(summary_df.end, errors="coerce")

        if output_fn:
            summary_df.to_csv(output_fn, index=False)

        return summary_df

    def get_station_from_csv(self, csv_fn, station):
        df = pd.read_csv(csv_fn)

        sdf = df.loc[df.station == station]
        if len(sdf) == 0:
            print(f"Could not find {station} in {csv_fn}")

        cfg_dict = dict(
            [(k, {}) for k in ["station", "run", "ex", "ey", "hx", "hy", "hz"]]
        )
        entry = [entry for entry in sdf.itertuples()][0]
        for key, metadata_key in self._csv_ch_translation.items():
            entry_value = getattr(entry, key)
            if isinstance(metadata_key, list):
                for mkey in metadata_key:
                    dkey = mkey.split(".", 1)[0]
                    dvalue = mkey.split(".", 1)[1]
                    cfg_dict[dkey][dvalue] = entry_value
            else:
                dkey = metadata_key.split(".", 1)[0]
                dvalue = metadata_key.split(".", 1)[1]
                cfg_dict[dkey][dvalue] = entry_value

        return cfg_dict

    def write_shp_file(self, survey_df, save_path=None):
        """


        Parameters
        ----------
        survey_df : TYPE
            DESCRIPTION.
        save_path : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        survey_db : TYPE
            DESCRIPTION.
        save_fn : TYPE
            DESCRIPTION.

        """

        if save_path is not None:
            save_fn = save_path
        else:
            save_fn = self.z3d_path.joinpath("survey_sites.shp")

        geometry = [
            Point(x, y) for x, y in zip(survey_df.longitude, survey_df.latitude)
        ]
        crs = {"init": "epsg:4326"}
        # survey_db = survey_db.drop(['latitude', 'longitude'], axis=1)
        survey_df = survey_df.rename(
            columns={
                "collected_by": "operator",
                "data_logger": "instr_id",
                "station": "siteID",
                "hz_azimuth": "hz_inclina",
                "start": "start_date",
                "end": "end_date",
            }
        )

        for col in ["start_date", "end_date"]:
            survey_df[col] = survey_df[col].astype(str)

        # list of columns to take from the database
        col_list = [
            "siteID",
            "latitude",
            "longitude",
            "elevation",
            "hx_azimuth",
            "hy_azimuth",
            "hz_inclina",
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
            "end_date",
        ]

        survey_df = survey_df[col_list]

        geo_db = gpd.GeoDataFrame(survey_df, crs=crs, geometry=geometry)

        geo_db.to_file(save_fn)

        print("*** Wrote survey shapefile to {0}".format(save_fn))
        return survey_df, save_fn
