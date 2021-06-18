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

from mth5.io import zen
from mth5.io.reader import read_file
from mth5.timeseries import RunTS

from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

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
        
        self._keys_dict = {
            "station": "station",
            "start": "start",
            "stop": "stop",
            "sample_rate": "sample_rate",
            "component": "component",
            "fn_z3d": "fn",
            "azimuth": "azimuth",
            "dipole_length": "dipole_len",
            "coil_number": "coil_num",
            "latitude": "latitude",
            "longitude": "longitude",
            "elevation": "elevation",
            "n_samples": "n_samples",
            "block": "block",
            "run": "run",
            "zen_num": "zen_num",
            "cal_fn": "cal_fn",
        }
        
        self._dtypes = {
            "station": str,
            "start": str,
            "stop": str,
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
            print("ERROR: Calibration path is None")
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
            z3d_obj.read_all_info()
            #z3d_obj.start = z3d_obj.zen_schedule.isoformat()
            # set some attributes to null to fill later
            z3d_obj.stop = None
            z3d_obj.n_samples = 0
            z3d_obj.block = 0
            z3d_obj.run = -666
            z3d_obj.zen_num = "ZEN{0:03.0f}".format(z3d_obj.header.box_number)
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
            entry["start"] = z3d_obj.zen_schedule.isoformat()
            z3d_info_list.append(entry)

        # make pandas dataframe and set data types
        z3d_df = pd.DataFrame(z3d_info_list)
        z3d_df = z3d_df.astype(self._dtypes)
        z3d_df.start = pd.to_datetime(z3d_df.start, errors="coerce")
        z3d_df.stop = pd.to_datetime(z3d_df.stop, errors="coerce")

        # assign block numbers
        for sr in z3d_df.sample_rate.unique():
            starts = sorted(z3d_df[z3d_df.sample_rate == sr].start.unique())
            for block_num, start in enumerate(starts):
                z3d_df.loc[(z3d_df.start == start), "block"] = block_num
                
        # assign run number
        for ii, start in enumerate(sorted(z3d_df.start.unique())):
            z3d_df.loc[(z3d_df.start == start), "run"] = ii

        return z3d_df

    def make_runts(self, run_df):
        """
        Create a RunTS object given a Dataframe of channels
        
        :param run_df: DESCRIPTION
        :type run_df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        ch_list = []
        fap_list = []
        for entry in run_df.itertuples():
            ch_obj = read_file(entry.fn_z3d)
            if entry.cal_fn not in [0, "0"]:
                fap_list.append(self._make_fap_filter(entry.cal_fn))
                
            ch_obj.run_metadata.id = f"{run_df.run.unique()[0]:03d}"
            ch_list.append(ch_obj)
        run_obj = RunTS(array_list=ch_list)

        return run_obj, fap_list
    
    def _make_fap_filter(self, cal_fn):
        """
        make a FAP filter object from calibration file
        """
        
        cal_fn = Path(cal_fn)
        if not cal_fn.exists():
            raise IOError(f"Could not find {cal_fn}")
        
        fap_df = pd.read_csv(cal_fn)
        fap = FrequencyResponseTableFilter()
        fap.units_in = "digital counts"
        fap.units_out = "nanotesla"
        fap.name = f"ant4_{cal_fn.stem}_response"
        fap.amplitudes = np.sqrt(fap_df.real**2 + fap_df.imaginary**2).to_numpy()
        fap.phases = np.rad2deg(np.arctan2(fap_df.imaginary, fap_df.real).to_numpy())
        fap.frequencies = fap_df.frequency.to_numpy()
        
        return fap
        
        
        

