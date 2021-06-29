# -*- coding: utf-8 -*-
"""
==============
USGS Archive
==============

    * Collect z3d files into logical scheduled blocks
    * Merge Z3D files into USGS ascii format
    * Collect metadata information
    * make .csv, .xml, .shp files.

Created on Tue Aug 29 16:38:28 2017

:copyright:
    Jared Peacock (jpeacock@usgs.gov)
    
:license: 
    MIT
"""

# ==============================================================================

import shutil
import datetime
import logging
from pathlib import Path
from configparser import ConfigParser
import pandas as pd

from mth5.mth5 import MTH5
from archive.utils import z3d_collection
from archive import mt_xml
from archive.utils import sb_tools

import urllib as url
import xml.etree.ElementTree as ET

LOG_FORMAT = logging.Formatter(
    "%(asctime)s [line %(lineno)d] %(name)s.%(funcName)s - %(levelname)s: %(message)s"
)
# =============================================================================
# data base error
# =============================================================================


class ArchiveError(Exception):
    pass


# =============================================================================
# Arvhive in Science Base class
# =============================================================================
class SBMTArcive:
    """
    Class to help archive MT data in Science Base or other data repositories

    * survey_name: name of the survey, string
    * survey_dir: directory where all station folders are
    * csv_fn: full path to a CSV file containing information about the
    survey.  If this is input it will overwrite metadata.
    * cfg_fn: Full path to configuration file that contains values to
    fill metadata with.
    * xml_cfg_fn: full path to configuration file that contains
    information for the XML files.
    * xml_root_template: full path to a XML template for the root
    XML file that goes onto the parent page in Science Base.
    * xml_child_template: full path to a XML template for the child
    pages in Science Base.
    * calibration_dir: Full path to a folder containing CSV files for
    ANT4 coils named by the coil number, ex. 4324.csv
    * edi_path: Full path to a folder containing all EDI files for the
    stations.
    * png_path: Full path to a folder containing all PNG files for the
    stations
    * mth5_compression: h5 compression type [ gzip | lfz | szip ]
    * mth5_compression_level: level of compression  [ 0-9 ]
    * mth5_chunks: create data chunks [ True | None | value]
    True with have h5py automaically control the chunking,
    if no compression set to None
    * mth5_shuffle: shuffle the blocks, set to True if using compression
    * mth5_fletcher: check for corruption, set to True if using compression


    """

    def __init__(self, **kwargs):
        self.survey_dir = None
        self.archive_dir = None
        self.survey_df = None

        # compression for the h5 file
        self.mth5_compression = None
        self.mth5_compression_level = None
        self.mth5_chunks = None
        self.mth5_shuffle = None
        self.mth5_fletcher = None

        # conifiguration files
        self.survey_csv_fn = None
        self.mth5_cfg_fn = None
        self.xml_cfg_fn = None

        # Science Base XML files
        self.xml_root_template = None
        self.xml_child_template = None
        self.calibration_dir = None
        self.edi_dir = None
        self.png_dir = None
        self.xml_dir = None
        self.mth5_cfg_dict = {}

        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logger()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.mth5_cfg_fn:
            self.mth5_cfg_dict = self.read_cfg_file(self.mth5_cfg_fn)

    def __str__(self):
        """overwrite the string representation"""
        lines = ["Variables Include:", "-" * 25]
        for k in sorted(self.__dict__.keys()):
            lines.append(f"\t{k} = {getattr(self, k)}")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def __setattr__(self, name, value):
        """
        Overwrite the set attribute to make anything with a fn, dir, path

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if "_dir" in name or "_fn" in name or "_template" in name:
            if value is not None:
                value = Path(value)
        super().__setattr__(name, value)

    def setup_logger(self):
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(LOG_FORMAT)
        # stream_handler.setLevel(logging.WARNING)
        # stream_handler.propagate = False

        # self.logger.addHandler(stream_handler)

    def setup_file_logger(self, station, save_dir):
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        logging_fn = save_dir.joinpath("sb_archiving.log")
        file_handler = logging.FileHandler(filename=logging_fn, mode="w")
        file_handler.setFormatter(LOG_FORMAT)
        file_handler.setLevel(logging.DEBUG)
        file_handler.propagate = False

        self.logger.addHandler(file_handler)

    def read_cfg_file(self, fn):
        """
        Read a configuration file

        Parameters
        ----------
        fn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        fn = Path(fn)
        if not fn.exists():
            raise IOError(f"Could not find {fn}")
        cfg_obj = ConfigParser()
        cfg_obj.read_file(fn.open())
        return cfg_obj._sections

    def get_station_directories(self, survey_dir=None):
        """
        Get a list of station directories in a survey directory

        Parameters
        ----------
        survey_dir : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if survey_dir:
            self.survey_dir = Path(survey_dir)

        station_dir_list = [
            station
            for station in self.survey_dir.iterdir()
            if self.survey_dir.joinpath(station).is_dir()
        ]

        return station_dir_list

    def copy_edi_file(self, edi_fn):
        """


        Parameters
        ----------
        station : TYPE
            DESCRIPTION.
        edi_fn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not edi_fn.exists():
            if self.edi_dir:
                try:
                    shutil.copy(self.edi_dir.joinpath(f"{edi_fn.name}"), edi_fn)
                except Exception as error:
                    self.logger.error(error)

    def copy_png_file(self, png_fn):
        """


        Parameters
        ----------
        station : TYPE
            DESCRIPTION.
        png_fn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not png_fn.exists():
            if self.png_dir:
                try:
                    shutil.copy(self.png_dir.joinpath(f"{png_fn.name}"), png_fn)
                except Exception as error:
                    self.logger.error(error)

    def copy_xml_file(self, xml_fn):
        """


        Parameters
        ----------
        station : TYPE
            DESCRIPTION.
        xml_fn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not xml_fn.exists():
            if self.xml_dir:
                try:
                    shutil.copy(self.xml_dir.joinpath(f"{xml_fn.name}"), xml_fn)
                except Exception as error:
                    # print(error)
                    self.logger.error(error)

    def copy_files_to_archive_dir(self, archive_dir, station):
        """


        Parameters
        ----------
        archive_dir : TYPE
            DESCRIPTION.
        station : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        edi_fn = archive_dir.joinpath(f"{station}.edi")
        self.copy_edi_file(edi_fn)

        png_fn = archive_dir.joinpath(f"{station}.png")
        self.copy_png_file(png_fn)

        xml_fn = archive_dir.joinpath(f"{station}.xml")
        self.copy_xml_file(xml_fn)

    def setup_station_archive_dir(self, station_dir):
        """
        Setup the directory structure for the archive

        survey_dir/Archive/station

        Parameters
        ----------
        station : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        station = station_dir.stem
        if not self.survey_dir:
            self.survey_dir = station_dir.parent
        if not self.archive_dir:
            self.archive_dir = self.survey_dir.joinpath("Archive")
            if not self.archive_dir.exists():
                self.archive_dir.mkdir()

        save_station_dir = self.archive_dir.joinpath(station)
        if not save_station_dir.exists():
            save_station_dir.mkdir()

        self.setup_file_logger(station, save_station_dir)

        return station, save_station_dir

    def make_child_xml(self, run_df, save_station_dir, survey_df=None, **kwargs):
        """
        Make a child XML file for a single station

        Parameters
        ----------
        run_df : TYPE
            DESCRIPTION.
        xml_child_template : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs: TYPE
            Can include xml_child_template, xml_cfg_fn

        Returns
        -------
        None.

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        s_xml = mt_xml.MTSBXML()
        if self.xml_child_template:
            s_xml.read_template_xml(self.xml_child_template)
        if self.xml_cfg_fn:
            s_xml.update_from_config(self.xml_cfg_fn, child=True)

        station = run_df.station.unique()[0]

        s_xml.update_with_station(station)

        # location
        if survey_df is not None:
            s_xml.update_bounding_box(
                survey_df.longitude.max(),
                survey_df.longitude.min(),
                survey_df.latitude.max(),
                survey_df.latitude.min(),
            )
        else:
            s_xml.update_bounding_box(
                run_df.longitude.max(),
                run_df.longitude.min(),
                run_df.latitude.max(),
                run_df.latitude.min(),
            )

        # start and end time
        s_xml.update_time_period(
            run_df.start.min().isoformat(), run_df.end.max().isoformat()
        )

        s_xml.update_metadate()

        # write station xml
        xml_fn = save_station_dir.joinpath(f"{station}.xml")
        s_xml.save(xml_fn)

        return xml_fn

    def make_station_mth5(self, station_dir, **kwargs):
        """
        make an mth5 for a single station

        Parameters
        ----------
        station_dir : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ArchiveError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        station_dir = Path(station_dir)
        
        # get the file names for each block of z3d files if none skip
        zc = z3d_collection.Z3DCollection(station_dir)
        try:
            fn_df = zc.get_z3d_df(calibration_path=self.calibration_dir)
            station, save_station_dir = self.setup_station_archive_dir(station_dir)
        except ValueError as error:
            msg = "folder %s because no Z3D files, %s"
            self.logger.error("folder %s because no Z3D files", station_dir)
            self.logger.error(str(error))
            raise ArchiveError(msg % (station_dir, error))

        self.logger.info("--- Creating MTH5 for %s ---", station)
        # capture output to put into a log file
        station_st = datetime.datetime.now()
        self.logger.info("Started %s at %s", station, station_st)

        # Make MTH5 File
        m = MTH5(
            shuffle=self.mth5_shuffle,
            fletcher32=self.mth5_fletcher,
            compression=self.mth5_compression,
            compression_opts=self.mth5_compression_level,
        )
        mth5_fn = save_station_dir.joinpath(f"{station}.h5")
        m.open_mth5(mth5_fn, "w")
        print(m.dataset_options)
        if not m.h5_is_write:
            msg = "Something went wrong with opening %, check logs"
            self.logger.error(msg, mth5_fn)
            raise ArchiveError(msg % mth5_fn)

        # loop over schedule blocks
        for run_num in fn_df.run.unique():
            run_df = fn_df.loc[fn_df.run == run_num]
            runts_obj, filters_list = zc.make_runts(
                run_df,
                logger_file_handler=self.logger.handlers[-1],
                config_dict=self.mth5_cfg_dict,
                survey_csv_fn=self.survey_csv_fn,
            )
            run_df.loc[:, ("end")] = pd.Timestamp(
                runts_obj.run_metadata.time_period.end
            )
            # add station
            station_group = m.add_station(
                runts_obj.station_metadata.id,
                station_metadata=runts_obj.station_metadata,
            )

            # add run
            run_group = station_group.add_run(
                runts_obj.run_metadata.id, runts_obj.run_metadata
            )
            _ = run_group.from_runts(runts_obj, chunks=self.mth5_chunks)
            run_group.validate_run_metadata()

            # need to update metadata
            station_group.validate_station_metadata()

            for f in filters_list:
                m.filters_group.add_filter(f)

        run_df.to_csv(save_station_dir.joinpath(f"{station}_summary.csv"), index=False)
        # update survey metadata from data and cfg file
        try:
            m.survey_group.update_survey_metadata(
                survey_dict=self.mth5_cfg_dict["survey"]
            )
        except KeyError:
            m.survey_group.update_survey_metadata()

        m.close_mth5()

        station_et = datetime.datetime.now()
        t_diff = (station_et - station_st).total_seconds()
        print(f"Processing Took: {t_diff // 60:0.0f}:{t_diff%60:04.1f} minutes")
        self.logger.info(
            f"Processing Took: {t_diff // 60:0.0f}:{t_diff%60:04.1f} minutes"
        )

        return run_df, mth5_fn

    def upload_data(self, page_id, station_dir, username, password, file_types):
        """
        Upload files to science base

        Parameters
        ----------
        page_id : TYPE
            DESCRIPTION.
        station_dir : TYPE
            DESCRIPTION.
        username : TYPE
            DESCRIPTION.
        file_types : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        try:
            sb_tools.sb_upload_data(
                page_id, 
                station_dir,
                username, 
                password, 
                f_types=file_types,
                child_xml=True,
            )
        except Exception as error:
            msg = "Upload failed %s"
            self.logger.error(msg, error)
            raise ArchiveError(msg % error)

    def archive_stations(
        self,
        station_dir_list,
        summarize=True,
        make_xml=True,
        copy_files=True,
        upload=False,
        page_id=None,
        username=None,
        password=None,
        file_types=[".zip", ".edi", ".png", ".xml", ".h5"],
        **kwargs,
    ):

        if not self.survey_dir:
            self.survey_dir = station_dir_list[0].parent

        if summarize:
            survey_zc = z3d_collection.Z3DCollection(self.survey_dir)
            if self.survey_csv_fn is None:
                self.survey_df = survey_zc.summarize_survey(
                    calibration_path=self.calibration_dir
                )
            else:
                self.survey_df = pd.read_csv(self.survey_csv_fn)
                self.survey_df.start = pd.to_datetime(self.survey_df.start)
                self.survey_df.end = pd.to_datetime(self.survey_df.end)

        archive_dirs = []
        for station_dir in station_dir_list:
            try:
                station_df, station_mth5_fn = self.make_station_mth5(
                    station_dir, **kwargs
                )
                archive_station_dir = station_mth5_fn.parent
                archive_dirs.append(archive_station_dir)
                station = station_df.station.unique()[0]
                if self.survey_df is not None:
                    self.survey_df.loc[
                        self.survey_df.station == station, ("end")
                    ] = pd.Timestamp(station_df.end.max())
                if make_xml:
                    _ = self.make_child_xml(
                        station_df,
                        save_station_dir=archive_station_dir,
                        survey_df=self.survey_df,
                        **kwargs,
                    )
                if copy_files:
                    self.copy_files_to_archive_dir(archive_station_dir, station)
                print("\a")
            except ArchiveError as error:
                print("Skipping %s, %s" % (station_dir.name, error))
                self.logger.warning("Skipping %s, %s", station_dir.name, error)

        if summarize:
            # write csv
            self.survey_df.to_csv(
                self.archive_dir.joinpath("survey_summary.csv"), index=False
            )
            
            ### write shape file
            shp_df, shp_fn = survey_zc.write_shp_file(self.survey_df)

            ### write survey xml
            # adjust survey information to align with data
            survey_xml = mt_xml.MTSBXML()
            if self.xml_root_template:
                survey_xml.read_template_xml(self.xml_root_template)
            if self.xml_cfg_fn:
                survey_xml.update_from_config(self.xml_cfg_fn)

            # location
            survey_xml.update_bounding_box(
                self.survey_df.longitude.min(),
                self.survey_df.longitude.max(),
                self.survey_df.latitude.max(),
                self.survey_df.latitude.min(),
            )

            # dates
            self.survey_df.start = pd.to_datetime(self.survey_df.start)
            self.survey_df.end = pd.to_datetime(self.survey_df.end)
            survey_xml.update_time_period(
                self.survey_df.start.min().isoformat(),
                self.survey_df.end.max().isoformat(),
            )

            # shape file attributes limits
            survey_xml.update_shp_attributes(shp_df)

            ### --> write survey xml file
            survey_xml.save(self.archive_dir.joinpath("parent_page.xml"))

            self.survey_df.to_csv(
                self.archive_dir.joinpath("survey_summary.csv"), index=False
            )

        if upload:
            if not page_id or not username or not password:
                raise ArchiveError("Must input page_id and username")

            for archive_station_dir in archive_dirs:
                self.upload_data(page_id, archive_station_dir, username, file_types)


    def upload_stations(self, page_id, archive_dir, username, password, 
                        file_types=[".zip", ".edi", ".png", ".xml", ".h5"]):
        """
        Upload stations to Science Base
        
        :param page_id: DESCRIPTION
        :type page_id: TYPE
        :param archive_dir: DESCRIPTION
        :type archive_dir: TYPE
        :param username: DESCRIPTION
        :type username: TYPE
        :param password: DESCRIPTION
        :type password: TYPE
        :param file_types: DESCRIPTION, defaults to [".zip", ".edi", ".png", ".xml", ".h5"]
        :type file_types: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(archive_dir, list):
            archive_dirs = archive_dir
        elif isinstance(archive_dir, (Path, str)):
            archive_dirs = [p for p in Path(archive_dir).iterdir() if p.is_dir()]
            
        for archive_station_dir in archive_dirs:
            print(f"Archiving: {archive_station_dir}")
            try:
                self.upload_data(page_id, 
                                 archive_station_dir,
                                 username,
                                 password,
                                 file_types)
                print("\a")
                print(f"Uploaded {archive_station_dir.name}")
            except ArchiveError as error:
                print("Could not archive %s" % error)
                self.logger.warning("Could not archive %s", error)
                