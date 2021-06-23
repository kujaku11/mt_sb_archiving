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

import shutil
import datetime
import logging
from pathlib import Path
from configparser import ConfigParser
import pandas as pd

from mth5.mth5 import MTH5
from archive import archive
from archive.utils import z3d_collection
from archive import mt_xml

import getpass

import urllib as url
import xml.etree.ElementTree as ET


# science base
import sciencebasepy as sb


LOG_FORMAT = logging.Formatter(
    "%(asctime)s [line %(lineno)d] %(name)s.%(funcName)s - %(levelname)s: %(message)s"
)
# =============================================================================
# data base error
# =============================================================================
class ArchiveError(Exception):
    pass

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
        
        ### compression for the h5 file
        self.mth5_compression = None
        self.mth5_compression_level = None
        self.mth5_chunks = None
        self.mth5_shuffle = None
        self.mth5_fletcher = None
        
        # conifiguration files
        self.csv_fn = None  
        self.cfg_fn = None
        self.xml_cfg_fn = None
        
        # Science Base XML files
        self.xml_root_template = None
        self.xml_child_template = None
        self.calibration_dir = None
        self.edi_dir = None
        self.png_dir = None
        self.xml_dir = None
        self.cfg_dict = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logger()
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        if self.cfg_fn:
            self.cfg_dict = self.read_cfg_file(self.cfg_fn)
            
    def __str__(self):
        """ overwrite the string representation """
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
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(LOG_FORMAT)
        stream_handler.setLevel(logging.WARNING)
        stream_handler.propagate = False
        
        self.logger.addHandler(stream_handler)
        
    def setup_file_logger(self, station, save_dir):
        logging_fn = save_dir.joinpath("sb_archiving.log")
        file_handler = logging.FileHandler(filename=logging_fn, 
                                           mode="w")
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
        
    def copy_edi_file(self, station, edi_fn):
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
        if self.edi_dir:
            try:
                shutil.copy(self.edi_dir.joinpath(f"{station}.edi"), edi_fn)
            except Exception as error:
                self.logger.error(error)
            
    def copy_png_file(self, station, png_fn):
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
        if self.png_dir: 
            try:
                shutil.copy(self.png_dir.joinpath(f"{station}.png"), png_fn)
            except Exception as error:
                self.logger.error(error)
                
    def copy_xml_file(self, station, xml_fn):
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
        if self.xml_dir: 
            try:
                shutil.copy(self.xml_dir.joinpath(f"{station}.xml"), xml_fn)
            except Exception as error:
                self.logger.error(error)
                
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
    
    def make_child_xml(self, run_df, survey_df=None, **kwargs):
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
            s_xml.update_from_config(self.xml_cfg_fn)

        s_xml.update_with_station(run_df.station.unique()[0])

        # location
        if survey_df:
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
        s_xml.update_time_period(run_df.start.min().isoformat(),
                                  run_df.end.max().isoformat())

        # write station xml
        s_xml.save(save_station_dir.joinpath(f"{station}.xml"))
        if not make_xml and xml_path:
            shutil.copy(
                xml_path.joinpath(f"{station}.xml"),
                save_station_dir.joinpath, f"{station}.xml",
            )

    def archive_station(self, station_dir, make_xml=True, upload_data=False,
                       survey_df=None, **kwargs):
        """
        Archive a single station

        Parameters
        ----------
        station_dir : TYPE
            DESCRIPTION.
        make_xml : TYPE, optional
            DESCRIPTION. The default is True.
        upload_data : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for k, v in kwargs.items:
            setattr(self, k, v)
        
        station_dir = Path(station_dir)
        station, save_station_dir = self.setup_station_archive_dir(station_dir)
        
        ### get the file names for each block of z3d files if none skip
        zc = z3d_collection.Z3DCollection(station_dir)
        try:
            fn_df = zc.get_z3d_df(calibration_path=self.calibration_dir)
        except ValueError as error:
            msg = "folder %s because no Z3D files"
            self.logger.error("folder %s because no Z3D files", station)
            self.logger.error(str(error))
            raise ArchiveError(msg % station)

        self.logger.info("--- Archiving Station %s ---", station)
        ### capture output to put into a log file
        station_st = datetime.datetime.now()
        self.logger.info("Started archiving %s at %s", station, station_st)
        
        ### copy edi and png into archive director
        edi_fn = save_station_dir.joinpath(f"{station}.edi")
        png_fn = save_station_dir.joinpath(f"{station}.png")
        if not edi_fn.is_file():
            self.copy_edi_file(station, edi_fn)
        if not png_fn.is_file():
            self.copy_png_file(station, png_fn)                
    
        ### Make MTH5 File
        m = MTH5(shuffle=self.mth5_shuffle,
                  fletcher32=self.mth5_fletcher,
                  compression=self.mth5_compression,
                  compression_opts=self.mth5_compression_level)
        mth5_fn = save_station_dir.joinpath(f"{station}.h5")
        m.open_mth5(mth5_fn, "w")
        if not m.h5_is_write:
            msg = "Something went wrong with opening %, check logs"
            self.logger.error(msg, mth5_fn)
            raise ArchiveError(msg % mth5_fn)
    
        ### loop over schedule blocks
        for run_num in fn_df.run.unique():
            run_df = fn_df.loc[fn_df.run == run_num]
            runts_obj, filters_list = zc.make_runts(run_df, 
                                                    self.logger,
                                                    self.cfg_dict)
            run_df.loc[:, ("end")] = pd.Timestamp(runts_obj.run_metadata.time_period.end)
            # add station
            station_group = m.add_station(
                runts_obj.station_metadata.id,
                station_metadata=runts_obj.station_metadata,
            )
    
            # add run
            run_group = station_group.add_run(runts_obj.run_metadata.id,
                                              runts_obj.run_metadata)
            channels = run_group.from_runts(runts_obj, chunks=self.mth5_chunks)
            run_group.validate_run_metadata()
    
            # need to update metadata
            station_group.validate_station_metadata()
            
            for f in filters_list:
                m.filters_group.add_filter(f)
        
        if survey_df:
            survey_df.loc[survey_df.station==runts_obj.station_metadata.id, "end"] = \
                pd.Timestamp(station_group.metadata.time_period.end)
                
        # update survey metadata from data and cfg file
        try:
            m.survey_group.update_survey_metadata(survey_dict=self.cfg_dict["survey"])
        except KeyError:
            m.survey_group.update_survey_metadata()
    
        m.close_mth5()
        ####------------------------------------------------------------------
        #### Make xml file for science base
        ####------------------------------------------------------------------
        # make xml file
        if make_xml:
            s_xml = mt_xml.MTSBXML()
            if self.xml_child_template:
                s_xml.read_template_xml(xml_child_template)
            if xml_cfg_fn:
                s_xml.update_from_config(xml_cfg_fn)
    
            s_xml.update_with_station(station)
    
            # location
            s_xml.update_bounding_box(
                survey_df.longitude.max(),
                survey_df.longitude.min(),
                survey_df.latitude.max(),
                survey_df.latitude.min(),
            )
    
            # start and end time
            s_xml.update_time_period(run_df.start.min().isoformat(),
                                      run_df.end.max().isoformat())
    
            # write station xml
            s_xml.save(save_station_dir.joinpath(f"{station}.xml"))
        if not make_xml and xml_path:
            shutil.copy(
                xml_path.joinpath(f"{station}.xml"),
                save_station_dir.joinpath, f"{station}.xml",
            )
    
        station_et = datetime.datetime.now()
        t_diff = station_et - station_st
        print("Took --> {0:.2f} seconds".format(t_diff.total_seconds()))
    
        ####------------------------------------------------------------------
        #### Upload data to science base
        #### -----------------------------------------------------------------
        if upload_data:
            try:
                archive.sb_upload_data(
                    page_id, save_station_dir, username, password, f_types=upload_files
                )
            except Exception as error:
                print("xxx FAILED TO UPLOAD {0} xxx".format(station))
                print(error)

    

