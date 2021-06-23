# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:53:51 2018

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
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

# =============================================================================
# Logging
# =============================================================================
LOG_FORMAT = logging.Formatter(
    "%(asctime)s [line %(lineno)d] %(name)s.%(funcName)s - %(levelname)s: %(message)s"
)

archive_logger = logging.getLogger("sb_archiving")
archive_logger.setLevel(logging.DEBUG)
archive_logger.propagate = False

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(LOG_FORMAT)
stream_handler.setLevel(logging.INFO)
stream_handler.propagate = False

archive_logger.addHandler(stream_handler)


# =============================================================================
# Inputs
# =============================================================================
survey_name = "gabbs_valley"

### path to station data
station_dir = Path(r"c:\Users\jpeacock\Documents\test_data")

### compression for the h5 file
mth5_compression = None
mth5_compression_level = None
mth5_chunks = None
mth5_shuffle = None
mth5_fletcher = None

### path to survey parameter spread sheet
### this can be made by running this code and setting csv_fn to None
# csv_fn =
csv_fn = None  # r"/mnt/hgfs/MTData/Geysers/Archive/survey_summary.csv"

### path to mth5 configuration file
### this is a configuration file that has metadata explaining most of the
### common information needed by the user.  See example files
cfg_fn = Path(r"C:\Users\jpeacock\Documents\GitHub\mt_sb_archiving\examples\example_mth5_config_new.cfg")
cfg_obj = ConfigParser()
cfg_obj.read_file(cfg_fn.open())
cfg_dict = cfg_obj._sections

### path to xml configuration file
### this is a file that has metadata common to the xml files that go into
### science base, see examples files
xml_cfg_fn = None
# xml_cfg_fn = Path(r"/mnt/hgfs/MT_Data/GV2020/gv_sb_config.cfg")

### path to main xml file template.  This could be made somewhere else and
### has been through review.
xml_root_template = Path(
    r"c:\Users\jpeacock\Documents\test_data\mt_root_template.xml"
)

### path to xml template for child item
### this is a file that has been created according to the metadata standards
### and only a few fields will be update with station specific information
xml_child_template = Path(
    r"c:\Users\jpeacock\Documents\test_data\mt_child_template.xml"
)

### path to calibration files
### path to the calibrations.  These are assumed to be
### in the format (frequency, real, imaginary) for each coil with the
### coil name in the file name, e.g. ant2284.[cal, txt, csv]
### see example script
calibration_dir = Path(r"c:\Users\jpeacock\OneDrive - DOI\mt\ant_responses")

### paths to edi and png files if not already copied over
### typically all edi files are stored in one folder, but to make it easier
### to archive the code copies the edi and png files into the archve/station
### folder that makes it easier to upload to science base
edi_path = Path(r"/mnt/hgfs/MTData/GV2020/final_edi")
png_path = Path(r"/mnt/hgfs/MTData/GV202/final_png")

### Make xml file each
make_xml = True

### if the chile xmls are already made, put them all in the same folder and add the
### path here.
xml_path = None

### SCIENCE BASE
### page id number
### this is the end of the link that science base sends you
page_id = "############"
username = "user@usgs.gov"
password = None

### summarize all runs [ True | False ]
### this will make a .csv file that summarizes the survey by summarizing
### all runs for a station for  all stations,
### you can use this file as the csv_fn above.
### also this will create a shape file of station locations
summarize = True
survey_zc = z3d_collection.Z3DCollection(station_dir)
survey_df = survey_zc.summarize_survey(calibration_path=calibration_dir)

### upload data [ True | False]
upload_data = False
### type of files to upload in case you want to upload different formats
upload_files = [".zip", ".edi", ".png", ".xml", ".mth5"]
### if upload_data is True need to get the password for the user
### NOTE: you should run this in a generic python shell, if you use an
### ipython shell or Spyder or other IDL the password will be visible.
if upload_data:
    password = getpass.getpass()


# =============================================================================
# Make an archive folder to put everything
# =============================================================================
save_dir = station_dir.joinpath("Archive")
if not save_dir.exists():
    save_dir.mkdir()
    
# =============================================================================
# Logging    
# =============================================================================
logging_fn = save_dir.joinpath("sb_archiving.log")
# if logging_fn.exists():
#     logging_fn.unlink()
file_handler = logging.FileHandler(filename=logging_fn, 
                                   mode="w")
file_handler.setFormatter(LOG_FORMAT)
file_handler.setLevel(logging.DEBUG)
file_handler.propagate = False

archive_logger.addHandler(file_handler)
# =============================================================================
# get station folders
# =============================================================================
station_list = [
    station
    for station in station_dir.iterdir()
    if station_dir.joinpath(station).is_dir()
]

# =============================================================================
# Loop over stations
# =============================================================================
st = datetime.datetime.now()
for station_dir in station_list:
    z3d_dir = station_dir
    if z3d_dir.is_dir():
        station = station_dir.stem
        ### get the file names for each block of z3d files if none skip
        zc = z3d_collection.Z3DCollection(z3d_dir)
        try:
            fn_df = zc.get_z3d_df(calibration_path=calibration_dir)
        except ValueError as error:
            archive_logger.warning("Skipping folder %s because no Z3D files",
                                   station)
            archive_logger.error(str(error))
            continue

        ### make a folder in the archive folder
        save_station_dir = save_dir.joinpath(station)
        if not save_station_dir.exists():
            save_station_dir.mkdir()
        archive_logger.info("--- Archiving Station %s ---", station)

        ### capture output to put into a log file
        station_st = datetime.datetime.now()
        archive_logger.info("starting at %s", station_st)
        
        ### copy edi and png into archive director
        edi_fn = save_station_dir.joinpath(f"{station}.edi")
        png_fn = save_station_dir.joinpath(f"{station}.png")
        if not edi_fn.is_file():
            try:
                shutil.copy(edi_path.joinpath(f"{station}.edi"), edi_fn)
            except Exception as error:
                archive_logger.error(error)
        if not png_fn.is_file():
            try:
                shutil.copy(edi_path.joinpath(f"{station}.png"), png_fn)
            except Exception as error:
                archive_logger.error(error)                

        ### Make MTH5 File
        m = MTH5(shuffle=mth5_shuffle,
                 fletcher32=mth5_fletcher,
                 compression=mth5_compression,
                 compression_opts=mth5_compression_level)
        mth5_fn = save_station_dir.joinpath(f"{station}.h5")
        m.open_mth5(mth5_fn, "w")
        if not m.h5_is_write:
            msg = f"Something went wrong with opening {mth5_fn}, check logs"
            archive_logger.error(msg)
            raise ValueError(msg)

        ### loop over schedule blocks
        for run_num in fn_df.run.unique():
        # for run_num in [1]:
            run_df = fn_df.loc[fn_df.run == run_num]
            runts_obj, filters_list = zc.make_runts(run_df, 
                                                    file_handler,
                                                    cfg_dict)
            run_df.loc[:, ("end")] = pd.Timestamp(runts_obj.run_metadata.time_period.end)
            # add station
            station_group = m.add_station(
                runts_obj.station_metadata.id,
                station_metadata=runts_obj.station_metadata,
            )
    
            # add run
            run_group = station_group.add_run(runts_obj.run_metadata.id,
                                              runts_obj.run_metadata)
            channels = run_group.from_runts(runts_obj, chunks=mth5_chunks)
            run_group.validate_run_metadata()
    
            # need to update metadata
            station_group.validate_station_metadata()
            
            for f in filters_list:
                m.filters_group.add_filter(f)
        
        if summarize:
            survey_df.loc[survey_df.station==runts_obj.station_metadata.id, "end"] = \
                pd.Timestamp(station_group.metadata.time_period.end)
                
        # update survey metadata from data and cfg file
        m.survey_group.update_survey_metadata(survey_dict=cfg_dict["survey"])

        m.close_mth5()
        ####------------------------------------------------------------------
        #### Make xml file for science base
        ####------------------------------------------------------------------
        # make xml file
        if make_xml:
            s_xml = mt_xml.MTSBXML()
            if xml_child_template:
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

# =============================================================================
# Combine all information into a database
# =============================================================================
if summarize:

    ### write shape file
    shp_df, shp_fn = survey_zc.write_shp_file(survey_df)

    ### write survey xml
    # adjust survey information to align with data
    survey_xml = mt_xml.MTSBXML()
    if xml_root_template:
        survey_xml.read_template_xml(xml_root_template)
    if xml_cfg_fn:
        survey_xml.update_from_config(xml_cfg_fn)

    # location
    survey_xml.update_bounding_box(
        survey_df.longitude.min(),
        survey_df.longitude.max(),
        survey_df.latitude.max(),
        survey_df.latitude.min())

    # dates
    survey_xml.update_time_period(survey_df.start.min().isoformat(), 
                                  survey_df.end.max().isoformat())
    
    # shape file attributes limits
    survey_xml.update_shp_attributes(shp_df)

    ### --> write survey xml file
    survey_xml.save(save_dir.joinpath(f"{survey_name}.xml"))

# print timing
et = datetime.datetime.now()
t_diff = et - st
archive_logger.info(
    "--> Archiving took: {0}:{1:05.2f}, finished at {2}".format(
        int(t_diff.total_seconds() // 60),
        t_diff.total_seconds() % 60,
        datetime.datetime.ctime(datetime.datetime.now()),
    )
)
print("\a")
