# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:53:51 2018

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from archive import archive
import getpass

# =============================================================================
password = None
# if you want to archive uncomment the line below, be sure you are in a
# python shell otherwise you password will be visable.
# password = getpass.getpass()

sbmt = archive.SBMTArcive()

# survey directory
sbmt.survey_dir = Path(r"c:\Users\jpeacock\Documents\test_data")

# MTH5 compression
sbmt.mth5_compression = "gzip" # [ gzip | lfx | szip ]
sbmt.mth5_compression_level = 5 # [0-9]
sbmt.mth5_chunks = True # [True | chunck size number ]
sbmt.mth5_shuffle = True # [ True | False]
sbmt.mth5_fletcher = True # [ True | False ]

sbmt.survey_csv_fn = Path(r"c:\Users\jpeacock\Documents\test_data\Archive\survey_summary.csv")

# path to mth5 configuration file
# this is a configuration file that has metadata explaining most of the
# common information needed by the user.  See example files
sbmt.mth5_cfg_fn = Path(
    r"c:\Users\jpeacock\Documents\test_data\gv_mth5_config.cfg"
)
if sbmt.mth5_cfg_fn is not None:
    sbmt.mth5_cfg_dict = sbmt.read_cfg_file(sbmt.mth5_cfg_fn)

# path to xml configuration file
# this is a file that has metadata common to the xml files that go into
# science base, see examples files
sbmt.xml_cfg_fn = r"c:\Users\jpeacock\Documents\test_data\gv_xml_configuration.cfg"
# xml_cfg_fn = Path(r"/mnt/hgfs/MT_Data/GV2020/gv_sb_config.cfg")

# path to main xml file template.  This could be made somewhere else and
# has been through review.
sbmt.xml_root_template = Path(
    r"c:\Users\jpeacock\Documents\test_data\mt_root_template.xml"
)

# path to xml template for child item
# this is a file that has been created according to the metadata standards
# and only a few fields will be update with station specific information
sbmt.xml_child_template = Path(
    r"c:\Users\jpeacock\Documents\test_data\mt_child_template.xml"
)

# path to calibration files
# path to the calibrations.  These are assumed to be
# in the format (frequency, real, imaginary) for each coil with the
# coil name in the file name, e.g. ant2284.[cal, txt, csv]
# see example script
sbmt.calibration_dir = Path(r"c:\Users\jpeacock\OneDrive - DOI\mt\ant_responses")

# paths to edi and png files if not already copied over
# typically all edi files are stored in one folder, but to make it easier
# to archive the code copies the edi and png files into the archve/station
# folder that makes it easier to upload to science base
sbmt.edi_path = Path(r"/mnt/hgfs/MTData/GV2020/final_edi")
sbmt.png_path = Path(r"/mnt/hgfs/MTData/GV202/final_png")

# =============================================================================
# Run the program
# =============================================================================
# get station directories
station_dirs = sbmt.get_station_directories()

# loop over stations and create mth5 files, copy edi, png's and summarize
sbmt.archive_stations(
    station_dirs,
    make_xml=True,
    copy_files=True,
    summarize=True,
    upload=False,
    page_id=None,
    username=None,
    password=None,
)
