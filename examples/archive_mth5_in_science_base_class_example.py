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
from archive import mt_xml, archive

import getpass

sbmt = archive.SBMTArcive()

# survey directory
sbmt.survey_dir = Path(r"c:\Users\jpeacock\Documents\test_data")

# MTH5 compression 
sbmt.mth5_compression = None
sbmt.mth5_compression_level = None
sbmt.mth5_chunks = None
sbmt.mth5_shuffle = None
sbmt.mth5_fletcher = None

sbmt.csv_fn = Path(r"c:\Users\jpeacock\Documents\test_data\Archive\survey_summary.csv")

### path to mth5 configuration file
### this is a configuration file that has metadata explaining most of the
### common information needed by the user.  See example files
sbmt.cfg_fn = Path(r"C:\Users\jpeacock\Documents\GitHub\mt_sb_archiving\examples\example_mth5_config_new.cfg")
sbmt.cfg_dict = sbmt.read_cfg_file(sbmt.cfg_fn)

### path to xml configuration file
### this is a file that has metadata common to the xml files that go into
### science base, see examples files
sbmt.xml_cfg_fn = None
# xml_cfg_fn = Path(r"/mnt/hgfs/MT_Data/GV2020/gv_sb_config.cfg")

### path to main xml file template.  This could be made somewhere else and
### has been through review.
sbmt.xml_root_template = Path(
    r"c:\Users\jpeacock\Documents\test_data\mt_root_template.xml"
)

### path to xml template for child item
### this is a file that has been created according to the metadata standards
### and only a few fields will be update with station specific information
sbmt.xml_child_template = Path(
    r"c:\Users\jpeacock\Documents\test_data\mt_child_template.xml"
)

### path to calibration files
### path to the calibrations.  These are assumed to be
### in the format (frequency, real, imaginary) for each coil with the
### coil name in the file name, e.g. ant2284.[cal, txt, csv]
### see example script
sbmt.calibration_dir = Path(r"c:\Users\jpeacock\OneDrive - DOI\mt\ant_responses")

### paths to edi and png files if not already copied over
### typically all edi files are stored in one folder, but to make it easier
### to archive the code copies the edi and png files into the archve/station
### folder that makes it easier to upload to science base
sbmt.edi_path = Path(r"/mnt/hgfs/MTData/GV2020/final_edi")
sbmt.png_path = Path(r"/mnt/hgfs/MTData/GV202/final_png")

station_dirs = sbmt.get_station_directories()

sbmt.archive_stations(station_dirs, summarize=False)