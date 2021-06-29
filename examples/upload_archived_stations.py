#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:11:31 2021

@author: peacock
"""

from pathlib import Path
from archive import archive
import getpass
from mt_metadata.utils import mttime

# archive_dir = r"/mnt/hgfs/MT_Data/GV2020/Archive"

archive_dir = r"c:\MT\GV2020\Archive"
page_id = "60d39cc1d34e12a1b009c64b"
username = "jpeacock@usgs.gov"


sbmt = archive.SBMTArcive()

password = getpass.getpass()

archive_dirs = [p for p in Path(archive_dir).iterdir() if p.is_dir()]

for archive_station_dir in archive_dirs[28:]:
    st = mttime.MTime(mttime.get_now_utc())
    print(f"Archiving: {archive_station_dir}")
    try:
        sbmt.upload_data(page_id, 
                         archive_station_dir,
                         username,
                         password,
                         [".edi", ".png", ".xml", ".h5"])
        et = mttime.MTime(mttime.get_now_utc())
        print("\a")
        print(f"Uploaded {archive_station_dir.name}, took: {et - st:.2f} seconds")
    except archive.ArchiveError as error:
        print("Could not archive %s" % error)
        sbmt.logger.warning("Could not archive %s", error)

# sbmt.upload_stations(page_id,
#                      archive_dir,
#                      r"jpeacock@usgs.gov", 
#                      password)