#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:11:31 2021

@author: peacock
"""


from archive import archive
import getpass

archive_dir = r"/mnt/hgfs/MT_Data/GV2020/Archive"
sbmt = archive.SBMTArcive()

password = getpass.getpass()
sbmt.upload_stations(r"",
                     archive_dir,
                     r"jpeacock@usgs.gov", 
                     password)