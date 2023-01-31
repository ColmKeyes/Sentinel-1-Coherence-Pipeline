# -*- coding: utf-8 -*-
"""
@Time    : 18/01/2023 16:02
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Backscatter_Preprocessing
"""

###########
## Example code:
## https://github.com/wajuqi/Sentinel-1-preprocessing-using-Snappy
###########

## Snappy requires python version 3.6 or below.

import datetime
import time
from snappy import ProductIO
from snappy import HashMap
from snappy import WKTReader
## Garbage collection to release memory of objects no longer in use.
import os, gc
from snappy import GPF
import shapefile
#import pygeoif
#import jpy
import zipfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import Sentinel_1_SLC_Preprocessing as slc

pols = 'VV' #'VH,VV'
iw_swath = 'IW2'
first_burst_index = 4
mode = 'backscatter'
last_burst_index = 7
product_type = 'GeoTIFF'
outpath = 'D:\Data\Results\Coherence_Results'
if mode == 'backscatter':
     outpath_window = '_multilook_window_'
elif mode == 'coherence':
     outpath_window = '_coherence_window_'
mode_path =
multilook_window_size = [[2,10],[3,15]]#,[4,20]] ## will not take 1 as a param! [1,5]
for ix, i in enumerate(multilook_window_size):
    slc.main(pols,
         iw_swath,
         first_burst_index,
         last_burst_index,
         multilook_window_size[ix],
         mode=mode,
         speckle_filter='Lee',
         speckle_filter_size=[5,5],
         product_type=product_type,
         outpath = outpath + '\\pol_'+str(pols) + str(outpath_window) + str(multilook_window_size[ix][0]*multilook_window_size[ix][1]))

