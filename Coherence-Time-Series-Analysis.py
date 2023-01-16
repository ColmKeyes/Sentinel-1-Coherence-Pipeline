# -*- coding: utf-8 -*-
"""
@Time    : 04/01/2023 21:56
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Coherence-Time-Series-Analysis
"""
#conda update -n base -c defaults conda



import pandas as pd

## need to import gdal for rasterio import errors
from osgeo import gdal
import rasterio
import rasterio as rasta

import numpy as np
import os
import h5py
import cv2
import mintpy as mint
from calc_coherence_change import calc_coherence_change
from plot_ccd import plot_ccd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import pause
from CCD_animation import ccd_animation
path = 'D:\Data\Results\Coherence_Results'

with rasterio.open(path + '\\' +str(os.listdir(path)[0])) as src0:
    meta = src0.meta
meta.update(count=len(os.listdir(path)))

titles = []
with rasterio.open('stack.tif', 'w', **meta) as dst:
    for ix, layer in enumerate(os.listdir(path), start=1):
#        if ix < 3:
        with rasterio.open(path + '\\' +str(layer)) as src1:
            titles.append([str(layer)[17:-4]])
            dst.write_band(ix, src1.read(1))
            #dst.write( src1.read(1),ix)
    print(f'Total Images stacked: {ix}')
    dst.close()


ccd, pre_coherence, co_coherence = calc_coherence_change(rasterio.open('stack.tif').read(), meta=meta, method='difference')    #       #
# plot_ccd(ccd,'20200826',method='difference')


from CCD_animation import ccd_animation

ccd_animation(rasterio.open('stack.tif'),titles)





#############
## Tree Cover 2000, Hansen, GLAD
#############
rasterio.open(path+'Hansen_GFC-2021-v1.9_treecover2000_00N_110E.tif')

#############
## Tree Cover loss 2021, Hansen
#############
rasterio.open(path+'Hansen_GFC-2021-v1.9_lossyear_00N_110E.tif')
