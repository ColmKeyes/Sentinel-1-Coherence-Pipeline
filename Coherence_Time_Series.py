# -*- coding: utf-8 -*-
"""
This script provides functions for building data-cubes from SLC processed coherence and backscatter data
"""
"""
@Time    : 17/02/2023 15:32
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Coherence-Time-Series
"""


import pandas as pd
## need to import gdal for rasterio import errors
## for some reaason this is the rule for in line, but in console needf to import rasterio first...
from osgeo import gdal
import rasterio
import rasterio as rasta
import rasterio.plot
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import pause
from CCD_animation import ccd_animation
import rioxarray
import rioxarray as riox
import xarray
import xarray as xar
import geopandas as gpd
from geocube.api.core import make_geocube

# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

from astropy.convolution import Box1DKernel, convolve
from scipy.signal import savgol_filter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



##########################################
## Writing backscatter and coherence rasterio raster stacks...
##########################################
def pct_clip(array,pct=[2,98]):#.02,99.98]):
    array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
    clip = (array - array_min) / (array_max - array_min)
    clip[clip>1]=1
    clip[clip<0]=0
    return clip




def write_rasterio_stack(path, write_file, titles=None,write=True ):
    ## Path = path to folder containing tiffs...

    ## get sample metadata..
    with rasterio.open(path + '\\' +str(os.listdir(path)[0])) as src0:
        meta = src0.meta
    meta.update(count=len(os.listdir(path)))
    titles = []
    with rasterio.open(write_file, 'w', **meta) as dst:
        for ix, layer in enumerate(os.listdir(path), start=1):
            with rasterio.open(path + '\\' +str(layer)) as src1:
                titles.append(str(layer)[17:25]) ## get list of image dates..
                if write == True:
                    dst.write_band(ix, pct_clip(src1.read(1)))
        print(f'Total Images stacked: {ix}')
        dst.close()
        return titles

def build_cube(tiff_stacks, shp=None ):
    cubes= []

    ## two slightly different ways to build an xarray datacube

    if shp is None:
        stack_coherence = rioxarray.open_rasterio(tiff_stacks[0], masked=False)
        stack_backscatter = rioxarray.open_rasterio(tiff_stacks[1], masked=False)

        ## open_rasterio opens as xarray DataArrays, we must convert to Datasets before combining both into a data-cube..
        stack_coherence = stack_coherence.to_dataset(name='coherence')
        stack_backscatter = stack_backscatter.to_dataset(name='backscatter')

        cube = xarray.merge([stack_coherence, stack_backscatter])

    if shp is not None:

        #for a,tiff_stack in enumerate(tiff_stacks):
            # shp_stack = rioxarray.open_rasterio(tiff_stack, masked=True).rio.clip(shp.geometry.values, shp.crs, from_disk=True)
            # if a==0:
            #     cube = make_geocube(shp, like=shp_stack, measurements=['code'])
            #
            # shp_stack['code'] = shp_stack.band + 1
            # if 'coherence' in tiff_stack:
            #     cube = cube.isel(x=range(0, 142),drop=True)  # x=142), drop=True)#y=133 ## Data is not lined up (ColLocated properly)
            # #if a != 0:
            # cube[f'{tiff_stack}'] = (shp_stack.dims,shp_stack.values,shp_stack.attrs,shp_stack.encoding)

            #shp_stack = rioxarray.open_rasterio(tiff_stack, masked=True).rio.clip(shp.geometry.values, shp.crs, from_disk=True)




        #cube['coherence_VV'] = (shp_stack.dims,shp_stack.values,shp_stack.attrs,shp_stack.encoding)




        shp_stack_backscatter = rioxarray.open_rasterio(tiff_stacks[0], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")


        shp['code'] = shp.index + 1

        shp_stack = rioxarray.open_rasterio(tiff_stacks[1], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")

        shp_stack_backscatter_VH = rioxarray.open_rasterio(tiff_stacks[2], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")

        shp_stack_coh_VH = rioxarray.open_rasterio(tiff_stacks[3], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")


        shp['code'] = shp.index + 1
        shp_stack_backscatter_VH['code']= shp_stack_backscatter_VH.band +1
        shp_stack_coh_VH['code']= shp_stack_coh_VH.band +1



        cube = make_geocube(shp,like=shp_stack ,measurements=['code'])
        cube['coherence_VV'] = (shp_stack.dims,shp_stack.values,shp_stack.attrs,shp_stack.encoding)
        cube["coherence_VH"] = (shp_stack_coh_VH.dims, shp_stack_coh_VH.values,shp_stack_coh_VH.attrs,shp_stack_coh_VH.encoding)
        ## squeezing last weird dim length...
        cube = cube.isel(x=range(0, len(cube.x)-1),drop=True)#x=142), drop=True)#y=133 ## reduce length by 1 in x axis....
        cube["backscatter_VV"] = (shp_stack_backscatter.dims, shp_stack_backscatter.values,shp_stack_backscatter.attrs,shp_stack_backscatter.encoding)
        cube["backscatter_VH"] = (shp_stack_backscatter_VH.dims, shp_stack_backscatter_VH.values,shp_stack_backscatter_VH.attrs,shp_stack_backscatter_VH.encoding)

    return cube

def calc_zonal_stats(cube):
    #################
    ## coh_stats
    #################
    grouped_coherence_cube = cube.groupby(cube.code)  ## so this is treating it as a geopandas geodataframe...
    grid_mean = grouped_coherence_cube.mean().rename({"coherence": "coherence_mean"})
    grid_min = grouped_coherence_cube.min().rename({"coherence": "coherence_min"})
    grid_max = grouped_coherence_cube.max().rename({"coherence": "coherence_max"})
    grid_std = grouped_coherence_cube.std().rename({"coherence": "coherence_std"})
    zonal_stats = xarray.merge([grid_mean, grid_min, grid_max, grid_std]).to_dataframe()
    #shp_with_statistics = shp.merge(zonal_stats,on='code')

    return zonal_stats

#def stat_analysis(cube):   Next up...

