# -*- coding: utf-8 -*-
"""
This script provides functions for building data-cubes from SLC processed coherence and backscatter data
"""
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
import pandas as pd
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
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



##########################################
## Writing backscatter and coherence rasterio raster stacks...
##########################################

def write_rasterio_stack(path, write_file, titles=None ):
    ## Path = path to folder containing tiffs...

    ## get sample metadata..
    with rasterio.open(path + '\\' +str(os.listdir(path)[0])) as src0:
        meta = src0.meta
    meta.update(count=len(os.listdir(path)))

    with rasterio.open(write_file, 'w', **meta) as dst:
        for ix, layer in enumerate(os.listdir(path), start=1):
            with rasterio.open(path + '\\' +str(layer)) as src1:
                titles.append([str(layer)[17:-4]]) ## get list of image dates..
                dst.write_band(ix, src1.read(1))
        print(f'Total Images stacked: {ix}')
        dst.close()

def build_cube(tiff_stacks, shp=None ):
    ## shp = polygon shape files

    if shp is None:
        stack_coherence = rioxarray.open_rasterio(tiff_stacks[0], masked=False)
        stack_backscatter = rioxarray.open_rasterio(tiff_stacks[1], masked=False)

        ## open_rasterio opens as xarray DataArrays, we must convert to Datasets before combining both into a data-cube..
        stack_coherence = stack_coherence.to_dataset(name='coherence')
        stack_backscatter = stack_backscatter.to_dataset(name='backscatter')

        cube = xarray.merge([stack_coherence, stack_backscatter])

    if shp is not None:
        shp_stack = rioxarray.open_rasterio(tiff_stacks[0], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")

        shp_stack_backscatter = rioxarray.open_rasterio(tiff_stacks[1], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")

        shp_stack_backscatter['code']= shp_stack_backscatter.band +1

        cube = make_geocube(shp,like=shp_stack ,measurements=['code'])
        cube['coherence'] = (shp_stack.dims,shp_stack.values,shp_stack.attrs,shp_stack.encoding)

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


if __name__ == '__main__':
    path = 'D:\Data\Results\Coherence_Results\pol_VV_coh_window_45'
    titles = []
    tiff_stack=[]
    for ix, layer in enumerate(os.listdir(path[:34])):  ## look at upper layer in path..
        write_rasterio_stack(layer, f'{layer}.tif')
        tiff_stack.append(layer)

    cube = build_cube(tiff_stack=tiff_stack)
    zonal_stats = calc_zonal_stats(cube)

    print(cube)
    print(zonal_stats)

