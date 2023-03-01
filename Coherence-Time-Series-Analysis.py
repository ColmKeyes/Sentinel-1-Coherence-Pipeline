# -*- coding: utf-8 -*-
"""
This script provides analysis of data-cubes from SLC processed coherence and backscatter data
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
from Coherence_Time_Series import CoherenceTimeSeries as cts

# TODO: plot change in backscatter between coherence fist and second images...
## show a stop-gap between some of the large gaps.
## Add Fill value of NAN for boxcar,
## combine smoothed trendline and real values as scatter.
## add nan values for large gap??
## Keep consistent X & Y axis limits,
## for RQ1, visualise how different coherence window sizes show the problem of overstimation at low number of looks :D
## for RQ2, we then want to deep dive into the disturbance events themselves, and their temporal coherence and backscatter characteristics

################################
##CODES:
## 5=Urban,6=Farmland,7=3rd_Compact, 3=2nd_Compact, 4=1stCompact, 10=Intact_Forest
################################

if __name__ == '__main__':


    window_size = 500
    asf_df,  coh_path_list, full_stack_path_list = cts.paths(window_size)

    shp = gpd.read_file('D:\Data\\geometries\\ordered_gcp_6_items_Point.shp')   #'D:\Data\\geometries\\combiend_polygons.shp')
    shp['code'] = shp.index + 1

    tiff_stack=[]

     ## I NEED TO FIX THESE!!!
    #if not os.path.exists(f"{results_path}\\{path[45:]}.tif"):

    stack_paths = []
    for i in range(len(coh_path_list)):
        stack_paths.append((coh_path_list[i], full_stack_path_list))#[i])))

    for input_path, output_path in stack_paths:
        #if not os.path.exists(output_path):
        cts.write_rasterio_stack(input_path, output_path,shp)

    ## get the titles of the images
    titles=cts.write_rasterio_stack(coh_path_list[0], full_stack_path_list,gcps=shp, write=False) ## struggling to get this to do what it's supposed to...

    cube = cts.build_cube(tiff_stacks=full_stack_path_list, shp =shp)

    coh_dates = pd.to_datetime(pd.Series(titles))
    cube['dates'] = coh_dates

    ### ccd animation
    ccd_animation(rasterio.open(f'{output_path}\\{os.listdir(output_path)[1]}'))

    #ccd_animation(rasterio.open("D:\Data\Results\S1A_IW_SLC__1SDV_20200611_20200623_pol_VH_coherence_window_500_Stack.tif"))

    #############################
    ## Radd alerts dont' work with single GCPs...
    #############################

    perp_dist_diff = np.abs(asf_df[" Reference Perpendicular Baseline (meters)"] - asf_df[" Secondary Perpendicular Baseline (meters)"])
    perp_dist_diff.name = 'Perpendicular_Distance'

    coh_VV_mean_df,coh_VH_mean_df,bsc_VV_mean_df,bsc_VH_mean_df = cts.calc_zonal_stats(cube)

    cts.single_plot(cube,coh_VV_mean_df,coh_VH_mean_df,bsc_VV_mean_df,bsc_VH_mean_df,window_size)

    #precipitation_plot()

    titles =  ['1st Disturbed Area', '2nd Disturbed Area', 'Sand & Water', 'Farmland','3rd Disturbed Area', 'Intact Forest'] #'Intact Forest','Farmland','Urban', '1st_Compact Event', '2nd_Compact Event' ,'3rd_Compact Event']

    #multiple_plots(cube,coh_VV_mean_df,coh_VH_mean_df,bsc_VV_mean_df,bsc_VH_mean_df,titles,window_size)



























