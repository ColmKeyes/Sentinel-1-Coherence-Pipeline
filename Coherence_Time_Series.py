# -*- coding: utf-8 -*-
"""
This script provides a class and methods for building data-cubes from SLC processed coherence and backscatter data
"""
"""
@Time    : 17/02/2023 15:32
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Coherence-Time-Series
"""


import pandas as pd
## need to import gdal for rasterio import errors
## for some reaason this is the rule for in line, but in console need to import rasterio first...
from osgeo import gdal
import rasterio
import rasterio as rasta
import rasterio.plot
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import pause
from CCD_animation import ccd_animation
import rioxarray
import rioxarray as riox
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import get_data_window, transform, shape

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



class CoherenceTimeSeries:

    def __init__(self, asf_df, path, stack_path_list, window_size,shp=None, normalised=False):
        self.asf_df = asf_df
        self.path = path
        self.stack_path_list = stack_path_list
        self.cube = None
        self.window_size = window_size
        self.normalised = normalised
        self.shp = shp
        self.titles = [f[17:25] for f in os.listdir(self.path[0]) if f.endswith('.tif')]

    #def paths(self,window_size, normalised=False):


        #return  asf_df,coh_path_list, full_self.stack_path_list




    ##########################################
    ## Writing backscatter and coherence rasterio raster stacks...
    ##########################################
    def pct_clip(self,array,pct=[2,98]):#.02,99.98]):
        array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
        clip = (array - array_min) / (array_max - array_min)
        clip[clip>1]=1
        clip[clip<0]=0
        return clip


    def write_rasterio_stack(self,write=True):
        """
        Write folder of GeoTIFFs to a GeoTIFF stack file.

        Args:
            path (str): Path to the folder containing the input GeoTIFF files.
            self.stack_path_list (str): Path to the output GeoTIFF stack file.
            write (bool): If True, write the output file; if False, return the
                metadata of the output file without writing it. Default is True.

        Returns:
            list: List of the titles of the input GeoTIFF files.

        ## https://gist.github.com/prakharcode/b83caaaa2fc6d2d62b7fe558656df0d1#file-resample-py-L14

        """

        for path in self.path:
            files = [f for f in os.listdir(path) if f.endswith('.tif')]
    
    
            # Read the source image and the GCPs
            src_image = rasterio.open(os.path.join(path, files[0]),"r+")
    
            dst_crs = src_image.crs
            dst_transform, dst_width, dst_height = calculate_default_transform(src_image.crs, dst_crs, src_image.width, src_image.height, *src_image.bounds)
    
    
            # Create a VRT dataset with the transformation applied
            dst_profile = src_image.profile.copy()
            dst_profile.update({
                "driver": "GTiff",
                "count": len(files),
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height
            })
    
    
            if not os.path.exists(f"{self.stack_path_list}"):
                os.makedirs(f"{self.stack_path_list}")
            with rasterio.open(f'{self.stack_path_list}\\{path[45:]}.tif', 'w', **dst_profile) as dst:
                for i, file in enumerate(files, start=1):
    
                    src=rasterio.open(os.path.join(path, file),"r+")
                    src.nodata = 0
                    src.close()
    
                    with rasterio.open(os.path.join(path, file),"r") as src:
                        profile = src.profile.copy()
                        data_window = get_data_window(src.read(masked=True))
                        data_transform = transform(data_window, src.transform)
                        profile.update(
                            transform=data_transform,
                            height=data_window.height,
                            width=data_window.width)
    
                        data = src.read(window=data_window)
                        src.close()
    
                    if write:
                        if self.normalised:
                            dst.write(self.pct_clip(data[0]), i)
                        else:
                            dst.write(data[0], i)
                dst.close()




    def build_cube(self):
        cubes = []

        if self.shp is not None:
            shp_stacks = [rioxarray.open_rasterio(os.path.join(self.stack_path_list,stack), masked=True).rio.clip(self.shp.geometry.values, self.shp.crs, from_disk=True) for stack in os.listdir(self.stack_path_list)]

            shp_stack_coh_VH = shp_stacks[0]
            shp_stack_coh_VV = shp_stacks[1] if len(shp_stacks) >= 2 else None
            shp_stack_backscatter_VH = shp_stacks[2] if len(shp_stacks) >= 3 else None
            shp_stack_backscatter_VV = shp_stacks[3] if len(shp_stacks) >= 4 else None

            self.cube = make_geocube(self.shp, like=shp_stack_coh_VH, measurements=["code"])
            if shp_stack_coh_VV is not None:
                shp_stack_coh_VV["code"] = shp_stack_coh_VV.band + 1
                self.cube['coherence_VV'] = (shp_stack_coh_VV.dims, shp_stack_coh_VV.values, shp_stack_coh_VV.attrs, shp_stack_coh_VV.encoding)

            if shp_stack_coh_VH is not None:
                shp_stack_coh_VH["code"] = shp_stack_coh_VH.band + 1
                self.cube["coherence_VH"] = (shp_stack_coh_VH.dims, shp_stack_coh_VH.values, shp_stack_coh_VH.attrs, shp_stack_coh_VH.encoding)


            if shp_stack_backscatter_VV is not None:
                shp_stack_backscatter_VV["code"] = shp_stack_backscatter_VV.band + 1
                self.cube["backscatter_VV"] = (shp_stack_backscatter_VV.dims, shp_stack_backscatter_VV.values, shp_stack_backscatter_VV.attrs, shp_stack_backscatter_VV.encoding)

            if shp_stack_backscatter_VH is not None:
                shp_stack_backscatter_VH["code"] = shp_stack_backscatter_VH.band + 1
                self.cube["backscatter_VH"] = (shp_stack_backscatter_VH.dims, shp_stack_backscatter_VH.values, shp_stack_backscatter_VH.attrs, shp_stack_backscatter_VH.encoding)

            coh_dates = pd.to_datetime(pd.Series(self.titles))
            self.cube['dates'] = coh_dates



    def calc_zonal_stats(self):
        #################
        ## coh_stats
        #################
        ## One really should't average coherence as it's an ensemble averaging!! One looses significant information when you average an average, so this is bad to be doing...
        zonal_stats = self.cube.groupby(self.cube.code).mean()
        return zonal_stats


    def single_plot(self,zonal_stats=None):

        # for var_name in zonal_stats:
        #     plt.plot(self.cube.dates, zonal_stats[var_name][5], label=var_name)


        grouped = self.cube.groupby('code')



        #for code, ds in grouped:
            # access the coherence_VH variable for this code

           #coherence_VH = ds['coherence_VH']

        #fig, axes = plt.subplots(ncols=len(grouped), figsize=(16, 4), sharex=True, sharey=True)

        for i, (code, ds) in enumerate(grouped):
            if i==5:
                plt.plot(ds.dates, ds.coherence_VH, label=f'Code {int(code)}')
                plt.title(f'Disturbance Event 3, {self.window_size}m Resolution')
                plt.legend()
                plt.show()
                plt.pause(100)

            #######
            ## multiplot example
            #######
            # ax = axes[i]
            # #for j in range(ds.shape[1]):
            # ax.plot(ds.dates, ds.isel(band=j), label=f'band {j + 1}')
            # ax.set_title(f'Code {int(code)}')
            # ax.legend()


        # plt.title(f'Disturbance Event 3, {self.window_size}m Resolution')
        # plt.legend()
        # plt.show()
        # plt.pause(100)




    def multiple_plots(self,coh_VV_mean_df,coh_VH_mean_df,bsc_VV_mean_df,bsc_VH_mean_df,titles):
        fig, ax = plt.subplots(3, 2, figsize=(21, 7), sharey=True, sharex=True)


        convolve_data = False  # set to True to convolve data, False to not convolve

        a = 0
        for i in range(3):
            for j in range(2):
                try:
                    for df, name in [(coh_VV_mean_df, coh_VV_mean_df.name),
                                     (coh_VH_mean_df, coh_VH_mean_df.name),
                                     (bsc_VV_mean_df, bsc_VV_mean_df.name),
                                     (bsc_VH_mean_df, bsc_VH_mean_df.name)]:
                        if convolve_data:
                            data = convolve(df[a], Box1DKernel(5), boundary=None)
                        else:
                            data = df[a]
                        ax[i, j].plot(self.cube.dates, data, label=name)

                    #ax[i, j].scatter(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_dates'][7:61],
                    #                 pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_values'][7:61], [.2, 99.8]),
                    #                 label='Radd Alert Detections')
                    ax[i, j].set_title(titles[a])
                except KeyError:
                    continue
                except IndexError:
                    continue
                a = a + 1



        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes][0]
        #lines, labels = [(label, []) for label in zip(*lines_labels1)][0] #sum(lol, [])
        lines_labels = [[line for line,label in zip(*lines_labels)],[label for line,label in zip(*lines_labels)]]
        # Finally, the legend (that maybe you'll customize differently)
        fig.legend(lines_labels[0], lines_labels[1], loc='upper right')#, ncol=4)
        fig.tight_layout()
        fig.suptitle(f"{self.window_size*2}m Pixel Spacing")
        plt.show()
        pause(1000)




    def precipitation_plot(self,perp_dist_diff,coh_mean_df):

        # Set time period
        start = datetime(2021, 1, 1)
        end = datetime(2023, 1, 31)

        # Create Point for Vancouver, BC
        # location = Point(49.2497, -123.1193, 70) #-1.8, 113.5, 0

        # Get daily data for 2018
        data = Daily(96655, start, end)
        data = data.fetch()


        prcp = data.groupby(pd.cut(data.index, self.cube.dates)).mean()['prcp'].to_frame()
        prcp['dates'] = self.cube.dates[:-1]  ## one less date as this is change between dates..
        prcp.name = 'Mean Precipitation'

        plt.scatter(coh_mean_df.index,self.pct_clip(perp_dist_diff))    # #plt.show()
        plt.plot(coh_mean_df.index, coh_mean_df, label=coh_mean_df.columns)
        plt.yticks([0, 1])
        plt.legend()
        plt.show()
        plt.pause(1000)


    def radd_alert_data(self):

        #####################
        ## Having difficulty here wrangling the radd alert dates into datetimes, so will pass it to a loop before plotting...
        ## save each code as aa band in a datacube...
        #####################

        radd = rioxarray.open_rasterio("D:/Data/Radd_Alert.tif", masked=True).rio.clip(
                self.shp.geometry.values, self.shp.crs, from_disk=True) ##.rio.reproject_match(cube)

        radd_cube = make_geocube(self.shp, like=radd, measurements=['code'])
        radd_cube["alert_date"] = (radd.dims, radd.values, radd.attrs,radd.encoding)
        radd_stats = radd_cube.groupby(radd_cube.code)

        radd_count = radd_stats.count()
        radd_count["alert_dates"] = (radd.dims, radd.values, radd.attrs,radd.encoding)

        radd_count['dates'] =  datetime.strptime(radd_cube.alert_date, '%y%j')

        # two slightly different ways to build an xarray datacube

        unique_codes = np.unique(radd_cube.code)[np.isfinite(np.unique(radd_cube.code))] ## or [~np.isnan()]
        radd_array = pd.DataFrame()
        for ix,i in enumerate(unique_codes):
            polygon = radd_cube.where(radd_cube.code == i).alert_date[1]  # .plot.imshow() plt.pause(100)
            polygon_counts = np.unique(polygon, return_counts=True)
            radd_array[f'polygon{i}_dates'] = pd.to_datetime(pd.Series(polygon_counts[0].astype(str)), format="%y%j.0", errors='coerce')
            radd_array[f'polygon{i}_values'] = pd.Series(polygon_counts[1])



        radd_array1 = radd_array.iloc[:, :2]
        radd_array2 = radd_array.iloc[:, 2:4]
        #radd_array3 = radd_array.iloc[:, 4:6]##only NaTTypes
        radd_array4 = radd_array.iloc[:, 6:8]
        radd_array5 = radd_array.iloc[:, 8:10]
        radd_array6 = radd_array.iloc[:, 10:12]

        radd_arrays_list = [radd_array1, radd_array2 , radd_array4, radd_array5,radd_array6]
        #radd_array1.groupby([pd.Grouper(key='polygon3.0_dates', freq='W-MON')]).mean()['polygon3.0_values'].to_frame()

        radd_arrays = [radd_df.groupby([pd.Grouper(key=str(radd_df.columns[0]) , freq='D')]).mean()[str(radd_df.columns[1])].to_frame() for radd_df in radd_arrays_list]#W-MON

        radd_xarray = pd.concat(radd_arrays).to_xarray()

        #ax[i, j].scatter(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_dates'][7:61] ,pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_values'][7:61],[.2,99.8]),label='Radd Alert Detections')#raddy_array.index,radd_xarray[f'polygon{np.unique(radd_cube.code)[a]}_values'],[0,100]))#f'polygon{np.unique(radd_cube.code)[a]}_dates'

