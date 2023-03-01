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

    def __init__(self, path, bsc_path, results_path,asf_df, window_size):
        self.path = path
        self.bsc_path = bsc_path
        self.results_path = results_path
        self.asf_df = asf_df
        self.cube = None
        self.window_size = window_size


    def paths(window_size, normalised=False):

        #######
        ## my paths are FUCKED
        ######
        if not normalised:
            stacks = 'Stacks_non_normalised'
        if normalised:
            stacks = 'Stacks_normalised'

        stack_path = f'D:\\Data\\Results\\Stacks\\{stacks}\\{window_size}m_window'
        coh_results_path = f'D:\\Data\\Results\\Coherence_Results\\{window_size}m_window'
        coh_path_list = [os.path.join(coh_results_path, directory) for directory in os.listdir(coh_results_path)]
        full_stack_path_list = stack_path   #[os.path.join(stack_path, directory) for directory in os.listdir(stack_path)]
        path_asf_csv = r'D:\Data\asf-sbas-pairs_12d_all_perp.csv'#asf-sbas-pairs_24d_35m_Jun20_Dec22.csv'
        asf_df = pd.read_csv(path_asf_csv)
        asf_df = asf_df.drop(index=61)
        return  asf_df,coh_path_list, full_stack_path_list




    ##########################################
    ## Writing backscatter and coherence rasterio raster stacks...
    ##########################################
    def pct_clip(array,pct=[2,98]):#.02,99.98]):
        array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
        clip = (array - array_min) / (array_max - array_min)
        clip[clip>1]=1
        clip[clip<0]=0
        return clip


    def write_rasterio_stack(path, write_file,gcps, titles=None,write=True):
        """
        Write folder of GeoTIFFs to a GeoTIFF stack file.

        Args:
            path (str): Path to the folder containing the input GeoTIFF files.
            write_file (str): Path to the output GeoTIFF stack file.
            write (bool): If True, write the output file; if False, return the
                metadata of the output file without writing it. Default is True.

        Returns:
            list: List of the titles of the input GeoTIFF files.

        ## https://gist.github.com/prakharcode/b83caaaa2fc6d2d62b7fe558656df0d1#file-resample-py-L14

        """


        titles = []
        files = [f for f in os.listdir(path) if f.endswith('.tif')]

        #
        coords = [(p.x, p.y) for p in gcps.geometry]
        # gcps= GroundControlPoint(gcps)
        # transform = from_gcps(gcps)
        # crs = rasterio.crs.CRS.from_epsg(28992)
        #
        # if not files:
        #     raise ValueError(f"No GeoTIFF files found in {path}.")
        # files = sorted(files)
        # for f in files:
        #     title = f[17:25]
        #     titles.append(title)
        #
        # gcps = []
        # for idx, image_file in enumerate(files):
        #     with rasterio.open(os.path.join(path, image_file)) as src:
        #         pixel_coords = [src.index(x, y) for x, y in coords]
        #         gcps.extend([(idx + 1, p[0], p[1], x, y) for p, (x, y) in zip(coords, pixel_coords)])

        import rasterio as rio
        from rasterio.windows import get_data_window, transform, shape

        from rasterio.vrt import WarpedVRT
        import geopandas as gpd

        # Read the source image and the GCPs
        src_image = rasterio.open(os.path.join(path, files[3]),"r+")
        #gcp_df = gcps

        first_culprit = rasterio.open(os.path.join(path, files[40]),"r+")

        # Convert the GCPs to a format that rasterio can use
        #gcps = []
        # for _, row in gcp_df.iterrows():
        #     gcps.append(rasterio.control.GroundControlPoint(row["x"], row["y"], row["lon"], row["lat"], row["z"]))

        # Calculate the transformation matrix and output image size
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


        with rasterio.open(f'{write_file}\\{path[45:]}.tif', 'w', **dst_profile) as dst:
            for i, file in enumerate(files, start=1):
                with rasterio.open(os.path.join(path, file)) as src:
                    dest = np.zeros(rasterio.band(src, 1).shape)
                    if write:
                        ## I`ve modelled this on a reporjection of current data, but I need to model it on the writing of new data... see rasterio warp...
                        reproject(source=rasterio.band(src, 1),
                                    destination=dest,   #rasterio.band(dst, i),
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=dst_transform,
                                    dst_crs=src.crs,
                                    resampling=Resampling.nearest)

                        dst.write(dest, i) #src.read(1), i)# firstly, i dont need to write this also , reproject already wrties...
                    src.close()

        dest = np.zeros(rasterio.band(first_culprit, 1).shape)
        from rasterio.windows import get_data_window
        reproject(source=rasterio.band(first_culprit, 1),
                  destination=dest,  # rasterio.band(dst, i),
                  src_transform=first_culprit.transform,
                  src_crs=first_culprit.crs,
                  dst_transform=dst_transform,
                  dst_crs=first_culprit.crs,
                  resampling=Resampling.nearest)

        dest1 = np.zeros(rasterio.band(src_image, 1).shape)
        reproject(source=rasterio.band(src_image, 1),
                  destination=dest1,  # rasterio.band(dst, i),
                  src_transform=src_image.transform,
                  src_crs=src_image.crs,
                  dst_transform=dst_transform,
                  dst_crs=src_image.crs,
                  resampling=Resampling.nearest)









        #
        # with WarpedVRT(src_image, crs=dst_crs, transform=dst_transform, width=dst_width, height=dst_height, resampling=Resampling.bilinear, add_alpha=True) as vrt:
        #     # Read the collocated image from the VRT
        #     collocated_image = vrt.read()
        #     collocated_profile = vrt.profile
        #
        # # Write the collocated image to file
        # with rasterio.open("path/to/output/image.tif", "w", **collocated_profile) as dst:
        #     dst.write(collocated_image)
























        with rasterio.open(os.path.join(path, files[0])) as src0:
            meta = src0.meta
        meta.update(count=len(files))

        with rasterio.open(f'{write_file}\\{path[45:]}.tif', 'w', **meta) as dst:
            for i, file in enumerate(files, start=1):
                with rasterio.open(os.path.join(path, file)) as src:
                    if write:
                        dst.write(src.read(1), i)
                    src.close()

        print(f"Total images stacked: {i}")
        return titles

        ##########################################




    def build_cube(tiff_stacks, shp=None ):
        cubes= []

        if shp is not None:

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

            ## make cube same length as backscatter cube...:
            shp_stack_backscatter=shp_stack_backscatter.isel(y=range(0, len(cube.y)),drop=True)#x=142)) len(cube.y)-1),drop=True)
            shp_stack_backscatter_VH=shp_stack_backscatter_VH.isel(y=range(0, len(cube.y)),drop=True)#x=142)) len(cube.y)-1),drop=True)

            cube = cube.isel(x=range(0, len(cube.x)-1),drop=True)#x=142), drop=True)#y=133 ## reduce length by 1 in x axis....

            cube["backscatter_VV"] = (shp_stack_backscatter.dims, shp_stack_backscatter.values,shp_stack_backscatter.attrs,shp_stack_backscatter.encoding)
            cube["backscatter_VH"] = (shp_stack_backscatter_VH.dims, shp_stack_backscatter_VH.values,shp_stack_backscatter_VH.attrs,shp_stack_backscatter_VH.encoding)

        return cube

    def calc_zonal_stats(cube):
        #################
        ## coh_stats
        #################
        ## My zonal stats are wrong, I can't be averaging coherence as it's an ensemble averaging!! One looses significant information when you average an average, so this is bad to be doin...

        ## Also, I should really improve this to be a single variable at output, so that I dont' keep having to mess around with 4 different variables.
        zonal_stats = cube.groupby(cube.code)#calc_zonal_stats(cube)
        zonal_stats = zonal_stats.mean()#.rename({"coherence": "coherence_mean"})
        #zonal_transpose = zonal_stats.unstack(level='code')
        coh_VV_mean_df = zonal_stats.coherence_VV #zonal_transpose.coherence_mean
        bsc_VV_mean_df = zonal_stats.backscatter_VV
        bsc_VH_mean_df = zonal_stats.backscatter_VH
        coh_VH_mean_df = zonal_stats.coherence_VH

        return coh_VV_mean_df,coh_VH_mean_df,bsc_VV_mean_df,bsc_VH_mean_df

        #
        # grouped_coherence_cube = cube.groupby(cube.code)  ## so this is treating it as a geopandas geodataframe...
        # grid_mean = grouped_coherence_cube.mean().rename({"coherence": "coherence_mean"})
        # grid_min = grouped_coherence_cube.min().rename({"coherence": "coherence_min"})
        # grid_max = grouped_coherence_cube.max().rename({"coherence": "coherence_max"})
        # grid_std = grouped_coherence_cube.std().rename({"coherence": "coherence_std"})
        # zonal_stats = xarray.merge([grid_mean, grid_min, grid_max, grid_std]).to_dataframe()
        # #shp_with_statistics = shp.merge(zonal_stats,on='code')

        return zonal_stats

    #def stat_analysis(cube):   Next up...


    def single_plot(cube,coh_VV_mean_df,coh_VH_mean_df,bsc_VV_mean_df,bsc_VH_mean_df,window_size):

        plt.plot(cube.dates, coh_VV_mean_df[4] )#, label=coh_VV_mean_df.name)  # ,, convolve(coh_VV_mean_df[4], Box1DKernel(5), boundary=None)
        plt.plot(cube.dates, coh_VH_mean_df[4] )#, label=coh_VH_mean_df.name)  # , coh_VH_mean_df[a],label=coh_VH_mean_df.name)  convolve(coh_VH_mean_df[4], Box1DKernel(5), boundary=None)
        plt.plot(cube.dates,  bsc_VV_mean_df[4])#, label=bsc_VV_mean_df.name)  # bsc_VV_mean_df[a],label=bsc_VV_mean_df.name)   convolve(bsc_VV_mean_df[4], Box1DKernel(5), boundary=None)
        plt.plot(cube.dates, bsc_VH_mean_df[4])#, label=bsc_VH_mean_df.name)  # , bsc_VH_mean_df[a],label=bsc_VH_mean_df.name) convolve(bsc_VH_mean_df[4], Box1DKernel(5), boundary=None)
        #plt.scatter(radd_array[f'polygon{np.unique(radd_cube.code)[4]}_dates'][7:61], pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[4]}_values'][7:61], [.2, 99.8]), label='Radd Alert Detections')
        #plt.scatter(prcp.dates, x_filtered, label='Precipitation')
        plt.title(f'Disturbance Event 3, {window_size*2}m Resolution')
        plt.legend()
        plt.show()
        plt.pause(100)



    def multiple_plots(cube,coh_VV_mean_df,coh_VH_mean_df,bsc_VV_mean_df,bsc_VH_mean_df,titles,window_size):
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
                        ax[i, j].plot(cube.dates, data, label=name)

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
        fig.suptitle(f"{window_size*2}m Pixel Spacing")
        plt.show()
        pause(1000)




    def precipitation_plot(cube,perp_dist_diff,coh_mean_df):
        # Set time period
        start = datetime(2021, 1, 1)
        end = datetime(2023, 1, 31)

        # Create Point for Vancouver, BC
        # location = Point(49.2497, -123.1193, 70) #-1.8, 113.5, 0

        # Get daily data for 2018
        data = Daily(96655, start, end)
        data = data.fetch()


        prcp = data.groupby(pd.cut(data.index, cube.dates)).mean()['prcp'].to_frame()
        prcp['dates'] = cube.dates[:-1]  ## one less date as this is change between dates..
        prcp.name = 'Mean Precipitation'

        #x_filtered = prcp[["prcp"]]

        plt.scatter(coh_mean_df.index,pct_clip(perp_dist_diff))    # #plt.show()
        plt.plot(coh_mean_df.index, coh_mean_df, label=coh_mean_df.columns)
        plt.yticks([0, 1])
        plt.legend()
        plt.show()
        plt.pause(1000)


    def radd_alert_data(shp,):

        #####################
        ## Having difficulty here wrangling the radd alert dates into datetimes, so will pass it to a loop before plotting...
        ## save each code as aa band in a datacube...
        #####################

        radd = rioxarray.open_rasterio("D:/Data/Radd_Alert.tif", masked=True).rio.clip(
                shp.geometry.values, shp.crs, from_disk=True) ##.rio.reproject_match(cube)

        radd_cube = make_geocube(shp, like=radd, measurements=['code'])
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

