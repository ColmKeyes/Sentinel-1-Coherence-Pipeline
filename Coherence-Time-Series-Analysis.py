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




def write_rasterio_stack(path, write_file, titles=None ):
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
                dst.write_band(ix, pct_clip(src1.read(1)))
        print(f'Total Images stacked: {ix}')
        dst.close()
        return titles

def build_cube(tiff_stacks, shp=None ):
    ## two slightly different ways to build an xarray datacube

    if shp is None:
        stack_coherence = rioxarray.open_rasterio(tiff_stacks[0], masked=False)
        stack_backscatter = rioxarray.open_rasterio(tiff_stacks[1], masked=False)

        ## open_rasterio opens as xarray DataArrays, we must convert to Datasets before combining both into a data-cube..
        stack_coherence = stack_coherence.to_dataset(name='coherence')
        stack_backscatter = stack_backscatter.to_dataset(name='backscatter')

        cube = xarray.merge([stack_coherence, stack_backscatter])

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
        shp_stack_backscatter['code']= shp_stack_backscatter.band +1
        shp_stack_backscatter_VH['code']= shp_stack_backscatter_VH.band +1
        shp_stack_coh_VH['code']= shp_stack_coh_VH.band +1



        cube = make_geocube(shp,like=shp_stack ,measurements=['code'])
        cube['coherence_VV'] = (shp_stack.dims,shp_stack.values,shp_stack.attrs,shp_stack.encoding)
        cube["coherence_VH"] = (shp_stack_coh_VH.dims, shp_stack_coh_VH.values,shp_stack_coh_VH.attrs,shp_stack_coh_VH.encoding)
        ## squeezing last weird dim length...
        cube = cube.isel(x=range(0, 142),drop=True)#x=142), drop=True)#y=133
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





if __name__ == '__main__':

    # if stack does not exist
    path = 'D:\Data\Results\Coherence_Results\pol_VV_coherence_window_500'
    bsc_path = 'D:\Data\Results\Coherence_Results\pol_VV_backscatter_multilook_window_500'
    bsc_path_VH = 'D:\Data\Results\Coherence_Results\pol_VH_backscatter_multilook_window_500'
    coh_path_VH = 'D:\Data\Results\Coherence_Results\pol_VH_coherence_window_500'
    shp1 = gpd.read_file('D:\Data\\geometries\\all_ground_control_points_2_Point_backup_Point.shp')
    shp2 = gpd.read_file("D:\Data\\geometries\\all_ground_control_points_Point_1_Point_backup_Point.shp")
    shp = shp1.append(shp2)
    #shp = shp.reset_index(drop='index')
    shp = gpd.read_file('combiend_polygons.shp')
    shp['code'] = shp.index + 1
    shp1['code'] = shp1.index + 1
    titles = []
    tiff_stack=[]

    #for ix, layer in enumerate(os.listdir(path[:34])):  ## look at upper layer in path..
    titles=write_rasterio_stack(path, f"{path[34:]}.tif")   #f'{layer}.tif')
    write_rasterio_stack(bsc_path, f"{bsc_path[34:]}.tif")   #f'{layer}.tif')
    write_rasterio_stack(bsc_path_VH, f"{bsc_path_VH[34:]}.tif")  # f'{layer}.tif')
    write_rasterio_stack(coh_path_VH, f"{coh_path_VH[34:]}.tif")  # f'{layer}.tif')


    #tiff_stack.append(layer)
    tiff_stack = [f"{bsc_path[34:]}.tif",f"{path[34:]}.tif",f"{bsc_path_VH[34:]}.tif",f"{coh_path_VH[34:]}.tif"]
    #cube = build_cube(tiff_stacks=tiff_stack, shp =shp )
    cube = build_cube(tiff_stacks=tiff_stack, shp =shp )
    coh_dates = pd.to_datetime(pd.Series(titles))
    cube['dates'] = coh_dates

    path_asf_csv = r'D:\Data\asf-sbas-pairs_12d_all_perp.csv'#asf-sbas-pairs_24d_35m_Jun20_Dec22.csv'
    asf_df = pd.read_csv(path_asf_csv)
    asf_df = asf_df.drop(index=61)
    perp_dist_diff = np.abs(asf_df[" Reference Perpendicular Baseline (meters)"] - asf_df[" Secondary Perpendicular Baseline (meters)"])
    perp_dist_diff.name = 'Perpendicular_Distance'

    zonal_stats = cube.groupby(cube.code)#calc_zonal_stats(cube)
    zonal_stats = zonal_stats.mean()#.rename({"coherence": "coherence_mean"})
    #zonal_transpose = zonal_stats.unstack(level='code')
    coh_mean_df = zonal_stats.coherence_VV #zonal_transpose.coherence_mean
    bsc_VV_mean_df = zonal_stats.backscatter_VV
    bsc_VH_mean_df = zonal_stats.backscatter_VH
    coh_VH_mean_df = zonal_stats.coherence_VH
    titles = ['Main_Large', '7th_Compact', '2nd_Compact', '1st_Compact', 'Urban', 'Central_Kalimantan','3rd_Compact', '2nd_Sporadtic' ,'5th_Compact']

    # plt.imshow(coh_std_df[0])
    # plt.scatter(coh_mean_df.index,pct_clip(perp_dist_diff))
    # #plt.show()
    # plt.plot(coh_mean_df.index, coh_mean_df, label=coh_mean_df.columns)
    # plt.yticks([0, 1])
    # plt.legend()
    # plt.show()
    # plt.pause(1000)

    # Import Meteostat library and dependencies
    from datetime import datetime
    import matplotlib.pyplot as plt
    from meteostat import Point, Daily

    # Set time period
    start = datetime(2021, 1, 1)
    end = datetime(2023, 1, 31)

    # Create Point for Vancouver, BC
    # location = Point(49.2497, -123.1193, 70) #-1.8, 113.5, 0

    # Get daily data for 2018
    data = Daily(96655, start, end)
    data = data.fetch()

    # Plot line chart including average, minimum and maximum temperature
    # data.plot(y=['prcp'])#tavg', 'tmin', 'tmax'])
    # plt.show()
    # plt.pause(10)
    # start = datetime(2021, 1, 1)
    # end = datetime(2021, 12, 31)
    # a=Daily(96655,start,end)

    # data['Date'] = pd.to_datetime(data.index) - pd.to_timedelta(7, unit='d')
    # prcp = data.groupby([pd.Grouper(key='Date', freq='W-MON')])['prcp'].sum().reset_index().sort_values('Date')

    prcp = data.groupby(pd.cut(data.index, cube.dates)).mean()['prcp'].to_frame()
    prcp['dates'] = cube.dates[:-1]  ## one less date as this is change between dates..
    prcp.name = 'Precipitation'






    fig, ax = plt.subplots(5, 2, figsize=(21, 7))
    # plt.suptitle('combined groundtruth 47:73')
    a = 0
    for i in range(5):#len(coh_mean_df.columns)%2:
        for j in range(2):
            # plt.subplot(4,4,i+1)
            try:
                ax[i,j].plot(cube.dates, coh_mean_df[a], label=coh_mean_df.name)
                ax[i,j].scatter(cube.dates,pct_clip(perp_dist_diff),label=perp_dist_diff.name)
                ax[i,j].plot(cube.dates, bsc_VV_mean_df[a],label=bsc_VV_mean_df.name)#, label=coh_mean_df.columns)
                ax[i,j].plot(cube.dates, bsc_VH_mean_df[a],label=bsc_VH_mean_df.name)#, label=coh_mean_df.columns)
                ax[i,j].plot(cube.dates, coh_VH_mean_df[a],label=coh_VH_mean_df.name)#, label=coh_mean_df.columns)
                ax[i, j].scatter(prcp.dates, pct_clip(prcp.prcp),label = prcp.name )
                ax[i, j].set_title(titles[a])
            except KeyError:
                continue
            except IndexError:
                continue
            #rasterio.plot.show(coh_mean_df[a], ax=ax[i, j])  # transform=transform, HRtif, ax=ax)
            # plt.title(titles[a],ax=ax[i,j])
            # ax[i,j].title.set_text(titles_colm[a])
            # plt.title('orthoHR - Rendiermos')

            ##plotting overlaying polygons
            # combined_groundtruth_colm.plot(ax=ax[i,j], facecolor='none', edgecolor='red')#combined_groundtruth_colm['Label'] == str(titles_colm[a]
            a = a + 1
    #ax.legend()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes][0]
    #lines, labels = [(label, []) for label in zip(*lines_labels1)][0] #sum(lol, [])
    lines_labels = [[line for line,label in zip(*lines_labels)],[label for line,label in zip(*lines_labels)]]
    # Finally, the legend (that maybe you'll customize differently)
    fig.legend(lines_labels[0], lines_labels[1], loc='upper center', ncol=4)
    fig.tight_layout()
    plt.show()
    #plt.show()
    plt.pause(10)
    #print(cube)
    #print(zonal_stats)













# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2021, 1, 1)
end = datetime(2023, 1, 31)

# Create Point for Vancouver, BC
#location = Point(49.2497, -123.1193, 70) #-1.8, 113.5, 0

# Get daily data for 2018
data = Daily(96655, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['prcp'])#tavg', 'tmin', 'tmax'])
plt.show()
plt.pause(10)
# start = datetime(2021, 1, 1)
# end = datetime(2021, 12, 31)
a=Daily(96655,start,end)


#data['Date'] = pd.to_datetime(data.index) - pd.to_timedelta(7, unit='d')
#prcp = data.groupby([pd.Grouper(key='Date', freq='W-MON')])['prcp'].sum().reset_index().sort_values('Date')

prcp = data.groupby(pd.cut(data.index,cube.dates)).sum()['prcp']
prcp['dates']= cube.dates[:-1] ## one less date as this is change between dates..
df.plot(y=['prcp'])
plt.show()
plt.pause(100)














