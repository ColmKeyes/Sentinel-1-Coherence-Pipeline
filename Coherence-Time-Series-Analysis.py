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
from Coherence_Time_Series import *


# TODO: plot change in backscatter between coherence fist and second images...
## show a stop-gap between some of the large gaps.
## Add Fill value of NAN for boxcar,
## combine smoothed trendline and real values as scatter.
## add nan values for large gap??
## Keep consistent X & Y axis limits,
## for RQ1, visualise how different coherence window sizes show the problem of overstimation at low number of looks :D
## for RQ2, we then want to deep dive into the disturbance events themselves, and their temporal coherence and backscatter characteristics



if __name__ == '__main__':

    window_size = 500

    path = f'D:\Data\Results\Coherence_Results\{window_size}m_window\pol_VV_coherence_window_{window_size}'
    bsc_path = f'D:\Data\Results\Coherence_Results\{window_size}m_window\pol_VV_backscatter_multilook_window_{window_size}'
    bsc_path_VH = f'D:\Data\Results\Coherence_Results\{window_size}m_window\pol_VH_backscatter_multilook_window_{window_size}'
    coh_path_VH = f'D:\Data\Results\Coherence_Results\{window_size}m_window\pol_VH_coherence_window_{window_size}'
    forest_baseline_polygon = gpd.read_file("D:\Data\\geometries\\Forest_Baseline_Polygon.shp")

    # # if stack does not exist
    # path = 'D:\Data\Results\Coherence_Results\\500m_window\\pol_VV_coherence_window_500'#20'#500
    # bsc_path = 'D:\Data\Results\Coherence_Results\\500m_window\\pol_VV_backscatter_multilook_window_500'#500'
    # bsc_path_VH = 'D:\Data\Results\Coherence_Results\\500m_window\\pol_VH_backscatter_multilook_window_500'#500'
    # coh_path_VH = 'D:\Data\Results\Coherence_Results\\500m_window\\pol_VH_coherence_window_500'#500'
    # #shp1 = gpd.read_file('D:\Data\\geometries\\all_ground_control_points_2_Point_backup_Point.shp')
    # #shp2 = gpd.read_file("D:\Data\\geometries\\all_ground_control_points_Point_1_Point_backup_Point.shp")
    # forest_baseline_polygon = gpd.read_file("D:\Data\\geometries\\Forest_Baseline_Polygon.shp")
    # #shp = shp1.append(shp2)

    #shp = shp.reset_index(drop='index')
    shp = gpd.read_file('D:\Data\\geometries\\combiend_polygons.shp')
    shp = shp.append(forest_baseline_polygon.iloc[0])
    shp = shp.reset_index(drop='index')
    shp['code'] = shp.index + 1
    ## Only include certain examples..
    ## I REALLY need a better way of keeping track of my polygons....

    shp = shp.iloc[[9,5,4,3,2,6]]#  2,3,4,5,6,8,9]] ##Number 5,7,8 Cenral Kalimantan, is the polygon that seesm to be giving me errors....
    titles =  ['Intact Forest','Farmland','Urban', '1st_Compact Event', '2nd_Compact Event' ,'3rd_Compact Event']
    #titles = #['2nd_Compact Event', '1st_Compact Event', 'Urban', 'Central_Kalimantan(Farmland)','3rd_Compact Event', 'Intact Forest']                         #'Main_Large', '7th_Compact', '2nd_Compact', '1st_Compact', 'Urban', 'Central_Kalimantan','3rd_Compact', '2nd_Sporadtic' ,'5th_Compact']

    #shp1['code'] = shp1.index + 1
    #titles = []
    tiff_stack=[]
    results_path = 'D:\\Data\\Results'
    #for ix, layer in enumerate(os.listdir(path[:45])):  ## look at upper layer in path..


    ## INEED TO FIX THESE!!!
    #if not os.path.exists(f"{results_path}\\{path[45:]}.tif"):
    titles=write_rasterio_stack(path, f"{results_path}\\{path[45:]}.tif", write=True) ## struggling to get this to do what it's supposed to...
    #titles=write_rasterio_stack(path, f"{results_path}\\{path[45:]}.tif",write=False)   #f'{layer}.tif') ## titles =

    if not os.path.exists(f"{results_path}\\{path[45:]}.tif"):
        write_rasterio_stack(path, f"{results_path}\\{path[45:]}.tif")
    if not os.path.exists(f"{results_path}\\{bsc_path[45:]}.tif"):
        write_rasterio_stack(bsc_path, f"{results_path}\\{bsc_path[45:]}.tif")   #f'{layer}.tif')
    if not os.path.exists(f"{results_path}\\{bsc_path_VH[45:]}.tif"):
        write_rasterio_stack(bsc_path_VH, f"{results_path}\\{bsc_path_VH[45:]}.tif")  # f'{layer}.tif')
    if not os.path.exists(f"{results_path}\\{coh_path_VH[45:]}.tif"):
        write_rasterio_stack(coh_path_VH, f"{results_path}\\{coh_path_VH[45:]}.tif")  # f'{layer}.tif')


    #tiff_stack.append(layer)
    tiff_stack = [f"{results_path}\\{bsc_path[45:]}.tif",f"{results_path}\\{path[45:]}.tif",f"{results_path}\\{bsc_path_VH[45:]}.tif",f"{results_path}\\{coh_path_VH[45:]}.tif"]
    #cube = build_cube(tiff_stacks=tiff_stack, shp =shp )
    cube = build_cube(tiff_stacks=tiff_stack, shp =shp )

    coh_dates = pd.to_datetime(pd.Series(titles))
    cube['dates'] = coh_dates
    #

    radd = rioxarray.open_rasterio("D:/Data/Radd_Alert.tif", masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True) ##.rio.reproject_match(cube)

    radd_cube = make_geocube(shp, like=radd, measurements=['code'])
    radd_cube["alert_date"] = (radd.dims, radd.values, radd.attrs,radd.encoding)
    #radd_stats = radd_cube.groupby(radd_cube.code)

    #radd_count = radd_stats.count()
    #radd_count["alert_dates"] = (radd.dims, radd.values, radd.attrs,radd.encoding)

    #radd_count['dates'] =  datetime.strptime(radd_cube.alert_date, '%y%j')

#####################
## Having difficulty here wrangling the radd alert dates into datetimes, so will pass it to a loop before plotting...
## save each code as aa band in a datacube...
#####################



    ################################
    ##CODES:
    ## 5=Urban,6=Farmland,7=3rd_Compact, 3=2nd_Compact, 4=1stCompact, 10=Intact_Forest
    ################################

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


    # code10 = radd_cube.where(radd_cube.code == 10).alert_date[1]#.plot.imshow() plt.pause(100)
    #
    #
    # code10_counts = np.unique(code10, return_counts=True)

    #pd.to_datetime(pd.Series(code10_counts[0].astype(str)), format="%y%j.0", errors='coerce')

    # radd_array['dates'] = pd.to_datetime(pd.Series(code10_counts[0].astype(str)), format="%y%j.0", errors='coerce')
    # radd_array['values'] = code10_counts[1]
    #radd_array.plot.line('dates', 'values')



    path_asf_csv = r'D:\Data\asf-sbas-pairs_12d_all_perp.csv'#asf-sbas-pairs_24d_35m_Jun20_Dec22.csv'
    asf_df = pd.read_csv(path_asf_csv)
    asf_df = asf_df.drop(index=61)
    perp_dist_diff = np.abs(asf_df[" Reference Perpendicular Baseline (meters)"] - asf_df[" Secondary Perpendicular Baseline (meters)"])
    perp_dist_diff.name = 'Perpendicular_Distance'

    zonal_stats = cube.groupby(cube.code)#calc_zonal_stats(cube)
    zonal_stats = zonal_stats.mean()#.rename({"coherence": "coherence_mean"})
    #zonal_transpose = zonal_stats.unstack(level='code')
    coh_VV_mean_df = zonal_stats.coherence_VV #zonal_transpose.coherence_mean
    bsc_VV_mean_df = zonal_stats.backscatter_VV
    bsc_VH_mean_df = zonal_stats.backscatter_VH
    coh_VH_mean_df = zonal_stats.coherence_VH
    # plt.imshow(coh_std_df[0])
    # plt.scatter(coh_mean_df.index,pct_clip(perp_dist_diff))
    # #plt.show()
    # plt.plot(coh_mean_df.index, coh_mean_df, label=coh_mean_df.columns)
    # plt.yticks([0, 1])
    # plt.legend()
    # plt.show()
    # plt.pause(1000)

    titles =  ['1st Disturbed Area', '2nd Disturbed Area', 'Sand & Water', 'Farmland','3rd Disturbed Area', 'Intact Forest'] #'Intact Forest','Farmland','Urban', '1st_Compact Event', '2nd_Compact Event' ,'3rd_Compact Event']

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
    prcp.name = 'Mean Precipitation'

    x_filtered = prcp[["prcp"]]

    #x_filtered =convolve(,Box1DKernel(50))

    #from statsmodels.nonparametric.smoothers_lowess import lowess

    ##so, now i want to plot the three coherence values togehter, to show the change in coherence, say for my third disturbance event,
    ## and see that due to the way coherence is estimated as the average over an infinite set of cohernece images,
    ## the higher 1km resolution images are closer to the real coherence values as they contain less bas than the lower resolution
    ## and thus lower window and higher bias images....

    plt.plot(cube.dates, convolve(coh_VV_mean_df[4], Box1DKernel(5), boundary=None), label=coh_VV_mean_df.name)  # , coh_VV_mean_df[a],
    plt.plot(cube.dates,  convolve(coh_VH_mean_df[4], Box1DKernel(5), boundary=None), label=coh_VH_mean_df.name)  # , coh_VH_mean_df[a],label=coh_VH_mean_df.name)
    plt.plot(cube.dates,  convolve(bsc_VV_mean_df[4], Box1DKernel(5), boundary=None) , label=bsc_VV_mean_df.name)  # bsc_VV_mean_df[a],label=bsc_VV_mean_df.name)
    plt.plot(cube.dates,convolve(bsc_VH_mean_df[4], Box1DKernel(5), boundary=None) , label=bsc_VH_mean_df.name)  # , bsc_VH_mean_df[a],label=bsc_VH_mean_df.name)
    plt.scatter(radd_array[f'polygon{np.unique(radd_cube.code)[4]}_dates'][7:61], pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[4]}_values'][7:61], [.2, 99.8]), label='Radd Alert Detections')
    plt.title(f'Disturbance Event 3, {window_size*2}m Resolution')
    plt.legend()
    plt.show()
    plt.pause(100)


#################
## FFT
#################
    # from matplotlib import dates as mdates
    #
    # data = [convolve(coh_VV_mean_df[4], Box1DKernel(5), boundary=None),
    #         convolve(coh_VH_mean_df[4], Box1DKernel(5), boundary=None),
    #         convolve(bsc_VV_mean_df[4], Box1DKernel(5), boundary=None),
    #         convolve(bsc_VH_mean_df[4], Box1DKernel(5), boundary=None)]
    # names = [coh_VV_mean_df.name, coh_VH_mean_df.name, bsc_VV_mean_df.name, bsc_VH_mean_df.name]
    # plt.figure(figsize=(12, 8))
    # for i, d in enumerate(data):
    #     plt.subplot(2, 2, i + 1)
    #     # Convert datetime64[ns] to matplotlib format
    #     x = mdates.date2num(cube.dates)
    #     plt.plot(x, d, label=names[i])
    #     plt.xlabel('Date')
    #     plt.ylabel('Value')
    #     plt.title(f'{names[i]}')
    #     plt.xlim(min(x), max(x))
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    #     plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    #     plt.legend()
    #     fft_data = np.fft.fft(d)
    #     freqs = np.fft.fftfreq(len(d), (cube.dates[1] - cube.dates[0]) / np.timedelta64(1, 'D'))
    #     plt.subplot(2, 2, i + 1 + len(data))
    #     plt.plot(freqs, np.abs(fft_data))
    #     plt.xlabel('Frequency (1/day)')
    #     plt.ylabel('Magnitude')
    #     plt.title(f'FFT of {names[i]}')
    #     plt.xlim(0, max(freqs))
    #     plt.ylim(0, 2 * np.abs(fft_data).max())
    #     plt.grid()
    # plt.tight_layout()
    # plt.show()
#####################


    fig, ax = plt.subplots(3, 2, figsize=(21, 7), sharey=True, sharex=True)

    convolve_data = True  # set to True to convolve data, False to not convolve

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

                ax[i, j].scatter(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_dates'][7:61],
                                 pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_values'][7:61], [.2, 99.8]),
                                 label='Radd Alert Detections')
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

## CHANGE URBAN TO WHATEVER THE HELL IT ACTUALLY IS..
    # fig, ax = plt.subplots(3, 2, figsize=(21, 7),sharey=True,sharex=True)
    # # plt.suptitle('combined groundtruth 47:73')
    # a = 0
    # for i in range(3):#len(coh_mean_df.columns)%2:
    #     for j in range(2):
    #         # plt.subplot(4,4,i+1)
    #         try:
    #             #ax[i,j].hist(cube.dates, radd_cube.where(np.unique(radd_cube.code)[a]).alert_date[1], label=coh_VV_mean_df.name)
    #             ## FILL IN WITH NAN INSTEAD OF ZEROS!!!
    #             ax[i,j].plot(cube.dates, convolve(coh_VV_mean_df[a], Box1DKernel(5),boundary=None), label=coh_VV_mean_df.name)#, coh_VV_mean_df[a],
    #             ax[i,j].plot(cube.dates, convolve(coh_VH_mean_df[a],Box1DKernel(5),boundary=None),label=coh_VH_mean_df.name)#, coh_VH_mean_df[a],label=coh_VH_mean_df.name)
    #             ax[i,j].plot(cube.dates,convolve(bsc_VV_mean_df[a],Box1DKernel(5),boundary=None),label=bsc_VV_mean_df.name)#   bsc_VV_mean_df[a],label=bsc_VV_mean_df.name)
    #             ax[i,j].plot(cube.dates, convolve(bsc_VH_mean_df[a],Box1DKernel(5),boundary=None),label=bsc_VH_mean_df.name)#, bsc_VH_mean_df[a],label=bsc_VH_mean_df.name)
    #             ax[i, j].scatter(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_dates'][7:61] ,pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_values'][7:61],[.2,99.8]),label='Radd Alert Detections')#raddy_array.index,radd_xarray[f'polygon{np.unique(radd_cube.code)[a]}_values'],[0,100]))#f'polygon{np.unique(radd_cube.code)[a]}_dates'
    #             ax[i, j].set_title(titles[a])
    #             #ax[i,j].set_ylim(0.8)
    #         except KeyError:
    #             continue
    #         except IndexError:
    #             continue
    #         #rasterio.plot.show(coh_mean_df[a], ax=ax[i, j])  # transform=transform, HRtif, ax=ax)
    #         # plt.title(titles[a],ax=ax[i,j])
    #         # ax[i,j].title.set_text(titles_colm[a])
    #         # plt.title('orthoHR - Rendiermos')
    #
    #         ##plotting overlaying polygons
    #         # combined_groundtruth_colm.plot(ax=ax[i,j], facecolor='none', edgecolor='red')#combined_groundtruth_colm['Label'] == str(titles_colm[a]
    #         a = a + 1
    # #ax.legend()




    #plt.show()


    # plt.scatter(cube.dates,pct_clip(perp_dist_diff),label=perp_dist_diff.name)
    # plt.scatter(prcp.dates, pct_clip(prcp.prcp), label=prcp.name)
    # plt.tight_layout()
    # plt.title(" Precipitation & Perpendicular Distance between Coherence Pairs")
    # plt.legend()
    # plt.show()
    #
    # plt.pause(1000)
    # #print(cube)
    # #print(zonal_stats)








#
#
#
# # Set time period
# start = datetime(2021, 1, 1)
# end = datetime(2023, 1, 31)
#
# # Create Point for Vancouver, BC
# #location = Point(49.2497, -123.1193, 70) #-1.8, 113.5, 0
#
# # Get daily data for 2018
# data = Daily(96655, start, end)
# data = data.fetch()
#
# # Plot line chart including average, minimum and maximum temperature
# # start = datetime(2021, 1, 1)
# # end = datetime(2021, 12, 31)
# a=Daily(96655,start,end)
#
#
# #data['Date'] = pd.to_datetime(data.index) - pd.to_timedelta(7, unit='d')
# #prcp = data.groupby([pd.Grouper(key='Date', freq='W-MON')])['prcp'].sum().reset_index().sort_values('Date')
#
# prcp = data.groupby(pd.cut(data.index,cube.dates)).sum()['prcp']
# prcp['dates']= cube.dates[:-1] ## one less date as this is change between dates..
# df.plot(y=['prcp'])
# plt.show()
# plt.pause(100)
#













