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
path = 'D:\Data\Results\Coherence_Results\pol_VV_coh_window_45'
titles = []



def write_rasterio_stack(path, mode, save_file, titles=None ):
    ## Path = path to folder containing tiffs...
    ##mode = coherence or backscatter ## I don't think this mode actually matters...
    ## get sample metadata..
    with rasterio.open(path + '\\' +str(os.listdir(path)[0])) as src0:
        meta = src0.meta
    meta.update(count=len(os.listdir(path)))


    with rasterio.open(save_file, 'w', **meta) as dst:
        for ix, layer in enumerate(os.listdir(path), start=1):
            with rasterio.open(path + '\\' +str(layer)) as src1:
                titles.append([str(layer)[17:-4]]) ## get list of image dates..
                dst.write_band(ix, src1.read(1))
        print(f'Total Images stacked: {ix}')
        dst.close()






def build_cube(tiff_stacks, shp=None ):


    #shp['label'] = ['Main_Large','7th_Compact','2nd_Compact','1st_Compact','Urban','Central_Kalimantan','3rd_Compact','2nd_Sporadtic','5th_Compact']
    #shp.loc[4,'label'] = 'Urban'

    if shp is None:
        ## Read in using rioxarray as it will properly read in tiffs,
        stack_coherence = rioxarray.open_rasterio(tiff_stacks[0], masked=False)
        stack_backscatter = rioxarray.open_rasterio(tiff_stacks[1], masked=False)

        ## as open_rasterio opens as xarray Dataarrays, we must convert to
        ## Datasets before we combine both into a data-cube..
        stack_coherence = stack_coherence.to_dataset(name='coherence')
        stack_backscatter = stack_backscatter.to_dataset(name='backscatter')

        ## Add code variable,to group by?...
        stack_coherence['code'] = stack_coherence.band + 1
        stack_backscatter['code'] = stack_backscatter.band + 1

        cube = xarray.merge([stack_coherence, stack_backscatter])




    if shp is not None:
        shp_stack = rioxarray.open_rasterio(tiff_stacks[0], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")

        shp_stack_backscatter = rioxarray.open_rasterio(tiff_stacks[1], masked=True).rio.clip(
            shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")

        shp_stack_backscatter['code']= shp_stack_backscatter.band +1

        ## so,firstly I need to load my data into dataarrays, then I need to looad my data arrays into a dataset, followed by computations...\=
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
    shp_with_statistics = shp.merge(zonal_stats,on='code')

    #cube.coherence[0].plot.imshow()
    #plt.title('Coherence with polygons - Oct 28th + Dec 10th 2022')
    #plt.pause(10)

    ## This part deoens't make any sense anymore, as I now don't have 'code' to combine on..
    ##
    #stack_groupby = combined_stack.groupby(combined_stack.code)

    grid_mean = stack_groupby.mean().rename({"coherence": "coherence_mean"})
    grid_min = stack_groupby.min().rename({"coherence": "coherence_min"})
    grid_max = stack_groupby.max().rename({"coherence": "coherence_max"})
    grid_std = stack_groupby.std().rename({"coherence": "coherence_std"})
    zonal_stats = xarray.merge([grid_mean, grid_min, grid_max, grid_std]).to_dataframe()
    shp_with_statistics = shp.merge(zonal_stats,on='code')































































#ccd, pre_coherence, co_coherence = calc_coherence_change(rasterio.open('stack.tif').read(), meta=meta, method='difference')    #       #
# plot_ccd(ccd,'20200826',method='difference')


#from CCD_animation import ccd_animation

#ccd_animation(rasterio.open('stack.tif'))

############################################
############################################
## some useful maps...
######################

############
## Tree Cover 2000, Hansen, GLAD
#############
rasterio.open(path+'Hansen_GFC-2021-v1.9_treecover2000_00N_110E.tif')
#############
## Tree Cover loss 2021, Hansen
#############
rasterio.open(path+'Hansen_GFC-2021-v1.9_lossyear_00N_110E.tif')

######################
############################################################
## creating coherence and backscatter xarray data cubes...
############################################################


# shp1 = gpd.read_file('D:\Data\intial_geometry_kalimantan_Polygon.shp')
# shp2 = gpd.read_file('D:\Data\second_geometry_kalimantan_Polygon_Polygon.shp')
# shp2 = shp2.drop(index=1)  ## drop a shp that is gone bad, can't find it manually to remove..
# shp = shp1.append(shp2)
# shp = shp.reset_index(drop='index')
# shp['code'] = shp.index + 1


#################
## GCP shapes...
#################

# shp_gcp = gpd.read_file('D:\Data\\all_ground_control_points_Point.shp')
# shp_gcp['code']= shp_gcp.index +1
# cube_gcp = make_geocube(shp_gcp,like=stack_coherence,measurements=['code'])


###########################################################################################################################
###########################################################################################################################
######################
## So, maybe here I don't need to join the zonal stats back to the subject area,
## because I'm just going to plot these points in a scatter, I'll leave zonal stats as it is & use its
## band index to identify the time. I don't need to imshow the zonal_stats as they are overall stats for my
## areas, => they will be in the form of single point numbers...
## so, plot the subject polygons through time using imshow and then show their statistics on a graph :)
## also, get some actual visual data through which I can show the physical evolution I'm comparing to.
## possibly use just a screenshot of me going through dates on the global forest watch planet data :))
######################



ax1 = zonal_stats.unstack(level='code')

coh_mean_df = ax1.coherence_mean.transpose()
#coh_mean_df['label'] = ['Main_Large','7th_Compact','2nd_Compact','1st_Compact','Urban','Central_Kalimantan','3rd_Compact','2nd_Sporadtic','5th_Compact']
coh_std_df = ax1.coherence_std
coh_min_df = ax1.coherence_min
coh_max_df = ax1.coherence_max


fig, ax = plt.subplots()
coh_mean_df.transpose().plot.line(ax=ax)
plt.legend(['Main_Large','7th_Compact','2nd_Compact','1st_Compact','Urban','Central_Kalimantan','3rd_Compact','2nd_Sporadtic','5th_Compact'])
plt.title('Mean Coherence for selected Polygons - June 2020 - Dec 2022' )
plt.xlabel('Image Number')
plt.ylabel('Coherence')
plt.ylim(0,0.8)

#plt.imshow(coh_std_df[0])
plt.plot(coh_mean_df.columns,coh_mean_df[coh_mean_df.columns[0:33]], label=coh_mean_df.label)
#plt.yticks([0,0.5])
#plt.legend()
#plt.show()
#plt.pause(1000)










#####################

# This assumes you are running this example from a clone of
# https://github.com/corteva/geocube/
# You could also use the full path:
# https://raw.githubusercontent.com/corteva/geocube/master/test/test_data/input/soil_data_group.geojson
ssurgo_data = gpd.read_file("../../test/test_data/input/soil_data_group.geojson")
ssurgo_data = ssurgo_data.loc[ssurgo_data.hzdept_r==0]
# convert the key to group to the vector data to an integer as that is one of the
# best data types for this type of mapping. If your data is not integer,
# then consider using a mapping of your data to an integer with something
# like a categorical dtype.
ssurgo_data["mukey"] = ssurgo_data.mukey.astype(int)
# load in source elevation data subset relevant for the vector data
elevation = rioxarray.open_rasterio(
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n42w091/USGS_13_n42w091.tif", masked=True
).rio.clip(
    ssurgo_data.geometry.values, ssurgo_data.crs, from_disk=True
).sel(band=1).drop("band")
elevation.name = "elevation"

out_grid = make_geocube(
    vector_data=ssurgo_data,
    measurements=["mukey"],
    like=elevation, # ensure the data are on the same grid
)












































#riox.open_rasterio("stack.tif")
shp1 = gpd.read_file('D:\Data\intial_geometry_kalimantan_Polygon.shp')
shp2 = gpd.read_file('D:\Data\second_geometry_kalimantan_Polygon_Polygon.shp')
shp2 = shp2.drop(index=1) ## drop a shp that is gone bad, can't find it manually to remove..
shp= shp1.append(shp2)
shp=shp.reset_index(drop='index')
shp['code']= shp.index +1
#shp['label'] = ['Main_Large','7th_Compact','2nd_Compact','1st_Compact','Urban','Central_Kalimantan','3rd_Compact','2nd_Sporadtic','5th_Compact']
#shp.loc[4,'label'] = 'Urban'

#xar.plot.imshow(riox.open_rasterio("stack.tif"))
#riox.open_rasterio("stack.tif")[1]
#xar.open_rasterio('stack.tif').plot.imshow()[1]



# load in source elevation data subset relevant for the vector data
## this will just clop the
shp_stack = rioxarray.open_rasterio("coh_45_stack.tif", masked=True).rio.clip(
    shp.geometry.values, shp.crs, from_disk=True)#.sel(band=1).drop("band")

#########

#coh_stack = rioxarray.open_rasterio("stack.tif")
cube = make_geocube(shp,like=shp_stack,measurements=['code'])

## So before this, we have just created a datacube of specific dimensions. with this, we can then
## go and add attribute values to this empty cube, by attaching the below 'coherece' data.
cube['coherence'] = (shp_stack.dims,shp_stack.values,shp_stack.attrs,shp_stack.encoding)

grouped_coherence = cube.groupby(cube.code)  ## so this is treating it as a geopandas geodataframe...
grid_mean = grouped_coherence.mean().rename({"coherence": "coherence_mean"})
grid_min = grouped_coherence.min().rename({"coherence": "coherence_min"})
grid_max = grouped_coherence.max().rename({"coherence": "coherence_max"})
grid_std = grouped_coherence.std().rename({"coherence": "coherence_std"})
zonal_stats = xarray.merge([grid_mean, grid_min, grid_max, grid_std]).to_dataframe()
##What I want
#cube1 = cube.merge(zonal_stats,'code')
#shp_with_statistics = shp.merge(zonal_stats,on='code')
#shp_with_statistics.plot(column='coherence_mean',legend=True)



#make_geocube(zonal_stats,shp_stack, group_by='code','band')


######################
## So, maybe here I don't need to join the zonal stats back to the subject area,
## because I'm just going to plot these points in a scatter, I'll leave zonal stats as it is & use its
## band index to identify the time. I don't need to imshow the zonal_stats as they are overall stats for my
## areas, => they will be in the form of single point numbers...
## so, plot the subject polygons through time using imshow and then show their statistics on a graph :)
## also, get some actual visual data through which I can show the physical evolution I'm comparing to.
## possibly use just a screenshot of me going through dates on the global forest watch planet data :))
######################



ax1 = zonal_stats.unstack(level='code')

coh_mean_df = ax1.coherence_mean.transpose()
coh_mean_df['label'] = ['Main_Large','7th_Compact','2nd_Compact','1st_Compact','Urban','Central_Kalimantan','3rd_Compact','2nd_Sporadtic','5th_Compact']
coh_std_df = ax1.coherence_std
coh_min_df = ax1.coherence_min
coh_max_df = ax1.coherence_max


#plt.imshow(coh_std_df[0])
plt.plot(coh_mean_df.index,coh_mean_df, label=coh_mean_df.columns)
plt.yticks([0,0.5])
plt.legend()
plt.show()
plt.pause(1000)






#####################

# This assumes you are running this example from a clone of
# https://github.com/corteva/geocube/
# You could also use the full path:
# https://raw.githubusercontent.com/corteva/geocube/master/test/test_data/input/soil_data_group.geojson
ssurgo_data = gpd.read_file("../../test/test_data/input/soil_data_group.geojson")
ssurgo_data = ssurgo_data.loc[ssurgo_data.hzdept_r==0]
# convert the key to group to the vector data to an integer as that is one of the
# best data types for this type of mapping. If your data is not integer,
# then consider using a mapping of your data to an integer with something
# like a categorical dtype.
ssurgo_data["mukey"] = ssurgo_data.mukey.astype(int)
# load in source elevation data subset relevant for the vector data
elevation = rioxarray.open_rasterio(
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n42w091/USGS_13_n42w091.tif", masked=True
).rio.clip(
    ssurgo_data.geometry.values, ssurgo_data.crs, from_disk=True
).sel(band=1).drop("band")
elevation.name = "elevation"

out_grid = make_geocube(
    vector_data=ssurgo_data,
    measurements=["mukey"],
    like=elevation, # ensure the data are on the same grid
)






































