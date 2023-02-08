# -*- coding: utf-8 -*-
"""
This script provides interface to Sentinel-1 preprocessing through SNAP's Graph Processing Framework defined in Javascript
"""
"""
@Time    : 07/12/2022 13:39
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Sentinel-1 SLC Preprocessing
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

##############
## steps needed are:
## TopSAR-Split
## Apply-Orbit-file
## Back-Geocoding
## Coherence
## TOPSAR-Deburst
## Terrain-Correction
## Write
##############

def plotBand(product, band, vmin, vmax):
    band = product.getBand(band)
    w = band.getRasterWidth()
    h = band.getRasterHeight()
    print(w, h)
    band_data = np.zeros(w * h, np.float32)
    band.readPixels(0, 0, w, h, band_data)
    band_data.shape = h, w
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    imgplot = plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
    return imgplot

# band_names = subset.getBandNames()
# band = subset.getBand(band_names[0])

#plotBand(subset,'coh_IW2_VV_02Mar2021_18Feb2021',0,1)
#plotBand(backgeocoding,'i_IW2_VV_mst_02Mar2021',0,1000)


def topsar_split(source,pols,iw_swath,first_burst_index,last_burst_index):
    print('\tOperator-TOPSAR-Split...')
    parameters = HashMap()
    parameters.put('subswath',iw_swath)#'IW2')
    parameters.put('selectedPolarisations',pols)
    parameters.put('firstBurstIndex',first_burst_index) #4)
    parameters.put('lastBurstIndex',last_burst_index) #7)
    output = GPF.createProduct('TOPSAR-Split', parameters, source)
    return output


def apply_orbit_file(source):
    print('\tApply orbit file...')
    parameters = HashMap()
    #parameters.put('Apply-Orbit-File', True)
    parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
    parameters.put('polyDegree', '3')
    parameters.put('continueOnFail', 'false')
    print(source.getBand(source.getBandNames()[0]))
    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output


def back_geocoding(sources):
    print('\tOperator-Back-Geocoding...')
    parameters = HashMap()
    parameters.put('demName','SRTM 3Sec')
    parameters.put('externalDEMNoDataValue',0.0)
    print(sources[0].getBand(sources[0].getBandNames()[0]))
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION') ## Perhaps give as a param...
    parameters.put('maskOutAreaWithoutElevation','false')
    parameters.put('outputRangeAzimuthOffset','false')
    parameters.put('outputDerampDemodPhase','false')
    parameters.put('disableReramp','false')
    output = GPF.createProduct('Back-Geocoding', parameters, sources) ##  I think I just need to change source here to an array of products, one from each apply orbit file...
    return output

def thermal_noise_reduction(source,pols):
    print('\tOperator-Thermal-Noise-Removal...')
    parameters = HashMap()
    parameters.put('selectedPolarisations', pols)
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval',parameters,source)
    return output

def calibration_(source, pols):
    print('\tOperator-Radiometric-Calibration...')
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)
    parameters.put('outputGammaBand', False)
    parameters.put('outputBetaBand', False)
    parameters.put('selectedPolarisations',pols)
    output = GPF.createProduct('Calibration', parameters, source)
    return output


def coherence_(source, coh_window_size):
    print('\tOperator-Coherence...')
    parameters = HashMap()
    parameters.put('cohWinAz',coh_window_size[0])#3)
    parameters.put('cohWinRg',coh_window_size[1])#15)
    print(source.getBand(source.getBandNames()[0]))
    parameters.put('subtractFlatEarthPhase',False)
    parameters.put('srpPolynomialDegree',5)
    parameters.put('srpNumberPoints',501)
    parameters.put('orbitDegree',3)
    parameters.put('subtractTopographicPhase',True)
    parameters.put('demName','SRTM 3Sec')
    parameters.put('externalDEMNoDataValue',0.0)
    parameters.put('externalDEMApplyEGM', True)
    parameters.put('tileExtensionPercent','100')
    parameters.put('singleMaster',True)
    #parameters.put('subtractTopographicPhase',True)
    parameters.put('squarePixel',False)
    output = GPF.createProduct('Coherence',parameters,source)
    return output

def speckle_filtering(source, filter, filter_size):
    print('\tSpeckle filtering...')
    parameters = HashMap()
    parameters.put('filter', filter)#'Lee')
    parameters.put('filterSizeX', filter_size[0])#5)
    parameters.put('filterSizeY', filter_size[1])#5)
    output = GPF.createProduct('Speckle-Filter', parameters, source)
    return output



def terrain_correction(source,coh_window_size):#, proj):
    print('\tTerrain correction...')
    parameters = HashMap()
    parameters.put('demName', 'SRTM 3Sec')
    #parameters.put('pixelSpacingInMeter',20)
    #parameters.put('demName', 'GETASSE30')
    band_names = source.getBandNames()
    #band = source.getBand(band_names[0])
    parameters.put('sourceBands', band_names[0])
    parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('SEMResamplingMethod', 'BILINEAR_INTERPOLATION')

    #parameters.put('mapProjection', proj)       # comment this line if no need to convert to UTM/WGS84, default is WGS84
    #parameters.put('saveProjectedLocalIncidenceAngle', True)
    parameters.put('saveSelectedSourceBand', True)

    #while downsample == 1:                      # downsample: 1 -- need downsample to 40m, 0 -- no need to downsample
    parameters.put('pixelSpacingInMeter',np.double(coh_window_size[0]*coh_window_size[1])) #, 45.0)
    #    break
    output = GPF.createProduct('Terrain-Correction', parameters, source)
    return output

def multi_look(source, window_size):
    print('\tOperator-Multilook...')
    parameters = HashMap()
    parameters.put('nAzLooks', window_size[0])
    parameters.put('nRgLooks', window_size[1])
    parameters.put('outputIntensity', True)
    parameters.put('grSquarePixel',False)
    output = GPF.createProduct('Multilook', parameters, source)
    return output

def topsar_deburst(source,pols):
    print('\tOperator-TOPSAR-Deburst...')
    parameters = HashMap()
    parameters.put('selectedPolarisations', pols)
    output = GPF.createProduct('TOPSAR-Deburst', parameters,source)
    return output


def main(pols,
         iw_swath,
         first_burst_index,
         last_burst_index,
         coh_window_size,
         mode,
         speckle_filter,
         speckle_filter_size,
         product_type,
         outpath):

    SLC_path = r'D:\Data\SLC'
    shapes = r'C:\Users\Lord Colm\Desktop\InSAR Thesis\Data\Primary_Disturbance-polygon.shp'
    path_asf_csv = r'D:\Data\asf-sbas-pairs_12d_all_perp.csv'#asf-sbas-pairs_24d_35m_Jun20_Dec22.csv'

    shpfile = 'D:\Data\geometry_Polygon.shp'
    if not os.path.exists(outpath):
        os.makedirs(outpath)


    folder_paths = []
    asf_csv = pd.read_csv(path_asf_csv)
    for ref, sec in asf_csv.iterrows():
        primary = sec['Reference']
        secondary = sec['Secondary']
        print(sec['Reference'])
        print(sec['Secondary'])

        gc.enable()
        gc.collect()
        loopstarttime=str(datetime.datetime.now())
        print('Start time:', loopstarttime)
        start_time = time.time()

        if mode == 'coherence':

            sentinel_1_1 = ProductIO.readProduct(SLC_path + "\\" + primary + '.zip')
            sentinel_1_2 = ProductIO.readProduct(SLC_path + "\\" + secondary + '.zip')# os.listdir(path)[os.listdir(path).index(folder) + 1])

            width = sentinel_1_1.getSceneRasterWidth()
            print("Width: {} px".format(width))
            height = sentinel_1_1.getSceneRasterHeight()
            print("Height: {} px".format(height))
            name = sentinel_1_1.getName()
            print("Name: {}".format(name))
            band_names = sentinel_1_1.getBandNames()
            print("Band names: {}".format(", ".join(band_names)))

            #thermalnoisereduction_1 = thermal_noise_reduction(sentinel_1_1,pols)
            #thermalnoisereduction_2 = thermal_noise_reduction(sentinel_1_2,pols)
            topsarsplit_1 = topsar_split(sentinel_1_1,pols,iw_swath,first_burst_index,last_burst_index)
            applyorbit_1 = apply_orbit_file(topsarsplit_1)
            topsarsplit_2 = topsar_split(sentinel_1_2,pols,iw_swath,first_burst_index,last_burst_index)
            applyorbit_2 = apply_orbit_file(topsarsplit_2)
            backgeocoding = back_geocoding([applyorbit_1,applyorbit_2])
            coherence = coherence_(backgeocoding,coh_window_size)
            topsardeburst = topsar_deburst(coherence,pols)
            terraincorrection = terrain_correction(topsardeburst,coh_window_size)#,proj)

            #del thermalnoisereduction_1
            #del thermalnoisereduction_2
            del applyorbit_1
            del applyorbit_2
            del topsarsplit_1
            del topsarsplit_2
            del backgeocoding
            del coherence
            del topsardeburst


        elif mode == 'backscatter':
            sentinel_1_1 = ProductIO.readProduct(SLC_path + "\\" + primary + '.zip')

            width = sentinel_1_1.getSceneRasterWidth()
            print("Width: {} px".format(width))
            height = sentinel_1_1.getSceneRasterHeight()
            print("Height: {} px".format(height))
            name = sentinel_1_1.getName()
            print("Name: {}".format(name))
            band_names = sentinel_1_1.getBandNames()
            print("Band names: {}".format(", ".join(band_names)))

            thermalnoisereduction = thermal_noise_reduction(sentinel_1_1,pols)
            topsarsplit_1 = topsar_split(thermalnoisereduction,pols,iw_swath,first_burst_index,last_burst_index)
            applyorbit_1 = apply_orbit_file(topsarsplit_1)
            calibration = calibration_(applyorbit_1,pols)
            topsardeburst = topsar_deburst(calibration,pols)
            multilook = multi_look(topsardeburst,coh_window_size)
            terraincorrection = terrain_correction(multilook,coh_window_size)#,proj)
            #speckle = speckle_filtering(terraincorrection, speckle_filter, speckle_filter_size)

            del thermalnoisereduction
            del applyorbit_1
            del topsarsplit_1
            del calibration
            del topsardeburst
            del multilook
            #del terraincorrection

        print("Plotting...")
        #plotBand(speckle, 'coh_IW2_VV_02Mar2021_18Feb2021', 0, 1)
        print("Writing...")




        if mode == 'coherence':
            write_tiff_path = outpath + '\\' + primary[:25] + '_' + secondary[17:25] + '_pol_' + str(pols) + '_coherence_window_' + str(coh_window_size[0] * coh_window_size[1])
            if not os.path.exists(write_tiff_path + '.tif'):
                ProductIO.writeProduct(terraincorrection, write_tiff_path, product_type)  # 'BEAM-DIMAP')
            sentinel_1_1.dispose()
            sentinel_1_1.closeIO()
            sentinel_1_2.dispose()
            sentinel_1_2.closeIO()
            del terraincorrection

        elif mode == "backscatter":
            write_tiff_path = outpath + '\\' + primary[:25] + '_pol_' + str(pols) + '_backscatter_multilook_window_' + str(coh_window_size[0] * coh_window_size[1])
            if not os.path.exists(write_tiff_path + '.tif'):
                ProductIO.writeProduct(terraincorrection, write_tiff_path, product_type)  # 'BEAM-DIMAP')
            sentinel_1_1.dispose()
            sentinel_1_1.closeIO()
            del terraincorrection
            #del speckle

        print('Done.')
        #del speckle
        print("--- %s seconds ---" % (time.time() - start_time))



#snappy thermal noise doesn't work for coherence