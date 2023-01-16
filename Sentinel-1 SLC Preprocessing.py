# -*- coding: utf-8 -*-
"""
@Time    : 07/12/2022 13:39
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Sentinel-1 SLC Preprocessing
"""



import datetime
import time
from snappy import ProductIO
from snappy import HashMap
from snappy import WKTReader
## Garbage collection to release memory of objects no longer in use.
import os, gc
from snappy import GPF
#import shapefile
#import pygeoif
#import jpy
import zipfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
##############
## To Do:
## Speckle filter loop,
## Coherence size loop & Pixel spacing
## Burst selection at start of file?
## & now for each of the areas that I have created on my google earth, I need to loop.
##############

##############
## Don't need this one, steps that I need are:
## TopSAR-Split
## Apply-Orbit-file
## Back-Geocoding
## Coherence
## TOPSAR-Deburst
## Terrain-Correction
## Write
##############

##########################
# each of the below steps contain the following common structures:
## Parameters = Hashmap,
# This is creating an essential dictionary like object that will be read into Java.
## Parameters.put(<Name>,input)
# This will take a set name of an input parameter for this Graph processing step, and will add the 'input' into the
# middle of this function name. e.g. <subswath>string</subswath>
## GPF.createProduct(<Operator-Name>, <Parameter-Hashmap>, <Source-file-to-change>)
# Here we simply create the new product/output by applying our operator, with the Hashmap of parameters for the operation
# & the source file to be changed.
## This is the structure of every on eof the SNAPPY operations!
##########################

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


def topsar_split(source,pols):
    print('\tOperator-TOPSAR-Split...')
    parameters = HashMap()
    parameters.put('subswath','IW2')
    parameters.put('selectedPolarisations',pols)
    parameters.put('firstBurstIndex',4)
    parameters.put('lastBurstIndex',7)
    output = GPF.createProduct('TOPSAR-Split', parameters, source)
    return output


def apply_orbit_file(source):
    print('\tApply orbit file...')
    parameters = HashMap()
    #parameters.put('Apply-Orbit-File', True)
    parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
    parameters.put('polyDegree', '3')
    parameters.put('continueOnFail', 'false')
    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output


def back_geocoding(sources):
    print('\tOperator-Back-Geocoding...')
    parameters = HashMap()
    parameters.put('demName','SRTM 3Sec')
    parameters.put('externalDEMNoDataValue',0.0)
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('maskOutAreaWithoutElevation','false')
    parameters.put('outputRangeAzimuthOffset','false')
    parameters.put('outputDerampDemodPhase','false')
    parameters.put('disableReramp','false')
    output = GPF.createProduct('Back-Geocoding', parameters, sources) ##  I think I just need to change source here to an array of products, one from each apply orbit file...
    return output


def coherence_(source):
    print('\tOperator-Coherence...')
    parameters = HashMap()
    parameters.put('cohWinAz',3)
    parameters.put('cohWinRg',15)
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

def speckle_filtering(source):
    print('\tSpeckle filtering...')
    parameters = HashMap()
    parameters.put('filter', 'Lee')
    parameters.put('filterSizeX', 5)
    parameters.put('filterSizeY', 5)
    output = GPF.createProduct('Speckle-Filter', parameters, source)
    return output



def terrain_correction(source):#, proj):
    print('\tTerrain correction...')
    parameters = HashMap()
    parameters.put('demName', 'SRTM 3Sec')
    #parameters.put('pixelSpacingInMeter',20)
    #parameters.put('demName', 'GETASSE30')
    band_names = source.getBandNames()
    #band = source.getBand(band_names[0])
    parameters.put('sourceBands', band_names[0])
    #parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    #parameters.put('mapProjection', proj)       # comment this line if no need to convert to UTM/WGS84, default is WGS84
    #parameters.put('saveProjectedLocalIncidenceAngle', True)
    parameters.put('saveSelectedSourceBand', True)

    #while downsample == 1:                      # downsample: 1 -- need downsample to 40m, 0 -- no need to downsample
    parameters.put('pixelSpacingInMeter', 45.0)
    #    break
    output = GPF.createProduct('Terrain-Correction', parameters, source)
    return output
#
# def subset_(source,pixels):
#     print('\tSubsetting...')
#     parameters = HashMap()
#     parameters.put('region',pixels )
#     output = GPF.createProduct('Subset', parameters, source)
#     return output

# def shpToWKT(shp_path):
#     r = shapefile.Reader(shp_path)
#     g = []
#     for s in r.shapes():
#         g.append(pygeoif.geometry.as_shape(s))
#     m = pygeoif.MultiPoint(g)
#     return str(m.wkt).replace("MULTIPOINT", "POLYGON(") + ")"

def subset_(product, shpPath):
    parameters = HashMap()
    wkt = shpPath #shpToWKT(shpPath)
    #SubsetOp = jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
    #geometry = WKTReader().read(wkt)
    parameters.put('copyMetadata', True)
    parameters.put('geoRegion',wkt)# geometry)#'113.7969632641358 -1.829160214922361, 113.7968845575969 -1.828459619961155, 113.7968556380623 -1.828222430127611, 113.7973768900133 -1.827964729931979, 113.7977070393482 -1.827952559488025, 113.7981472741146 -1.827936336182638, 113.798711664528 -1.828034831575053')  #geometry)
    #parameters.put('region', '0, 0,2000, 2000')
    return GPF.createProduct('Subset', parameters, product)



def topsar_deburst(source,pols):
    print('\tOperator-TOPSAR-Deburst...')
    parameters = HashMap()
    parameters.put('selectedPolarisations', pols)
    output = GPF.createProduct('TOPSAR-Deburst', parameters,source)
    return output

#def write_sent_1(source,outpath,folder):
#    print("Writing...")
#    ProductIO.writeProduct(source, outpath + '\\' + folder[:-5], 'GeoTIFF')


#'(113.7969632641358 -1.829160214922361, 113.7968845575969 -1.828459619961155, 113.7968556380623 -1.828222430127611, 113.7973768900133 -1.827964729931979, 113.7977070393482 -1.827952559488025, 113.7981472741146 -1.827936336182638, 113.798711664528 -1.828034831575053)'
#'0, 0, 500, 500'

#'MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)),((15 5, 40 10, 10 20, 5 10, 15 5)))'


def main():
    ## All Sentinel-1 data sub folders are located within a super folder (make sure the data is already unzipped and each sub folder name ends with '.SAFE'):
    path = r'C:\Users\Lord Colm\Desktop\InSAR Thesis\Data\SLC'
    outpath = r'C:\Users\Lord Colm\Desktop\InSAR Thesis\Data\Results\BEAM-DIMAP_Results'
    shapes = r'C:\Users\Lord Colm\Desktop\InSAR Thesis\Data\Primary_Disturbance-polygon.shp'
    path_asf_csv = r'C:\Users\Lord Colm\Desktop\InSAR Thesis\Data\asf-sbas-pairs_24d_35m_Jun20_Dec22.csv'

    ## The below I think will have to be imported as a list of strings to parse into the next functions...
    wkt = 'MULTIPOLYGON (((113.687055646045 -1.88041872846218,113.508693183751 -1.824370167858,113.482694843255 -1.91365021106271,113.653545702012 -1.96710447209521,113.687055646045 -1.88041872846218)) , ((113.432832703583 -1.35287854799105,113.372010436089 -1.4498460421418,113.561345523232 -1.58203176674169,113.617122226634 -1.47983207045205,113.432832703583 -1.35287854799105)) , ((113.647454712203 -1.61881337410528,113.643135431336 -1.62063749600287,113.642638764428 -1.62432828452934,113.64509887982 -1.62708372530131,113.649283296908 -1.62859681101286,113.652466927419 -1.62834960067563,113.65372134645 -1.62653454705797,113.653594274749 -1.62330631414612,113.653348512064 -1.62110373101174,113.651987680338 -1.62001511164443,113.650276905518 -1.61895953296043,113.647454712203 -1.61881337410528)) , ((113.200202822709 -1.52365780078576,113.050525292778 -1.65863192255451,113.270296815809 -1.92198227272771,113.419195971107 -1.82337950434567,113.200202822709 -1.52365780078576)))'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    ## well-known-text (WKT) file for subsetting (can be obtained from SNAP by drawing a polygon)
    #wkt = 'POLYGON ((-157.79579162597656 71.36872100830078, -155.4447021484375 71.36872100830078, \
    #-155.4447021484375 70.60020446777344, -157.79579162597656 70.60020446777344, -157.79579162597656 71.36872100830078))'

    ## With the SLC pairs in a single folder, we assume that the pairs are downloaded together and listed one after another.

    pols = 'VV'#'VH,VV'
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

        sentinel_1_1 = ProductIO.readProduct(path + "\\" + primary + '.zip')
        sentinel_1_2 = ProductIO.readProduct(path + "\\" + secondary + '.zip')# os.listdir(path)[os.listdir(path).index(folder) + 1])

        width = sentinel_1_1.getSceneRasterWidth()
        print("Width: {} px".format(width))
        height = sentinel_1_1.getSceneRasterHeight()
        print("Height: {} px".format(height))
        name = sentinel_1_1.getName()
        print("Name: {}".format(name))
        band_names = sentinel_1_1.getBandNames()
        print("Band names: {}".format(", ".join(band_names)))

        topsarsplit_1 = topsar_split(sentinel_1_1,pols)
        applyorbit_1 = apply_orbit_file(topsarsplit_1)
        topsarsplit_2 = topsar_split(sentinel_1_2,pols)
        applyorbit_2 = apply_orbit_file(topsarsplit_2)
        backgeocoding = back_geocoding([applyorbit_1,applyorbit_2])
        coherence = coherence_(backgeocoding)
        topsardeburst = topsar_deburst(coherence,pols)
        terraincorrection = terrain_correction(topsardeburst)#,proj)
        #subset = subset_(terraincorrection, wkt)#shapes)
        speckle = speckle_filtering(terraincorrection)
        del applyorbit_1
        del applyorbit_2
        del topsarsplit_1
        del topsarsplit_2
        del backgeocoding
        del coherence
        del topsardeburst
        del terraincorrection

        print("Plotting...")
        #plotBand(speckle, 'coh_IW2_VV_02Mar2021_18Feb2021', 0, 1)
        print("Writing...")
        ProductIO.writeProduct(speckle, outpath + '\\' + primary[:25] +'_'+ secondary[17:25], 'BEAM-DIMAP')
        print('Done.')
        del speckle
        sentinel_1_1.dispose()
        sentinel_1_1.closeIO()
        sentinel_1_2.dispose()
        sentinel_1_2.closeIO()
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__== "__main__":
    main()
