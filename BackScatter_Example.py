## This is a script where I will play around with snappy, process some backscatter data for the island of Hawaii
## & show some flooding.. This is a backscatter Analysis..


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import snappy
from snappy import Product
from snappy import ProductIO
from snappy import ProductUtils
from snappy import WKTReader
from snappy import HashMap
from snappy import GPF

import shapefile
import pygeoif



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



path_to_sentinel_data = "C:/Users/Colm The Creator/Downloads/S1A_IW_GRDH_1SDV_20220909T163215_20220909T163244_044930_055DF2_BA72.zip"
product = ProductIO.readProduct(path_to_sentinel_data)


width = product.getSceneRasterWidth()
print("Width: {} px".format(width))
height = product.getSceneRasterHeight()
print("Height: {} px".format(height))
name = product.getName()
print("Name: {}".format(name))
band_names = product.getBandNames()
print("Band names: {}".format(", ".join(band_names)))


parameters = HashMap()
## So, we register the SPI to the GPF? or we get a regular SPI in GPF format..
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis() ## this will load available operators..
## a set of random parameters that I don't understand, these will have to be researched..'
parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
parameters.put('polyDegree', '3')
parameters.put('continueOnFail', 'false')
## & create a GPF product based on those parameters, where the product is a read in Sentinel 1 GRD zip :)
apply_orbit_file = GPF.createProduct('Apply-Orbit-File', parameters, product)

######################################################
## Image pre-processing
######################################################
#########################
## Orbit file application
#########################
# Before any SAR pre-processing steps occur, the product subset should be properly
# orthorectified to improve accuracy. To properly orthorectify the image, the orbit file is applied
# using the Apply-Orbit-File GPF module. SNAP is able to generate and apply a high accuracy
# satellite orbit file to a product.
########################


######################
## Some notes on SPIs:
## The OperatorSpi class is the service provider interface (SPI) for Operators. Therefore this abstract class is intended to be derived by clients.
## An SPI is required for your operator if you want to make it accessible via an alias name in the various GPF.create methods or within GPF Graph XML code.
## SPI are registered either pragmatically using the OperatorSpiRegistry or automatically via standard Java services lookup mechanism
######################
######################
## Some notes on GPFs:
## The facade for the Graph Processing Framework.
## A graph processing framework (GPF) is a set of tools oriented to process graphs. Graph vertices are used to model data and edges model relationships between vertices.
#####################



####################
## Image Subsetting:
## using an island shapefile..
## Skipping this unnecessary part for now...
####################

plotBand(apply_orbit_file, "Intensity_VV", 0, 100000)

####################
## Data is now orthorectified.
## orthorectification is a subset of georeferencing.
## this is the process of removing sensor, satellite/aircraft motion and terrain-related geometric distortions from raw imagery.
## This prepares images for use in maps..
## This is one of the main processing steps for evaluating remote sensing data..
####################
##############
## our next step is to calibrate the image to sigma-naught values..
##

parameters = HashMap()
parameters.put('outputSigmaBand', True)
parameters.put('sourceBands', 'Intensity_VV')
parameters.put('selectedPolarisations', "VV")
parameters.put('outputImageScaleInDb', False)
product_calibrated = GPF.createProduct("Calibration", parameters,apply_orbit_file)

##############
## The data is now stored in a band called Sigma0_VV in the product object product_calibrated.
## Sigma0 is the calibrated backscatter coefficient, measured in dB, compared to the strength observed to that expected from 1m^2.
## it is a normalized dimensionless number..
##############
#plotBand(product_calibrated, "Sigma0_VV", 0, 1)

###########################
## Viewing all hashmap SPIs (operators/parameters;)
# op_spi_it = GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpis().iterator()
# while op_spi_it.hasNext():
#     op_spi = op_spi_it.next()
#     print("op_spi: ", op_spi.getOperatorAlias())

###########################


