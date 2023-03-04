# -*- coding: utf-8 -*-
"""
This script calculates the coherence(or bsc) change between images in a stack and saves a new ccd_stack
"""
"""
@Time    : 07/01/2023 20:05
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : calc_coherence_change
"""


##################
## Credit: https://github.com/insarlab/MintPy-tutorial/blob/main/applications/coherence_change_detection.ipynb
##################


## needed for rasterio..
from osgeo import gdal
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


def calc_coherence_change(coherence_stack,
                          #date12_list,
                          #event_date,
                          meta,
                          method='difference',
                          start_date=None,
                          coherence_threshold=None,
                          min_ccd_value=None,
                          subset=None,
                          mask=None,
                          plot=False,
                          print_msg=False):


    with rasterio.open('../ccd_stack.tif', 'w', **meta) as dst:
        for ix,coherence in enumerate(coherence_stack):

            pre_coherence = coherence_stack[ix]
            try:
                post_coherence = coherence_stack[ix+1]
            except IndexError:
                print('This is the end of the line!')
                break
            # 1. Difference between mean pre_event coh.  - mean. co_event coh.
            if method == 'difference':
                ccd = np.abs(np.abs(post_coherence) - np.abs(pre_coherence)) #pre_coherence - co_coherence

            # 2. Histogram matching (Yun et al. 2015a and Yun et al. 2015b)
            # between mean pre_event coh.(ref image) and mean co_event  coh. (matched image.)
            elif method == 'histogram_matching':
                ccd = 1 - exposure.match_histograms(post_coherence, pre_coherence, channel_axis=False)

                # 3. Ratio between pre- & co-event coherence stack (Washaya, et al. 2018):
            elif method == 'ratio':
                ccd = ((pre_coherence - post_coherence) / post_coherence) * 100

            dst.write_band(ix+1, ccd)


    return ccd, pre_coherence, post_coherence
