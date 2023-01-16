# -*- coding: utf-8 -*-
"""
@Time    : 07/01/2023 20:05
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : calc_coherence_change
"""


from osgeo import gdal
import rasterio

import os
import sys

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


    with rasterio.open('ccd_stack.tif', 'w', **meta) as dst:
        for ix,coherence in enumerate(coherence_stack):   # , date12   date12_list):
            #date1, date2 = date12.split('_')


            #if subset is None:
                #subset = [0, coherence_stack.shape[2], 0, coherence_stack.shape[1]]

            pre_coherence = coherence_stack[ix] # , subset[2]:subset[3], subset[0]:subset[1]]
            try:
                post_coherence = coherence_stack[ix+1] # ,subset[2]:subset[3], subset[0]:subset[1]]
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











































    # vprint = print if print_msg else lambda *args, **kwargs: None

    # vprint('\n###############################################')
    # vprint('########## Coherence Change Detection #########')
    # vprint('###############################################\n')

    # if date1 < date2:
    #     pre_event_ix.append(ix)
    # elif date1 > date2:
    #     post_event_ix.append(ix)
    #
    #pre_coherence_stack = #pre_coherence_stack = coherence_stack[pre_event_ix, subset[2]:subset[3], subset[0]:subset[1]]
    #co_coherence_stack #co_coherence_stack = coherence_stack[co_event_ix, subset[2]:subset[3], subset[0]:subset[1]]


    ##################### CCD ##################################
    # vprint('\nCoherence Change Detection Calculation')
    # vprint('     Method: {}'.format(method))

    ###################################
    ######### Mean pre-event and co-event stack coherence ######
    ## The only important calculations are below.
    ###################################
    #pre_coherence = np.mean(pre_coherence_stack, axis=0)
    #co_coherence = np.mean(co_coherence_stack, axis=0)

        ######
        #
        # ###
        ## Below is required only if we need
        ## an event date to look at, we don't know our event date, we want to find this
        ## using our algorithm.
        #########


            # Coherence stacks

    ## Here I want to use have my whole list of coherence images passed in as my coherence stack,
    ## then between each two coherence images I want to obtain a change between the coherence,
    ## so that I have a dataset of the change in coherence between images.
    ##so What will I do with the cahgne in coherence over time then?
    ## Similarly here I can build the change in backscatter over time, using the same code.


  #################### Refinement #############################
    ## Only based on a coherence threshold (possibly multiple thresholds)for reducing data,
    ## This will take the images that have an image mean less than the threshold & not use them.
    ####################
    # methods = ['difference', 'histogram_matching', 'ratio']
    # if method not in methods:
    #     raise ValueError(f"Un-recognized CCD method: {method}!")
    # # If startDate is not defined, use the first date in timeseries
    # if start_date is None:
    #     start_date = date12_list[0]

    #
    # pre_event_ix = []
    # co_event_ix = []


    # Mask coherence stack
    # if mask is not None:
    #     pre_coherence_stack[:, mask == 0] = np.nan
    #     co_coherence_stack[:, mask == 0] = np.nan


    # ###############################
    # ############ Remove values below min_ccd_value #################.
    # ###############################
    # if min_ccd_value is not None:
    #     #vprint('Removing CCD values below {}'.format(min_ccd_value))
    #     ccd = np.ma.filled(np.ma.masked_where(ccd < min_ccd_value, ccd), fill_value=np.nan)

'''
    if coherence_threshold is not None:
        vprint('Refine pre-event Coherence Stack:')
        pre_coherence_stack_bt = pre_coherence_stack
        mean_coherence_bt = np.nanmean(pre_coherence_stack, axis=(1, 2))
        # threshold can be manually defined as list of dates to keep
        if isinstance(coherence_threshold, list) is True:
            pre_coherence_stack = np.take(pre_coherence_stack, coherence_threshold, axis=0)
        else:
            ix = np.where(mean_coherence_bt > coherence_threshold)
            pre_coherence_stack = np.take(pre_coherence_stack, ix[0], axis=0)
            vprint('List of dates to keep:', ix[0])
        vprint(f'Number of pre-Event Datasets After Thresholding: {pre_coherence_stack.shape[0]}')
        mean_coherence = np.nanmean(pre_coherence_stack, axis=(1, 2))
        ###################
        ##Plotting before and after threshold is applied.
        ###################
        if plot:
            from matplotlib.ticker import FormatStrFormatter
            fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=100)
            # Histograms
            img1 = axs[0, 0].hist(mean_coherence_bt, bins=50, density=False)
            img2 = axs[0, 1].hist(mean_coherence, bins=20, density=False)
            axs[0, 0].set_title('Pre-Event Coherence Before Thresh.')
            axs[0, 1].set_title('Pre-Event Coherence After Thresh.')
            axs[0, 0].set_ylabel('# Images', fontsize=8)
            for ax in axs[0, :]:
                ax.set_xlabel('Avg. Spatial Coherence', fontsize=8, weight='bold')
                ax.xaxis.set_label_coords(0.5, 0.06)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            # CCD plots
            img3 = axs[1, 0].imshow(np.mean(pre_coherence_stack_bt, axis=0), cmap='gray', clim=[0, 1])
            img4 = axs[1, 1].imshow(np.mean(pre_coherence_stack, axis=0), cmap='gray', clim=[0, 1])
            for ax in axs[1, :]:
                ax.set_xticks([]), ax.set_xticklabels([])
                ax.set_yticks([]), ax.set_yticklabels([])
                
'''

'''
Calculate Coherence Change for the defined eventDate (earthquake, volcano eruption, flooding)

Input:
  coherence_stack      ::  ndarray           - timeseries coherence stack
  date12_list          ::  list              - dates of SAR acquistions used to generate coherence, example: '20170828_20170904'
  event_date           ::  str               - date of the event for which CCD is calculated, example: '20170904'
                                               (event: earthquake, volcano eruption, flooding)
  method               ::  str               - method for calculate CCD:
                                               'difference'         : difference between mean pre- and co- coherence
                                               'histogram_matching' : histogram matching using pre-event coherence as reference (
                                                                      Yun et al. 2015a and Yun et al. 2015b)
                                               'ratio'              : ratio between pre- & co- coherence stack (Washaya, et al. 2018)
  start_date           ::  str               - define the first date to get reference, pre-coherence stack: [start_date:event_date],
                                               default is first date of timeseries
  coherence_threshold  ::  float/list        - mean spatial coherence threshold used to refine the pre-coherence stack
  min_ccd_value        ::  float             - used to remove the CCD values below the defined value
  subset               ::  list              - calculate CCD on image subset defined as [x1,x2,y1,y2]
  mask                 ::  2D boolean array  - mask the CCD results with external data such as watermask, e.g. waterMask.h5
  plot                 ::  bool              - [True/False] plot intermediate steps (default = False)

Output:
  ccd                  ::  array             - Coherence Change Detection
  pre_coherence        ::  array             - Mean Pre-Event Coherence, i.e. before the event
  co_coherence         ::  array             - Mean Co-Event Coherence, i.e. during the event
'''